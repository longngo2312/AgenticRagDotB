"""
DAY 3 — STEP 1: Caption screenshots with a vision-LLM, using their step
context, and splice the caption into the document text before chunking.

Why context-aware, not a bare visual description: "shows a blue icon" is
useless for retrieval; "shows the Tài khoản icon the user taps in this step"
is exactly what a search query would match. The cost is that captions can't
be deduped by image hash alone — the same screenshot reused across two
different steps needs two different captions. Cache key is therefore
(image_hash, context_hash): a true duplicate (same image, same step) still
short-circuits; a genuinely different context costs one more call.

Generation is gated by `allow_generate` (wired to the --caption CLI flag):
a cache HIT is always applied, even on a plain preview run, so already-paid-for
captions never regress. Only a cache MISS is skipped when allow_generate=False
— the placeholder is left bare rather than burning quota mid-preview.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import httpx
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GOOGLE_API_KEY, BASE_DIR, DATA_DIR

CAPTION_MODEL = "models/gemini-flash-lite-latest"
CAPTIONS_PATH = DATA_DIR / "images" / "captions.json"

# Free tier: 15 requests/minute for this model (confirmed empirically via a 429).
_RATE_LIMIT_BACKOFF_SEC = [60, 60, 120, 120]
# Transient network failures (timeouts, connection resets) — not a quota issue,
# a longer 950-image run over an unattended connection will hit a few of these.
_NETWORK_BACKOFF_SEC = [15, 30, 60]
_RETRYABLE_NETWORK_ERRORS = (httpx.TransportError, httpx.TimeoutException, ServerError)

_client = genai.Client(api_key=GOOGLE_API_KEY)
_IMAGE_ONLY_RE = re.compile(r"^\[IMAGE:\s*([^\]]+)\]$")


class DailyQuotaExhausted(RuntimeError):
    """The free tier's per-day request cap (not per-minute) is exhausted.
    Retrying within this process is pointless — the quota resets on a rolling
    daily window. Callers should fall back to cache-only for the rest of the run."""


def _is_daily_quota_error(e: ClientError) -> bool:
    try:
        for d in e.details.get("error", {}).get("details", []):
            for v in d.get("violations", []):
                if "PerDay" in v.get("quotaId", ""):
                    return True
    except AttributeError:
        pass
    return False


def _load_cache() -> dict:
    if CAPTIONS_PATH.exists():
        return json.loads(CAPTIONS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CAPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAPTIONS_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


_cache = _load_cache()
_daily_quota_exhausted = False


def _normalize_context(text: str) -> str:
    # Collapse whitespace and drop trailing punctuation so a trivial copy-edit
    # to the step text doesn't spuriously bust the cache and re-trigger a call.
    return " ".join(text.split()).rstrip(".:;,").lower()


def _generate_with_backoff(image_bytes: bytes, mime: str, prompt: str) -> str:
    rate_limit_retries = 0
    network_retries = 0
    while True:
        try:
            resp = _client.models.generate_content(
                model=CAPTION_MODEL,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime), prompt],
            )
            return resp.text.strip()
        except ClientError as e:
            if e.code == 429 and _is_daily_quota_error(e):
                raise DailyQuotaExhausted(e.message) from e
            if e.code == 429 and rate_limit_retries < len(_RATE_LIMIT_BACKOFF_SEC):
                backoff = _RATE_LIMIT_BACKOFF_SEC[rate_limit_retries]
                rate_limit_retries += 1
                print(f"\n  [rate-limit] waiting {backoff}s before retry {rate_limit_retries}...")
                time.sleep(backoff)
                continue
            raise
        except _RETRYABLE_NETWORK_ERRORS as e:
            if network_retries < len(_NETWORK_BACKOFF_SEC):
                backoff = _NETWORK_BACKOFF_SEC[network_retries]
                network_retries += 1
                print(f"\n  [network] {type(e).__name__}, waiting {backoff}s before retry {network_retries}...")
                time.sleep(backoff)
                continue
            raise
    raise RuntimeError("unreachable")


def caption_image(local_path: str, context_text: str, allow_generate: bool = True) -> str | None:
    """
    Returns a caption, or None if uncaptionable (file missing on disk — can
    happen when image resolution fell back to an unresolvable placeholder —
    or a cache miss while allow_generate=False).
    """
    global _daily_quota_exhausted
    abs_path = BASE_DIR / local_path
    if not abs_path.exists():
        return None

    image_bytes = abs_path.read_bytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    context_hash = hashlib.sha256(_normalize_context(context_text).encode()).hexdigest()
    key = f"{image_hash}:{context_hash}"

    if key in _cache:
        return _cache[key]
    if not allow_generate or _daily_quota_exhausted:
        return None

    mime = "image/png" if abs_path.suffix == ".png" else "image/jpeg"
    prompt = (
        f'This is a screenshot illustrating the step: "{context_text.strip()}"\n'
        "Silently judge, from the step's wording, whether it is an ACTION "
        "(chọn/click/nhấn/select — tells the user to do something) or a RESULT "
        "(hệ thống hiển thị/the system displays/shows — describes what appeared "
        "after an action). Never write that judgment down or mention 'action step' "
        "/ 'result step' in your answer — it only governs which of the two "
        "sentences below you write.\n"
        "If it's an ACTION: describe the specific button, field, or control to "
        "interact with. If it's a RESULT: describe what information or status is "
        "now visible — do not invent a click, button, or 'opens a window' action.\n"
        "Always write 1-2 sentences of plain documentation text describing what "
        "is actually visible, especially anything circled, arrowed, boxed, or "
        "highlighted. Never reply with just a label and no description.\n"
        "Numbered circles or labels (①②③ / 1, 2, 3) do not always mean "
        "'click these in order' — they are often just reference callouts "
        "pointing at different elements or states (e.g. contrasting a "
        "'before' vs 'after' status on two different rows, or labeling parts "
        "of a result). Only describe something as a click target if it is "
        "visually a button, link, or control — never a status badge, label, "
        "or table cell — and never invent a click action the image doesn't "
        "actually show."
    )
    try:
        caption = _generate_with_backoff(image_bytes, mime, prompt)
    except DailyQuotaExhausted as e:
        _daily_quota_exhausted = True
        print(f"\n  [daily-quota] {e} — falling back to cache-only for the rest of this run.")
        return None
    _cache[key] = caption
    _save_cache(_cache)
    return caption


def caption_document(content: str, fallback_context: str, allow_generate: bool = True) -> str:
    """
    Walk a parsed document's paragraphs; for every `[IMAGE: path]` placeholder,
    splice in a caption generated from the nearest preceding non-image
    paragraph as context (falling back to `fallback_context` — the doc title —
    if the image opens its section with nothing before it). Chunking
    downstream never needs to know captioning happened at all.
    """
    paragraphs = re.split(r"\n{2,}", content)
    out: list[str] = []
    last_context = fallback_context

    for para in paragraphs:
        stripped = para.strip()
        m = _IMAGE_ONLY_RE.match(stripped)
        if m:
            image_path = m.group(1).strip()
            caption = caption_image(image_path, last_context, allow_generate=allow_generate)
            out.append(f"{stripped} {caption}" if caption else stripped)
        else:
            out.append(para)
            if stripped:
                last_context = stripped

    return "\n\n".join(out)

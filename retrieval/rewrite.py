"""
DAY 4 — STEP 1: Query rewrite (condense history + glossary expansion).

Two independent problems, two independent fixes:

1. Follow-up questions are ambiguous out of context — "còn phí thì sao?" after
   a question about Placement Test means nothing to a retriever on its own.
   `condense_query` uses the LLM to fold the last few turns into one
   standalone question, in the user's own language (no translation).

2. DotB's docs and its users don't always use the same term for the same
   concept ("delay" in the UI vs. "bảo lưu" in how support staff actually
   talk about it). BM25 is a lexical matcher — it can't bridge that gap on
   its own. `expand_glossary` appends known aliases from data/glossary.json
   so both vocabularies are present in the query BM25 searches against.
   (Dense search is less affected since embeddings already capture some of
   this synonymy, but the appended terms don't hurt it either.)

chat_history format: list[{"role": "user" | "assistant", "content": str}],
oldest first — the same shape agent/state.py will use once it exists (Day 5).

Interactive path, not ingestion: a demo target of <4s means this can't afford
ingestion-style 60s rate-limit backoff. On any failure it falls back to the
original (un-condensed) question rather than blocking the user.
"""
import json
import re
import sys
from pathlib import Path

from google import genai

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GOOGLE_API_KEY, GLOSSARY_PATH, LLM_MODEL, REWRITE_HISTORY_TURNS

_client = genai.Client(api_key=GOOGLE_API_KEY)


def _load_glossary() -> dict[str, list[str]]:
    if not GLOSSARY_PATH.exists():
        return {}
    return json.loads(GLOSSARY_PATH.read_text(encoding="utf-8")).get("aliases", {})


_GLOSSARY = _load_glossary()

_CONDENSE_PROMPT = """Given the conversation so far and a follow-up question, rewrite the \
follow-up into a standalone question that makes sense without the conversation.

Rules:
- Keep it in the same language the user is writing in (Vietnamese or English) — do not translate.
- Only resolve what the follow-up is actually referring to; don't add information that wasn't asked for.
- If the follow-up already stands on its own, return it unchanged.
- Output only the rewritten question, nothing else.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""


def _contains_term(text: str, term: str) -> bool:
    # Word-boundary match, not substring containment: short abbreviation
    # aliases ("ph", "hv", "hs") would otherwise false-positive inside
    # unrelated words (e.g. "ph" inside "phí" — fee, not phụ huynh/parent).
    return re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text) is not None


def expand_glossary(query: str) -> str:
    """Append any glossary terms (canonical or alias) implied by the query
    but not already present in it, so BM25 can match whichever term the
    docs actually use."""
    lower_query = query.lower()
    additions: list[str] = []

    for canonical, aliases in _GLOSSARY.items():
        terms = [canonical, *aliases]
        if not any(_contains_term(lower_query, t) for t in terms):
            continue
        for t in terms:
            if not _contains_term(lower_query, t) and t not in additions:
                additions.append(t)

    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def _format_history(chat_history: list[dict]) -> str:
    turns = chat_history[-(REWRITE_HISTORY_TURNS * 2):]
    lines = [f"{turn['role']}: {turn['content']}" for turn in turns]
    return "\n".join(lines)


def condense_query(question: str, chat_history: list[dict] | None = None) -> str:
    """Fold prior turns into a standalone question via the LLM. No-op (and no
    API call) when there's no history — most turns are the first turn."""
    if not chat_history:
        return question

    prompt = _CONDENSE_PROMPT.format(
        history=_format_history(chat_history),
        question=question,
    )
    try:
        resp = _client.models.generate_content(model=LLM_MODEL, contents=[prompt])
        rewritten = resp.text.strip()
    except Exception:
        # Interactive path: never block the user on a rewrite failure —
        # fall back to the original question rather than retrying/sleeping.
        return question

    # Strip a leading label if the model echoed one despite instructions.
    rewritten = re.sub(r"^(standalone question|câu hỏi)\s*:\s*", "", rewritten, flags=re.IGNORECASE)
    return rewritten or question


def rewrite_query(question: str, chat_history: list[dict] | None = None) -> str:
    standalone = condense_query(question, chat_history)
    return expand_glossary(standalone)

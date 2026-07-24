"""
DAY 2 — STEP 2: Parse raw markdown into structured documents

Input:  CrawledPage (url, title, description, raw_content)
Output: ParsedDocument with metadata extracted from URL structure

GitBook preprocessing (before chunking):
  - `> Step N: text`            → `**Step N:** text`
  - `{% hint %}...{% endhint %}` → `> **Note:** ...`
  - `<figure><img src="...">   → `[IMAGE: url]` placeholder
  - other HTML tags             → stripped

URL path encodes the business hierarchy, e.g.
  /tuyen-sinh-ban-hang/lead/convert-lead.md
    → breadcrumb = ['tuyen-sinh-ban-hang', 'lead', 'convert-lead']
    → section    = 'tuyen-sinh-ban-hang'
    → product    = 'EMS' (default) / 'SEA' / 'TEA' / 'Metrikal'
"""
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ParsedDocument:
    url: str
    title: str
    description: str
    breadcrumb: list[str]
    breadcrumb_str: str
    section: str
    product: str
    content: str
    content_hash: str
    last_seen: str
    image_urls: list[str]   # local cached paths (data/images/<hash>.ext), figure order


# GitBook's llms.txt exporter prepends this boilerplate to every .md page.
_LLMS_FOOTER_RE = re.compile(
    r"^>\s*For the complete documentation index,.*?\.md\)\.\s*$\n*",
    re.MULTILINE,
)


# ── GitBook preprocessing ──────────────────────────────────────────────────────

def _preprocess_gitbook(
    content: str,
    resolved_images: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Convert GitBook-specific syntax to plain text before chunking.

    `resolved_images` is the list of local cached-image paths (already
    downloaded from the real GitBook proxy URLs), in the same top-to-bottom
    order as the <figure> images on the rendered HTML page. The N-th
    markdown `![]()` / `<figure><img>` occurrence in this .md is matched
    positionally to the N-th entry — the short-id path inside the .md
    itself (e.g. `/files/abc123`) does not resolve to anything and is
    only used as a placeholder to detect "an image goes here".

    Returns (cleaned_content, list_of_image_paths_actually_used).
    """
    resolved_images = resolved_images or []
    used_images: list[str] = []
    image_counter = {"i": 0}

    def _next_image(placeholder_src: str) -> str:
        idx = image_counter["i"]
        image_counter["i"] += 1
        if idx < len(resolved_images):
            resolved = resolved_images[idx]
        else:
            # fall back to the raw (likely unresolvable) src if we ran out
            resolved = placeholder_src
        if resolved not in used_images:
            used_images.append(resolved)
        return resolved

    # 1. {% hint style="..." %}...{% endhint %}  →  > **Note:** ...
    def replace_hint(m: re.Match) -> str:
        inner = m.group(1).strip()
        # strip any nested HTML inside hints
        inner = re.sub(r"<[^>]+>", "", inner)
        inner = inner.strip()
        return f"> **Lưu ý:** {inner}"

    content = re.sub(
        r'\{%\s*hint\s+style=["\'][^"\']*["\']\s*%\}(.*?)\{%\s*endhint\s*%\}',
        replace_hint,
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 2. <figure><img src="...">...</figure>  →  [IMAGE: resolved_local_path]
    def replace_figure(m: re.Match) -> str:
        resolved = _next_image(m.group(1))
        return f"[IMAGE: {resolved}]"

    content = re.sub(
        r'<figure[^>]*>.*?<img[^>]+src=["\']([^"\']+)["\'][^>]*>.*?</figure>',
        replace_figure,
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # also catch bare <img> outside figures
    def replace_img(m: re.Match) -> str:
        resolved = _next_image(m.group(1))
        return f"[IMAGE: {resolved}]"

    content = re.sub(
        r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',
        replace_img,
        content,
        flags=re.IGNORECASE,
    )

    # 2b. Markdown images: ![alt](/files/short-id)  →  [IMAGE: resolved_local_path]
    #     The path inside the .md is a GitBook internal short-id that never
    #     resolves on its own — it only marks position. The real asset was
    #     already downloaded (via the rendered HTML page) and passed in as
    #     `resolved_images`, matched positionally.
    def replace_md_image(m: re.Match) -> str:
        resolved = _next_image(m.group(2))
        return f"[IMAGE: {resolved}]"

    content = re.sub(
        r'!\[([^\]]*)\]\(([^)]+)\)',
        replace_md_image,
        content,
    )

    # 3. > Step N: text  (GitBook blockquote used as step indicator)
    #    → **Step N:** text
    def replace_step(m: re.Match) -> str:
        step_label = m.group(1).strip()   # e.g. "Step 1"
        step_body  = m.group(2).strip()
        return f"**{step_label}:** {step_body}"

    content = re.sub(
        r"^>\s*(Step\s+\d+[:.:]?)\s+(.+)$",
        replace_step,
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 4. Other GitBook block tags: {% tabs %}, {% tab %}, {% endtabs %}, etc.
    content = re.sub(r"\{%-?\s*\w[^%]*%\}", "", content)

    return content, used_images


def _clean_markdown(content: str) -> str:
    # strip the llms.txt exporter's boilerplate footer (present on every page)
    content = _LLMS_FOOTER_RE.sub("", content)
    # strip remaining HTML tags (but keep [IMAGE: ...] placeholders)
    content = re.sub(r"<[^>]+>", "", content)
    # collapse 3+ blank lines to 2
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


# ── Metadata helpers ───────────────────────────────────────────────────────────

def extract_breadcrumb(url: str) -> list[str]:
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    if parts:
        parts[-1] = re.sub(r"\.md$", "", parts[-1])
    return parts


def infer_product(breadcrumb: list[str]) -> str:
    for segment in breadcrumb:
        seg = segment.lower()
        if "sea" in seg:
            return "SEA"
        if "tea" in seg:
            return "TEA"
        if "metrikal" in seg:
            return "Metrikal"
    return "EMS"


def _humanize(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_document(
    url: str,
    title: str,
    description: str,
    raw_content: str,
    resolved_images: list[str] | None = None,
) -> ParsedDocument:
    """
    `resolved_images`: local cached-image paths (from ingestion.image_cache),
    in the same order as the <figure> images on the rendered HTML page —
    matched positionally to markdown/HTML image placeholders in raw_content.
    """
    # GitBook → plain text
    content, image_urls = _preprocess_gitbook(raw_content, resolved_images)
    content = _clean_markdown(content)

    breadcrumb   = extract_breadcrumb(url)
    product      = infer_product(breadcrumb)
    section      = breadcrumb[0] if breadcrumb else ""
    breadcrumb_str = " > ".join(_humanize(b) for b in breadcrumb)
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    last_seen    = datetime.now(timezone.utc).isoformat()

    return ParsedDocument(
        url=url,
        title=title,
        description=description,
        breadcrumb=breadcrumb,
        breadcrumb_str=breadcrumb_str,
        section=section,
        product=product,
        content=content,
        content_hash=content_hash,
        last_seen=last_seen,
        image_urls=image_urls,
    )

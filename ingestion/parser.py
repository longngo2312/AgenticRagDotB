"""
DAY 2 — STEP 2: Parse raw markdown into structured documents

Input:  CrawledPage (url, title, description, raw_content)
Output: ParsedDocument with metadata extracted from URL structure

Key insight from design doc:
  URL path encodes the business hierarchy, e.g.
  /tuyen-sinh-ban-hang/lead/convert-lead.md
    → breadcrumb = ['tuyen-sinh-ban-hang', 'lead', 'convert-lead']
    → section    = 'tuyen-sinh-ban-hang'
    → product    = 'EMS' (default) / 'SEA' / 'TEA' / 'Metrikal'
"""
import hashlib
import re
import sys
from dataclasses import dataclass, field
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


def extract_breadcrumb(url: str) -> list[str]:
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    if parts:
        # strip .md extension from last segment
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


def clean_markdown(content: str) -> str:
    # strip HTML tags that may have leaked through
    content = re.sub(r"<[^>]+>", "", content)
    # collapse 3+ blank lines to 2
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _humanize(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def parse_document(
    url: str,
    title: str,
    description: str,
    raw_content: str,
) -> ParsedDocument:
    breadcrumb = extract_breadcrumb(url)
    product = infer_product(breadcrumb)
    section = breadcrumb[0] if breadcrumb else ""
    content = clean_markdown(raw_content)
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    breadcrumb_str = " > ".join(_humanize(b) for b in breadcrumb)
    last_seen = datetime.now(timezone.utc).isoformat()

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
    )

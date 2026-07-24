"""
DAY 2 — STEP 1: Crawl help.dotb.vn

Flow:
  GET help.dotb.vn/llms.txt
    → parse every "[Title](URL.md): description" line
    → async-download each .md AND its HTML counterpart (URL without .md)
    → extract <figure> img src values from HTML (real GitBook proxy URLs)
    → return list[CrawledPage]
    → optionally save raw .md files to data/raw_docs/
"""
import asyncio
import hashlib
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LLMS_TXT_URL, RAW_DOCS_DIR, CRAWLER_CONCURRENCY, CRAWLER_DELAY_SEC


@dataclass
class CrawledPage:
    url: str
    title: str
    description: str
    content: str
    content_hash: str
    image_urls: list[str] = field(default_factory=list)  # real GitBook proxy URLs, figure order


# ── HTML image extractor ───────────────────────────────────────────────────────

class _ZoomImgParser(HTMLParser):
    """
    Extract <img src> for content images, in document order.

    GitBook's rendered HTML does not wrap content images in <figure> — nav
    icons, logos, and avatars are plain <img> tags scattered in the header,
    while every actual content image carries data-testid="zoom-image" (the
    click-to-zoom affordance). That attribute is what distinguishes content
    images from chrome.
    """
    def __init__(self):
        super().__init__()
        self.srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag != "img":
            return
        d = dict(attrs)
        if d.get("data-testid") == "zoom-image":
            src = d.get("src", "")
            if src:
                self.srcs.append(src)


def _extract_content_images(html: str) -> list[str]:
    p = _ZoomImgParser()
    try:
        p.feed(html)
    except Exception:
        pass
    return p.srcs


# ── Crawl helpers ──────────────────────────────────────────────────────────────

def parse_llms_txt(text: str, base_url: str) -> list[dict]:
    base = base_url.rstrip("/")
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^-?\s*\[([^\]]+)\]\(([^)]+)\)(?::\s*(.*))?$", line)
        if m:
            title, url, desc = m.group(1), m.group(2), m.group(3) or ""
            if not url.startswith("http"):
                url = urljoin(base + "/", url.lstrip("/"))
            entries.append({"title": title.strip(), "url": url.strip(), "description": desc.strip()})
            continue
        if line.startswith("http"):
            path = urlparse(line).path
            title = Path(path).stem.replace("-", " ").title()
            entries.append({"title": title, "url": line, "description": ""})
    return entries


async def _fetch_one(
    client: httpx.AsyncClient,
    page: dict,
    semaphore: asyncio.Semaphore,
) -> CrawledPage | None:
    async with semaphore:
        # Fetch .md
        try:
            r = await client.get(page["url"], timeout=20.0)
            r.raise_for_status()
            content = r.text
        except Exception as e:
            print(f"\n  [WARN] Failed {page['url']}: {e}")
            return None
        finally:
            await asyncio.sleep(CRAWLER_DELAY_SEC)

    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Co-fetch HTML to extract real image URLs (GitBook proxy URLs)
    image_urls: list[str] = []
    html_url = page["url"].removesuffix(".md")
    if html_url != page["url"]:
        try:
            r_html = await client.get(html_url, timeout=20.0)
            if r_html.status_code == 200:
                image_urls = _extract_content_images(r_html.text)
        except Exception:
            pass  # image URLs are best-effort

    return CrawledPage(
        url=page["url"],
        title=page["title"],
        description=page["description"],
        content=content,
        content_hash=content_hash,
        image_urls=image_urls,
    )


async def crawl_all(save_raw: bool = True) -> list[CrawledPage]:
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        print(f"  Fetching index: {LLMS_TXT_URL}")
        r = await client.get(LLMS_TXT_URL, timeout=30.0)
        r.raise_for_status()
        base_url = f"{urlparse(LLMS_TXT_URL).scheme}://{urlparse(LLMS_TXT_URL).netloc}"
        pages = parse_llms_txt(r.text, base_url)
        print(f"  Found {len(pages)} pages in llms.txt")

        semaphore = asyncio.Semaphore(CRAWLER_CONCURRENCY)
        tasks = [_fetch_one(client, p, semaphore) for p in pages]

        results: list[CrawledPage] = []
        with tqdm(total=len(tasks), desc="  Downloading", unit="page") as pbar:
            for coro in asyncio.as_completed(tasks):
                page = await coro
                if page:
                    results.append(page)
                    if save_raw:
                        safe_name = hashlib.md5(page.url.encode()).hexdigest() + ".md"
                        (RAW_DOCS_DIR / safe_name).write_text(page.content, encoding="utf-8")
                pbar.update(1)

    print(f"  Crawled {len(results)}/{len(pages)} pages successfully")
    return results

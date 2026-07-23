"""
DAY 2 — STEP 1: Crawl help.dotb.vn

Flow:
  GET help.dotb.vn/llms.txt
    → parse every "[Title](URL.md): description" line
    → async-download each .md (rate-limited, CRAWLER_CONCURRENCY workers)
    → return list[CrawledPage]
    → optionally save raw .md files to data/raw_docs/
"""
import asyncio
import hashlib
import re
import sys
from dataclasses import dataclass
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


def parse_llms_txt(text: str, base_url: str) -> list[dict]:
    base = base_url.rstrip("/")
    entries = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue

        # format: - [Title](URL): description  OR  [Title](URL): description
        m = re.match(r"^-?\s*\[([^\]]+)\]\(([^)]+)\)(?::\s*(.*))?$", line)
        if m:
            title, url, desc = m.group(1), m.group(2), m.group(3) or ""
            # resolve relative URLs
            if not url.startswith("http"):
                url = urljoin(base + "/", url.lstrip("/"))
            entries.append({"title": title.strip(), "url": url.strip(), "description": desc.strip()})
            continue

        # plain URL per line fallback
        if line.startswith("http"):
            path = urlparse(line).path
            title = Path(path).stem.replace("-", " ").title()
            entries.append({"title": title, "url": line, "description": ""})
    print(entries)
    return entries


async def _fetch_one(
    client: httpx.AsyncClient,
    page: dict,
    semaphore: asyncio.Semaphore,
) -> CrawledPage | None:
    async with semaphore:
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
    return CrawledPage(
        url=page["url"],
        title=page["title"],
        description=page["description"],
        content=content,
        content_hash=content_hash,
    )


async def crawl_all(save_raw: bool = True) -> list[CrawledPage]:
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # fetch llms.txt index
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

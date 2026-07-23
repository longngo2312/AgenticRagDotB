"""
DAY 2 — STEP 1: Crawl help.dotb.vn

Flow:
  GET help.dotb.vn/llms.txt
    → parse every "[Title](URL.md): description" line
    → async-download each .md (rate-limited, CRAWLER_CONCURRENCY workers)
    → return list[CrawledPage]
    → optionally save raw .md files to data/raw_docs/
"""
# TODO D2-1a: import asyncio, hashlib, re, dataclasses, httpx, tqdm
# TODO D2-1b: import config constants (LLMS_TXT_URL, RAW_DOCS_DIR, etc.)

# TODO D2-2: define CrawledPage dataclass
#   fields: url, title, description, content (raw md), content_hash (sha256)

# TODO D2-3: def parse_llms_txt(text: str, base_url: str) -> list[dict]
#   - iterate lines, skip # comments and > description lines
#   - regex match:  -? [Title](URL): description
#   - also handle plain-URL-per-line fallback
#   - resolve relative URLs against base_url
#   - return list of {title, url, description}

# TODO D2-4: async def _fetch_one(client, page, semaphore) -> CrawledPage | None
#   - acquire semaphore, GET page['url'] with timeout=20s
#   - on HTTP error → print warning, return None
#   - compute content_hash = sha256(content)
#   - always sleep CRAWLER_DELAY_SEC before releasing semaphore

# TODO D2-5: async def crawl_all(save_raw=True) -> list[CrawledPage]
#   - create RAW_DOCS_DIR
#   - fetch + parse llms.txt  → list[dict]
#   - build asyncio tasks with Semaphore(CRAWLER_CONCURRENCY)
#   - tqdm progress bar over asyncio.as_completed(tasks)
#   - if save_raw: write each .md to RAW_DOCS_DIR/{md5(url)}.md
#   - print summary: "Crawled X/Y pages"

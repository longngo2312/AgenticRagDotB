"""
DAY 2 — STEP 2b: Cache GitBook-hosted images locally

GitBook serves images through a signed resize proxy:
  https://help.dotb.vn/~gitbook/image?url=...&token=...&sign=...
The token/sign params are not guaranteed to stay valid, so we download the
bytes once at ingestion time and re-host our own copy under a stable,
content-hashed filename. That local path is what gets embedded in chunk
text, captioned (Day 3), and cited back to the user — a token rotation on
GitBook's side can never break a previously-answered citation.

Idempotent: skips download if a file with the same content hash already
exists in the manifest.
"""
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

IMAGES_DIR = DATA_DIR / "images"
MANIFEST_PATH = IMAGES_DIR / "manifest.json"


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _ext_from_content_type(content_type: str) -> str:
    ct = content_type.split(";")[0].strip().lower()
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
    }.get(ct, ".jpg")


async def _fetch_one(client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore) -> tuple[str, str | None]:
    async with semaphore:
        try:
            r = await client.get(url, timeout=30.0)
            r.raise_for_status()
            return url, r.content, r.headers.get("content-type", "image/jpeg")
        except Exception as e:
            print(f"\n  [WARN] Failed image {url}: {e}")
            return url, None, None


async def cache_images(urls: list[str], concurrency: int = 5) -> dict[str, str]:
    """
    Download each GitBook image URL once, store under data/images/<hash>.<ext>.
    Returns {gitbook_url: local_relative_path}. Skips already-cached URLs.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()

    to_fetch = [u for u in set(urls) if u not in manifest]
    if not to_fetch:
        return {u: manifest[u] for u in urls if u in manifest}

    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [_fetch_one(client, u, semaphore) for u in to_fetch]
        results = await asyncio.gather(*tasks)

    for url, content, content_type in results:
        if content is None:
            continue
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = _ext_from_content_type(content_type)
        filename = f"{url_hash}{ext}"
        (IMAGES_DIR / filename).write_bytes(content)
        manifest[url] = f"data/images/{filename}"

    _save_manifest(manifest)
    return {u: manifest[u] for u in urls if u in manifest}

"""
DAY 2 — Entry point: run the ingestion pipeline.

Usage:
  python scripts/run_ingestion.py              # crawl + parse + chunk, save preview
  python scripts/run_ingestion.py --full       # also embed + index (Day 2 end)

Steps today (--preview mode, default):
  1. crawl_all()       → download all .md pages from help.dotb.vn/llms.txt
  2. parse_document()  → extract metadata + GitBook preprocessing
  3. chunk_document()  → split into parent + child chunks
  → saves data/chunks_preview.json for manual review

Idempotent: safe to re-run. Only changed pages re-embedded (when --full).
"""
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.crawler import crawl_all
from ingestion.image_cache import cache_images
from ingestion.parser import parse_document
from ingestion.chunker import chunk_document
from ingestion.indexer import index_chunks
from config import DATA_DIR


PREVIEW_PATH = DATA_DIR / "chunks_preview.json"


def _chunk_summary(chunks: list) -> dict:
    children = [c for c in chunks if c.chunk_type == "child"]
    parents  = [c for c in chunks if c.chunk_type == "parent"]
    token_counts = [c.token_count for c in children]
    return {
        "total_chunks": len(chunks),
        "children": len(children),
        "parents": len(parents),
        "child_tokens_min": min(token_counts) if token_counts else 0,
        "child_tokens_max": max(token_counts) if token_counts else 0,
        "child_tokens_avg": int(sum(token_counts) / len(token_counts)) if token_counts else 0,
    }


async def main(full: bool = False) -> None:
    print("=== DotB RAG Ingestion Pipeline ===")

    # ── Step 1: Crawl ──────────────────────────────────────────────────────────
    print("\n[1/3] Crawling help.dotb.vn...")
    pages = await crawl_all(save_raw=True)
    if not pages:
        print("  ERROR: no pages crawled. Check LLMS_TXT_URL and network.")
        sys.exit(1)

    # ── Step 1b: Cache images ──────────────────────────────────────────────────
    print("\n[1b/3] Caching images (GitBook proxy → local)...")
    all_gitbook_urls = [url for p in pages for url in p.image_urls]
    image_map = await cache_images(all_gitbook_urls) if all_gitbook_urls else {}
    print(f"  Cached {len(image_map)} unique images")

    # ── Step 2: Parse ──────────────────────────────────────────────────────────
    print("\n[2/3] Parsing documents...")
    documents = []
    total_images = 0
    for p in pages:
        resolved = [image_map[u] for u in p.image_urls if u in image_map]
        doc = parse_document(p.url, p.title, p.description, p.content, resolved_images=resolved)
        documents.append(doc)
        total_images += len(doc.image_urls)
    print(f"  Parsed {len(documents)} documents")
    print(f"  Found {total_images} images (will caption on Day 3)")

    # ── Step 3: Chunk ──────────────────────────────────────────────────────────
    print("\n[3/3] Chunking...")
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    summary = _chunk_summary(all_chunks)
    print(f"  {summary['total_chunks']} total chunks "
          f"({summary['children']} children, {summary['parents']} parents)")
    print(f"  Child token range: {summary['child_tokens_min']}–{summary['child_tokens_max']} "
          f"(avg {summary['child_tokens_avg']})")

    # ── Save preview ───────────────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    preview = {
        "summary": summary,
        "sample_children": [
            asdict(c) for c in all_chunks
            if c.chunk_type == "child"
        ][:20],  # first 20 children for review
        "sample_parents": [
            asdict(c) for c in all_chunks
            if c.chunk_type == "parent"
        ][:5],   # first 5 parents for review
    }
    # convert breadcrumb list to list (already is, but ensure JSON-safe)
    PREVIEW_PATH.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Preview saved → {PREVIEW_PATH}")
    print("  Open data/chunks_preview.json to review chunk quality before embedding.")

    if full:
        print("\n[--full] Embedding & indexing...")
        result = index_chunks(all_chunks)
        print(f"  Indexed {result['children_indexed']} children, "
              f"{result['parents_indexed']} parents "
              f"({result['total_chunks']} total chunks)")
        print("\n=== Ingestion complete ===")
    else:
        print("\n=== Stopped after chunking — review chunks_preview.json ===")


if __name__ == "__main__":
    full_mode = "--full" in sys.argv
    asyncio.run(main(full=full_mode))

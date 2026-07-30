"""
Entry point: run the ingestion pipeline.

Usage:
  python scripts/run_ingestion.py              # crawl + parse + chunk, save preview
  python scripts/run_ingestion.py --caption    # also generate new image captions (slow, rate-limited)
  python scripts/run_ingestion.py --full       # also embed + index

Pipeline: crawl -> cache images -> parse -> [caption] -> chunk -> [embed + index]
"""
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.crawler import crawl_all
from ingestion.image_cache import cache_images
from ingestion.parser import parse_document
from ingestion.image_captioner import caption_document
from ingestion.chunker import chunk_document
from ingestion.indexer import index_chunks
from retrieval.bm25_index import build_bm25_index
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


async def main(full: bool = False, caption: bool = False) -> None:
    print("=== DotB RAG Ingestion Pipeline ===")

    # ── Step 1: Crawl ──────────────────────────────────────────────────────────
    print("\n[1/4] Crawling help.dotb.vn...")
    pages = await crawl_all(save_raw=True)
    if not pages:
        print("  ERROR: no pages crawled. Check LLMS_TXT_URL and network.")
        sys.exit(1)

    # ── Step 1b: Cache images ──────────────────────────────────────────────────
    print("\n[1b/4] Caching images (GitBook proxy → local)...")
    all_gitbook_urls = [url for p in pages for url in p.image_urls]
    image_map = await cache_images(all_gitbook_urls) if all_gitbook_urls else {}
    print(f"  Cached {len(image_map)} unique images")

    # ── Step 2: Parse ──────────────────────────────────────────────────────────
    print("\n[2/4] Parsing documents...")
    documents = []
    total_images = 0
    for p in pages:
        resolved = [image_map[u] for u in p.image_urls if u in image_map]
        doc = parse_document(p.url, p.title, p.description, p.content, resolved_images=resolved)
        documents.append(doc)
        total_images += len(doc.image_urls)
    print(f"  Parsed {len(documents)} documents")
    print(f"  Found {total_images} images")

    # ── Step 2b: Caption images ─────────────────────────────────────────────────
    label = "generating new + applying cached" if caption else "applying cached only"
    print(f"\n[2b/4] Captioning images ({label})...")
    captioned = 0
    for doc in tqdm(documents, desc="  Captioning"):
        new_content = caption_document(doc.content, fallback_context=doc.title, allow_generate=caption)
        if new_content != doc.content:
            captioned += 1
        doc.content = new_content
    print(f"  {captioned} documents had a caption applied")
    if not caption:
        print("  (pass --caption to generate captions for cache misses — rate-limited, ~70-90 min for all images)")

    # ── Step 3: Chunk ──────────────────────────────────────────────────────────
    print("\n[3/4] Chunking...")
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
        # A sample, not a dump: the preview exists to be read by a human before
        # spending embedding quota, and 600+ chunks is not readable.
        "sample_children": [
            asdict(c) for c in all_chunks if c.chunk_type == "child"
        ][:20],
        "sample_parents": [
            asdict(c) for c in all_chunks if c.chunk_type == "parent"
        ][:5],
    }
    PREVIEW_PATH.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Preview saved → {PREVIEW_PATH}")
    print("  Open data/chunks_preview.json to review chunk quality before embedding.")

    if full:
        print("\n[4/4] Embedding & indexing...")
        result = index_chunks(all_chunks)
        print(f"  Indexed {result['children_indexed']} children "
              f"({result['children_skipped']} already up to date, "
              f"{result['children_purged']} stale purged), "
              f"{result['parents_indexed']} parents "
              f"({result['parents_purged']} stale purged) "
              f"({result['total_chunks']} total chunks)")

        print("  Rebuilding BM25 index from ChromaDB...")
        bm25_index = build_bm25_index()
        print(f"  BM25 corpus: {len(bm25_index['chunk_ids'])} children")
        print("\n=== Ingestion complete ===")
    else:
        print("\n=== Stopped after chunking — review chunks_preview.json ===")


if __name__ == "__main__":
    full_mode = "--full" in sys.argv
    caption_mode = "--caption" in sys.argv
    asyncio.run(main(full=full_mode, caption=caption_mode))

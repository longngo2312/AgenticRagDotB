"""
DAY 2 — Entry point: run the full ingestion pipeline.

Usage:
  python scripts/run_ingestion.py

Steps:
  1. crawl_all()       → download all .md pages from help.dotb.vn/llms.txt
  2. parse_document()  → extract structured metadata from each page
  3. chunk_document()  → split into parent + child chunks
  4. index_chunks()    → embed children → ChromaDB; parents → JSON file

Idempotent: safe to re-run. Only changed pages (new content_hash) are re-embedded.
"""
import asyncio
import sys
from pathlib import Path

# Make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO D2-27: from ingestion.crawler import crawl_all
# TODO D2-27: from ingestion.parser import parse_document
# TODO D2-27: from ingestion.chunker import chunk_document
# TODO D2-27: from ingestion.indexer import index_chunks

# TODO D2-28: async def main():
#   print("=== DotB RAG Ingestion Pipeline ===")
#
#   # Step 1: Crawl
#   print("\n[1/4] Crawling help.dotb.vn...")
#   pages = await crawl_all(save_raw=True)
#
#   # Step 2: Parse
#   print("\n[2/4] Parsing documents...")
#   documents = [parse_document(p.url, p.title, p.description, p.content) for p in pages]
#   print(f"  Parsed {len(documents)} documents")
#
#   # Step 3: Chunk
#   print("\n[3/4] Chunking...")
#   all_chunks = []
#   for doc in documents:
#       all_chunks.extend(chunk_document(doc))
#   n_children = sum(1 for c in all_chunks if c.chunk_type == 'child')
#   n_parents  = sum(1 for c in all_chunks if c.chunk_type == 'parent')
#   print(f"  {len(all_chunks)} total chunks ({n_children} children, {n_parents} parents)")
#
#   # Step 4: Index
#   print("\n[4/4] Embedding & indexing...")
#   stats = index_chunks(all_chunks)
#   print(f"  Done: {stats}")
#
#   print("\n=== Ingestion complete ===")

# TODO D2-29: if __name__ == '__main__':
#                 asyncio.run(main())

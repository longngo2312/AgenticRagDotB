"""
DAY 2 — STEP 5: Upsert chunks into storage

Two stores:
  1. ChromaDB  (data/chroma_db/)  — child chunks with embeddings, fast vector search
  2. JSON file (data/parent_docs.json) — parent docs keyed by parent_id, no embedding needed
     (parents are fetched by ID after child retrieval, not by similarity)

Idempotency: chunks are upserted by chunk_id.
  - ChromaDB.upsert() is a no-op if the ID already exists with the same content.
  - The run_ingestion script compares content_hash before calling this to skip
    unchanged documents entirely.
"""
import json
import sys
from pathlib import Path

import chromadb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, CHROMA_CHILD_COLLECTION, PARENTS_PATH
from ingestion.chunker import Chunk
from ingestion.embedder import embed_texts


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _chunk_to_metadata(chunk: Chunk) -> dict:
    return {
        "parent_id": chunk.parent_id or "",
        "chunk_type": chunk.chunk_type,
        "doc_url": chunk.doc_url,
        "doc_title": chunk.doc_title,
        "breadcrumb": json.dumps(chunk.breadcrumb, ensure_ascii=False),
        "section": chunk.section,
        "product": chunk.product,
        "heading_path": chunk.heading_path,
        "raw_text": chunk.raw_text,
        "token_count": chunk.token_count,
        "content_hash": chunk.content_hash,
    }


def load_parents() -> dict:
    if PARENTS_PATH.exists():
        return json.loads(PARENTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_parents(parents: dict) -> None:
    PARENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARENTS_PATH.write_text(json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8")


def index_chunks(chunks: list[Chunk], batch_size: int = 20) -> dict:
    children = [c for c in chunks if c.chunk_type == "child"]
    parents = [c for c in chunks if c.chunk_type == "parent"]

    # ── Children → ChromaDB ────────────────────────────────────────────────────
    client = get_chroma_client()
    col = client.get_or_create_collection(
        CHROMA_CHILD_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Resumable: chunk_id is a content hash, so anything already in the
    # collection is unchanged and can be skipped (saves scarce embedding quota).
    existing_ids = set(col.get(include=[])["ids"])
    todo = [c for c in children if c.chunk_id not in existing_ids]
    skipped = len(children) - len(todo)
    if skipped:
        print(f"  Skipping {skipped} children already indexed (unchanged)")

    for i in tqdm(range(0, len(todo), batch_size), desc="  Embedding+indexing children"):
        batch = todo[i : i + batch_size]
        texts = [c.content for c in batch]
        embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
        col.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[_chunk_to_metadata(c) for c in batch],
        )

    # ── Parents → JSON file ─────────────────────────────────────────────────────
    parents_store = load_parents()
    for p in parents:
        parents_store[p.chunk_id] = {
            "content": p.content,
            "metadata": _chunk_to_metadata(p),
        }
    save_parents(parents_store)

    return {
        "children_indexed": len(todo),
        "children_skipped": skipped,
        "parents_indexed": len(parents),
        "total_chunks": len(chunks),
    }

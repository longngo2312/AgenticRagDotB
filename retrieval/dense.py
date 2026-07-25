"""DAY 4 — Dense vector search via ChromaDB (HNSW cosine)."""
import json
import sys
from pathlib import Path

import chromadb
import httpx
from google.genai.errors import APIError

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, CHROMA_CHILD_COLLECTION, DENSE_TOP_K
from ingestion.embedder import embed_query

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_collection(CHROMA_CHILD_COLLECTION)
    return _collection


def dense_search(query: str, top_k: int = DENSE_TOP_K) -> list[dict]:
    """Embed `query` and return the top-k nearest children by cosine distance,
    ranked closest first. Each result carries chunk_id/content/metadata/score
    so it's directly fusable with bm25_search results in hybrid.rrf_fuse.

    Interactive path: embed_query already fails fast (no multi-minute
    ingestion-style backoff). If the embedding call still fails — rate limit
    or network — degrade to an empty list rather than raising, so hybrid
    search falls back to BM25-only instead of blocking or crashing the turn."""
    col = _get_collection()
    try:
        vector = embed_query(query)
    except (APIError, httpx.TransportError, httpx.TimeoutException) as e:
        print(f"\n  [dense-search] embedding failed ({type(e).__name__}), falling back to BM25-only.")
        return []
    result = col.query(
        query_embeddings=[vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    out = []
    for rank, (chunk_id, content, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        metadata = dict(metadata)
        metadata["breadcrumb"] = json.loads(metadata["breadcrumb"])
        metadata["image_paths"] = json.loads(metadata["image_paths"])
        out.append({
            "chunk_id": chunk_id,
            "content": content,
            "metadata": metadata,
            "score": 1 - distance,  # cosine distance -> similarity, informational only
            "rank": rank,
        })
    return out

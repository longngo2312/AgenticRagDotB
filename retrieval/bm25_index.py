"""
DAY 4 — BM25 lexical search using rank_bm25.

The corpus is built directly from ChromaDB's own children (their raw_text
metadata), not re-derived from chunks separately — one source of truth, so
BM25 and dense search can never drift out of sync with each other.

Tokenizer: lowercase + Unicode \\w+ regex. Python's `re` is Unicode-aware for
str patterns by default, so Vietnamese diacritic letters count as word
characters and split correctly on whitespace/punctuation — no dedicated
Vietnamese segmenter needed for a BM25 baseline.
"""
import json
import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, CHROMA_CHILD_COLLECTION, BM25_INDEX_PATH, BM25_TOP_K

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_bm25_index() -> dict:
    """Pull every child from ChromaDB, tokenize its raw_text, build a BM25Okapi
    index, and pickle-cache it (index + parallel chunk_id/content/metadata
    lists) to BM25_INDEX_PATH so queries don't rebuild it every time."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(CHROMA_CHILD_COLLECTION)
    data = col.get(include=["documents", "metadatas"])

    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    corpus_tokens = [_tokenize(m["raw_text"]) for m in metadatas]
    bm25 = BM25Okapi(corpus_tokens)

    index = {
        "bm25": bm25,
        "chunk_ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    return index


def _load_index() -> dict:
    if BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return build_bm25_index()


_index: dict | None = None


def bm25_search(query: str, top_k: int = BM25_TOP_K) -> list[dict]:
    global _index
    if _index is None:
        _index = _load_index()

    scores = _index["bm25"].get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    out = []
    for rank, i in enumerate(ranked, start=1):
        if scores[i] <= 0:
            break  # BM25 score 0 means no lexical overlap at all — not a real match
        metadata = dict(_index["metadatas"][i])
        metadata["breadcrumb"] = json.loads(metadata["breadcrumb"])
        metadata["image_paths"] = json.loads(metadata["image_paths"])
        out.append({
            "chunk_id": _index["chunk_ids"][i],
            "content": _index["documents"][i],
            "metadata": metadata,
            "score": float(scores[i]),
            "rank": rank,
        })
    return out

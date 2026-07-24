"""DAY 4 — Cross-encoder reranker (BAAI/bge-reranker-v2-m3, local, free)."""
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RERANKER_MODEL, RERANK_TOP_K, RERANK_SCORE_THRESHOLD

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """Score each (query, chunk.content) pair with the cross-encoder, sort
    descending, and drop anything below RERANK_SCORE_THRESHOLD — a low score
    there means neither dense nor BM25's candidates actually answer the
    query, which the caller should treat as an abstain signal, not force-feed
    the top-k anyway."""
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c["content"]) for c in candidates]
    raw_scores = model.predict(pairs)

    for c, score in zip(candidates, raw_scores):
        c["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    ranked = [c for c in ranked if c["rerank_score"] >= RERANK_SCORE_THRESHOLD][:top_k]
    for i, c in enumerate(ranked, start=1):
        c["rank"] = i
    return ranked

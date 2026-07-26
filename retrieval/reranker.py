"""DAY 4 — Cross-encoder reranker (BAAI/bge-reranker-v2-m3, local, free)."""
import sys
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RERANKER_MODEL, RERANK_TOP_K, RERANK_SCORE_THRESHOLD,
    RERANK_MAX_SEQ_LEN, RERANK_BATCH_SIZE,
)

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        # max_length caps the default 8192 → 512 (our child chunks are ~500 tok);
        # fp16 halves precision on GPU. Together: ~9s → ~1.8s per rerank, no
        # measurable quality loss. fp16 only on CUDA — half() is unsupported/slow
        # on CPU, so fall back to full precision there.
        _model = CrossEncoder(RERANKER_MODEL, max_length=RERANK_MAX_SEQ_LEN)
        if torch.cuda.is_available():
            _model.model.half()
    return _model


def warmup() -> None:
    """Load the ~1.1GB cross-encoder AND run one dummy prediction now, so both
    the model load and the first-inference CUDA-kernel compilation land at
    startup instead of on the user's first query. Call this once when a
    long-lived process (chat CLI, API server) boots — without the dummy
    predict, the model is loaded but the first real rerank still eats several
    seconds of kernel warm-up and blows the <4s latency target on cold start."""
    _get_model().predict([("warmup", "warmup")])


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
    raw_scores = model.predict(pairs, batch_size=RERANK_BATCH_SIZE)

    for c, score in zip(candidates, raw_scores):
        c["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    ranked = [c for c in ranked if c["rerank_score"] >= RERANK_SCORE_THRESHOLD][:top_k]
    for i, c in enumerate(ranked, start=1):
        c["rank"] = i
    return ranked

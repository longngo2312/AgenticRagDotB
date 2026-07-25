"""DAY 4 — Reciprocal Rank Fusion (RRF) of dense + BM25 results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RRF_K, DENSE_TOP_K, BM25_TOP_K
from retrieval.dense import dense_search
from retrieval.bm25_index import bm25_search


def rrf_fuse(dense_results: list[dict], bm25_results: list[dict], k: int = RRF_K) -> list[dict]:
    """score(d) = Σ 1/(k + rank_i(d)) over each ranking the chunk appears in.
    A chunk found by only one method still scores — being found at all by
    either search is exactly the point of running both."""
    scores: dict[str, float] = {}
    info: dict[str, dict] = {}

    for results in (dense_results, bm25_results):
        for r in results:
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r["rank"])
            info.setdefault(cid, r)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        {**info[cid], "rrf_score": scores[cid], "rank": rank}
        for rank, cid in enumerate(ranked_ids, start=1)
    ]


def hybrid_search(
    query: str,
    bm25_query: str | None = None,
    dense_top_k: int = DENSE_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
) -> list[dict]:
    """Run dense + BM25 in parallel search spaces and fuse via RRF. This is
    the hybrid retrieval deliverable itself — reranking and parent-fetch are
    the full retriever.py pipeline's job, not this function's.

    `bm25_query` lets the caller pass a glossary-expanded variant to BM25
    while dense search gets the clean, un-expanded `query` — see
    rewrite.RewrittenQuery for why they're kept separate. Defaults to `query`
    for direct/standalone callers that don't need the distinction."""
    dense_results = dense_search(query, top_k=dense_top_k)
    bm25_results = bm25_search(bm25_query if bm25_query is not None else query, top_k=bm25_top_k)
    return rrf_fuse(dense_results, bm25_results)

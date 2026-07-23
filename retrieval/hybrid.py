"""DAY 4 — Reciprocal Rank Fusion (RRF) of dense + BM25 results."""
# TODO D4: def rrf_fuse(dense_results, bm25_results, k=RRF_K) -> list[dict]
#          score(d) = Σ 1/(k + rank_i(d))  for each ranking list

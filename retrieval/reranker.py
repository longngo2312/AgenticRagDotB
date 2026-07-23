"""DAY 4 — Cross-encoder reranker (BAAI/bge-reranker-v2-m3, local, free)."""
# TODO D4: load CrossEncoder(RERANKER_MODEL) once at module level
#          def rerank(query: str, candidates: list[dict], top_k=RERANK_TOP_K) -> list[dict]
#          score each (query, chunk.content) pair → sort → return top_k

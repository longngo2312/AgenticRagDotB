"""DAY 4 — Full retrieval pipeline: rewrite → hybrid → rerank → fetch parents."""
# TODO D4: def retrieve(query: str, chat_history: list, filters: dict = None) -> list[dict]
#   1. rewrite query (condense history + glossary expansion)
#   2. dense_search + bm25_search → top-30 each
#   3. rrf_fuse → merged list
#   4. rerank → top-5 children
#   5. fetch parent docs from parent_docs.json by parent_id
#   6. return parent docs (with source metadata for citation)

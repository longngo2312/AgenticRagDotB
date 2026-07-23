"""
DAY 2 — STEP 4: Batch embedding via Google text-embedding-004

Free tier limits (as of 2026):
  - 100 requests/minute
  - 1,500 requests/day
  - Batch up to 100 texts per call

task_type must differ for indexing vs querying:
  - RETRIEVAL_DOCUMENT  → when embedding chunks for the index
  - RETRIEVAL_QUERY     → when embedding a user question at runtime
"""
# TODO D2-18: import time, google.generativeai as genai
#             import GOOGLE_API_KEY, EMBEDDING_MODEL, EMBED_BATCH_SIZE, EMBED_RETRY_DELAY_SEC

# TODO D2-19: genai.configure(api_key=GOOGLE_API_KEY) at module level

# TODO D2-20: def embed_texts(texts: list[str], task_type="RETRIEVAL_DOCUMENT") -> list[list[float]]
#   - loop in steps of EMBED_BATCH_SIZE
#   - call genai.embed_content(model=EMBEDDING_MODEL, content=batch, task_type=task_type)
#   - on exception: sleep EMBED_RETRY_DELAY_SEC, retry once, then raise
#   - sleep 0.5s between batches to respect rate limit
#   - return flat list of embedding vectors

# TODO D2-21: def embed_query(text: str) -> list[float]
#   - single call with task_type="RETRIEVAL_QUERY"
#   - returns one embedding vector (list[float])

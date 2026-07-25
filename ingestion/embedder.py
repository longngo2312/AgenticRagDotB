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
import sys
import time
from pathlib import Path

import httpx
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GOOGLE_API_KEY, EMBEDDING_MODEL, EMBED_BATCH_SIZE

_client = genai.Client(api_key=GOOGLE_API_KEY)

# gemini-embedding-001 free tier enforces a low per-minute rate/token cap.
# On 429 we pause a full minute (clears the per-minute window) and retry.
_RATE_LIMIT_BACKOFF_SEC = [60, 60, 120, 120]
# Transient network failures (timeouts, connection resets) — distinct from quota.
_NETWORK_BACKOFF_SEC = [15, 30, 60]
_RETRYABLE_NETWORK_ERRORS = (httpx.TransportError, httpx.TimeoutException, ServerError)

# Interactive query path: the <4s chat SLA can't sit in the ingestion backoff
# (up to 6 min). Fail fast on a rate limit — the caller degrades to BM25-only —
# and allow just one quick retry for a transient network blip.
_QUERY_RATE_LIMIT_BACKOFF_SEC: list[int] = []
_QUERY_NETWORK_BACKOFF_SEC = [1]


def _embed_batch(
    batch: list[str],
    task_type: str,
    rate_limit_backoff: list[int] = _RATE_LIMIT_BACKOFF_SEC,
    network_backoff: list[int] = _NETWORK_BACKOFF_SEC,
) -> list[list[float]]:
    config = types.EmbedContentConfig(task_type=task_type)
    rate_limit_retries = 0
    network_retries = 0
    while True:
        try:
            resp = _client.models.embed_content(
                model=EMBEDDING_MODEL, contents=batch, config=config
            )
            return [e.values for e in resp.embeddings]
        except ClientError as e:
            if e.code == 429 and rate_limit_retries < len(rate_limit_backoff):
                backoff = rate_limit_backoff[rate_limit_retries]
                rate_limit_retries += 1
                print(f"\n  [rate-limit] waiting {backoff}s before retry {rate_limit_retries}...")
                time.sleep(backoff)
                continue
            raise
        except _RETRYABLE_NETWORK_ERRORS as e:
            if network_retries < len(network_backoff):
                backoff = network_backoff[network_retries]
                network_retries += 1
                print(f"\n  [network] {type(e).__name__}, waiting {backoff}s before retry {network_retries}...")
                time.sleep(backoff)
                continue
            raise


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vectors.extend(_embed_batch(batch, task_type))
        if i + EMBED_BATCH_SIZE < len(texts):
            time.sleep(1.0)
    return vectors


def embed_query(text: str) -> list[float]:
    """Interactive path (chat turn): fails fast on rate limits instead of the
    ingestion-style multi-minute backoff — see retrieval/dense.py, which
    catches failures here and degrades to BM25-only rather than blocking."""
    return _embed_batch(
        [text],
        "RETRIEVAL_QUERY",
        rate_limit_backoff=_QUERY_RATE_LIMIT_BACKOFF_SEC,
        network_backoff=_QUERY_NETWORK_BACKOFF_SEC,
    )[0]

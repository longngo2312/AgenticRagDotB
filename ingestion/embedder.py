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

from google import genai
from google.genai import types
from google.genai.errors import ClientError

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GOOGLE_API_KEY, EMBEDDING_MODEL, EMBED_BATCH_SIZE

_client = genai.Client(api_key=GOOGLE_API_KEY)

# gemini-embedding-001 free tier enforces a low per-minute rate/token cap.
# On 429 we pause a full minute (clears the per-minute window) and retry.
_RATE_LIMIT_BACKOFF_SEC = [60, 60, 120, 120]


def _embed_batch(batch: list[str], task_type: str) -> list[list[float]]:
    config = types.EmbedContentConfig(task_type=task_type)
    for attempt, backoff in enumerate([0, *_RATE_LIMIT_BACKOFF_SEC]):
        if backoff:
            print(f"\n  [rate-limit] waiting {backoff}s before retry {attempt}...")
            time.sleep(backoff)
        try:
            resp = _client.models.embed_content(
                model=EMBEDDING_MODEL, contents=batch, config=config
            )
            return [e.values for e in resp.embeddings]
        except ClientError as e:
            if e.code == 429 and attempt < len(_RATE_LIMIT_BACKOFF_SEC):
                continue
            raise
    raise RuntimeError("unreachable")


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vectors.extend(_embed_batch(batch, task_type))
        if i + EMBED_BATCH_SIZE < len(texts):
            time.sleep(1.0)
    return vectors


def embed_query(text: str) -> list[float]:
    return _embed_batch([text], "RETRIEVAL_QUERY")[0]

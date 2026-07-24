"""
Central config — all tuneable knobs live here.
Change a setting here and it propagates everywhere.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
RAW_DOCS_DIR    = DATA_DIR / "raw_docs"
CHROMA_DIR      = DATA_DIR / "chroma_db"
GLOSSARY_PATH   = DATA_DIR / "glossary.json"
PARENTS_PATH    = DATA_DIR / "parent_docs.json"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"

# ── Google Gemini ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
# text-embedding-004 was retired from the Gemini Developer API; gemini-embedding-001
# is Google's current multilingual embedding model (3072-dim, free tier).
EMBEDDING_MODEL = "models/gemini-embedding-001"
# gemini-1.5-flash was also retired; gemini-flash-latest is the current free-tier alias.
LLM_MODEL       = "models/gemini-flash-latest"

# ── Crawling ──────────────────────────────────────────────────────────────────
LLMS_TXT_URL        = "https://help.dotb.vn/llms.txt"
CRAWLER_CONCURRENCY = 5      # parallel downloads
CRAWLER_DELAY_SEC   = 0.3    # delay between requests per worker

# ── Chunking ──────────────────────────────────────────────────────────────────
CHILD_CHUNK_MAX_TOKENS  = 500
PARENT_CHUNK_MAX_TOKENS = 1500

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_CHILD_COLLECTION = "dotb_child_chunks"

# ── Retrieval (Day 4) ─────────────────────────────────────────────────────────
DENSE_TOP_K             = 30
BM25_TOP_K              = 30
RERANK_TOP_K            = 5
RRF_K                   = 60    # RRF constant — 60 is standard
RERANKER_MODEL          = "BAAI/bge-reranker-v2-m3"
RERANK_SCORE_THRESHOLD  = 0.1   # below this → abstain

# ── Query rewrite (Day 4) ──────────────────────────────────────────────────────
REWRITE_HISTORY_TURNS = 3   # prior user/assistant turns fed to the condenser

# ── Embedding API ─────────────────────────────────────────────────────────────
EMBED_BATCH_SIZE      = 100
EMBED_RETRY_DELAY_SEC = 2.0

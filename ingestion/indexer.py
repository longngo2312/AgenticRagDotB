"""
DAY 2 — STEP 5: Upsert chunks into storage

Two stores:
  1. ChromaDB  (data/chroma_db/)  — child chunks with embeddings, fast vector search
  2. JSON file (data/parent_docs.json) — parent docs keyed by parent_id, no embedding needed
     (parents are fetched by ID after child retrieval, not by similarity)

Idempotency: chunks are upserted by chunk_id.
  - ChromaDB.upsert() is a no-op if the ID already exists with the same content.
  - The run_ingestion script compares content_hash before calling this to skip
    unchanged documents entirely.
"""
# TODO D2-22: import json, chromadb, tqdm
#             import Chunk from ingestion.chunker
#             import embed_texts from ingestion.embedder
#             import CHROMA_DIR, CHROMA_CHILD_COLLECTION, PARENTS_PATH from config

# TODO D2-23: def get_chroma_client() -> chromadb.PersistentClient
#   - CHROMA_DIR.mkdir(parents=True, exist_ok=True)
#   - return chromadb.PersistentClient(path=str(CHROMA_DIR))

# TODO D2-24: def _chunk_to_metadata(chunk: Chunk) -> dict
#   - ChromaDB metadata must be flat (str/int/float only)
#   - serialize breadcrumb list as json.dumps(chunk.breadcrumb)
#   - include: parent_id, chunk_type, doc_url, doc_title, breadcrumb (json str),
#              section, product, heading_path, raw_text, token_count, content_hash

# TODO D2-25: def load_parents() -> dict  /  def save_parents(parents: dict)
#   - load/save PARENTS_PATH as JSON (dict keyed by parent_id → {content, metadata})
#   - return empty dict if file doesn't exist

# TODO D2-26: def index_chunks(chunks: list[Chunk], batch_size=50) -> dict
#   - split into children and parents
#   - CHILDREN → ChromaDB:
#       * get_or_create_collection(CHROMA_CHILD_COLLECTION, metadata={'hnsw:space':'cosine'})
#       * batch loop with tqdm: embed texts → col.upsert(ids, embeddings, documents, metadatas)
#   - PARENTS → JSON file:
#       * load existing parents dict
#       * upsert by chunk_id: {content: chunk.content, metadata: _chunk_to_metadata(chunk)}
#       * save_parents()
#   - return {'children_indexed': N, 'parents_indexed': M, 'total_chunks': T}

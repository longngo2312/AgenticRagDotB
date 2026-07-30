"""DAY 6 — FastAPI server backing the chat UI and the metrics/pipeline dashboard.

Endpoints:
  GET  /health                    liveness check
  POST /api/chat                  one turn through the LangGraph agent
  POST /api/feedback              thumbs up/down on an answer → logs/feedback.jsonl
  POST /api/handoff               user-initiated "talk to a person"
  GET  /api/eval/latest           last `python eval/evaluate.py` snapshot
  GET  /api/pipeline/agent-graph  the compiled LangGraph's real nodes/edges
  GET  /api/pipeline/ingestion    live stats from the actual index artifacts
  GET  /api/data-model            chunk schema + AgentState fields + config knobs
  /                               static chat + dashboard frontend

/api/chat is request/response, not streaming — see the README's "faithfulness
check is a gate" note for why.
"""
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    BASE_DIR, BM25_INDEX_PATH, CHROMA_DIR, CHROMA_CHILD_COLLECTION,
    CHILD_CHUNK_MAX_TOKENS, PARENT_CHUNK_MAX_TOKENS, DENSE_TOP_K, BM25_TOP_K,
    RERANK_TOP_K, RRF_K, RERANK_SCORE_THRESHOLD, AGENT_MAX_ATTEMPTS,
    AGENT_FAITHFULNESS_MIN, EMBEDDING_MODEL, LLM_MODEL,
)
from agent.graph import app as agent_app, ask as agent_ask
from agent.nodes import UNVERIFIED_CONFIDENCE
from agent.prompts import HANDOFF_MESSAGE
from agent.tools import create_handoff
from ingestion.indexer import load_parents
from retrieval import reranker

EVAL_RESULTS_PATH = BASE_DIR / "eval" / "results" / "latest.json"
FEEDBACK_LOG_PATH = BASE_DIR / "logs" / "feedback.jsonl"
FRONTEND_DIR = BASE_DIR / "frontend"



def _browsable_url(doc_url: str) -> str:
    """Convert an indexed doc_url into one a human can open.

    doc_url is always the crawled `.md` fetch URL (see eval/evaluate.py's
    _with_md docstring); help.dotb.vn serves the same page without that
    suffix, and that's the link worth putting in a citation.
    """
    return doc_url.removesuffix(".md")


app = FastAPI(title="DotB RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warmup() -> None:
    reranker.warmup()  # load the cross-encoder once, not on the first chat request


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Chat ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    messages: list[dict] = []


class Source(BaseModel):
    title: str
    url: str
    breadcrumb: list[str]
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    confidence: float | None = None
    handoff: bool
    handoff_reason: str | None = None
    sources: list[Source]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    state = agent_ask(req.question, req.messages)
    confidence = state.get("confidence")
    docs = state.get("docs") or []

    # Mirrors scripts/agent_cli.py: only cite sources on a path that actually
    # answered from them — a handoff or clarify reply isn't grounded in docs
    # it happened to see, and an unverified/failed faithfulness check means
    # the citations can't be vouched for either.
    show_sources = (
        bool(docs) and confidence is not None
        and confidence != UNVERIFIED_CONFIDENCE
        and confidence >= AGENT_FAITHFULNESS_MIN
    )
    sources = [
        Source(
            title=d["metadata"].get("doc_title", ""),
            url=_browsable_url(d["metadata"].get("doc_url", "")),
            breadcrumb=d["metadata"].get("breadcrumb", []),
            snippet=d.get("matched_child", "")[:240],
        )
        for d in docs
    ] if show_sources else []

    return ChatResponse(
        answer=state.get("answer", ""),
        route=state.get("route", ""),
        confidence=None if confidence == UNVERIFIED_CONFIDENCE else confidence,
        handoff=state.get("route") == "handoff",
        handoff_reason=state.get("handoff_reason"),
        sources=sources,
    )


# ── Feedback ─────────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    vote: str  # "up" | "down"


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if req.vote not in ("up", "down"):
        raise HTTPException(status_code=400, detail="vote must be 'up' or 'down'")
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **req.model_dump()}
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"ok": True}


# ── Manual "talk to a person" handoff ───────────────────────────────────────
class HandoffRequest(BaseModel):
    question: str = ""


@app.post("/api/handoff")
def handoff(req: HandoffRequest):
    """User-initiated handoff (the chat's 'Talk to a person' button), as
    distinct from the agent's own automatic handoff route. Reuses the same
    create_handoff tool the graph calls internally, so both paths land in the
    same logs/handoffs.jsonl for support triage."""
    reason = f"user requested handoff: {req.question}" if req.question else "user requested handoff"
    message = create_handoff.invoke({"reason": reason})
    return {"message": message or HANDOFF_MESSAGE}


# ── Eval snapshot ────────────────────────────────────────────────────────────
@app.get("/api/eval/latest")
def eval_latest():
    if not EVAL_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No eval snapshot yet — run `python eval/evaluate.py` to generate one.",
        )
    return json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))


# ── Pipeline: agent graph ────────────────────────────────────────────────────
@app.get("/api/pipeline/agent-graph")
def agent_graph():
    """Introspects the real compiled StateGraph rather than hand-describing
    it, so this can never drift out of sync with agent/graph.py.

    The loop/threshold values ride along from config so the dashboard renders
    the limits actually in force instead of hardcoding its own copy.
    """
    g = agent_app.get_graph()
    nodes = [{"id": n} for n in g.nodes]
    edges = [
        {"source": e.source, "target": e.target, "label": e.data, "conditional": e.conditional}
        for e in g.edges
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "max_attempts": AGENT_MAX_ATTEMPTS,
        "faithfulness_min": AGENT_FAITHFULNESS_MIN,
    }


# ── Pipeline: ingestion stats ────────────────────────────────────────────────
_ingestion_cache: dict | None = None


@app.get("/api/pipeline/ingestion")
def ingestion_stats():
    """Computed live from the actual persisted artifacts (parent_docs.json,
    the ChromaDB collection, the BM25 pickle) — not parsed from ingestion log
    text, which is a one-off run's console output and can go stale or drift."""
    global _ingestion_cache
    if _ingestion_cache is not None:
        return _ingestion_cache

    parents = load_parents()
    doc_urls = {p["metadata"]["doc_url"] for p in parents.values()}
    image_paths: set[str] = set()
    for p in parents.values():
        image_paths.update(json.loads(p["metadata"].get("image_paths", "[]")))

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(CHROMA_CHILD_COLLECTION)
    child_count = col.count()
    child_meta = col.get(include=["metadatas"])["metadatas"]
    token_counts = [int(m["token_count"]) for m in child_meta if m.get("token_count") is not None]

    bm25_corpus_size = None
    if BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_corpus_size = len(pickle.load(f)["chunk_ids"])

    _ingestion_cache = {
        "documents": len(doc_urls),
        "parent_chunks": len(parents),
        "child_chunks": child_count,
        "bm25_corpus_size": bm25_corpus_size,
        "child_token_range": [min(token_counts), max(token_counts)] if token_counts else None,
        "child_token_avg": round(sum(token_counts) / len(token_counts), 1) if token_counts else None,
        "images_referenced": len(image_paths),
        "stages": [
            {"name": "crawler", "description": "async-fetch all .md pages from help.dotb.vn/llms.txt"},
            {"name": "parser", "description": "URL → breadcrumb/product/section metadata"},
            {"name": "image_captioner", "description": "caption embedded screenshots (cached by image hash)"},
            {"name": "chunker", "description": "heading-aware parent-child split"},
            {"name": "embedder", "description": f"batch embed children via {EMBEDDING_MODEL}"},
            {"name": "indexer", "description": "upsert children → ChromaDB, parents → JSON"},
        ],
    }
    return _ingestion_cache


# ── Data model ───────────────────────────────────────────────────────────────
@app.get("/api/data-model")
def data_model():
    parents = load_parents()
    sample = None
    if parents:
        parent_id, entry = next(iter(parents.items()))
        meta = entry["metadata"]
        sample = {
            "parent_id": parent_id,
            "doc_title": meta.get("doc_title", ""),
            "doc_url": _browsable_url(meta.get("doc_url", "")),
            "breadcrumb": json.loads(meta.get("breadcrumb", "[]")),
            "heading_path": meta.get("heading_path", ""),
            "token_count": meta.get("token_count"),
            "content_preview": entry["content"][:400],
        }

    chunk_fields = [
        "chunk_id", "parent_id", "chunk_type", "doc_url", "doc_title", "breadcrumb",
        "section", "product", "heading_path", "content", "raw_text", "token_count",
        "content_hash", "image_paths",
    ]
    return {
        "chunk_schema": {
            "child": {"approx_tokens": CHILD_CHUNK_MAX_TOKENS, "stored_in": "ChromaDB (embedded)", "fields": chunk_fields},
            "parent": {"approx_tokens": PARENT_CHUNK_MAX_TOKENS, "stored_in": "data/parent_docs.json (no embedding)", "fields": chunk_fields},
            "sample_parent": sample,
        },
        "agent_state_fields": [
            {"name": "messages", "type": "list", "description": "prior conversation, oldest first (excludes this turn)"},
            {"name": "question", "type": "str", "description": "this turn's raw user input, verbatim"},
            {"name": "query", "type": "str", "description": "standalone (history-condensed) query used for retrieval"},
            {"name": "docs", "type": "list", "description": "retrieved + reranked parent chunks"},
            {"name": "context", "type": "str", "description": "docs formatted as numbered [Nguồn n] blocks"},
            {"name": "attempts", "type": "int", "description": "retrieve→grade loop counter"},
            {"name": "confidence", "type": "float", "description": "faithfulness self-check score, 0.0–1.0"},
            {"name": "route", "type": "str", "description": "retrieve | clarify | handoff | generate | end"},
            {"name": "answer", "type": "str", "description": "what the user actually sees, whichever path produced it"},
            {"name": "handoff_reason", "type": "str", "description": "why the agent gave up, logged for support triage"},
        ],
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "child_chunk_max_tokens": CHILD_CHUNK_MAX_TOKENS,
            "parent_chunk_max_tokens": PARENT_CHUNK_MAX_TOKENS,
            "dense_top_k": DENSE_TOP_K,
            "bm25_top_k": BM25_TOP_K,
            "rrf_k": RRF_K,
            "rerank_top_k": RERANK_TOP_K,
            "rerank_score_threshold": RERANK_SCORE_THRESHOLD,
            "agent_max_attempts": AGENT_MAX_ATTEMPTS,
            "agent_faithfulness_min": AGENT_FAITHFULNESS_MIN,
        },
    }


# Mounted last so it never shadows the /api/* and /health routes above.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

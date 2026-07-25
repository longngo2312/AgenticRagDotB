"""DAY 4 — Full retrieval pipeline: rewrite → hybrid → rerank → fetch parents."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RERANK_TOP_K
from retrieval.rewrite import rewrite_query
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from ingestion.indexer import load_parents


def _apply_filters(candidates: list[dict], filters: dict | None) -> list[dict]:
    if not filters:
        return candidates
    return [
        c for c in candidates
        if all(c["metadata"].get(k) == v for k, v in filters.items())
    ]


_parents_cache: dict | None = None


def _get_parents() -> dict:
    # Cached for the process lifetime — re-reading and JSON-parsing the full
    # parent store from disk on every chat turn is avoidable latency on the
    # interactive path. A long-running server (Day 6) that re-ingests while
    # live will need an explicit reload hook; out of scope for the CLI.
    global _parents_cache
    if _parents_cache is None:
        _parents_cache = load_parents()
    return _parents_cache


def retrieve(
    query: str,
    chat_history: list[dict] | None = None,
    filters: dict | None = None,
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
    """rewrite -> hybrid -> rerank -> fetch parents.

    Returns parent docs (full context for generation), deduplicated by
    parent_id and ordered by rerank score — the same reranked child can point
    at the same parent as another, and the LLM only needs it once. Each
    result also carries `matched_child`, the specific child text that scored
    highest for this parent, since that's often what should be highlighted
    in a citation even though the parent is what gets fed to the LLM.

    An empty list is a valid result: every candidate scored below
    RERANK_SCORE_THRESHOLD, which the caller (Day 5's agent) should treat as
    "no good match" — not force-fed as an answer.
    """
    rewritten = rewrite_query(query, chat_history)
    fused = hybrid_search(rewritten.standalone, bm25_query=rewritten.bm25)
    fused = _apply_filters(fused, filters)
    reranked = rerank(rewritten.standalone, fused, top_k=top_k)

    parents_store = _get_parents()
    seen_parent_ids: set[str] = set()
    results = []

    for c in reranked:
        parent_id = c["metadata"]["parent_id"]
        if not parent_id or parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        entry = parents_store.get(parent_id)
        if not entry:
            continue

        metadata = dict(entry["metadata"])
        metadata["breadcrumb"] = json.loads(metadata["breadcrumb"])
        metadata["image_paths"] = json.loads(metadata["image_paths"])

        results.append({
            "parent_id": parent_id,
            "content": entry["content"],
            "metadata": metadata,
            "rerank_score": c["rerank_score"],
            "matched_child": c["content"],
        })

    return results

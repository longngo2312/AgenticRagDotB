"""DAY 5 — LangGraph AgentState definition."""
from typing import TypedDict


class AgentState(TypedDict, total=False):
    """State threaded through every node in the graph.

    `messages` stays a plain list of {"role", "content"} dicts rather than using
    LangGraph's `add_messages` reducer: retrieval/rewrite.py's condense_query()
    already expects exactly that shape (see its docstring), and add_messages
    would coerce entries into LangChain Message objects that would need
    converting back on every turn. One less lossy translation on the hot path.

    total=False so nodes can return partial updates (LangGraph merges them) and
    a caller can invoke the graph with just {"messages": [...], "question": ...}.
    """
    messages:       list    # prior conversation, oldest first (excludes this turn)
    question:       str     # this turn's raw user input, verbatim
    query:          str     # standalone (history-condensed) query used for retrieval
    docs:           list    # retrieved + reranked parent chunks
    context:        str     # docs formatted as numbered "[Nguồn n]" blocks
    attempts:       int     # retrieve→grade loop counter
    confidence:     float   # faithfulness self-check score, 0.0-1.0
    route:          str     # 'retrieve' | 'clarify' | 'handoff' | 'generate' | 'end'
    answer:         str     # what the user actually sees, whichever path produced it
    handoff_reason: str     # why the agent gave up, logged for support triage

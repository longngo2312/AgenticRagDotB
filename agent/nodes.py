"""
DAY 5 — LangGraph node functions (each reads AgentState, returns a partial update).

Every LLM call here is on the interactive path, so each one degrades instead of
raising: a 429 or a network blip must not take down the turn. The fallback per
node is chosen so the failure mode is the *safe* one (see each except block) —
same principle retrieval/rewrite.py already applies to query condensing.

Guardrails is deliberately rule-based. An LLM guardrail would add a full round
trip to every single turn — quota we don't have on the free tier (15 req/min)
and latency the <4s target can't absorb — to check things a regex checks better.
"""
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GOOGLE_API_KEY, LC_LLM_MODEL, REWRITE_HISTORY_TURNS,
    AGENT_MAX_ATTEMPTS, AGENT_MAX_INPUT_CHARS,
)
from agent.prompts import (
    SYSTEM_PROMPT, ROUTER_PROMPT, GRADING_PROMPT, REFORMULATE_PROMPT,
    FAITHFULNESS_PROMPT, CLARIFY_PROMPT, HANDOFF_MESSAGE, BLOCKED_MESSAGE,
)
from agent.tools import search_docs, create_handoff

# Sentinel: the faithfulness self-check itself failed (rate limit / network), as
# distinct from "the answer scored badly". A broken checker is not evidence of a
# bad answer, so this must not be treated as a low score — see graph.py's
# after_faithfulness edge.
UNVERIFIED_CONFIDENCE = -1.0

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=LC_LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,  # deterministic routing/grading; eval needs repeatability
        )
    return _llm


def _text(response) -> str:
    """Extract plain text from an AIMessage.

    langchain-core 1.5 returns `.content` as a list of content-block dicts, not a
    str, so `.content.strip()` raises AttributeError. `.text` is the accessor for
    the concatenated text; str() on it works whether this version exposes it as a
    property or (deprecated) a method."""
    return str(response.text).strip()


def _format_history(messages: list[dict]) -> str:
    turns = (messages or [])[-(REWRITE_HISTORY_TURNS * 2):]
    if not turns:
        return "(chưa có hội thoại trước đó)"
    return "\n".join(
        f"{'Người dùng' if t['role'] == 'user' else 'Trợ lý'}: {t['content']}"
        for t in turns
    )


# ── Structured-output schemas ──────────────────────────────────────────────────
# These are what .with_structured_output() enforces, which is why no prompt here
# asks the model for JSON and nothing parses fenced code blocks by hand.

class RouteDecision(BaseModel):
    route: str = Field(description="Chính xác một trong: retrieve, clarify, handoff")
    standalone_question: str = Field(description="Câu hỏi đã viết lại thành câu độc lập")


class GradeDecision(BaseModel):
    sufficient: bool = Field(description="Ngữ cảnh có đủ để trả lời câu hỏi không")
    reason: str = Field(description="Một câu ngắn giải thích")
    improved_query: str = Field(
        default="", description="Truy vấn viết lại để tra lại; rỗng nếu sufficient=true"
    )


class FaithfulnessScore(BaseModel):
    score: float = Field(description="0.0 đến 1.0, mức độ câu trả lời bám sát ngữ cảnh")
    unsupported: str = Field(default="", description="Nội dung không được ngữ cảnh hỗ trợ")


# ── Guardrails ─────────────────────────────────────────────────────────────────
# Instruction-override attempts, in both languages the docs and users mix.
_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?(previous|above)\s+instructions"
    r"|disregard\s+(all\s+)?(previous|prior)"
    r"|bỏ\s+qua\s+(mọi\s+|tất\s+cả\s+)?(hướng\s+dẫn|chỉ\s+dẫn|yêu\s+cầu)\s+(trước|trên)"
    r"|system\s+prompt"
    r"|you\s+are\s+now\s+a\b"
    r"|bạn\s+bây\s+giờ\s+là\b)",
    re.IGNORECASE,
)


def guardrails_node(state: dict) -> dict:
    """Cheap rule-based input validation — no LLM call. Blocks empty input,
    oversized input, and instruction-override attempts before they reach any
    model or burn any quota."""
    question = (state.get("question") or "").strip()

    if not question:
        return {"route": "blocked", "answer": BLOCKED_MESSAGE,
                "handoff_reason": "empty input"}

    if len(question) > AGENT_MAX_INPUT_CHARS:
        return {"route": "blocked", "answer": BLOCKED_MESSAGE,
                "handoff_reason": f"input too long ({len(question)} chars)"}

    if _INJECTION_RE.search(question):
        return {"route": "blocked", "answer": BLOCKED_MESSAGE,
                "handoff_reason": "prompt-injection pattern"}

    # Seed the loop counter here so retrieve_node never has to guess.
    return {"route": "router", "attempts": 0}


# ── Router ─────────────────────────────────────────────────────────────────────
def router_node(state: dict) -> dict:
    """One LLM call that both picks the route and condenses the follow-up into a
    standalone query — see ROUTER_PROMPT on why they're merged."""
    question = state["question"]
    prompt = ROUTER_PROMPT.format(
        history=_format_history(state.get("messages", [])),
        question=question,
    )
    try:
        decision = _get_llm().with_structured_output(RouteDecision).invoke(prompt)
        route = decision.route.strip().lower()
        if route not in ("retrieve", "clarify", "handoff"):
            route = "retrieve"
        query = decision.standalone_question.strip() or question
    except Exception as e:
        # Safe default: attempt retrieval with the raw question. If nothing
        # relevant exists, the reranker's score threshold makes retrieval return
        # empty and grade routes to handoff anyway — no fabricated answer.
        print(f"  [router] fell back to retrieve ({type(e).__name__})")
        route, query = "retrieve", question

    update = {"route": route, "query": query}
    if route == "handoff":
        # Record it here or handoff_node has nothing to log but "unspecified" —
        # out-of-scope questions are the signal support most wants to see.
        update["handoff_reason"] = "router: out of documentation scope"
    return update


# ── Retrieve ───────────────────────────────────────────────────────────────────
def retrieve_node(state: dict) -> dict:
    """Invoke the search_docs tool, taking the numbered context from the tool's
    content and the structured parent docs from its artifact — one retrieval,
    both representations.

    Reads state['query'], which grade_node overwrites with `improved_query` on a
    retry, so looping back here naturally re-searches with better wording rather
    than re-running the identical search.
    """
    query = state.get("query") or state["question"]
    attempts = state.get("attempts", 0) + 1

    try:
        message = search_docs.invoke({
            "name": "search_docs",
            "args": {"query": query},
            "id": f"search_{attempts}",
            "type": "tool_call",
        })
        docs = message.artifact or []
        context = message.content
    except Exception as e:
        print(f"  [retrieve] failed ({type(e).__name__}: {e})")
        docs, context = [], "(không tìm thấy tài liệu liên quan)"

    return {"docs": docs, "context": context, "attempts": attempts}


# ── Grade ──────────────────────────────────────────────────────────────────────
def _reformulate(query: str) -> str:
    """Reword a query that matched nothing, so the retry is a real second look
    rather than a byte-identical repeat of a search we know returns nothing."""
    try:
        return _text(_get_llm().invoke(REFORMULATE_PROMPT.format(query=query)))
    except Exception as e:
        print(f"  [grade] reformulation failed ({type(e).__name__})")
        return ""


def grade_node(state: dict) -> dict:
    """Judge whether the retrieved context can actually answer the question.
    On 'no', propose a reworded query so the retry is a genuinely different
    search — then graph.py decides loop-vs-handoff on the attempt count."""
    docs = state.get("docs") or []
    query = state.get("query", state["question"])

    # Retrieval abstained outright — everything scored below the rerank
    # threshold. Grading absent context is pointless (it's definitionally
    # insufficient), but a reworded query may well clear that threshold, so
    # spend one call on a reformulation *while attempts remain*. On the last
    # attempt, skip even that: nothing downstream would use the result.
    if not docs:
        update = {"route": "insufficient", "confidence": 0.0,
                  "handoff_reason": "no documents matched the query"}
        if state.get("attempts", 0) < AGENT_MAX_ATTEMPTS:
            improved = _reformulate(query)
            if improved and improved != query:
                update["query"] = improved
        return update

    # Grade against the user's ORIGINAL wording as well as the search query: the
    # query may have been narrowed or re-worded (by router condensing or a prior
    # retry), and judging only against that lets a drifted query reject context
    # that plainly answers what the user actually asked.
    prompt = GRADING_PROMPT.format(
        question=state["question"], query=query, context=state["context"],
    )
    try:
        decision = _get_llm().with_structured_output(GradeDecision).invoke(prompt)
    except Exception as e:
        # We do have documents; failing open (answer them) beats failing closed
        # (hand off a question we probably could have answered). SYSTEM_PROMPT
        # still forbids going beyond the context, and faithfulness_node still
        # checks the result.
        print(f"  [grade] assuming sufficient ({type(e).__name__})")
        return {"route": "generate"}

    if decision.sufficient:
        return {"route": "generate"}

    update = {"route": "insufficient", "handoff_reason": f"context insufficient: {decision.reason}"}
    if decision.improved_query.strip():
        update["query"] = decision.improved_query.strip()
    return update


# ── Generate ───────────────────────────────────────────────────────────────────
def generate_node(state: dict) -> dict:
    """Answer strictly from the retrieved context, with [n] citations."""
    parts = []
    history = _format_history(state.get("messages", []))
    if state.get("messages"):
        parts.append(f"Hội thoại trước đó:\n{history}")
    parts.append(f"Ngữ cảnh:\n{state['context']}")
    parts.append(f"Câu hỏi: {state['question']}")

    try:
        response = _get_llm().invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(parts)),
        ])
        answer = _text(response)
    except Exception as e:
        print(f"  [generate] failed ({type(e).__name__})")
        return {"route": "handoff", "handoff_reason": f"generation failed: {type(e).__name__}"}

    if not answer:
        return {"route": "handoff", "handoff_reason": "empty generation"}
    return {"answer": answer, "route": "faithfulness"}


# ── Faithfulness self-check ────────────────────────────────────────────────────
def faithfulness_node(state: dict) -> dict:
    """Score the generated answer against the context it was given. Below
    AGENT_FAITHFULNESS_MIN, graph.py hands off rather than shipping it."""
    prompt = FAITHFULNESS_PROMPT.format(context=state["context"], answer=state["answer"])
    try:
        result = _get_llm().with_structured_output(FaithfulnessScore).invoke(prompt)
        score = max(0.0, min(1.0, float(result.score)))
        if score < 1.0 and result.unsupported.strip():
            print(f"  [faithfulness] {score:.2f} — unsupported: {result.unsupported[:120]}")
        return {"confidence": score}
    except Exception as e:
        print(f"  [faithfulness] unverified ({type(e).__name__})")
        return {"confidence": UNVERIFIED_CONFIDENCE,
                "handoff_reason": f"faithfulness unverified: {type(e).__name__}"}


# ── Clarify ────────────────────────────────────────────────────────────────────
def clarify_node(state: dict) -> dict:
    """Ask one targeted follow-up instead of guessing at a vague question.
    This is the capability the Day 3/4 baseline lacked — it could only answer or
    abstain, which is exactly why the eval's ambiguous items capped at 0.5."""
    try:
        response = _get_llm().invoke(CLARIFY_PROMPT.format(question=state["question"]))
        question = _text(response)
    except Exception as e:
        print(f"  [clarify] failed ({type(e).__name__})")
        question = ""

    return {
        "answer": question or "Bạn có thể nói rõ hơn một chút về vấn đề bạn đang gặp không?",
        "route": "end",
    }


# ── Handoff ────────────────────────────────────────────────────────────────────
def handoff_node(state: dict) -> dict:
    """Log the handoff (for support triage) and return the polite message."""
    reason = state.get("handoff_reason") or "unspecified"
    attempts = state.get("attempts", 0)
    if attempts >= AGENT_MAX_ATTEMPTS:
        reason = f"{reason} (after {attempts} retrieval attempts)"

    try:
        message = create_handoff.invoke({"reason": f"[{state.get('question','')}] {reason}"})
    except Exception as e:
        print(f"  [handoff] logging failed ({type(e).__name__})")
        message = HANDOFF_MESSAGE

    return {"answer": message, "route": "end", "handoff_reason": reason}

"""
DAY 5 — Interactive REPL for the LangGraph agent, with its reasoning shown.

Usage:
  python scripts/agent_cli.py           # show the agent's thinking trace (default)
  python scripts/agent_cli.py --quiet   # answers only, no trace

Commands:  /reset   /sources   /trace   /exit  (also /quit, Ctrl-C, Ctrl-D)
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")  # Vietnamese input on Windows consoles
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AGENT_MAX_ATTEMPTS, AGENT_FAITHFULNESS_MIN
from agent.graph import app
from agent.nodes import UNVERIFIED_CONFIDENCE
from retrieval import reranker

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
CYAN, GREEN, YELLOW, RED, MAGENTA = "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[35m"

# One line per node, phrased as what the agent just decided rather than as a
# function name — the point is to read the reasoning, not the call stack.
NODE_LABELS = {
    "guardrails":   ("guardrails",   CYAN),
    "router":       ("router",       CYAN),
    "retrieve":     ("retrieve",     YELLOW),
    "grade":        ("grade",        YELLOW),
    "generate":     ("generate",     GREEN),
    "faithfulness": ("faithfulness", MAGENTA),
    "clarify":      ("clarify",      CYAN),
    "handoff":      ("handoff",      RED),
}


def _describe(node: str, update: dict) -> str:
    """Turn a node's real state update into one human-readable line."""
    if node == "guardrails":
        if update.get("route") == "blocked":
            return f"BLOCKED — {update.get('handoff_reason')}"
        return "input ok"

    if node == "router":
        return f"route={update.get('route')}  ·  standalone: \"{update.get('query', '')[:60]}\""

    if node == "retrieve":
        docs = update.get("docs") or []
        attempt = update.get("attempts", 0)
        titles = ", ".join(d["metadata"].get("doc_title", "?")[:28] for d in docs[:3])
        if not docs:
            return f"attempt {attempt}/{AGENT_MAX_ATTEMPTS} — no docs cleared the rerank threshold"
        return f"attempt {attempt}/{AGENT_MAX_ATTEMPTS} — {len(docs)} parent doc(s): {titles}"

    if node == "grade":
        if update.get("route") == "generate":
            return "context sufficient → generate"
        reason = update.get("handoff_reason", "insufficient")
        requeried = update.get("query")
        line = f"INSUFFICIENT — {reason}"
        if requeried:
            line += f"\n           retrying with: \"{requeried[:60]}\""
        return line

    if node == "generate":
        if update.get("route") == "handoff":
            return f"failed — {update.get('handoff_reason')}"
        return f"drafted {len(update.get('answer', ''))} chars"

    if node == "faithfulness":
        confidence = update.get("confidence")
        if confidence == UNVERIFIED_CONFIDENCE:
            return "could not verify (rate limit / network) — shipping answer anyway"
        verdict = "PASS" if confidence >= AGENT_FAITHFULNESS_MIN else "FAIL → handoff"
        return f"score={confidence:.2f} (min {AGENT_FAITHFULNESS_MIN}) → {verdict}"

    if node == "clarify":
        return "question too vague → asking a follow-up instead of guessing"

    if node == "handoff":
        return f"reason: {update.get('handoff_reason')}"

    return ""


def _print_sources(docs: list[dict]) -> None:
    if not docs:
        return
    print(f"\n{DIM}Nguồn tham khảo:{RESET}")
    for i, d in enumerate(docs, start=1):
        meta = d["metadata"]
        print(f"  {DIM}[{i}] {meta.get('doc_title','')} — {meta.get('doc_url','')}{RESET}")


def run_turn(question: str, history: list[dict], show_trace: bool) -> dict:
    """Stream the graph so each node's decision prints as it happens, then return
    the accumulated final state."""
    state: dict = {}

    if show_trace:
        print(f"\n{DIM}┌─ agent{RESET}")

    for chunk in app.stream({"question": question, "messages": history},
                            stream_mode="updates"):
        for node, update in chunk.items():
            state.update(update or {})
            if not show_trace:
                continue
            label, color = NODE_LABELS.get(node, (node, DIM))
            detail = _describe(node, update or {})
            print(f"{DIM}│{RESET} {color}{label:<13}{RESET}{DIM}{detail}{RESET}")

    if show_trace:
        print(f"{DIM}└{'─' * 58}{RESET}")

    return state


def main() -> None:
    show_trace = "--quiet" not in sys.argv

    print("Đang khởi động agent DotB EMS...")
    reranker.warmup()  # load the cross-encoder now, not on the first question
    print(f"Sẵn sàng (trace={'on' if show_trace else 'off'}). "
          f"Gõ câu hỏi, /exit để thoát. (/reset, /sources, /trace)\n")

    history: list[dict] = []
    last_docs: list[dict] = []

    while True:
        try:
            question = input(f"{BOLD}Bạn:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not question:
            continue
        if question in ("/exit", "/quit"):
            print("Tạm biệt!")
            break
        if question == "/reset":
            history.clear()
            last_docs = []
            print(f"{DIM}(đã xóa lịch sử hội thoại){RESET}\n")
            continue
        if question == "/sources":
            if last_docs:
                _print_sources(last_docs)
                print()
            else:
                print(f"{DIM}(chưa có nguồn nào){RESET}\n")
            continue
        if question == "/trace":
            show_trace = not show_trace
            print(f"{DIM}(trace = {show_trace}){RESET}\n")
            continue

        try:
            state = run_turn(question, history, show_trace)
        except Exception as e:
            print(f"{RED}[lỗi agent] {type(e).__name__}: {e}{RESET}\n")
            continue

        answer = state.get("answer", "(không có câu trả lời)")
        print(f"\n{BOLD}Trợ lý:{RESET} {answer}")

        docs = state.get("docs") or []
        # Only cite sources on a path that actually answered from them — a
        # handoff or clarify reply isn't grounded in the docs it happened to see.
        if docs and state.get("confidence", 0) >= AGENT_FAITHFULNESS_MIN:
            _print_sources(docs)
            last_docs = docs
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

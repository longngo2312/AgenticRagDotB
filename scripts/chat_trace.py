"""
Instrumented chat REPL — see what every stage of retrieval + generation does,
and how long it takes, on each turn.

Usage:
  python scripts/chat_trace.py

Same conversation loop as chat_cli.py, but instead of hiding the pipeline it
decomposes retrieve() into its real sub-functions and times each one:

  rewrite (condense) → rewrite (glossary) → dense → bm25 → RRF fusion →
  rerank → parent-fetch → LLM generation (with time-to-first-token)

Commands:  /reset   /sources   /exit  (also /quit, Ctrl-C, Ctrl-D)
"""
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")  # Vietnamese input, incl. piped test input on Windows
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root (config, retrieval)
sys.path.insert(0, str(Path(__file__).parent))         # scripts/ (chat_cli)

from google import genai
from google.genai import types

from config import (
    GOOGLE_API_KEY, LLM_MODEL,
    DENSE_TOP_K, BM25_TOP_K, RRF_K, RERANK_TOP_K, RERANK_SCORE_THRESHOLD,
)
from retrieval.rewrite import condense_query, expand_glossary
from retrieval.dense import dense_search
from retrieval.bm25_index import bm25_search
from retrieval.hybrid import rrf_fuse
from retrieval.reranker import rerank, warmup
from retrieval.retriever import _apply_filters, fetch_parents

# Reuse the exact persona + context-building + source display from the real CLI.
import chat_cli

_client = genai.Client(api_key=GOOGLE_API_KEY)

LOG_PATH = Path(__file__).parent.parent / "logs" / "chat_trace.jsonl"

# ANSI
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
YELLOW, RED, GREEN, CYAN = "\033[33m", "\033[31m", "\033[32m", "\033[36m"


@contextmanager
def stage(trace: list, label: str, fn: str):
    """Time a pipeline stage; append its record to `trace`. Set rec['out'] in
    the body to describe what came back."""
    rec = {"stage": label, "fn": fn, "out": ""}
    trace.append(rec)
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)


def _color_for(ms: float) -> str:
    if ms >= 500:
        return RED
    if ms >= 100:
        return YELLOW
    return GREEN


def _print_trace(question: str, trace: list, total_ms: float) -> None:
    print(f"\n{DIM}┌─ trace {RESET}{CYAN}\"{question[:56]}\"{RESET}")
    for rec in trace:
        c = _color_for(rec["ms"])
        line = f"{DIM}│{RESET} {rec['stage']:<9}{DIM}{rec['fn']:<26}{RESET}"
        line += f"{c}{rec['ms']:>8.1f}ms{RESET}"
        if rec["out"]:
            line += f"  {DIM}→ {rec['out']}{RESET}"
        print(line)
    print(f"{DIM}│{RESET} {BOLD}{'TOTAL':<9}{'':<26}{total_ms:>8.1f}ms{RESET}")
    print(f"{DIM}└{'─' * 52}{RESET}")


def _log_jsonl(question: str, standalone: str, trace: list, total_ms: float) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "standalone": standalone,
        "total_ms": round(total_ms, 1),
        "stages": {r["stage"] + ("_" + r["fn"].split("(")[0] if r["stage"] in ("rewrite",) else ""): r["ms"]
                   for r in trace},
        "stage_detail": trace,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def traced_retrieve(question: str, history: list, trace: list) -> tuple[list, str]:
    """Mirror of retriever.retrieve(), decomposed so each real sub-function is
    timed individually. Returns (parent_docs, standalone_query)."""
    with stage(trace, "rewrite", "condense_query()") as rec:
        standalone = condense_query(question, history)
        rec["out"] = (f'"{standalone[:50]}"' if standalone != question
                      else "(no history — passthrough)")

    with stage(trace, "rewrite", "expand_glossary()") as rec:
        bm25_query = expand_glossary(standalone)
        extra = len(bm25_query.split()) - len(standalone.split())
        rec["out"] = f"+{extra} BM25 term(s)" if extra else "no glossary hit"

    with stage(trace, "dense", f"dense_search(k={DENSE_TOP_K})") as rec:
        dense_results = dense_search(standalone, top_k=DENSE_TOP_K)
        rec["out"] = f"{len(dense_results)} hits (embed+HNSW cosine)"

    with stage(trace, "bm25", f"bm25_search(k={BM25_TOP_K})") as rec:
        bm25_results = bm25_search(bm25_query, top_k=BM25_TOP_K)
        rec["out"] = f"{len(bm25_results)} hits (lexical)"

    with stage(trace, "fusion", f"rrf_fuse(k={RRF_K})") as rec:
        fused = rrf_fuse(dense_results, bm25_results)
        rec["out"] = f"{len(fused)} unique candidates"

    with stage(trace, "rerank", f"cross-encoder ×{len(fused)}") as rec:
        reranked = rerank(standalone, _apply_filters(fused, None), top_k=RERANK_TOP_K)
        top = reranked[0]["rerank_score"] if reranked else 0.0
        rec["out"] = f"{len(reranked)} kept (≥{RERANK_SCORE_THRESHOLD}, top={top:.3f})"

    with stage(trace, "parents", "fetch_parents()") as rec:
        docs = fetch_parents(reranked)
        rec["out"] = f"{len(docs)} parent doc(s), deduped"

    return docs, standalone


def _stream_traced(contents: str, trace: list) -> str:
    """Stream the answer while capturing time-to-first-token and total gen time."""
    full = []
    with stage(trace, "generate", "gemini stream") as rec:
        t0 = time.perf_counter()
        ttft_ms = None
        stream = _client.models.generate_content_stream(
            model=LLM_MODEL,
            contents=[contents],
            config=types.GenerateContentConfig(system_instruction=chat_cli.SYSTEM_PROMPT),
        )
        for chunk in stream:
            if chunk.text:
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - t0) * 1000, 1)
                    rec["ttft_ms"] = ttft_ms
                print(chunk.text, end="", flush=True)
                full.append(chunk.text)
        print()
        answer = "".join(full)
        rec["out"] = f"{len(answer)} chars (ttft {ttft_ms}ms)" if ttft_ms else "empty"
    return answer


def main() -> None:
    print("Đang khởi động (chế độ trace)...")
    warmup()
    print(f"Sẵn sàng. Trace ghi vào {LOG_PATH}")
    print("Gõ câu hỏi, /exit để thoát. (/reset, /sources)\n")

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
            history.clear(); last_docs = []
            print(f"{DIM}(đã xóa lịch sử){RESET}\n")
            continue
        if question == "/sources":
            chat_cli._print_sources(last_docs) if last_docs else print(f"{DIM}(chưa có nguồn){RESET}")
            print()
            continue

        trace: list = []
        turn_t0 = time.perf_counter()

        try:
            docs, standalone = traced_retrieve(question, history, trace)
        except Exception as e:
            print(f"{RED}[lỗi truy xuất] {type(e).__name__}: {e}{RESET}\n")
            continue

        if not docs:
            total_ms = (time.perf_counter() - turn_t0) * 1000
            _print_trace(question, trace, total_ms)
            print(f"\n{BOLD}Trợ lý:{RESET} {chat_cli.ABSTAIN_MESSAGE}\n")
            _log_jsonl(question, standalone, trace, total_ms)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": chat_cli.ABSTAIN_MESSAGE})
            last_docs = []
            continue

        print(f"\n{BOLD}Trợ lý:{RESET} ", end="", flush=True)
        contents = chat_cli._build_contents(question, docs, history)
        try:
            answer = _stream_traced(contents, trace)
        except Exception as e:
            answer = None
            print(f"\n{RED}[lỗi sinh câu trả lời] {type(e).__name__}: {str(e)[:140]}{RESET}")

        # Print the trace regardless — the retrieval stages still ran and their
        # latencies are exactly what we're here to see, even if generation failed.
        total_ms = (time.perf_counter() - turn_t0) * 1000
        _print_trace(question, trace, total_ms)
        chat_cli._print_sources(docs)
        print()
        _log_jsonl(question, standalone, trace, total_ms)

        if answer:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            last_docs = docs


if __name__ == "__main__":
    main()

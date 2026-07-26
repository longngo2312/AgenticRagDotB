"""
DAY 3 — Terminal Q&A REPL for testing the RAG backbone without the web UI.

Usage:
  python scripts/chat_cli.py            # normal chat
  python scripts/chat_cli.py --debug    # also print retrieved sources + scores

Flow per turn:  retrieve() -> ground the LLM on the retrieved parent docs ->
stream a cited answer. This is the pre-agent baseline: a direct RAG loop, no
LangGraph (that's Day 5). Retrieval already condenses follow-ups against the
running history, so multi-turn works here too.

Commands (typed at the prompt):
  /reset     clear the conversation history
  /sources   show sources for the last answer again
  /debug     toggle retrieval debug output
  /exit      quit  (also: /quit, Ctrl-C, Ctrl-D)
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")  # Vietnamese input on Windows consoles
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, LLM_MODEL, REWRITE_HISTORY_TURNS
from retrieval.retriever import retrieve
from retrieval import reranker

_client = genai.Client(api_key=GOOGLE_API_KEY)

# Vietnamese persona + grounding rules. The retriever can return nothing
# (abstain), and when it does we never even reach the LLM — but these rules
# still bind the model to the supplied context and to citing its sources.
SYSTEM_PROMPT = """Bạn là trợ lý hỗ trợ khách hàng của DotB EMS — phần mềm quản lý đào tạo.
Nhiệm vụ: trả lời câu hỏi của người dùng DỰA HOÀN TOÀN vào phần "Ngữ cảnh" được cung cấp bên dưới.

Quy tắc:
- Chỉ dùng thông tin trong Ngữ cảnh. TUYỆT ĐỐI không bịa thông tin không có trong đó.
- Sau mỗi ý lấy từ một nguồn, trích dẫn số nguồn tương ứng, ví dụ [1], [2].
- Nếu Ngữ cảnh có các bước (Bước 1, Bước 2...), trình bày lại rõ ràng theo thứ tự.
- Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm.
- Nếu Ngữ cảnh không đủ để trả lời, hãy nói thẳng là bạn chưa có thông tin và đề nghị chuyển cho nhân viên hỗ trợ."""

# Shown when retrieval abstains — we don't invent an answer from nothing.
ABSTAIN_MESSAGE = (
    "Xin lỗi, mình chưa tìm thấy thông tin phù hợp trong tài liệu hỗ trợ để trả lời câu hỏi này. "
    "Bạn có thể diễn đạt lại câu hỏi, hoặc mình sẽ chuyển bạn tới nhân viên hỗ trợ nhé."
)


def _format_context(docs: list[dict]) -> str:
    """Number each retrieved parent so the model can cite [n], and label it
    with its human breadcrumb so citations are meaningful."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        title = d["metadata"].get("doc_title", "")
        crumb = " > ".join(d["metadata"].get("breadcrumb", [])) or title
        blocks.append(f"[Nguồn {i}] {crumb}\n{d['content']}")
    return "\n\n---\n\n".join(blocks)


def _build_contents(question: str, docs: list[dict], history: list[dict]) -> str:
    """Recent history (for natural pronoun/topic continuity in the answer) +
    the grounding context + the question, as one user message. Retrieval has
    already resolved the follow-up for *search*; this gives *generation* the
    same continuity."""
    parts = []
    recent = history[-(REWRITE_HISTORY_TURNS * 2):]
    if recent:
        convo = "\n".join(
            f"{'Người dùng' if t['role'] == 'user' else 'Trợ lý'}: {t['content']}"
            for t in recent
        )
        parts.append(f"Hội thoại trước đó:\n{convo}")
    parts.append(f"Ngữ cảnh:\n{_format_context(docs)}")
    parts.append(f"Câu hỏi: {question}")
    return "\n\n".join(parts)


def _stream_answer(contents: str) -> str:
    """Stream the grounded answer to the terminal, returning the full text so
    it can be appended to history."""
    full = []
    stream = _client.models.generate_content_stream(
        model=LLM_MODEL,
        contents=[contents],
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    for chunk in stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full.append(chunk.text)
    print()
    return "".join(full)


def _print_sources(docs: list[dict]) -> None:
    if not docs:
        return
    print("\n\033[2mNguồn tham khảo:\033[0m")
    for i, d in enumerate(docs, start=1):
        title = d["metadata"].get("doc_title", "")
        url = d["metadata"].get("doc_url", "")
        print(f"  \033[2m[{i}] {title} — {url}\033[0m")


def _print_debug(docs: list[dict]) -> None:
    print(f"\n\033[2m[debug] {len(docs)} parent doc(s) after rerank:\033[0m")
    for i, d in enumerate(docs, start=1):
        print(f"  \033[2m[{i}] score={d['rerank_score']:.3f}  {d['metadata'].get('doc_title','')}\033[0m")


def main() -> None:
    debug = "--debug" in sys.argv

    print("Đang khởi động trợ lý hỗ trợ DotB EMS...")
    reranker.warmup()  # load the cross-encoder now, not on the first question
    print("Sẵn sàng. Gõ câu hỏi, hoặc /exit để thoát. (/reset, /sources, /debug)\n")

    history: list[dict] = []
    last_docs: list[dict] = []

    while True:
        try:
            question = input("\033[1mBạn:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not question:
            continue

        # ── commands ────────────────────────────────────────────────────────
        if question in ("/exit", "/quit"):
            print("Tạm biệt!")
            break
        if question == "/reset":
            history.clear()
            last_docs = []
            print("\033[2m(đã xóa lịch sử hội thoại)\033[0m\n")
            continue
        if question == "/sources":
            if last_docs:
                _print_sources(last_docs)
                print()
            else:
                print("\033[2m(chưa có nguồn nào)\033[0m\n")
            continue
        if question == "/debug":
            debug = not debug
            print(f"\033[2m(debug = {debug})\033[0m\n")
            continue

        # ── retrieve ────────────────────────────────────────────────────────
        try:
            docs = retrieve(question, chat_history=history)
        except Exception as e:
            print(f"\033[31m[lỗi truy xuất] {type(e).__name__}: {e}\033[0m\n")
            continue

        if debug:
            _print_debug(docs)

        # ── abstain path ────────────────────────────────────────────────────
        if not docs:
            print(f"\n\033[1mTrợ lý:\033[0m {ABSTAIN_MESSAGE}\n")
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": ABSTAIN_MESSAGE})
            last_docs = []
            continue

        # ── generate ────────────────────────────────────────────────────────
        print("\n\033[1mTrợ lý:\033[0m ", end="", flush=True)
        contents = _build_contents(question, docs, history)
        try:
            answer = _stream_answer(contents)
        except Exception as e:
            print(f"\n\033[31m[lỗi sinh câu trả lời] {type(e).__name__}: {e}\033[0m\n")
            continue

        _print_sources(docs)
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        last_docs = docs


if __name__ == "__main__":
    main()

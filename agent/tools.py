"""
DAY 5 — LangChain tools available to the agent.

search_docs uses response_format="content_and_artifact" so one invocation yields
BOTH the numbered context string (what the LLM reads) and the structured parent
docs (what faithfulness checking and citation rendering need). The obvious
alternative — a tool returning only a string, with the node calling retrieve()
again for the docs — would run the whole hybrid+rerank pipeline twice per turn,
paying the cross-encoder cost for nothing.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GLOSSARY_PATH, HANDOFF_LOG_PATH
from agent.prompts import HANDOFF_MESSAGE
from retrieval.retriever import retrieve


def format_docs(docs: list[dict]) -> str:
    """Number each retrieved parent so the model can cite [n], labelled with its
    human breadcrumb so the citation means something. Same shape as the Day 3
    CLI's context blocks — the wording SYSTEM_PROMPT's citation rule expects."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d["metadata"]
        crumb = " > ".join(meta.get("breadcrumb", [])) or meta.get("doc_title", "")
        blocks.append(f"[Nguồn {i}] {crumb}\n{d['content']}")
    return "\n\n---\n\n".join(blocks)


@tool(response_format="content_and_artifact")
def search_docs(query: str, section: str | None = None) -> tuple[str, list[dict]]:
    """Tra tài liệu hướng dẫn DotB EMS để tìm thông tin trả lời câu hỏi.

    Args:
        query: câu truy vấn độc lập, đã đủ ngữ cảnh để tìm kiếm.
        section: (tùy chọn) giới hạn trong một phân hệ, ví dụ 'bo-phan-giao-vu'.
    """
    filters = {"section": section} if section else None
    docs = retrieve(query, filters=filters)
    if not docs:
        # Empty is a real answer, not an error: every candidate scored below
        # RERANK_SCORE_THRESHOLD, which the grade node should read as "no match".
        return "(không tìm thấy tài liệu liên quan)", []
    return format_docs(docs), docs


@tool
def lookup_glossary(term: str) -> str:
    """Tra thuật ngữ EMS: trả về các cách gọi tương đương (Việt ↔ Anh).

    Args:
        term: thuật ngữ cần tra, ví dụ 'bảo lưu' hoặc 'delay'.
    """
    if not GLOSSARY_PATH.exists():
        return "(chưa có dữ liệu thuật ngữ)"
    aliases = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8")).get("aliases", {})

    # Match in both directions — the caller may hand us either the canonical
    # English term or any Vietnamese alias of it.
    needle = term.strip().lower()
    for canonical, alts in aliases.items():
        if needle == canonical.lower() or needle in [a.lower() for a in alts]:
            return f"{canonical}: {', '.join(alts)}"
    return f"(không tìm thấy thuật ngữ '{term}')"


@tool
def create_handoff(reason: str) -> str:
    """Ghi nhận yêu cầu chuyển cho nhân viên hỗ trợ, trả về tin nhắn cho người dùng.

    Args:
        reason: lý do agent không tự trả lời được, dùng cho việc phân loại nội bộ.
    """
    HANDOFF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), "reason": reason}
    with open(HANDOFF_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return HANDOFF_MESSAGE


AGENT_TOOLS = [search_docs, lookup_glossary, create_handoff]

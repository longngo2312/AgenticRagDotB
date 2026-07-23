"""
DAY 2 — STEP 3: Structure-aware Markdown chunker (parent-child)

Strategy (from design doc §5):
  - Split by heading hierarchy (#, ##, ###, …)
  - Each child chunk carries FULL heading path as prefix:
      "[Tuyển sinh > Lead > Convert Lead] ## Chuyển đổi hàng loạt\n\n<body>"
  - Group heading sections into PARENT chunks (~1500 tok)
  - Split each parent into CHILD chunks (~500 tok) at paragraph boundaries
  - Index & search on children (precise match)
  - Retrieve PARENT for LLM (full context)

Token estimate: 1 token ≈ 4 chars (mixed Viet/English, no tiktoken needed)
"""
import hashlib
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHILD_CHUNK_MAX_TOKENS, PARENT_CHUNK_MAX_TOKENS
from ingestion.parser import ParsedDocument


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str | None
    chunk_type: Literal["child", "parent"]
    doc_url: str
    doc_title: str
    breadcrumb: list[str]
    section: str
    product: str
    heading_path: str
    content: str       # heading prefix + body — what gets embedded
    raw_text: str      # body only — stored for BM25 / display
    token_count: int
    content_hash: str


def _split_by_headings(markdown: str) -> list[tuple[int, str, str]]:
    """Return list of (level, heading_text, body_text). level=0 = preamble."""
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[int, str, str]] = []
    pos = 0
    for m in heading_re.finditer(markdown):
        body = markdown[pos:m.start()].strip()
        if pos == 0 and body:
            sections.append((0, "", body))
        pos = m.end()
        level = len(m.group(1))
        heading = m.group(2).strip()
        sections.append((level, heading, ""))  # body filled on next iteration
    # fill bodies
    matches = list(heading_re.finditer(markdown))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        # sections index: preamble may have been inserted at 0
        offset = 1 if sections and sections[0][0] == 0 else 0
        sections[i + offset] = (sections[i + offset][0], sections[i + offset][1], body)
    return sections


def _build_heading_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(h for _, h in stack if h)


def _split_into_children(text: str, max_tokens: int) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text)
    children: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        t = estimate_tokens(para)
        if current_tokens + t > max_tokens and current_parts:
            children.append("\n\n".join(current_parts))
            current_parts = [para]
            current_tokens = t
        else:
            current_parts.append(para)
            current_tokens += t

    if current_parts:
        children.append("\n\n".join(current_parts))

    return children or [text]


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    sections = _split_by_headings(doc.content)
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    chunks: list[Chunk] = []

    # group sections into parent groups
    parent_groups: list[list[tuple[int, str, str]]] = []
    current_group: list[tuple[int, str, str]] = []
    current_tokens = 0

    for level, heading, body in sections:
        section_text = (f"{'#' * level} {heading}\n\n{body}" if heading else body).strip()
        t = estimate_tokens(section_text)
        if current_tokens + t > PARENT_CHUNK_MAX_TOKENS and current_group:
            parent_groups.append(current_group)
            current_group = [(level, heading, body)]
            current_tokens = t
        else:
            current_group.append((level, heading, body))
            current_tokens += t

    if current_group:
        parent_groups.append(current_group)

    for group in parent_groups:
        # reconstruct parent text and heading path
        heading_stack = []
        group_parts: list[str] = []
        group_heading_path = ""

        for level, heading, body in group:
            if heading:
                # maintain stack — pop anything at same or deeper level
                heading_stack = [(l, h) for l, h in heading_stack if l < level]
                heading_stack.append((level, heading))
                prefix = "#" * level + " " + heading
                group_parts.append(prefix + ("\n\n" + body if body else ""))
            else:
                group_parts.append(body)
            group_heading_path = _build_heading_path(heading_stack)

        parent_raw = "\n\n".join(p for p in group_parts if p).strip()
        if not parent_raw:
            continue

        parent_id = str(uuid.uuid4())
        parent_content = f"[{doc.breadcrumb_str}] {group_heading_path}\n\n{parent_raw}".strip()
        parent_hash = hashlib.sha256(parent_content.encode()).hexdigest()

        parent_chunk = Chunk(
            chunk_id=parent_id,
            parent_id=None,
            chunk_type="parent",
            doc_url=doc.url,
            doc_title=doc.title,
            breadcrumb=doc.breadcrumb,
            section=doc.section,
            product=doc.product,
            heading_path=group_heading_path,
            content=parent_content,
            raw_text=parent_raw,
            token_count=estimate_tokens(parent_content),
            content_hash=parent_hash,
        )
        chunks.append(parent_chunk)

        # split parent into children
        child_texts = _split_into_children(parent_raw, CHILD_CHUNK_MAX_TOKENS)
        for child_raw in child_texts:
            prefix = f"[{doc.breadcrumb_str}] {group_heading_path}\n\n" if group_heading_path else f"[{doc.breadcrumb_str}]\n\n"
            child_content = (prefix + child_raw).strip()
            child_hash = hashlib.sha256(child_content.encode()).hexdigest()
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                parent_id=parent_id,
                chunk_type="child",
                doc_url=doc.url,
                doc_title=doc.title,
                breadcrumb=doc.breadcrumb,
                section=doc.section,
                product=doc.product,
                heading_path=group_heading_path,
                content=child_content,
                raw_text=child_raw,
                token_count=estimate_tokens(child_content),
                content_hash=child_hash,
            ))

    return chunks

"""
DAY 2 — STEP 3: Structure-aware Markdown chunker (parent-child)

- Split by heading hierarchy (#, ##, ###, …)
- Each child chunk carries FULL heading path as prefix:
    "[Tuyển sinh > Lead > Convert Lead] ## Chuyển đổi hàng loạt\n\n<body>"
- Group heading sections into PARENT chunks (~1500 tok)
- Split each parent into CHILD chunks (~500 tok) at paragraph boundaries
- Index & search on children (precise match)
- Retrieve PARENT for LLM (full context)

"""
import hashlib
import re
import sys
from dataclasses import dataclass, field
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
    image_paths: list[str] = field(default_factory=list)  # local paths of screenshots in this chunk


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
        sections.append((level, heading, ""))

    # Second pass to fill bodies: a heading's body runs until the *next*
    # heading, which the first pass hasn't seen yet when it appends the entry.
    matches = list(heading_re.finditer(markdown))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        # `sections` gains a leading preamble entry when the doc opens with
        # text before its first heading — shift past it to stay aligned.
        offset = 1 if sections and sections[0][0] == 0 else 0
        sections[i + offset] = (sections[i + offset][0], sections[i + offset][1], body)
    return sections


def _build_heading_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(h for _, h in stack if h)


_IMAGE_RE = re.compile(r"^\[IMAGE:", re.IGNORECASE)
_HINT_RE  = re.compile(r"^>\s*\*\*Lưu ý:\*\*")
_IMAGE_PATH_RE = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


def _extract_image_paths(content: str) -> list[str]:
    return _IMAGE_PATH_RE.findall(content)


def _is_bond_backward(para: str) -> bool:
    """
    A paragraph that must stay glued to the step/text that precedes it:
      - [IMAGE: ...] lines   (the screenshot illustrating the step above)
      - > **Lưu ý:** ...     (a hint/note qualifying the step above)
    Step markers (> Bước N / > Step N) are NOT bonded — they open a new block.
    """
    return bool(_IMAGE_RE.match(para) or _HINT_RE.match(para))


def _merge_atomic_blocks(paragraphs: list[str]) -> list[str]:
    """
    Group paragraphs into atomic blocks: an image or hint attaches backward to
    the preceding paragraph so `step text + image + hint` never gets split
    across two child chunks. A block may exceed the child token budget — atomicity
    wins over the size cap (the greedy splitter below emits it as a lone child).
    """
    blocks: list[list[str]] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if _is_bond_backward(para) and blocks:
            blocks[-1].append(para)
        else:
            blocks.append([para])
    return ["\n\n".join(b) for b in blocks]


def _split_into_children(text: str, max_tokens: int) -> list[str]:
    blocks = _merge_atomic_blocks(re.split(r"\n{2,}", text))
    children: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for block in blocks:
        t = estimate_tokens(block)
        if current_tokens + t > max_tokens and current_parts:
            children.append("\n\n".join(current_parts))
            current_parts = [block]
            current_tokens = t
        else:
            current_parts.append(block)
            current_tokens += t

    if current_parts:
        children.append("\n\n".join(current_parts))

    return children or [text]


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """Split one parsed document into parent and child chunks.

    Returns both kinds in one flat list (each child carries its `parent_id`):
    children are what gets embedded and searched, parents are what the LLM
    reads at answer time. Callers separate them by `chunk_type`.
    """
    sections = _split_by_headings(doc.content)
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    chunks: list[Chunk] = []

    # Pack consecutive sections up to the parent budget. Greedy rather than
    # balanced: keeping adjacent headings together matters more for context
    # than making the parents equal in size.
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
        heading_stack = []
        group_parts: list[str] = []
        group_heading_path = ""

        for level, heading, body in group:
            if heading:
                # Pop to the current level before pushing, so the stack always
                # holds this heading's true ancestors — an h2 following an h3
                # is a sibling of that h3's parent, not a child of the h3.
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

        parent_content = f"[{doc.breadcrumb_str}] {group_heading_path}\n\n{parent_raw}".strip()
        parent_hash = hashlib.sha256(parent_content.encode()).hexdigest()
        # Deterministic ID = content hash → re-runs are idempotent (upsert is a
        # true no-op for unchanged content) and embedding is resumable.
        parent_id = parent_hash

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
            image_paths=_extract_image_paths(parent_content),
        )
        chunks.append(parent_chunk)

        # Children are split from parent_raw (body only) but re-prefixed with
        # the breadcrumb + heading path below, so an embedded child still
        # carries the context its parent gave it.
        child_texts = _split_into_children(parent_raw, CHILD_CHUNK_MAX_TOKENS)
        for child_raw in child_texts:
            prefix = f"[{doc.breadcrumb_str}] {group_heading_path}\n\n" if group_heading_path else f"[{doc.breadcrumb_str}]\n\n"
            child_content = (prefix + child_raw).strip()
            child_hash = hashlib.sha256(child_content.encode()).hexdigest()
            chunks.append(Chunk(
                chunk_id=child_hash,
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
                image_paths=_extract_image_paths(child_content),
            ))

    return chunks

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
# TODO D2-11: import uuid, hashlib, re, dataclasses
#             import ParsedDocument from ingestion.parser
#             import CHILD_CHUNK_MAX_TOKENS, PARENT_CHUNK_MAX_TOKENS from config

# TODO D2-12: def estimate_tokens(text: str) -> int
#   - return max(1, len(text) // 4)

# TODO D2-13: define Chunk dataclass
#   fields: chunk_id (uuid4 str), parent_id (str | None),
#           chunk_type ('child' | 'parent'),
#           doc_url, doc_title, breadcrumb (list[str]),
#           section, product, heading_path (str),
#           content (heading prefix + body — what gets embedded),
#           raw_text (body only — stored for BM25 / display),
#           token_count (int), content_hash (sha256)

# TODO D2-14: def _split_by_headings(markdown: str) -> list[tuple[int, str, str]]
#   - regex scan for ^(#{1,6}) heading lines
#   - return list of (level, heading_text, body_text)
#   - level=0 means preamble before first heading

# TODO D2-15: def _build_heading_path(stack: list[tuple[int,str]]) -> str
#   - join non-empty headings in stack with ' > '

# TODO D2-16: def _split_into_children(text: str, max_tokens: int) -> list[str]
#   - split on double newlines (paragraph boundaries)
#   - greedily group paragraphs until max_tokens
#   - start new child when limit exceeded

# TODO D2-17: def chunk_document(doc: ParsedDocument) -> list[Chunk]
#   - call _split_by_headings(doc.content)
#   - maintain heading_stack to compute heading_path at each section
#   - group sections into PARENT groups (≤ PARENT_CHUNK_MAX_TOKENS)
#   - for each parent group:
#       * create parent Chunk (chunk_type='parent', parent_id=None)
#       * call _split_into_children on parent text
#       * create child Chunks (chunk_type='child', parent_id=parent.chunk_id)
#       * prefix each child content with "[breadcrumb_str] heading_path\n\n"
#   - return all chunks (parents + children interleaved)

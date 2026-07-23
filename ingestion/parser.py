"""
DAY 2 — STEP 2: Parse raw markdown into structured documents

Input:  CrawledPage (url, title, description, raw_content)
Output: ParsedDocument with metadata extracted from URL structure

Key insight from design doc:
  URL path encodes the business hierarchy, e.g.
  /tuyen-sinh-ban-hang/lead/convert-lead.md
    → breadcrumb = ['tuyen-sinh-ban-hang', 'lead', 'convert-lead']
    → section    = 'tuyen-sinh-ban-hang'
    → product    = 'EMS' (default) / 'SEA' / 'TEA' / 'Metrikal'
"""
# TODO D2-6: define ParsedDocument dataclass
#   fields: url, title, description, breadcrumb (list[str]),
#           breadcrumb_str ("Tuyển sinh > Lead > Convert Lead"),
#           section (str), product (str), content (str),
#           content_hash (sha256), last_seen (ISO 8601)

# TODO D2-7: def extract_breadcrumb(url: str) -> list[str]
#   - urlparse → split path on '/'
#   - strip empty strings and remove .md extension from last segment

# TODO D2-8: def infer_product(breadcrumb: list[str]) -> str
#   - look for 'sea' / 'tea' / 'metrikal' in any segment (lowercase)
#   - default → 'EMS'

# TODO D2-9: def clean_markdown(content: str) -> str
#   - strip HTML tags that may have leaked through
#   - collapse 3+ blank lines to 2
#   - strip leading/trailing whitespace

# TODO D2-10: def parse_document(url, title, description, raw_content) -> ParsedDocument
#   - call extract_breadcrumb, infer_product, clean_markdown
#   - build breadcrumb_str: join with ' > ', title-case each segment, replace '-' with ' '
#   - compute content_hash = sha256(cleaned content)
#   - set last_seen = datetime.now(utc).isoformat()

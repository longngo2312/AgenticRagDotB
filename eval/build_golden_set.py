"""
DAY 4 — Draft eval/golden_set.json by sampling real parent docs and asking
Gemini to write a plausible support question + grounded answer for each.

Usage:
  python eval/build_golden_set.py

This produces a DRAFT. eval/evaluate.py should not be trusted against
golden_set.json until a human has reviewed and corrected every
ground_truth_answer — the LLM only sees one doc at a time and can still
misread it.

Sampling: stratified across every `section` present in data/parent_docs.json,
weighted toward the sections with the most real pages (bo-phan-giao-vu,
tuyen-sinh-ban-hang, admin-guide, quan-li-dang-ki-hoc-va-thu-tien), one pick
minimum everywhere else, so the golden set has real coverage breadth instead
of clustering wherever sampling happened to land.

Schema (one object per array entry):
  id                 unique string, "q001", "q002", ...
  question           Vietnamese question as a real user/staff member would ask it
  ground_truth_answer  correct answer in Vietnamese
  relevant_doc_ids   doc_url(s) that should be retrieved; [] for out-of-scope
  section            top-level section slug
  difficulty         easy | medium | hard
  type               single-step | multi-step | conditional | out-of-scope | ambiguous

The last 5 entries (out-of-scope + ambiguous) are hand-written, not
LLM-derived — there's no single source doc to draft them from, since the
whole point is testing what happens when no doc should match, or several
plausibly could.
"""
import json
import random
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai

from config import GOOGLE_API_KEY, LLM_MODEL, PARENTS_PATH

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RANDOM_SEED = 42

# Picks per section; every section not listed here gets DEFAULT_QUOTA.
SECTION_QUOTAS = {
    "bo-phan-giao-vu": 5,
    "tuyen-sinh-ban-hang": 5,
    "admin-guide": 4,
    "quan-li-dang-ki-hoc-va-thu-tien": 3,
    "mobile": 2,
    "nhom-tinh-nang-tiep-thi": 2,
}
DEFAULT_QUOTA = 1

_client = genai.Client(api_key=GOOGLE_API_KEY)

_DRAFT_PROMPT = """Bạn là một chuyên viên hỗ trợ khách hàng của DotB EMS (phần mềm quản lý trung tâm giáo dục). Dựa CHỈ vào nội dung tài liệu dưới đây, hãy soạn:

1. Một câu hỏi thực tế mà nhân viên trung tâm có thể hỏi bộ phận hỗ trợ, bằng tiếng Việt tự nhiên (không trích dẫn nguyên văn tiêu đề tài liệu).
2. Câu trả lời đúng, đầy đủ, CHỈ dựa trên thông tin có trong tài liệu bên dưới (không suy diễn, không thêm thông tin ngoài tài liệu).

Tài liệu (breadcrumb: {breadcrumb}):
---
{content}
---

Trả lời CHÍNH XÁC theo định dạng JSON sau, không thêm giải thích, không thêm markdown code fence:
{{"question": "...", "answer": "..."}}
"""

# Hand-written: things a support inbox realistically receives that no page on
# help.dotb.vn answers (out-of-scope), or that are too vague to point at one
# doc (ambiguous). relevant_doc_ids is [] for out-of-scope; for ambiguous it
# lists every doc that could plausibly be the right one, since the correct
# retrieval behavior is "surface several candidates / ask a clarifying
# question," not "find the one true answer."
HAND_WRITTEN = [
    {
        "question": "Anh chị có thể giảm giá gói phần mềm nếu trung tâm em thanh toán trọn năm không?",
        "ground_truth_answer": "Đây là câu hỏi về giá/hợp đồng, không phải hướng dẫn sử dụng phần mềm — tài liệu help.dotb.vn không có thông tin về chính sách giá hoặc chiết khấu. Agent nên từ chối trả lời trực tiếp và chuyển yêu cầu này cho bộ phận kinh doanh (handoff) thay vì suy đoán.",
        "relevant_doc_ids": [],
        "section": "out-of-scope",
        "difficulty": "medium",
        "type": "out-of-scope",
    },
    {
        "question": "Phần mềm có tích hợp gửi tin nhắn qua Zalo OA cho phụ huynh không?",
        "ground_truth_answer": "Tài liệu help.dotb.vn không đề cập đến tích hợp Zalo OA. Agent không nên khẳng định có hoặc không có tính năng này nếu không tìm thấy trong tài liệu — nên trả lời là không tìm thấy thông tin và đề nghị chuyển cho đội ngũ hỗ trợ xác nhận trực tiếp.",
        "relevant_doc_ids": [],
        "section": "out-of-scope",
        "difficulty": "medium",
        "type": "out-of-scope",
    },
    {
        "question": "Wifi ở trung tâm em bị chậm quá, có cách nào khắc phục không?",
        "ground_truth_answer": "Đây là vấn đề hạ tầng mạng của trung tâm, không liên quan đến phần mềm DotB EMS. Agent nên từ chối lịch sự vì ngoài phạm vi hỗ trợ, không nên cố tìm tài liệu liên quan trong help.dotb.vn.",
        "relevant_doc_ids": [],
        "section": "out-of-scope",
        "difficulty": "easy",
        "type": "out-of-scope",
    },
    {
        "question": "Sao em không vào được hệ thống?",
        "ground_truth_answer": "Câu hỏi quá chung chung để xác định đúng tài liệu — có thể do sai mật khẩu, tài khoản bị khóa, xung đột trình duyệt/cache, hoặc không có quyền truy cập. Agent nên hỏi lại để làm rõ (ví dụ: có thông báo lỗi cụ thể nào không, đăng nhập trên web hay mobile app) thay vì đoán và trả lời ngay.",
        "relevant_doc_ids": [
            "https://help.dotb.vn/huong-dan-xoa-cache-trinh-duyet.md"
        ],
        "section": "ambiguous",
        "difficulty": "hard",
        "type": "ambiguous",
    },
    {
        "question": "Học viên này bị sao vậy, sao em thấy trạng thái lạ vậy?",
        "ground_truth_answer": "Câu hỏi thiếu thông tin cụ thể (trạng thái nào, học viên nào) — có thể liên quan đến Delay/bảo lưu, Outstanding (học nợ), hoặc trạng thái điểm danh. Agent nên hỏi lại để biết chính xác trạng thái hiển thị là gì trước khi trả lời, thay vì chọn đại một tài liệu.",
        "relevant_doc_ids": [
            "https://help.dotb.vn/bo-phan-giao-vu/quan-li-su-vu/quan-li-delay/hoc-vien-delay-bao-luu.md",
            "https://help.dotb.vn/tinh-nang-cap-nhat-moi/cap-nhat-ems/cap-nhat-giao-dien-tinh-nang.md",
        ],
        "section": "ambiguous",
        "difficulty": "hard",
        "type": "ambiguous",
    },
]


def _infer_difficulty_and_type(content: str, token_count: int) -> tuple[str, str]:
    step_count = len(re.findall(r"\*\*Bước\s+\d+", content))
    has_condition = bool(re.search(r"(Lưu ý|Nếu |Trường hợp)", content))
    if step_count >= 3:
        q_type = "multi-step"
    elif has_condition:
        q_type = "conditional"
    else:
        q_type = "single-step"

    if token_count < 150:
        difficulty = "easy"
    elif token_count < 600:
        difficulty = "medium"
    else:
        difficulty = "hard"
    return difficulty, q_type


def _sample_parents(parents: dict) -> list[tuple[str, dict]]:
    by_section: dict[str, list[tuple[str, dict]]] = {}
    for pid, entry in parents.items():
        by_section.setdefault(entry["metadata"]["section"], []).append((pid, entry))

    rng = random.Random(RANDOM_SEED)
    selected = []
    for section, items in sorted(by_section.items()):
        quota = min(SECTION_QUOTAS.get(section, DEFAULT_QUOTA), len(items))
        selected.extend(rng.sample(items, quota))
    return selected


def _draft_one(entry: dict) -> dict | None:
    meta = entry["metadata"]
    content = entry["content"]
    breadcrumb = meta["breadcrumb"]
    if isinstance(breadcrumb, str):
        breadcrumb = json.loads(breadcrumb)

    prompt = _DRAFT_PROMPT.format(breadcrumb=" > ".join(breadcrumb), content=content)
    try:
        resp = _client.models.generate_content(model=LLM_MODEL, contents=[prompt])
        text = re.sub(r"^```(json)?|```$", "", resp.text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(text)
    except Exception as e:
        print(f"  [skip] {meta['doc_title'][:40]!r} — draft failed: {type(e).__name__}: {e}")
        return None

    difficulty, q_type = _infer_difficulty_and_type(content, meta["token_count"])
    return {
        "question": parsed["question"],
        "ground_truth_answer": parsed["answer"],
        "relevant_doc_ids": [meta["doc_url"]],
        "section": meta["section"],
        "difficulty": difficulty,
        "type": q_type,
    }


def _load_existing_by_doc_url() -> dict:
    """Resumability: a prior run may have already drafted some items before
    hitting the free-tier 15 req/min wall. Re-running shouldn't re-spend
    quota on doc_urls that already have a good draft — keyed by doc_url since
    that's stable across runs (sampling itself is deterministic via
    RANDOM_SEED, but re-drafting costs a real API call either way)."""
    if not GOLDEN_SET_PATH.exists():
        return {}
    try:
        existing = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        item["relevant_doc_ids"][0]: item
        for item in existing
        if item.get("relevant_doc_ids") and item["type"] not in ("out-of-scope", "ambiguous")
    }


def main() -> None:
    parents = json.loads(PARENTS_PATH.read_text(encoding="utf-8"))
    sampled = _sample_parents(parents)
    print(f"Sampled {len(sampled)} parents across {len(set(e['metadata']['section'] for _, e in sampled))} sections")

    cached = _load_existing_by_doc_url()
    if cached:
        print(f"Resuming: {len(cached)} doc_url(s) already drafted in an earlier run, reusing those")

    drafted = []
    for i, (pid, entry) in enumerate(sampled, start=1):
        doc_url = entry["metadata"]["doc_url"]
        if doc_url in cached:
            print(f"  [{i}/{len(sampled)}] {entry['metadata']['doc_title'][:50]} — cached, skipping")
            drafted.append(cached[doc_url])
            continue
        print(f"  [{i}/{len(sampled)}] {entry['metadata']['doc_title'][:50]}")
        item = _draft_one(entry)
        if item:
            drafted.append(item)
        time.sleep(4.5)  # free-tier LLM_MODEL: hard 15 req/min cap

    print(f"Drafted {len(drafted)}/{len(sampled)} doc-grounded items "
          f"({len(sampled) - len(drafted)} skipped on draft failure)")

    all_items = drafted + HAND_WRITTEN
    for i, item in enumerate(all_items, start=1):
        all_items[i - 1] = {**item, "id": f"q{i:03d}"}  # id set last: overrides any cached id

    GOLDEN_SET_PATH.write_text(
        json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(all_items)} draft items -> {GOLDEN_SET_PATH}")
    print("This is a DRAFT — review every ground_truth_answer before trusting eval scores against it.")


if __name__ == "__main__":
    main()

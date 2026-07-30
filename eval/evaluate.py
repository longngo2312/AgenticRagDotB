"""
DAY 4 — Evaluation harness: retrieval metrics + LLM-judge generation eval.

RAGAS was dropped: ragas 0.4.3 hard-imports langchain_community.chat_models.vertexai,
which was removed from the installed langchain-community — broken independent of
anything this project does, and nothing else here uses LangChain (that only enters
with the Day 5 agent, and downgrading langchain-community to satisfy ragas would
risk fighting whatever version LangGraph wants). Faithfulness/relevance/correctness
are judged directly via LLM_MODEL instead — see _judge_answer().

Usage:
  python eval/evaluate.py
"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, LLM_MODEL, DENSE_TOP_K, BM25_TOP_K, RERANK_TOP_K
from retrieval.rewrite import rewrite_query
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank, warmup
from retrieval.retriever import fetch_parents, retrieve
import chat_cli

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_PATH = Path(__file__).parent / "results" / "latest.json"

_client = genai.Client(api_key=GOOGLE_API_KEY)

# eval_generation makes two LLM_MODEL calls per item (generate + judge) —
# free-tier hard caps at 15 req/min, so every call funnels through here to
# stay spaced out regardless of which function is calling.
_MIN_CALL_INTERVAL_SEC = 4.2
_last_call_ts = 0.0


def _throttled_generate(**kwargs):
    global _last_call_ts
    elapsed = time.perf_counter() - _last_call_ts
    if elapsed < _MIN_CALL_INTERVAL_SEC:
        time.sleep(_MIN_CALL_INTERVAL_SEC - elapsed)
    resp = _client.models.generate_content(**kwargs)
    _last_call_ts = time.perf_counter()
    return resp


def load_golden_set() -> list[dict]:
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


def _with_md(url: str) -> str:
    """golden_set.json stores doc_urls without the trailing .md for
    readability, but the index's real doc_url metadata always has it — that's
    the actual crawled/markdown-fetch URL and the one source of truth, so
    comparisons normalize toward it rather than stripping it off."""
    return url if url.endswith(".md") else url + ".md"


def _rank_of_first_hit(candidate_urls: list[str], relevant: set[str]) -> int | None:
    for i, url in enumerate(candidate_urls, start=1):
        if url in relevant:
            return i
    return None


def _recall_at(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def _precision_at(candidate_lists: list[list[str]], relevant_sets: list[set[str]], k: int) -> float:
    vals = [
        sum(1 for u in urls[:k] if u in relevant) / k
        for urls, relevant in zip(candidate_lists, relevant_sets)
    ]
    return sum(vals) / len(vals)


def _mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r is not None) / len(ranks)


def eval_retrieval(golden_set: list[dict]) -> dict:
    """Stage the check at two points so a low score says WHERE recall was
    lost, not just THAT it was lost:
      - fusion: dense+BM25 -> RRF, before reranking narrows anything at all
        (up to DENSE_TOP_K+BM25_TOP_K unique candidates) — this is what
        CLAUDE.md's "Recall@10" demo target actually measures, since
        RERANK_TOP_K caps the final stage below 10.
      - final: retrieve()'s real output after rerank + top-k cutoff — what
        the LLM would actually be shown.

    Items with relevant_doc_ids == [] (out-of-scope) have no "correct doc" to
    check recall against, so they're excluded here — eval_generation covers
    whether the agent correctly abstains on those instead.
    """
    scored_items = [g for g in golden_set if g["relevant_doc_ids"]]

    fusion_urls_all, final_urls_all, relevant_all = [], [], []
    per_question = []

    for item in scored_items:
        relevant = {_with_md(u) for u in item["relevant_doc_ids"]}

        rewritten = rewrite_query(item["question"])
        fused = hybrid_search(rewritten.standalone, bm25_query=rewritten.bm25,
                               dense_top_k=DENSE_TOP_K, bm25_top_k=BM25_TOP_K)
        fusion_urls = [c["metadata"]["doc_url"] for c in fused]

        reranked = rerank(rewritten.standalone, fused, top_k=RERANK_TOP_K)
        final_docs = fetch_parents(reranked)
        final_urls = [d["metadata"]["doc_url"] for d in final_docs]

        fusion_urls_all.append(fusion_urls)
        final_urls_all.append(final_urls)
        relevant_all.append(relevant)

        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "fusion_rank": _rank_of_first_hit(fusion_urls, relevant),
            "final_rank": _rank_of_first_hit(final_urls, relevant),
        })

    fusion_ranks = [q["fusion_rank"] for q in per_question]
    final_ranks = [q["final_rank"] for q in per_question]

    return {
        "n_items": len(scored_items),
        "fusion": {
            "recall_at_5": _recall_at(fusion_ranks, 5),
            "recall_at_10": _recall_at(fusion_ranks, 10),
            "precision_at_5": _precision_at(fusion_urls_all, relevant_all, 5),
            "precision_at_10": _precision_at(fusion_urls_all, relevant_all, 10),
            "mrr": _mrr(fusion_ranks),
        },
        "final": {
            "recall_at_5": _recall_at(final_ranks, 5),
            "precision_at_5": _precision_at(final_urls_all, relevant_all, 5),
            "mrr": _mrr(final_ranks),
        },
        "per_question": per_question,
    }


_JUDGE_PROMPT = """Bạn là người đánh giá chất lượng câu trả lời của một trợ lý hỗ trợ khách hàng DotB EMS.

Câu hỏi: {question}

Ngữ cảnh được cung cấp cho trợ lý (nếu ghi "(không có ngữ cảnh)" nghĩa là hệ thống không tìm thấy tài liệu liên quan và trợ lý phải từ chối trả lời):
---
{context}
---

Câu trả lời của trợ lý:
---
{answer}
---

Câu trả lời đúng tham khảo (ground truth — có thể mô tả hành vi đúng thay vì một câu trả lời cụ thể, ví dụ khi câu hỏi ngoài phạm vi tài liệu hoặc quá mơ hồ):
---
{ground_truth}
---

Chấm điểm câu trả lời của trợ lý trên 3 tiêu chí, mỗi tiêu chí một số thực từ 0.0 đến 1.0:
- faithfulness: câu trả lời có bịa thông tin KHÔNG có trong Ngữ cảnh không? (1.0 = bám sát ngữ cảnh hoặc không bịa gì khi không có ngữ cảnh; 0.0 = bịa đặt nghiêm trọng)
- relevance: câu trả lời có thực sự giải quyết đúng câu hỏi không?
- correctness: câu trả lời có khớp với câu trả lời đúng tham khảo về nội dung/hành vi không?

Trả lời CHÍNH XÁC theo định dạng JSON sau, không thêm giải thích, không thêm markdown code fence:
{{"faithfulness": 0.0, "relevance": 0.0, "correctness": 0.0}}
"""


def _judge_answer(question: str, context: str, answer: str, ground_truth: str) -> dict:
    prompt = _JUDGE_PROMPT.format(
        question=question, context=context or "(không có ngữ cảnh)",
        answer=answer, ground_truth=ground_truth,
    )
    try:
        resp = _throttled_generate(model=LLM_MODEL, contents=[prompt])
        text = re.sub(r"^```(json)?|```$", "", resp.text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(text)
        return {
            "faithfulness": float(parsed["faithfulness"]),
            "relevance": float(parsed["relevance"]),
            "correctness": float(parsed["correctness"]),
        }
    except Exception as e:
        print(f"  [judge failed] {type(e).__name__}: {e}")
        return {"faithfulness": None, "relevance": None, "correctness": None}


def _generate_answer(question: str, docs: list[dict]) -> str:
    contents = chat_cli._build_contents(question, docs, [])
    resp = _throttled_generate(
        model=LLM_MODEL, contents=[contents],
        config=types.GenerateContentConfig(system_instruction=chat_cli.SYSTEM_PROMPT),
    )
    return resp.text.strip()


def eval_generation(golden_set: list[dict]) -> dict:
    """Runs the real retrieve() -> generate pipeline per item — the same
    functions scripts/chat_cli.py uses, reused directly rather than
    reimplemented — then scores the answer with an LLM judge on
    faithfulness/relevance/correctness (replacing RAGAS; see module
    docstring). Out-of-scope/ambiguous items aren't special-cased: their
    ground_truth_answer describes the *correct behavior* (decline, ask for
    clarification), so the same judge naturally scores a hallucinated answer
    low and an appropriate abstain/clarify high.
    """
    per_question = []
    for item in golden_set:
        docs = retrieve(item["question"])
        if docs:
            answer = _generate_answer(item["question"], docs)
            context = chat_cli._format_context(docs)
        else:
            answer = chat_cli.ABSTAIN_MESSAGE
            context = ""

        scores = _judge_answer(item["question"], context, answer, item["ground_truth_answer"])
        per_question.append({
            "id": item["id"],
            "type": item["type"],
            "retrieved_n": len(docs),
            "answer": answer,
            **scores,
        })

    def _avg(key: str) -> float | None:
        vals = [q[key] for q in per_question if q[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_items": len(per_question),
        "faithfulness": _avg("faithfulness"),
        "relevance": _avg("relevance"),
        "correctness": _avg("correctness"),
        "per_question": per_question,
    }


def run_full_eval() -> dict:
    warmup()  # load the cross-encoder once, not on the first golden-set item
    golden_set = load_golden_set()
    print(f"Loaded {len(golden_set)} golden items\n")

    print("=== Retrieval eval ===")
    retrieval_metrics = eval_retrieval(golden_set)
    f = retrieval_metrics["fusion"]
    z = retrieval_metrics["final"]
    print(f"  fusion ({retrieval_metrics['n_items']} items)  "
          f"recall@5={f['recall_at_5']:.2f}  recall@10={f['recall_at_10']:.2f}  "
          f"precision@5={f['precision_at_5']:.2f}  precision@10={f['precision_at_10']:.2f}  mrr={f['mrr']:.2f}")
    print(f"  final                "
          f"recall@5={z['recall_at_5']:.2f}  precision@5={z['precision_at_5']:.2f}  mrr={z['mrr']:.2f}")

    print("\n=== Generation eval ===")
    generation_metrics = eval_generation(golden_set)
    print(f"  ({generation_metrics['n_items']} items)  "
          f"faithfulness={generation_metrics['faithfulness']:.2f}  "
          f"relevance={generation_metrics['relevance']:.2f}  "
          f"correctness={generation_metrics['correctness']:.2f}")

    print("\n=== Demo targets (CLAUDE.md) ===")
    r10 = retrieval_metrics["fusion"]["recall_at_10"]
    faith = generation_metrics["faithfulness"]
    print(f"  Recall@10 >= 0.80:     {'PASS' if r10 >= 0.80 else 'FAIL'}  ({r10:.2f})")
    print(f"  Faithfulness >= 0.85:  {'PASS' if faith >= 0.85 else 'FAIL'}  ({faith:.2f})")

    return {
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "demo_targets": {
            "recall_at_10": {"target": 0.80, "actual": r10, "pass": r10 >= 0.80},
            "faithfulness": {"target": 0.85, "actual": faith, "pass": faith >= 0.85},
        },
    }


def save_results(results: dict) -> None:
    """Persist the full eval output for the dashboard (api/main.py's
    /api/eval/latest) to read — running the harness live on every dashboard
    load would burn free-tier quota (2 LLM calls/golden item, throttled to
    ~4.2s apart). The dashboard always shows the numbers from the last time
    this script was run, timestamped so staleness is visible."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), **results}
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved eval snapshot -> {RESULTS_PATH}")


if __name__ == "__main__":
    save_results(run_full_eval())

"""DAY 4 — Evaluation harness: retrieval metrics + RAGAS faithfulness."""
# TODO D4: def load_golden_set() -> list[dict]   — load eval/golden_set.json

# TODO D4: def eval_retrieval(golden_set, retriever_fn) -> dict
#   - for each item: retrieve docs, check if relevant_doc_ids appear in results
#   - compute Recall@k (k=5,10), Precision@k, MRR
#   - return metrics dict + per-question breakdown

# TODO D4: def eval_generation(golden_set, agent_fn) -> dict
#   - for each item: run agent_fn(question), get answer + retrieved docs
#   - run RAGAS: faithfulness, answer_relevance, answer_correctness
#   - return metrics dict + per-question breakdown

# TODO D4: def run_full_eval() -> dict
#   - run both retrieval + generation eval
#   - print summary table
#   - return combined metrics

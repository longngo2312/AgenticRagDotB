# DotB EMS — Agentic RAG Support Assistant

A retrieval-augmented customer-support agent for **DotB EMS**, a Vietnamese
education-management SaaS. It answers "how do I…" questions about the product
by retrieving from the official help site (`help.dotb.vn`, ~250 pages of mixed
Vietnamese/English documentation) and answering **only** from what it retrieved
— with citations, and with an explicit hand-off to a human when it can't.

The goal is deflecting tier-1 support tickets, so the design optimises for
*not being wrong* over *always having an answer*: retrieval can abstain, a
grading step can send the agent back to search again, and every generated
answer is checked against its own sources before the user ever sees it.

**Current measured results** (39-question hand-reviewed golden set, see
[Evaluation](#evaluation)):

| Metric | Target | Actual |
|---|---|---|
| Recall@10 (retrieval) | ≥ 0.80 | **0.97** ✅ |
| Faithfulness (generation) | ≥ 0.85 | **1.00** ✅ |
| Answer relevance | — | 0.97 |
| Answer correctness | — | 0.93 |

Everything runs on free tiers: Google Gemini for the LLM and embeddings, a
local ChromaDB file for vectors, and a locally-run cross-encoder for reranking.

---

## Running it

### 1. Install and configure

```bash
pip install -r requirements.txt

cp .env.example .env      # then edit .env and set GOOGLE_API_KEY
```

Get a free key at [aistudio.google.com](https://aistudio.google.com) → *Get API Key*.
Nothing else in `.env` is required — LangSmith tracing is optional.

### 2. Build the index (first time only)

The index is not committed to git, so a fresh clone has to build it once:

```bash
python scripts/run_ingestion.py --full
```

This crawls `help.dotb.vn`, chunks every page, embeds the chunks, and writes
`data/chroma_db/`, `data/parent_docs.json`, and `data/bm25_index.pkl`. Expect
several minutes. It is **idempotent** — re-running skips unchanged documents,
so it is safe to interrupt and resume.

> Run `python scripts/run_ingestion.py` without `--full` to stop after chunking
> and inspect `data/chunks_preview.json` before spending any embedding quota.

### 3. Start the app

```bash
python scripts/launch.py --no-eval
```

Then open:

| | |
|---|---|
| **Chat** | <http://127.0.0.1:8000/> |
| **Dashboard** | <http://127.0.0.1:8000/dashboard.html> |

`--no-eval` starts the server immediately and shows the last saved evaluation
snapshot. **Drop the flag** (`python scripts/launch.py`) to re-run the full
evaluation against live data first — that makes two throttled Gemini calls per
golden-set question, so it takes several minutes and spends real free-tier
quota before the server comes up.

| Flag | Effect |
|---|---|
| `--no-eval` | Skip the eval refresh (fastest start) |
| `--port 8080` | Serve on a different port |
| `--reload` | Auto-restart on backend edits (development only) |

<details>
<summary>Other entry points</summary>

```bash
python scripts/agent_cli.py      # terminal REPL, prints the agent's reasoning trace
python scripts/chat_cli.py       # terminal REPL, pre-agent RAG baseline
python eval/evaluate.py          # run the eval harness and save a snapshot
uvicorn api.main:app --port 8000 # start the server directly, no eval step
```

`scripts/agent_cli.py` is the most useful of these for understanding the
system: it streams each node's decision as it fires, so you can watch a
question get routed, retrieved, graded, and fact-checked in real time.
</details>

---

## User flow

What happens between a user typing a question and seeing a cited answer:

```
   Browser (frontend/index.html)
        │  POST /api/chat  { question, messages }
        ▼
   FastAPI (api/main.py)
        │
        ▼
   LangGraph agent (agent/graph.py)
        │
        ├─ guardrails ── rule-based: empty / oversized / prompt-injection?
        │                   └── blocked ─────────────────────────────► reply
        │
        ├─ router ────── one LLM call: pick a route AND rewrite the
        │                question to stand on its own
        │                   ├── clarify ── question too vague ───────► reply
        │                   └── handoff ── out of scope ────────────► reply
        │
        ├─ retrieve ──── the retrieval pipeline (below)
        │
        ├─ grade ─────── is the retrieved context enough to answer?
        │                   └── no ── rewrite the query, search again
        │                              (up to AGENT_MAX_ATTEMPTS, then handoff)
        │
        ├─ generate ──── answer in Vietnamese, citing [1] [2] per source
        │
        └─ faithfulness ─ score the answer against its own context
                            ├── below threshold ── handoff ─────────► reply
                            └── passes ─────────────────────────────► reply
                                                                        │
   Browser renders answer + source cards + 👍/👎 ◄─────────────────────┘
```

Only the last box is what the user sees. Every terminal path writes the same
`answer` field, so the UI never needs to know which branch ran.

### Inside `retrieve`

```
question
   │
   ├─► rewrite ──── condense against chat history; expand Vietnamese ↔ English
   │                glossary terms for the lexical search only
   │
   ├─► dense search (ChromaDB, cosine, top-30) ─┐
   │                                            ├─► RRF fusion ──► rerank ──► top-5
   ├─► BM25 search (rank_bm25, top-30) ─────────┘                    │
   │                                                                 │
   └─► fetch parents ◄───────────────────────────────────────────────┘
              │
              ▼
       full parent documents handed to the LLM
```

Two ideas do most of the work here:

**Parent–child chunking.** Small ~500-token children are what get embedded and
searched, because a small chunk matches a specific question precisely. But the
LLM is handed the ~1500-token *parent* those children came from, because a
precise match is useless if the surrounding steps got cut off. Children are
searched; parents are read.

**Hybrid retrieval.** Dense vector search catches meaning ("how do I pause a
student's enrolment"), BM25 catches exact product terms that embeddings blur
("Mass Update", "Convert Lead"). The Vietnamese docs mix both constantly, so
neither alone is enough — Reciprocal Rank Fusion merges the two rankings, then
a local cross-encoder reranks the merged shortlist and drops anything below a
score threshold. That threshold is what lets retrieval return *nothing*, which
is the honest answer surprisingly often.

---

## Architecture

```
config.py               single source of truth for every tuneable knob

ingestion/              build the index (run once, re-runnable)
  crawler.py              async-fetch every .md listed in help.dotb.vn/llms.txt
  image_cache.py          download GitBook-hosted screenshots to a stable local path
  parser.py               GitBook markup → plain text; URL → breadcrumb metadata
  image_captioner.py      caption screenshots with a vision LLM, using step context
  chunker.py              heading-aware parent/child splitter
  embedder.py             batch embeddings, with rate-limit backoff
  indexer.py              children → ChromaDB, parents → JSON; purges stale chunks

retrieval/              answer-time search
  rewrite.py              history condensing + glossary expansion
  dense.py                ChromaDB cosine search
  bm25_index.py           BM25 lexical search
  hybrid.py               Reciprocal Rank Fusion
  reranker.py             bge-reranker-v2-m3 cross-encoder + abstain threshold
  retriever.py            the whole pipeline, end to end

agent/                  the decision-making layer
  state.py                AgentState — what flows between nodes
  nodes.py                one function per node
  prompts.py              all Vietnamese prompts
  tools.py                search_docs, lookup_glossary, create_handoff
  graph.py                StateGraph wiring

api/main.py             FastAPI: chat, feedback, handoff, and dashboard data
frontend/               chat widget + dashboard (vanilla JS, no build step)
eval/                   golden set + evaluation harness
scripts/                entry points (launch, ingestion, CLIs, tracing)
```

### Design decisions worth knowing

**Degrade, never crash.** Every LLM call on the interactive path has a fallback
chosen so the failure mode is the *safe* one. If the router fails, retrieval
runs anyway. If grading fails, the answer proceeds (it still gets
fact-checked). If the embedding API rate-limits, dense search returns empty and
BM25 carries the query alone. A rate limit degrades answer quality; it doesn't
take the assistant down.

**The faithfulness check is a gate, not a metric.** `faithfulness_node` scores
each answer against the context it was given, and `graph.py` hands off to a
human if it scores too low. This is also why `/api/chat` doesn't stream tokens:
an answer that fails the check is never shown at all, so there is nothing
correct to stream word-by-word. The UI does a client-side typewriter reveal
instead.

**Idempotent ingestion.** Chunk IDs are content hashes, so re-running skips
unchanged documents and re-embeds only what actually changed. Editing a
document's content mints a new ID, so the indexer also purges the superseded
chunk — scoped per document, so a partial crawl can never wipe pages it simply
didn't reach.

---

## Evaluation

```bash
python eval/evaluate.py
```

Scores the system against `eval/golden_set.json` — 39 questions drafted from
real documentation pages and then hand-corrected, deliberately including
out-of-scope and ambiguous questions where the *correct* behaviour is to
decline or ask a follow-up rather than answer.

Results are written to `eval/results/latest.json`, which is what the dashboard
displays. Two things are measured separately:

- **Retrieval** — Recall@k, Precision@k, and MRR, measured at two points: after
  fusion (before reranking narrows anything) and after the final rerank. Two
  checkpoints mean a bad score says *where* recall was lost, not just that it
  was. This runs the same `retrieve()` the live chat uses.
- **Generation** — an LLM judge scores faithfulness, relevance, and correctness
  against the reference answer. Note this path generates via the simpler
  pre-agent prompt rather than the full LangGraph agent, so it measures the
  retrieval-plus-grounding quality, not the routing and self-correction logic.

Both stages funnel every API call through one throttle to stay inside the free
tier's 15 requests/minute, which is why a full run takes minutes.

---

## Notes and constraints

- **Free-tier quota is the binding constraint.** The daily cap is reachable
  during a heavy ingestion run. Anything that would hit Gemini repeatedly is
  built to persist its results and read them back rather than recompute.
- **First start is slow.** The cross-encoder (~1.1 GB) loads at startup and
  runs a dummy prediction to force CUDA kernel compilation, so the first real
  question isn't the one that pays for it.
- **Vietnamese-first.** Prompts, answers, and the UI are Vietnamese; the docs
  mix in English product terms, which is exactly why BM25 is in the pipeline
  alongside embeddings.
```

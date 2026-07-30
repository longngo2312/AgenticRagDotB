// DotB EMS dashboard — reads /api/eval/latest, /api/pipeline/*, /api/data-model.
const $ = (id) => document.getElementById(id);

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    const err = new Error(detail);
    err.status = r.status;
    throw err;
  }
  return r.json();
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function fmt(v, digits = 2) {
  return (v === undefined || v === null) ? "—" : Number(v).toFixed(digits);
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s ?? "";
  return d.innerHTML;
}

// ── KPI strip ────────────────────────────────────────────────────────────
function renderKpis(evalData) {
  const grid = $("kpiGrid");
  grid.innerHTML = "";
  if (!evalData) {
    grid.appendChild(el("div", "empty-state",
      "Chưa có số liệu eval. Chạy <code>python eval/evaluate.py</code> để tạo snapshot, " +
      "sau đó tải lại trang này."));
    $("kpiMeta").textContent = "";
    return;
  }
  const t = evalData.demo_targets || {};
  const rows = [
    { label: "Recall@10 (fusion)", value: t.recall_at_10?.actual, target: t.recall_at_10?.target, pass: t.recall_at_10?.pass },
    { label: "Faithfulness", value: t.faithfulness?.actual, target: t.faithfulness?.target, pass: t.faithfulness?.pass },
    { label: "Relevance", value: evalData.generation?.relevance },
    { label: "Correctness", value: evalData.generation?.correctness },
    { label: "MRR (final rerank)", value: evalData.retrieval?.final?.mrr },
  ];
  for (const r of rows) {
    const card = el("div", "card kpi");
    const badge = r.pass === undefined
      ? ""
      : `<span class="badge ${r.pass ? "pass" : "fail"}">${r.pass ? "PASS" : "FAIL"}</span>`;
    card.innerHTML = `
      <div class="label">${r.label}</div>
      <div class="value">${fmt(r.value)} ${badge}</div>
      <div class="target">${r.target ? `mục tiêu ≥ ${r.target}` : "&nbsp;"}</div>
    `;
    grid.appendChild(card);
  }
  const when = evalData.generated_at ? new Date(evalData.generated_at).toLocaleString("vi-VN") : "?";
  $("kpiMeta").textContent = `Snapshot: ${when} · ${evalData.generation?.n_items ?? "?"} câu hỏi golden set`;
}

// ── Agent flow diagram ──────────────────────────────────────────────────
const SPINE = ["guardrails", "router", "retrieve", "grade", "generate", "faithfulness", "__end__"];
const BRANCH = ["clarify", "handoff"];
const NODE_LABEL = { __end__: "END", __start__: "START" };

function displayName(id) { return NODE_LABEL[id] || id; }

function renderAgentGraph(graph) {
  const host = $("agentGraph");
  host.innerHTML = "";
  if (!graph) {
    host.appendChild(el("div", "empty-state", "Không tải được cấu trúc graph."));
    return;
  }
  const edges = graph.edges.filter(e => e.source !== "__start__");
  const spineIndex = Object.fromEntries(SPINE.map((id, i) => [id, i]));

  const wrap = el("div", "graph-grid card");
  wrap.style.padding = "1rem";

  const spineCol = el("div", "spine");
  for (const id of SPINE) {
    const card = el("div", "node-card card");
    const outgoing = edges.filter(e => e.source === id && !BRANCH.includes(e.target));
    const chips = outgoing.map(e => {
      const isLoop = spineIndex[e.target] !== undefined && spineIndex[e.target] < spineIndex[id];
      const label = e.label ? `${displayName(e.target)} · ${e.label}` : displayName(e.target);
      return `<span class="edge-chip${isLoop ? " loop" : ""}">→ ${escapeHtml(label)}${isLoop ? " (loop)" : ""}</span>`;
    }).join("");
    const entry = id === "guardrails" ? '<span class="badge blue">ENTRY</span>' : "";
    card.innerHTML = `<div class="id">${displayName(id)} ${entry}</div><div class="edges">${chips}</div>`;
    spineCol.appendChild(card);
  }

  const branchCol = el("div", "branch-lane");
  for (const id of BRANCH) {
    const incoming = edges.filter(e => e.target === id);
    const outgoing = edges.filter(e => e.source === id);
    const inChips = incoming.map(e =>
      `<span class="edge-chip">← ${escapeHtml(displayName(e.source))}${e.label ? " · " + escapeHtml(e.label) : ""}</span>`
    ).join("");
    const outChips = outgoing.map(e => `<span class="edge-chip">→ ${escapeHtml(displayName(e.target))}</span>`).join("");
    const card = el("div", "node-card card terminal");
    card.innerHTML = `<div class="id">${displayName(id)}</div><div class="edges">${inChips}${outChips}</div>`;
    branchCol.appendChild(card);
  }

  wrap.append(spineCol, branchCol);
  host.appendChild(wrap);

  const legend = el("div", "stat-strip");
  legend.innerHTML = `
    <span class="stat-pill">dashed chip = graph edge</span>
    <span class="stat-pill">amber "(loop)" = grade retries retrieve, max <b>${graph.max_attempts ?? "3"}</b> lần</span>
  `;
  host.appendChild(legend);
}

// ── Ingestion pipeline ───────────────────────────────────────────────────
function renderIngestion(ing) {
  const host = $("ingestion");
  host.innerHTML = "";
  if (!ing) {
    host.appendChild(el("div", "empty-state", "Không tải được số liệu ingestion."));
    return;
  }
  const countFor = {
    crawler: `${ing.documents} tài liệu`,
    image_captioner: `${ing.images_referenced} ảnh tham chiếu`,
    chunker: `${ing.parent_chunks} parent · ${ing.child_chunks} child`,
    embedder: `avg ${ing.child_token_avg ?? "?"} tok/child`,
    indexer: `BM25 corpus: ${ing.bm25_corpus_size ?? "?"}`,
  };

  const row = el("div", "stage-row card");
  row.style.padding = "1rem";
  ing.stages.forEach((s, i) => {
    if (i > 0) row.appendChild(el("div", "stage-arrow", "→"));
    const box = el("div", "stage-box");
    const n = countFor[s.name] ? `<div class="n">${countFor[s.name]}</div>` : "";
    box.innerHTML = `<div class="name">${escapeHtml(s.name)}</div><div class="desc">${escapeHtml(s.description)}</div>${n}`;
    row.appendChild(box);
  });
  host.appendChild(row);

  const strip = el("div", "stat-strip");
  strip.innerHTML = `
    <span class="stat-pill">Tài liệu: <b>${ing.documents}</b></span>
    <span class="stat-pill">Parent chunks: <b>${ing.parent_chunks}</b></span>
    <span class="stat-pill">Child chunks: <b>${ing.child_chunks}</b></span>
    <span class="stat-pill">BM25 corpus: <b>${ing.bm25_corpus_size ?? "—"}</b></span>
    <span class="stat-pill">Child token range: <b>${ing.child_token_range ? ing.child_token_range.join("–") : "—"}</b></span>
    <span class="stat-pill">Ảnh tham chiếu: <b>${ing.images_referenced}</b></span>
  `;
  host.appendChild(strip);
}

// ── Data model ───────────────────────────────────────────────────────────
function renderDataModel(dm) {
  const host = $("dataModel");
  host.innerHTML = "";
  if (!dm) {
    host.appendChild(el("div", "empty-state", "Không tải được data model."));
    return;
  }

  const schemaCard = el("div", "card");
  schemaCard.style.padding = "1rem";
  const child = dm.chunk_schema.child, parent = dm.chunk_schema.parent;
  schemaCard.innerHTML = `
    <div class="stat-strip" style="margin-top:0">
      <span class="stat-pill">Child ~${child.approx_tokens} tok → ${escapeHtml(child.stored_in)}</span>
      <span class="stat-pill">Parent ~${parent.approx_tokens} tok → ${escapeHtml(parent.stored_in)}</span>
    </div>
    <div class="desc" style="margin-top:0.6rem;font-size:0.82rem;color:var(--ink-2)">
      Fields chung: <span class="mono">${child.fields.join(", ")}</span>
    </div>
  `;
  host.appendChild(schemaCard);

  if (dm.chunk_schema.sample_parent) {
    const s = dm.chunk_schema.sample_parent;
    const sampleCard = el("div", "card");
    sampleCard.style.cssText = "padding:1rem;margin-top:0.7rem";
    sampleCard.innerHTML = `
      <div class="label" style="font-size:0.72rem;color:var(--ink-3);text-transform:uppercase">Ví dụ parent chunk thật</div>
      <div style="font-weight:600;margin-top:0.2rem">${escapeHtml(s.doc_title)}</div>
      <div class="crumb" style="font-size:0.76rem;color:var(--ink-3)">${escapeHtml((s.breadcrumb || []).join(" > "))} · ${s.token_count} tok</div>
      <pre class="mono" style="white-space:pre-wrap;font-size:0.76rem;color:var(--ink-2);margin-top:0.5rem;max-height:160px;overflow:auto">${escapeHtml(s.content_preview)}…</pre>
      <a href="${s.doc_url}" target="_blank" rel="noopener" style="font-size:0.78rem">${s.doc_url}</a>
    `;
    host.appendChild(sampleCard);
  }

  const stateCard = el("div", "card");
  stateCard.style.cssText = "padding:1rem;margin-top:0.7rem";
  const rows = dm.agent_state_fields.map(f =>
    `<tr><td class="name">${f.name}</td><td class="type">${f.type}</td><td>${escapeHtml(f.description)}</td></tr>`
  ).join("");
  stateCard.innerHTML = `
    <div class="label" style="font-size:0.72rem;color:var(--ink-3);text-transform:uppercase;margin-bottom:0.4rem">AgentState</div>
    <table class="field-table"><tbody>${rows}</tbody></table>
  `;
  host.appendChild(stateCard);

  const cfgCard = el("div", "card");
  cfgCard.style.cssText = "padding:1rem;margin-top:0.7rem";
  const pills = Object.entries(dm.config).map(([k, v]) =>
    `<span class="stat-pill">${k}: <b>${escapeHtml(String(v))}</b></span>`
  ).join("");
  cfgCard.innerHTML = `
    <div class="label" style="font-size:0.72rem;color:var(--ink-3);text-transform:uppercase;margin-bottom:0.4rem">config.py</div>
    <div class="stat-strip">${pills}</div>
  `;
  host.appendChild(cfgCard);
}

// ── Eval detail ──────────────────────────────────────────────────────────
function renderEvalDetail(evalData) {
  const host = $("evalDetail");
  host.innerHTML = "";
  if (!evalData) {
    host.appendChild(el("div", "empty-state",
      "Chưa có snapshot eval. Chạy <code>python eval/evaluate.py</code> rồi tải lại trang."));
    $("evalSub").textContent = "";
    return;
  }
  $("evalSub").textContent = `${evalData.retrieval?.n_items ?? 0} câu có nhãn recall · ${evalData.generation?.n_items ?? 0} câu chấm generation`;

  const f = evalData.retrieval?.fusion || {}, z = evalData.retrieval?.final || {};
  const retCard = el("div", "card");
  retCard.style.padding = "1rem";
  retCard.innerHTML = `
    <table class="evtable">
      <thead><tr><th>Giai đoạn</th><th>Recall@5</th><th>Recall@10</th><th>Precision@5</th><th>Precision@10</th><th>MRR</th></tr></thead>
      <tbody>
        <tr><td>Fusion (dense+BM25, trước rerank)</td><td class="num">${fmt(f.recall_at_5)}</td><td class="num">${fmt(f.recall_at_10)}</td><td class="num">${fmt(f.precision_at_5)}</td><td class="num">${fmt(f.precision_at_10)}</td><td class="num">${fmt(f.mrr)}</td></tr>
        <tr><td>Final (sau rerank, top-${z.k ?? 5})</td><td class="num">${fmt(z.recall_at_5)}</td><td class="num">—</td><td class="num">${fmt(z.precision_at_5)}</td><td class="num">—</td><td class="num">${fmt(z.mrr)}</td></tr>
      </tbody>
    </table>
  `;
  host.appendChild(retCard);

  const g = evalData.generation || {};
  const genCard = el("div", "card");
  genCard.style.cssText = "padding:1rem;margin-top:0.7rem";
  genCard.innerHTML = `
    <div class="stat-strip" style="margin-top:0">
      <span class="stat-pill">Faithfulness: <b>${fmt(g.faithfulness)}</b></span>
      <span class="stat-pill">Relevance: <b>${fmt(g.relevance)}</b></span>
      <span class="stat-pill">Correctness: <b>${fmt(g.correctness)}</b></span>
    </div>
  `;
  host.appendChild(genCard);

  // Per-question — merge retrieval + generation rows by id.
  const retByld = Object.fromEntries((evalData.retrieval?.per_question || []).map(q => [q.id, q]));
  const genRows = evalData.generation?.per_question || [];
  if (genRows.length) {
    const listCard = el("div", "card");
    listCard.style.cssText = "padding:0.5rem 1rem;margin-top:0.7rem";
    for (const g of genRows) {
      const r = retByld[g.id];
      const details = el("details", "qa");
      details.innerHTML = `
        <summary>
          <span class="qtext">${escapeHtml(g.id)} — ${escapeHtml(g.question || "")}</span>
          <span class="badge violet">${escapeHtml(g.type || "")}</span>
          <span class="badge ${g.faithfulness >= 0.85 ? "pass" : "fail"}">F ${fmt(g.faithfulness)}</span>
        </summary>
        <div class="qa-body">
          ${r ? `Fusion rank: <b>${r.fusion_rank ?? "miss"}</b> · Final rank: <b>${r.final_rank ?? "miss"}</b><br>` : ""}
          Relevance: ${fmt(g.relevance)} · Correctness: ${fmt(g.correctness)} · retrieved_n: ${g.retrieved_n}<br>
          <div style="margin-top:0.4rem">${escapeHtml(g.answer || "")}</div>
        </div>
      `;
      listCard.appendChild(details);
    }
    host.appendChild(listCard);
  }
}

// ── Boot ─────────────────────────────────────────────────────────────────
async function safe(promise) {
  try { return await promise; } catch (e) { console.error(e); return null; }
}

(async function main() {
  const [evalData, graph, ingestion, dataModel] = await Promise.all([
    safe(getJSON("/api/eval/latest")),
    safe(getJSON("/api/pipeline/agent-graph")),
    safe(getJSON("/api/pipeline/ingestion")),
    safe(getJSON("/api/data-model")),
  ]);
  renderKpis(evalData);
  renderAgentGraph(graph);
  renderIngestion(ingestion);
  renderDataModel(dataModel);
  renderEvalDetail(evalData);
})();

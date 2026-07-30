// DotB EMS chat widget — talks to POST /api/chat, /api/feedback, /api/handoff.
const thread = document.getElementById("thread");
const composer = document.getElementById("composer");
const questionEl = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");
const handoffBtn = document.getElementById("handoffBtn");

let history = []; // [{role, content}, ...] sent to the agent for follow-up continuity
let lastQuestion = "";

function scrollToEnd() {
  thread.scrollTop = thread.scrollHeight;
}

function addMessage(role, { text = "", handoff = false } = {}) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}` + (handoff ? " handoff" : "");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  msg.appendChild(bubble);
  thread.appendChild(msg);
  scrollToEnd();
  return { msg, bubble };
}

function addTyping() {
  const { msg, bubble } = addMessage("assistant");
  bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
  return msg;
}

// Client-side typewriter reveal — the answer already passed the agent's
// faithfulness check server-side before this ever runs, so there's nothing
// left to stream token-by-token; this is purely a UX touch.
function typewriter(bubble, text, onDone) {
  bubble.textContent = "";
  let i = 0;
  const step = Math.max(1, Math.round(text.length / 120));
  const id = setInterval(() => {
    i += step;
    bubble.textContent = text.slice(0, i);
    if (i >= text.length) {
      clearInterval(id);
      bubble.textContent = text;
      onDone && onDone();
    }
  }, 12);
}

function addSources(afterMsg, sources) {
  if (!sources || !sources.length) return;
  const wrap = document.createElement("div");
  wrap.className = "sources";
  for (const s of sources) {
    const a = document.createElement("a");
    a.className = "source-card";
    a.href = s.url || "#";
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    const crumb = (s.breadcrumb || []).join(" > ");
    a.innerHTML = `<span class="title">${escapeHtml(s.title || s.url || "Nguồn")}</span>` +
                  (crumb ? `<span class="crumb">${escapeHtml(crumb)}</span>` : "");
    wrap.appendChild(a);
  }
  afterMsg.appendChild(wrap);
  scrollToEnd();
}

function addFeedback(afterMsg, question, answer) {
  const row = document.createElement("div");
  row.className = "feedback-row";
  const up = document.createElement("button");
  up.textContent = "👍";
  const down = document.createElement("button");
  down.textContent = "👎";

  async function vote(kind, btn, otherBtn) {
    otherBtn.classList.remove("active");
    btn.classList.toggle("active", true);
    if (kind === "down") btn.classList.add("down");
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, answer, vote: kind }),
      });
    } catch (e) { /* best-effort — feedback is non-blocking */ }
  }
  up.addEventListener("click", () => vote("up", up, down));
  down.addEventListener("click", () => vote("down", down, up));
  row.append(up, down);
  afterMsg.appendChild(row);
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

async function sendQuestion(question) {
  addMessage("user", { text: question });
  lastQuestion = question;
  const typingMsg = addTyping();
  sendBtn.disabled = true;

  let data;
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, messages: history }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
  } catch (e) {
    typingMsg.remove();
    addMessage("assistant", {
      text: "Xin lỗi, hệ thống đang gặp sự cố kết nối. Bạn thử lại sau ít phút nhé.",
      handoff: true,
    });
    sendBtn.disabled = false;
    return;
  }

  typingMsg.remove();
  const { msg, bubble } = addMessage("assistant", { handoff: data.handoff });
  typewriter(bubble, data.answer || "(không có câu trả lời)", () => {
    addSources(msg, data.sources);
    addFeedback(msg, question, data.answer || "");
  });

  history.push({ role: "user", content: question });
  history.push({ role: "assistant", content: data.answer || "" });
  sendBtn.disabled = false;
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = questionEl.value.trim();
  if (!q) return;
  questionEl.value = "";
  sendQuestion(q);
});

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

handoffBtn.addEventListener("click", async () => {
  handoffBtn.disabled = true;
  try {
    const resp = await fetch("/api/handoff", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: lastQuestion }),
    });
    const data = await resp.json();
    addMessage("assistant", { text: data.message, handoff: true });
  } catch (e) {
    addMessage("assistant", {
      text: "Không thể ghi nhận yêu cầu lúc này, bạn thử lại sau nhé.",
      handoff: true,
    });
  }
  handoffBtn.disabled = false;
});

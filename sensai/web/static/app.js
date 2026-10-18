"use strict";

/* Duolingoy front end: plain JavaScript, no build step, and no request to
   anything but the local Sensai server (the app works offline).
   Everything the model or the learner writes is set as text, or goes through
   renderMarkdown(), which escapes HTML before adding its own tags. */

const MASCOT = "/static/mascot.svg";

const $ = (selector) => document.querySelector(selector);

/** Build an element. `html` is reserved for the output of renderMarkdown(). */
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/* ---------- markdown (the tutor writes bold, lists, headings, tables) ---------- */

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const escapeHtml = (text) => text.replace(/[&<>"']/g, (c) => ESCAPES[c]);

function inline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?!\s)([^*]+?)\*(?!\w)/g, "$1<em>$2</em>");
}

const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
const isTableRule = (line) => /^\s*\|?(\s*:?-{2,}:?\s*\|)+\s*(:?-{2,}:?\s*)?$/.test(line);
const tableCells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());

function renderMarkdown(source) {
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let paragraph = [];
  let list = null;
  const flush = () => {
    if (paragraph.length) out.push(`<p>${paragraph.map(inline).join("<br>")}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (list) out.push(`</${list}>`);
    list = null;
  };
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    let m;
    if (/^\s*```/.test(line)) {
      flush();
      closeList();
      const code = [];
      while (i + 1 < lines.length && !/^\s*```/.test(lines[i + 1])) code.push(lines[++i]);
      i += 1; // closing fence (absent while the answer is still streaming)
      out.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
    } else if (isTableRow(line) && i + 1 < lines.length && isTableRule(lines[i + 1])) {
      flush();
      closeList();
      const head = tableCells(line).map((cell) => `<th>${inline(cell)}</th>`).join("");
      const rows = [];
      i += 1;
      while (i + 1 < lines.length && isTableRow(lines[i + 1])) {
        rows.push(`<tr>${tableCells(lines[++i]).map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`);
      }
      out.push(`<table><thead><tr>${head}</tr></thead><tbody>${rows.join("")}</tbody></table>`);
    } else if ((m = line.match(/^\s*(#{1,6})\s+(.+?)\s*#*\s*$/))) {
      flush();
      closeList();
      const level = Math.min(6, m[1].length + 2);
      out.push(`<h${level}>${inline(m[2])}</h${level}>`);
    } else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      flush();
      closeList();
      out.push("<hr>");
    } else if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {
      flush();
      if (list !== "ul") {
        closeList();
        out.push("<ul>");
        list = "ul";
      }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*(\d+)[.)]\s+(.*)$/))) {
      flush();
      if (list !== "ol") {
        closeList();
        out.push(`<ol start="${Number(m[1])}">`);
        list = "ol";
      }
      out.push(`<li>${inline(m[2])}</li>`);
    } else if (!line.trim()) {
      flush();
      closeList();
    } else {
      closeList();
      paragraph.push(line);
    }
  }
  flush();
  closeList();
  return out.join("");
}

/* ---------- talking to the server ---------- */

async function errorText(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join(", ");
  } catch {
    /* not JSON */
  }
  return `${response.status} ${response.statusText}`;
}

async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorText(response));
  return response.json();
}

async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (data) yield JSON.parse(data);
    }
  }
}

/** POST `body` to a streaming endpoint and dispatch each event to handlers[event.type]. */
async function stream(path, body, handlers) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorText(response));
  for await (const event of readEvents(response)) handlers[event.type]?.(event);
}

/* ---------- state and page elements ---------- */

const ui = {
  thread: $("#thread"),
  composer: $("#composer"),
  input: $("#input"),
  send: $("#send"),
  stop: $("#stop"),
  status: $("#status"),
  device: $("#device"),
  model: $("#model"),
  sidebar: $("#sidebar"),
  toggle: $("#panel-toggle"),
  close: $("#sidebar-close"),
  stats: $("#stats"),
  words: $("#words"),
  wordsCount: $("#words-count"),
  mistakes: $("#mistakes"),
  gauge: $("#gauge"),
  legend: $("#legend"),
  ollama: $("#ollama-count"),
  sources: $("#sources"),
  summary: $("#summary"),
  compress: $("#compress"),
  metrics: $("#metrics"),
  learner: $("#learner"),
  profile: $("#profile"),
  profileForm: $("#profile-form"),
  profileKey: $("#profile-key"),
  profileValue: $("#profile-value"),
  sessions: $("#sessions"),
  newSession: $("#new-session"),
  toasts: $("#toasts"),
  user: $("#user"),
  banner: $("#switch-banner"),
  bannerText: $("#switch-banner-text"),
  bannerReset: $("#switch-banner-reset"),
  compareBar: $("#compare-bar"),
  compareFeature: $("#compare-feature"),
  compareJudge: $("#compare-judge"),
  switches: $("#switches"),
  switchesReset: $("#switches-reset"),
  demos: $("#demos"),
  users: $("#users"),
  userForm: $("#user-form"),
  userName: $("#user-name"),
  historyUser: $("#history-user"),
  historyStats: $("#history-stats"),
  activity: $("#activity"),
  preview: $("#preview"),
  previewTitle: $("#preview-title"),
  previewResume: $("#preview-resume"),
  previewClose: $("#preview-close"),
  previewThread: $("#preview-thread"),
};

const app = { mode: "chat", busy: false, state: null, memory: null };

function setStatus(state, text) {
  ui.status.dataset.state = state;
  ui.status.querySelector(".status-text").textContent = text;
}

function setBusy(busy, text = "") {
  app.busy = busy;
  ui.send.hidden = busy;
  ui.stop.hidden = !busy;
  ui.stop.disabled = false;
  const controls = ".mode, #model, #user, #compress, #new-session, .session .btn, .forget, #switches-reset, #switch-banner-reset, .user .btn, #user-form .btn, #compare-feature";
  for (const control of document.querySelectorAll(controls)) {
    control.disabled = busy || (control.dataset.mode === "agent" && !app.state?.features.agent);
  }
  if (app.state) renderSwitches(app.state.switches); // switches and demos: disabled while busy, or when unavailable
  setStatus(busy ? "busy" : "idle", busy ? text : "Prêt");
}

/** Where Ollama put the model: GPU when it can, CPU for the rest (it decides by itself). */
function renderDevice(placement) {
  ui.device.hidden = !placement;
  if (!placement) return;
  const share = placement.gpu_share;
  ui.device.textContent = share >= 0.995 ? "100 % GPU" : share > 0 ? `${Math.max(1, Math.round(share * 100))} % GPU` : "CPU";
  ui.device.dataset.kind = share > 0 ? "gpu" : "cpu";
  ui.device.title = `Modèle placé par Ollama : ${placement.label}`;
}

function toast(text, kind = "info") {
  const node = h("div", { class: `toast toast-${kind}`, role: kind === "error" ? "alert" : null, text });
  ui.toasts.append(node);
  setTimeout(() => node.remove(), kind === "error" ? 7000 : 3500);
}

const plural = (n, one, many) => `${n} ${n > 1 ? many : one}`;

const formatDate = (seconds) =>
  new Date(seconds * 1000).toLocaleString("fr-FR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });

/* ---------- conversation thread ---------- */

const SUGGESTIONS = {
  chat: [
    "Comment dit-on « chat » en japonais ?",
    "Quelle est la différence entre wa et ga ?",
    "Fais-moi réviser mes mots avec un petit exercice.",
  ],
  agent: [
    "Écris un quiz Python à partir de docs/lecons/compteurs.md, teste-le, puis enregistre-le dans workspace/quiz.py",
    "Crée une fiche de révision à partir de docs/lecons/animaux.md dans workspace/fiche.md",
  ],
  compare: [
    "Qu'est-ce qui tombe à l'interro de vendredi ?",
    "Comment compter trois crayons en japonais ?",
    "Comment dit-on « chat » ?",
  ],
};

const WELCOME = {
  chat: ["Konnichiwa !", "Je suis ton tuteur de langues. Je retiens tes mots et tes erreurs, et je m'appuie sur tes leçons."],
  agent: ["Que dois-je préparer ?", "Confie une tâche à l'agent : il lit tes leçons, écrit et teste ses programmes, et te demande toujours avant de modifier un fichier."],
  compare: ["Avec ou sans ?", "Pose une question : elle est envoyée deux fois, avec puis sans la fonctionnalité choisie au-dessus du champ, et les deux réponses s'affichent côte à côte."],
};

function welcome() {
  const [title, text] = WELCOME[app.mode];
  return h(
    "div",
    { class: "welcome" },
    h("img", { src: MASCOT, alt: "" }),
    h("h1", { text: title }),
    h("p", { text }),
    h(
      "div",
      { class: "chips" },
      SUGGESTIONS[app.mode].map((text) =>
        h("button", {
          type: "button",
          class: "chip",
          text,
          onclick: () => {
            ui.input.value = text;
            autosize();
            ui.input.focus();
          },
        }),
      ),
    ),
  );
}

function scrollToEnd(force = false) {
  const thread = ui.thread;
  if (force || thread.scrollHeight - thread.scrollTop - thread.clientHeight < 160) thread.scrollTop = thread.scrollHeight;
}

function append(node) {
  ui.thread.querySelector(".welcome")?.remove();
  ui.thread.append(node);
  scrollToEnd(true);
  return node;
}

const AGENT_PREFIX = /^\[tâche agent\]\s*/;

function userMessage(text, summarized = false) {
  const agent = AGENT_PREFIX.test(text);
  return h(
    "div",
    { class: `msg msg-user${summarized ? " is-summarized" : ""}` },
    h("div", { class: "bubble" }, agent ? [h("span", { class: "msg-tag", text: "Tâche agent" }), "\n"] : null, text.replace(AGENT_PREFIX, "")),
  );
}

const featureLabel = (id) => app.state?.switches.find((f) => f.id === id)?.label ?? id;
const fileName = (path) => path.split("/").pop();

/** Under an answer: which features were switched off, which lessons were used. */
function msgTags(meta) {
  if (!meta) return null;
  const tags = [];
  if (meta.off?.length) tags.push(h("span", { class: "tag tag-off", title: "Fonctionnalités coupées pour cette réponse", text: `✂ sans ${meta.off.map(featureLabel).join(", ")}` }));
  if (meta.sources?.length) {
    const files = [...new Set(meta.sources.map((s) => fileName(s.source)))];
    tags.push(h("span", { class: "tag tag-rag", title: "Leçons injectées dans le prompt", text: `📄 ${files.join(", ")}` }));
  }
  return tags.length ? h("div", { class: "msg-tags" }, tags) : null;
}

function assistantMessage(text, summarized = false, meta = null) {
  return h(
    "div",
    { class: `msg msg-assistant${summarized ? " is-summarized" : ""}` },
    h("img", { class: "msg-avatar", src: MASCOT, alt: "" }),
    h("div", { class: "msg-body" }, h("div", { class: "bubble markdown", html: renderMarkdown(text) }), msgTags(meta)),
  );
}

const OPS = {
  add_word: ["mot ajouté", "mots ajoutés"],
  review_word: ["mot révisé", "mots révisés"],
  delete_word: ["mot oublié", "mots oubliés"],
  add_mistake: ["erreur notée", "erreurs notées"],
  resolve_mistake: ["erreur maîtrisée", "erreurs maîtrisées"],
  set_profile: ["info de profil", "infos de profil"],
};

function notice(text, kind = "") {
  const clean = text
    .replace(/^\[|\]$/g, "")
    .replace(/(\d+)× (\w+)/g, (all, n, op) => (OPS[op] ? `${n} ${OPS[op][n > 1 ? 1 : 0]}` : all));
  const memory = clean.startsWith("mémoire");
  append(h("div", { class: `notice${memory ? " notice-memory" : ""}${kind ? ` notice-${kind}` : ""}`, text: clean }));
}

function renderThread(state) {
  const until = state.summarized_until;
  const nodes = [];
  let dividerShown = false;
  for (const message of state.messages) {
    const summarized = message.id <= until;
    if (!summarized && until && !dividerShown) {
      nodes.push(h("div", { class: "divider", text: "Au-dessus : résumé dans la mémoire de session" }));
      dividerShown = true;
    }
    if (message.role === "user") nodes.push(userMessage(message.content, summarized));
    else if (message.role === "assistant") nodes.push(assistantMessage(message.content, summarized, message.meta));
  }
  ui.thread.replaceChildren(...(nodes.length ? nodes : [welcome()]));
  scrollToEnd(true);
}

/** An assistant bubble that fills up as tokens arrive. */
function liveAnswer() {
  const elapsed = h("span", { text: "Le tuteur réfléchit…" });
  const thinking = h("div", { class: "thinking" }, h("span", { class: "dots", "aria-hidden": "true" }, h("span"), h("span"), h("span")), elapsed);
  const body = h("div", { class: "markdown" });
  const meta = h("div", { class: "bubble-meta" });
  const bubble = h("div", { class: "bubble", "aria-busy": "true" }, thinking, body);
  const column = h("div", { class: "msg-body" }, bubble);
  const node = append(h("div", { class: "msg msg-assistant" }, h("img", { class: "msg-avatar", src: MASCOT, alt: "" }), column));
  const started = performance.now();
  const seconds = () => ((performance.now() - started) / 1000).toFixed(1);
  const timer = setInterval(() => {
    elapsed.textContent = `Le tuteur réfléchit… ${Math.round((performance.now() - started) / 1000)} s`;
  }, 1000);
  let text = "";
  let frame = 0;
  const stopThinking = () => {
    clearInterval(timer);
    thinking.remove();
    bubble.removeAttribute("aria-busy");
  };
  return {
    add(piece) {
      if (!text) {
        clearInterval(timer);
        thinking.remove();
      }
      text += piece;
      frame ||= requestAnimationFrame(() => {
        frame = 0;
        body.innerHTML = renderMarkdown(text);
        scrollToEnd();
      });
    },
    tag(meta) {
      const tags = msgTags(meta);
      if (tags) column.append(tags);
    },
    finish(interrupted, outputTokens) {
      stopThinking();
      cancelAnimationFrame(frame);
      frame = 0;
      body.innerHTML = renderMarkdown(text);
      if (!text) body.append(h("p", { class: "bubble-meta", text: "Interrompu avant la réponse." }));
      meta.textContent = [interrupted ? "Interrompu" : null, `${seconds()} s`, outputTokens ? `${outputTokens} tokens` : null]
        .filter(Boolean)
        .join(" · ");
      bubble.append(meta);
      scrollToEnd();
    },
    fail(message) {
      stopThinking();
      bubble.append(h("div", { class: "bubble-error", text: message }));
      if (!text) node.querySelector(".markdown").remove();
    },
  };
}

/* ---------- agent mode (A1, T2, T4, A3) ---------- */

function diffLine(line) {
  let kind = null;
  if (line.startsWith("@@")) kind = "hunk";
  else if (line.startsWith("+") && !line.startsWith("+++")) kind = "add";
  else if (line.startsWith("-") && !line.startsWith("---")) kind = "del";
  return h("span", { class: kind, text: `${line}\n` });
}

function confirmCard(request) {
  const refuse = h("button", { type: "button", class: "btn btn-secondary btn-small", text: "Refuser" });
  const accept = h("button", { type: "button", class: "btn btn-primary btn-small", text: "Confirmer" });
  const result = h("span", { class: "confirm-result" });
  const timer = h("span", { class: "confirm-timer" });
  const node = h(
    "li",
    { class: "confirm", role: "group", "aria-label": "Confirmation requise" },
    h("h3", { text: "⚠ Confirmation requise" }),
    h(
      "dl",
      {},
      h("dt", { text: "Action" }),
      h("dd", { text: `${request.action} — ${request.title}` }),
      request.resources.length ? [h("dt", { text: "Ressources" }), h("dd", { text: request.resources.join(", ") })] : null,
      request.reason ? [h("dt", { text: "Motif" }), h("dd", { text: request.reason })] : null,
    ),
    request.summary ? h("pre", { class: "diff" }, request.summary.split("\n").map(diffLine)) : null,
    request.risks.length ? h("ul", {}, request.risks.map((risk) => h("li", { text: risk }))) : null,
    h("div", { class: "confirm-actions" }, refuse, accept, result, timer),
  );
  const deadline = request.timeout ? Date.now() + request.timeout * 1000 : null;
  const tick = () => {
    if (!deadline) return;
    const left = Math.max(0, Math.round((deadline - Date.now()) / 1000));
    timer.textContent = `annulation dans ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
  };
  tick();
  const interval = setInterval(tick, 1000);
  const close = (text) => {
    clearInterval(interval);
    refuse.remove();
    accept.remove();
    timer.remove();
    result.textContent = text;
  };
  const answer = async (approved) => {
    refuse.disabled = accept.disabled = true;
    try {
      await api("/api/confirm", { method: "POST", body: { id: request.id, approved } });
      close(approved ? "✓ Confirmé" : "✗ Refusé");
    } catch (err) {
      close(`Trop tard : ${err.message}`);
    }
  };
  refuse.addEventListener("click", () => answer(false));
  accept.addEventListener("click", () => answer(true));
  node.expire = () => close("⏱ Sans réponse : action annulée");
  requestAnimationFrame(() => refuse.focus({ preventScroll: true })); // the safe choice gets the focus
  return node;
}

function stepCall(step) {
  if (step.action === "final_answer") return "réponse finale";
  const args = Object.entries(step.args)
    .filter(([, value]) => value !== "" && value !== null && !(Array.isArray(value) && !value.length))
    .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`);
  return `${step.action}(${args.join(", ")})`;
}

const STEP_ICONS = { final: "✓", denied: "⛔", refused: "✗", error: "!", invalid: "?" };

/** A card that lists the agent's steps and confirmation requests as they come. */
function liveTask() {
  const badge = h("span", { class: "badge", text: "en cours" });
  const steps = h("ol", { class: "steps" });
  const card = append(h("div", { class: "agent-card" }, h("p", { class: "agent-title" }, "🛠 Agent", badge), steps));
  const confirmations = new Map();
  const setBadge = (text, kind) => {
    badge.textContent = text;
    badge.className = `badge${kind ? ` badge-${kind}` : ""}`;
  };
  return {
    step(step) {
      let detail = null;
      if (step.error) detail = h("div", { class: "step-detail", text: step.error });
      else if (step.sandbox) {
        const { outcome, exit_code: code, last_line: line } = step.sandbox;
        detail = h("div", {
          class: `step-detail${outcome === "ok" ? " is-ok" : ""}`,
          text: `sandbox : ${outcome}, code ${code ?? "—"}${line ? ` — ${line}` : ""}`,
        });
      }
      steps.append(
        h(
          "li",
          { class: "step", dataset: { status: step.status } },
          h("span", { class: "step-icon", text: STEP_ICONS[step.status] ?? String(step.index) }),
          h(
            "div",
            {},
            h("div", { class: "step-summary", text: step.summary || "—" }),
            h("div", { class: "step-call", text: `${stepCall(step)} · ${(step.duration_ms / 1000).toFixed(1)} s` }),
            detail,
          ),
        ),
      );
      scrollToEnd();
    },
    confirm(request) {
      const node = confirmCard(request);
      confirmations.set(request.id, node);
      steps.append(node);
      scrollToEnd(true);
      setStatus("busy", "En attente de ta confirmation");
    },
    closed(event) {
      confirmations.get(event.id)?.expire();
    },
    done(result) {
      const labels = { done: ["terminé", "ok"], max_steps: ["limite d'étapes", "warn"], error: ["arrêté sur erreur", "bad"] };
      setBadge(...(labels[result.outcome] ?? [result.outcome, null]));
      card.append(
        h("div", { class: "agent-answer markdown", html: renderMarkdown(result.answer) }),
        h("div", { class: "bubble-meta", text: `${result.steps} étapes · ${(result.duration_ms / 1000).toFixed(1)} s` }),
      );
      scrollToEnd();
    },
    stopped() {
      setBadge("arrêté", "warn");
    },
    fail(message) {
      setBadge("erreur", "bad");
      card.append(h("div", { class: "bubble-error", text: message }));
    },
  };
}

/* ---------- comparison with and without a feature ---------- */

const SECTION_NAMES = { system: "rôle et profil", memory: "mémoire", summary: "résumé", rag: "leçons", history: "historique" };

/** Two answers side by side, each with what its prompt contained, then the judge's scores. */
function liveCompare(feature) {
  const badge = h("span", { class: "badge", text: "en cours" });
  const cols = h("div", { class: "compare-cols" });
  const footer = h("div", { class: "compare-footer" });
  const card = append(
    h("div", { class: "compare-card" }, h("p", { class: "agent-title" }, `⚖ Avec et sans : ${featureLabel(feature)}`, badge, h("span", { class: "badge", text: "non enregistré" })), cols, footer),
  );
  const sides = {};
  let current = null;
  let frame = 0;
  const paint = (side) => {
    side.body.innerHTML = renderMarkdown(side.text);
    scrollToEnd();
  };
  return {
    side(event) {
      const body = h("div", { class: "markdown" }, h("div", { class: "thinking" }, h("span", { class: "dots", "aria-hidden": "true" }, h("span"), h("span"), h("span")), "Le tuteur réfléchit…"));
      const meta = h("div", { class: "compare-meta" });
      const node = h("section", { class: "compare-col", dataset: { side: event.side } }, h("h3", { text: event.label }), body, meta);
      cols.append(node);
      current = sides[event.side] = { body, meta, node, text: "" };
      scrollToEnd(true);
    },
    token(event) {
      const side = sides[event.side];
      side.text += event.text;
      frame ||= requestAnimationFrame(() => {
        frame = 0;
        paint(side);
      });
    },
    sideDone(event) {
      cancelAnimationFrame(frame);
      frame = 0;
      const side = sides[event.side];
      paint(side);
      const ctx = event.context;
      const parts = Object.keys(SECTION_NAMES).filter((key) => ctx[key] > 0).map((key) => `${SECTION_NAMES[key]} ${ctx[key]}`);
      side.meta.append(
        h("div", { text: `${ctx.used} tokens de prompt (${parts.join(" · ")}) · ${(event.duration_ms / 1000).toFixed(1)} s` }),
        event.sources.length
          ? h("ul", { class: "compare-sources" }, event.sources.map((src) => h("li", { dataset: { key: `${src.source}#${src.part}` }, text: `📄 ${fileName(src.source)} · partie ${src.part + 1}` })))
          : h("div", { class: "muted", text: "aucune leçon dans le prompt" }),
      );
    },
    verdict(event) {
      const row = (label, result, best) =>
        h(
          "div",
          { class: `score${best ? " is-best" : ""}` },
          h("span", { class: "score-label", text: label }),
          h("span", { class: "score-bar", role: "img", "aria-label": `${result.score} sur 10` }, h("i", { style: `width:${result.score * 10}%` })),
          h("span", { class: "score-value", text: `${result.score}/10` }),
          h("span", { class: "score-reason", text: result.reason }),
        );
      footer.append(
        h(
          "div",
          { class: "verdict" },
          h("h3", { text: "Arbitrage du modèle — fidélité aux informations apportées" }),
          row(event.on_label, event.on, event.on.score > event.off.score),
          row(event.off_label, event.off, event.off.score > event.on.score),
          h("p", { class: "verdict-text", text: event.verdict }),
        ),
      );
      scrollToEnd();
    },
    notice(text) {
      footer.append(h("p", { class: "compare-notice", text }));
    },
    end(event) {
      badge.textContent = "terminé";
      badge.className = "badge badge-ok";
      const { only_on: onlyOn, only_off: onlyOff } = event.sources_diff;
      for (const [key, list] of [["on", onlyOn], ["off", onlyOff]]) {
        for (const src of list) sides[key]?.node.querySelector(`[data-key="${CSS.escape(`${src.source}#${src.part}`)}"]`)?.classList.add("is-unique");
      }
      if (onlyOn.length || onlyOff.length) footer.prepend(h("p", { class: "compare-notice", text: "Surlignées : les leçons présentes d'un seul côté." }));
    },
    stopped() {
      badge.textContent = "arrêté";
      badge.className = "badge badge-warn";
    },
    fail(message) {
      badge.textContent = "erreur";
      badge.className = "badge badge-bad";
      (current?.node ?? card).append(h("div", { class: "bubble-error", text: message }));
    },
  };
}

async function sendCompare(text) {
  const feature = ui.compareFeature.value;
  append(h("div", { class: "msg msg-user" }, h("div", { class: "bubble" }, h("span", { class: "msg-tag", text: "Comparaison" }), "\n", text)));
  const card = liveCompare(feature);
  setBusy(true, "Comparaison : première réponse…");
  try {
    await stream("/api/compare", { text, feature, judge: ui.compareJudge.checked }, {
      side: (event) => {
        card.side(event);
        setStatus("busy", event.side === "on" ? "Comparaison : première réponse…" : "Comparaison : seconde réponse…");
      },
      token: (event) => card.token(event),
      side_done: (event) => {
        card.sideDone(event);
        if (event.side === "off" && ui.compareJudge.checked) setStatus("busy", "Arbitrage…");
      },
      verdict: (event) => card.verdict(event),
      notice: (event) => card.notice(event.text),
      end: (event) => card.end(event),
      stopped: () => card.stopped(),
      error: (event) => {
        card.fail(event.message);
        toast(event.message, "error");
      },
    });
  } catch (err) {
    card.fail(err.message);
    toast(err.message, "error");
  } finally {
    setBusy(false);
    refreshMetrics().catch(() => {});
  }
}

/* ---------- sending ---------- */

async function sendChat(text) {
  append(userMessage(text));
  const answer = liveAnswer();
  setBusy(true, "Le tuteur réfléchit…");
  try {
    await stream("/api/chat", { text }, {
      notice: (event) => notice(event.text),
      token: (event) => {
        answer.add(event.text);
        setStatus("busy", "Le tuteur écrit…");
      },
      done: (event) => {
        answer.finish(event.interrupted, event.ollama?.output_tokens);
        answer.tag({ off: event.off, sources: event.sources });
        renderTurn(event);
        if (event.placement) renderDevice(event.placement);
        setStatus("busy", "Mise à jour de la mémoire…");
      },
      memory: (event) => renderMemory(event, true),
      error: (event) => {
        answer.fail(event.message);
        toast(event.message, "error");
      },
    });
  } catch (err) {
    answer.fail(err.message);
    toast(err.message, "error");
  } finally {
    setBusy(false);
    refreshAfterTurn();
  }
}

async function runAgent(task) {
  append(userMessage(`[tâche agent] ${task}`));
  const card = liveTask();
  setBusy(true, "L'agent travaille…");
  try {
    await stream("/api/agent", { task }, {
      notice: (event) => notice(event.text),
      step: (event) => {
        card.step(event);
        setStatus("busy", "L'agent travaille…");
      },
      confirm: (event) => card.confirm(event),
      confirm_closed: (event) => card.closed(event),
      done: (event) => card.done(event),
      stopped: () => card.stopped(),
      error: (event) => {
        card.fail(event.message);
        toast(event.message, "error");
      },
    });
  } catch (err) {
    card.fail(err.message);
    toast(err.message, "error");
  } finally {
    setBusy(false);
    refreshAfterTurn();
  }
}

/* ---------- side panel ---------- */

function stat(value, label, kind = "") {
  return h("div", { class: `stat${kind ? ` stat-${kind}` : ""}` }, h("span", { class: "stat-value", text: value }), h("span", { class: "stat-label", text: label }));
}

function wordItem(word) {
  const nextReview = word.due ? "à réviser maintenant" : `prochaine révision le ${formatDate(word.next_review)}`;
  return h(
    "li",
    { class: `word${word.due ? " is-due" : ""}` },
    h("span", { class: "word-term", text: word.word }),
    h(
      "span",
      { class: "word-side" },
      word.due ? h("span", { class: "badge badge-warn", text: "à réviser" }) : null,
      h("span", { class: "boxes", role: "img", "aria-label": `boîte ${word.box} sur 5, ${nextReview}`, title: `Boîte ${word.box} sur 5 — ${nextReview}` }, [1, 2, 3, 4, 5].map((i) => h("i", { class: i <= word.box ? "on" : null }))),
      h("button", { type: "button", class: "forget", title: `Oublier « ${word.word} »`, "aria-label": `Oublier ${word.word}`, text: "×", onclick: () => forgetWord(word.word) }),
    ),
    h("span", { class: "word-translation", text: `${word.translation} · ${word.correct} ✔ / ${word.wrong} ✘` }),
  );
}

function mistakeItem(mistake) {
  return h(
    "li",
    { class: "mistake" },
    h("div", { class: "mistake-pattern" }, mistake.pattern, " ", h("span", { class: "badge badge-warn", text: `×${mistake.count}` })),
    h("div", { class: "mistake-fix", text: `→ ${mistake.correction}` }),
    mistake.example ? h("div", { class: "mistake-example", text: `ex. « ${mistake.example} »` }) : null,
  );
}

function renderMemory(memory, announce = false) {
  const before = app.memory?.stats;
  app.memory = memory;
  const s = memory.stats;
  if (announce && before && s.mastered > before.mastered) toast(`🎉 Mot maîtrisé ! (${s.mastered} au total)`);
  ui.stats.replaceChildren(stat(s.words, "mots vus"), stat(s.mastered, "maîtrisés", "mastered"), stat(s.due, "à réviser", "due"), stat(s.mistakes, "erreurs suivies"));
  ui.wordsCount.textContent = memory.language ? `· ${memory.language}` : "";
  const emptyWords = memory.language ? "Aucun mot pour l'instant : discute avec le tuteur !" : "Renseigne ta langue cible dans l'onglet Profil.";
  ui.words.replaceChildren(...(memory.words.length ? memory.words.map(wordItem) : [h("li", { class: "empty", text: emptyWords })]));
  ui.mistakes.replaceChildren(...(memory.mistakes.length ? memory.mistakes.map(mistakeItem) : [h("li", { class: "empty", text: "Aucune erreur récurrente repérée." })]));
}

const SEGMENTS = [
  ["system", "Rôle et profil"],
  ["memory", "Mémoire de l'apprenant"],
  ["summary", "Résumé des anciens échanges"],
  ["rag", "Extraits de leçons"],
  ["history", "Historique récent"],
];

function legendRow(label, value, swatch, note = "") {
  return h("li", {}, h("i", { class: swatch, "aria-hidden": "true" }), label, h("span", { class: "value", text: `${value}${note}` }));
}

function renderTurn(turn) {
  const ctx = turn.context;
  if (!ctx) return;
  const total = ctx.num_ctx;
  const width = (value) => `width:${((100 * Math.max(0, value)) / total).toFixed(2)}%`;
  const free = Math.max(0, ctx.budget - ctx.used);
  const reserve = total - ctx.budget;
  ui.gauge.replaceChildren(
    ...SEGMENTS.filter(([key]) => ctx[key] > 0).map(([key, label]) => h("span", { class: `seg-${key}`, style: width(ctx[key]), title: `${label} : ${ctx[key]} tokens` })),
    h("span", { class: "seg-free", style: width(free), title: `Libre : ${free} tokens` }),
    h("span", { class: "seg-reserve", style: width(reserve), title: `Réservé à la réponse : ${reserve} tokens` }),
  );
  ui.gauge.setAttribute("aria-label", `${ctx.used} tokens utilisés sur ${ctx.budget} disponibles pour le prompt`);
  const caps = { memory: ctx.caps.memory, summary: ctx.caps.summary, rag: ctx.caps.rag };
  ui.legend.replaceChildren(
    ...SEGMENTS.map(([key, label]) => legendRow(label, ctx[key], `seg-${key}`, caps[key] ? ` / ${caps[key]}` : "")),
    legendRow("Libre", free, "seg-free"),
    legendRow("Réservé à la réponse", reserve, "seg-reserve"),
  );
  const dropped = ctx.dropped ? `, ${plural(ctx.dropped, "écarté", "écartés")}` : "";
  ui.ollama.textContent =
    `Estimation : ${ctx.used} tokens sur ${ctx.budget} (fenêtre de ${total}), ${plural(ctx.kept, "message gardé", "messages gardés")}${dropped}. ` +
    `Ollama : ${turn.ollama.prompt_tokens} tokens de prompt traités (hors cache), ${turn.ollama.output_tokens} générés.`;
  if (!app.state?.features.rag) {
    ui.sources.replaceChildren(h("li", { class: "empty", text: "RAG désactivé." }));
  } else if (!turn.sources.length) {
    ui.sources.replaceChildren(h("li", { class: "empty", text: "Aucune leçon assez pertinente pour ce message." }));
  } else {
    ui.sources.replaceChildren(
      ...turn.sources.map((s) =>
        h(
          "li",
          { class: "source" },
          h("div", { class: "source-name", text: `📄 ${s.source.split("/").pop()} · partie ${s.part + 1}` }),
          h("div", { class: "source-scores", text: `cosinus ${s.cosine.toFixed(2)} · BM25 ${s.bm25.toFixed(1)}${s.rerank === null ? "" : ` · rerank ${s.rerank}/10`}` }),
        ),
      ),
    );
  }
}

function renderSummary(state) {
  if (state.summary) ui.summary.innerHTML = renderMarkdown(state.summary.replace(/^## .*\n+/, ""));
  else ui.summary.replaceChildren(h("p", { class: "empty", text: "Pas encore de résumé : il se crée quand la conversation devient longue." }));
}

function renderProfile(state) {
  ui.learner.textContent = `· ${state.user}`;
  const entries = Object.entries(state.profile);
  ui.profile.replaceChildren(
    ...(entries.length
      ? entries.map(([key, value]) =>
          h(
            "div",
            { class: "profile-row" },
            h("dt", { text: key }),
            h("dd", {}, value, h("button", { type: "button", class: "forget", title: `Supprimer « ${key} »`, "aria-label": `Supprimer ${key}`, text: "×", onclick: () => deleteProfileKey(key) })),
          ),
        )
      : [h("div", { class: "empty", text: "Profil vide : ajoute au moins ta langue cible." })]),
  );
}

/* ---------- lab: feature switches (features.py) ---------- */

const DEMOS = [
  ["rag", "Qu'est-ce qui tombe à l'interro de vendredi ?", "Seule la leçon de la classe connaît la date, le barème et la liste de mots : sans RAG, le tuteur invente ou botte en touche."],
  ["history", "Où est-ce que je pars en voyage ?", "Dis d'abord au tuteur où tu pars (en mode Tuteur). Sans historique, il ne voit que la question."],
  ["profile", "Donne-moi un petit exercice adapté à mon niveau.", "Sans profil, le tuteur ignore ta langue cible et ton niveau."],
  ["memory", "Fais-moi réviser mes mots.", "Sans mémoire, l'exercice ne reprend pas tes mots à réviser ni tes erreurs."],
  ["rag_filters", "Comment dit-on « chat » ?", "Sans le filtre de langue, la leçon de coréen (고양이) peut entrer dans le prompt d'un apprenant de japonais."],
  ["rag_rerank", "Comment compter trois crayons ?", "Sans reclassement, des passages du même thème mais inutiles occupent le prompt."],
  ["rag_hybrid", "Que veut dire kippu ?", "Le mot rare « kippu » est trouvé par BM25 ; le cosinus seul le classe parfois moins bien."],
];

const activeSwitches = () => (app.state?.switches ?? []).filter((f) => f.available);
const cutSwitches = () => activeSwitches().filter((f) => !f.enabled && !(f.requires && !activeSwitches().find((p) => p.id === f.requires)?.enabled));

function switchRow(feature) {
  const parent = feature.requires ? app.state.switches.find((f) => f.id === feature.requires) : null;
  const inactive = !feature.available || (parent && !parent.enabled);
  const input = h("input", {
    type: "checkbox",
    role: "switch",
    checked: feature.enabled,
    disabled: !feature.available || app.busy,
    "aria-describedby": `switch-${feature.id}-does`,
    onchange: () => toggleFeature(feature.id, input.checked, input),
  });
  return h(
    "li",
    { class: `switch${feature.enabled ? "" : " is-off"}${inactive ? " is-inactive" : ""}${parent ? " is-child" : ""}` },
    h(
      "label",
      { class: "switch-head" },
      input,
      h("span", { class: "switch-track", "aria-hidden": "true" }),
      h("span", { class: "switch-label", text: feature.label }),
      h("span", { class: "badge", text: feature.code }),
    ),
    h("p", { class: "switch-does", id: `switch-${feature.id}-does`, text: !feature.available ? "Indisponible : RAG désactivé au démarrage." : feature.enabled ? feature.does : `Coupé — ${feature.when_off}` }),
  );
}

function renderSwitches(list) {
  app.state.switches = list;
  const groups = [...new Set(list.map((f) => f.group))];
  ui.switches.replaceChildren(
    ...groups.map((group) => h("div", { class: "switch-group" }, h("h3", { text: group }), h("ul", { class: "switch-list" }, list.filter((f) => f.group === group).map(switchRow)))),
  );
  const cut = cutSwitches();
  ui.banner.hidden = !cut.length;
  ui.bannerText.textContent = `✂ Coupé : ${cut.map((f) => f.label).join(", ")}`;
  const selected = ui.compareFeature.value;
  ui.compareFeature.replaceChildren(...activeSwitches().map((f) => h("option", { value: f.id, text: `${f.label} (${f.code})`, selected: f.id === selected })));
  ui.demos.replaceChildren(
    ...DEMOS.filter(([id]) => activeSwitches().some((f) => f.id === id)).map(([id, question, why]) =>
      h(
        "li",
        { class: "demo" },
        h("div", { class: "demo-head" }, h("span", { class: "badge", text: featureLabel(id) }), h("button", { type: "button", class: "btn btn-secondary btn-small", text: "Essayer", disabled: app.busy, onclick: () => tryDemo(id, question) })),
        h("div", { class: "demo-question", text: `« ${question} »` }),
        h("div", { class: "demo-why", text: why }),
      ),
    ),
  );
}

async function toggleFeature(id, enabled, input) {
  try {
    const { features } = await api(`/api/features/${encodeURIComponent(id)}`, { method: "PUT", body: { enabled } });
    renderSwitches(features);
    toast(`${featureLabel(id)} : ${enabled ? "rétabli" : "coupé"}`);
  } catch (err) {
    input.checked = !enabled;
    toast(err.message, "error");
  }
}

async function resetSwitches() {
  try {
    renderSwitches((await api("/api/features/reset", { method: "POST" })).features);
    toast("Réglages de config.json rétablis");
  } catch (err) {
    toast(err.message, "error");
  }
}

function tryDemo(id, question) {
  setMode("compare");
  ui.compareFeature.value = id;
  ui.input.value = question;
  autosize();
  if (matchMedia("(max-width: 960px)").matches) openSidebar(false);
  ui.input.focus();
}

/* ---------- learners and their history (M1) ---------- */

function renderUsers(users) {
  ui.user.replaceChildren(
    ...users.map((u) => h("option", { value: u.name, text: u.name, selected: u.current })),
    h("option", { value: "", text: "＋ Nouvel apprenant…" }),
  );
  ui.users.replaceChildren(
    ...users.map((u) =>
      h(
        "li",
        { class: `user${u.current ? " is-current" : ""}` },
        h(
          "div",
          { class: "session-text" },
          h("div", { class: "session-title", text: u.name }),
          h("div", {
            class: "session-meta",
            text: [u.language || "langue non précisée", plural(u.words, "mot", "mots"), plural(u.sessions, "session", "sessions"), u.last_active ? `vu le ${formatDate(u.last_active)}` : "jamais vu"].join(" · "),
          }),
        ),
        u.current
          ? h("span", { class: "badge badge-ok", text: "actif" })
          : [
              h("button", { type: "button", class: "btn btn-secondary btn-small", text: "Choisir", onclick: () => selectUser(u.name) }),
              h("button", { type: "button", class: "forget", title: `Supprimer « ${u.name} »`, "aria-label": `Supprimer ${u.name}`, text: "×", onclick: () => deleteUser(u.name) }),
            ],
      ),
    ),
  );
}

async function refreshUsers() {
  renderUsers((await api("/api/users")).users);
}

/** After a learner change: everything on screen belongs to the new learner. */
async function showLearner(state, message) {
  applyState(state);
  await Promise.all([refreshUsers(), refreshMemory(), refreshHistory()]);
  ui.legend.replaceChildren(h("li", { class: "empty", text: "Envoie un message pour voir la répartition." }));
  ui.gauge.replaceChildren();
  ui.sources.replaceChildren(h("li", { class: "empty", text: "Les extraits de leçons utilisés apparaîtront ici." }));
  toast(message);
}

async function selectUser(name) {
  try {
    await showLearner(await api(`/api/users/${encodeURIComponent(name)}/select`, { method: "POST" }), `Bon retour, ${name} !`);
  } catch (err) {
    toast(err.message, "error");
    refreshUsers().catch(() => {});
  }
}

async function createUser(name) {
  try {
    await showLearner(await api("/api/users", { method: "POST", body: { name } }), `Profil « ${name} » créé : renseigne sa langue cible.`);
    selectTab($("#tab-profile"));
    return true;
  } catch (err) {
    toast(err.message, "error");
    refreshUsers().catch(() => {});
    return false;
  }
}

async function deleteUser(name) {
  if (!window.confirm(`Supprimer « ${name} » ? Ses sessions, son profil, son vocabulaire et ses erreurs seront effacés.`)) return;
  try {
    renderUsers((await api(`/api/users/${encodeURIComponent(name)}`, { method: "DELETE" })).users);
    toast(`« ${name} » supprimé`);
  } catch (err) {
    toast(err.message, "error");
  }
}

const DAY_NAMES = ["D", "L", "M", "M", "J", "V", "S"];

function renderActivity(days) {
  const max = Math.max(1, ...days.map((d) => d.messages));
  ui.activity.replaceChildren(
    ...days.map((d) => {
      const date = new Date(`${d.day}T12:00:00`);
      const label = `${date.toLocaleDateString("fr-FR", { weekday: "long", day: "numeric", month: "long" })} : ${plural(d.messages, "message", "messages")}`;
      return h(
        "div",
        { class: "activity-day", role: "listitem", "aria-label": label, title: label, tabindex: "0" },
        h("span", { class: "activity-bar" }, h("i", { class: d.messages ? null : "is-zero", style: `height:${d.messages ? Math.max(6, (100 * d.messages) / max) : 0}%` })),
        h("span", { class: "activity-label", text: DAY_NAMES[date.getDay()] }),
      );
    }),
  );
}

function renderHistory(history) {
  ui.historyUser.textContent = `· ${history.user}`;
  ui.historyStats.replaceChildren(
    stat(`🔥 ${history.streak}`, history.streak > 1 ? "jours d'affilée" : "jour d'affilée", "due"),
    stat(history.active_days, "jours actifs"),
    stat(history.messages, "messages envoyés"),
    stat(history.mastered, `mots maîtrisés / ${history.words}`, "mastered"),
  );
  renderActivity(history.days);
  ui.sessions.replaceChildren(
    ...(history.sessions.length
      ? history.sessions.map((s) =>
          h(
            "li",
            { class: `session${s.current ? " is-current" : ""}` },
            h(
              "div",
              { class: "session-text" },
              h("div", { class: "session-title", text: s.title || "(vide)" }),
              h("div", { class: "session-meta", text: `#${s.id} · ${formatDate(s.updated_at)} · ${plural(s.messages, "message", "messages")}` }),
            ),
            s.messages ? h("button", { type: "button", class: "btn btn-secondary btn-small", text: "Lire", onclick: () => openPreview(s) }) : null,
            s.current
              ? h("span", { class: "badge badge-ok", text: "active" })
              : h("button", { type: "button", class: "btn btn-secondary btn-small", text: "Reprendre", onclick: () => loadSession(s.id) }),
          ),
        )
      : [h("li", { class: "empty", text: "Aucune session pour l'instant." })]),
  );
}

async function refreshHistory() {
  renderHistory(await api("/api/history"));
}

async function openPreview(session) {
  try {
    const { messages } = await api(`/api/sessions/${session.id}`);
    ui.previewTitle.textContent = `#${session.id} · ${session.title || "(vide)"} · ${formatDate(session.updated_at)}`;
    ui.previewThread.replaceChildren(...messages.map((m) => (m.role === "user" ? userMessage(m.content) : assistantMessage(m.content, false, m.meta))));
    ui.previewResume.hidden = session.current;
    ui.previewResume.onclick = () => {
      ui.preview.close();
      loadSession(session.id);
    };
    ui.preview.showModal();
  } catch (err) {
    toast(err.message, "error");
  }
}

async function refreshMetrics() {
  const m = await api("/api/metrics");
  const calls = m.model_calls;
  const confirmations = Object.entries(m.confirmations).map(([decision, n]) => `${decision} ${n}`).join(", ");
  const rows = [
    ["Appels au modèle", calls.count],
    ["Latence moyenne", `${(calls.avg_ms / 1000).toFixed(1)} s`],
    ["Latence p95", `${(calls.p95_ms / 1000).toFixed(1)} s`],
    ["Tokens prompt / sortie", `${calls.prompt_tokens} / ${calls.output_tokens}`],
    ["Erreurs", `${m.errors} (${(m.error_rate * 100).toFixed(1)} %)`],
    ["Tâches de l'agent", m.react.runs],
    ["Confirmations", confirmations || "—"],
  ];
  ui.metrics.replaceChildren(...rows.flatMap(([label, value]) => [h("dt", { text: label }), h("dd", { text: String(value) })]));
}

function applyState(state) {
  app.state = state;
  renderThread(state);
  renderProfile(state);
  renderSummary(state);
  renderSwitches(state.switches);
  if (state.last_turn) renderTurn(state.last_turn);
  const agentTab = document.querySelector('.mode[data-mode="agent"]');
  agentTab.disabled = !state.features.agent;
  agentTab.title = state.features.agent ? "" : "Mode agent indisponible";
  if (!state.features.agent && app.mode === "agent") setMode("chat");
}

async function refreshAfterTurn() {
  try {
    const [state, history, users] = await Promise.all([api("/api/state"), api("/api/history"), api("/api/users")]);
    app.state = state;
    renderProfile(state);
    renderSummary(state);
    renderSwitches(state.switches);
    renderHistory(history);
    renderUsers(users.users);
    await refreshMetrics();
  } catch (err) {
    toast(err.message, "error");
  }
}

async function refreshMemory() {
  renderMemory(await api("/api/memory"));
}

/* ---------- actions ---------- */

async function forgetWord(word) {
  if (!window.confirm(`Oublier « ${word} » ? Le mot sera retiré de ta révision.`)) return;
  try {
    renderMemory(await api(`/api/vocab/${encodeURIComponent(word)}`, { method: "DELETE" }));
    toast(`« ${word} » oublié`);
  } catch (err) {
    toast(err.message, "error");
  }
}

async function deleteProfileKey(key) {
  try {
    const { profile } = await api(`/api/profile/${encodeURIComponent(key)}`, { method: "DELETE" });
    app.state.profile = profile;
    renderProfile(app.state);
    await refreshMemory();
  } catch (err) {
    toast(err.message, "error");
  }
}

async function loadSession(id) {
  try {
    applyState(await api(`/api/sessions/${id}/load`, { method: "POST" }));
    await refreshHistory();
    toast(`Session #${id} reprise`);
  } catch (err) {
    toast(err.message, "error");
  }
}

function setMode(mode) {
  app.mode = mode;
  for (const button of document.querySelectorAll(".mode")) {
    const on = button.dataset.mode === mode;
    button.classList.toggle("is-active", on);
    button.setAttribute("aria-checked", String(on));
  }
  ui.input.placeholder = { chat: "Écris à ton tuteur…", agent: "Décris une tâche pour l'agent…", compare: "Une question, posée avec et sans…" }[mode];
  ui.compareBar.hidden = mode !== "compare";
  if (ui.thread.querySelector(".welcome")) ui.thread.replaceChildren(welcome());
  ui.input.focus();
}

function selectTab(tab) {
  for (const t of document.querySelectorAll('[role="tab"]')) {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
  }
  if (tab.id === "tab-context") refreshMetrics().catch(() => {});
}

function openSidebar(open) {
  ui.sidebar.classList.toggle("is-open", open);
  ui.toggle.setAttribute("aria-expanded", String(open));
  if (open) ui.sidebar.querySelector('[aria-selected="true"]').focus();
}

function autosize() {
  ui.input.style.height = "auto";
  ui.input.style.height = `${Math.min(ui.input.scrollHeight, 200)}px`;
}

async function loadModels() {
  try {
    const { current, models } = await api("/api/models");
    const names = models.includes(current) ? models : [current, ...models];
    ui.model.replaceChildren(...names.map((name) => h("option", { value: name, text: name, selected: name === current })));
  } catch {
    ui.model.replaceChildren(h("option", { text: app.state?.model ?? "?" }));
  }
}

function bind() {
  ui.composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = ui.input.value.trim();
    if (!text || app.busy) return;
    ui.input.value = "";
    autosize();
    await { chat: sendChat, agent: runAgent, compare: sendCompare }[app.mode](text);
    ui.input.focus();
  });
  ui.input.addEventListener("input", autosize);
  ui.input.addEventListener("keydown", (event) => {
    // isComposing: Enter also validates a kana/kanji conversion, it must not send the message.
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ui.composer.requestSubmit();
    }
  });
  ui.stop.addEventListener("click", async () => {
    ui.stop.disabled = true;
    setStatus("busy", "Arrêt en cours…");
    try {
      await api("/api/stop", { method: "POST" });
    } catch (err) {
      toast(err.message, "error");
    }
  });

  for (const button of document.querySelectorAll(".mode")) button.addEventListener("click", () => setMode(button.dataset.mode));

  const tabs = [...document.querySelectorAll('[role="tab"]')];
  for (const tab of tabs) {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      const next = tabs[(tabs.indexOf(tab) + step + tabs.length) % tabs.length];
      selectTab(next);
      next.focus();
    });
  }

  ui.toggle.addEventListener("click", () => openSidebar(!ui.sidebar.classList.contains("is-open")));
  ui.close.addEventListener("click", () => {
    openSidebar(false);
    ui.toggle.focus();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !ui.preview.open && ui.sidebar.classList.contains("is-open")) {
      openSidebar(false);
      ui.toggle.focus();
    }
  });

  ui.model.addEventListener("change", async () => {
    const previous = app.state.model;
    ui.model.disabled = true;
    setStatus("busy", "Chargement du modèle…");
    try {
      const { model, placement } = await api("/api/model", { method: "PUT", body: { model: ui.model.value } });
      app.state.model = model;
      renderDevice(placement);
      toast(`Modèle : ${model}${placement ? ` (${placement.label})` : ""}`);
    } catch (err) {
      ui.model.value = previous;
      toast(err.message, "error");
    } finally {
      ui.model.disabled = false;
      setStatus("idle", "Prêt");
    }
  });

  ui.compress.addEventListener("click", async () => {
    setBusy(true, "Résumé des anciens échanges…");
    try {
      const result = await api("/api/compress", { method: "POST" });
      applyState(result.state);
      result.notices.forEach((text) => notice(text));
      if (!result.compressed) toast("Rien à résumer : il faut au moins deux messages avant le dernier.");
    } catch (err) {
      toast(err.message, "error");
    } finally {
      setBusy(false);
    }
  });

  ui.newSession.addEventListener("click", async () => {
    try {
      const state = await api("/api/sessions", { method: "POST" });
      applyState(state);
      await refreshHistory();
      toast(`Nouvelle session #${state.session}`);
    } catch (err) {
      toast(err.message, "error");
    }
  });

  ui.user.addEventListener("change", async () => {
    const name = ui.user.value;
    if (name) return selectUser(name);
    const fresh = window.prompt("Nom du nouvel apprenant (lettres, chiffres, - ou _) :")?.trim();
    if (!fresh || !(await createUser(fresh))) refreshUsers().catch(() => {});
  });
  ui.userForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (await createUser(ui.userName.value.trim())) ui.userForm.reset();
  });
  ui.switchesReset.addEventListener("click", resetSwitches);
  ui.bannerReset.addEventListener("click", resetSwitches);
  ui.previewClose.addEventListener("click", () => ui.preview.close());
  ui.preview.addEventListener("click", (event) => {
    if (event.target === ui.preview) ui.preview.close(); // click on the backdrop
  });

  ui.profileForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const { profile } = await api("/api/profile", { method: "PUT", body: { key: ui.profileKey.value, value: ui.profileValue.value } });
      app.state.profile = profile;
      renderProfile(app.state);
      ui.profileForm.reset();
      toast("Profil mis à jour");
      await refreshMemory();
    } catch (err) {
      toast(err.message, "error");
    }
  });
}

async function init() {
  bind();
  ui.legend.replaceChildren(h("li", { class: "empty", text: "Envoie un message pour voir la répartition." }));
  ui.sources.replaceChildren(h("li", { class: "empty", text: "Les extraits de leçons utilisés apparaîtront ici." }));
  ui.metrics.replaceChildren(h("dt", { text: "Aucun appel au modèle pour l'instant." }));
  try {
    const [state, memory, history, users] = await Promise.all([api("/api/state"), api("/api/memory"), api("/api/history"), api("/api/users")]);
    applyState(state);
    renderDevice(state.placement);
    renderMemory(memory);
    renderHistory(history);
    renderUsers(users.users);
    setStatus("idle", "Prêt");
  } catch (err) {
    setStatus("error", "Serveur injoignable");
    toast(err.message, "error");
    return;
  }
  loadModels();
  refreshMetrics().catch(() => {});
  const tab = { "#progres": "#tab-progress", "#contexte": "#tab-context", "#labo": "#tab-lab", "#profil": "#tab-profile" }[location.hash];
  if (tab) {
    selectTab($(tab));
    if (matchMedia("(max-width: 960px)").matches) openSidebar(true);
  }
  ui.input.focus();
}

init();

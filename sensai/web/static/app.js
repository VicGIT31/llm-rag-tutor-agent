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
  for (const control of document.querySelectorAll(".mode, #model, #compress, #new-session, .session .btn, .forget")) {
    control.disabled = busy || (control.dataset.mode === "agent" && !app.state?.features.agent);
  }
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
};

function welcome() {
  const agent = app.mode === "agent";
  return h(
    "div",
    { class: "welcome" },
    h("img", { src: MASCOT, alt: "" }),
    h("h1", { text: agent ? "Que dois-je préparer ?" : "Konnichiwa !" }),
    h("p", {
      text: agent
        ? "Confie une tâche à l'agent : il lit tes leçons, écrit et teste ses programmes, et te demande toujours avant de modifier un fichier."
        : "Je suis ton tuteur de langues. Je retiens tes mots et tes erreurs, et je m'appuie sur tes leçons.",
    }),
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

function assistantMessage(text, summarized = false) {
  return h(
    "div",
    { class: `msg msg-assistant${summarized ? " is-summarized" : ""}` },
    h("img", { class: "msg-avatar", src: MASCOT, alt: "" }),
    h("div", { class: "bubble markdown", html: renderMarkdown(text) }),
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
    else if (message.role === "assistant") nodes.push(assistantMessage(message.content, summarized));
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
  const node = append(h("div", { class: "msg msg-assistant" }, h("img", { class: "msg-avatar", src: MASCOT, alt: "" }), bubble));
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

function renderSessions(sessions) {
  ui.sessions.replaceChildren(
    ...sessions.map((s) =>
      h(
        "li",
        { class: `session${s.current ? " is-current" : ""}` },
        h(
          "div",
          { class: "session-text" },
          h("div", { class: "session-title", text: s.title || "(vide)" }),
          h("div", { class: "session-meta", text: `#${s.id} · ${formatDate(s.updated_at)} · ${plural(s.messages, "message", "messages")}` }),
        ),
        s.current
          ? h("span", { class: "badge badge-ok", text: "active" })
          : h("button", { type: "button", class: "btn btn-secondary btn-small", text: "Reprendre", onclick: () => loadSession(s.id) }),
      ),
    ),
  );
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
  if (state.last_turn) renderTurn(state.last_turn);
  const agentTab = document.querySelector('.mode[data-mode="agent"]');
  agentTab.disabled = !state.features.agent;
  agentTab.title = state.features.agent ? "" : "Mode agent indisponible";
  if (!state.features.agent && app.mode === "agent") setMode("chat");
}

async function refreshAfterTurn() {
  try {
    const [state, sessions] = await Promise.all([api("/api/state"), api("/api/sessions")]);
    app.state = state;
    renderProfile(state);
    renderSummary(state);
    renderSessions(sessions.sessions);
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
    renderSessions((await api("/api/sessions")).sessions);
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
  ui.input.placeholder = mode === "agent" ? "Décris une tâche pour l'agent…" : "Écris à ton tuteur…";
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
    await (app.mode === "agent" ? runAgent(text) : sendChat(text));
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
    if (event.key === "Escape" && ui.sidebar.classList.contains("is-open")) {
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
      renderSessions((await api("/api/sessions")).sessions);
      toast(`Nouvelle session #${state.session}`);
    } catch (err) {
      toast(err.message, "error");
    }
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
    const [state, memory, sessions] = await Promise.all([api("/api/state"), api("/api/memory"), api("/api/sessions")]);
    applyState(state);
    renderDevice(state.placement);
    renderMemory(memory);
    renderSessions(sessions.sessions);
    setStatus("idle", "Prêt");
  } catch (err) {
    setStatus("error", "Serveur injoignable");
    toast(err.message, "error");
    return;
  }
  loadModels();
  refreshMetrics().catch(() => {});
  const tab = { "#progres": "#tab-progress", "#contexte": "#tab-context", "#profil": "#tab-profile" }[location.hash];
  if (tab) {
    selectTab($(tab));
    if (matchMedia("(max-width: 960px)").matches) openSidebar(true);
  }
  ui.input.focus();
}

init();

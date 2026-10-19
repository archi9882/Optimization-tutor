// Prediction questions ("預測再看動畫") above the lab.
// The answer of each question was decided at build time by running the
// scenario and checking each option's predicate (scripts/build.py), so the
// reveal needs no Python; the lab only plays the embedded traces.

import { warmUp } from "./py.js";

// Separate key: progress.js (ot.progress.v1) has no public hook for this yet.
export const KEY = "ot.predict.v1";

/** {v, answers: {questionId: {tries, right, last, ts}}}; empty when storage fails. */
export function loadPredictions() {
  try {
    const obj = JSON.parse(localStorage.getItem(KEY) || "null");
    if (obj && obj.v === 1 && obj.answers && typeof obj.answers === "object") return obj;
  } catch (e) { /* unavailable or corrupt: start empty */ }
  return { v: 1, answers: {} };
}

/** Record one prediction; returns false when storage is unavailable. */
export function recordPrediction(id, choice, correct, ts = Date.now()) {
  const state = loadPredictions();
  const a = state.answers[id] || { tries: 0, right: 0 };
  a.tries += 1;
  a.right += correct ? 1 : 0;
  a.last = { choice, correct };
  a.ts = ts;
  state.answers[id] = a;
  try { localStorage.setItem(KEY, JSON.stringify(state)); return true; } catch (e) { return false; }
}

const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  el.append(...kids);
  return el;
};

/**
 * Mount the question card before the lab.
 * lab = {applyScenario(sc) -> Promise, playThrough() -> Promise<finished>, pause(), play()}
 */
export function mountPredict(mount, questions, lab) {
  let qi = 0;
  let busy = false;
  const uid = `pq-${Math.random().toString(36).slice(2, 8)}`;
  const counter = h("span", { class: "predict-count" });
  const skipBtn = h("button", { type: "button", class: "link-btn predict-skip", onclick: () => skip() }, "直接看動畫");
  const head = h("div", { class: "predict-head" },
    h("span", { class: "predict-kicker", id: `${uid}-title` }, "先預測，再看動畫"), counter, skipBtn);
  const legend = h("legend", { class: "predict-prompt" });
  const choices = h("div", { class: "predict-options" });
  const fieldset = h("fieldset", { class: "predict-q" }, legend, choices);
  const goBtn = h("button", { type: "button", class: "btn btn-primary", disabled: "", onclick: () => submit() }, "確定預測並播放");
  const status = h("span", { class: "status-line predict-status", role: "status" });
  const revealNow = h("button", { type: "button", class: "link-btn", hidden: "", onclick: () => lab.pause() }, "略過動畫，直接揭曉");
  const actions = h("div", { class: "widget-row predict-actions" }, goBtn, revealNow, status);
  const reveal = h("div", { class: "predict-reveal", tabindex: "-1", hidden: "" });
  const body = h("div", { class: "predict-body" }, fieldset, actions, reveal);
  const reopen = h("button", { type: "button", class: "link-btn", hidden: "", onclick: () => resume() }, "回到預測題");
  const card = h("section", { class: "widget predict", "aria-labelledby": `${uid}-title` }, head, body, reopen);
  mount.before(card);

  // Preload Python while the learner reads and chooses (lab.js also warms up).
  const warm = () => warmUp().catch(() => {});
  card.addEventListener("focusin", warm, { once: true });
  card.addEventListener("pointerdown", warm, { once: true });

  // Blur the answer away and keep it from screen readers and the pointer;
  // the lab's controls stay usable.
  const hideLab = (on) => {
    mount.classList.toggle("predict-hidden", on);
    for (const el of mount.querySelectorAll(".lab-plots, .algo-result, .step-note")) el.inert = on;
  };

  function render() {
    const q = questions[qi];
    busy = false;
    counter.textContent = questions.length > 1 ? `第 ${qi + 1}／${questions.length} 題` : "";
    legend.textContent = q.prompt;
    choices.replaceChildren(...q.options.map((o, i) => {
      const input = h("input", { type: "radio", name: uid, value: String(i) });
      input.addEventListener("change", () => { goBtn.disabled = false; });
      return h("label", { class: "predict-option" }, input, h("span", {}, o.label));
    }));
    fieldset.disabled = false;
    actions.hidden = false;
    goBtn.hidden = false;
    goBtn.disabled = true;
    revealNow.hidden = true;
    status.textContent = "";
    reveal.hidden = true;
    reveal.replaceChildren();
    hideLab(true);
  }

  const chosen = () => {
    const el = choices.querySelector("input:checked");
    return el ? Number(el.value) : -1;
  };

  async function submit() {
    const choice = chosen();
    if (choice < 0 || busy) return;
    busy = true;
    const q = questions[qi];
    recordPrediction(q.id, choice, choice === q.answer);
    fieldset.disabled = true;
    goBtn.hidden = true;
    let note = "";
    try {
      if (!q.scenario.same_grid) status.textContent = "等待 Python 載入，以切換到這個問題設定…";
      await lab.applyScenario(q.scenario);
      hideLab(false);
      status.textContent = "播放中，播完揭曉結果";
      revealNow.hidden = false;
      revealNow.focus({ preventScroll: true });
      const plots = mount.querySelector(".lab-plots");
      if (plots) {
        const r = plots.getBoundingClientRect();
        if (r.top < 0 || r.bottom > innerHeight) {
          const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
          plots.scrollIntoView({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
        }
      }
      await lab.playThrough();
    } catch (err) {
      hideLab(false);
      note = "Python 無法載入，動畫沒有切換到這個設定；下面的結果是建置時實際執行得出的。";
    }
    showReveal(q, choice, note);
  }

  function showReveal(q, choice, note) {
    busy = false;
    actions.hidden = true;
    revealNow.hidden = true;
    status.textContent = "";
    const right = choice === q.answer;
    const next = qi + 1 < questions.length;
    reveal.replaceChildren(
      h("p", { class: `predict-verdict ${right ? "good" : "bad"}` },
        h("strong", {}, right ? "答對了" : "沒猜中"), "：你預測「", q.options[choice].label,
        "」→ 實際「", q.options[q.answer].label, "」"),
      h("p", { class: "predict-explain" }, q.explain),
      h("p", { class: "hint-text predict-measured" },
        `實測（${q.subject_name}）：${q.summary}。判定依據：${q.options[q.answer].criterion}。`),
      ...(note ? [h("p", { class: "hint-text" }, note)] : []),
      h("div", { class: "widget-row" },
        h("button", { type: "button", class: "btn btn-small", onclick: () => lab.play() }, "重看動畫"),
        next ? h("button", { type: "button", class: "link-btn predict-next", onclick: () => { qi++; render(); focusPrompt(); } }, "下一題 →")
          : h("button", { type: "button", class: "link-btn", onclick: () => { qi = 0; render(); focusPrompt(); } }, "從第 1 題重來")));
    reveal.hidden = false;
    reveal.focus();
  }

  function focusPrompt() {
    const first = choices.querySelector("input");
    if (first) first.focus();
  }

  function skip() {
    if (busy) lab.pause();
    body.hidden = true;
    counter.hidden = true;
    skipBtn.hidden = true;
    reopen.hidden = false;
    hideLab(false);
    reopen.focus();
    lab.play();
  }

  function resume() {
    body.hidden = false;
    counter.hidden = false;
    skipBtn.hidden = false;
    reopen.hidden = true;
    render();
    focusPrompt();
  }

  render();
}

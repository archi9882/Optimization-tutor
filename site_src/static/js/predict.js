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
 * lab = {applyScenario(sc) -> Promise (rejects with err.superseded when a later
 *        edit or cancelScenario() wins), cancelScenario(),
 *        playThrough() -> Promise<finished>, pause(), play(), showEnd()}
 */
export function mountPredict(mount, questions, lab) {
  let qi = 0;
  let busy = false;
  let attempt = 0;            // bumped by skip / question change: stale awaits stop
  const uid = `pq-${Math.random().toString(36).slice(2, 8)}`;
  const nav = h("div", { class: "predict-nav", role: "group", "aria-label": "預測題" });
  const skipBtn = h("button", { type: "button", class: "link-btn predict-skip", onclick: () => skip() }, "直接看動畫");
  const head = h("div", { class: "predict-head" },
    h("span", { class: "predict-kicker", id: `${uid}-title` }, "先預測，再看動畫"), nav, skipBtn);
  const legend = h("legend", { class: "predict-prompt" });
  const choices = h("div", { class: "predict-options" });
  const fieldset = h("fieldset", { class: "predict-q" }, legend, choices);
  const prev = h("p", { class: "hint-text predict-prev", hidden: "" });
  const goBtn = h("button", { type: "button", class: "btn btn-primary", disabled: "", onclick: () => submit(chosen(), true) }, "確定預測並播放");
  const status = h("span", { class: "status-line predict-status", role: "status" });
  const revealNow = h("button", { type: "button", class: "link-btn", hidden: "", onclick: () => lab.showEnd() }, "略過動畫，直接揭曉");
  const actions = h("div", { class: "widget-row predict-actions" }, goBtn, revealNow, status);
  const reveal = h("div", { class: "predict-reveal", tabindex: "-1", hidden: "" });
  const body = h("div", { class: "predict-body" }, fieldset, prev, actions, reveal);
  const reopen = h("button", { type: "button", class: "link-btn", hidden: "", onclick: () => resume() }, "回到預測題");
  const card = h("section", { class: "widget predict", "aria-labelledby": `${uid}-title` }, head, body, reopen);
  mount.before(card);
  const labWidget = mount.querySelector(".lab");
  const plotsEl = mount.querySelector(".lab-plots");
  if (plotsEl) plotsEl.dataset.cover = "先預測，再看結果";

  // Preload Python while the learner reads and chooses (lab.js also warms up).
  const warm = () => warmUp().catch(() => {});
  card.addEventListener("focusin", warm, { once: true });
  card.addEventListener("pointerdown", warm, { once: true });

  // While a question is pending the plots and result lines are covered by a
  // solid panel and the whole lab is dimmed and inert (its default values
  // would only confuse); "直接看動畫" gives it back.
  const hideLab = (on) => {
    mount.classList.toggle("predict-hidden", on);
    if (labWidget) labWidget.inert = on;
  };

  const last = (q) => loadPredictions().answers[q.id]?.last;

  function renderNav() {
    nav.replaceChildren(...(questions.length > 1 ? questions.map((q, i) => {
      const l = last(q);
      const mark = !l ? "" : l.correct ? " ✓" : " ✗";
      const desc = !l ? "尚未作答" : l.correct ? "上次答對" : "上次沒猜中";
      const b = h("button", { type: "button", class: `predict-pill${l ? (l.correct ? " good" : " bad") : ""}`,
        "aria-label": `第 ${i + 1} 題，${desc}`, onclick: () => go(i) }, `${i + 1}${mark}`);
      if (i === qi) b.setAttribute("aria-current", "step");
      return b;
    }) : []));
  }

  function go(i) {
    if (busy) { lab.cancelScenario(); lab.pause(); }
    attempt++;
    qi = i;
    render();
    focusPrompt();
  }

  function render() {
    const q = questions[qi];
    busy = false;
    renderNav();
    legend.textContent = q.prompt;
    choices.replaceChildren(...q.options.map((o, i) => {
      const input = h("input", { type: "radio", name: uid, value: String(i) });
      input.addEventListener("change", () => { goBtn.disabled = false; });
      return h("label", { class: "predict-option" }, input, h("span", {}, o.label));
    }));
    const l = last(q);
    prev.hidden = !l;
    if (l) prev.replaceChildren(`你上次預測「${q.options[l.choice]?.label ?? "?"}」，${l.correct ? "答對了" : "沒猜中"}。可以重新作答，或 `,
      h("button", { type: "button", class: "link-btn", onclick: () => submit(l.choice, false) }, "看上次的結果"));
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

  function scrollToPlots() {
    const plots = mount.querySelector(".lab-plots");
    if (!plots) return;
    const r = plots.getBoundingClientRect();
    if (r.top < 0 || r.bottom > innerHeight) {
      const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
      plots.scrollIntoView({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
    }
  }

  // record = false replays the learner's previous prediction without logging it again.
  async function submit(choice, record) {
    if (choice < 0 || busy) return;
    busy = true;
    const token = ++attempt;
    const q = questions[qi];
    if (record) { recordPrediction(q.id, choice, choice === q.answer); renderNav(); }
    for (const el of choices.querySelectorAll("input")) el.checked = Number(el.value) === choice;
    fieldset.disabled = true;
    goBtn.hidden = true;
    prev.hidden = true;
    let note = "";
    try {
      if (!q.scenario.same_grid) status.textContent = "等待 Python 載入，以切換到這個問題設定…";
      await lab.applyScenario(q.scenario);
      if (token !== attempt) return;
      hideLab(false);
      status.textContent = "播放中，播完揭曉結果";
      revealNow.hidden = false;
      revealNow.focus({ preventScroll: true });
      scrollToPlots();
      await lab.playThrough();
    } catch (err) {
      if (token !== attempt) return;          // skipped or moved to another question
      hideLab(false);
      note = err && err.superseded
        ? "你在載入期間調整了動畫，所以它沒有切換到題目設定；按「重看動畫」可載入題目設定。"
        : "Python 無法載入，動畫沒有切換到這個設定；下面的結果是建置時實際執行得出的。";
    }
    if (token !== attempt) return;
    showReveal(q, choice, note);
  }

  async function replay(q) {
    const token = ++attempt;
    try {
      await lab.applyScenario(q.scenario);
      if (token === attempt) { scrollToPlots(); lab.play(); }
    } catch (err) {
      if (token === attempt && !(err && err.superseded)) status.textContent = "Python 無法載入，無法切換到題目設定。";
    }
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
      h("p", { class: "predict-explain" },
        h("span", { class: "predict-lead" }, right ? "為什麼：" : "看看實際發生了什麼："), q.explain),
      h("p", { class: "hint-text predict-measured" },
        `實測（${q.subject_name}）：${q.summary}。判定依據：${q.options[q.answer].criterion}。`),
      ...(note ? [h("p", { class: "hint-text" }, note)] : []),
      h("div", { class: "widget-row" },
        h("button", { type: "button", class: "btn btn-small", onclick: () => replay(q) }, "重看動畫"),
        h("button", { type: "button", class: "link-btn", onclick: () => go(qi) }, "重新作答"),
        next ? h("button", { type: "button", class: "link-btn predict-next", onclick: () => go(qi + 1) }, "下一題 →")
          : h("button", { type: "button", class: "link-btn", onclick: () => go(0) }, "從第 1 題重來")));
    reveal.hidden = false;
    reveal.focus();
  }

  function focusPrompt() {
    const first = choices.querySelector("input");
    if (first) first.focus();
  }

  function skip() {
    attempt++;                 // a pending submit stops after its await
    lab.cancelScenario();      // ...and a scenario waiting for Python is dropped
    if (busy) lab.pause();
    busy = false;
    body.hidden = true;
    nav.hidden = true;
    skipBtn.hidden = true;
    reopen.hidden = false;
    hideLab(false);
    reopen.focus();
    lab.play();
  }

  function resume() {
    body.hidden = false;
    nav.hidden = false;
    skipBtn.hidden = false;
    reopen.hidden = true;
    render();
    focusPrompt();
  }

  // Start at the first question not yet answered (all answered: the first).
  const open = questions.findIndex((q) => !last(q));
  qi = open < 0 ? 0 : open;
  render();
}

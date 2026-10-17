// "程式與練習": two folds under the animation.
//   1. 虛擬碼與示範程式 – pseudocode plus the exact Python the animation runs
//   2. 挖空練習        – fill in the blanks of that code (or write it from
//                        scratch), graded in Pyodide, trajectory drawn over
//                        the reference solution

import { ContourPlot, seriesColor } from "./plots.js";
import { call, onState, warmUp } from "./py.js";

const ex = JSON.parse(document.getElementById("practice-data").textContent);
const mount = document.getElementById("practice");
const KEY = `ot-practice-${ex.id}`;

const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { return {}; } };
const saved = load();
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) { /* no storage */ } };

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

// Mute comments and docstrings so the executable lines stand out.
function codeView(src) {
  const pre = h("pre", { class: "code" });
  let inDoc = false;
  for (const line of src.replace(/\n$/, "").split("\n")) {
    const quotes = (line.match(/"""/g) || []).length;
    if (inDoc || quotes) {
      pre.append(h("span", { class: "c-doc" }, line), "\n");
      if (quotes === 1) inDoc = !inDoc;
      continue;
    }
    const i = line.indexOf("#");
    if (i >= 0) pre.append(line.slice(0, i), h("span", { class: "c-com" }, line.slice(i)), "\n");
    else pre.append(line, "\n");
  }
  return pre;
}

// Two-step confirmation inside the page (embedded viewers block confirm()).
function confirmTwice(btn, label, confirmLabel, action) {
  let armed = false, timer = null;
  btn.addEventListener("click", () => {
    if (!armed) {
      armed = true;
      btn.textContent = confirmLabel;
      timer = setTimeout(() => { armed = false; btn.textContent = label; }, 4000);
      return;
    }
    clearTimeout(timer);
    armed = false;
    btn.textContent = label;
    action();
  });
}

// ---------- fold 1: pseudocode + demo code ----------
const fold1 = h("details", { class: "fold" },
  h("summary", {}, "虛擬碼與示範程式", h("small", {}, "動畫執行的就是這段程式")),
  h("div", { class: "fold-body" },
    h("h4", {}, "虛擬碼"),
    h("pre", { class: "pseudo" }, ex.pseudocode),
    h("h4", {}, "Python 示範程式"),
    codeView(ex.solution),
    h("p", { class: "hint-text" }, ex.loop_note ||
      `上方動畫每一步呼叫一次 ${ex.func}()；外層迴圈負責提供梯度、保存狀態、記錄軌跡，並偵測發散與收斂。`)));

// ---------- fold 2: cloze / from-scratch practice ----------
// Split the solution into text segments and blanks, in source order.
const parts = [];
{
  let rest = ex.solution;
  const order = [...ex.blanks].sort((a, b) => ex.solution.indexOf(a) - ex.solution.indexOf(b));
  for (const b of order) {
    const i = rest.indexOf(b);
    parts.push(rest.slice(0, i), { blank: b });
    rest = rest.slice(i + b.length);
  }
  parts.push(rest);
}
saved.blanks = saved.blanks || [];
const inputs = [];
const cloze = h("div", { class: "cloze", role: "group", "aria-label": "挖空程式碼" });
for (const p of parts) {
  if (typeof p === "string") { cloze.append(p); continue; }
  const n = inputs.length;
  const input = h("input", {
    type: "text", class: "blank", id: `blank-${ex.id}-${n}`, autocomplete: "off", autocapitalize: "off",
    autocorrect: "off", spellcheck: "false", placeholder: `（${n + 1}）`, "aria-label": `第 ${n + 1} 個空格`,
  });
  input.style.width = `${Math.max(p.blank.length, 5) + 2}ch`;
  input.value = saved.blanks[n] || "";
  input.addEventListener("input", () => { saved.blanks[n] = input.value; save(); });
  inputs.push(input);
  cloze.append(input);
}

const editor = h("textarea", { class: "editor", id: `scratch-${ex.id}`, spellcheck: "false", wrap: "off", "aria-label": "你的 Python 程式" });
editor.value = saved.scratch || ex.skeleton;
editor.addEventListener("input", () => { saved.scratch = editor.value; save(); });
editor.addEventListener("keydown", (e) => {
  if (e.key !== "Tab") return;
  e.preventDefault();
  editor.setRangeText("    ", editor.selectionStart, editor.selectionEnd, "end");
  saved.scratch = editor.value; save();
});

let mode = saved.mode === "scratch" ? "scratch" : "cloze";
const clozeBtn = h("button", { class: "seg", type: "button" }, `挖空（${inputs.length} 格）`);
const scratchBtn = h("button", { class: "seg", type: "button" }, "從零寫");
const clozeBox = h("div", {}, h("p", { class: "hint-text" }, "把示範程式中被挖掉的運算式填回去。"), cloze);
const scratchBox = h("div", {}, h("p", { class: "hint-text" }, "只給函式骨架，整個函式自己寫。"), editor);
function setMode(m) {
  mode = m; saved.mode = m; save();
  clozeBtn.setAttribute("aria-pressed", String(m === "cloze"));
  scratchBtn.setAttribute("aria-pressed", String(m === "scratch"));
  clozeBox.hidden = m !== "cloze";
  scratchBox.hidden = m !== "scratch";
}
clozeBtn.addEventListener("click", () => setMode("cloze"));
scratchBtn.addEventListener("click", () => setMode("scratch"));

const runBtn = h("button", { class: "btn btn-primary", type: "button" }, "執行測試");
let hintsShown = 0;
const hintList = h("ol", { class: "hints" });
const hintBtn = h("button", { class: "btn", type: "button" }, `提示（0/${ex.hints.length}）`);
const ansBtn = h("button", { class: "btn", type: "button" }, "看答案");
const resetBtn = h("button", { class: "btn", type: "button" }, "清除");
const pyStatus = h("span", { class: "py-status" });
const verdict = h("div", { class: "verdict" });
const results = h("ul", { class: "results" });
const plotBox = h("div", {}, h("div", { class: "plot-title" }, "你的軌跡（實線）與參考解（虛線）"));

const fold2 = h("details", { class: "fold" },
  h("summary", {}, "挖空練習", h("small", {}, "在瀏覽器裡執行並自動評分")),
  h("div", { class: "fold-body" },
    h("div", { class: "segmented", role: "group", "aria-label": "練習模式" }, clozeBtn, scratchBtn),
    h("div", { class: "practice-grid" },
      h("div", { class: "practice-main" }, clozeBox, scratchBox,
        h("div", { class: "widget-row", style: "margin-top:10px" }, runBtn, hintBtn, ansBtn, resetBtn, pyStatus),
        hintList),
      h("div", {}, plotBox, verdict, results)),
    h("p", { class: "hint-text" }, "評分分三層：單步數值、在測試問題上的收斂行為、整條軌跡是否和參考解重合。")));

mount.append(h("div", { class: "folds" }, fold1, fold2));
setMode(mode);

const plot = new ContourPlot(plotBox, { maxHeight: 300 });
plot.setGrid(ex.grid);
if (ex.region) plot.setRegion(ex.region);

onState((s) => {
  pyStatus.textContent = { idle: "", loading: "Python 載入中…", ready: "Python 已就緒", error: "Python 載入失敗" }[s];
  pyStatus.classList.toggle("ready", s === "ready");
});

hintBtn.addEventListener("click", () => {
  if (hintsShown >= ex.hints.length) return;
  hintList.append(h("li", {}, ex.hints[hintsShown++]));
  hintBtn.textContent = `提示（${hintsShown}/${ex.hints.length}）`;
  hintBtn.disabled = hintsShown >= ex.hints.length;
});

confirmTwice(ansBtn, "看答案", "再按一次顯示答案", () => {
  if (mode === "cloze") {
    const order = parts.filter((p) => typeof p !== "string");
    inputs.forEach((input, i) => {
      input.value = order[i].blank;
      saved.blanks[i] = input.value;
      input.classList.add("revealed");
    });
    save();
  } else {
    editor.value = ex.solution;
    saved.scratch = editor.value; save();
  }
});
confirmTwice(resetBtn, "清除", "再按一次清除", () => {
  if (mode === "cloze") {
    inputs.forEach((input, i) => { input.value = ""; saved.blanks[i] = ""; input.classList.remove("revealed"); });
  } else {
    editor.value = ex.skeleton; saved.scratch = editor.value;
  }
  save();
});

function show(res) {
  results.replaceChildren(...res.results.map((r) => h("li", { class: r.passed ? "pass" : "fail" },
    h("span", { class: "mark", "aria-label": r.passed ? "通過" : "未通過" }, r.passed ? "✓" : "✗"),
    h("span", {}, h("span", { class: "layer-tag" }, r.layer ? `第 ${r.layer} 層` : "執行"), r.name),
    h("span", { class: "msg" }, r.message))));
  const n = res.results.filter((r) => r.passed).length;
  verdict.textContent = res.passed ? `全部通過（${n}/${res.results.length}）` : `通過 ${n}/${res.results.length}`;
  verdict.className = `verdict ${res.passed ? "pass" : "fail"}`;
  const series = [];
  if (res.ref_trace) series.push({ ...res.ref_trace, label: "參考解", color: getComputedStyle(document.documentElement).getPropertyValue("--ink-2").trim(), dash: true });
  if (res.user_trace) series.push({ ...res.user_trace, label: "你的程式", color: seriesColor(0), dots: !!ex.region });
  plot.setSeries(series);
}

function clozeSource() {
  let n = 0;
  return parts.map((p) => (typeof p === "string" ? p : inputs[n++].value.trim())).join("");
}

runBtn.addEventListener("click", async () => {
  if (mode === "cloze") {
    const empty = inputs.filter((i) => !i.value.trim()).length;
    if (empty) {
      show({ passed: false, results: [{ layer: 0, name: "填空", passed: false, message: `還有 ${empty} 個空格沒填。` }] });
      return;
    }
  }
  runBtn.disabled = true;
  runBtn.textContent = "執行中…";
  try {
    const source = mode === "cloze" ? clozeSource() : editor.value;
    show(await call("grade", { exercise: ex.id, source }, 8000));
  } catch (err) {
    show({ passed: false, results: [{ layer: 0, name: "執行", passed: false, message: err.message }] });
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "執行測試";
  }
});

// Python is only needed once someone opens the practice fold.
fold2.addEventListener("toggle", () => { if (fold2.open) warmUp().catch(() => {}); });

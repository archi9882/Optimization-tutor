// Scratch practice: write the step function, grade it in Pyodide, and see
// your own trajectory drawn over the reference solution.

import { ContourPlot, seriesColor } from "./plots.js";
import { call, onState, warmUp } from "./py.js";

const ex = JSON.parse(document.getElementById("practice-data").textContent);
const mount = document.getElementById("practice");
const KEY = `ot-code-${ex.id}`;

const load = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
const save = (v) => { try { localStorage.setItem(KEY, v); } catch (e) { /* no storage */ } };

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

const editor = h("textarea", { class: "editor", spellcheck: "false", wrap: "off", "aria-label": "你的 Python 程式" });
editor.value = load() || ex.skeleton;
editor.addEventListener("input", () => save(editor.value));
editor.addEventListener("keydown", (e) => {
  if (e.key !== "Tab") return;
  e.preventDefault();
  const { selectionStart: s, selectionEnd: t } = editor;
  editor.setRangeText("    ", s, t, "end");
  save(editor.value);
});

const runBtn = h("button", { class: "btn btn-primary", type: "button" }, "執行測試");
let hintsShown = 0;
const hintList = h("ol", { class: "hints" });
const hintBtn = h("button", { class: "btn", type: "button" }, `提示（0/${ex.hints.length}）`);
const solBtn = h("button", { class: "btn", type: "button" }, "看解答");
const resetBtn = h("button", { class: "btn", type: "button" }, "重設");
const solBox = h("div");
const pyStatus = h("span", { class: "py-status" });
const verdict = h("div", { class: "verdict" });
const results = h("ul", { class: "results" });
const plotBox = h("div", {}, h("div", { class: "plot-title" }, "你的軌跡（實線）與參考解（虛線）"));

mount.append(h("div", { class: "widget practice" },
  h("p", { style: "margin:0" }, "完成 ", h("code", {}, `${ex.func}()`), "，按「執行測試」。程式在你的瀏覽器中執行，分三層檢查：單步數值、收斂行為、整條軌跡。"),
  h("div", { class: "practice-grid" },
    h("div", {}, editor,
      h("div", { class: "widget-row", style: "margin-top:8px" }, runBtn, hintBtn, solBtn, resetBtn, pyStatus),
      hintList, solBox),
    h("div", {}, plotBox, verdict, results))));

const plot = new ContourPlot(plotBox, { maxHeight: 300 });
plot.setGrid(ex.grid);

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
solBtn.addEventListener("click", () => {
  if (!confirm("先自己試試看？確定要看解答嗎？")) return;
  solBox.replaceChildren(h("p", { class: "hint-text" }, "參考解答（看完建議關掉，再自己重寫一次）："),
    h("pre", { class: "solution" }, ex.solution));
});
resetBtn.addEventListener("click", () => {
  if (!confirm("清除你的程式，回到初始骨架？")) return;
  editor.value = ex.skeleton;
  save(editor.value);
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
  if (res.user_trace) series.push({ ...res.user_trace, label: "你的程式", color: seriesColor(0) });
  plot.setSeries(series);
}

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  runBtn.textContent = "執行中…";
  try {
    show(await call("grade", { exercise: ex.id, source: editor.value }, 8000));
  } catch (err) {
    show({ passed: false, results: [{ layer: 0, name: "執行", passed: false, message: err.message }] });
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "執行測試";
  }
});

warmUp().catch(() => {});

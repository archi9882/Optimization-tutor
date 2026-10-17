// Interactive lab: race optimizers on a 2D problem.
// First frame comes from build-time data; edits re-run optlab in Pyodide.

import { ContourPlot, ConvergencePlot, seriesColor, fmt } from "./plots.js";
import { call, onState, warmUp } from "./py.js";

const data = JSON.parse(document.getElementById("lab-data").textContent);
const mount = document.getElementById("lab");

const PROBLEMS = {
  quadratic: { name: "二次碗", params: [
    { name: "kappa", label: "條件數 κ", lo: 1, hi: 200, log: true, default: 10 },
    { name: "theta_deg", label: "旋轉 θ", lo: 0, hi: 90, log: false, default: 0, unit: "°" },
  ] },
  rosenbrock: { name: "Rosenbrock 香蕉谷", params: [] },
  himmelblau: { name: "Himmelblau 四谷", params: [] },
};

const st = {
  spec: structuredClone(data.spec),
  enabled: new Set(data.enabled),
  params: Object.fromEntries(data.spec.runs.map((r) => [r.algo, { ...r.params }])),
  traces: data.traces,
  grid: data.grid,
  frame: Infinity,
  playing: false,
  speed: 30,
  version: 0,
};

// ---------- small DOM helpers ----------
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

function slider({ label, lo, hi, log, value, unit = "", onInput }) {
  const toPos = (v) => (log ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo)) * 1000;
  const fromPos = (p) => {
    const u = p / 1000;
    const v = log ? Math.exp(Math.log(lo) + u * (Math.log(hi) - Math.log(lo))) : lo + u * (hi - lo);
    return Number(v.toPrecision(3));
  };
  const out = h("output", {}, `${value}${unit}`);
  const input = h("input", { type: "range", min: 0, max: 1000, step: 1, "aria-label": label });
  input.value = toPos(value);
  input.addEventListener("input", () => {
    const v = fromPos(Number(input.value));
    out.textContent = `${v}${unit}`;
    onInput(v);
  });
  return h("label", { class: "slider" }, label, input, out);
}

// ---------- layout ----------
const pyStatus = h("span", { class: "py-status" });
const problemSel = h("select", { "aria-label": "測試問題" },
  ...Object.entries(PROBLEMS).map(([id, p]) => h("option", { value: id }, p.name)));
problemSel.value = st.spec.problem;
const problemParams = h("span", { class: "widget-row" });
const iterSlider = slider({ label: "迭代數", lo: 20, hi: 500, log: false, value: st.spec.n_iter,
  onInput: (v) => { st.spec.n_iter = Math.round(v); schedule(false); } });

const contourBox = h("div", {}, h("div", { class: "plot-title" }, "等高線與路徑（等比例座標，＋為最小值）"));
const convBox = h("div", {}, h("div", { class: "plot-title" }, "f(xₖ) − f*（對數座標）"));
const rows = h("div", { class: "algo-rows" });
const playBtn = h("button", { class: "btn btn-primary", type: "button", onclick: () => (st.playing ? pause() : play()) }, "▶ 播放");
const stepBtn = h("button", { class: "btn", type: "button", onclick: () => stepOnce() }, "單步");
const speedSel = h("select", { "aria-label": "播放速度" },
  h("option", { value: 10 }, "慢"), h("option", { value: 30, selected: "" }, "中"), h("option", { value: 90 }, "快"));
speedSel.addEventListener("change", () => { st.speed = Number(speedSel.value); });
const frameOut = h("span", { class: "status-line" });

mount.append(h("div", { class: "widget lab" },
  h("div", { class: "widget-row" }, h("label", { class: "ctl" }, "問題 ", problemSel), problemParams, iterSlider, pyStatus),
  h("div", { class: "lab-plots" }, contourBox, convBox),
  rows,
  h("div", { class: "widget-row", style: "margin-top:12px" }, playBtn, stepBtn, h("label", { class: "ctl" }, "速度 ", speedSel), frameOut),
  h("p", { class: "hint-text" }, "點擊等高線圖可以更換起點；滑鼠移到路徑上可以看每一步的 f 值。")));

const contour = new ContourPlot(contourBox);
const conv = new ConvergencePlot(convBox, { height: 300 });
contour.onPick((x0) => { st.spec.x0 = x0.map((v) => Number(v.toFixed(3))); schedule(false); });

onState((s) => {
  pyStatus.textContent = { idle: "", loading: "Python 載入中…（調整參數會在載入後生效）", ready: "Python 已就緒", error: "Python 載入失敗，只能看預設動畫" }[s];
  pyStatus.classList.toggle("ready", s === "ready");
});

function renderProblemParams() {
  problemParams.replaceChildren(...PROBLEMS[st.spec.problem].params.map((p) => slider({
    ...p, value: st.spec.problem_params[p.name] ?? p.default,
    onInput: (v) => { st.spec.problem_params[p.name] = v; schedule(true); },
  })));
}

const resultEls = {};
function renderRows() {
  rows.replaceChildren(...data.algos.map((id, i) => {
    const meta = data.optimizers[id];
    const box = h("input", { type: "checkbox", "aria-label": `顯示 ${meta.name}` });
    box.checked = st.enabled.has(id);
    const row = h("div", { class: `algo-row${box.checked ? "" : " off"}` });
    box.addEventListener("change", () => {
      box.checked ? st.enabled.add(id) : st.enabled.delete(id);
      row.classList.toggle("off", !box.checked);
      redraw();
    });
    const sw = h("span", { class: "swatch" });
    sw.style.background = seriesColor(i);
    resultEls[id] = h("span", { class: "algo-result" });
    row.append(h("label", { class: "algo-name" }, box, sw, meta.name),
      ...meta.params.map((p) => slider({ ...p, value: st.params[id][p.name],
        onInput: (v) => { st.params[id][p.name] = v; schedule(false); } })),
      resultEls[id]);
    return row;
  }));
}

function describe(t) {
  const gap = t.fs[t.fs.length - 1] - (st.grid.fstar ?? 0);
  if (t.status === "diverged") return `第 ${t.n_iter + 1} 步發散`;
  const ev = t.evals !== undefined ? ` · ${t.evals} 次 f 計算` : "";
  if (t.status === "converged") return `${t.n_iter} 步收斂 · f−f* = ${fmt(gap)}${ev}`;
  if (t.status === "stopped") return `${t.n_iter} 步停止 · f−f* = ${fmt(gap)}${ev}`;
  return `${t.n_iter} 步 · f−f* = ${fmt(gap)}${ev}`;
}

function visibleSeries() {
  return data.algos.map((id, i) => ({ id, i, t: st.traces.find((t) => t.algo === id) }))
    .filter((s) => s.t && st.enabled.has(s.id))
    .map((s) => ({ xs: s.t.xs, fs: s.t.fs, status: s.t.status, radius: s.t.radius, poly: s.t.poly, polyKind: s.t.poly_kind, label: data.optimizers[s.id].name, color: seriesColor(s.i) }));
}

function maxLen() { return Math.max(1, ...visibleSeries().map((s) => s.xs.length)); }

function redraw() {
  const series = visibleSeries();
  contour.setSeries(series);
  conv.setSeries(series, st.grid.fstar ?? 0, st.spec.n_iter);
  for (const t of st.traces) if (resultEls[t.algo]) {
    resultEls[t.algo].textContent = describe(t);
    resultEls[t.algo].classList.toggle("diverged", t.status === "diverged");
  }
  setFrame(st.frame);
}

function setFrame(k) {
  st.frame = k;
  contour.setFrame(k);
  conv.setFrame(k);
  const shown = Math.min(k, maxLen() - 1);
  frameOut.textContent = Number.isFinite(k) ? `第 ${shown} 步` : "";
}

// ---------- animation ----------
let last = 0, acc = 0;
function tick(ts) {
  if (!st.playing) return;
  acc += ((ts - last) / 1000) * st.speed;
  last = ts;
  const adv = Math.floor(acc);
  if (adv > 0) {
    acc -= adv;
    const next = (Number.isFinite(st.frame) ? st.frame : 0) + adv;
    if (next >= maxLen() - 1) { setFrame(Infinity); pause(); return; }
    setFrame(next);
  }
  requestAnimationFrame(tick);
}
function play() {
  if (!Number.isFinite(st.frame) || st.frame >= maxLen() - 1) setFrame(0);
  st.playing = true; playBtn.textContent = "⏸ 暫停";
  last = performance.now(); acc = 0;
  requestAnimationFrame(tick);
}
function pause() { st.playing = false; playBtn.textContent = "▶ 播放"; }
function stepOnce() {
  pause();
  const cur = Number.isFinite(st.frame) ? st.frame : -1;
  setFrame(cur + 1 >= maxLen() ? 0 : cur + 1);
}

// ---------- live recompute through Pyodide ----------
let timer = null;
let gridDirty = false;
function schedule(needGrid) {
  gridDirty ||= needGrid;
  clearTimeout(timer);
  timer = setTimeout(recompute, 150);
}

async function recompute() {
  const version = ++st.version;
  const needGrid = gridDirty;
  gridDirty = false;
  try {
    const spec = { ...st.spec, runs: data.algos.map((a) => ({ algo: a, params: st.params[a] })) };
    const grid = needGrid ? await call("grid", spec) : null;
    if (grid && !spec.x0) spec.x0 = grid.x0;
    const traces = await call("traces", spec);
    if (version !== st.version) return; // a newer request superseded this one
    if (grid) { st.grid = grid; contour.setGrid(grid); }
    st.traces = traces;
    pause();
    st.frame = Infinity;
    redraw();
  } catch (err) {
    frameOut.textContent = `計算失敗：${err.message}`;
  }
}

problemSel.addEventListener("change", () => {
  st.spec.problem = problemSel.value;
  st.spec.problem_params = Object.fromEntries(PROBLEMS[st.spec.problem].params.map((p) => [p.name, p.default]));
  st.spec.x0 = null;
  renderProblemParams();
  schedule(true);
});

// ---------- boot ----------
renderProblemParams();
renderRows();
contour.setGrid(st.grid);
redraw();
warmUp().catch(() => {});
// Play once the first time the lab scrolls into view.
new IntersectionObserver((entries, obs) => {
  if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); play(); }
}, { threshold: 0.4 }).observe(mount);

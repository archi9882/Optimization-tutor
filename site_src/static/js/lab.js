// Interactive lab: race optimizers on a 2D problem.
// First frame comes from build-time data; edits re-run optlab in Pyodide.

import { ContourPlot, ConvergencePlot, seriesColor, fmt, touchUI } from "./plots.js";
import { call, onState, warmUp } from "./py.js";

const data = JSON.parse(document.getElementById("lab-data").textContent);
const mount = document.getElementById("lab");

// Slider grids (data.problem_params / optimizers[..].params[..].values) come
// from optlab.sliders at build time, so every position is a "nice" number.
const PROBLEMS = {
  quadratic: "二次碗",
  rosenbrock: "Rosenbrock 香蕉谷",
  himmelblau: "Himmelblau 四谷",
  // Constrained problems (offered only on pages that list them in lab.problems).
  fert_budget: "施肥預算（三角形）",
  fert_eq: "預算全部用完（等式）",
  fert_disk: "土壤養分圓盤",
  reservoir: "兩座水庫放水（箱型）",
  rosen_disk: "Rosenbrock + 圓盤（非凸）",
  maratos: "圓上等式（Maratos 例）",
  fert_lasso: "施肥稀疏（ℓ1）",
  // Test functions (optlab.testfunctions), offered on pages that list them.
  sphere: "Sphere 球面",
  ackley: "Ackley",
  levy: "Levy",
  cross_in_tray: "Cross-in-tray 十字托盤",
  bukin6: "Bukin N.6",
  rastrigin: "Rastrigin",
  bohachevsky1: "Bohachevsky N.1",
  griewank: "Griewank",
};
const problemParams = (id) => data.problem_params[id] || [];
// "f − f* < tol 於第 k 步" uses the same tolerance as the page text.
const REACH_TOL = data.reach_tol ?? 1e-10;
const PROBLEM_IDS = data.problems || ["quadratic", "rosenbrock", "himmelblau"];
// With constraints (or an l1 term, grid.distance) the gradient of f is not zero
// at the optimum, so the convergence plot shows the distance to x* instead.
const constrained = () => !!(st.grid.constraints || st.grid.distance);

const initial = () => ({
  spec: structuredClone(data.spec),
  enabled: new Set(data.enabled),
  params: Object.fromEntries(data.spec.runs.map((r) => [r.algo, { ...r.params }])),
  traces: data.traces,
  grid: data.grid,
  frame: Infinity,
});
const st = { ...initial(), playing: false, speed: 30, version: 0 };

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

// A slider over a fixed list of values: the range input moves an index, so
// every position is exactly one of `values` (no rounding of a continuous scale).
function slider({ label, values, value, unit = "", onInput }) {
  const index = (v) => {
    let best = 0;
    values.forEach((x, i) => { if (Math.abs(x - v) < Math.abs(values[best] - v)) best = i; });
    return best;
  };
  const out = h("output", {});
  const input = h("input", { type: "range", min: 0, max: values.length - 1, step: 1, "aria-label": label });
  const show = (i) => {
    const text = `${values[i]}${unit}`;
    out.textContent = text;
    input.setAttribute("aria-valuetext", text);
  };
  const el = h("label", { class: "slider" }, label, input, out);
  el.setValue = (v) => { input.value = index(v); show(Number(input.value)); };
  el.setValue(value);
  input.addEventListener("input", () => {
    const i = Number(input.value);
    show(i);
    onInput(values[i]);
  });
  return el;
}

// ---------- layout ----------
const pyStatus = h("span", { class: "py-status" });
const problemSel = h("select", { "aria-label": "測試問題" },
  ...PROBLEM_IDS.map((id) => h("option", { value: id }, PROBLEMS[id])));
problemSel.value = st.spec.problem;
const problemParamBox = h("span", { class: "widget-row" });
const iterSlider = slider({ label: data.n_iter.label, values: data.n_iter.values, value: st.spec.n_iter,
  onInput: (v) => { st.spec.n_iter = v; schedule(false); } });
const resetBtn = h("button", { class: "btn btn-small", type: "button", title: "回到本頁預設的問題、參數、起點與迭代數",
  onclick: () => reset() }, "重設");

const contourBox = h("div", {}, h("div", { class: "plot-title" }, "等高線與路徑（等比例座標，＋為最小值）"));
const convBox = h("div", {}, h("div", { class: "plot-title" }, "f(xₖ) − f*（對數座標）"));
const rows = h("div", { class: "algo-rows" });
const playBtn = h("button", { class: "btn btn-primary", type: "button", onclick: () => (st.playing ? pause() : play()) }, "▶ 播放");
const stepBtn = h("button", { class: "btn", type: "button", onclick: () => stepOnce() }, "單步");
const speedSel = h("select", { "aria-label": "播放速度" },
  h("option", { value: 10 }, "慢"), h("option", { value: 30, selected: "" }, "中"), h("option", { value: 90 }, "快"));
speedSel.addEventListener("change", () => { st.speed = Number(speedSel.value); });
const frameOut = h("span", { class: "status-line" });
const problemNote = h("p", { class: "hint-text" });
const stepNote = h("p", { class: "step-note", "aria-live": "polite", style: "white-space: pre-line" });
const popNote = h("p", { class: "hint-text pop-note", hidden: "" });
const popText = (trail, local) => `族群方法：小圓點是這一步的族群${trail ? "，細線是粒子這一步的移動" : ""}，◎ 是到目前為止找到的最佳點；` +
  `右圖實線是最佳值、虛線是族群平均 f。${local ? "○ 為局部最小值、" : ""}＋ 為全域最小值。`;
const convTitle = convBox.querySelector(".plot-title");

const touch = touchUI();
mount.append(h("div", { class: "widget lab" },
  h("div", { class: "widget-row" }, h("label", { class: "ctl" }, "問題 ", problemSel), problemParamBox, iterSlider, resetBtn, pyStatus),
  problemNote,
  h("div", { class: "lab-plots" }, contourBox, convBox),
  popNote,
  stepNote,
  rows,
  h("div", { class: "widget-row", style: "margin-top:12px" }, playBtn, stepBtn, h("label", { class: "ctl" }, "速度 ", speedSel), frameOut),
  h("p", { class: "hint-text" }, touch
    ? "點一下路徑上的點可以看那一步的 f 值；點其他位置可以更換起點。"
    : "點擊等高線圖可以更換起點；滑鼠移到路徑上可以看每一步的 f 值。")));

const contour = new ContourPlot(contourBox);
const conv = new ConvergencePlot(convBox, { height: 300 });
// Phones stack the plots in one column, where an equal-aspect contour plot of a
// wide problem is short: match the convergence plot to it instead of towering over it.
const narrow = typeof matchMedia === "function" ? matchMedia("(max-width: 760px)") : { matches: false };
contour.onResize((ch) => conv.setHeight(narrow.matches ? Math.round(Math.max(180, Math.min(300, ch))) : 300));
contour.onPick((x0) => { st.spec.x0 = x0.map((v) => Number(v.toFixed(3))); schedule(false); });

onState((s) => {
  pyStatus.textContent = { idle: "", loading: "Python 載入中…（調整參數會在載入後生效）", ready: "Python 已就緒", error: "Python 載入失敗，只能看預設動畫" }[s];
  pyStatus.classList.toggle("ready", s === "ready");
});

function renderProblemParams() {
  problemParamBox.replaceChildren(...problemParams(st.spec.problem).map((p) => slider({
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

// 1e-10 -> "10⁻¹⁰"
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const pow10 = (v) => `10${String(Math.round(Math.log10(v))).replace(/./g, (c) => SUP[c])}`;

// Two step counts, each with its criterion: where the run stopped (the
// runner's rule) and the first step with f − f* below the page's tolerance.
function describe(t) {
  if (t.status === "diverged") return `第 ${t.n_iter + 1} 步發散`;
  if (constrained()) {
    const x = t.xs[t.xs.length - 1], xs = st.grid.minima[0];
    const d = `‖x−x*‖ = ${fmt(Math.hypot(x[0] - xs[0], x[1] - xs[1]))}` + (st.grid.constraints ? ` · 違反量 ${fmt(t.violation)}` : "");
    if (t.message) return `第 ${t.n_iter} 步停止：${t.message} · ${d}`;
    return `${t.status === "stopped" ? `第 ${t.n_iter} 步停止` : `${t.n_iter} 步內未停止`} · ${d}`;
  }
  const fstar = st.grid.fstar ?? 0;
  const gap = t.fs[t.fs.length - 1] - fstar;
  const stop = { converged: `第 ${t.n_iter} 步停止（‖∇f‖ < 10⁻¹⁰）`, stopped: `第 ${t.n_iter} 步停止（演算法本身的停止條件）` }[t.status]
    || `${t.n_iter} 步內未停止`;
  const k = t.fs.findIndex((f) => f - fstar < REACH_TOL);
  const reach = k >= 0 ? `f−f* < ${pow10(REACH_TOL)} 於第 ${k} 步` : `未達 f−f* < ${pow10(REACH_TOL)}`;
  const ev = t.evals !== undefined ? ` · ${t.evals} 次 f 計算` : "";
  return `${reach} · ${stop} · 最後 f−f* = ${fmt(gap)}${ev}`;
}

function visibleSeries() {
  return data.algos.map((id, i) => ({ id, i, t: st.traces.find((t) => t.algo === id) }))
    .filter((s) => s.t && st.enabled.has(s.id))
    .flatMap((s) => [{ xs: s.t.xs, fs: s.t.fs, status: s.t.status, radius: s.t.radius, poly: s.t.poly, polyKind: s.t.poly_kind, label: data.optimizers[s.id].name, color: seriesColor(s.i),
      info: s.t.info, ...(s.t.outer ? { dots: true, big: s.t.outer } : s.t.dots ? { dots: true } : {}),
      ...(s.t.pop ? { pop: s.t.pop, trail: !!s.t.pop_trail } : {}) },
    // A second iterate (ADMM's z) as a dashed path in the same color, with a hollow marker.
    ...(s.t.zs ? [{ xs: s.t.zs, status: s.t.status, label: `${data.optimizers[s.id].name} 的 z`, color: seriesColor(s.i), dash: true, mark: true }] : []),
    // Population methods (renderer ③): the points of each step on the contour plot,
    // and the population's mean f as a dashed line on the convergence plot only.
    ...(s.t.pop ? [{ xs: s.t.xs, fs: s.t.fmean, status: s.t.status, label: `${data.optimizers[s.id].name} 族群平均`, color: seriesColor(s.i), dash: true, convOnly: true }] : [])]);
}

function maxLen() { return Math.max(1, ...visibleSeries().map((s) => s.xs.length)); }

function redraw() {
  const series = visibleSeries();
  contour.setSeries(series.filter((s) => !s.convOnly));
  popNote.hidden = !series.some((s) => s.pop);
  if (!popNote.hidden) popNote.textContent = popText(series.some((s) => s.pop && s.trail), !!st.grid.local_minima?.length);
  if (constrained()) {
    const xs = st.grid.minima[0];
    const dist = series.map((s) => s.xs.map(([a, b]) => Math.hypot(a - xs[0], b - xs[1])));
    // An exact hit (distance 0) would sit at 1e-16 and stretch the log axis:
    // clamp to a floor one decade below the smallest nonzero distance (at least 1e-12).
    const nonzero = dist.flat().filter((v) => v > 0);
    const floor = Math.max(1e-12, (nonzero.length ? Math.min(...nonzero) : 1) / 10);
    conv.setSeries(series.map((s, i) => ({ ...s, fs: dist[i].map((v) => Math.max(v, floor)) })), 0, st.spec.n_iter);
  } else conv.setSeries(series, st.grid.fstar ?? 0, st.spec.n_iter);
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
  if (constrained()) stepNote.textContent = visibleSeries().filter((s) => s.info).map((s) => {
    const i = Math.min(k, s.xs.length - 1), info = s.info[i];
    if (!info) return `${s.label}：起點`;
    const num = (v) => (Number.isInteger(v) ? String(v) : fmt(v));
    const fv = (v) => (Array.isArray(v) ? `(${v.map(num).join(", ")})` : num(v));
    return `${s.label} 第 ${i} 步：` + Object.entries(info).map(([n, v]) => `${n} = ${fv(v)}`).join(" · ");
  }).join("\n");
}

function renderProblemNote() {
  const c = constrained();
  problemNote.hidden = !c;
  stepNote.hidden = !c;
  problemNote.textContent = c ? st.grid.note : "";
  convTitle.textContent = !c ? "f(xₖ) − f*（對數座標）"
    : st.grid.constraints ? "‖xₖ − x*‖：到有限制最優解的距離（對數座標）" : "‖xₖ − x*‖：到最優解的距離（對數座標）";
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
// Callers of playThrough() wait for the next pause: the end of the run, the
// pause button, or any edit that recomputes (finished tells which).
let pauseWaiters = [];
function pause() {
  st.playing = false; playBtn.textContent = "▶ 播放";
  const waiters = pauseWaiters;
  pauseWaiters = [];
  for (const fn of waiters) fn(!Number.isFinite(st.frame));
}
function playThrough() { return new Promise((resolve) => { pauseWaiters.push(resolve); play(); }); }
function stepOnce() {
  pause();
  const cur = Number.isFinite(st.frame) ? st.frame : -1;
  setFrame(cur + 1 >= maxLen() ? 0 : cur + 1);
}

// ---------- live recompute through Pyodide ----------
let timer = null;
let gridDirty = false;
function schedule(needGrid) {
  st.version++;               // an edit supersedes any pending scenario or recompute
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
    if (grid) { st.grid = grid; contour.setGrid(grid); renderProblemNote(); }
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
  st.spec.problem_params = Object.fromEntries(problemParams(st.spec.problem).map((p) => [p.name, p.default]));
  st.spec.x0 = null;
  renderProblemParams();
  schedule(true);
});

// Back to the page's defaults; the build-time traces are still valid, so no
// Python call is needed.
function reset() {
  clearTimeout(timer);
  gridDirty = false;
  st.version++;               // drop any recompute still in flight
  pause();
  Object.assign(st, initial());
  problemSel.value = st.spec.problem;
  iterSlider.setValue(st.spec.n_iter);
  renderProblemParams();
  renderRows();
  contour.setGrid(st.grid);
  renderProblemNote();
  frameOut.textContent = "";
  redraw();
}

// ---------- boot ----------
renderProblemParams();
renderRows();
renderProblemNote();
contour.setGrid(st.grid);
redraw();
warmUp().catch(() => {});
// Prediction questions (frontmatter predict:): put a scenario computed at
// build time on the lab. The traces are embedded; only a scenario on another
// problem or problem parameter needs Python for its contour grid.
// Any later edit, reset or cancelScenario() bumps st.version; a scenario
// still waiting for Python then rejects with err.superseded instead of
// overwriting what the learner did meanwhile.
const superseded = () => Object.assign(new Error("superseded"), { superseded: true });
function cancelScenario() { st.version++; }
function showEnd() { pause(); setFrame(Infinity); }
async function applyScenario(sc) {
  clearTimeout(timer);
  gridDirty = false;
  const version = ++st.version;
  pause();
  const grid = sc.same_grid ? data.grid : await call("grid", sc.spec);
  if (version !== st.version) throw superseded();
  st.spec = structuredClone(sc.spec);
  st.params = Object.fromEntries(sc.spec.runs.map((r) => [r.algo, { ...r.params }]));
  st.enabled = new Set(sc.enabled);
  // {algo, same: true}: identical to the page's default trace (not embedded twice).
  st.traces = sc.traces.map((t) => (t.same ? data.traces.find((d) => d.algo === t.algo) : t));
  st.grid = grid;
  st.frame = Infinity;
  problemSel.value = st.spec.problem;
  iterSlider.setValue(st.spec.n_iter);
  renderProblemParams();
  renderRows();
  contour.setGrid(grid);
  renderProblemNote();
  frameOut.textContent = "";
  redraw();
}

// Play once the first time the lab scrolls into view (pages with prediction
// questions wait for the learner to predict or skip instead).
function autoplay() {
  new IntersectionObserver((entries, obs) => {
    if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); play(); }
  }, { threshold: 0.4 }).observe(mount);
}
if (data.predict) {
  import("./predict.js")
    .then((m) => m.mountPredict(mount, data.predict, { applyScenario, cancelScenario, playThrough, pause, play, showEnd }))
    .catch(autoplay);
} else autoplay();

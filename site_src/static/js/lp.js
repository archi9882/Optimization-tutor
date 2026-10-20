// LP lab (renderer ②): feasible polygon + iso-objective lines + the path of
// the method, stepped together with its tableau / central path / working set.
// First frame comes from build-time data; slider edits re-run optlab.lp in Pyodide.

import { ContourPlot, seriesColor, fmt, touchUI } from "./plots.js";
import { call, onState, warmUp } from "./py.js";

const data = JSON.parse(document.getElementById("lp-data").textContent);
const mount = document.getElementById("lp");

// Coefficient sliders {label, lo, hi, step, index}: defined once in
// scripts/build.py (LP_SLIDERS), where prediction scenarios are checked against them.
const SLIDERS = data.slider_defs || {};

const st = {
  spec: structuredClone(data.spec),
  res: data.result,
  frame: 0,
  playing: false,
  version: 0,
};

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

// Tableau entries are mostly small fractions; show 1/3 rather than 0.3333.
function frac(v) {
  if (Math.abs(v) < 1e-9) return "0";
  for (let d = 1; d <= 12; d++) {
    const n = Math.round(v * d);
    if (Math.abs(v * d - n) < 1e-7) return d === 1 ? String(n) : `${n}/${d}`;
  }
  return fmt(v);
}

// Slider value of a spec: its preset_params, else the preset's own c.
const valueOf = (spec, res, name) => spec.preset_params[name] ?? res.region.c[SLIDERS[name].index];
const initialValue = (name) => valueOf(data.spec, data.result, name);

const sliderSetters = {};
function slider(name, value) {
  const m = SLIDERS[name];
  const out = h("output", {}, String(value));
  const input = h("input", { type: "range", min: m.lo, max: m.hi, step: m.step, value, "aria-label": m.label });
  input.addEventListener("input", () => {
    out.textContent = input.value;
    st.spec.preset_params[name] = Number(input.value);
    schedule();
  });
  sliderSetters[name] = (v) => { input.value = String(v); out.textContent = input.value; };
  return h("label", { class: "slider" }, m.label, input, out);
}

// ---------- layout ----------
const pyStatus = h("span", { class: "py-status" });
const sliders = data.sliders.map((n) => slider(n, initialValue(n)));
const plotBox = h("div", {}, h("div", { class: "plot-title" }, "可行域（亮區）、等目標值線與路徑"));
const panelTitle = h("div", { class: "plot-title" });
const panel = h("div", { class: "lp-panel" });
const note = h("p", { class: "step-note", "aria-live": "polite" });
const counter = h("span", { class: "status-line" });
const resetBtn = h("button", { class: "btn", type: "button", onclick: () => { pause(); setFrame(0); } }, "⏮ 重來");
const defaultsBtn = h("button", { class: "btn btn-small", type: "button", title: "回到本頁預設的係數與求解結果",
  onclick: () => resetParams() }, "重設");
const prevBtn = h("button", { class: "btn", type: "button", onclick: () => { pause(); setFrame(st.frame - 1); } }, "◀ 上一步");
const playBtn = h("button", { class: "btn btn-primary", type: "button", onclick: () => (st.playing ? pause() : play()) }, "▶ 播放");
const nextBtn = h("button", { class: "btn", type: "button", onclick: () => { pause(); setFrame(st.frame + 1); } }, "下一步 ▶");

mount.append(h("div", { class: "widget lab lp-lab" },
  h("p", { class: "lp-problem" }, h("strong", {}, st.res.name), " ", h("span", { class: "hint-text" }, st.res.note)),
  sliders.length ? h("div", { class: "widget-row" }, ...sliders, defaultsBtn, pyStatus) : "",
  h("div", { class: `lab-plots${st.res.method === "cutting_plane" ? " even" : ""}` }, plotBox, h("div", {}, panelTitle, panel)),
  note,
  h("div", { class: "widget-row" }, resetBtn, prevBtn, playBtn, nextBtn, counter),
  h("p", { class: "hint-text" }, touchUI()
    ? "點一下路徑上的點可以看那一步的座標與目標值；拖動係數滑桿會重新求解。"
    : "滑鼠移到路徑上可以看每一步的座標與目標值；拖動係數滑桿會重新求解。")));

const plot = new ContourPlot(plotBox, { maxHeight: 440 });

onState((s) => {
  pyStatus.textContent = { idle: "", loading: "Python 載入中…（調整係數會在載入後生效）", ready: "Python 已就緒", error: "Python 載入失敗，只能看預設動畫" }[s];
  pyStatus.classList.toggle("ready", s === "ready");
});

// ---------- right-hand panels ----------
function tableauPanel(k) {
  const step = st.res.steps[k];
  const next = st.res.steps[k + 1]?.pivot;
  const [pr, pc] = next || [-1, -1];
  const T = step.tableau, cols = step.cols;
  const head = h("tr", {}, h("th", { scope: "col" }, "基"), ...cols.map((c, j) => h("th", { scope: "col", class: j === pc ? "piv-col" : "" }, c)));
  const body = T.map((row, i) => {
    const last = i === T.length - 1;
    const label = last ? "目標列" : step.basis[i];
    return h("tr", { class: `${last ? "obj-row" : ""}${i === pr ? " piv-row" : ""}` },
      h("th", { scope: "row" }, label),
      ...row.map((v, j) => h("td", { class: `${j === pc ? "piv-col" : ""}${i === pr && j === pc ? " piv-cell" : ""}` }, frac(v))));
  });
  const caption = next
    ? `下一步：${cols[pc]} 進基（欄），${step.basis[pr]} 出基（列），框起來的是樞紐元素。`
    : k < st.res.steps.length - 1 ? "下一步：由這張表產生一條割平面。"
    : st.res.status === "optimal" ? "已達最優，沒有下一個樞紐。" : `停止：${st.res.status}`;
  panelTitle.textContent = "單純形表（每一步對應圖上的一個頂點）";
  panel.replaceChildren(h("div", { class: "tableau-wrap" }, h("table", { class: "tableau" }, h("thead", {}, head), h("tbody", {}, ...body))),
    h("p", { class: "hint-text" }, caption));
}

function barrierPanel(k) {
  panelTitle.textContent = "中心路徑上的點（每個 μ 做完牛頓步後）";
  const rows = st.res.steps.map((s, i) => ({ ...s, i })).filter((s) => s.central && s.i <= k);
  const cur = st.res.steps[k];
  panel.replaceChildren(
    h("div", { class: "tableau-wrap" }, h("table", { class: "tableau" },
      h("thead", {}, h("tr", {}, ...["步", "μ", "x₁", "x₂", "z"].map((c) => h("th", { scope: "col" }, c)))),
      h("tbody", {}, ...rows.map((s) => h("tr", { class: s.i === k ? "piv-row" : "" },
        h("td", {}, String(s.i)), h("td", {}, fmt(s.mu)), h("td", {}, s.x[0].toFixed(4)), h("td", {}, s.x[1].toFixed(4)), h("td", {}, s.z.toFixed(4))))))),
    h("p", { class: "hint-text" }, `目前 μ = ${fmt(cur.mu)}。μ 越小，障礙越弱，點越靠近邊界上的最優頂點。`));
}

function activeSetPanel(k) {
  const s = st.res.steps[k];
  panelTitle.textContent = "工作集 W（目前當作等式的限制）";
  const list = s.W.length ? h("ul", { class: "wset" }, ...s.W.map((w) => h("li", {}, w))) : h("p", {}, "W = ∅（不受任何限制）");
  panel.replaceChildren(list,
    h("p", { class: "hint-text" }, `目前 x = (${fmt(s.x[0])}, ${fmt(s.x[1])})，目標值 ${fmt(s.z)}。`));
}

const ACTION = { branch: "分支", incumbent: "新的最佳解", prune: "以界限剪枝", infeasible: "不可行" };

function treePanel(k) {
  const step = st.res.steps[k];
  panelTitle.textContent = "分支樹（深度優先，先走 ≤ 的分支）";
  const nodes = step.tree;
  const kids = (id) => nodes.filter((n) => n.parent === id);   // in the order they were explored
  const num = (v) => String(Number(v.toFixed(3)));
  const item = (n) => {
    const lp = n.x ? `LP (${num(n.x[0])}, ${num(n.x[1])})，z = ${num(n.z)}` : "LP 不可行";
    const li = h("li", { class: n.id === step.node ? "cur" : "" },
      h("span", { class: "node-id" }, `節點 ${n.id}`), " ", h("span", { class: "node-label" }, n.label), h("br"),
      h("span", { class: "hint-text" }, lp), " ", h("span", { class: `badge badge-${n.action}` }, ACTION[n.action]));
    const sub = kids(n.id);
    if (sub.length) li.append(h("ul", {}, ...sub.map(item)));
    return li;
  };
  const best = step.best_z === null ? "尚無" : `(${num(step.incumbent[0])}, ${num(step.incumbent[1])})，z = ${num(step.best_z)}`;
  panel.replaceChildren(h("ul", { class: "bb-tree" }, ...kids(null).map(item)),
    h("p", { class: "hint-text" }, `現有最佳解：${best}。圖上深色區域是目前節點的子問題，綠圈是現有最佳解。`));
}

function cutPanel(k) {
  tableauPanel(k);
  panelTitle.textContent = "單純形表（加入割之後多一列、一欄）";
}

const PANELS = {
  simplex: tableauPanel, dual_simplex: tableauPanel, barrier: barrierPanel, active_set: activeSetPanel,
  branch_bound: treePanel, cutting_plane: cutPanel,
};

// ---------- drawing ----------
function last() { return st.res.steps.length - 1; }

function setFrame(k) {
  st.frame = Math.max(0, Math.min(last(), k));
  plot.setFrame(st.frame);
  PANELS[st.res.method](st.frame);
  const s = st.res.steps[st.frame];
  if (s.polygon || s.cuts) {
    const marks = st.res.method === "branch_bound"
      ? [{ x: s.x, kind: "lp", label: s.x ? "LP 最優" : "" }, { x: s.incumbent, kind: "best", label: "最佳解" }]
      : [{ x: s.x, kind: "lp", label: "目前的解" }];
    plot.setOverlay({ polygon: s.polygon, cuts: s.cuts || [], marks });
  }
  note.textContent = st.res.steps[st.frame].note;
  counter.textContent = `第 ${st.frame} / ${last()} 步`;
  prevBtn.disabled = st.frame === 0;
  nextBtn.disabled = st.frame === last();
}

function render() {
  plot.setGrid(st.res.grid);
  plot.setRegion(st.res.region);
  const steps = st.res.steps;
  // Integer methods: no path (dual-simplex points may lie outside the region); marks only.
  if (st.res.method === "branch_bound" || st.res.method === "cutting_plane") { plot.setSeries([]); setFrame(st.frame); return; }
  plot.setSeries([{
    xs: steps.map((s) => s.x), fs: steps.map((s) => s.z), status: st.res.status,
    label: { simplex: "單純形法", dual_simplex: "對偶單純形法", barrier: "內點法", active_set: "有效集法", cutting_plane: "割平面法" }[st.res.method],
    color: seriesColor(0), dots: true, big: steps.map((s) => s.central !== false),
  }]);
  setFrame(st.frame);
}

// ---------- animation ----------
let timer = null;
function play() {
  if (st.frame >= last()) setFrame(0);
  st.playing = true;
  playBtn.textContent = "⏸ 暫停";
  const tick = () => {
    if (!st.playing) return;
    if (st.frame >= last()) { pause(); return; }
    setFrame(st.frame + 1);
    timer = setTimeout(tick, 1000 / data.speed);
  };
  timer = setTimeout(tick, 1000 / data.speed);
}
// Callers of playThrough() (prediction cards) wait for the next pause:
// the end of the run, the pause button or an edit (finished tells which).
let pauseWaiters = [];
function pause() {
  st.playing = false; clearTimeout(timer); playBtn.textContent = "▶ 播放";
  const waiters = pauseWaiters;
  pauseWaiters = [];
  for (const fn of waiters) fn(st.frame >= last());
}
function playThrough() { return new Promise((resolve) => { pauseWaiters.push(resolve); play(); }); }

// ---------- live recompute through Pyodide ----------
let debounce = null;
function schedule() { st.version++; clearTimeout(debounce); debounce = setTimeout(recompute, 150); }

async function recompute() {
  const version = ++st.version;
  try {
    const res = await call("lp", st.spec);
    if (version !== st.version) return;
    pause();
    st.res = res;
    st.frame = last();
    render();
  } catch (err) {
    counter.textContent = `計算失敗：${err.message}`;
  }
}

// Back to the build-time coefficients and result; it is embedded in the page,
// so no Python call is needed.
function resetParams() {
  clearTimeout(debounce);
  st.version++;               // drop any recompute still in flight
  pause();
  st.spec = structuredClone(data.spec);
  st.res = data.result;
  st.frame = 0;
  for (const n of data.sliders) sliderSetters[n](initialValue(n));
  render();
}

// ---------- boot ----------
render();
if (data.sliders.length) warmUp().catch(() => {});

// ---------- prediction questions (frontmatter predict:) ----------
// Each scenario's result was computed at build time and is embedded, so no
// Python is needed: {same: true} is the page default; other LP results come
// without their objective grid, redrawn here from c (the objective is linear
// on every page with sliders).
function withGrid(res) {
  if (res.grid) return res;
  const g = data.result.grid;
  if (!g.linear) throw new Error("scenario result without grid on a quadratic objective");
  const [c1, c2] = res.region.c;
  return { ...res, grid: { ...g, z: g.y.map((y) => g.x.map((x) => g.sign * (c1 * x + c2 * y))) } };
}
function applyScenario(sc) {
  clearTimeout(debounce);
  st.version++;               // drop any recompute still in flight
  pause();
  st.spec = structuredClone(sc.spec);
  st.res = sc.result.same ? data.result : withGrid(sc.result);
  st.frame = 0;
  for (const n of data.sliders) sliderSetters[n](valueOf(st.spec, st.res, n));
  render();
  return Promise.resolve();
}
function cancelScenario() { st.version++; }
function showEnd() { pause(); setFrame(last()); }
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

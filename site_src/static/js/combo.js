// Combinatorial lab: one stepper, three views.
//   graph  – field locations with a tour (2-opt, tabu) or a growing tree (Kruskal)
//   dp     – value table V_t(s), filled backwards, then the optimal path
//   matrix – cost matrix with covering lines and the final assignment (Hungarian)
// First frame comes from build-time data; sliders re-run optlab.combo in Pyodide.

import { call, onState, warmUp } from "./py.js";

const data = JSON.parse(document.getElementById("combo-data").textContent);
const mount = document.getElementById("combo");
const num = (v) => String(Number(v.toFixed(2)));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const st = { spec: structuredClone(data.spec), res: data.result, frame: 0, playing: false, version: 0 };

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

// ---------- layout ----------
const pyStatus = h("span", { class: "py-status" });
const sliders = (data.sliders || []).map((sl) => {
  const value = st.spec.params[sl.name] ?? sl.default;
  const out = h("output", {}, String(value));
  const input = h("input", { type: "range", min: sl.lo, max: sl.hi, step: sl.step || 1, value, "aria-label": sl.label });
  input.addEventListener("input", () => {
    out.textContent = input.value;
    st.spec.params[sl.name] = Number(input.value);
    schedule();
  });
  return h("label", { class: "slider" }, sl.label, input, out);
});
const viewTitle = h("div", { class: "plot-title" });
const view = h("div", { class: "combo-view" });
const panelTitle = h("div", { class: "plot-title" });
const panel = h("div", { class: "lp-panel" });
const note = h("p", { class: "step-note", "aria-live": "polite" });
const counter = h("span", { class: "status-line" });
const prevBtn = h("button", { class: "btn", type: "button", onclick: () => { pause(); setFrame(st.frame - 1); } }, "◀ 上一步");
const playBtn = h("button", { class: "btn btn-primary", type: "button", onclick: () => (st.playing ? pause() : play()) }, "▶ 播放");
const nextBtn = h("button", { class: "btn", type: "button", onclick: () => { pause(); setFrame(st.frame + 1); } }, "下一步 ▶");
const resetBtn = h("button", { class: "btn", type: "button", onclick: () => { pause(); setFrame(0); } }, "⏮ 重來");

mount.append(h("div", { class: "widget lab lp-lab" },
  h("p", { class: "lp-problem" }, h("strong", {}, st.res.name), " ", h("span", { class: "hint-text" }, st.res.note)),
  sliders.length ? h("div", { class: "widget-row" }, ...sliders, pyStatus) : "",
  h("div", { class: `lab-plots${st.res.kind === "graph" ? "" : " combo-wide"}` }, h("div", {}, viewTitle, view), h("div", {}, panelTitle, panel)),
  note,
  h("div", { class: "widget-row" }, resetBtn, prevBtn, playBtn, nextBtn, counter)));

if (sliders.length) onState((s) => {
  pyStatus.textContent = { idle: "", loading: "Python 載入中…（調整參數會在載入後生效）", ready: "Python 已就緒", error: "Python 載入失敗，只能看預設動畫" }[s];
  pyStatus.classList.toggle("ready", s === "ready");
});

// ---------- graph view (canvas) ----------
const canvas = h("canvas", { role: "img", "aria-label": "田區位置與路線" });
let ctx = null, W = 0, H = 0;
function sizeCanvas() {
  const w = view.clientWidth;
  if (!w) return;
  W = w; H = Math.min(440, Math.round(w * 0.72));
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
const toPx = ([x, y]) => [18 + (x / 10) * (W - 36), H - 18 - (y / 7) * (H - 36)];

function line(c, p, q, color, width, dash) {
  const [a, b] = [toPx(p), toPx(q)];
  c.save();
  c.strokeStyle = color; c.lineWidth = width;
  if (dash) c.setLineDash(dash);
  c.beginPath(); c.moveTo(...a); c.lineTo(...b); c.stroke();
  c.restore();
}

function drawGraph(k) {
  if (!ctx) return;
  const c = ctx, pts = st.res.points, s = st.res.steps[k];
  c.fillStyle = css("--surface"); c.fillRect(0, 0, W, H);
  c.strokeStyle = css("--grid"); c.lineWidth = 1;
  for (let x = 0; x <= 10; x++) { const [px] = toPx([x, 0]); c.beginPath(); c.moveTo(px, 0); c.lineTo(px, H); c.stroke(); }
  for (let y = 0; y <= 7; y++) { const [, py] = toPx([0, y]); c.beginPath(); c.moveTo(0, py); c.lineTo(W, py); c.stroke(); }
  if (st.res.tour) {
    const t = s.tour, n = t.length;
    if (s.best && s.best_len < s.length - 1e-9) {
      for (let i = 0; i < n; i++) line(c, pts[s.best[i]], pts[s.best[(i + 1) % n]], css("--good"), 1.5, [5, 4]);
    }
    const seg = s.seg;
    for (let i = 0; i < n; i++) {
      // The two edges created by this step's reversal are highlighted.
      const fresh = seg && (i === seg[0] || i === seg[1]);
      line(c, pts[t[i]], pts[t[(i + 1) % n]], fresh ? css("--s2") : css("--s1"), fresh ? 3.5 : 2);
    }
  } else {
    for (const [u, v] of s.edges) line(c, pts[u], pts[v], css("--s1"), 3);
    if (s.consider) line(c, pts[s.consider[0]], pts[s.consider[1]], s.accepted ? css("--good") : css("--bad"), 2.5, s.accepted ? null : [6, 4]);
  }
  c.font = `12px ${css("--font")}`;
  c.textAlign = "center"; c.textBaseline = "middle";
  pts.forEach((p, i) => {
    const [x, y] = toPx(p);
    c.fillStyle = css("--surface"); c.strokeStyle = css("--ink-2"); c.lineWidth = 1.5;
    c.beginPath(); c.arc(x, y, 10, 0, 7); c.fill(); c.stroke();
    c.fillStyle = css("--ink"); c.fillText(String(i + 1), x, y + 0.5);
  });
}

function graphPanel(k) {
  const s = st.res.steps[k];
  if (!st.res.tour) {
    panelTitle.textContent = "已選的管線";
    const D = (u, v) => Math.hypot(st.res.points[u][0] - st.res.points[v][0], st.res.points[u][1] - st.res.points[v][1]);
    const total = s.edges.reduce((a, [u, v]) => a + D(u, v), 0);
    panel.replaceChildren(
      h("p", {}, `${s.edges.length} / ${st.res.points.length - 1} 條，總長 ${total.toFixed(2)} km`),
      h("p", { class: "hint-text" }, "藍線是已選的管線；綠線是剛選進來的，紅色虛線是因為會形成迴圈而跳過的。"));
    return;
  }
  panelTitle.textContent = st.res.method === "tabu" ? "目前路線、歷史最佳與禁忌表" : "路線長度";
  const kids = [h("p", {}, `目前 ${s.length.toFixed(3)} km`)];
  if (s.best_len !== undefined) {
    kids.push(h("p", {}, `歷史最佳 ${s.best_len.toFixed(3)} km`));
    kids.push(h("p", { class: "hint-text" }, "禁忌的邊（剩餘回合）："));
    kids.push(s.tabu.length
      ? h("ul", { class: "wset" }, ...s.tabu.map(([[u, v], left]) => h("li", {}, `${u + 1}–${v + 1}（${left}）`)))
      : h("p", { class: "hint-text" }, "（空）"));
    kids.push(h("p", { class: "hint-text" }, "綠色虛線是歷史最佳路線（比目前短時才畫）；橘色是這一步新換上的兩條邊。"));
  } else {
    kids.push(h("p", { class: "hint-text" }, "橘色是這一步新換上的兩條邊：反轉中間那一段，就等於把兩條交叉的邊換掉。"));
  }
  panel.replaceChildren(...kids);
}

// ---------- DP view (table) ----------
function drawDP(k) {
  const r = st.res, s = r.steps[k], T = r.months.length;
  const onPath = (t, sv) => s.path.some((p) => p.t === t && p.s === sv);
  const head = h("tr", {}, h("th", { scope: "col" }, "蓄水"), ...r.months.map((m, t) => h("th", { scope: "col", class: t === s.col ? "piv-col" : "" }, m)), h("th", { scope: "col" }, "結束"));
  const rows = [];
  for (let sv = r.S; sv >= 0; sv--) {
    const cells = [];
    for (let t = 0; t <= T; t++) {
      const v = s.V[t][sv], pol = t < T ? s.P[t][sv] : null;
      const cls = [t === s.col ? "piv-col" : "", onPath(t, sv) ? "piv-cell" : ""].join(" ");
      cells.push(h("td", { class: cls }, v === null ? "" : num(v), pol === null || pol === undefined ? "" : h("small", { class: "dp-r" }, ` r${pol}`)));
    }
    rows.push(h("tr", {}, h("th", { scope: "row" }, String(sv)), ...cells));
  }
  const foot = h("tr", { class: "obj-row" }, h("th", { scope: "row" }, "入流"), ...r.inflow.map((v) => h("td", {}, String(v))), h("td", {}, ""));
  const wrow = h("tr", {}, h("th", { scope: "row" }, "權重"), ...r.weight.map((v) => h("td", {}, String(v))), h("td", {}, ""));
  view.replaceChildren(h("div", { class: "tableau-wrap" }, h("table", { class: "tableau" }, h("thead", {}, head), h("tbody", {}, ...rows, foot, wrow))));
  viewTitle.textContent = "價值表 V（每格右下 r = 最佳放水量；框起來的是最佳路徑）";
  panelTitle.textContent = "放水計畫";
  panel.replaceChildren(s.path.length
    ? h("ul", { class: "wset" }, ...s.path.filter((p) => p.r !== null).map((p) => h("li", {}, `${r.months[p.t]}：蓄水 ${p.s}，放水 ${p.r}`)))
    : h("p", { class: "hint-text" }, "先由後往前填完整張表，再從 3 月、蓄水 2 照著 r 往前走。"));
}

// ---------- matrix view (Hungarian) ----------
function drawMatrix(k) {
  const r = st.res, s = r.steps[k];
  const covered = (i, j) => s.rows.includes(i) || s.cols.includes(j);
  const head = h("tr", {}, h("th", {}, ""), ...r.col_labels.map((c, j) => h("th", { scope: "col", class: s.cols.includes(j) ? "piv-col" : "" }, c)));
  const rows = s.M.map((row, i) => h("tr", { class: s.rows.includes(i) ? "piv-row" : "" },
    h("th", { scope: "row" }, r.row_labels[i]),
    ...row.map((v, j) => {
      const chosen = s.assign && s.assign[i] === j;
      const cls = [s.cols.includes(j) ? "piv-col" : "", chosen ? "piv-cell" : "", Math.abs(v) < 1e-9 ? "zero" : "", !covered(i, j) && s.rows.length + s.cols.length ? "uncov" : ""].join(" ");
      return h("td", { class: cls }, num(v));
    })));
  view.replaceChildren(h("div", { class: "tableau-wrap" }, h("table", { class: "tableau" }, h("thead", {}, head), h("tbody", {}, ...rows))));
  viewTitle.textContent = "矩陣（底色列／欄 = 蓋住 0 的直線；框起來 = 指派）";
  panelTitle.textContent = "說明";
  const n = s.M.length, lines = s.rows.length + s.cols.length;
  panel.replaceChildren(
    h("p", {}, lines ? `目前最少需要 ${lines} 條線（要 ${n} 條才能指派）` : "尚未畫線"),
    s.assign ? h("p", {}, `總時數 ${s.assign.reduce((a, j, i) => a + r.C[i][j], 0)}`) : "",
    h("p", { class: "hint-text" }, "粗體的 0 是可以考慮指派的位置。"));
}

// ---------- stepping ----------
const last = () => st.res.steps.length - 1;
function setFrame(k) {
  st.frame = Math.max(0, Math.min(last(), k));
  const kind = st.res.kind;
  if (kind === "graph") { drawGraph(st.frame); graphPanel(st.frame); }
  else if (kind === "dp") drawDP(st.frame);
  else drawMatrix(st.frame);
  note.textContent = st.res.steps[st.frame].note;
  counter.textContent = `第 ${st.frame} / ${last()} 步`;
  prevBtn.disabled = st.frame === 0;
  nextBtn.disabled = st.frame === last();
}

let timer = null;
function play() {
  if (st.frame >= last()) setFrame(0);
  st.playing = true; playBtn.textContent = "⏸ 暫停";
  const tick = () => {
    if (!st.playing) return;
    if (st.frame >= last()) { pause(); return; }
    setFrame(st.frame + 1);
    timer = setTimeout(tick, 1000 / data.speed);
  };
  timer = setTimeout(tick, 1000 / data.speed);
}
function pause() { st.playing = false; clearTimeout(timer); playBtn.textContent = "▶ 播放"; }

let debounce = null;
function schedule() { clearTimeout(debounce); debounce = setTimeout(recompute, 150); }
async function recompute() {
  const version = ++st.version;
  try {
    const res = await call("combo", st.spec);
    if (version !== st.version) return;
    pause();
    st.res = res;
    setFrame(last());
  } catch (err) {
    counter.textContent = `計算失敗：${err.message}`;
  }
}

// ---------- boot ----------
if (st.res.kind === "graph") {
  view.append(h("div", { class: "plot" }, canvas));
  viewTitle.textContent = "田區位置（格線間距 1 km）";
  new ResizeObserver(() => { sizeCanvas(); drawGraph(st.frame); }).observe(view);
  window.addEventListener("ot-theme", () => drawGraph(st.frame));
}
setFrame(0);
if (sliders.length) warmUp().catch(() => {});
new IntersectionObserver((entries, obs) => {
  if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); play(); }
}, { threshold: 0.4 }).observe(mount);

// 測試函數圖鑑: one contour plot, a menu of test functions, one info card at a time.
// The cards (formula, domain, minima, why it is hard) are static HTML from the
// build (KaTeX typesets them on load); the grids come from gallery-data.

import { ContourPlot } from "./plots.js";

const data = JSON.parse(document.getElementById("gallery-data").textContent);
const mount = document.getElementById("gallery");
const cards = Object.fromEntries([...mount.querySelectorAll(".tf-card")].map((c) => [c.dataset.fn, c]));

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

const KEY = "ot-gallery-fn";
const saved = (() => { try { return localStorage.getItem(KEY); } catch (e) { return null; } })();
let current = data.functions.some((f) => f.id === saved) ? saved : data.default;

const tabs = h("div", { class: "tf-tabs", role: "tablist", "aria-label": "測試函數" });
const buttons = {};
for (const fn of data.functions) {
  buttons[fn.id] = h("button", { type: "button", class: "tf-tab", role: "tab", id: `tf-tab-${fn.id}`,
    "aria-controls": `tf-${fn.id}`, onclick: () => show(fn.id, true) }, fn.name);
  tabs.append(buttons[fn.id]);
}
tabs.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  const ids = data.functions.map((f) => f.id);
  const i = (ids.indexOf(current) + (e.key === "ArrowRight" ? 1 : ids.length - 1)) % ids.length;
  show(ids[i], true);
  buttons[ids[i]].focus();
});

const plotBox = h("div", { class: "tf-plot" },
  h("div", { class: "plot-title" }, "對數等高線（等比例座標）：＋ 全域最小值、○ 局部最小值；滑鼠移到圖上（手機點一下）可讀 f 值"));
const cardBox = h("div", { class: "tf-info" });
mount.replaceChildren(tabs, h("div", { class: "tf-layout" }, plotBox, cardBox));
for (const c of Object.values(cards)) { c.setAttribute("role", "tabpanel"); cardBox.append(c); }
mount.classList.add("ready");

const plot = new ContourPlot(plotBox, { maxHeight: 460 });

// Only the default grid is in the page; the others are fetched once when chosen.
const grids = Object.fromEntries(data.functions.filter((f) => f.grid).map((f) => [f.id, f.grid]));
async function gridOf(fn) {
  if (!grids[fn.id]) {
    const res = await fetch(fn.src);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    grids[fn.id] = await res.json();
  }
  return grids[fn.id];
}
const status = h("p", { class: "hint-text", role: "status" });
plotBox.append(status);

async function show(id, remember) {
  current = id;
  const fn = data.functions.find((f) => f.id === id);
  for (const [k, b] of Object.entries(buttons)) {
    b.setAttribute("aria-selected", String(k === id));
    b.tabIndex = k === id ? 0 : -1;
  }
  for (const [k, c] of Object.entries(cards)) {
    c.hidden = k !== id;
    c.setAttribute("aria-labelledby", `tf-tab-${k}`);
  }
  try {
    status.textContent = grids[id] ? "" : "載入等高線資料…";
    const grid = await gridOf(fn);
    if (current !== id) return;           // another function was chosen meanwhile
    status.textContent = "";
    plot.setGrid(grid);
    plot.setSeries([]);
  } catch (err) {
    if (current === id) status.textContent = `等高線資料載入失敗（${err.message}）；右側說明仍可閱讀。`;
  }
  if (remember) { try { localStorage.setItem(KEY, id); } catch (e) { /* no storage */ } }
}
show(current, false);

// review.html: today's queue (due recall cards mixed with practice redos),
// the per-family progress table, and export / import / clear.

import { renderMath } from "./common.js";
import * as P from "./progress.js";
import { dueText, h, recallCard } from "./recall.js";
import { masteryBar } from "./game.js";

const data = JSON.parse(document.getElementById("review-data").textContent);
const cardsById = Object.fromEntries(data.cards.map((c) => [c.id, c]));
const queueEl = document.getElementById("queue");
const countEl = document.getElementById("queue-count");
const status = document.getElementById("io-status");

const pageHref = (id) => `algorithms/${id}.html`;

// ---------- queue ----------
/**
 * Due cards are interleaved across algorithms (mixed practice beats blocked
 * practice for later recall), with one practice redo after every 3 cards.
 */
function buildQueue(state) {
  const day = P.today();
  const groups = new Map();
  for (const id of P.dueCards(day, state)) {
    const c = cardsById[id];
    if (!c) continue;  // card no longer exists in the content
    if (!groups.has(c.algo)) groups.set(c.algo, []);
    groups.get(c.algo).push(c);
  }
  const cards = [];
  const lists = [...groups.values()];
  for (let i = 0; lists.some((l) => i < l.length); i++) {
    for (const l of lists) if (i < l.length) cards.push({ kind: "card", card: l[i] });
  }
  const practice = P.duePractice(day, state)
    .filter((id) => data.algos[id]?.practice)
    .map((id) => ({ kind: "practice", id, info: P.practiceDue(id, state) }));
  const out = [];
  while (cards.length || practice.length) {
    out.push(...cards.splice(0, 3));
    if (practice.length) out.push(practice.shift());
  }
  return out;
}

function nextDue(state) {
  const day = P.today();
  const dates = Object.entries(state.cards).filter(([id]) => cardsById[id]).map(([, c]) => c.due)
    .concat(Object.keys(state.practice).map((id) => P.practiceDue(id, state)?.due).filter(Boolean))
    .filter((d) => d > day).sort();
  return dates.length ? { due: dates[0], n: dates.filter((d) => d === dates[0]).length } : null;
}

function practiceItem(item) {
  const a = data.algos[item.id];
  const reason = item.info.reason === "fail"
    ? "上次練習沒有全過，趁記憶還新再試一次。"
    : `上次通過是 ${P.dayStr(item.info.lastPass).slice(5).replace("-", "/")}，隔一段時間再做一次，記得更牢。`;
  return h("section", { class: "review-practice", "aria-label": `重做練習：${a.name}` },
    h("span", { class: "review-kind" }, "練習"),
    h("div", { class: "review-practice-body" },
      h("strong", {}, `重做練習：${a.name}`),
      h("p", { class: "hint-text" }, reason)),
    h("a", { class: "btn btn-primary", href: `${pageHref(item.id)}#practice` }, "去做練習"));
}

function renderQueue() {
  const state = P.load();
  const items = buildQueue(state);
  queueEl.replaceChildren();
  let graded = 0;
  const setCount = () => {
    countEl.textContent = items.length ? `（${items.length} 項${graded ? `，已自評 ${graded} 張` : ""}）` : "";
  };
  setCount();
  if (!items.length) {
    const nd = nextDue(state);
    const anyCards = Object.keys(state.cards).length > 0;
    queueEl.append(h("div", { class: "review-empty" },
      h("p", { class: "review-empty-title" }, "今天沒有到期的項目。"),
      h("p", {}, nd ? `下一次到期：${dueText(nd.due)}，共 ${nd.n} 項。`
        : anyCards ? "" : "到任何一頁的「面試追問」試答幾題，自評後題目就會依間隔回到這裡。"),
      h("p", {}, h("a", { href: "index.html" }, "回目錄挑一個演算法"))));
    return;
  }
  for (const item of items) {
    if (item.kind === "practice") { queueEl.append(practiceItem(item)); continue; }
    const c = item.card;
    const a = data.algos[c.algo];
    let done = false;
    const el = recallCard({
      cardId: c.id, question: c.q_html, answer: c.a_html,
      source: { href: pageHref(c.algo), text: `${a.name}・第 ${c.n} 題` },
      onGraded: () => {
        if (!done) { done = true; graded++; setCount(); }
        el.classList.add("done");
      },
    });
    queueEl.append(el);
  }
  if (!renderMath(queueEl)) window.addEventListener("load", () => renderMath(queueEl), { once: true });
}

// ---------- progress table ----------
function renderTable() {
  const state = P.load();
  const fams = {};
  for (const [id, a] of Object.entries(data.algos)) {
    const lv = P.mastery(id, a.cards, state).level;
    (fams[a.family] ||= [0, 0, 0, 0, 0])[lv]++;
  }
  const levels = [0, 1, 2, 3];
  const total = [0, 0, 0, 0, 0];
  const rows = Object.entries(data.families).filter(([f]) => fams[f]).map(([f, name]) => {
    fams[f].forEach((n, i) => { total[i] += n; });
    return h("tr", {}, h("th", { scope: "row" }, name),
      ...levels.map((l) => h("td", { class: fams[f][l] ? "" : "zero-count" }, String(fams[f][l]))));
  });
  const table = h("table", { class: "progress-table" },
    h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "族群"),
      ...levels.map((l) => h("th", { scope: "col" }, masteryBar(l), h("span", {}, P.LEVELS[l]))))),
    h("tbody", {}, ...rows),
    h("tfoot", {}, h("tr", {}, h("th", { scope: "row" }, "合計"), ...levels.map((l) => h("td", {}, String(total[l]))))));
  document.getElementById("progress-table").replaceChildren(h("div", { class: "table-wrap" }, table));
  if (!P.storageAvailable()) {
    status.textContent = "這個瀏覽器無法使用本機儲存，進度只保留到關閉分頁為止；記得匯出備份。";
  }
}

// ---------- export / import / clear ----------
document.getElementById("export-btn").addEventListener("click", () => {
  const text = P.exportJSON();
  const name = `optimization-tutor-progress-${P.today()}.json`;
  try {
    const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    const a = h("a", { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = `已匯出 ${name}。若沒有開始下載，可展開「貼上 JSON 匯入」複製下方內容保存。`;
  } catch (e) {
    status.textContent = "無法下載檔案，已把內容放到下方文字框，請自行複製保存。";
  }
  const box = document.getElementById("import-text");
  box.value = text;
});

function afterImport(res) {
  if (res.ok) {
    status.textContent = `匯入完成：${Object.keys(res.state.cards).length} 張卡片、${Object.keys(res.state.practice).length} 題練習紀錄。`;
    renderQueue();
  } else {
    status.textContent = `匯入失敗：${res.error} 目前的進度沒有變動。`;
  }
}

document.getElementById("import-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  afterImport(P.importJSON(await file.text()));
  e.target.value = "";
});
document.getElementById("import-paste").addEventListener("click", () => {
  afterImport(P.importJSON(document.getElementById("import-text").value));
});

// Two-step confirmation inside the page (embedded viewers block confirm()).
const clearBtn = document.getElementById("clear-btn");
let armed = false, timer = null;
clearBtn.addEventListener("click", () => {
  if (!armed) {
    armed = true;
    clearBtn.textContent = "再按一次確認清除";
    clearBtn.classList.add("btn-danger");
    timer = setTimeout(() => { armed = false; clearBtn.textContent = "清除全部進度"; clearBtn.classList.remove("btn-danger"); }, 4000);
    return;
  }
  clearTimeout(timer);
  armed = false;
  clearBtn.textContent = "清除全部進度";
  clearBtn.classList.remove("btn-danger");
  P.clearAll();
  status.textContent = "已清除全部進度。";
  renderQueue();
});

renderQueue();
renderTable();
window.addEventListener("ot-progress", renderTable);
window.addEventListener("storage", (e) => { if (e.key === P.KEY) { renderQueue(); renderTable(); } });

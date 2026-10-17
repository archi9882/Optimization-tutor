// Progress layer shared by every page:
//   - nav 「複習」 badge with the number of due items
//   - algorithm pages: record the visit, mastery indicator, recall cards
//   - home page: a small mastery ring on each algorithm card
// No points for opening pages: a visit only moves 未開始 -> 看過.

import * as P from "./progress.js";
import { enhanceDetails, h } from "./recall.js";
import { CARD_COUNTS, PRACTICE } from "./recall-index.js";

// ---------- nav badge ----------
/** True if a stored card id still exists in the built content. */
export function knownCard(id) {
  const m = /^(.+)#q(\d+)$/.exec(id);
  return !!m && Number(m[2]) >= 1 && Number(m[2]) <= (CARD_COUNTS[m[1]] || 0);
}

export function dueCount(state = P.load()) {
  return P.dueCards(P.today(), state).filter(knownCard).length +
    P.duePractice(P.today(), state).filter((id) => PRACTICE.includes(id)).length;
}

function renderBadge() {
  const badge = document.querySelector(".nav-badge");
  if (!badge) return;
  const n = dueCount();
  badge.hidden = n === 0;
  badge.textContent = String(n);
  badge.closest("a")?.setAttribute("aria-label", n ? `複習（${n} 項到期）` : "複習");
}

// ---------- mastery indicator ----------
export function masteryBar(level) {
  const bar = h("span", { class: "mastery-bar", "aria-hidden": "true" });
  for (let i = 1; i <= 4; i++) bar.append(h("i", { class: i <= level ? "on" : i === 4 ? "future" : "" }));
  return bar;
}

function nextStep(m, hasPractice) {
  if (m.level >= 3) return "精通需要模擬面試，之後開放。";
  if (m.demoted && m.spaced && m.ratio >= 0.7) return "最近一次練習沒全過，再通過一次就回到熟練。";
  if (m.level < 2) return hasPractice ? "下一步：在「挖空練習」通過三層測試。" : "這頁還沒有練習題。";
  const todo = [];
  if (!m.spaced) {
    const from = P.addDays(m.firstPass, 2);
    todo.push(from <= P.today() ? "再通過一次練習" : `${from.slice(5).replace("-", "/")} 以後再通過一次練習`);
  }
  if (m.ratio < 0.7) todo.push(`面試追問答「會」達七成（目前 ${m.known}/${m.cardCount}）`);
  return `下一步：${todo.join("，並且")}。`;
}

function renderMastery(el, algoId, cardCount, hasPractice) {
  const m = P.mastery(algoId, cardCount);
  el.replaceChildren(
    masteryBar(m.level),
    h("span", { class: "mastery-label" }, "熟練度：", h("strong", {}, m.label)),
    h("span", { class: "mastery-next" }, nextStep(m, hasPractice)));
  el.dataset.level = String(m.level);
  el.hidden = false;
}

// ---------- algorithm page ----------
const article = document.querySelector("article.algo[data-algo]");
if (article && article.dataset.detailed === "1") {
  const id = article.dataset.algo;
  P.recordVisit(id);
  const cardCount = enhanceDetails(article);
  const hasPractice = !!document.getElementById("practice");
  const el = document.getElementById("mastery");
  const draw = () => el && renderMastery(el, id, cardCount, hasPractice);
  draw();
  window.addEventListener("ot-progress", draw);
  window.addEventListener("storage", (e) => { if (e.key === P.KEY) draw(); });
}

// ---------- home page ----------
const cardsData = document.getElementById("cards-data");
if (cardsData) {
  const cards = JSON.parse(cardsData.textContent);
  const byId = Object.fromEntries(cards.map((c) => [c.id, c]));
  const drawRings = () => {
    const state = P.load();
    for (const el of document.querySelectorAll(".card[data-id]")) {
      const c = byId[el.dataset.id];
      el.querySelector(".mring")?.remove();
      if (!c || !c.status.page) continue;
      const m = P.mastery(c.id, c.cards || 0, state);
      if (m.level === 0) continue;
      el.prepend(h("span", {
        class: `mring lv${m.level}`, role: "img", "aria-label": `熟練度：${m.label}`, title: `熟練度：${m.label}`,
        style: `--lv:${m.level}`,
      }, h("span", { class: "mring-label", "aria-hidden": "true" }, m.label)));
    }
  };
  drawRings();
  window.addEventListener("ot-progress", drawRings);
  window.addEventListener("storage", (e) => { if (e.key === P.KEY) drawRings(); });
}

renderBadge();
window.addEventListener("ot-progress", renderBadge);
window.addEventListener("storage", (e) => { if (e.key === P.KEY) renderBadge(); });

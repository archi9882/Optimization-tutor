// Recall card: question visible, answer hidden until the learner has tried,
// then a self-grade (忘了 / 模糊 / 會) that feeds the Leitner schedule.
// Used on algorithm pages (built from the page's <details>) and on review.html.

import { GRADE_LABELS, cardState, daysBetween, gradeCard, today } from "./progress.js";

const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...kids);
  return el;
};
export { h };

/** "10/19（2 天後）" */
export function dueText(due) {
  const n = daysBetween(today(), due);
  const md = due.slice(5).replace("-", "/").replace(/^0/, "");
  if (n < 0) return `${md}（已到期）`;
  if (n === 0) return `${md}（今天）`;
  if (n === 1) return `${md}（明天）`;
  return `${md}（${n} 天後）`;
}

function metaText(c) {
  const last = c && c.history[c.history.length - 1];
  if (!last) return "還沒複習過這題。";
  return `上次：${GRADE_LABELS[last.grade]}・下次複習 ${dueText(c.due)}`;
}

/**
 * Build one card.
 *   cardId   "algoId#qN"
 *   question Node | string (HTML), answer Node | string (HTML)
 *   source   optional {href, text} link shown above the question
 *   onGraded optional callback(cardState)
 */
export function recallCard({ cardId, question, answer, source = null, onGraded = null, headingLevel = 3 }) {
  const slug = `rc-${cardId.replace(/[^A-Za-z0-9]+/g, "-")}`;
  const fill = (el, x) => { if (typeof x === "string") el.innerHTML = x; else el.append(x); return el; };

  const q = fill(h("div", { class: "recall-q", role: "heading", "aria-level": String(headingLevel), id: `${slug}-q` }), question);
  const meta = h("p", { class: "recall-meta" }, metaText(cardState(cardId)));
  const note = h("textarea", { class: "recall-note", id: `${slug}-note`, rows: "2", spellcheck: "false",
    placeholder: "先用自己的話寫下答案（選填，不評分）" });
  const ans = fill(h("div", { class: "recall-a", id: `${slug}-a`, hidden: true, tabindex: "-1" }), answer);
  const reveal = h("button", { class: "btn recall-reveal", type: "button", "aria-expanded": "false", "aria-controls": `${slug}-a` },
    "先想一想，再看答案");
  const result = h("p", { class: "recall-result", "aria-live": "polite" });
  const btns = GRADE_LABELS.map((label, g) => h("button", {
    class: `btn grade-btn grade-${g}`, type: "button", "aria-pressed": "false", "data-grade": String(g),
  }, label));
  const grade = h("div", { class: "recall-grade", role: "group", "aria-label": "自評：這題你答得出來嗎？", hidden: true },
    h("span", { class: "recall-ask" }, "你答得出來嗎？"), ...btns);

  reveal.addEventListener("click", () => {
    const open = ans.hidden;
    ans.hidden = !open;
    reveal.setAttribute("aria-expanded", String(open));
    reveal.textContent = open ? "收起答案" : "先想一想，再看答案";
    if (open) { grade.hidden = false; ans.focus({ preventScroll: true }); }
  });

  for (const b of btns) {
    b.addEventListener("click", () => {
      const g = Number(b.dataset.grade);
      const before = cardState(cardId);
      const after = gradeCard(cardId, g);
      for (const o of btns) o.setAttribute("aria-pressed", String(o === b));
      const held = g === 2 && before && before.due === after.due && before.box === after.box;
      result.textContent = held
        ? `記下了。這題還沒到期，維持 ${dueText(after.due)} 複習。`
        : `記下「${GRADE_LABELS[g]}」，下次複習：${dueText(after.due)}。`;
      meta.textContent = metaText(after);
      if (onGraded) onGraded(after, g);
    });
  }

  const card = h("section", { class: "recall-card", "data-card": cardId, "aria-labelledby": `${slug}-q` });
  if (source) card.append(h("a", { class: "recall-src", href: source.href }, source.text));
  card.append(q, meta, note, h("div", { class: "recall-actions" }, reveal), ans, grade, result);
  return card;
}

/** Replace every <details class="recall"> inside `scope` with a card. */
export function enhanceDetails(scope = document) {
  const list = [...scope.querySelectorAll("details.recall[data-card]")];
  for (const d of list) {
    const summary = d.querySelector(":scope > summary");
    const qFrag = document.createDocumentFragment();
    qFrag.append(...summary.childNodes);
    summary.remove();
    const aFrag = document.createDocumentFragment();
    aFrag.append(...d.childNodes);
    d.replaceWith(recallCard({ cardId: d.dataset.card, question: qFrag, answer: aFrag }));
  }
  return list.length;
}

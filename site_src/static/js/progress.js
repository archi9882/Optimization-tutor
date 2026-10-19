// Learner progress, stored client-side under one localStorage key.
//
//   {v: 1,
//    visits:   {algoId: ts},                          last visit
//    practice: {algoId: {passes: [ts...], lastFail: ts}},
//    cards:    {cardId: {box: 0..4, due: "YYYY-MM-DD", history: [{ts, grade}]}},
//    week:     {}}                                    reserved (weekly goals)
//
// Mastery is never stored: it is recomputed from these records, so a failed
// attempt can move a learner back down. Every storage access is wrapped in
// try/catch; when storage is unavailable the state lives in memory only.

export const KEY = "ot.progress.v1";
export const VERSION = 1;
export const INTERVALS = [1, 2, 7, 21, 60];        // Leitner box -> days
export const PRACTICE_INTERVALS = [2, 7, 21, 60];  // n-th distinct pass day -> days
export const GRADE_LABELS = ["忘了", "模糊", "會"];
export const LEVELS = ["未開始", "看過", "熟悉", "熟練", "精通"]; // 4 reserved (模擬面試)
const HISTORY_CAP = 50;
const DAY = 86400000;

let memory = null;          // fallback copy when storage is unavailable
let storageOk = true;

export const empty = () => ({ v: VERSION, visits: {}, practice: {}, cards: {}, week: {} });

// ---------- dates (local calendar days) ----------
export function dayStr(ts = Date.now()) {
  const d = new Date(ts);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}
export const today = () => dayStr();
export function addDays(day, n) {
  const [y, m, d] = day.split("-").map(Number);
  return dayStr(new Date(y, m - 1, d + n, 12).getTime());
}
/** Whole calendar days from day a to day b (b - a). */
export function daysBetween(a, b) {
  const t = (s) => { const [y, m, d] = s.split("-").map(Number); return Date.UTC(y, m - 1, d); };
  return Math.round((t(b) - t(a)) / DAY);
}

// ---------- load / save ----------
export function storageAvailable() { return storageOk; }

/** Try a throwaway write, for pages that only read (the home page). */
export function probeStorage() {
  try {
    localStorage.setItem(`${KEY}.probe`, "1");
    localStorage.removeItem(`${KEY}.probe`);
  } catch (e) {
    storageOk = false;
  }
  return storageOk;
}

export function load() {
  if (!storageOk && memory) return structuredClone(memory);
  let raw = null;
  try {
    raw = localStorage.getItem(KEY);
  } catch (e) {
    storageOk = false;
    return memory ? structuredClone(memory) : empty();
  }
  if (raw == null) return empty();
  const parsed = validate(safeParse(raw));
  if (parsed.ok) return parsed.state;
  // Keep the unreadable copy aside instead of silently overwriting it.
  try { localStorage.setItem(`${KEY}.corrupt`, raw); } catch (e) { /* ignore */ }
  return empty();
}

export function save(state) {
  memory = structuredClone(state);
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
    storageOk = true;
  } catch (e) {
    storageOk = false;
  }
  try { window.dispatchEvent(new CustomEvent("ot-progress")); } catch (e) { /* no window */ }
}

function update(fn) {
  const s = load();
  const out = fn(s);
  save(s);
  return out;
}

// ---------- records ----------
export function recordVisit(id, ts = Date.now()) {
  update((s) => { s.visits[id] = ts; });
}

export function recordPractice(id, passed, ts = Date.now()) {
  update((s) => {
    const p = s.practice[id] || (s.practice[id] = { passes: [], lastFail: 0 });
    if (passed) p.passes.push(ts);
    else p.lastFail = ts;
  });
}

/**
 * Self-grade a recall card: 0 忘了, 1 模糊, 2 會.
 *   會   -> up one box (only when the card is new or due, so repeated
 *           same-day grading cannot inflate the box), due = today + interval
 *   模糊 -> same box, due tomorrow
 *   忘了 -> box 0, due tomorrow
 */
export function gradeCard(cardId, grade, ts = Date.now()) {
  if (![0, 1, 2].includes(grade)) throw new Error(`bad grade ${grade}`);
  return update((s) => {
    const t = dayStr(ts);
    const old = s.cards[cardId];
    const c = old ? { ...old, history: [...old.history] } : { box: 0, due: t, history: [] };
    if (grade === 2) {
      if (!old || old.due <= t) {
        c.box = old ? Math.min(c.box + 1, INTERVALS.length - 1) : 1;
        c.due = addDays(t, INTERVALS[c.box]);
      }
    } else {
      if (grade === 0) c.box = 0;
      c.due = addDays(t, 1);
    }
    c.history.push({ ts, grade });
    if (c.history.length > HISTORY_CAP) c.history = c.history.slice(-HISTORY_CAP);
    s.cards[cardId] = c;
    return { ...c };
  });
}

export function cardState(cardId, state = load()) {
  return state.cards[cardId] || null;
}

/** Card ids due on or before `day`, oldest due first. */
export function dueCards(day = today(), state = load()) {
  return Object.entries(state.cards)
    .filter(([, c]) => c.due <= day)
    .sort((a, b) => (a[1].due < b[1].due ? -1 : a[1].due > b[1].due ? 1 : 0))
    .map(([id]) => id);
}

/** When the practice for `id` should be redone, or null if never passed. */
export function practiceDue(id, state = load()) {
  const p = state.practice[id];
  if (!p || !p.passes.length) return null;
  const last = Math.max(...p.passes);
  if (p.lastFail > last) return { due: dayStr(p.lastFail), reason: "fail", lastPass: last };
  const days = new Set(p.passes.map((t) => dayStr(t))).size;
  const gap = PRACTICE_INTERVALS[Math.min(days - 1, PRACTICE_INTERVALS.length - 1)];
  return { due: addDays(dayStr(last), gap), reason: "spaced", lastPass: last };
}

export function duePractice(day = today(), state = load()) {
  return Object.keys(state.practice).filter((id) => {
    const d = practiceDue(id, state);
    return d && d.due <= day;
  });
}

/**
 * Mastery of one algorithm, computed from the records.
 *   0 未開始  nothing recorded
 *   1 看過    page visited (or any activity)
 *   2 熟悉    practice passed (all layers) at least once
 *   3 熟練    a pass >= 2 days after the first pass AND >= 70% of the
 *             page's recall cards last graded 會; a failed attempt after the
 *             last pass drops it back to 熟悉
 *   4 精通    reserved for the future mock interview
 */
export function mastery(id, cardCount = 0, state = load()) {
  const p = state.practice[id];
  const passes = p ? [...p.passes].sort((a, b) => a - b) : [];
  let known = 0;
  for (let i = 1; i <= cardCount; i++) {
    const c = state.cards[`${id}#q${i}`];
    // Counts only if the card has climbed at least one box: a 會 pressed right
    // after 忘了 on the same day is not yet remembered knowledge.
    if (c && c.box >= 1 && c.history.length && c.history[c.history.length - 1].grade === 2) known++;
  }
  const ratio = cardCount ? known / cardCount : 1;
  const touched = id in state.visits || !!p ||
    Object.keys(state.cards).some((k) => k.startsWith(`${id}#`));
  const first = passes.length ? dayStr(passes[0]) : null;
  const spaced = passes.some((t) => daysBetween(first, dayStr(t)) >= 2);
  const demoted = !!(p && passes.length && p.lastFail > passes[passes.length - 1]);
  let level = touched ? 1 : 0;
  if (passes.length) level = 2;
  if (level === 2 && spaced && ratio >= 0.7 && !demoted) level = 3;
  return {
    level, label: LEVELS[level], passes: passes.length, firstPass: first,
    spaced, demoted, known, cardCount, ratio,
  };
}

// ---------- export / import ----------
export function exportJSON(state = load()) {
  return JSON.stringify({ ...state, exported: new Date().toISOString() }, null, 2);
}

function safeParse(text) {
  try { return JSON.parse(text); } catch (e) { return undefined; }
}

const isObj = (o) => o !== null && typeof o === "object" && !Array.isArray(o);
// Timestamps must be real and not in the future (one day of clock skew allowed).
const isTs = (t) => typeof t === "number" && Number.isFinite(t) && t >= 0 && t <= Date.now() + 864e5;
const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
// A calendar day that exists (rejects 9999-99-99, 0000-00-00) within a sane range.
const isDay = (d) => typeof d === "string" && DAY_RE.test(d) && d >= "2000-01-01" && d <= "2200-12-31" &&
  dayStr(new Date(`${d}T12:00:00`).getTime()) === d;

/** Validate an untrusted object; returns {ok, state} or {ok: false, error}. */
export function validate(obj) {
  const fail = (error) => ({ ok: false, error });
  if (!isObj(obj)) return fail("不是有效的 JSON 物件。");
  if (obj.v !== VERSION) return fail(`版本不符（需要 v=${VERSION}）。`);
  const out = empty();
  for (const k of ["visits", "practice", "cards"]) {
    if (obj[k] !== undefined && !isObj(obj[k])) return fail(`「${k}」格式錯誤。`);
  }
  for (const [id, ts] of Object.entries(obj.visits || {})) {
    if (!isTs(ts)) return fail(`visits.${id} 不是時間戳記。`);
    out.visits[id] = ts;
  }
  for (const [id, p] of Object.entries(obj.practice || {})) {
    if (!isObj(p) || !Array.isArray(p.passes) || !p.passes.every(isTs)) return fail(`practice.${id} 格式錯誤。`);
    const lastFail = p.lastFail ?? 0;
    if (!isTs(lastFail)) return fail(`practice.${id}.lastFail 格式錯誤。`);
    out.practice[id] = { passes: [...p.passes], lastFail };
  }
  for (const [id, c] of Object.entries(obj.cards || {})) {
    if (!isObj(c) || !Number.isInteger(c.box) || c.box < 0 || c.box >= INTERVALS.length ||
        !isDay(c.due) || !Array.isArray(c.history) || c.history.length === 0 ||
        !c.history.every((h) => isObj(h) && isTs(h.ts) && [0, 1, 2].includes(h.grade))) {
      return fail(`cards.${id} 格式錯誤。`);
    }
    out.cards[id] = { box: c.box, due: c.due, history: c.history.map((h) => ({ ts: h.ts, grade: h.grade })) };
  }
  if (isObj(obj.week)) out.week = structuredClone(obj.week);
  return { ok: true, state: out };
}

/** Replace the stored progress with `text` (JSON) after validation. */
export function importJSON(text) {
  const parsed = safeParse(text);
  if (parsed === undefined) return { ok: false, error: "無法解析 JSON。" };
  const res = validate(parsed);
  if (res.ok) save(res.state);
  return res;
}

export function clearAll() {
  save(empty());
}

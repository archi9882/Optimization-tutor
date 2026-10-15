// Main-thread client for the Pyodide worker: one shared worker per page,
// promise-based calls, and a timeout that restarts the worker.

const cfg = window.OT_CONFIG;
let worker = null;
let ready = null;
let seq = 0;
const pending = new Map();
const listeners = new Set();
let state = "idle";

function setState(s) {
  state = s;
  for (const fn of listeners) fn(s);
}

function spawn() {
  worker = new Worker(`${cfg.root}/static/js/py-worker.js`);
  worker.onmessage = (e) => {
    const { id, ok, result, error } = e.data;
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    clearTimeout(p.timer);
    ok ? p.resolve(result) : p.reject(new Error(error));
  };
  setState("loading");
  // Absolute URLs: the worker resolves relative paths against its own script.
  const abs = (u) => new URL(u, location.href).href;
  ready = send("init", { pyodideBase: abs(cfg.pyodideBase), root: abs(cfg.root).replace(/\/$/, "") }, 120000)
    .then(() => setState("ready"))
    .catch((err) => { setState("error"); throw err; });
  return ready;
}

function send(type, payload, timeoutMs) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      if (type === "init") { reject(new Error("Pyodide 載入逾時")); return; }
      // A stuck worker cannot be interrupted: kill it and start fresh.
      worker.terminate();
      for (const p of pending.values()) { clearTimeout(p.timer); p.reject(new Error("worker restarted")); }
      pending.clear();
      spawn();
      reject(new Error(`執行逾時（超過 ${timeoutMs / 1000} 秒），可能有無窮迴圈。`));
    }, timeoutMs);
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ id, type, payload });
  });
}

/** Start loading Pyodide in the background (idempotent). */
export function warmUp() {
  if (!worker) spawn();
  return ready;
}

/** Call optlab.api: "grid" | "traces" | "grade". */
export async function call(type, payload, timeoutMs = 10000) {
  warmUp();
  await ready;
  return send(type, payload, timeoutMs);
}

/** Subscribe to "loading" | "ready" | "error"; fires immediately. */
export function onState(fn) {
  listeners.add(fn);
  fn(state);
}

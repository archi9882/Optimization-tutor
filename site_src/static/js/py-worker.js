// Web Worker hosting Pyodide + optlab, so Python never blocks the page and
// a runaway learner loop can be killed by terminating the worker.

let pyReady = null;

async function boot(pyodideBase, root) {
  importScripts(pyodideBase + "pyodide.js");
  const pyodide = await loadPyodide({ indexURL: pyodideBase });
  const files = await (await fetch(`${root}/py/manifest.json`)).json();
  for (const rel of files) {
    const text = await (await fetch(`${root}/py/${rel}`)).text();
    const dir = rel.split("/").slice(0, -1);
    let path = "/home/pyodide";
    for (const part of dir) {
      path += "/" + part;
      try { pyodide.FS.mkdir(path); } catch (e) { /* exists */ }
    }
    pyodide.FS.writeFile(`/home/pyodide/${rel}`, text);
  }
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide'); from optlab import api");
  return pyodide;
}

self.onmessage = async (e) => {
  const { id, type, payload } = e.data;
  try {
    if (type === "init") {
      pyReady = pyReady || boot(payload.pyodideBase, payload.root);
      await pyReady;
      self.postMessage({ id, ok: true, result: "ready" });
      return;
    }
    const pyodide = await pyReady;
    if (payload.source) {
      // Learner code may import numpy etc.; fetch those packages on demand.
      await pyodide.loadPackagesFromImports(payload.source);
    }
    pyodide.globals.set("_name", type);
    pyodide.globals.set("_payload", JSON.stringify(payload));
    const out = pyodide.runPython("api.call(_name, _payload)");
    self.postMessage({ id, ok: true, result: JSON.parse(out) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message || err) });
  }
};

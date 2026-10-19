"""Build the static site into _site/.

    python scripts/build.py            # build
    python scripts/build.py --serve    # build, then serve on localhost:8000
    python scripts/build.py --vendor DIR --out DIR   # self-hosted libraries

`--vendor DIR` expects DIR/pyodide (the npm "pyodide" package) and DIR/katex
(the npm "katex" package). Their files are copied into the site so it works
where pages may not fetch from CDNs (e.g. a claude.ai Artifact).

Inputs:  content/catalog.yaml, content/algorithms/*.md, site_src/, optlab/
Output:  _site/ (ready for GitHub Pages), including review.html, whose
         recall cards are extracted here from each page's 面試追問 section.
"""
from __future__ import annotations

import argparse
import html as htmlmod
import http.server
import json
import re
import shutil
import sys
from functools import partial
from pathlib import Path

import markdown
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optlab import OPTIMIZERS, api, demos, make_problem, lp as lpmod, predict as predmod  # noqa: E402
from optlab.optimizers import N_ITER, PROBLEM_PARAMS, Param  # noqa: E402
from optlab.exercises import EXERCISES  # noqa: E402

CONTENT = ROOT / "content"
SRC = ROOT / "site_src"
OUT = ROOT / "_site"

PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/"
KATEX_BASE = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/"

MATH_RE = re.compile(r"(\$\$.+?\$\$|\$[^$\n]+?\$)", re.S)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split '---\\nyaml\\n---\\nbody' into (meta, body)."""
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        return yaml.safe_load(fm) or {}, body
    return {}, text


def render_markdown(body: str) -> str:
    """Markdown -> HTML, protecting $...$ / $$...$$ from the parser."""
    stash: list[str] = []

    def keep(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"\x00M{len(stash) - 1}\x00"

    protected = MATH_RE.sub(keep, body)
    html = markdown.markdown(protected, extensions=[
        "tables", "fenced_code", "md_in_html", "toc"])
    escape = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
    return re.sub(r"\x00M(\d+)\x00",
                  lambda m: "".join(escape.get(c, c)
                                    for c in stash[int(m.group(1))]), html)


RECALL_HEADING = "面試追問"
RECALL_SECTION_RE = re.compile(
    r"(<h2[^>]*>\s*" + RECALL_HEADING + r"\s*</h2>)(.*?)(?=<h2[\s>]|\Z)", re.S)
DETAILS_RE = re.compile(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", re.S)
TAG_RE = re.compile(r"<[^>]+>")


def extract_recall(algo_id: str, html: str) -> tuple[str, list[dict]]:
    """Find the 面試追問 Q&A of a rendered page and turn them into recall cards.

    Returns (html, cards). In the returned html each <details> of that section
    carries class="recall" and data-card="<algo>#q<n>" so the page script can
    turn it into a card; without JS it stays a plain <details>. Each card is
    {id, algo, n, q (plain text), q_html, a_html}.
    """
    m = RECALL_SECTION_RE.search(html)
    if not m:
        return html, []
    cards: list[dict] = []

    def mark(d: re.Match) -> str:
        n = len(cards) + 1
        cid = f"{algo_id}#q{n}"
        q_html, a_html = d.group(1).strip(), d.group(2).strip()
        cards.append({"id": cid, "algo": algo_id, "n": n,
                      "q": htmlmod.unescape(TAG_RE.sub("", q_html)).strip(),
                      "q_html": q_html, "a_html": a_html})
        return (f'<details class="recall" data-card="{cid}">'
                f"<summary>{q_html}</summary>\n{a_html}\n</details>")

    section = DETAILS_RE.sub(mark, m.group(2))
    return html[:m.start(2)] + section + html[m.end(2):], cards


def to_script_json(obj: object) -> str:
    """JSON safe to inline in <script type="application/json">."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def slider_meta(p: Param) -> dict:
    """One lab slider: label, default, unit and every value it can take."""
    return {"name": p.name, "default": p.default, "label": p.label or p.name,
            "unit": p.unit, "values": p.values()}


def optimizer_meta() -> dict:
    """Optimizer ids -> {name, params[]} for the lab's sliders."""
    return {o.id: {"name": o.name, "params": [slider_meta(p) for p in o.params]}
            for o in OPTIMIZERS.values()}


def lab_data(cfg: dict) -> dict:
    """Precompute the lab's first frame so it renders before Pyodide loads."""
    algos = cfg["algos"] + cfg.get("compare", [])
    params = {a: OPTIMIZERS[a].defaults() | cfg.get("params", {}).get(a, {})
              for a in algos}
    spec = {"problem": cfg["problem"],
            "problem_params": cfg.get("problem_params", {}),
            "n_iter": cfg.get("n_iter", 150)}
    grid = api.problem_grid(spec)
    spec["x0"] = grid["x0"]
    spec["runs"] = [{"algo": a, "params": params[a]} for a in algos]
    out = {"spec": spec, "enabled": cfg["algos"], "algos": algos,
           "grid": grid, "traces": api.traces(spec),
           "optimizers": {a: optimizer_meta()[a] for a in algos},
           "problem_params": {k: [slider_meta(p) for p in v]
                              for k, v in PROBLEM_PARAMS.items()},
           "n_iter": slider_meta(N_ITER),
           # Tolerance of the "f - f* < tol 於第 k 步" read-out; pages of
           # derivative-free methods quote 1e-8 and set lab.reach_tol.
           "reach_tol": cfg.get("reach_tol", 1e-10)}
    if "problems" in cfg:  # problem menu of this page (default: the unconstrained zoo)
        assert cfg["problem"] in cfg["problems"], "lab.problem must be in lab.problems"
        out["problems"] = cfg["problems"]
    return out


def _on_grid(what: str, values: list, v: float) -> None:
    """Fail the build unless v is exactly one of a slider's values."""
    if not any(abs(x - v) <= 1e-12 * max(1.0, abs(v)) for x in values):
        near = min(values, key=lambda x: abs(x - v))
        raise ValueError(f"{what} = {v} is not on its slider grid (nearest {near})")


SCENARIO_KEYS = {"problem", "problem_params", "params", "n_iter", "x0"}
QUESTION_KEYS = {"prompt", "scenario", "options", "explain", "subject", "show"}
OPTION_KEYS = {"label", "when"}


def _known(what: str, got: dict, allowed: set) -> None:
    """Fail on keys outside `allowed` (a typo would otherwise be ignored)."""
    if not isinstance(got, dict):
        raise ValueError(f"{what} must be a mapping, got {got!r}")
    extra = set(got) - allowed
    if extra:
        raise ValueError(f"unknown {what} key(s) {sorted(extra)}; allowed: {sorted(allowed)}")


def scenario_spec(lab_cfg: dict, sc: dict) -> dict:
    """Full lab spec of one prediction scenario, every value checked on the grids.

    sc = {"problem", "problem_params", "params": {algo: {...}}, "n_iter", "x0"?}
    Unset fields fall back to the page's lab settings; algorithms keep the
    page's parameters unless the scenario overrides them.
    """
    _known("scenario", sc, SCENARIO_KEYS)
    algos = lab_cfg["algos"] + lab_cfg.get("compare", [])
    problem = sc.get("problem", lab_cfg["problem"])
    menu = lab_cfg.get("problems", ["quadratic", "rosenbrock", "himmelblau"])
    if problem not in menu:
        raise ValueError(f"scenario problem {problem!r} is not in the lab's menu {menu}")
    same = problem == lab_cfg["problem"]
    pparams = dict(lab_cfg.get("problem_params", {}) if same else {})
    pparams |= sc.get("problem_params", {})
    sliders = {p.name: p for p in PROBLEM_PARAMS.get(problem, [])}
    for k, v in pparams.items():
        if k not in sliders:
            raise ValueError(f"problem {problem!r} has no slider {k!r}")
        _on_grid(f"{problem}.{k}", sliders[k].values(), v)
    for a, ps in sc.get("params", {}).items():
        if a not in algos:
            raise ValueError(f"scenario sets params of {a!r}, which is not in this lab {algos}")
        names = {p.name for p in OPTIMIZERS[a].params}
        _known(f"{a} parameter", ps, names)
    runs = []
    for a in algos:
        merged = OPTIMIZERS[a].defaults() | lab_cfg.get("params", {}).get(a, {}) \
            | sc.get("params", {}).get(a, {})
        for p in OPTIMIZERS[a].params:
            _on_grid(f"{a}.{p.name}", p.values(), merged[p.name])
        runs.append({"algo": a, "params": merged})
    n_iter = sc.get("n_iter", lab_cfg.get("n_iter", 150))
    _on_grid("n_iter", N_ITER.values(), n_iter)
    prob = make_problem(problem, **pparams)
    x0 = list(sc["x0"]) if "x0" in sc else list(prob.x0)
    return {"problem": problem, "problem_params": pparams, "n_iter": n_iter,
            "x0": x0, "runs": runs, "fstar": prob.fstar}


def round_sig(obj: object, digits: int = 6) -> object:
    """Round every float in a JSON-like object to `digits` significant digits."""
    if isinstance(obj, float):
        return float(f"{obj:.{digits}g}")
    if isinstance(obj, list):
        return [round_sig(v, digits) for v in obj]
    if isinstance(obj, dict):
        return {k: round_sig(v, digits) for k, v in obj.items()}
    return obj


def _same_run(spec: dict, run: dict, default: dict | None) -> bool:
    """True when this run of a scenario is the page's default run (same trace)."""
    if not default:
        return False
    d = default["spec"]
    keys = ("problem", "problem_params", "n_iter")
    if any(spec[k] != d[k] for k in keys) or list(spec["x0"]) != list(d["x0"]):
        return False
    return any(r["algo"] == run["algo"] and r["params"] == run["params"] for r in d["runs"])


def predict_data(algo_id: str, lab_cfg: dict, questions: list[dict],
                 default: dict | None = None) -> list[dict]:
    """Run every prediction scenario and decide its answer from the trace.

    Raises (failing the build) when a value is off its slider grid, a key or
    predicate is malformed, or not exactly one option holds. With `default`
    (the page's lab_data), a trace identical to the default lab's is not
    embedded again: it becomes {"algo": id, "same": true}. Other traces are
    rounded to 6 significant digits (the answer is measured before rounding).
    """
    reach_tol = lab_cfg.get("reach_tol", 1e-10)
    algos = lab_cfg["algos"] + lab_cfg.get("compare", [])
    out = []
    for n, q in enumerate(questions, 1):
        qid = f"{algo_id}#p{n}"
        try:
            _known("question", q, QUESTION_KEYS)
            for key in ("prompt", "scenario", "options", "explain"):
                if key not in q:
                    raise ValueError(f"missing {key!r}")
            spec = scenario_spec(lab_cfg, q["scenario"])
            fstar = spec.pop("fstar")
            subject = q.get("subject", lab_cfg["algos"][0])
            if subject not in algos:
                raise ValueError(f"subject {subject!r} is not in this lab {algos}")
            shown = q.get("show", [subject])
            bad = [a for a in shown if a not in algos]
            if bad:
                raise ValueError(f"show: {bad} not in this lab {algos}")
            if not isinstance(q["options"], list) or not 2 <= len(q["options"]) <= 4:
                raise ValueError("a question needs 2 to 4 options")
            for o in q["options"]:
                _known("option", o, OPTION_KEYS)
                predmod.validate(o.get("when"))
            traces = api.traces(spec)
            sub = next(t for t in traces if t["algo"] == subject)
            m = predmod.measure(sub, fstar, reach_tol)
            ans = predmod.answer(q["options"], m)
        except (ValueError, KeyError, TypeError) as err:
            raise ValueError(f"{qid}: {err}") from err
        embedded = [{"algo": t["algo"], "same": True}
                    if _same_run(spec, run, default) else round_sig(t)
                    for t, run in zip(traces, spec["runs"])]
        out.append({
            "id": qid, "prompt": q["prompt"], "explain": q["explain"],
            "subject": subject, "subject_name": OPTIMIZERS[subject].name,
            "answer": ans, "measured": m,
            "summary": predmod.summary(m, reach_tol),
            "options": [{"label": o["label"],
                         "criterion": predmod.criterion(o["when"], reach_tol)}
                        for o in q["options"]],
            "scenario": {"spec": spec, "enabled": shown, "traces": embedded,
                         "same_grid": spec["problem"] == lab_cfg["problem"] and
                         spec["problem_params"] == lab_cfg.get("problem_params", {})},
        })
    return out


def attach_predict(algo_id: str, meta: dict, lab: dict | None) -> None:
    """Add the page's prediction questions to its lab data (if it has any)."""
    if "predict" not in meta:
        return
    if lab is None:
        raise ValueError(f"{algo_id}: predict: needs a lab: (renderer 1) on the same page")
    lab["predict"] = predict_data(algo_id, meta["lab"], meta["predict"], lab)


def lp_data(cfg: dict) -> dict:
    """Precompute the LP lab (renderer 2) so it renders before Pyodide loads.

    cfg = {"method": runner id, "preset": LP preset id,
           "preset_params": {...}, "sliders": [param names], "speed": steps/s}
    """
    spec = {"method": cfg["method"], "preset": cfg["preset"],
            "preset_params": cfg.get("preset_params", {})}
    return {"spec": spec, "sliders": cfg.get("sliders", []),
            "speed": cfg.get("speed", 1.0), "result": api.lp_lab(spec)}


def combo_data(cfg: dict) -> dict:
    """Precompute a combinatorial lab: {"method", "params", "speed"}."""
    spec = {"method": cfg["method"], "params": cfg.get("params", {})}
    return {"spec": spec, "sliders": cfg.get("sliders", []), "speed": cfg.get("speed", 1.0),
            "result": api.combo_lab(spec)}


def practice_data(ex_id: str) -> dict:
    """Skeleton, hints, solution and the plot background for one exercise."""
    ex = EXERCISES[ex_id]
    out = {"id": ex.id, "func": ex.func_name, "skeleton": ex.skeleton,
           "hints": ex.hints, "solution": ex.solution,
           "blanks": list(ex.blanks), "loop_note": ex.loop_note,
           "pseudocode": demos.MODULES[ex_id].PSEUDOCODE}
    if ex.lp_preset:
        prob = lpmod.make_lp(ex.lp_preset)
        out["grid"] = lpmod.objective_grid(prob)
        out["region"] = lpmod.region_data(prob)
    elif ex.demo:
        out["grid"] = api.problem_grid(ex.demo)
    return out


def review_data(cat: dict, cards: list[dict]) -> dict:
    """Build-time data for review.html: every recall card plus page metadata."""
    algos = {a["id"]: {"name": a["name"], "family": a["family"],
                       "cards": a.get("n_cards", 0),
                       "practice": a["status"]["practice"]}
             for a in cat["algorithms"] if a["status"]["page"]}
    return {"families": cat["families"], "algos": algos, "cards": cards}


def load_catalog() -> dict:
    """Read catalog.yaml and attach completion status to each entry."""
    cat = yaml.safe_load((CONTENT / "catalog.yaml").read_text("utf-8"))
    ids = [a["id"] for a in cat["algorithms"]]
    assert len(ids) == len(set(ids)), "duplicate algorithm id"
    known = set(ids)
    children: dict[str, list[str]] = {i: [] for i in ids}
    for a in cat["algorithms"]:
        for parent in a.get("after", []):
            assert parent in known, f"{a['id']}: unknown 'after' {parent}"
            children[parent].append(a["id"])
    for a in cat["algorithms"]:
        a.setdefault("after", [])
        a["children"] = children[a["id"]]
        md = CONTENT / "algorithms" / f"{a['id']}.md"
        meta = split_frontmatter(md.read_text("utf-8"))[0] if md.exists() else {}
        a["status"] = {"page": md.exists(),
                       "lab": "lab" in meta or "lp" in meta or "combo" in meta,
                       "practice": "practice" in meta}
    cat["by_id"] = {a["id"]: a for a in cat["algorithms"]}
    return cat


PYODIDE_FILES = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm",
                 "python_stdlib.zip", "pyodide-lock.json"]
WRAPPER_RE = re.compile(r"<!doctype html>\s*|</?html[^>]*>\s*|</?head>\s*|</?body>\s*",
                        re.I)


def copy_vendor(vendor: Path, out: Path) -> None:
    """Copy Pyodide core and KaTeX (CSS, JS, woff2 fonts) into the site."""
    pyo = out / "static" / "vendor" / "pyodide"
    pyo.mkdir(parents=True)
    for name in PYODIDE_FILES:
        shutil.copy2(vendor / "pyodide" / name, pyo / name)
    kdst = out / "static" / "vendor" / "katex"
    ksrc = vendor / "katex" / "dist"
    (kdst / "contrib").mkdir(parents=True)
    (kdst / "fonts").mkdir()
    for name in ["katex.min.css", "katex.min.js"]:
        shutil.copy2(ksrc / name, kdst / name)
    shutil.copy2(ksrc / "contrib" / "auto-render.min.js",
                 kdst / "contrib" / "auto-render.min.js")
    for font in (ksrc / "fonts").glob("*.woff2"):
        shutil.copy2(font, kdst / "fonts" / font.name)


def build(out: Path = OUT, vendor: Path | None = None) -> None:
    """Render every page and copy static assets into `out`.

    With `vendor`, Pyodide and KaTeX are served from the site itself and
    the home page is emitted without its <html>/<head>/<body> wrapper (the
    Artifact host adds its own).
    """
    OUT = out  # noqa: N806 - shadow the module default for this build
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "algorithms").mkdir(parents=True)
    shutil.copytree(SRC / "static", OUT / "static")

    # Ship optlab's source so the Pyodide worker can import it.
    py_files = sorted(p.relative_to(ROOT).as_posix()
                      for p in (ROOT / "optlab").rglob("*.py"))
    for rel in py_files:
        dst = OUT / "py" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    (OUT / "py" / "manifest.json").write_text(json.dumps(py_files), "utf-8")

    cat = load_catalog()
    env = Environment(loader=FileSystemLoader(SRC / "templates"),
                      autoescape=select_autoescape(["html"]))
    if vendor:
        copy_vendor(vendor, OUT)
    pyodide_base = "{root}/static/vendor/pyodide/" if vendor else PYODIDE_BASE
    katex_base = "{root}/static/vendor/katex/" if vendor else KATEX_BASE

    def bases(root: str) -> dict:
        return {"pyodide_base": pyodide_base.replace("{root}", root),
                "katex_base": katex_base.replace("{root}", root)}

    common = {"cat": cat}

    page_tpl = env.get_template("algorithm.html")
    all_cards: list[dict] = []
    for a in cat["algorithms"]:
        md = CONTENT / "algorithms" / f"{a['id']}.md"
        meta, body = split_frontmatter(md.read_text("utf-8")) if md.exists() \
            else ({}, "")
        html = render_markdown(body) if body else ""
        html, recall = extract_recall(a["id"], html)
        a["n_cards"] = len(recall)
        all_cards.extend(recall)
        lab = lab_data(meta["lab"]) if "lab" in meta else None
        attach_predict(a["id"], meta, lab)
        lpd = lp_data(meta["lp"]) if "lp" in meta else None
        combod = combo_data(meta["combo"]) if "combo" in meta else None
        practice = practice_data(meta["practice"]) if "practice" in meta \
            else None
        html = html.replace("<!-- lab -->", '<div id="lab"></div>')
        html = html.replace("<!-- lp -->", '<div id="lp"></div>')
        html = html.replace("<!-- combo -->", '<div id="combo"></div>')
        html = html.replace("<!-- practice -->", '<div id="practice"></div>')
        (OUT / "algorithms" / f"{a['id']}.html").write_text(page_tpl.render(
            root="..", a=a, meta=meta, body=html, **bases(".."),
            lab_json=to_script_json(lab) if lab else "",
            lp_json=to_script_json(lpd) if lpd else "",
            combo_json=to_script_json(combod) if combod else "",
            practice_json=to_script_json(practice) if practice else "",
            **common), "utf-8")

    cards = [{k: a[k] for k in ("id", "name", "en", "family", "problem",
                                 "info", "domains", "summary", "status")}
             | {"cards": a["n_cards"]}
             for a in cat["algorithms"]]
    index = env.get_template("index.html").render(
        root=".", cards_json=to_script_json(cards), **bases("."), **common)
    if vendor:
        index = WRAPPER_RE.sub("", index)
    (OUT / "index.html").write_text(index, "utf-8")

    review = env.get_template("review.html").render(
        root=".", review_json=to_script_json(review_data(cat, all_cards)),
        **bases("."), **common)
    (OUT / "review.html").write_text(review, "utf-8")

    # Card counts for the nav badge on every page (drops ids of removed cards).
    (OUT / "static" / "js" / "recall-index.js").write_text(
        "// Generated by scripts/build.py: recall cards per page, pages with practice.\n"
        f"export const CARD_COUNTS = {json.dumps({a['id']: a['n_cards'] for a in cat['algorithms'] if a['n_cards']})};\n"
        f"export const PRACTICE = {json.dumps([a['id'] for a in cat['algorithms'] if a['status']['practice']])};\n",
        "utf-8")

    n_pages = sum(a["status"]["page"] for a in cat["algorithms"])
    print(f"built {len(cat['algorithms'])} algorithms "
          f"({n_pages} detailed, {len(all_cards)} recall cards) -> {OUT}/")


def serve(port: int, out: Path = OUT) -> None:
    """Serve the built site locally (Pyodide needs http://, not file://)."""
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    print(f"serving http://localhost:{port}/  (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("", port), handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--vendor", type=Path, default=None)
    args = parser.parse_args()
    build(args.out.resolve(), args.vendor)
    if args.serve:
        serve(args.port, args.out.resolve())

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

from optlab import OPTIMIZERS, api, demos, lp as lpmod  # noqa: E402
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

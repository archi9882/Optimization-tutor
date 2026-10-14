"""Build the static site into _site/.

    python scripts/build.py            # build
    python scripts/build.py --serve    # build, then serve on localhost:8000

Inputs:  content/catalog.yaml, content/algorithms/*.md, site_src/, optlab/
Output:  _site/ (ready for GitHub Pages)
"""
from __future__ import annotations

import argparse
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

from optlab import OPTIMIZERS, api  # noqa: E402
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


def to_script_json(obj: object) -> str:
    """JSON safe to inline in <script type="application/json">."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def optimizer_meta() -> dict:
    """Optimizer ids -> {name, params[]} for the lab's sliders."""
    return {o.id: {"name": o.name,
                   "params": [{"name": p.name, "default": p.default,
                               "lo": p.lo, "hi": p.hi, "log": p.log,
                               "label": p.label or p.name}
                              for p in o.params]}
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
    return {"spec": spec, "enabled": cfg["algos"], "algos": algos,
            "grid": grid, "traces": api.traces(spec),
            "optimizers": {a: optimizer_meta()[a] for a in algos}}


def practice_data(ex_id: str) -> dict:
    """Skeleton, hints, solution and demo contour for one exercise."""
    ex = EXERCISES[ex_id]
    return {"id": ex.id, "func": ex.func_name, "skeleton": ex.skeleton,
            "hints": ex.hints, "solution": ex.solution,
            "grid": api.problem_grid(ex.demo)}


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
        a["status"] = {"page": md.exists(), "lab": "lab" in meta,
                       "practice": "practice" in meta}
    cat["by_id"] = {a["id"]: a for a in cat["algorithms"]}
    return cat


def build() -> None:
    """Render every page and copy static assets into _site/."""
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
    common = {"cat": cat, "pyodide_base": PYODIDE_BASE,
              "katex_base": KATEX_BASE}

    cards = [{k: a[k] for k in ("id", "name", "en", "family", "problem",
                                 "info", "domains", "summary", "status")}
             for a in cat["algorithms"]]
    (OUT / "index.html").write_text(env.get_template("index.html").render(
        root=".", cards_json=to_script_json(cards),
        **common), "utf-8")

    page_tpl = env.get_template("algorithm.html")
    for a in cat["algorithms"]:
        md = CONTENT / "algorithms" / f"{a['id']}.md"
        meta, body = split_frontmatter(md.read_text("utf-8")) if md.exists() \
            else ({}, "")
        html = render_markdown(body) if body else ""
        lab = lab_data(meta["lab"]) if "lab" in meta else None
        practice = practice_data(meta["practice"]) if "practice" in meta \
            else None
        html = html.replace("<!-- lab -->", '<div id="lab"></div>')
        html = html.replace("<!-- practice -->", '<div id="practice"></div>')
        (OUT / "algorithms" / f"{a['id']}.html").write_text(page_tpl.render(
            root="..", a=a, meta=meta, body=html,
            lab_json=to_script_json(lab) if lab else "",
            practice_json=to_script_json(practice) if practice else "",
            **common), "utf-8")

    n_pages = sum(a["status"]["page"] for a in cat["algorithms"])
    print(f"built {len(cat['algorithms'])} algorithms "
          f"({n_pages} detailed) -> {OUT.relative_to(ROOT)}/")


def serve(port: int) -> None:
    """Serve _site/ locally (Pyodide needs http://, not file://)."""
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
    print(f"serving http://localhost:{port}/  (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("", port), handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    build()
    if args.serve:
        serve(args.port)

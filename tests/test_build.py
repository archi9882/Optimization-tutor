"""Tests for the static-site build: catalog integrity and page rendering."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_catalog_is_consistent() -> None:
    """Ids are unique; family/problem/info/domain/tag keys all exist."""
    cat = build.load_catalog()
    for a in cat["algorithms"]:
        assert a["family"] in cat["families"], a["id"]
        assert a["problem"] in cat["problems"], a["id"]
        assert a["info"] in cat["info"], a["id"]
        assert set(a["domains"]) <= set(cat["domains"]), a["id"]
        assert set(a["tags"]) <= set(cat["tags"]), a["id"]
        assert a["summary"].strip(), a["id"]


def test_detail_pages_reference_known_algorithms_and_exercises() -> None:
    """Every algorithm .md matches a catalog id and valid lab/practice ids."""
    from optlab import OPTIMIZERS
    from optlab.exercises import EXERCISES
    ids = {a["id"] for a in build.load_catalog()["algorithms"]}
    for md in (ROOT / "content" / "algorithms").glob("*.md"):
        assert md.stem in ids, md.name
        meta, body = build.split_frontmatter(md.read_text("utf-8"))
        if "lab" in meta:
            for a in meta["lab"]["algos"] + meta["lab"].get("compare", []):
                assert a in OPTIMIZERS, (md.name, a)
            assert "<!-- lab -->" in body, md.name
        if "practice" in meta:
            assert meta["practice"] in EXERCISES, md.name
            assert "<!-- practice -->" in body, md.name


def test_markdown_keeps_math_intact() -> None:
    """Underscores and asterisks inside $...$ must not become <em>."""
    html = build.render_markdown("令 $x_k * y_k$ 與 $$a_{i}*b_{j}$$ 和 *強調*")
    assert "$x_k * y_k$" in html
    assert "$$a_{i}*b_{j}$$" in html
    assert "<em>強調</em>" in html


def test_math_with_angle_brackets_is_escaped() -> None:
    """'<' inside math must be HTML-escaped so it is not parsed as a tag."""
    html = build.render_markdown("需要 $0<\\eta<2/L$。")
    assert "$0&lt;\\eta&lt;2/L$" in html


@pytest.fixture(scope="module")
def site() -> Path:
    build.build()
    return build.OUT


def test_build_writes_every_page(site: Path) -> None:
    """One HTML file per catalog entry plus the index and Python manifest."""
    ids = [a["id"] for a in build.load_catalog()["algorithms"]]
    assert (site / "index.html").exists()
    for i in ids:
        assert (site / "algorithms" / f"{i}.html").exists(), i
    files = json.loads((site / "py" / "manifest.json").read_text("utf-8"))
    assert "optlab/api.py" in files
    for f in files:
        assert (site / "py" / f).exists(), f


def test_inline_json_cannot_close_script_tag(site: Path) -> None:
    """Embedded data must never contain a raw '</' sequence."""
    for page in (site / "algorithms").glob("*.html"):
        text = page.read_text("utf-8")
        for marker in ('id="lab-data">', 'id="practice-data">'):
            if marker in text:
                blob = text.split(marker, 1)[1].split("</script>", 1)[0]
                assert "</" not in blob
                json.loads(blob)

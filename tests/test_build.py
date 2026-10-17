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
        if "lp" in meta:
            from optlab.ip import RUNNERS as IP_RUNNERS
            from optlab.lp import PRESETS, RUNNERS
            assert meta["lp"]["method"] in RUNNERS | IP_RUNNERS, md.name
            assert meta["lp"]["preset"] in PRESETS, md.name
            assert "<!-- lp -->" in body, md.name
        if "combo" in meta:
            from optlab.combo import RUNNERS as COMBO
            assert meta["combo"]["method"] in COMBO, md.name
            assert "<!-- combo -->" in body, md.name
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
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site")
    build.build(out)
    return out


def test_build_writes_every_page(site: Path) -> None:
    """One HTML file per catalog entry plus the index and Python manifest."""
    ids = [a["id"] for a in build.load_catalog()["algorithms"]]
    assert (site / "index.html").exists()
    assert (site / "review.html").exists()
    assert "CARD_COUNTS" in (site / "static" / "js" / "recall-index.js").read_text("utf-8")
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
        for marker in ('id="lab-data">', 'id="lp-data">', 'id="combo-data">', 'id="practice-data">'):
            if marker in text:
                blob = text.split(marker, 1)[1].split("</script>", 1)[0]
                assert "</" not in blob
                json.loads(blob)


def test_extract_recall_marks_only_interview_section() -> None:
    """Only <details> under 面試追問 become cards; ids are algo#qN."""
    html = build.render_markdown(
        "## 定義\n\n<details markdown=\"1\">\n<summary>推導</summary>\n\n略\n\n</details>\n\n"
        "## 面試追問\n\n<details markdown=\"1\">\n<summary>Q1. 為何 $a<b$？</summary>\n\n因為 *好*。\n\n</details>\n\n"
        "<details markdown=\"1\">\n<summary>Q2. 再一題</summary>\n\n答。\n\n</details>\n\n## 後記\n\n<details><summary>x</summary>y</details>\n")
    out, cards = build.extract_recall("demo", html)
    assert [c["id"] for c in cards] == ["demo#q1", "demo#q2"]
    assert cards[0]["q"] == "Q1. 為何 $a<b$？"
    assert "<em>好</em>" in cards[0]["a_html"]
    assert out.count('class="recall"') == 2
    assert out.count("<details") == 4


def _review_blob(site: Path) -> dict:
    text = (site / "review.html").read_text("utf-8")
    blob = text.split('id="review-data">', 1)[1].split("</script>", 1)[0]
    assert "</" not in blob
    return json.loads(blob)


def test_every_detailed_page_yields_recall_cards(site: Path) -> None:
    """Each detailed page has >= 1 card, ids are unique and match the page."""
    data = _review_blob(site)
    ids = [c["id"] for c in data["cards"]]
    assert len(ids) == len(set(ids))
    cat = build.load_catalog()
    for a in cat["algorithms"]:
        if not a["status"]["page"]:
            continue
        mine = [c for c in data["cards"] if c["algo"] == a["id"]]
        assert mine, a["id"]
        assert data["algos"][a["id"]]["cards"] == len(mine)
        page = (site / "algorithms" / f"{a['id']}.html").read_text("utf-8")
        for c in mine:
            assert c["id"] == f"{a['id']}#q{c['n']}"
            assert c["q"].strip() and c["a_html"].strip(), c["id"]
            assert f'data-card="{c["id"]}"' in page


def test_home_cards_carry_recall_counts(site: Path) -> None:
    """The home page data has the card count used by the mastery ring."""
    text = (site / "index.html").read_text("utf-8")
    blob = text.split('id="cards-data">', 1)[1].split("</script>", 1)[0]
    assert "</" not in blob
    cards = {c["id"]: c for c in json.loads(blob)}
    assert cards["gd"]["cards"] >= 1
    assert all(c["cards"] == 0 for c in cards.values() if not c["status"]["page"])

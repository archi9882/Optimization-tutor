"""Prediction questions: predicates, the exactly-one-true rule, slider grids."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from optlab import make_problem, predict

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

PAGES = [p for p in sorted((ROOT / "content" / "algorithms").glob("*.md"))
         if "predict" in build.split_frontmatter(p.read_text("utf-8"))[0]]


def meta_of(algo: str) -> dict:
    path = ROOT / "content" / "algorithms" / f"{algo}.md"
    return build.split_frontmatter(path.read_text("utf-8"))[0]


def trace(xs: list, fs: list, status: str = "max_iter") -> dict:
    return {"xs": xs, "fs": fs, "status": status, "n_iter": len(xs) - 1}


def test_measure_counts_zigzags_rises_and_reach() -> None:
    """Back-and-forth steps are turns; f going up is a rise; reach uses the tolerance."""
    t = trace([[0, 0], [1, 0], [0.5, 0], [1, 0], [2, 0]], [4, 1, 2, 1e-3, 1e-12])
    m = predict.measure(t, fstar=0.0, reach_tol=1e-10)
    assert m["zigzag"] == 2          # 1 -> 0.5 and 0.5 -> 1 both reverse
    assert m["rises"] == 1           # 1 -> 2
    assert m["reach"] == 4
    assert m["final_gap"] == pytest.approx(1e-12)


def test_predicates_on_a_diverged_run() -> None:
    """A diverged run never 'reaches' and has an infinite final gap."""
    m = predict.measure(trace([[0, 0], [1e5, 0]], [0.5, 1e10], "diverged"), 0.0, 1e-10)
    assert predict.check({"status": "diverged"}, m)
    assert predict.check({"not_reach_within": 150}, m)
    assert not predict.check({"reach_within": 150}, m)
    assert predict.check({"final_gap_above": 1.0}, m)


def test_answer_requires_exactly_one_true_option() -> None:
    m = predict.measure(trace([[0, 0], [1, 0]], [1.0, 0.0]), 0.0, 1e-10)
    yes, no = {"label": "a", "when": {"reach_within": 5}}, {"label": "b", "when": {"status": "diverged"}}
    assert predict.answer([no, yes], m) == 1
    with pytest.raises(ValueError, match="exactly one"):
        predict.answer([no, no], m)
    with pytest.raises(ValueError, match="exactly one"):
        predict.answer([yes, yes], m)
    with pytest.raises(ValueError, match="unknown predicate"):
        predict.check({"converged_fast": True}, m)


def test_criterion_text_is_generated_from_the_predicate() -> None:
    assert predict.criterion({"reach_within": 150, "zigzag_min": 3}, 1e-10) == \
        "150 步內 f−f* < 10⁻¹⁰，且折返（相鄰兩步方向相反）≥ 3 次"
    assert predict.criterion({"status": ["max_iter", "converged", "stopped"]}, 1e-10) == "未發散"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_every_question_has_one_build_time_answer(page: Path) -> None:
    """The build runs each scenario and finds exactly one true option."""
    meta = build.split_frontmatter(page.read_text("utf-8"))[0]
    qs = build.predict_data(page.stem, meta["lab"], meta["predict"])
    assert 2 <= len(qs) <= 3
    for q in qs:
        assert 0 <= q["answer"] < len(q["options"])
        sub = next(t for t in q["scenario"]["traces"] if t["algo"] == q["subject"])
        sc = q["scenario"]["spec"]
        fstar = make_problem(sc["problem"], **sc["problem_params"]).fstar
        m = predict.measure(sub, fstar, meta["lab"].get("reach_tol", 1e-10))
        assert m == q["measured"]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_scenario_values_are_on_slider_grids(page: Path) -> None:
    """Every value a question puts on the lab is a position of its slider."""
    meta = build.split_frontmatter(page.read_text("utf-8"))[0]
    for q in meta["predict"]:
        spec = build.scenario_spec(meta["lab"], q["scenario"])
        assert spec["n_iter"] in build.N_ITER.values()
        for p in build.PROBLEM_PARAMS.get(spec["problem"], []):
            assert spec["problem_params"].get(p.name, p.default) in p.values()
        for r in spec["runs"]:
            for p in build.OPTIMIZERS[r["algo"]].params:
                assert r["params"][p.name] in p.values(), (r["algo"], p.name)


def test_known_outcomes() -> None:
    """The authored gd questions: zigzag below 2/L, divergence above, κ = 100 too slow."""
    qs = build.predict_data("gd", meta_of("gd")["lab"], meta_of("gd")["predict"])
    got = [q["options"][q["answer"]]["label"] for q in qs]
    assert got == ["來回震盪，但仍收斂", "發散", "150 步內到不了"]


@pytest.mark.parametrize("bad, match", [
    ({"params": {"gd": {"lr": 0.205}}}, "not on its slider grid"),
    ({"problem_params": {"kappa": 10.5}}, "not on its slider grid"),
    ({"n_iter": 151}, "not on its slider grid"),
    ({"problem": "fert_disk"}, "not in the lab's menu"),
    ({"params": {"bfgs": {}}}, "not in this lab"),
])
def test_off_grid_scenarios_fail_the_build(bad: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build.scenario_spec(meta_of("gd")["lab"], bad)


def test_ambiguous_question_fails_the_build() -> None:
    q = {"prompt": "?", "explain": ".", "scenario": {"params": {"gd": {"lr": 0.21}}},
         "options": [{"label": "a", "when": {"status": "diverged"}},
                     {"label": "b", "when": {"not_reach_within": 150}}]}
    with pytest.raises(ValueError, match="gd#p1: expected exactly one"):
        build.predict_data("gd", meta_of("gd")["lab"], [q])


def test_built_page_embeds_answers(tmp_path: Path) -> None:
    """The reveal reads the embedded answer; pages without predict: have none."""
    build.build(tmp_path)

    def lab(page: str) -> dict:
        text = (tmp_path / "algorithms" / f"{page}.html").read_text("utf-8")
        return json.loads(text.split('id="lab-data">', 1)[1].split("</script>", 1)[0])
    assert [q["id"] for q in lab("gd")["predict"]] == ["gd#p1", "gd#p2", "gd#p3"]
    assert "predict" not in lab("bfgs")

"""Prediction questions: predicates, the exactly-one-true rule, slider grids."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from optlab import predict

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

# Pages whose questions run on the continuous lab (lab:); LP and combinatorial
# pages (lp:, combo:) are tested in the "widgets" section at the end.
PAGES = [p for p in sorted((ROOT / "content" / "algorithms").glob("*.md"))
         if {"predict", "lab"} <= set(build.split_frontmatter(p.read_text("utf-8"))[0])]


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
        "150 步內 f−f* < 10⁻¹⁰，且折返（轉向超過 90°）≥ 3 次"
    assert predict.criterion({"status": ["max_iter", "converged", "stopped"]}, 1e-10) == "未發散"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_every_question_has_one_build_time_answer(page: Path) -> None:
    """The build runs each scenario and finds exactly one true option."""
    meta = build.split_frontmatter(page.read_text("utf-8"))[0]
    qs = build.predict_data(page.stem, meta["lab"], meta["predict"])
    assert 1 <= len(qs) <= 5           # owner rule: 1 to 5 questions per page
    for raw, q in zip(meta["predict"], qs):
        assert 0 <= q["answer"] < len(q["options"])
        spec = build.scenario_spec(meta["lab"], raw["scenario"])
        fstar = spec.pop("fstar")
        prob = build.make_problem(spec["problem"], **spec["problem_params"])
        sub = next(t for t in build.api.traces(spec) if t["algo"] == q["subject"])
        m = predict.measure(sub, fstar, meta["lab"].get("reach_tol", 1e-10), prob)
        assert m == q["measured"]
        assert q["options"][q["answer"]] is not None


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
    q = {"id": "p1", "prompt": "?", "explain": ".", "scenario": {"params": {"gd": {"lr": 0.21}}},
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
    assert [q["id"] for q in lab("penalty")["predict"]] == [f"penalty#p{n}" for n in range(1, 5)]
    without = [p.stem for p in sorted((ROOT / "content" / "algorithms").glob("*.md"))
               if "lab" in meta_of(p.stem) and "predict" not in meta_of(p.stem)]
    for page in without:
        assert "predict" not in lab(page), page


GOOD_Q = {"id": "p1", "prompt": "?", "explain": ".", "scenario": {"params": {"gd": {"lr": 0.21}}},
          "options": [{"label": "a", "when": {"status": "diverged"}},
                      {"label": "b", "when": {"reach_within": 150}}]}


@pytest.mark.parametrize("patch, match", [
    ({"scenario": {"problem_param": {"kappa": 10}}}, "unknown scenario key"),
    ({"scenario": {"params": {"gd": {"eta": 0.21}}}}, "unknown gd parameter key"),
    ({"options": [{"label": "a", "when": {"status": "blew_up"}},
                  {"label": "b", "when": {"reach_within": 150}}]}, "status must be one or more"),
    ({"options": [{"label": "a", "when": {"status": "diverged"}},
                  {"label": "b", "when": {"reach_within": "150"}}]}, "reach_within needs a number"),
    ({"show": ["gd", "bfgs"]}, "show:"),
    ({"subject": "bfgs"}, "subject 'bfgs' is not in this lab"),
    ({"hint": "x"}, "unknown question key"),
    ({"options": [{"label": "a", "when": {"status": "diverged"}, "correct": True},
                  {"label": "b", "when": {"reach_within": 150}}]}, "unknown option key"),
])
def test_malformed_questions_fail_with_their_id(patch: dict, match: str) -> None:
    q = GOOD_Q | {"id": "p2"} | patch
    build.predict_data("gd", meta_of("gd")["lab"], [GOOD_Q])   # the base question is fine
    with pytest.raises(ValueError, match=f"gd#p2: .*{match}"):
        build.predict_data("gd", meta_of("gd")["lab"], [GOOD_Q, q])


def test_predict_without_lab_fails() -> None:
    with pytest.raises(ValueError, match="needs a lab"):
        build.attach_predict("simplex", {"predict": [GOOD_Q]}, None)


def test_default_traces_are_referenced_and_others_rounded() -> None:
    """A scenario equal to the page default embeds no trace twice; others are rounded."""
    for page, algo in [("momentum", "momentum"), ("penalty", "penalty")]:
        meta = meta_of(page)
        lab = build.lab_data(meta["lab"])
        qs = build.predict_data(page, meta["lab"], meta["predict"], lab)
        same = [q for q in qs if all(t.get("same") for t in q["scenario"]["traces"])]
        subject = [next(t for t in q["scenario"]["traces"] if t["algo"] == algo) for q in qs]
        other = [t for t in subject if not t.get("same")]
        assert same and other, page
        for q in same:
            assert all(t == {"algo": t["algo"], "same": True} for t in q["scenario"]["traces"])
        assert all(float(f"{v:.6g}") == v for v in other[0]["fs"])


def test_momentum_beta_numbers_quoted_in_q3() -> None:
    """The explanation of momentum#p3 quotes these measured step counts."""
    from optlab.trace import run_named

    def reach(lr: float, beta: float) -> int | None:
        t = run_named("momentum", "quadratic", {"kappa": 50}, {"lr": lr, "beta": beta}, None, 150)
        return next((k for k, f in enumerate(t["fs"]) if f < 1e-10), None)
    assert (reach(0.02, 0.76), reach(0.02, 0.8), reach(0.061, 0.57)) == (88, 117, 57)
    for beta in (0.5, 0.55, 0.6, 0.65, 0.85, 0.9, 0.95):
        assert reach(0.02, beta) is None, beta
    explain = meta_of("momentum")["predict"][2]["explain"]
    for s in ("β = 0.76 第 88 步", "β = 0.8 第 117 步", "η = 4/(√L + √μ)² ≈ 0.061", "第 57 步"):
        assert s in explain


# ---------- derivative-free and constrained predicates ----------

FERT = build.make_problem("fert_budget")          # x* = (1.5, 1.5), budget x1 + x2 <= 3


def ctrace(xs: list, status: str = "stopped", outer: list | None = None, **extra) -> dict:
    """A constrained trace; f values are irrelevant to the constrained predicates."""
    t = trace(xs, [FERT.f(tuple(x)) for x in xs], status) | extra
    if outer is not None:
        t["outer"] = outer
    return t


def test_evals_predicates_and_summary() -> None:
    """evals_* bound the run's total f evaluations; the summary quotes the count."""
    t = trace([[0, 0], [1, 0], [1, 1]], [1.0, 0.5, 1e-12], "stopped") | {"evals": 245}
    m = predict.measure(t, 0.0, 1e-8)
    assert m["evals"] == 245
    assert predict.check({"evals_above": 150, "evals_below": 400}, m)
    assert predict.check({"evals_below": 246}, m) and not predict.check({"evals_below": 245}, m)
    assert predict.check({"evals_above": 245}, m) and not predict.check({"evals_above": 246}, m)
    assert predict.summary(m, 1e-8) == \
        "f−f* < 10⁻⁸ 於第 2 步 · 折返 0 次 · f 上升 0 步 · 第 2 步停止，共 245 次 f 計算"


def test_evals_need_a_method_that_counts_them() -> None:
    m = predict.measure(trace([[0, 0], [1, 0]], [1.0, 0.0]), 0.0, 1e-10)
    with pytest.raises(ValueError, match="counts f evaluations"):
        predict.check({"evals_below": 10}, m)


def test_constrained_measure_distance_violation_and_outer() -> None:
    """dist and violation of the last iterate, worst violation on the path, outer rounds."""
    xs = [[0.5, 0.5], [2.0, 2.0], [1.5, 1.5 + 3e-4]]        # middle point is 1 over budget
    m = predict.measure(ctrace(xs, outer=[False, True, True]), FERT.fstar, 1e-10, FERT)
    assert m["dist"] == pytest.approx(3e-4)
    assert m["violation"] == pytest.approx(3e-4)
    assert m["max_violation"] == pytest.approx(1.0)
    assert m["outer"] == 2
    assert predict.check({"final_dist_above": 1e-4, "final_dist_below": 1e-3}, m)
    assert predict.check({"final_violation_above": 1e-6, "final_violation_below": 1e-3}, m)
    assert predict.check({"max_violation_above": 1.0 - 1e-9, "max_violation_below": 2.0}, m)
    assert predict.check({"stays_feasible": False}, m)
    assert not predict.check({"stays_feasible": True}, m)
    assert predict.check({"outer_min": 2, "outer_max": 2}, m)
    assert not predict.check({"outer_max": 1}, m)


def test_stays_feasible_allows_rounding_on_the_boundary() -> None:
    """A projection lands on the boundary up to rounding: FEAS_TOL = 1e-9 counts as feasible."""
    m = predict.measure(ctrace([[0.5, 0.5], [1.5, 1.5 + 1e-12]]), FERT.fstar, 1e-10, FERT)
    assert m["max_violation"] > 0
    assert predict.check({"stays_feasible": True}, m)
    m = predict.measure(ctrace([[0.5, 0.5], [1.5, 1.5 + 1e-8]]), FERT.fstar, 1e-10, FERT)
    assert predict.check({"stays_feasible": False}, m)


def test_diverged_constrained_run_is_far_and_infeasible() -> None:
    m = predict.measure(ctrace([[0.5, 0.5], [9e5, 0.0]], "diverged"), FERT.fstar, 1e-10, FERT)
    assert m["dist"] is None and m["violation"] is None
    assert predict.check({"final_dist_above": 1e3, "final_violation_above": 1e3}, m)
    assert not predict.check({"final_dist_below": 1e9}, m)
    assert predict.check({"stays_feasible": False, "not_stops_within": 500}, m)
    assert predict.summary(m, 1e-10) == "第 2 步發散"


def test_stops_within() -> None:
    """Stopped (or a zero gradient) at step <= N; max_iter and diverged never 'stop'."""
    def m_of(status: str, n: int) -> dict:
        return predict.measure(trace([[0, 0]] * (n + 1), [1.0] * (n + 1), status), 0.0, 1e-10)
    assert predict.check({"stops_within": 3}, m_of("stopped", 3))
    assert predict.check({"stops_within": 3}, m_of("converged", 2))
    assert predict.check({"not_stops_within": 2}, m_of("stopped", 3))
    assert predict.check({"stops_within": 4, "not_stops_within": 2}, m_of("stopped", 3))
    assert predict.check({"not_stops_within": 150}, m_of("max_iter", 150))
    assert predict.check({"not_stops_within": 150}, m_of("diverged", 3))


@pytest.mark.parametrize("when, match", [
    ({"reach_within": 150}, "does not show on a constrained problem"),
    ({"not_reach_within": 150}, "does not show on a constrained problem"),
    ({"final_gap_below": 1e-3}, "does not show on a constrained problem"),
])
def test_f_gap_predicates_are_refused_on_constrained_runs(when: dict, match: str) -> None:
    m = predict.measure(ctrace([[0.5, 0.5], [1.5, 1.5]]), FERT.fstar, 1e-10, FERT)
    with pytest.raises(ValueError, match=match):
        predict.check(when, m)


@pytest.mark.parametrize("when", [{"final_dist_below": 1e-3}, {"final_violation_above": 1e-3},
                                  {"max_violation_below": 1.0}, {"stays_feasible": True},
                                  {"outer_min": 1}])
def test_constrained_predicates_need_a_constrained_problem(when: dict) -> None:
    m = predict.measure(trace([[0, 0], [1, 0]], [1.0, 0.0]), 0.0, 1e-10)
    with pytest.raises(ValueError, match="needs a constrained problem"):
        predict.check(when, m)


@pytest.mark.parametrize("when, match", [
    ({"stays_feasible": 1}, "stays_feasible needs true or false"),
    ({"stays_feasible": "yes"}, "stays_feasible needs true or false"),
    ({"evals_below": 1.5}, "evals_below needs a whole number"),
    ({"outer_min": -1}, "outer_min needs a whole number"),
    ({"stops_within": 2.5}, "stops_within needs a whole number"),
    ({"final_dist_below": 0}, "final_dist_below needs a positive number"),
    ({"max_violation_above": -1e-3}, "max_violation_above needs a positive number"),
    ({"final_violation_below": float("inf")}, "needs a positive number"),
    ({"final_dist_below": True}, "final_dist_below needs a number"),
    ({"evals_above": 400, "evals_below": 150}, "leave no possible run"),
    ({"final_dist_above": 1e-2, "final_dist_below": 1e-2}, "leave no possible run"),
    ({"stops_within": 100, "not_stops_within": 100}, "leave no possible run"),
    ({"outer_min": 3, "outer_max": 2}, "leave no possible run"),
])
def test_new_predicates_are_validated_strictly(when: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        predict.validate(when)


def test_criterion_text_of_the_new_predicates() -> None:
    c = predict.criterion
    assert c({"evals_above": 150, "evals_below": 400}, 1e-8) == "150 ≤ 整次執行的 f 計算次數 < 400"
    assert c({"evals_below": 150}, 1e-8) == "整次執行的 f 計算次數 < 150"
    assert c({"final_dist_below": 1e-2, "final_dist_above": 1e-3}, 1e-10) == \
        "10⁻³ ≤ 最後 ‖x−x*‖ < 10⁻²"
    assert c({"final_dist_above": 0.0025, "final_dist_below": 0.014}, 1e-10) == \
        "0.0025 ≤ 最後 ‖x−x*‖ < 0.014"
    assert c({"final_violation_above": 1e-6}, 1e-10) == "最後的違反量 ≥ 10⁻⁶"
    assert c({"max_violation_below": 0.5}, 1e-10) == "路徑上最大違反量 < 0.5"
    assert c({"stays_feasible": True}, 1e-10) == "每個迭代點都在可行域內（違反量 ≤ 10⁻⁹）"
    assert c({"stays_feasible": False}, 1e-10) == "有迭代點離開可行域（違反量 > 10⁻⁹）"
    assert c({"stops_within": 1, "outer_max": 0, "final_dist_above": 1e-3}, 1e-10) == \
        "1 步內停止，且沒有完成任何一輪外圈，且最後 ‖x−x*‖ ≥ 10⁻³"
    assert c({"stops_within": 400, "not_stops_within": 208}, 1e-10) == \
        "400 步內停止，且 208 步內未停止"
    assert c({"outer_min": 2, "outer_max": 5}, 1e-10) == "完成的外圈 ≥ 2 輪，且完成的外圈 ≤ 5 輪"
    # the existing keys keep their text (no range merging for f − f*)
    assert c({"final_gap_above": 1e-3, "final_gap_below": 1e-2}, 1e-10) == \
        "最後 f−f* ≥ 10⁻³，且最後 f−f* < 10⁻²"


@pytest.mark.parametrize("v, js", [(2.3e-7, "2.30e-7"), (0.005, "0.0050"), (0.949, "0.9490"),
                                   (1.41, "1.410"), (123.456, "123.5"), (0.0, "0"),
                                   (1.5e5, "1.50e+5"), (-2.5e-4, "-2.50e-4")])
def test_fmt_matches_the_lab(v: float, js: str) -> None:
    """Same strings as plots.js fmt (toExponential(2) / toFixed)."""
    assert predict.fmt(v) == js


def test_constrained_summary_is_the_lab_result_line() -> None:
    """lab.js describe(): stop · ‖x−x*‖ · 違反量, then the path's numbers."""
    m = predict.measure(ctrace([[0.5, 0.5], [2.0, 2.0], [1.5, 1.5 + 3e-4]], outer=[False, True, True]),
                        FERT.fstar, 1e-10, FERT)
    assert predict.summary(m, 1e-10) == \
        "第 2 步停止 · ‖x−x*‖ = 3.00e-4 · 違反量 3.00e-4 · 路徑上最大違反量 1.000 · 完成 2 輪外圈"
    m = predict.measure(ctrace([[3.0, 2.5], [3.0, 2.5]], message="起點不在可行域內部"),
                        FERT.fstar, 1e-10, FERT)
    assert predict.summary(m, 1e-10).startswith("第 1 步停止：起點不在可行域內部 · ‖x−x*‖ = 1.803 · 違反量 2.500")
    m = predict.measure(ctrace([[0.5, 0.5], [1.0, 1.0]], "max_iter"), FERT.fstar, 1e-10, FERT)
    assert predict.summary(m, 1e-10).startswith("1 步內未停止 · ")


ROLL_B = {
    "nelder-mead": ["150～399 次", "120 步內", "不到 200 次，和 0° 差不多"],
    "powell": ["1000 次以上", "2 輪內", "2 輪內"],
    "pattern-search": ["51～150 步", "500 步內達不到", "500 步內達不到"],
    "penalty": ["10⁻⁶～10⁻³：還在三角形外一點點", "10⁻³～10⁻²", "0.1 以上", "更遠（0.0050 以上）"],
    "barrier": ["每一步都留在三角形內部", "無法開始：第 1 步就停在起點", "不到 10⁻⁶", "10⁻² 以上"],
    "augmented-lagrangian": ["不到 10⁻⁶", "更晚：209～400 步才停止", "10⁻² 以上",
                             "比預設更早：208 步內停止"],
    "projected-gradient": ["第一步被拉回圓盤，之後收斂到 x*", "收斂：200 步內停止，且 ‖x − x*‖ < 10⁻⁶",
                           "收斂：200 步內停止，且 ‖x − x*‖ < 10⁻⁶", "不收斂：200 步內停不下來"],
    "frank-wolfe": ["每一步都留在三角形內", "151 次以上", "0.0025～0.014（比 k = 100 時小了 1.5～8 倍）", "10⁻⁴ 以下"],
}


@pytest.mark.parametrize("page", sorted(ROLL_B))
def test_known_outcomes_derivative_free_and_constrained(page: str) -> None:
    qs = build.predict_data(page, meta_of(page)["lab"], meta_of(page)["predict"])
    assert [q["options"][q["answer"]]["label"] for q in qs] == ROLL_B[page]


def _q(page: str, n: int) -> dict:
    meta = meta_of(page)
    return build.predict_data(page, meta["lab"], meta["predict"], build.lab_data(meta["lab"]))[n - 1]


@pytest.mark.parametrize("page, n, quoted", [
    ("nelder-mead", 1, ["245 次", "116 步"]), ("nelder-mead", 2, ["第 94 步"]),
    ("nelder-mead", 3, ["153 次"]), ("powell", 1, ["1803 次", "44 次一維搜尋"]),
    ("pattern-search", 1, ["第 59 步"]), ("penalty", 1, ["2.97 × 10⁻⁴"]),
    ("penalty", 2, ["0.0050"]), ("penalty", 3, ["0.185", "第 200 步"]),
    ("penalty", 4, ["第 243 步", "0.042"]), ("barrier", 1, ["第 50 步", "2.6 × 10⁻⁷"]),
    ("barrier", 3, ["第 14 步", "1.6 × 10⁻⁷"]), ("barrier", 4, ["第 7 步", "0.204"]),
    ("augmented-lagrangian", 1, ["第 209 步", "2.3 × 10⁻⁷", "9.9 × 10⁻⁸"]),
    ("augmented-lagrangian", 2, ["第 367 步", "5 次"]), ("augmented-lagrangian", 3, ["0.0595", "3 次"]),
    ("augmented-lagrangian", 4, ["第 73 步", "13 次", "1.9 × 10⁻⁷"]),
    ("projected-gradient", 1, ["第 32 步", "3.5 × 10⁻¹³"]), ("projected-gradient", 2, ["第 35 步"]),
    ("projected-gradient", 3, ["第 33 步"]), ("projected-gradient", 4, ["1.41"]),
    ("frank-wolfe", 2, ["299 次"]), ("frank-wolfe", 3, ["0.0070"]),
    ("frank-wolfe", 4, ["5.7 × 10⁻⁶", "折返 3 次"]),
])
def test_quoted_numbers_are_the_measured_ones(page: str, n: int, quoted: list[str]) -> None:
    """Each number an explanation quotes appears in the text and matches the measure."""
    q = _q(page, n)
    for s in quoted:
        assert s in q["explain"], (page, n, s)
    m = q["measured"]
    checks = {
        "245 次": m.get("evals") == 245, "116 步": m.get("evals", 0) - 3 - m["steps"] == 116,
        "第 94 步": m["reach"] == 94, "153 次": m.get("evals") == 153,
        "1803 次": m.get("evals") == 1803, "第 59 步": m["reach"] == 59,
        "299 次": m["zigzag"] == 299, "折返 3 次": m["zigzag"] == 3,
        "5 次": m.get("outer") == 5, "3 次": m.get("outer") == 3, "13 次": m.get("outer") == 13,
    }
    for s in quoted:
        if s in checks:
            assert checks[s], (page, n, s, m)
        elif s.startswith("第 ") and s.endswith(" 步"):
            assert m["steps"] == int(s[2:-2]), (page, n, s, m)
        elif "×" in s:   # "2.6 × 10⁻⁷": the distance or the violation, same digits
            mant, exp = s.split(" × 10")
            v = float(f"{mant}e{exp.translate(SUPERSCRIPT)}")
            digits = len(mant.replace(".", ""))
            assert any(m.get(k) and float(f"{m[k]:.{digits - 1}e}") == v
                       for k in ("dist", "violation")), (page, n, s, m)
        elif s in ("44 次一維搜尋",):
            pass         # counted by test_powell_line_searches below
        else:            # a plain distance such as 0.0050 or 1.41
            assert f"{m['dist']:.{len(s.split('.')[1])}f}" == s, (page, n, s, m)


SUPERSCRIPT = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


def test_powell_line_searches(monkeypatch: pytest.MonkeyPatch) -> None:
    """powell#p1 quotes 44 line searches for 1803 evaluations on Rosenbrock."""
    import optlab.demos.powell as pw
    from optlab.trace import run_named
    calls = [0]
    orig = pw.line_min

    def counted(*args):
        calls[0] += 1
        return orig(*args)
    monkeypatch.setattr(pw, "line_min", counted)
    assert run_named("powell", "rosenbrock", {}, {}, None, 60)["evals"] == 1803
    assert calls[0] == 44


def test_constrained_scenarios_embed_full_precision_iterates() -> None:
    """The lab recomputes ‖x−x*‖ from the embedded iterates: it must equal the reveal's."""
    meta = meta_of("projected-gradient")
    q = _q("projected-gradient", 1)          # x0 = (3, 3): not the page default, so embedded
    t = next(t for t in q["scenario"]["traces"] if t["algo"] == "projected_gradient")
    assert "same" not in t
    xstar = build.make_problem(meta["lab"]["problem"]).minima[0]
    x = t["xs"][-1]
    d = ((x[0] - xstar[0]) ** 2 + (x[1] - xstar[1]) ** 2) ** 0.5
    assert f"‖x−x*‖ = {predict.fmt(d)}" in q["summary"]
    # unconstrained scenarios are still rounded to 6 significant digits
    g = next(t for t in _q("pattern-search", 3)["scenario"]["traces"] if "xs" in t)
    assert all(float(f"{v:.6g}") == v for xy in g["xs"] for v in xy)


def test_gd_summary_unchanged() -> None:
    """Pages of gradient methods (no evals, unconstrained) keep their reveal text."""
    q = _q("gd", 1)
    assert "f 計算" not in q["summary"] and "‖x−x*‖" not in q["summary"]
    assert q["summary"].startswith("f−f* < 10⁻¹⁰ 於第 ")


def test_other_numbers_quoted_in_rollb_questions() -> None:
    """Numbers in prompts / explanations that come from runs outside the scenario."""
    from optlab.trace import run_named
    t = run_named("pattern_search", "quadratic", {"kappa": 10, "theta_deg": 30},
                  {"step0": 0.01}, None, 1200)
    assert next(k for k, f in enumerate(t["fs"]) if f < 1e-8) == 1149       # pattern-search#p3
    t = run_named("frank_wolfe", "fert_budget", {}, {}, None, 100)
    assert f"{((t['xs'][-1][0] - 1.5) ** 2 + (t['xs'][-1][1] - 1.5) ** 2) ** 0.5:.3f}" == "0.021"
    t = run_named("penalty", "fert_budget", {}, {}, None, 400)              # penalty#p2
    assert [i["內圈步"] for i, o in zip(t["info"], t["outer"]) if o] == [43, 24, 100, 100, 100]
    rho = 1e4      # exact minimiser of the rho = 1e4 penalty problem (budget active)
    s = 2 / (1 + 2 * rho / 3)
    d = ((3 - rho * s / 2 - 1.5) ** 2 + (2 - rho * s / 6 - 1.5) ** 2) ** 0.5
    assert f"{d:.1e}" == "2.4e-04"
    t = run_named("augmented_lagrangian", "fert_budget", {}, {"rho": 1000.0}, None, 400)
    assert [i["內圈步"] for i, o in zip(t["info"], t["outer"]) if o] == [101, 101, 101]
    t = run_named("barrier_nl", "fert_budget", {}, {"inner": 1}, None, 150)   # barrier#p4
    assert [round(v, 2) for v in t["xs"][4]] == [1.66, 1.34]
    assert f"{t['info'][4]['最小餘裕 −g']:.1e}" == "3.9e-04"
    # projected-gradient#p2: tangential error factor (1 − η τᵀ∇²f τ)/(1 + 2ηλ*) on the disk
    disk = build.make_problem("fert_disk")
    (a, b), lam = disk.minima[0], disk.multipliers[0]
    tau = (-(b - 1), a - 1)                       # unit tangent (the radius is 1)
    curv = 2 * tau[0] ** 2 + 6 * tau[1] ** 2      # ∇²f = diag(2, 6)
    assert (round(curv, 2), round(lam, 2)) == (4.32, 1.63)
    assert round((1 - 0.5 * curv) / (1 + 2 * 0.5 * lam), 2) == -0.44
    # projected-gradient#p4: the period-2 orbit on the budget line
    t = run_named("projected_gradient", "fert_budget", {}, {"lr": 0.5}, None, 200)
    assert sorted(tuple(round(v, 9) for v in x) for x in t["xs"][-2:]) == [(0.5, 2.5), (2.5, 0.5)]


def test_questions_need_explicit_unique_ids() -> None:
    """Saved answers are keyed by the id, so it must be explicit and unique."""
    import copy
    lab = meta_of("gd")["lab"]
    good = copy.deepcopy(GOOD_Q)
    missing = {k: v for k, v in good.items() if k != "id"}
    with pytest.raises(ValueError, match="explicit id"):
        build.predict_data("gd", lab, [missing])
    with pytest.raises(ValueError, match="duplicate question id"):
        build.predict_data("gd", lab, [good, copy.deepcopy(good)])


def test_every_page_question_has_an_id() -> None:
    """All authored questions carry ids p1, p2, ... (unique per page)."""
    for md in (build.CONTENT / "algorithms").glob("*.md"):
        meta, _ = build.split_frontmatter(md.read_text("utf-8"))
        ids = [q.get("id") for q in meta.get("predict", [])]
        assert all(isinstance(i, str) and i.startswith("p") for i in ids), md.name
        assert len(ids) == len(set(ids)), md.name


# ---------- LP / integer / combinatorial widgets (lp.js, combo.js) ----------

WIDGET_PAGES = {
    "simplex": ["2 次", "3 次以上", "頂點 (4, 3)"],
    "dual-simplex": ["一路上升", "1 次", "(7, 0)：只用肥料甲"],
    "lp-interior-point": ["不會，每一步都嚴格在內部", "差不到 0.001", "邊的中間 (3, 4.5)"],
    "qp-active-set": ["灌溉水那條邊的中間 (3, 4.5)", "2 步"],
    "branch-and-bound": ["(4, 0)", "4～6 個", "只有 1 個", "(3, 1)"],
    "cutting-plane": ["不會，一個也不會", "2～3 條", "0 條"],
    "dynamic-programming": ["60 以上", "1～2", "兩個月都放 2 以上"],
    "greedy": ["會，跳過 1～3 條", "是，等於最小生成樹"],
    "hungarian": ["要調整矩陣 1 次", "相同"],
    "two-opt": ["不會，每一步都變短", "4～6 次", "不是，比最短的長"],
    "tabu-search": ["能，而且找到最短的路線", "3 條以下", "4～10 條"],
}


def widget_of(page: str) -> tuple[str, dict]:
    meta = meta_of(page)
    kind = "lp" if "lp" in meta else "combo"
    return kind, meta[kind]


def wq(page: str) -> list[dict]:
    """The page's built questions (with its default data, as the build does)."""
    kind, cfg = widget_of(page)
    default = build.lp_data(cfg) if kind == "lp" else build.combo_data(cfg)
    return build.widget_predict_data(page, kind, cfg, meta_of(page)["predict"], default)


def test_every_lp_and_combo_page_has_questions() -> None:
    """Phase B covers the 11 LP / integer / combinatorial pages (1 to 5 questions each)."""
    pages = {p.stem for p in (ROOT / "content" / "algorithms").glob("*.md")
             if "predict" in meta_of(p.stem) and ("lp" in meta_of(p.stem) or "combo" in meta_of(p.stem))}
    assert pages == set(WIDGET_PAGES)
    for page in pages:
        assert 1 <= len(meta_of(page)["predict"]) <= 5, page


@pytest.mark.parametrize("page", sorted(WIDGET_PAGES))
def test_widget_known_outcomes(page: str) -> None:
    """Each page's answers, decided by running the widget's method at build time."""
    qs = wq(page)
    assert [q["options"][q["answer"]]["label"] for q in qs] == WIDGET_PAGES[page]
    assert [q["id"] for q in qs] == [f"{page}#p{n}" for n in range(1, len(qs) + 1)]


@pytest.mark.parametrize("page", sorted(WIDGET_PAGES))
def test_widget_scenarios_are_on_the_pages_sliders(page: str) -> None:
    """Only the page's own sliders are moved, to positions the range input can take."""
    kind, cfg = widget_of(page)
    sliders = build.widget_sliders(kind, cfg)
    key = build.WIDGET_SCENARIO_KEYS[kind]
    for q in meta_of(page)["predict"]:
        for name, v in q.get("scenario", {}).get(key, {}).items():
            sl = sliders[name]
            assert sl["lo"] <= v <= sl["hi"]
            assert ((v - sl["lo"]) / sl["step"]).is_integer(), (page, name, v)


def test_widget_measure_matches_embedded_and_default_is_not_embedded() -> None:
    """measured is what the predicates saw; default runs are referenced, LP grids dropped."""
    from optlab import api, lp as lpmod
    for page in WIDGET_PAGES:
        kind, cfg = widget_of(page)
        for q in wq(page):
            spec = q["scenario"]["spec"]
            res = api.lp_lab(spec) if kind == "lp" else api.combo_lab(spec)
            m = (predict.measure_lp(res, lpmod.make_lp(spec["preset"], **spec["preset_params"]))
                 if kind == "lp" else predict.measure_combo(res))
            assert m == q["measured"], q["id"]
            emb = q["scenario"]["result"]
            default = (cfg.get("preset_params", {}) == spec.get("preset_params", {}) if kind == "lp"
                       else cfg.get("params", {}) == spec["params"])
            assert emb.get("same", False) == default, q["id"]
            if not default:
                assert "grid" not in emb and len(emb["steps"]) == len(res["steps"])
                assert q["scenario"]["same_grid"] is True


def test_lp_js_grid_formula_reproduces_objective_grid() -> None:
    """lp.js withGrid(): z = sign (c1 x + c2 y) on the default grid's x, y (linear presets)."""
    from optlab import lp as lpmod
    g = lpmod.objective_grid(lpmod.make_lp("farm"))
    other = lpmod.objective_grid(lpmod.make_lp("farm", c1=1.0, c2=1.0))
    z = [[g["sign"] * (1.0 * x + 1.0 * y) for x in g["x"]] for y in g["y"]]
    assert other["x"] == g["x"] and other["y"] == g["y"] and g["linear"]
    assert all(abs(a - b) < 1e-12 for ra, rb in zip(z, other["z"]) for a, b in zip(ra, rb))


@pytest.mark.parametrize("page, sc, match", [
    ("simplex", {"preset_params": {"c1": 1.25}}, "not on its slider"),
    ("simplex", {"preset_params": {"c1": 8.5}}, "not on its slider"),
    ("simplex", {"preset_params": {"c3": 1}}, "unknown scenario preset_params key"),
    ("simplex", {"params": {"c1": 1}}, "unknown scenario key"),
    ("qp-active-set", {"preset_params": {"c1": 1}}, "unknown scenario preset_params key"),
    ("tabu-search", {"params": {"tenure": 9}}, "not on its slider"),
    ("tabu-search", {"params": {"tenure": 2.5}}, "not on its slider"),
    ("tabu-search", {"params": {"iters": 40}}, "unknown scenario params key"),
    ("two-opt", {"params": {"max_steps": 2}}, "unknown scenario params key"),
])
def test_widget_scenarios_off_the_sliders_fail(page: str, sc: dict, match: str) -> None:
    kind, cfg = widget_of(page)
    with pytest.raises(ValueError, match=match):
        build.widget_spec(kind, cfg, sc)


WQ = {"id": "p1", "prompt": "?", "explain": ".",
      "options": [{"label": "a", "when": {"pivots_max": 2}}, {"label": "b", "when": {"pivots_min": 3}}]}


@pytest.mark.parametrize("patch, match", [
    ({"options": [{"label": "a", "when": {"nodes_max": 2}},
                  {"label": "b", "when": {"pivots_min": 3}}]}, "nodes_max does not apply to method 'simplex'"),
    ({"options": [{"label": "a", "when": {"reach_within": 2}},
                  {"label": "b", "when": {"pivots_min": 3}}]}, "unknown predicate key.*lp widget"),
    ({"options": [{"label": "a", "when": {"pivots_max": 2.5}},
                  {"label": "b", "when": {"pivots_min": 3}}]}, "needs a whole number"),
    ({"options": [{"label": "a", "when": {"final_point": [2]}},
                  {"label": "b", "when": {"pivots_min": 3}}]}, r"final_point needs \[x1, x2\]"),
    ({"options": [{"label": "a", "when": {"z_trend": "rising"}},
                  {"label": "b", "when": {"pivots_min": 3}}]}, "z_trend must be one of"),
    ({"options": [{"label": "a", "when": {"pivots_min": 3, "pivots_max": 2}},
                  {"label": "b", "when": {"pivots_min": 3}}]}, "leave no possible run"),
    ({"options": [{"label": "a", "when": {"pivots_max": 3}},
                  {"label": "b", "when": {"pivots_min": 2}}]}, "expected exactly one true option, got 2"),
    ({"scenario": {"preset_params": {"c1": 0.25}}}, "not on its slider"),
    ({"hint": "x"}, "unknown question key"),
])
def test_malformed_widget_questions_fail_with_their_id(patch: dict, match: str) -> None:
    kind, cfg = widget_of("simplex")
    build.widget_predict_data("simplex", kind, cfg, [WQ])        # the base question is fine
    with pytest.raises(ValueError, match=f"simplex#p2: .*{match}"):
        build.widget_predict_data("simplex", kind, cfg, [WQ, WQ | {"id": "p2"} | patch])


def test_widget_questions_need_explicit_unique_ids() -> None:
    kind, cfg = widget_of("simplex")
    with pytest.raises(ValueError, match="explicit id"):
        build.widget_predict_data("simplex", kind, cfg, [{k: v for k, v in WQ.items() if k != "id"}])
    with pytest.raises(ValueError, match="duplicate question id"):
        build.widget_predict_data("simplex", kind, cfg, [WQ, dict(WQ)])


@pytest.mark.parametrize("kind, method, when, match", [
    ("combo", "dp", {"length_below": 3.0}, "does not apply to method 'dp'"),
    ("combo", "two_opt", {"total_above": 3.0}, "does not apply"),
    ("combo", "kruskal", {"adjustments_max": 0}, "does not apply"),
    ("combo", "tabu", {"reaches_optimum": 1}, "needs true or false"),
    ("combo", "dp", {"release_min": {"9月": 1}}, "needs {month: whole number}"),
    ("combo", "dp", {"release_min": {"3月": 1.5}}, "needs {month: whole number}"),
    ("combo", "dp", {"release_min": {"3月": 2}, "release_max": {"3月": 1}}, "leave no possible run"),
    ("combo", "hungarian", {"total_above": 40, "total_below": 40}, "leave no possible run"),
    ("combo", "tabu", {"pivots_min": 1}, "unknown predicate key"),
    ("lp", "simplex", {"stays_interior": True}, "does not apply"),
    ("lp", "barrier", {"stays_interior": "yes"}, "needs true or false"),
    ("lp", "barrier", {"pivots_min": 1}, "does not apply"),
    ("lp", "branch_bound", {"cuts_min": 1}, "does not apply"),
    ("lp", "simplex", {"final_z_below": "36"}, "needs a number"),
    ("lp", "simplex", {}, "non-empty mapping"),
])
def test_widget_predicates_are_validated_strictly(kind: str, method: str, when: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        predict.validate_widget(when, kind, method)


def _lp_measure(method: str, preset: str, **pp) -> dict:
    from optlab import api, lp as lpmod
    res = api.lp_lab({"method": method, "preset": preset, "preset_params": pp})
    return predict.measure_lp(res, lpmod.make_lp(preset, **pp))


def _combo_measure(method: str, **params) -> dict:
    from optlab import api
    return predict.measure_combo(api.combo_lab({"method": method, "params": params}))


def test_lp_measures() -> None:
    """Pivots, final point, z path; B&B answer is the incumbent; cuts keep integer points."""
    m = _lp_measure("simplex", "farm", c1=1.0, c2=1.0)
    assert (m["pivots"], m["x"], m["zs"], m["z_trend"]) == (3, [2.0, 6.0], [0.0, 4.0, 7.0, 8.0], "up")
    assert predict.check_widget({"final_point": [2, 6], "pivots_min": 3, "final_z_above": 8}, m)
    assert not predict.check_widget({"final_z_below": 8}, m)
    m = _lp_measure("branch_bound", "greenhouse")
    assert (m["nodes"], m["x"], m["z"], m["steps"]) == (5, [4.0, 0.0], 20.0, 4)
    m = _lp_measure("cutting_plane", "greenhouse")
    assert (m["cuts"], m["pivots"], m["int_points"], m["cut_points"]) == (3, 6, 13, 0)
    m = _lp_measure("barrier", "farm")
    assert m["steps"] == 27 and 0 < m["min_slack"] < 1e-3 and m["z_trend"] == "mixed"


def test_cut_points_counts_a_cut_that_removes_an_integer_point() -> None:
    """A fabricated cut x1 + x2 <= 3 removes the greenhouse points with x1 + x2 = 4."""
    from optlab import api, lp as lpmod
    res = api.lp_lab({"method": "cutting_plane", "preset": "greenhouse"})
    res["steps"][-1]["cuts"] = [{"a": [1.0, 1.0], "b": 3.0, "label": "bad"}]
    m = predict.measure_lp(res, lpmod.make_lp("greenhouse"))
    assert m["cut_points"] == 3                    # (4, 0), (3, 1), (2, 2)
    assert predict.check_widget({"cut_points_min": 1}, m)


def test_combo_measures_against_brute_force() -> None:
    """Optima come from enumeration (Prim for the MST), not from the methods' code."""
    m = _combo_measure("dp")
    assert (m["total"], m["optimum"], m["optimal"]) == (71.0, 71.0, True)
    assert m["releases"] == {"3月": 1, "4月": 1, "5月": 1, "6月": 2, "7月": 3, "8月": 2}
    m = _combo_measure("kruskal")
    assert (m["considered"], m["rejected"], m["optimal"]) == (10, 2, True)
    assert f"{m['total']:.3f}" == "11.944"
    m = _combo_measure("hungarian")
    assert (m["total"], m["optimum"], m["adjustments"], m["assignments"]) == (38.0, 38, 1, 120)
    m = _combo_measure("two_opt")
    assert (m["steps"], m["uphill"], m["distinct_tours"], m["optimal"]) == (4, 0, 5, False)
    assert (f"{m['length']:.3f}", f"{m['optimum']:.3f}") == ("18.641", "18.195")
    got = {t: _combo_measure("tabu", tenure=t)["distinct_tours"] for t in (0, 2, 3, 5)}
    assert got == {0: 3, 2: 6, 3: 15, 5: 13}
    assert all(_combo_measure("tabu", tenure=t)["optimal"] for t in range(9))


def test_widget_criterion_text() -> None:
    c = predict.criterion_widget
    assert c({"pivots_min": 2, "pivots_max": 2}, "simplex") == "樞紐次數 = 2"
    assert c({"nodes_min": 4, "nodes_max": 6}, "branch_bound") == "解過的節點數 4～6"
    assert c({"final_z_above": 35.999, "final_z_below": 36}, "barrier") == "35.999 ≤ 目標值 z < 36"
    assert c({"final_point": [4, 0]}, "branch_bound") == "找到的整數最優解是 (4, 0)"
    assert c({"final_point": [3, 4.5]}, "barrier") == "最後的點在 (3, 4.5)（距離 < 10⁻³）"
    assert c({"steps_min": 4, "steps_max": 6}, "two_opt") == "交換次數 4～6"
    assert c({"reaches_optimum": True}, "kruskal") == "等於最小生成樹（Prim 獨立計算）"
    assert c({"length_below": 18.64, "reaches_optimum": False}, "tabu") == \
        "最佳路線長（km） < 18.64，且比窮舉找到的最優值差"
    assert c({"release_min": {"6月": 2, "7月": 2}}, "dp") == "6月放水 ≥ 2、7月放水 ≥ 2"
    assert c({"total_above": 60}, "dp") == "總收益 ≥ 60"


def test_widget_summary_is_the_widgets_result() -> None:
    """The reveal's measured line uses the widget's own numbers and formats."""
    assert predict.summary_widget(_lp_measure("simplex", "farm")) == \
        "2 次樞紐，停在 (2, 6)，z 依序 0 → 30 → 36"
    assert predict.summary_widget(_lp_measure("dual_simplex", "fertilizer")) == \
        "2 次樞紐，停在 (3, 2)，成本依序 0 → 8 → 12"
    assert predict.summary_widget(_lp_measure("barrier", "farm")) == \
        "27 步牛頓法後停在 (2.0000, 5.9999)，z = 35.9997；每個點都在可行域內部"
    assert predict.summary_widget(_lp_measure("branch_bound", "greenhouse")) == \
        "解了 5 個節點，整數最優解 (4, 0)，z = 20"
    assert predict.summary_widget(_combo_measure("hungarian")) == \
        "調整矩陣 1 次，總時數 38（120 種分派中最少為 38）"
    assert predict.summary_widget(_combo_measure("tabu", tenure=0)).startswith(
        "20 回合，最佳路線 18.195 km（窮舉最短 18.195 km）；路線變長 10 次，共走過 3 條不同路線")


def _explain(page: str, n: int) -> str:
    return meta_of(page)["predict"][n - 1]["explain"]


def test_numbers_quoted_in_widget_explanations() -> None:
    """Every number an explanation (or prompt) quotes, measured here."""
    from optlab import api, combo as C
    lp = lambda method, preset, **pp: api.lp_lab(  # noqa: E731
        {"method": method, "preset": preset, "preset_params": pp})["steps"]
    # simplex: paths and z
    assert [s["x"] for s in lp("simplex", "farm")] == [[0, 0], [0, 6], [2, 6]]
    assert "0 → 30 → 36" in _explain("simplex", 1) and "−5" in _explain("simplex", 1)
    assert [s["x"] for s in lp("simplex", "farm", c1=1.0, c2=1.0)] == [[0, 0], [4, 0], [4, 3], [2, 6]]
    assert [s["x"] for s in lp("simplex", "farm", c1=3.0, c2=2.0)] == [[0, 0], [4, 0], [4, 3]]
    # dual simplex
    st = lp("dual_simplex", "fertilizer", c1=1.0, c2=3.0)
    assert [(s["x"], s["z"]) for s in st] == [([0, 0], 0), ([4, 0], 4), ([7, 0], 7)]
    assert [(s["x"], s["z"]) for s in lp("dual_simplex", "fertilizer", c1=3.0, c2=1.0)][-1] == ([0, 8], 8)
    # interior point: last point, smallest slack, gap and m·mu
    m = _lp_measure("barrier", "farm")
    st = lp("barrier", "farm")
    mu, gap = st[-1]["mu"], 36 - m["z"]
    assert (f"{mu:.1e}", f"{gap:.2e}", f"{5 * mu:.1e}", f"{m['min_slack']:.1e}") == \
        ("1.5e-04", "3.05e-04", "7.6e-04", "1.0e-04")
    for s in ("1.5 × 10⁻⁴",):
        assert s in meta_of("lp-interior-point")["predict"][1]["prompt"]
    for s in ("35.9997", "3.05 × 10⁻⁴", "7.6 × 10⁻⁴"):
        assert s in _explain("lp-interior-point", 2)
    assert "1.0 × 10⁻⁴" in _explain("lp-interior-point", 1) and "(2.0000, 5.9999)" in _explain("lp-interior-point", 1)
    x = _lp_measure("barrier", "farm", c1=3.0, c2=2.0)["x"]
    assert f"({x[0]:.4f}, {x[1]:.4f})" == "(3.0000, 4.5000)"
    # active set: the blocked first step
    st = lp("active_set", "farm_qp")
    assert [f"{v:.3f}" for v in st[1]["x"]] == ["3.103", "4.345"]
    # branch and bound: the order of the incumbents and the pruned bound
    st = lp("branch_bound", "greenhouse")
    assert [(n["x"], n["z"], n["action"]) for n in st[-1]["tree"]][2:] == \
        [([3, 1], 19, "incumbent"), ([4, 0], 20, "incumbent"), ([2, 2], 18, "prune")]
    st = lp("branch_bound", "greenhouse", c1=5.0, c2=5.0)
    assert [n["action"] for n in st[-1]["tree"]] == ["branch", "branch", "incumbent", "prune", "prune"]
    # cutting plane: the points after each cut
    st = lp("cutting_plane", "greenhouse")
    after = [s["x"] for s, nxt in zip(st, st[1:] + [None]) if nxt is None or nxt["phase"] == "cut"]
    assert [[round(v, 2) for v in x] for x in after] == [[3, 1.5], [3.33, 1], [2.67, 1.67], [4, 0]]
    assert len(lp("cutting_plane", "greenhouse", c1=6.0, c2=2.0)) == 2      # start + 1 pivot
    # dynamic programming: release everything every month
    s, total = C.RESERVOIR["s0"], 0.0
    for t in range(6):
        avail = s + C.RESERVOIR["inflow"][t]
        total += C.reservoir_value(t, avail)
        s = 0
    assert total == 32.5 and "32.5" in meta_of("dynamic-programming")["predict"][0]["prompt"]
    assert C.RESERVOIR["benefit"][:4] == [0, 4, 7, 9]
    # Hungarian: the cover after the reduction, and the adjustment amount
    st = api.combo_lab({"method": "hungarian"})["steps"]
    assert (len(st[2]["rows"]), len(st[2]["cols"])) == (3, 1) and "最小值是 1" in st[2]["note"]
    assert "4 條線（3 列 + 1 欄）" in _explain("hungarian", 1)
    # 2-opt lengths
    st = api.combo_lab({"method": "two_opt"})["steps"]
    assert " → ".join(f"{s['length']:.2f}" for s in st) == "32.42 → 26.59 → 22.62 → 20.56 → 18.64"
    assert "32.42 → 26.59 → 22.62 → 20.56 → 18.64" in _explain("two-opt", 1)
    # tabu: first moves, the 2-tour cycle at tenure 0 and the 6-cycle at tenure 2
    st = api.combo_lab({"method": "tabu", "params": {"tenure": 5}})["steps"]
    assert [f"{s['length']:.3f}" for s in st[:3]] == ["18.641", "18.657", "18.195"]
    tours = lambda ten: [predict._canon_tour(s["tour"]) for s in  # noqa: E731
                         api.combo_lab({"method": "tabu", "params": {"tenure": ten}})["steps"]]
    t0 = tours(0)
    assert all(t0[k] == t0[k + 2] for k in range(1, len(t0) - 2))
    t2 = tours(2)
    assert all(t2[k] == t2[k + 6] for k in range(len(t2) - 6))
    assert "6 回合" in _explain("tabu-search", 3) and "15 條" in _explain("tabu-search", 3)


def test_built_widget_pages_embed_questions(tmp_path: Path) -> None:
    """lp-data / combo-data carry the questions; lp.js reads its slider grid from the page."""
    build.build(tmp_path)

    def data(page: str, kind: str) -> dict:
        text = (tmp_path / "algorithms" / f"{page}.html").read_text("utf-8")
        return json.loads(text.split(f'id="{kind}-data">', 1)[1].split("</script>", 1)[0])
    for page, answers in WIDGET_PAGES.items():
        kind, _ = widget_of(page)
        d = data(page, kind)
        assert [q["options"][q["answer"]]["label"] for q in d["predict"]] == answers
    assert data("simplex", "lp")["slider_defs"]["c1"] == build.LP_SLIDERS["c1"]
    assert data("qp-active-set", "lp")["slider_defs"] == {}

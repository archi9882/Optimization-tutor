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

"""Population methods (PSO, GA), renderer-3 trace data and their predicates."""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path

import pytest

from optlab import OPTIMIZERS, make_problem, run_named
from optlab import predict
from optlab.trace import pop_decimals

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "build.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def _found(t: dict, pid: str) -> bool:
    p = make_problem(pid)
    return min(math.dist(t["xs"][-1], m) for m in p.minima) < predict.FOUND_TOL


@pytest.mark.parametrize("algo", ["pso", "ga"])
def test_trace_carries_a_population_per_iterate(algo: str) -> None:
    """pop[k] belongs to xs[k]; points are inside the view and rounded for drawing;
    every f evaluation is counted; particle 0 starts at x0."""
    prob = make_problem("rastrigin")
    params = OPTIMIZERS[algo].defaults()
    t = run_named(algo, "rastrigin", n_iter=30)
    n = params["n_pop"]
    assert len(t["pop"]) == len(t["xs"]) == len(t["fmean"]) == 31
    assert t["status"] == "max_iter" and t["n_iter"] == 30
    d = pop_decimals(prob)
    assert d == 3
    x0, x1, y0, y1 = prob.bounds
    for pop in t["pop"]:
        assert len(pop) == n
        for x, y in pop:
            assert x0 <= x <= x1 and y0 <= y <= y1
            assert round(x, d) == x and round(y, d) == y
    assert t["pop"][0][0] == [round(v, d) for v in prob.x0]
    per_step = n if algo == "pso" else n - params["n_elite"]
    assert t["evals"] == n + 30 * per_step
    assert all(b <= a for a, b in zip(t["fs"], t["fs"][1:]))      # best so far / elitism
    assert t.get("pop_trail", False) == (algo == "pso")
    assert all(fm >= f - 1e-9 for fm, f in zip(t["fmean"], t["fs"]))


@pytest.mark.parametrize("algo", ["pso", "ga"])
def test_same_seed_same_run_other_seed_other_run(algo: str) -> None:
    a = run_named(algo, "ackley", params={"seed": 3}, n_iter=20)
    b = run_named(algo, "ackley", params={"seed": 3}, n_iter=20)
    c = run_named(algo, "ackley", params={"seed": 4}, n_iter=20)
    assert a["xs"] == b["xs"] and a["pop"] == b["pop"]
    assert a["pop"] != c["pop"]


def test_default_settings_find_the_global_minimum_as_the_pages_say() -> None:
    """PSO on Rastrigin and GA on Ackley (page defaults, seed 0, 100 steps)."""
    assert _found(run_named("pso", "rastrigin", n_iter=100), "rastrigin")
    ga = run_named("ga", "ackley", n_iter=100)
    assert _found(ga, "ackley")
    assert 1e-4 < ga["fs"][-1] < 1e-3                  # quoted: 3.1e-4


def test_quoted_success_rates_over_100_seeds() -> None:
    """Headline numbers of the PSO, GA and gallery pages (seeds 0-99, 100 steps)."""
    def rate(algo: str, pid: str, params: dict) -> int:
        return sum(_found(run_named(algo, pid, params=params | {"seed": s}, n_iter=100), pid)
                   for s in range(100))
    assert rate("pso", "rastrigin", {}) == 97
    assert rate("pso", "rastrigin", {"c2": 0.0}) == 0
    assert rate("ga", "ackley", {}) == 100
    assert rate("ga", "ackley", {"pm": 0.0}) == 0


def test_gd_and_nelder_mead_are_trapped_on_rastrigin() -> None:
    """The comparison rows: both local methods stop in a dimple from (2.2, 2.3)."""
    gd = run_named("gd", "rastrigin", params={"lr": 0.002}, n_iter=150)
    nm = run_named("nelder_mead", "rastrigin", params={"size": 0.5}, n_iter=150)
    assert gd["status"] == "converged" and gd["n_iter"] == 19
    assert gd["fs"][-1] == pytest.approx(7.9597, abs=1e-3)
    assert nm["status"] == "stopped" and nm["fs"][-1] == pytest.approx(12.934, abs=1e-2)
    assert "pop" not in gd and "pop" not in nm


def test_pso_stability_region_of_one_particle() -> None:
    """Page derivation: [[1 - phi, w], [-phi, w]] has spectral radius < 1
    exactly when |w| < 1 and 0 < phi < 2 (1 + w)."""
    def radius(w: float, phi: float) -> float:
        b, c = 1 + w - phi, w                # lambda^2 - b lambda + c = 0
        disc = complex(b * b - 4 * c) ** 0.5
        return max(abs((b + disc) / 2), abs((b - disc) / 2))
    for w in (0.0, 0.4, 0.72, 0.9, 0.99, 1.0, 1.1):
        for phi in (0.01, 0.5, 1.5, 3.0, 3.43, 3.45, 3.9, 4.5):
            inside = abs(w) < 1 and 0 < phi < 2 * (1 + w)
            assert (radius(w, phi) < 1) == inside, (w, phi)
    chi = 2 / abs(2 - 4.1 - math.sqrt(4.1 ** 2 - 4 * 4.1))
    assert chi == pytest.approx(0.7298, abs=1e-4) and chi * 2.05 == pytest.approx(1.4962, abs=1e-4)


def test_tournament_win_probability_by_enumeration() -> None:
    """P(rank r wins a binary tournament with replacement) = (2(N - r) + 1) / N^2."""
    from optlab.demos.ga import tournament
    from optlab.exercises._rng import Draws
    N = 20
    pop = [(float(i), 0.0) for i in range(N)]
    fv = [float(i) for i in range(N)]                  # rank r = i + 1
    wins = [0] * N
    for i, j in itertools.product(range(N), repeat=2):
        w = tournament(pop, fv, Draws([(i + 0.5) / N, (j + 0.5) / N]))
        wins[int(w[0])] += 1
    assert wins == [2 * (N - (r + 1)) + 1 for r in range(N)]
    assert wins[0] / N ** 2 == 0.0975


def test_sbx_and_mutation_spread_quoted_on_the_page() -> None:
    eta = 15
    p = (0.5 - 0.5 * 0.9 ** (eta + 1)) + (0.5 - 1 / (2 * 1.1 ** (eta + 1)))
    assert p == pytest.approx(0.80, abs=0.005)
    from optlab.demos.ga import sbx
    from optlab.exercises._rng import Draws
    for u in (0.01, 0.3, 0.5, 0.7, 0.99):
        c1, c2 = sbx((1.0, -2.0), (4.0, 3.0), Draws([u, u]), 15)
        assert c1[0] + c2[0] == pytest.approx(5.0) and c1[1] + c2[1] == pytest.approx(1.0)


# ---------- predicates ----------

def _measure(algo: str, pid: str, **params) -> dict:
    p = make_problem(pid)
    t = run_named(algo, pid, params=params, n_iter=100)
    return predict.measure(t, p.fstar, 1e-8, p)


def test_found_global_and_diversity_measures() -> None:
    m = _measure("pso", "rastrigin")
    assert predict.check({"found_global": True}, m)
    assert not predict.check({"found_global": False}, m)
    m = _measure("pso", "rastrigin", c2=0.0)
    assert predict.check({"found_global": False, "diversity_above": 3.0}, m)
    assert predict.check({"diversity_below": 3.5}, m)
    assert predict.criterion({"diversity_above": 0.5, "diversity_below": 2}, 1e-8) == \
        "0.5 ≤ 最後族群離重心的平均距離 < 2"
    assert "全域最小值" in predict.criterion({"found_global": True}, 1e-8)


def test_found_global_counts_a_diverged_run_as_not_found() -> None:
    m = _measure("gd", "sphere", lr=1.5)
    assert m["status"] == "diverged" and m["global_dist"] is None
    assert predict.check({"found_global": False}, m)


@pytest.mark.parametrize("when,msg", [
    ({"found_global": 1}, "needs true or false"),
    ({"found_global": "yes"}, "needs true or false"),
    ({"diversity_below": -1}, "positive"),
    ({"diversity_above": 0}, "positive"),
    ({"diversity_below": "1"}, "needs a number"),
    ({"diversity_above": 2, "diversity_below": 1}, "leave no possible run"),
])
def test_new_predicates_are_validated(when: dict, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        predict.validate(when)


def test_new_predicates_refuse_runs_they_do_not_apply_to() -> None:
    p = make_problem("rosenbrock")
    t = run_named("gd", "rosenbrock", params={"lr": 0.001}, n_iter=10)
    m = predict.measure(t, p.fstar, 1e-8, p)
    assert "global_dist" not in m and "diversity" not in m
    with pytest.raises(ValueError, match="test function"):
        predict.check({"found_global": True}, m)
    m = _measure("gd", "rastrigin", lr=0.002)
    with pytest.raises(ValueError, match="population"):
        predict.check({"diversity_below": 1.0}, m)


def test_measure_of_existing_problems_is_unchanged() -> None:
    """Pages built before the test functions keep their measured keys."""
    p = make_problem("quadratic", kappa=10)
    t = run_named("gd", "quadratic", {"kappa": 10}, n_iter=20)
    assert set(predict.measure(t, 0.0, 1e-10, p)) == \
        {"status", "steps", "reach", "zigzag", "rises", "final_gap"}


# ---------- gallery build ----------

def test_gallery_rejects_bad_frontmatter() -> None:
    ok = {"id": "rastrigin", "formula": "f", "hard": "h"}
    build.gallery_entries({"functions": [ok]})
    for cfg, msg in [({"functions": []}, "non-empty"),
                     ({"functions": [ok | {"extra": 1}]}, "unknown"),
                     ({"functions": [{"id": "rastrigin", "formula": "f"}]}, "needs 'hard'"),
                     ({"functions": [ok, ok]}, "duplicate"),
                     ({"functions": [ok], "default": "ackley"}, "default")]:
        with pytest.raises(ValueError, match=msg):
            build.gallery_entries(cfg)
    with pytest.raises(KeyError):
        build.gallery_entries({"functions": [ok | {"id": "nope"}]})


def test_gallery_page_cards_and_lazy_grids(tmp_path: Path) -> None:
    """One card per function; only the default grid is inline, the rest are files."""
    meta = build.split_frontmatter((ROOT / "content" / "algorithms" / "test-functions.md")
                                   .read_text("utf-8"))[0]
    ids = [f["id"] for f in meta["gallery"]["functions"]]
    assert len(ids) == 11 and set(ids) >= {"quadratic", "rosenbrock", "himmelblau"}
    html = build.gallery_html(meta["gallery"])
    assert all(f'data-fn="{i}"' in html for i in ids)
    build.build(tmp_path)
    page = (tmp_path / "algorithms" / "test-functions.html").read_text("utf-8")
    data = json.loads(page.split('id="gallery-data">', 1)[1].split("</script>", 1)[0])
    inline = [f["id"] for f in data["functions"] if "grid" in f]
    assert inline == [data["default"]] == ["rastrigin"]
    for f in data["functions"]:
        if "src" in f:
            grid = json.loads((tmp_path / "algorithms" / f["src"]).read_text("utf-8"))
            assert len(grid["z"]) == len(grid["y"]) and len(grid["z"][0]) == len(grid["x"]) == 140


# ---------- graders catch the classic slips ----------

_GA_SLIPS = {
    "no clipping": ("out.append(min(max(xd, lo), hi))", "out.append(xd)"),
    "clip only when mutated": ("            xd = xd + delta * (hi - lo)\n        out.append(min(max(xd, lo), hi))",
                               "            xd = min(max(xd + delta * (hi - lo), lo), hi)\n        out.append(xd)"),
    "tie goes to j": ("fvals[j] < fvals[i]", "fvals[j] <= fvals[i]"),
    "mutates the extra child": (
        "            if len(new_pop) < len(pop):\n                child = mutate(c, rng, pm, eta_m, bounds)",
        "            child = mutate(c, rng, pm, eta_m, bounds)\n            if len(new_pop) < len(pop):"),
    "no r drawn when pc = 0": ("        if rng.random() < pc:", "        if pc > 0 and rng.random() < pc:"),
    "re-evaluates elites": ("    new_f = [fvals[i] for i in elite]", "    new_f = [f(pop[i]) for i in elite]"),
}


@pytest.mark.parametrize("slip", list(_GA_SLIPS))
def test_ga_grader_catches(slip: str) -> None:
    from optlab.exercises import EXERCISES
    from optlab.grading import grade_source
    ex = EXERCISES["ga"]
    old, new = _GA_SLIPS[slip]
    assert ex.solution.count(old) == 1
    res = grade_source(ex, ex.solution.replace(old, new))
    assert not res["passed"]
    assert any(not r["passed"] for r in res["results"] if r["layer"] == 1), slip


def test_pso_grader_catches_in_place_update() -> None:
    from optlab.exercises import EXERCISES
    from optlab.grading import grade_source
    ex = EXERCISES["pso"]
    old = "    new_p, new_pf = list(pbest), list(pbest_f)"
    res = grade_source(ex, ex.solution.replace(old, "    new_p, new_pf = pbest, pbest_f"))
    assert not res["passed"] and any("修改了傳進來的" in r["message"] for r in res["results"])


def test_layer3_compares_the_whole_population(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same best path, different population: layer 3 must fail."""
    from optlab import grading
    from optlab.exercises import EXERCISES
    real = grading.run
    calls = []

    def fake_run(*args, **kw):
        t = real(*args, **kw)
        calls.append(1)
        if len(calls) == 2:                    # the learner's run (the reference runs first)
            t["pop"][-1][0] = [t["pop"][-1][0][0] + 0.5, t["pop"][-1][0][1]]
        return t
    monkeypatch.setattr(grading, "run", fake_run)
    ex = EXERCISES["ga"]
    res = grading.grade_source(ex, ex.solution)
    assert res["user_trace"]["xs"] == res["ref_trace"]["xs"]
    assert not [r for r in res["results"] if r["layer"] == 3][0]["passed"]

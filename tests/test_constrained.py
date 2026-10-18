"""Property tests for the constrained-optimization family."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optlab import api  # noqa: E402
from optlab.demos import augmented_lagrangian as AL  # noqa: E402
from optlab.demos import penalty as PEN  # noqa: E402
from optlab.exercises import EXERCISES  # noqa: E402
from optlab.problems import make_problem, violation  # noqa: E402
from optlab.sets import lmo_polygon, project_disk, project_polygon  # noqa: E402
from optlab.trace import run_named  # noqa: E402

CONSTRAINED = ["fert_budget", "fert_eq", "fert_disk", "reservoir", "rosen_disk"]
INEQ_ONLY = ["fert_budget", "fert_disk", "reservoir", "rosen_disk"]
CONVEX_POLY = ["fert_budget", "reservoir"]


@pytest.mark.parametrize("pid", CONSTRAINED)
def test_listed_optimum_satisfies_kkt(pid: str) -> None:
    """x* is feasible, stationary for the Lagrangian, lambda >= 0, complementary."""
    p = make_problem(pid)
    x = p.minima[0]
    assert violation(p, x) < 1e-12
    r = list(p.grad(x))
    for c, lam in zip(p.constraints, p.multipliers):
        gc = c.grad(x)
        r[0] += lam * gc[0]
        r[1] += lam * gc[1]
        if c.kind == "ineq":
            assert lam >= 0
            assert abs(lam * c.g(x)) < 1e-12
    assert math.hypot(*r) < 1e-9
    assert abs(p.f(x) - p.fstar) < 1e-12


def test_hand_derived_values() -> None:
    p = make_problem("fert_budget")
    assert p.minima[0] == (1.5, 1.5) and p.multipliers[0] == 3.0 and p.fstar == 3.0
    r = make_problem("reservoir")
    assert r.minima[0] == pytest.approx((5 / 3, 1.0)) and r.multipliers[1] == pytest.approx(4 / 3)


def _penalty_minimiser(rho: float) -> tuple:
    """Run the displayed penalty step at fixed rho until ||grad P|| < 2e-7.

    (Armijo backtracking in double precision stalls near 1.3e-7 at rho = 10.)"""
    p = make_problem("fert_budget")
    x = p.x0
    for _ in range(20000):
        x, gnorm = PEN.penalty_step(x, rho, p.f, p.grad, p.constraints)
        if gnorm < 2e-7:
            return x
    raise AssertionError("inner loop did not converge")


def test_penalty_error_is_order_one_over_rho() -> None:
    """Distance to x* and violation shrink ~10x per decade of rho; rho*g -> lambda*."""
    dists = []
    for rho in (10.0, 100.0, 1000.0):
        x = _penalty_minimiser(rho)
        g = x[0] + x[1] - 3
        assert g > 0                                     # approached from outside
        assert g == pytest.approx(6 / (3 + 2 * rho), abs=1e-7)   # hand formula
        assert abs(rho * g - 3) * rho < 5                # rho*g = 3 - O(1/rho)
        dists.append(math.dist(x, (1.5, 1.5)))
    assert 8.5 < dists[0] / dists[1] < 11 and 8.5 < dists[1] / dists[2] < 11


@pytest.mark.parametrize("pid", CONSTRAINED)
def test_augmented_lagrangian_multiplier_converges_to_kkt(pid: str) -> None:
    p = make_problem(pid)
    t = run_named("augmented_lagrangian", pid, params={"rho": 100}, n_iter=3000)
    assert t["status"] == "stopped"
    lam = t["info"][-1]["λ"]
    assert max(abs(a - b) for a, b in zip(lam, p.multipliers)) < 1e-5
    assert math.dist(t["xs"][-1], p.minima[0]) < 1e-5


def test_augmented_lagrangian_rate_matches_hand_formula() -> None:
    """Multiplier error shrinks by 1/(1 + 2 rho/3) per update (exact-ish inner solves)."""
    t = run_named("augmented_lagrangian", "fert_budget", params={"rho": 10}, n_iter=400)
    lams = [t["info"][i]["λ"][0] for i in range(len(t["xs"])) if t["outer"][i]]
    errs = [la - 3 for la in lams]
    assert lams[0] == pytest.approx(60 / 23, abs=1e-6)   # first round = penalty with rho = 10
    for a, b in zip(errs[:3], errs[1:4]):
        assert b / a == pytest.approx(3 / 23, abs=1e-3)
    assert AL.INNER_TOL > 0


@pytest.mark.parametrize("pid", INEQ_ONLY)
def test_barrier_iterates_are_strictly_feasible(pid: str) -> None:
    p = make_problem(pid)
    t = run_named("barrier_nl", pid, n_iter=300)
    assert t["status"] == "stopped"
    for x in t["xs"]:
        assert all(c.g(tuple(x)) < 0 for c in p.constraints)
    assert math.dist(t["xs"][-1], p.minima[0]) < 1e-5


def test_barrier_gap_bound_on_central_path() -> None:
    """Convex problem: f(x(mu)) - f* <= m mu at the end of every outer iteration."""
    p = make_problem("fert_budget")
    t = run_named("barrier_nl", "fert_budget", n_iter=300)
    ends = [i for i in range(len(t["xs"])) if t["outer"][i]]
    assert len(ends) == 7
    for i in ends:
        mu = t["info"][i]["μ"]
        assert p.f(tuple(t["xs"][i])) - p.fstar <= 3 * mu


def test_barrier_refuses_infeasible_start() -> None:
    t = run_named("barrier_nl", "fert_budget", x0=(2.5, 2.5), n_iter=50)
    assert t["status"] == "stopped" and t["n_iter"] == 1 and "起點" in t["message"]


@pytest.mark.parametrize("pid", INEQ_ONLY)
@pytest.mark.parametrize("x0", [None, (3.0, 3.0)])
def test_projected_gradient_iterates_stay_feasible(pid: str, x0) -> None:
    p = make_problem(pid)
    lr = 0.002 if pid == "rosen_disk" else 0.1
    t = run_named("projected_gradient", pid, params={"lr": lr}, x0=x0, n_iter=2000)
    assert all(violation(p, tuple(x)) < 1e-12 for x in t["xs"][1:])
    assert t["status"] == "stopped"
    assert math.dist(t["xs"][-1], p.minima[0]) < 1e-8


@pytest.mark.parametrize("pid", CONVEX_POLY + ["fert_disk"])
def test_frank_wolfe_gap_bounds_error_and_decreases(pid: str) -> None:
    p = make_problem(pid)
    t = run_named("frank_wolfe", pid, n_iter=1000)
    gaps = [i["gap"] for i in t["info"][1:]]
    for x, i in zip(t["xs"][:-1], t["info"][1:]):        # gap is computed at the old x
        assert violation(p, tuple(x)) < 1e-12
        assert i["gap"] >= p.f(tuple(x)) - p.fstar - 1e-12
    if t["status"] != "stopped":
        assert max(gaps[-100:]) < max(gaps[100:200]) / 4
        assert min(gaps[-100:]) < 0.05


def test_frank_wolfe_zigzags_on_a_face_with_order_one_over_k() -> None:
    t = run_named("frank_wolfe", "fert_budget", n_iter=1000)
    s = [tuple(i["頂點 s"]) for i in t["info"][1:]]
    assert set(s) == {(0.0, 3.0), (3.0, 0.0)}
    assert all(a != b for a, b in zip(s, s[1:]))         # strict alternation
    for k in (100, 300, 1000):
        assert 2.0 < k * math.dist(t["xs"][k], (1.5, 1.5)) < 2.2


def test_projection_and_oracle_helpers() -> None:
    rng = random.Random(0)
    box = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    for _ in range(200):
        y = (rng.uniform(-3, 5), rng.uniform(-3, 4))
        assert project_polygon(y, box) == pytest.approx((min(2, max(0, y[0])), min(1, max(0, y[1]))))
        p = project_disk(y, (1.0, 1.0), 1.0)
        assert math.dist(p, (1, 1)) <= 1 + 1e-12
        g = (rng.uniform(-1, 1), rng.uniform(-1, 1))
        v = lmo_polygon(g, box)
        assert all(g[0] * v[0] + g[1] * v[1] <= g[0] * w[0] + g[1] * w[1] for w in box)


def test_grid_carries_constraints_for_the_renderer() -> None:
    grid = api.problem_grid({"problem": "fert_budget"})
    assert [c["kind"] for c in grid["constraints"]] == ["ineq"] * 3
    j, i = len(grid["y"]) - 1, len(grid["x"]) - 1          # top-right corner (4, 3.5)
    assert grid["constraints"][0]["z"][j][i] > 0
    assert "constraints" not in api.problem_grid({"problem": "quadratic"})


def test_lp_barrier_exercise_is_kept_separate() -> None:
    assert EXERCISES["barrier"].lp_preset == "farm"
    assert EXERCISES["barrier_nl"].func_name == "barrier_step"
    for ex_id in ("barrier", "barrier_nl"):
        ex = EXERCISES[ex_id]
        from optlab.grading import grade_source
        assert grade_source(ex, ex.solution)["passed"]

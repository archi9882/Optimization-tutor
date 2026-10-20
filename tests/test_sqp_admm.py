"""Property tests for SQP and ADMM (second half of the constrained family)."""
from __future__ import annotations

import math

import pytest

from optlab import api, predict
from optlab.demos import admm as ADMM
from optlab.demos import sqp as SQP
from optlab.problems import make_problem, violation
from optlab.trace import run, run_named

QP_LIKE = ["fert_budget", "fert_eq", "reservoir"]          # quadratic f, linear constraints
SQP_PROBLEMS = QP_LIKE + ["fert_disk", "rosen_disk", "maratos"]
ADMM_PROBLEMS = ["fert_lasso", "fert_budget", "fert_disk", "reservoir"]


# ---------- the two new problems ----------

def test_maratos_optimum_satisfies_kkt() -> None:
    p = make_problem("maratos")
    x, lam = p.minima[0], p.multipliers[0]
    g, gc = p.grad(x), p.constraints[0].grad(x)
    assert violation(p, x) == 0 and lam == -1.5 and p.fstar == p.f(x) == -1.0
    assert math.hypot(g[0] + lam * gc[0], g[1] + lam * gc[1]) < 1e-15


def test_lasso_optimum_satisfies_subgradient_condition() -> None:
    p = make_problem("fert_lasso")
    x = p.minima[0]
    g = p.grad(x)                                   # smooth part only
    assert x == (5 / 3, 0.0) and p.l1 == 3.0
    assert g[0] + p.l1 * 1.0 == pytest.approx(0, abs=1e-14)    # x1 > 0: s' + λ = 0
    assert abs(g[1]) == pytest.approx(8 / 3) and abs(g[1]) <= p.l1   # x2 = 0: |s'| ≤ λ
    # brute force on a fine grid around x*: nothing lower
    best = min(p.f((x[0] + i / 400, x[1] + j / 400)) for i in range(-40, 41) for j in range(-40, 41))
    assert best >= p.fstar - 1e-12


def test_lasso_grid_plots_distance_and_marks_the_smooth_minimum() -> None:
    grid = api.problem_grid({"problem": "fert_lasso"})
    assert grid["distance"] is True and "constraints" not in grid
    assert grid["unconstrained"] == [2.0, 1.0] and "ℓ1" in grid["unconstrained_label"]
    assert "distance" not in api.problem_grid({"problem": "fert_budget"})


@pytest.mark.parametrize("algo", ["augmented_lagrangian", "projected_gradient", "penalty", "sqp"])
def test_methods_without_l1_support_refuse_the_lasso(algo: str) -> None:
    t = run_named(algo, "fert_lasso", n_iter=50)
    assert t["status"] == "stopped" and t["n_iter"] == 0 and "ℓ1" in t["message"]


# ---------- SQP ----------

@pytest.mark.parametrize("pid", QP_LIKE)
def test_sqp_solves_a_qp_in_one_step(pid: str) -> None:
    """Quadratic f + linear constraints: the QP subproblem is the problem itself."""
    p = make_problem(pid)
    t = run_named("sqp", pid, n_iter=30)
    assert t["status"] == "stopped" and t["n_iter"] == 2
    assert math.dist(t["xs"][1], p.minima[0]) < 1e-14
    assert t["info"][-1]["λ"] == pytest.approx(list(p.multipliers), abs=1e-12)


@pytest.mark.parametrize("pid", SQP_PROBLEMS)
@pytest.mark.parametrize("algo,most", [("sqp", 16), ("sqp_bfgs", 23)])
def test_sqp_needs_few_steps_and_ends_at_a_kkt_point(pid: str, algo: str, most: int) -> None:
    p = make_problem(pid)
    t = run_named(algo, pid, n_iter=100)
    assert t["status"] == "stopped" and t["n_iter"] <= most
    x, lam = tuple(t["xs"][-1]), t["info"][-1]["λ"]
    assert math.dist(x, p.minima[0]) < 1e-12 and violation(p, x) < 1e-12
    r = SQP.grad_lagrangian(x, lam, p.grad, p.constraints)       # KKT residual
    assert math.hypot(*r) < 1e-9
    assert lam == pytest.approx(list(p.multipliers), abs=1e-8)
    for c, li in zip(p.constraints, lam):
        if c.kind == "ineq":
            assert li >= 0 and abs(li * c.g(x)) < 1e-12


def test_sqp_converges_quadratically_on_the_disk() -> None:
    p = make_problem("fert_disk")
    t = run_named("sqp", "fert_disk", n_iter=30)
    e = [math.dist(x, p.minima[0]) for x in t["xs"]]
    assert [i["步長 t"] for i in t["info"][1:]] == [1.0] * 7
    tail = [(a, b) for a, b in zip(e, e[1:]) if 1e-10 < a < 0.1]
    assert len(tail) == 3 and all(b <= 3 * a * a for a, b in tail)
    assert [f"{v:.2g}" for v in e[3:7]] == ["0.051", "0.0012", "7.6e-07", "1.3e-12"]


@pytest.mark.parametrize("pid", SQP_PROBLEMS)
def test_sqp_merit_decreases_every_step(pid: str) -> None:
    """The accepted point satisfies the Armijo test on the l1 merit with that step's μ."""
    p = make_problem(pid)
    t = run_named("sqp", pid, n_iter=50)
    for k in range(1, len(t["xs"])):
        mu = t["info"][k]["μ"]
        a, b = tuple(t["xs"][k - 1]), tuple(t["xs"][k])
        assert SQP.merit(b, mu, p.f, p.constraints) <= SQP.merit(a, mu, p.f, p.constraints) + 1e-12


def test_damped_bfgs_stays_positive_definite() -> None:
    p = make_problem("rosen_disk")
    step = SQP.bfgs_adapter(SQP.FUNC)
    x, state = p.x0, {}
    for _ in range(23):
        x, state = step(x, p, state, {})
        (a, b), (_, d) = state["B"]
        assert a > 0 and a * d - b * b > 0


def test_hessian_modification_on_rosenbrock() -> None:
    p = make_problem("rosen_disk")
    _, tau = SQP.hessian_lagrangian((-0.6, 0.4), [0.0], p.hess, p.constraints)
    assert tau == pytest.approx(5.84 + 1e-4, abs=0.01)
    raw = run_named("sqp_raw", "rosen_disk", x0=(-0.6, 0.4), n_iter=50)
    assert raw["n_iter"] == 1 and raw["xs"][-1] == [-0.6, 0.4] and "線搜尋" in raw["message"]
    mod = run_named("sqp", "rosen_disk", x0=(-0.6, 0.4), n_iter=50)
    assert mod["n_iter"] == 13 and mod["info"][1]["修正 τ"] == pytest.approx(tau)


def test_infeasible_linearisation_stops_at_the_centre() -> None:
    t = run_named("sqp", "maratos", x0=(0.0, 0.0), n_iter=20)
    assert t["n_iter"] == 1 and t["xs"] == [[0.0, 0.0], [0.0, 0.0]] and "QP" in t["message"]


def test_maratos_effect_full_steps_would_be_faster() -> None:
    """Line search on the l1 merit cuts the (quadratically convergent) full step."""
    p = make_problem("maratos")
    x0 = (0.99875, 0.04998)
    ls = run_named("sqp", "maratos", x0=x0, n_iter=50)
    assert ls["n_iter"] == 10
    assert [i["步長 t"] for i in ls["info"][1:7]] == [0.5, 0.125, 0.125, 0.25, 0.25, 0.5]

    def full_step(x, B, mu, f, grad, cons):
        d, lam = SQP.qp_subproblem(B, grad(x), [(c.kind, c.grad(x), c.g(x)) for c in cons])
        return (x[0] + d[0], x[1] + d[1]), lam, mu, 1.0, d

    full = run(SQP.adapter(full_step), p, x0, {}, 50)
    assert full["n_iter"] == 5
    x1 = tuple(full["xs"][1])
    h = p.constraints[0].g
    assert p.f(x1) < p.f(x0) and abs(h(x1)) == pytest.approx(1.56e-4, rel=0.01)
    assert abs(h(x0)) == pytest.approx(4.4e-7, rel=0.01)
    mu = ls["info"][1]["μ"]
    assert SQP.merit(x1, mu, p.f, p.constraints) > SQP.merit(x0, mu, p.f, p.constraints)


def test_sqp_can_converge_to_a_maximum() -> None:
    t = run_named("sqp", "maratos", x0=(-0.5, 0.0), n_iter=50)
    assert t["n_iter"] == 6 and t["xs"][-1] == pytest.approx([-1.0, 0.0], abs=1e-12)
    off = run_named("sqp", "maratos", x0=(-0.5, 0.05), n_iter=100)
    assert off["n_iter"] == 52 and math.dist(off["xs"][-1], (1, 0)) < 1e-12


# ---------- ADMM ----------

def _admm(pid: str, rho: float, n_iter: int = 500) -> dict:
    return run_named("admm", pid, params={"rho": rho}, n_iter=n_iter)


@pytest.mark.parametrize("pid", ADMM_PROBLEMS)
@pytest.mark.parametrize("rho", [0.3, 1.0, 3.0, 10.0])
def test_admm_residuals_vanish_and_z_is_always_feasible(pid: str, rho: float) -> None:
    p = make_problem(pid)
    t = _admm(pid, rho)
    assert t["status"] == "stopped"
    last = t["info"][-1]
    assert last["原始殘差 r"] < ADMM.TOL and last["對偶殘差 s"] < ADMM.TOL
    assert math.dist(t["xs"][-1], p.minima[0]) < 1e-8
    assert math.dist(t["zs"][-1], p.minima[0]) < 1e-8
    assert len(t["zs"]) == len(t["xs"]) and t["zs"][0] == t["xs"][0]
    assert all(violation(p, tuple(z)) < 1e-12 for z in t["zs"][1:])


def test_admm_x_iterates_leave_the_triangle() -> None:
    p = make_problem("fert_budget")
    t = _admm("fert_budget", 1.0)
    assert max(violation(p, tuple(x)) for x in t["xs"]) == pytest.approx(1.24, abs=0.005)


def test_admm_soft_threshold_gives_exact_zeros_in_z_only() -> None:
    t = _admm("fert_lasso", 1.0)
    assert t["n_iter"] == 34
    assert all(z[1] == 0.0 for z in t["zs"][1:])
    assert all(x[1] != 0.0 for x in t["xs"][1:])
    assert t["zs"][1] == t["zs"][2] == [0.0, 0.0]


def test_rho_too_small_or_too_large_is_slow() -> None:
    steps = {rho: _admm("fert_lasso", rho)["n_iter"] for rho in
             (0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0)}
    assert steps == {0.1: 167, 0.3: 58, 1.0: 34, 2.0: 35, 3.0: 37, 5.0: 51, 10.0: 105,
                     30.0: 303, 100.0: 500}
    big = _admm("fert_lasso", 100.0)
    assert math.dist(big["xs"][-1], (5 / 3, 0)) == pytest.approx(6.2e-4, rel=0.01)
    # rho large: r tiny but s large; rho small: s zero but r large
    assert big["info"][10]["原始殘差 r"] < 1e-15 and big["info"][10]["對偶殘差 s"] > 1
    small = _admm("fert_lasso", 0.1)
    assert all(small["info"][k]["對偶殘差 s"] == 0 for k in range(2, 21))
    assert small["info"][20]["原始殘差 r"] > 1


@pytest.mark.parametrize("pid", ["fert_budget", "fert_disk", "reservoir"])
def test_admm_multiplier_matches_the_augmented_lagrangian_fixed_point(pid: str) -> None:
    """At the fixed point rho*u = sum_i lambda_i grad g_i(x*) = -grad f(x*), the
    multipliers the method of multipliers (joint minimisation) converges to."""
    p = make_problem(pid)
    al = run_named("augmented_lagrangian", pid, params={"rho": 100}, n_iter=3000)
    lam = al["info"][-1]["λ"]
    x = p.minima[0]
    target = [sum(li * c.grad(x)[i] for c, li in zip(p.constraints, lam)) for i in range(2)]
    rho_u = _admm(pid, 3.0)["info"][-1]["ρu"]
    assert rho_u == pytest.approx(target, abs=1e-6)
    assert rho_u == pytest.approx([-v for v in p.grad(x)], abs=1e-6)


def test_admm_lasso_multiplier_is_a_subgradient() -> None:
    rho_u = _admm("fert_lasso", 1.0)["info"][-1]["ρu"]
    assert rho_u == pytest.approx([3.0, 8 / 3], abs=1e-6)


def test_admm_x_update_is_exact_for_a_quadratic() -> None:
    p = make_problem("fert_budget")
    x = ADMM.x_update((0.0, 0.0), 2.0, p.grad, p.hess, (10.0, -4.0))
    g = p.grad(x)
    assert math.hypot(g[0] + 2 * x[0], g[1] + 2 * x[1]) < 1e-12


# ---------- predictions on an l1 problem ----------

def test_lasso_measure_has_distance_but_no_violation() -> None:
    p = make_problem("fert_lasso")
    m = predict.measure(_admm("fert_lasso", 1.0, 150), p.fstar, 1e-10, p)
    assert m["dist"] < 1e-9 and "violation" not in m
    assert "違反量" not in predict.summary(m, 1e-10)
    with pytest.raises(ValueError, match="constraints to violate"):
        predict.check({"stays_feasible": True}, m)

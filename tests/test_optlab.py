"""Tests for problems, optimizers and the exercise grader."""
from __future__ import annotations

import math
import random

import pytest

from optlab import OPTIMIZERS, make_problem, run, run_named
from optlab.exercises import EXERCISES, grade_json
from optlab.grading import grade, grade_source

PROBLEMS = [("quadratic", {"kappa": 10, "theta_deg": 30}),
            ("rosenbrock", {}), ("himmelblau", {})]


@pytest.mark.parametrize("pid,params", PROBLEMS)
def test_gradient_matches_finite_difference(pid: str, params: dict) -> None:
    """Analytic gradients agree with central differences."""
    prob = make_problem(pid, **params)
    rng = random.Random(0)
    h = 1e-6
    for _ in range(20):
        x = (rng.uniform(-2, 2), rng.uniform(-2, 2))
        g = prob.grad(x)
        fd = ((prob.f((x[0] + h, x[1])) - prob.f((x[0] - h, x[1]))) / (2 * h),
              (prob.f((x[0], x[1] + h)) - prob.f((x[0], x[1] - h))) / (2 * h))
        assert g == pytest.approx(fd, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("pid,params", PROBLEMS)
def test_hessian_matches_finite_difference(pid: str, params: dict) -> None:
    """Analytic Hessians agree with differences of the gradient."""
    prob = make_problem(pid, **params)
    rng = random.Random(1)
    h = 1e-6
    for _ in range(20):
        x = (rng.uniform(-2, 2), rng.uniform(-2, 2))
        H = prob.hess(x)
        for j, e in enumerate([(h, 0.0), (0.0, h)]):
            gp = prob.grad((x[0] + e[0], x[1] + e[1]))
            gm = prob.grad((x[0] - e[0], x[1] - e[1]))
            col = ((gp[0] - gm[0]) / (2 * h), (gp[1] - gm[1]) / (2 * h))
            assert (H[0][j], H[1][j]) == pytest.approx(col, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("pid,params", PROBLEMS)
def test_listed_minima_are_stationary(pid: str, params: dict) -> None:
    """Every listed minimum has (near) zero gradient and f = f*."""
    prob = make_problem(pid, **params)
    for m in prob.minima:
        assert math.hypot(*prob.grad(m)) < 1e-4
        assert prob.f(m) == pytest.approx(prob.fstar, abs=1e-8)


def test_quadratic_condition_number_is_kappa() -> None:
    """Eigenvalues of the rotated quadratic's Hessian are (1, kappa)."""
    (a, b), (_, d) = make_problem("quadratic", kappa=25, theta_deg=40).hess((0, 0))
    tr, det = a + d, a * d - b * b
    disc = math.sqrt(tr * tr / 4 - det)
    assert (tr / 2 - disc, tr / 2 + disc) == pytest.approx((1.0, 25.0))


def test_gd_matches_theoretical_rate_on_quadratic() -> None:
    """With lr = 2/(L+mu), ||x_k|| <= ((k-1)/(k+1))^k ||x_0|| exactly."""
    kappa = 10.0
    prob = make_problem("quadratic", kappa=kappa, theta_deg=20)
    lr = 2.0 / (kappa + 1.0)
    out = run(OPTIMIZERS["gd"].step, prob, prob.x0, {"lr": lr}, n_iter=50)
    r = (kappa - 1) / (kappa + 1)
    n0 = math.hypot(*out["xs"][0])
    for k, x in enumerate(out["xs"]):
        assert math.hypot(*x) <= r ** k * n0 * (1 + 1e-9)


def test_gd_diverges_above_two_over_l() -> None:
    """GD on a quadratic is unstable once lr > 2/L."""
    out = run_named("gd", "quadratic", {"kappa": 10}, {"lr": 0.21}, n_iter=500)
    assert out["status"] == "diverged"
    ok = run_named("gd", "quadratic", {"kappa": 10}, {"lr": 0.19}, n_iter=500)
    assert ok["fs"][-1] < 1e-8


@pytest.mark.parametrize("algo", ["gd", "momentum", "nesterov", "rmsprop",
                                  "adam"])
def test_family_converges_on_quadratic(algo: str) -> None:
    """Default hyper-parameters reach the bowl's bottom (loosely)."""
    out = run_named(algo, "quadratic", {"kappa": 10}, n_iter=2000)
    assert out["status"] != "diverged"
    assert math.hypot(*out["xs"][-1]) < 0.05


def test_momentum_beats_gd_on_ill_conditioned_quadratic() -> None:
    """Tuned heavy ball needs far fewer steps than tuned GD at kappa = 100."""
    kappa = 100.0
    lr_gd = 2 / (kappa + 1)
    lr_hb = 4 / (math.sqrt(kappa) + 1) ** 2
    beta = ((math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)) ** 2

    def steps_to(algo: str, params: dict) -> int:
        out = run_named(algo, "quadratic", {"kappa": kappa}, params, n_iter=5000)
        return next(i for i, f in enumerate(out["fs"]) if f < 1e-8)

    assert steps_to("momentum", {"lr": lr_hb, "beta": beta}) * 4 < \
        steps_to("gd", {"lr": lr_gd})


def test_adam_first_step_has_length_lr_per_coordinate() -> None:
    """Bias correction makes Adam's first step exactly lr * sign(g)."""
    prob = make_problem("rosenbrock")
    x0 = (-1.5, 2.5)
    x1, _ = OPTIMIZERS["adam"].step(x0, prob, {}, OPTIMIZERS["adam"].defaults())
    lr = OPTIMIZERS["adam"].defaults()["lr"]
    g = prob.grad(x0)
    for i in range(2):
        assert x0[i] - x1[i] == pytest.approx(lr * math.copysign(1, g[i]),
                                              rel=1e-6)


@pytest.mark.parametrize("ex_id", list(EXERCISES))
def test_reference_solution_passes(ex_id: str) -> None:
    """The shipped solution passes every layer of its own exercise."""
    ex = EXERCISES[ex_id]
    result = grade_source(ex, ex.solution)
    assert result["passed"], result["results"]


@pytest.mark.parametrize("ex_id", list(EXERCISES))
def test_skeleton_fails_cleanly(ex_id: str) -> None:
    """The untouched skeleton fails with messages, never with a crash."""
    result = grade_source(EXERCISES[ex_id], EXERCISES[ex_id].skeleton)
    assert not result["passed"]
    assert all("NotImplementedError" in r["message"]
               for r in result["results"] if r["layer"] == 1)


def test_momentum_with_old_velocity_is_caught() -> None:
    """Moving x with the *old* v is a classic bug; layer 1 must catch it."""
    def buggy(x, v, grad, lr, beta):
        x_new = (x[0] - lr * v[0], x[1] - lr * v[1])
        return x_new, (beta * v[0] + grad[0], beta * v[1] + grad[1])

    result = grade(EXERCISES["momentum"], buggy)
    assert not result["passed"]
    assert not result["results"][0]["passed"]


def test_syntax_error_and_missing_function_are_reported() -> None:
    """Broken source returns a layer-0 failure instead of raising."""
    ex = EXERCISES["gd"]
    assert grade_source(ex, "def gd_step(:")["results"][0]["layer"] == 0
    assert "gd_step" in grade_source(ex, "x = 1")["results"][0]["message"]


def test_numpy_style_returns_are_accepted() -> None:
    """Returning lists (or arrays) instead of tuples is fine."""
    src = "def gd_step(x, grad, lr):\n    return [x[0]-lr*grad[0], x[1]-lr*grad[1]]\n"
    assert grade_source(EXERCISES["gd"], src)["passed"]


def test_grade_json_round_trips() -> None:
    """The browser entry point returns valid JSON."""
    import json
    data = json.loads(grade_json("gd", EXERCISES["gd"].solution))
    assert data["passed"] and data["user_trace"]["xs"]

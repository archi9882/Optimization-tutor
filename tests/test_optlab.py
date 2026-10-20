"""Tests for problems, optimizers and the exercise grader."""
from __future__ import annotations

import math
import random

import pytest

from optlab import OPTIMIZERS, make_problem, run, run_named
from optlab.exercises import EXERCISES, grade_json
from optlab.grading import grade, grade_source

CONTINUOUS = [k for k, ex in EXERCISES.items() if ex.check is None]

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


@pytest.mark.parametrize("ex_id", CONTINUOUS)
def test_exercise_reference_matches_lab_optimizer(ex_id: str) -> None:
    """Practice reference and the animated optimizer give identical traces."""
    ex = EXERCISES[ex_id]
    prob = make_problem(ex.demo["problem"], **ex.demo.get("problem_params", {}))
    params = OPTIMIZERS[ex_id].defaults() | ex.demo["params"]
    a = run(ex.adapter(ex.reference), prob, prob.x0, params, 100)
    b = run(OPTIMIZERS[ex_id].step, prob, prob.x0, params, 100)
    assert a["xs"] == b["xs"]


def test_sgd_is_reproducible_and_decay_helps() -> None:
    """Same seed -> same path; decaying lr ends closer than constant lr."""
    base = {"lr": 0.15, "sigma": 1.0, "seed": 0}
    a = run_named("sgd", "quadratic", {"kappa": 10}, base | {"decay": 0.0}, n_iter=400)
    b = run_named("sgd", "quadratic", {"kappa": 10}, base | {"decay": 0.0}, n_iter=400)
    assert a["xs"] == b["xs"]
    c = run_named("sgd", "quadratic", {"kappa": 10}, base | {"decay": 0.05}, n_iter=400)
    tail = lambda t: sum(t["fs"][-100:]) / 100  # noqa: E731
    assert tail(c) < tail(a)


def test_coordinate_descent_is_exact_on_axis_aligned_quadratic() -> None:
    """With omega = 1 and no coupling, two coordinate steps hit the minimum."""
    out = run_named("coordinate", "quadratic", {"kappa": 10, "theta_deg": 0},
                    {"omega": 1.0}, n_iter=10)
    assert out["status"] == "converged" and out["n_iter"] == 2


def test_sor_beats_gauss_seidel_on_coupled_quadratic() -> None:
    """Over-relaxation (omega = 1.5) needs fewer sweeps than omega = 1."""
    def steps(omega: float) -> int:
        out = run_named("coordinate", "quadratic",
                        {"kappa": 50, "theta_deg": 30}, {"omega": omega},
                        n_iter=2000)
        return next(i for i, f in enumerate(out["fs"]) if f < 1e-10)
    assert steps(1.5) < steps(1.0)


def test_nesterov_gradient_is_taken_at_look_ahead_point() -> None:
    """Nesterov differs from heavy ball once velocity is non-zero."""
    p = {"lr": 0.05, "beta": 0.8}
    hb = run_named("momentum", "quadratic", {"kappa": 10}, p, n_iter=5)
    nag = run_named("nesterov", "quadratic", {"kappa": 10}, p, n_iter=5)
    assert hb["xs"][1] == nag["xs"][1]   # v = 0 on the first step
    assert hb["xs"][2] != nag["xs"][2]


@pytest.mark.parametrize("algo,lr", [("adam", 0.1), ("adagrad", 0.5)])
def test_adaptive_methods_are_not_rotation_invariant(algo: str, lr: float) -> None:
    """Per-coordinate scaling helps on axis-aligned valleys, not rotated ones."""
    def final_f(theta: float) -> float:
        return run_named(algo, "quadratic", {"kappa": 10, "theta_deg": theta},
                         {"lr": lr}, n_iter=300)["fs"][-1]
    assert final_f(0) * 100 < final_f(45)


def test_gd_is_rotation_invariant() -> None:
    """GD's f-trajectory does not depend on how the valley is rotated."""
    a = run_named("gd", "quadratic", {"kappa": 10, "theta_deg": 0}, {"lr": 0.1}, n_iter=50)
    b = run_named("gd", "quadratic", {"kappa": 10, "theta_deg": 45}, {"lr": 0.1}, n_iter=50)
    # Same start point is not rotated with the valley, so compare rates instead.
    ra = (a["fs"][-1] / a["fs"][-11]) ** 0.1
    rb = (b["fs"][-1] / b["fs"][-11]) ** 0.1
    assert ra == pytest.approx(rb, rel=1e-6)


@pytest.mark.parametrize("ex_id", CONTINUOUS)
def test_displayed_demo_source_is_what_the_lab_runs(ex_id: str) -> None:
    """exec() the code shown on the page; it must reproduce the lab trace."""
    from optlab import demos
    ns: dict = {}
    exec(demos.source(ex_id), ns)
    fn = ns[EXERCISES[ex_id].func_name]
    ex = EXERCISES[ex_id]
    prob = make_problem(ex.demo["problem"], **ex.demo.get("problem_params", {}))
    params = OPTIMIZERS[ex_id].defaults() | ex.demo["params"]
    shown = run(demos.MODULES[ex_id].adapter(fn), prob, prob.x0, params, 100)
    lab = run(OPTIMIZERS[ex_id].step, prob, prob.x0, params, 100)
    assert shown["xs"] == lab["xs"]


@pytest.mark.parametrize("ex_id", list(EXERCISES))
def test_every_blank_is_unique_and_matters(ex_id: str) -> None:
    """Each blank appears once in the solution, and a wrong fill fails."""
    ex = EXERCISES[ex_id]
    assert ex.blanks, "exercise needs at least one blank"
    for blank in ex.blanks:
        assert ex.solution.count(blank) == 1, blank
        broken = ex.solution.replace(blank, "0")
        assert not grade_source(ex, broken)["passed"], blank


def test_every_demo_has_pseudocode() -> None:
    """Each demo module ships non-empty pseudocode."""
    from optlab import demos
    for demo_id, mod in demos.MODULES.items():
        assert mod.PSEUDOCODE.strip(), demo_id


# ---------- second-order family ----------

@pytest.mark.parametrize("algo", ["newton", "gauss_newton"])
def test_newton_type_methods_solve_a_quadratic_in_one_step(algo: str) -> None:
    """The quadratic model is exact, so one full step lands on the minimum."""
    out = run_named(algo, "quadratic", {"kappa": 50, "theta_deg": 30}, n_iter=5)
    assert out["n_iter"] == 1 and out["fs"][-1] < 1e-20


def test_newton_converges_to_a_maximum_on_himmelblau() -> None:
    """Newton seeks grad f = 0, not a minimum: from (-0.5,-0.5) it finds the local max."""
    out = run_named("newton", "himmelblau", n_iter=50)
    prob = make_problem("himmelblau")
    x = tuple(out["xs"][-1])
    assert out["status"] == "converged"
    assert x == pytest.approx((-0.270845, -0.923039), abs=1e-5)
    (h11, h12), (_, h22) = prob.hess(x)
    assert h11 < 0 and h11 * h22 - h12 * h12 > 0      # negative definite


def test_gauss_newton_stalls_where_lm_succeeds() -> None:
    """J is singular at Himmelblau's (-0.5, -0.5); LM's lambda*I fixes it."""
    gn = run_named("gauss_newton", "himmelblau", n_iter=50)
    assert all(x == [-0.5, -0.5] for x in gn["xs"])
    lm = run_named("lm", "himmelblau", n_iter=50)
    assert lm["status"] == "converged" and lm["fs"][-1] < 1e-20


def test_bfgs_update_satisfies_the_secant_equation() -> None:
    """After an update, H_new @ y == s exactly (up to rounding)."""
    from optlab.demos.bfgs import bfgs_step
    prob = make_problem("rosenbrock")
    x0 = (-1.5, 2.5)
    x1, H = bfgs_step(x0, [[1.0, 0.0], [0.0, 1.0]], prob.f, prob.grad)
    s = (x1[0] - x0[0], x1[1] - x0[1])
    g0, g1 = prob.grad(x0), prob.grad(x1)
    y = (g1[0] - g0[0], g1[1] - g0[1])
    Hy = (H[0][0] * y[0] + H[0][1] * y[1], H[1][0] * y[0] + H[1][1] * y[1])
    assert Hy == pytest.approx(s, rel=1e-9, abs=1e-12)


def test_conjugate_gradient_finishes_a_2d_quadratic_in_two_steps() -> None:
    """With exact line search, CG minimises an n-dim quadratic in n steps."""
    out = run_named("ncg", "quadratic", {"kappa": 50, "theta_deg": 30}, n_iter=10)
    assert out["fs"][2] < 1e-15


def test_lbfgs_memory_matters_on_an_ill_conditioned_bowl() -> None:
    """m = 1 throws away curvature information and needs many more steps."""
    def steps(m: int) -> int:
        out = run_named("lbfgs", "quadratic", {"kappa": 10}, {"m": m}, n_iter=500)
        return next(i for i, f in enumerate(out["fs"]) if f < 1e-10)
    assert steps(5) * 2 < steps(1)


def test_trust_region_records_radii_and_rejects_bad_steps() -> None:
    """Each step reports its radius; a rejected step leaves x unchanged."""
    from optlab.demos.trust_region import dogleg_step
    prob = make_problem("rosenbrock")
    x, r = dogleg_step((0.0, 0.0), 1.0, prob.f, prob.grad, prob.hess)
    assert x == (0.0, 0.0) and r == 0.25
    out = run_named("trust_region", "rosenbrock", n_iter=200)
    assert out["status"] == "converged"
    assert len(out["radius"]) == out["n_iter"] and min(out["radius"]) > 0


def test_trust_region_indefinite_step_stops_at_cauchy_point() -> None:
    """Uphill Newton step but gᵀHg > 0: stop at the model minimum along −g,
    not at the boundary (f = 2x² − y², g = (4, −4), ‖g‖³/gᵀHg = 4√2 < Δ)."""
    from optlab.demos.trust_region import dogleg_step
    x, r = dogleg_step((1.0, 2.0), 8.0, lambda p: 2 * p[0] ** 2 - p[1] ** 2,
                       lambda p: (4 * p[0], -2 * p[1]), lambda p: ((4.0, 0.0), (0.0, -2.0)))
    assert x == pytest.approx((-3.0, 6.0)) and r == 8.0


# ---------- derivative-free family ----------

@pytest.mark.parametrize("algo", ["nelder_mead", "powell", "pattern_search"])
@pytest.mark.parametrize("pid,params", PROBLEMS)
def test_derivative_free_methods_reach_a_minimum(algo: str, pid: str, params: dict) -> None:
    """Every method stops on its own rule (or by gradient) at a listed minimum,
    except compass search on Rosenbrock, which needs far more than 2000 polls."""
    out = run_named(algo, pid, params, n_iter=2000)
    if (algo, pid) == ("pattern_search", "rosenbrock"):
        assert out["status"] == "max_iter" and out["fs"][-1] > 1e-3
        return
    assert out["status"] in ("stopped", "converged")
    assert out["fs"][-1] < 1e-12
    prob = make_problem(pid, **params)
    assert min(math.dist(out["xs"][-1], m) for m in prob.minima) < 1e-5


@pytest.mark.parametrize("algo", ["nelder_mead", "powell", "pattern_search"])
def test_derivative_free_methods_never_increase_f_and_count_evals(algo: str) -> None:
    """The reported best value is monotone and evaluations are counted exactly."""
    import dataclasses
    prob = make_problem("rosenbrock")
    calls = [0]

    def f(x):
        calls[0] += 1
        return prob.f(x)

    counted = dataclasses.replace(prob, f=f)
    opt = OPTIMIZERS[algo]
    seen = [0]

    def step(x, problem, state, p):
        before = calls[0]
        out = opt.step(x, problem, state, p)
        seen[0] += calls[0] - before          # only the method's own calls
        return out

    out = run(step, counted, prob.x0, opt.defaults(), n_iter=300)
    assert all(b <= a for a, b in zip(out["fs"], out["fs"][1:]))
    assert out["evals"] == seen[0]
    assert len(out["poly"]) == len(out["xs"])


def test_nelder_mead_poly_is_the_current_simplex() -> None:
    """Each recorded triangle contains the iterate as its best vertex."""
    prob = make_problem("himmelblau")
    out = run_named("nelder_mead", "himmelblau", n_iter=50)
    assert out["poly"][0] == [[-0.5, -0.5], [0.0, -0.5], [-0.5, 0.0]]
    for x, tri in zip(out["xs"][1:], out["poly"][1:]):
        assert len(tri) == 3 and x in tri
        assert prob.f(x) == min(prob.f(v) for v in tri)


def test_nelder_mead_is_insensitive_to_rotating_the_bowl() -> None:
    """Unlike compass search, NM barely cares how the valley is rotated."""
    def evals(algo: str, theta: float) -> int:
        out = run_named(algo, "quadratic", {"kappa": 100, "theta_deg": theta},
                        n_iter=5000)
        return out["evals"]
    assert evals("nelder_mead", 30) < 2 * evals("nelder_mead", 0)
    assert evals("pattern_search", 30) > 5 * evals("nelder_mead", 30)


def test_nelder_mead_stalls_on_mckinnon_function() -> None:
    """McKinnon's example: NM collapses onto (0, 0), where df/dy = 1."""
    from optlab.demos.nelder_mead import diameter, nelder_mead_step

    def f(p):
        x, y = p
        return (360 * x * x if x <= 0 else 6 * x * x) + y + y * y

    simplex = [(0.0, 0.0), (1.0, 1.0),
               ((1 + math.sqrt(33)) / 8, (1 - math.sqrt(33)) / 8)]
    fvals = [f(v) for v in simplex]
    for _ in range(200):
        simplex, fvals = nelder_mead_step(simplex, fvals, f)
    best = simplex[min(range(3), key=fvals.__getitem__)]
    assert diameter(simplex) < 1e-12
    assert math.hypot(*best) < 1e-12           # not the minimiser (0, -0.5)
    assert min(fvals) > f((0.0, -0.5)) + 0.2
    # Same code, another starting triangle: finds the true minimiser.
    simplex = [(0.5, 0.5), (1.0, 0.5), (0.5, 1.0)]
    fvals = [f(v) for v in simplex]
    for _ in range(200):
        simplex, fvals = nelder_mead_step(simplex, fvals, f)
    best = simplex[min(range(3), key=fvals.__getitem__)]
    assert math.dist(best, (0.0, -0.5)) < 1e-6


def test_compass_step_halves_on_failure_and_keeps_it_on_success() -> None:
    """Step stays after a successful poll; halves (and x stays) on failure."""
    from optlab.demos.pattern_search import compass_step
    prob = make_problem("quadratic", kappa=10, theta_deg=30)
    out = run_named("pattern_search", "quadratic", {"kappa": 10, "theta_deg": 30},
                    n_iter=60)
    x, fx, h = tuple(out["xs"][0]), prob.f(tuple(out["xs"][0])), 0.5
    for _ in range(60):
        x_new, f_new, h_new = compass_step(x, fx, h, prob.f)
        if x_new == x:
            assert h_new == h / 2 and f_new == fx
        else:
            assert h_new == h and f_new < fx
            assert sorted(abs(a - b) for a, b in zip(x, x_new)) == [0.0, h]
        x, fx, h = x_new, f_new, h_new


def test_powell_finishes_a_2d_quadratic_in_two_sweeps() -> None:
    """Conjugate directions: after two sweeps f is at rounding level, from any start."""
    rng = random.Random(3)
    for theta in (0, 30, 60):
        prob = make_problem("quadratic", kappa=50, theta_deg=theta)
        for _ in range(5):
            x0 = (rng.uniform(-4, 4), rng.uniform(-2, 2))
            out = run(OPTIMIZERS["powell"].step, prob, x0, {}, n_iter=10)
            assert out["fs"][min(2, out["n_iter"])] < 1e-12


def test_powell_drops_the_direction_of_largest_decrease() -> None:
    """On 4x^2 + y^2 the x-search gains most, so e1 is replaced."""
    from optlab.demos.powell import powell_step
    _, _, dirs = powell_step((1.0, 1.0), 5.0, [(1.0, 0.0), (0.0, 1.0)],
                             lambda p: 4 * p[0] ** 2 + p[1] ** 2)
    assert dirs[0] == (0.0, 1.0)
    assert dirs[1] == pytest.approx((-1 / math.sqrt(2), -1 / math.sqrt(2)), abs=1e-7)


def test_failed_poll_bounds_the_gradient() -> None:
    """With L-smooth f and the 2n compass directions, a poll where every point
    fails implies ||grad f(x)|| <= sqrt(n) * L * h / 2 (page derivation)."""
    from optlab.demos.pattern_search import compass_step
    kappa = 10.0                                      # L = kappa for this bowl
    prob = make_problem("quadratic", kappa=kappa, theta_deg=30)
    x, h = prob.x0, 0.5
    fx, failures = prob.f(x), 0
    for _ in range(300):
        x_new, fx, h_new = compass_step(x, fx, h, prob.f)
        if x_new == x:
            failures += 1
            assert math.hypot(*prob.grad(x)) <= math.sqrt(2) * kappa * h / 2
        x, h = x_new, h_new
    assert failures > 10

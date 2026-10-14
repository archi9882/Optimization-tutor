"""Run an optimizer and record its trajectory as a JSON-friendly trace."""
from __future__ import annotations

import math

from .first_order import OPTIMIZERS, StepFn
from .problems import Problem, Vec, make_problem

DIVERGE_AT = 1e6


def run(step: StepFn, problem: Problem, x0: Vec, params: dict,
        n_iter: int = 200, tol: float = 1e-10) -> dict:
    """Iterate `step` from x0 and return {xs, fs, status, n_iter}.

    status is "converged" (gradient norm below tol), "diverged" (iterate
    blew up or became NaN) or "max_iter".
    """
    x = (float(x0[0]), float(x0[1]))
    state: dict = {}
    xs = [list(x)]
    fs = [problem.f(x)]
    status = "max_iter"
    for _ in range(n_iter):
        x, state = step(x, problem, state, params)
        x = (float(x[0]), float(x[1]))
        if not all(math.isfinite(v) and abs(v) < DIVERGE_AT for v in x):
            status = "diverged"
            break
        xs.append(list(x))
        fs.append(problem.f(x))
        g = problem.grad(x)
        if math.hypot(g[0], g[1]) < tol:
            status = "converged"
            break
    return {"xs": xs, "fs": fs, "status": status, "n_iter": len(xs) - 1}


def run_named(algo_id: str, problem_id: str, problem_params: dict | None = None,
              params: dict | None = None, x0: Vec | None = None,
              n_iter: int = 200) -> dict:
    """Run a registered optimizer on a registered problem by id."""
    opt = OPTIMIZERS[algo_id]
    prob = make_problem(problem_id, **(problem_params or {}))
    merged = opt.defaults() | (params or {})
    out = run(opt.step, prob, x0 or prob.x0, merged, n_iter)
    out.update({"algo": algo_id, "name": opt.name, "params": merged})
    return out

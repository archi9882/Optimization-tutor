"""Run an optimizer and record its trajectory as a JSON-friendly trace."""
from __future__ import annotations

import math

from .optimizers import OPTIMIZERS, StepFn
from .problems import Problem, Vec, make_problem, violation

DIVERGE_AT = 1e6


def run(step: StepFn, problem: Problem, x0: Vec, params: dict,
        n_iter: int = 200, tol: float = 1e-10) -> dict:
    """Iterate `step` from x0 and return {xs, fs, status, n_iter}.

    status is "converged" (gradient norm below tol), "diverged" (iterate
    blew up or became NaN), "stopped" (the method's own stopping rule, signalled
    by state["stop"]; derivative-free methods never see the gradient) or
    "max_iter".

    Optional per-step extras read from the returned state:
      radius_used  trust-region radius used for the step -> out["radius"]
      poly         geometry at the new iterate (simplex, poll stencil, ...);
                   poly_used is the geometry the first step started from, so
                   out["poly"][k] belongs to xs[k]; poly_kind "open" draws a
                   polyline instead of a filled polygon
      evals        cumulative number of f evaluations -> out["evals"]
      info         {name: number or list} describing the step (rho, mu, lambda,
                   gap, ...) -> out["info"][k] belongs to xs[k] (info[0] is None)
      outer_end    True when the new iterate ends an outer iteration of a
                   penalty / barrier / multiplier method -> out["outer"][k]

    Constrained problems never stop on a small gradient (grad f != 0 at a
    constrained optimum); they report the final constraint violation.
    """
    x = (float(x0[0]), float(x0[1]))
    state: dict = {}
    xs = [list(x)]
    fs = [problem.f(x)]
    status = "max_iter"
    radii: list[float] = []             # trust-region radius used at each step
    polys: list = []                    # per-iterate geometry (see docstring)
    infos: list = [None]
    outer: list[bool] = [False]
    for _ in range(n_iter):
        x, state = step(x, problem, state, params)
        if "radius_used" in state:
            radii.append(state["radius_used"])
        x = (float(x[0]), float(x[1]))
        if not all(math.isfinite(v) and abs(v) < DIVERGE_AT for v in x):
            status = "diverged"
            break
        xs.append(list(x))
        fs.append(problem.f(x))
        if "info" in state:
            infos.append(state["info"])
        outer.append(bool(state.get("outer_end")))
        if "poly" in state:
            if not polys:
                polys.append(state.get("poly_used"))
            polys.append(state["poly"])
        g = problem.grad(x)
        if not problem.constraints and math.hypot(g[0], g[1]) < tol:
            status = "converged"
            break
        if state.get("stop"):
            status = "stopped"
            break
    out = {"xs": xs, "fs": fs, "status": status, "n_iter": len(xs) - 1}
    if radii:
        out["radius"] = radii
    if polys:
        out["poly"] = polys
        if state.get("poly_kind"):
            out["poly_kind"] = state["poly_kind"]
    if "evals" in state:
        out["evals"] = state["evals"]
    if len(infos) > 1:
        out["info"] = infos
    if any(outer):
        out["outer"] = outer
    if problem.constraints:
        out["violation"] = violation(problem, tuple(xs[-1]))
        if "message" in state:
            out["message"] = state["message"]
    return out


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

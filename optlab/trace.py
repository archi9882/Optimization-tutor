"""Run an optimizer and record its trajectory as a JSON-friendly trace."""
from __future__ import annotations

import math

from .optimizers import OPTIMIZERS, StepFn
from .problems import Problem, Vec, by_distance, make_problem, violation

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
      dots         True: draw every iterate as a dot (SQP) -> out["dots"]
      z            a second iterate drawn dashed next to x (ADMM's z) ->
                   out["zs"][k] belongs to xs[k]; z_used is the z the first
                   step started from (default x0)
      pop          population-based methods (renderer 3): the current points
                   -> out["pop"][k] belongs to xs[k] (rounded to the view's
                   resolution, see pop_decimals); fmean -> out["fmean"][k]
                   is their mean f; pop_trail True: draw each point's move
                   from the previous step (PSO velocities) -> out["pop_trail"]

    A step function with an `init(x0, problem, params) -> (x, state)` attribute
    is a population method: init builds the first population (x0 is one of its
    points) and returns its best point, which becomes xs[0]. Such methods
    never stop on a small gradient (they do not use it) and run all n_iter steps
    unless their own rule stops them.

    Constrained problems never stop on a small gradient (grad f != 0 at a
    constrained optimum); they report the final constraint violation. Neither
    do problems with an l1 term (their grad is that of the smooth part only).
    """
    x = (float(x0[0]), float(x0[1]))
    state: dict = {}
    init = getattr(step, "init", None)
    if init is not None:
        x, state = init(x, problem, params)
        x = (float(x[0]), float(x[1]))
    xs = [list(x)]
    fs = [problem.f(x)]
    pops: list = []                     # population per iterate (see docstring)
    fmeans: list = []
    if "pop" in state:
        pops.append(pop_round(state["pop"], problem))
        fmeans.append(float(f"{state['fmean']:.6g}"))
    status = "max_iter"
    radii: list[float] = []             # trust-region radius used at each step
    polys: list = []                    # per-iterate geometry (see docstring)
    infos: list = [None]
    outer: list[bool] = [False]
    zs: list = []                       # second iterate (see docstring)
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
        if "z" in state:
            if not zs:
                zs.append(list(state.get("z_used", xs[0])))
            zs.append([float(state["z"][0]), float(state["z"][1])])
        if "pop" in state:
            pops.append(pop_round(state["pop"], problem))
            fmeans.append(float(f"{state['fmean']:.6g}"))
        if "poly" in state:
            if not polys:
                polys.append(state.get("poly_used"))
            polys.append(state["poly"])
        g = problem.grad(x)
        if init is None and not by_distance(problem) and math.hypot(g[0], g[1]) < tol:
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
    if state.get("dots"):
        out["dots"] = True
    if zs:
        out["zs"] = zs
    if pops:
        out["pop"] = pops
        out["fmean"] = fmeans
        if state.get("pop_trail"):
            out["pop_trail"] = True
    if by_distance(problem):
        out["violation"] = violation(problem, tuple(xs[-1]))
        if "message" in state:
            out["message"] = state["message"]
    return out


def pop_decimals(problem: Problem) -> int:
    """Decimals that resolve 1/10000 of the view's larger side (enough to draw)."""
    x0, x1, y0, y1 = problem.bounds
    span = max(x1 - x0, y1 - y0)
    return max(0, math.ceil(-math.log10(span / 1e4) - 1e-9))


def pop_round(pop, problem: Problem) -> list:
    """A population as embedded in the trace: [[x1, x2], ...] rounded for drawing."""
    d = pop_decimals(problem)
    return [[round(float(p[0]), d) + 0.0, round(float(p[1]), d) + 0.0] for p in pop]


def run_named(algo_id: str, problem_id: str, problem_params: dict | None = None,
              params: dict | None = None, x0: Vec | None = None,
              n_iter: int = 200) -> dict:
    """Run a registered optimizer on a registered problem by id."""
    opt = OPTIMIZERS[algo_id]
    prob = make_problem(problem_id, **(problem_params or {}))
    merged = opt.defaults() | (params or {})
    if prob.l1 and not opt.handles_l1:
        # Its step would see only the smooth part's gradient: refuse, do not mislead.
        x = tuple(x0 or prob.x0)
        out = {"xs": [list(x)], "fs": [prob.f(x)], "status": "stopped", "n_iter": 0,
               "violation": 0.0, "message": "這個方法不處理 ℓ1 項，不適用於這個問題"}
    else:
        out = run(opt.step, prob, x0 or prob.x0, merged, n_iter)
    out.update({"algo": algo_id, "name": opt.name, "params": merged})
    return out

"""Readable demo implementations: the exact code shown on each page.

Each module defines one learner-style step function (FUNC), its pseudocode,
and an `adapter` that plugs the function into the shared lab loop. The lab's
optimizers in optlab.optimizers are built from these, so the animation runs
precisely the code the page displays.
"""
from __future__ import annotations

import inspect
from types import ModuleType

from . import (active_set, adagrad, adam, augmented_lagrangian, barrier, barrier_nl,
               bfgs, branch_bound, coordinate, cutting_plane, frank_wolfe, dual_simplex, dynamic_programming, gauss_newton, gd,
               greedy, hungarian, lbfgs, lm, momentum, ncg, nelder_mead, nesterov,
               newton, pattern_search, penalty, powell, projected_gradient, rmsprop,
               sgd, simplex, tabu,
               trust_region, two_opt)

MODULES: dict[str, ModuleType] = {
    "gd": gd, "sgd": sgd, "momentum": momentum, "nesterov": nesterov,
    "adagrad": adagrad, "rmsprop": rmsprop, "adam": adam,
    "coordinate": coordinate,
    "newton": newton, "gauss_newton": gauss_newton, "lm": lm, "bfgs": bfgs,
    "lbfgs": lbfgs, "ncg": ncg, "trust_region": trust_region,
    "simplex": simplex, "dual_simplex": dual_simplex, "barrier": barrier,
    "active_set": active_set,
    "branch_bound": branch_bound, "cutting_plane": cutting_plane,
    "dynamic_programming": dynamic_programming, "greedy": greedy,
    "hungarian": hungarian, "two_opt": two_opt, "tabu": tabu,
    "nelder_mead": nelder_mead, "powell": powell, "pattern_search": pattern_search,
    "penalty": penalty, "barrier_nl": barrier_nl,
    "augmented_lagrangian": augmented_lagrangian,
    "projected_gradient": projected_gradient, "frank_wolfe": frank_wolfe,
}


def source(demo_id: str) -> str:
    """Display source: imports, helper functions, then the step function."""
    mod = MODULES[demo_id]
    parts = [inspect.getsource(h) for h in getattr(mod, "HELPERS", [])]
    parts.append(inspect.getsource(mod.FUNC))
    return mod.HEADER + "\n\n".join(parts)


def helpers_source(demo_id: str) -> str:
    """Imports plus helper functions only (given to learners in the skeleton)."""
    mod = MODULES[demo_id]
    parts = [inspect.getsource(h) for h in getattr(mod, "HELPERS", [])]
    return mod.HEADER + "".join(p + "\n\n" for p in parts)


def lab_step(demo_id: str):
    """The demo function wrapped for the lab loop (StepFn signature)."""
    mod = MODULES[demo_id]
    return mod.adapter(mod.FUNC)

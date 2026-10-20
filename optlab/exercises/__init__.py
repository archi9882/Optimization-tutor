"""Registry of scratch-practice exercises.

Continuous-optimizer exercises share their id with the optimizer in
optlab.optimizers; LP-family and combinatorial exercises instead define
`check`, which runs the full method (optlab.lp, optlab.ip, optlab.combo) with
the learner's functions.
"""
from __future__ import annotations

from ..grading import Exercise
from .active_set import EXERCISE as ACTIVE_SET
from .admm import EXERCISE as ADMM
from .adagrad import EXERCISE as ADAGRAD
from .adam import EXERCISE as ADAM
from .augmented_lagrangian import EXERCISE as AUGMENTED_LAGRANGIAN
from .barrier import EXERCISE as BARRIER
from .barrier_nl import EXERCISE as BARRIER_NL
from .bfgs import EXERCISE as BFGS
from .branch_bound import EXERCISE as BRANCH_BOUND
from .coordinate import EXERCISE as COORDINATE
from .cutting_plane import EXERCISE as CUTTING_PLANE
from .dual_simplex import EXERCISE as DUAL_SIMPLEX
from .dynamic_programming import EXERCISE as DYNAMIC_PROGRAMMING
from .frank_wolfe import EXERCISE as FRANK_WOLFE
from .gauss_newton import EXERCISE as GAUSS_NEWTON
from .gd import EXERCISE as GD
from .greedy import EXERCISE as GREEDY
from .hungarian import EXERCISE as HUNGARIAN
from .lbfgs import EXERCISE as LBFGS
from .lm import EXERCISE as LM
from .momentum import EXERCISE as MOMENTUM
from .ncg import EXERCISE as NCG
from .nelder_mead import EXERCISE as NELDER_MEAD
from .nesterov import EXERCISE as NESTEROV
from .newton import EXERCISE as NEWTON
from .pattern_search import EXERCISE as PATTERN_SEARCH
from .penalty import EXERCISE as PENALTY
from .powell import EXERCISE as POWELL
from .projected_gradient import EXERCISE as PROJECTED_GRADIENT
from .rmsprop import EXERCISE as RMSPROP
from .sgd import EXERCISE as SGD
from .simplex import EXERCISE as SIMPLEX
from .sqp import EXERCISE as SQP
from .tabu import EXERCISE as TABU
from .trust_region import EXERCISE as TRUST_REGION
from .two_opt import EXERCISE as TWO_OPT

EXERCISES: dict[str, Exercise] = {e.id: e for e in [
    GD, SGD, MOMENTUM, NESTEROV, ADAGRAD, RMSPROP, ADAM, COORDINATE,
    NEWTON, GAUSS_NEWTON, LM, BFGS, LBFGS, NCG, TRUST_REGION,
    SIMPLEX, DUAL_SIMPLEX, BARRIER, ACTIVE_SET,
    BRANCH_BOUND, CUTTING_PLANE, DYNAMIC_PROGRAMMING, GREEDY, HUNGARIAN, TWO_OPT, TABU,
    NELDER_MEAD, POWELL, PATTERN_SEARCH,
    PENALTY, BARRIER_NL, AUGMENTED_LAGRANGIAN, PROJECTED_GRADIENT, FRANK_WOLFE,
    SQP, ADMM]}


def grade_json(exercise_id: str, source: str) -> str:
    """Entry point for the browser: grade source code, return JSON text."""
    import json

    from ..grading import grade_source
    return json.dumps(grade_source(EXERCISES[exercise_id], source))

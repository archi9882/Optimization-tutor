"""Registry of scratch-practice exercises.

Continuous-optimizer exercises share their id with the optimizer in
optlab.optimizers; LP-family exercises (simplex, dual_simplex, barrier,
active_set) instead define `check`, which runs the full method in optlab.lp.
"""
from __future__ import annotations

from ..grading import Exercise
from .active_set import EXERCISE as ACTIVE_SET
from .adagrad import EXERCISE as ADAGRAD
from .adam import EXERCISE as ADAM
from .barrier import EXERCISE as BARRIER
from .bfgs import EXERCISE as BFGS
from .coordinate import EXERCISE as COORDINATE
from .dual_simplex import EXERCISE as DUAL_SIMPLEX
from .gauss_newton import EXERCISE as GAUSS_NEWTON
from .gd import EXERCISE as GD
from .lbfgs import EXERCISE as LBFGS
from .lm import EXERCISE as LM
from .momentum import EXERCISE as MOMENTUM
from .ncg import EXERCISE as NCG
from .nesterov import EXERCISE as NESTEROV
from .newton import EXERCISE as NEWTON
from .rmsprop import EXERCISE as RMSPROP
from .sgd import EXERCISE as SGD
from .simplex import EXERCISE as SIMPLEX
from .trust_region import EXERCISE as TRUST_REGION

EXERCISES: dict[str, Exercise] = {e.id: e for e in [
    GD, SGD, MOMENTUM, NESTEROV, ADAGRAD, RMSPROP, ADAM, COORDINATE,
    NEWTON, GAUSS_NEWTON, LM, BFGS, LBFGS, NCG, TRUST_REGION,
    SIMPLEX, DUAL_SIMPLEX, BARRIER, ACTIVE_SET]}


def grade_json(exercise_id: str, source: str) -> str:
    """Entry point for the browser: grade source code, return JSON text."""
    import json

    from ..grading import grade_source
    return json.dumps(grade_source(EXERCISES[exercise_id], source))

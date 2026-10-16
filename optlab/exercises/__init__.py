"""Registry of scratch-practice exercises.

Each exercise id equals the id of the optimizer in optlab.optimizers that
it re-implements, so tests can check the two produce identical traces.
"""
from __future__ import annotations

from ..grading import Exercise
from .adagrad import EXERCISE as ADAGRAD
from .adam import EXERCISE as ADAM
from .bfgs import EXERCISE as BFGS
from .coordinate import EXERCISE as COORDINATE
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
from .trust_region import EXERCISE as TRUST_REGION

EXERCISES: dict[str, Exercise] = {e.id: e for e in [
    GD, SGD, MOMENTUM, NESTEROV, ADAGRAD, RMSPROP, ADAM, COORDINATE,
    NEWTON, GAUSS_NEWTON, LM, BFGS, LBFGS, NCG, TRUST_REGION]}


def grade_json(exercise_id: str, source: str) -> str:
    """Entry point for the browser: grade source code, return JSON text."""
    import json

    from ..grading import grade_source
    return json.dumps(grade_source(EXERCISES[exercise_id], source))

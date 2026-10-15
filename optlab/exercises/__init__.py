"""Registry of scratch-practice exercises.

Each exercise id equals the id of the optimizer in optlab.first_order that
it re-implements, so tests can check the two produce identical traces.
"""
from __future__ import annotations

from ..grading import Exercise
from .adagrad import EXERCISE as ADAGRAD
from .adam import EXERCISE as ADAM
from .coordinate import EXERCISE as COORDINATE
from .gd import EXERCISE as GD
from .momentum import EXERCISE as MOMENTUM
from .nesterov import EXERCISE as NESTEROV
from .rmsprop import EXERCISE as RMSPROP
from .sgd import EXERCISE as SGD

EXERCISES: dict[str, Exercise] = {e.id: e for e in [
    GD, SGD, MOMENTUM, NESTEROV, ADAGRAD, RMSPROP, ADAM, COORDINATE]}


def grade_json(exercise_id: str, source: str) -> str:
    """Entry point for the browser: grade source code, return JSON text."""
    import json

    from ..grading import grade_source
    return json.dumps(grade_source(EXERCISES[exercise_id], source))

"""Registry of scratch-practice exercises."""
from __future__ import annotations

from ..grading import Exercise
from .gd import EXERCISE as GD
from .momentum import EXERCISE as MOMENTUM

EXERCISES: dict[str, Exercise] = {e.id: e for e in [GD, MOMENTUM]}


def grade_json(exercise_id: str, source: str) -> str:
    """Entry point for the browser: grade source code, return JSON text."""
    import json

    from ..grading import grade_source
    return json.dumps(grade_source(EXERCISES[exercise_id], source))

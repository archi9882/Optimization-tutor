"""JSON-in / JSON-out entry points called from the browser (Pyodide worker)
and from the static-site build. Keeping one API guarantees the build-time
default traces and the live in-browser traces come from identical code."""
from __future__ import annotations

import json

from .exercises import EXERCISES
from .grading import grade_source
from .problems import grid as _grid, make_problem
from .trace import run_named


def problem_grid(spec: dict, nx: int = 120, ny: int = 100) -> dict:
    """Contour grid for {"problem": id, "problem_params": {...}}."""
    prob = make_problem(spec["problem"], **spec.get("problem_params", {}))
    out = _grid(prob, nx, ny)
    out["x0"] = list(prob.x0)
    return out


def traces(spec: dict) -> list[dict]:
    """Run each requested optimizer on one problem.

    spec = {"problem", "problem_params", "x0", "n_iter",
            "runs": [{"algo": id, "params": {...}}, ...]}
    """
    return [run_named(r["algo"], spec["problem"],
                      spec.get("problem_params", {}), r.get("params", {}),
                      tuple(spec["x0"]) if spec.get("x0") else None,
                      spec.get("n_iter", 150))
            for r in spec["runs"]]


def grade(exercise_id: str, source: str) -> dict:
    """Grade learner source for one exercise."""
    return grade_source(EXERCISES[exercise_id], source)


def call(name: str, payload_json: str) -> str:
    """Dispatch used by the worker: call(name, json) -> json."""
    payload = json.loads(payload_json)
    if name == "grid":
        result = problem_grid(payload)
    elif name == "traces":
        result = traces(payload)
    elif name == "grade":
        result = grade(payload["exercise"], payload["source"])
    else:
        raise ValueError(f"unknown api call: {name}")
    return json.dumps(result)

"""Readable demo implementations: the exact code shown on each page.

Each module defines one learner-style step function (FUNC), its pseudocode,
and an `adapter` that plugs the function into the shared lab loop. The lab's
optimizers in optlab.first_order are built from these, so the animation runs
precisely the code the page displays.
"""
from __future__ import annotations

import inspect
from types import ModuleType

from . import adagrad, adam, coordinate, gd, momentum, nesterov, rmsprop, sgd

MODULES: dict[str, ModuleType] = {
    "gd": gd, "sgd": sgd, "momentum": momentum, "nesterov": nesterov,
    "adagrad": adagrad, "rmsprop": rmsprop, "adam": adam,
    "coordinate": coordinate,
}


def source(demo_id: str) -> str:
    """Display source: needed imports plus the step function's own code."""
    mod = MODULES[demo_id]
    return mod.HEADER + inspect.getsource(mod.FUNC)


def lab_step(demo_id: str):
    """The demo function wrapped for the lab loop (StepFn signature)."""
    mod = MODULES[demo_id]
    return mod.adapter(mod.FUNC)

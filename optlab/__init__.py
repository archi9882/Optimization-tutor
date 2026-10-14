"""optlab: reference implementations behind the Optimization Tutor site.

Pure Python (no numpy) so the same code runs in CPython for tests and
build-time traces, and in the browser through Pyodide.
"""
from .first_order import OPTIMIZERS
from .problems import make_problem
from .trace import run, run_named

__all__ = ["OPTIMIZERS", "make_problem", "run", "run_named"]

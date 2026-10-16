"""Grade a learner's scratch implementation against the reference.

Runs unchanged in CPython (pytest) and in the browser (Pyodide worker).
Three layers of checks:
  1. single-step cases  -- exact arithmetic on hand-picked inputs
  2. behaviour          -- convergence / properties on a real problem
  3. trace match        -- the whole trajectory overlays the reference
"""
from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Callable

from .problems import make_problem
from .trace import run


@dataclass(frozen=True)
class StepCase:
    """One hand-checkable step: call the learner's function with `args`."""

    name: str
    args: dict
    expected: tuple  # tuple of expected vectors (one entry = bare vector)


@dataclass(frozen=True)
class Exercise:
    """A scratch-practice exercise definition."""

    id: str
    func_name: str
    skeleton: str
    hints: list[str]
    solution: str
    cases: list[StepCase]
    adapter: Callable[[Callable], Callable]  # learner fn -> StepFn
    reference: Callable                       # reference learner-style fn
    demo: dict                                # problem/params for layers 2-3
    converge_below: float = 1e-8
    blanks: tuple[str, ...] = ()              # substrings of `solution` to hide


def flatten(obj) -> list[float]:
    """All numbers in a nested structure (tuples, lists, arrays), in order."""
    if obj is None:
        raise TypeError("函式回傳了 None，是不是忘了 return？")
    if isinstance(obj, (int, float)) or (
            hasattr(obj, "__float__") and not hasattr(obj, "__iter__")):
        return [float(obj)]
    try:
        return [v for item in obj for v in flatten(item)]
    except TypeError:
        raise TypeError(f"無法解讀的回傳值：{obj!r}") from None


def _fmt(obj) -> str:
    """Compact display of a nested numeric structure."""
    if isinstance(obj, (int, float)):
        return f"{obj:.6g}"
    try:
        return "(" + ", ".join(_fmt(v) for v in obj) + ")"
    except TypeError:
        return repr(obj)


def _err(exc: BaseException) -> str:
    """Short, learner-friendly error text (last frame only)."""
    lines = traceback.format_exception_only(type(exc), exc)
    return "".join(lines).strip()


def grade(ex: Exercise, fn: Callable) -> dict:
    """Run all checks; return {passed, results, user_trace, ref_trace}."""
    results: list[dict] = []

    for case in ex.cases:
        try:
            out = fn(**case.args)
            want = case.expected[0] if len(case.expected) == 1 else case.expected
            got_flat, want_flat = flatten(out), flatten(want)
            ok = len(got_flat) == len(want_flat) and all(
                abs(g - w) <= 1e-7 * (1 + abs(w))
                for g, w in zip(got_flat, want_flat))
            msg = "通過" if ok else f"預期 {_fmt(want)}，得到 {_fmt(out)}"
        except Exception as exc:  # learner code may raise anything
            ok, msg = False, _err(exc)
        results.append({"layer": 1, "name": case.name, "passed": ok,
                        "message": msg})

    demo = ex.demo
    prob = make_problem(demo["problem"], **demo.get("problem_params", {}))
    x0 = tuple(demo.get("x0", prob.x0))
    n_iter = demo.get("n_iter", 300)
    ref_trace = run(ex.adapter(ex.reference), prob, x0, demo["params"], n_iter)
    user_trace = None
    try:
        user_trace = run(ex.adapter(fn), prob, x0, demo["params"], n_iter)
        f_end = user_trace["fs"][-1] - prob.fstar
        ok = user_trace["status"] != "diverged" and f_end < ex.converge_below
        msg = (f"{user_trace['n_iter']} 步後 f−f* = {f_end:.2e}"
               + ("" if ok else f"（需 < {ex.converge_below:.0e}）"))
    except Exception as exc:
        ok, msg = False, _err(exc)
    results.append({"layer": 2, "name": f"在 {prob.name} 上收斂",
                    "passed": ok, "message": msg})

    if user_trace is not None:
        n = min(len(user_trace["xs"]), len(ref_trace["xs"]))
        dev = max((math.dist(user_trace["xs"][i], ref_trace["xs"][i])
                   for i in range(n)), default=math.inf)
        ok = len(user_trace["xs"]) == len(ref_trace["xs"]) and dev < 1e-6
        msg = f"最大偏差 {dev:.2e}，步數 {len(user_trace['xs']) - 1}" \
              f" / 參考 {len(ref_trace['xs']) - 1}"
    else:
        ok, msg = False, "無軌跡可比對"
    results.append({"layer": 3, "name": "軌跡與參考解重合", "passed": ok,
                    "message": msg})

    return {"passed": all(r["passed"] for r in results), "results": results,
            "user_trace": user_trace, "ref_trace": ref_trace}


def grade_source(ex: Exercise, source: str) -> dict:
    """Exec learner source code, find `ex.func_name`, then grade it."""
    ns: dict = {}
    try:
        exec(compile(source, "<你的程式>", "exec"), ns)
    except Exception as exc:
        return {"passed": False, "user_trace": None, "ref_trace": None,
                "results": [{"layer": 0, "name": "程式可執行", "passed": False,
                             "message": _err(exc)}]}
    fn = ns.get(ex.func_name)
    if not callable(fn):
        return {"passed": False, "user_trace": None, "ref_trace": None,
                "results": [{"layer": 0, "name": "找到函式", "passed": False,
                             "message": f"找不到函式 {ex.func_name}()"}]}
    return grade(ex, fn)

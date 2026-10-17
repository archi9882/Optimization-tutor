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
    func: str | None = None  # call this learner function instead of func_name


@dataclass(frozen=True)
class Exercise:
    """A scratch-practice exercise definition."""

    id: str
    func_name: str
    skeleton: str
    hints: list[str]
    solution: str
    cases: list[StepCase]
    adapter: Callable[[Callable], Callable] | None  # learner fn -> StepFn
    reference: Callable | None                # reference learner-style fn
    demo: dict | None                         # problem/params for layers 2-3
    converge_below: float = 1e-8
    blanks: tuple[str, ...] = ()              # substrings of `solution` to hide
    # Optional replacement for layers 2-3: check(namespace) ->
    # (results, user_trace, ref_trace). Used by the LP family.
    check: Callable[[dict], tuple] | None = None
    required: tuple[str, ...] = ()            # extra function names that must exist
    lp_preset: str | None = None              # LP family: preset drawn under the traces
    loop_note: str = ""                       # how the animation calls func_name


def _is_number(obj) -> bool:
    return isinstance(obj, (int, float)) or (
        hasattr(obj, "__float__") and not hasattr(obj, "__iter__"))


def same(got, want, tol: float = 1e-7) -> bool:
    """Structural comparison of nested numbers; None must match None."""
    if want is None:
        return got is None
    if _is_number(want):
        return _is_number(got) and abs(float(got) - want) <= tol * (1 + abs(want))
    try:
        got, want = list(got), list(want)
    except TypeError:
        return False
    return len(got) == len(want) and all(same(g, w, tol) for g, w in zip(got, want))


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
    if obj is None:
        return "None"
    if isinstance(obj, bool):
        return str(obj)
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


def grade(ex: Exercise, fn: Callable | dict) -> dict:
    """Run all checks; return {passed, results, user_trace, ref_trace}.

    `fn` is the learner's main function, or the whole namespace of their
    source (needed when an exercise involves several functions).
    """
    ns = fn if isinstance(fn, dict) else {ex.func_name: fn}
    fn = ns[ex.func_name]
    results: list[dict] = []

    for case in ex.cases:
        try:
            out = ns[case.func or ex.func_name](**case.args)
            want = case.expected[0] if len(case.expected) == 1 else case.expected
            if out is None and want is not None:
                raise TypeError("函式回傳了 None，是不是忘了 return？")
            ok = same(out, want)
            msg = "通過" if ok else f"預期 {_fmt(want)}，得到 {_fmt(out)}"
        except Exception as exc:  # learner code may raise anything
            ok, msg = False, _err(exc)
        results.append({"layer": 1, "name": case.name, "passed": ok,
                        "message": msg})

    if ex.check is not None:
        try:
            more, user_trace, ref_trace = ex.check(ns)
        except Exception as exc:
            more, user_trace, ref_trace = [{"layer": 2, "name": "完整執行", "passed": False,
                                            "message": _err(exc)}], None, None
        results.extend(more)
        return {"passed": all(r["passed"] for r in results), "results": results,
                "user_trace": user_trace, "ref_trace": ref_trace}

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
    for name in (ex.func_name, *ex.required):
        if not callable(ns.get(name)):
            return {"passed": False, "user_trace": None, "ref_trace": None,
                    "results": [{"layer": 0, "name": "找到函式", "passed": False,
                                 "message": f"找不到函式 {name}()"}]}
    return grade(ex, ns)

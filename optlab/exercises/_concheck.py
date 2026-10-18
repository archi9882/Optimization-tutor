"""Layer 2-3 checks shared by the constrained-optimization exercises.

The learner's functions run inside the same outer loop as the animation
(the demo's adapter) on a constrained problem. Helpers such as a projection or
a linear oracle are taken from the learner's namespace, so blanks inside the
helpers matter too.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable

from ..problems import make_problem, violation
from ..trace import run


def make_check(demo, func_name: str, runs: list[dict],
               wire: Callable[[dict, object], object] | None = None):
    """Build Exercise.check for the constrained family.

    runs: [{"problem", "params", "n_iter", "dist" (max ‖x_end − x*‖),
            "viol" (max violation), "goal" (text), optional "path": True to
            measure the violation over every iterate instead of the last one,
            optional "stop": False to accept running out of iterations}];
    layer 3 compares the whole trajectory of every run with its reference.
    wire(ns, problem) -> problem with learner helpers plugged in (projection,
    linear oracle); None keeps the problem as is.
    """
    def check(ns: dict) -> tuple:
        results, pairs = [], []
        for spec in runs:
            prob = make_problem(spec["problem"])
            ref = run(demo.adapter(demo.FUNC), prob, prob.x0, spec["params"], spec["n_iter"])
            user_prob = wire(ns, prob) if wire else prob
            try:
                user = run(demo.adapter(ns[func_name]), user_prob, prob.x0, spec["params"],
                           spec["n_iter"])
            except Exception as exc:
                results.append({"layer": 2, "name": spec["goal"], "passed": False,
                                "message": f"{type(exc).__name__}: {exc}"})
                pairs.append((None, ref))
                continue
            end = tuple(user["xs"][-1])
            dist = math.dist(end, prob.minima[0])
            pts = user["xs"] if spec.get("path") else [end]
            viol = max(violation(prob, tuple(p)) for p in pts)
            status_ok = user["status"] == "stopped" or (
                not spec.get("stop", True) and user["status"] == "max_iter")
            ok = status_ok and dist <= spec["dist"] and viol <= spec["viol"]
            where = "所有迭代點的最大違反量" if spec.get("path") else "終點違反量"
            results.append({"layer": 2, "name": spec["goal"], "passed": ok,
                            "message": f"{user['n_iter']} 步，狀態 {user['status']}，"
                                       f"‖x−x*‖ = {dist:.2e}（需 ≤ {spec['dist']:.0e}），"
                                       f"{where} {viol:.1e}（需 ≤ {spec['viol']:.0e}）"})
            pairs.append((user, ref))
        ok, devs = True, []
        for user, ref in pairs:
            if user is None:
                ok = False
                devs.append("無軌跡")
                continue
            n = min(len(user["xs"]), len(ref["xs"]))
            dev = max((math.dist(user["xs"][i], ref["xs"][i]) for i in range(n)),
                      default=math.inf)
            ok = ok and len(user["xs"]) == len(ref["xs"]) and dev < 1e-6
            devs.append(f"{dev:.1e}（{len(user['xs']) - 1}/{len(ref['xs']) - 1} 步）")
        results.append({"layer": 3, "name": "每個問題的軌跡都與參考解重合", "passed": ok,
                        "message": "最大偏差 " + "、".join(devs)})
        user, ref = pairs[0]
        return results, user, ref
    return check


def with_helpers(ns: dict, prob, **fields):
    """Copy of prob whose callables (project, lmo) use the learner's helpers."""
    return dataclasses.replace(prob, **{k: v(ns) for k, v in fields.items()})

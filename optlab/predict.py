"""Prediction questions ("預測再看動畫"): measure a trace, check predicates.

A page lists questions in its frontmatter (`predict:`). Each option carries a
machine-checkable predicate on the scenario's trace, so the correct answer is
decided by running the code at build time, never by hand:

    when: {status: diverged}
    when: {reach_within: 150, zigzag_min: 3}

Every key of `when` must hold (logical AND). Keys:

    status            str or list of str: the runner's status
    reach_within      int: f - f* < reach_tol at some step k <= N
    not_reach_within  int: no step k <= N with f - f* < reach_tol
                      (a diverged run never reaches)
    zigzag_min/max    int: number of turns sharper than 90 degrees (a step whose
                      dot product with the previous step is negative),
                      inclusive bounds
    rises_min/max     int: number of steps where f went up, inclusive bounds
    final_gap_below   float: last f - f* < X
    final_gap_above   float: last f - f* >= X
    stops_within      int: the run stopped (its own rule or a zero gradient,
                      not diverged and not max_iter) at a step k <= N
    not_stops_within  int: the negation: still running after N steps,
                      stopped later, or diverged

Derivative-free methods (the trace reports its f-evaluation count):

    evals_below       int: f evaluations of the whole run < N
    evals_above       int: f evaluations of the whole run >= N

Constrained problems (measure() is given the problem). The lab plots
||x - x*|| there, not f - f*, so the f - f* keys above are refused and these
are used instead (a diverged run has an infinite distance and violation):

    final_dist_below/above       float: last ||x - x*|| < X / >= X
    final_violation_below/above  float: constraint violation of the last
                                 iterate < X / >= X (the lab's 違反量)
    max_violation_below/above    float: largest violation over every iterate,
                                 x0 included, < X / >= X
    stays_feasible               bool: every iterate has violation <= FEAS_TOL
                                 (false: some iterate violates by more)
    outer_min/max                int: completed outer iterations (penalty /
                                 barrier rounds, multiplier updates), inclusive
"""
from __future__ import annotations

import math

from .problems import Problem, violation

Measure = dict

FEAS_TOL = 1e-9      # stays_feasible: an iterate counts as feasible up to this


def measure(trace: dict, fstar: float, reach_tol: float,
            problem: Problem | None = None) -> Measure:
    """Numbers the predicates look at, also shown in the reveal.

    With a constrained `problem`, also the distance to its optimum x* and the
    constraint violation; with an `evals` count in the trace, that count.
    """
    xs, fs = trace["xs"], trace["fs"]
    zigzag = 0
    for k in range(2, len(xs)):
        a = (xs[k - 1][0] - xs[k - 2][0], xs[k - 1][1] - xs[k - 2][1])
        b = (xs[k][0] - xs[k - 1][0], xs[k][1] - xs[k - 1][1])
        if a[0] * b[0] + a[1] * b[1] < 0:
            zigzag += 1
    rises = sum(fs[k] > fs[k - 1] for k in range(1, len(fs)))
    reach = next((k for k, f in enumerate(fs) if f - fstar < reach_tol), None)
    gap = fs[-1] - fstar
    m = {"status": trace["status"], "steps": trace["n_iter"],
         "reach": reach, "zigzag": zigzag, "rises": rises,
         "final_gap": gap if math.isfinite(gap) else None}
    if "evals" in trace:
        m["evals"] = int(trace["evals"])
    if problem is not None and problem.constraints:
        diverged = trace["status"] == "diverged"
        xstar = problem.minima[0]
        viol = [violation(problem, (x[0], x[1])) for x in xs]
        m.update({
            "dist": None if diverged else math.dist(xs[-1], xstar),
            "violation": None if diverged else viol[-1],
            "max_violation": None if diverged else max(viol),
            "outer": sum(bool(v) for v in trace.get("outer", [])),
        })
        if trace.get("message"):
            m["message"] = trace["message"]
    return m


def _reached(m: Measure, n: int) -> bool:
    return m["status"] != "diverged" and m["reach"] is not None and m["reach"] <= n


def _gap(m: Measure) -> float:
    return math.inf if m["final_gap"] is None or m["status"] == "diverged" \
        else m["final_gap"]


def _stopped(m: Measure, n: int) -> bool:
    return m["status"] in ("stopped", "converged") and m["steps"] <= n


def _num(m: Measure, key: str) -> float:
    """A constrained measure; a diverged run counts as infinitely far / infeasible."""
    return math.inf if m[key] is None else m[key]


CHECKS = {
    "status": lambda m, v: m["status"] in ([v] if isinstance(v, str) else v),
    "reach_within": lambda m, v: _reached(m, int(v)),
    "not_reach_within": lambda m, v: not _reached(m, int(v)),
    "zigzag_min": lambda m, v: m["zigzag"] >= v,
    "zigzag_max": lambda m, v: m["zigzag"] <= v,
    "rises_min": lambda m, v: m["rises"] >= v,
    "rises_max": lambda m, v: m["rises"] <= v,
    "final_gap_below": lambda m, v: _gap(m) < v,
    "final_gap_above": lambda m, v: _gap(m) >= v,
    "stops_within": lambda m, v: _stopped(m, int(v)),
    "not_stops_within": lambda m, v: not _stopped(m, int(v)),
    "evals_below": lambda m, v: m["evals"] < v,
    "evals_above": lambda m, v: m["evals"] >= v,
    "final_dist_below": lambda m, v: _num(m, "dist") < v,
    "final_dist_above": lambda m, v: _num(m, "dist") >= v,
    "final_violation_below": lambda m, v: _num(m, "violation") < v,
    "final_violation_above": lambda m, v: _num(m, "violation") >= v,
    "max_violation_below": lambda m, v: _num(m, "max_violation") < v,
    "max_violation_above": lambda m, v: _num(m, "max_violation") >= v,
    "stays_feasible": lambda m, v: (_num(m, "max_violation") <= FEAS_TOL) == v,
    "outer_min": lambda m, v: m["outer"] >= v,
    "outer_max": lambda m, v: m["outer"] <= v,
}

# Keys that only make sense for one kind of run (checked against the measure).
F_GAP_KEYS = {"reach_within", "not_reach_within", "final_gap_below", "final_gap_above"}
EVALS_KEYS = {"evals_below", "evals_above"}
CONSTRAINED_KEYS = {"final_dist_below", "final_dist_above", "final_violation_below",
                    "final_violation_above", "max_violation_below", "max_violation_above",
                    "stays_feasible", "outer_min", "outer_max"}
NEW_INT_KEYS = {"stops_within", "not_stops_within", "evals_below", "evals_above",
                "outer_min", "outer_max"}
# (a, b): the predicate is empty unless when[a] < when[b]; outer_min/max: unless <=.
EMPTY_IF_NOT_BELOW = [(f"{q}_above", f"{q}_below") for q in
                      ("evals", "final_dist", "final_violation", "max_violation")] + \
    [("not_stops_within", "stops_within")]
POSITIVE_KEYS = CONSTRAINED_KEYS - {"stays_feasible", "outer_min", "outer_max"}

STATUS_TEXT = {"diverged": "發散", "converged": "梯度歸零而停止",
               "stopped": "演算法自行停止", "max_iter": "跑滿迭代數"}


def _sci(v: float) -> str:
    """1e-10 -> '10⁻¹⁰'; other numbers as %g."""
    e = math.log10(v) if v > 0 else 0.5
    if abs(e - round(e)) < 1e-9 and round(e) <= -2:      # 1, 10, 0.1 read better as plain numbers
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "10" + str(int(round(e))).translate(sup)
    return f"{v:g}"


# key -> (quantity, partner key, is the lower bound): a pair of bounds on the
# same quantity reads "lo ≤ quantity < hi" in the criterion.
RANGE_PAIRS = {f"{q}_{side}": (name, f"{q}_{'below' if side == 'above' else 'above'}",
                               side == "above")
               for q, name in [("evals", "整次執行的 f 計算次數"), ("final_dist", "最後 ‖x−x*‖"),
                               ("final_violation", "最後的違反量"),
                               ("max_violation", "路徑上最大違反量")]
               for side in ("above", "below")}


def criterion(when: dict, reach_tol: float) -> str:
    """Human-readable form of a predicate (generated, so it cannot drift)."""
    tol = _sci(reach_tol)
    feas = _sci(FEAS_TOL)

    def status(v: str | list[str]) -> str:
        names = [v] if isinstance(v, str) else list(v)
        if set(names) == set(STATUS_TEXT) - {"diverged"}:
            return "未發散"
        return "或".join(STATUS_TEXT.get(s, s) for s in names)

    text = {
        "status": status,
        "reach_within": lambda v: f"{v} 步內 f−f* < {tol}",
        "not_reach_within": lambda v: f"{v} 步內未達 f−f* < {tol}",
        "zigzag_min": lambda v: f"折返（轉向超過 90°）≥ {v} 次",
        "zigzag_max": lambda v: f"折返 ≤ {v} 次",
        "rises_min": lambda v: f"f 上升的步數 ≥ {v}",
        "rises_max": lambda v: "f 每步都不上升" if v == 0 else f"f 上升的步數 ≤ {v}",
        "final_gap_below": lambda v: f"最後 f−f* < {_sci(v)}",
        "final_gap_above": lambda v: f"最後 f−f* ≥ {_sci(v)}",
        "stops_within": lambda v: f"{v} 步內停止",
        "not_stops_within": lambda v: f"{v} 步內未停止",
        "evals_below": lambda v: f"整次執行的 f 計算次數 < {v}",
        "evals_above": lambda v: f"整次執行的 f 計算次數 ≥ {v}",
        "final_dist_below": lambda v: f"最後 ‖x−x*‖ < {_sci(v)}",
        "final_dist_above": lambda v: f"最後 ‖x−x*‖ ≥ {_sci(v)}",
        "final_violation_below": lambda v: f"最後的違反量 < {_sci(v)}",
        "final_violation_above": lambda v: f"最後的違反量 ≥ {_sci(v)}",
        "max_violation_below": lambda v: f"路徑上最大違反量 < {_sci(v)}",
        "max_violation_above": lambda v: f"路徑上最大違反量 ≥ {_sci(v)}",
        "stays_feasible": lambda v: (f"每個迭代點都在可行域內（違反量 ≤ {feas}）" if v
                                     else f"有迭代點離開可行域（違反量 > {feas}）"),
        "outer_min": lambda v: f"完成的外圈 ≥ {v} 輪",
        "outer_max": lambda v: "沒有完成任何一輪外圈" if v == 0 else f"完成的外圈 ≤ {v} 輪",
    }
    parts, done = [], set()
    for k, v in when.items():
        pair = RANGE_PAIRS.get(k)
        if pair is None or pair[1] not in when:
            parts.append(text[k](v))
        elif pair[1] not in done:     # first key of a pair: "lo ≤ quantity < hi"
            lo, hi = (v, when[pair[1]]) if pair[2] else (when[pair[1]], v)
            num = str if k.startswith("evals") else _sci
            parts.append(f"{num(lo)} ≤ {pair[0]} < {num(hi)}")
        done.add(k)
    out = parts[0]
    for p in parts[1:]:  # a space between CJK and a leading digit
        out += ("，且 " if p[:1].isdigit() else "，且") + p
    return out


def fmt(v: float) -> str:
    """The lab's number format (plots.js fmt): 3 significant digits or so."""
    if not math.isfinite(v):
        return "Infinity" if v > 0 else "-Infinity" if v < 0 else "NaN"
    if v == 0:
        return "0"
    a = abs(v)
    if a < 1e-3 or a >= 1e5:
        mant, exp = f"{v:.2e}".split("e")
        return f"{mant}e{'-' if exp[0] == '-' else '+'}{int(exp[1:])}"
    return f"{v:.{4 if a < 1 else 3 if a < 100 else 1}f}"


def _stop(m: Measure) -> str:
    return f"第 {m['steps']} 步停止" if m["status"] in ("stopped", "converged") \
        else f"{m['steps']} 步內未停止"


def summary(m: Measure, reach_tol: float) -> str:
    """What the run did, in the words of the lab's result line."""
    if m["status"] == "diverged":
        return f"第 {m['steps'] + 1} 步發散"
    if "dist" in m:   # constrained: the lab's line, then the path's numbers
        head = f"第 {m['steps']} 步停止：{m['message']}" if "message" in m else _stop(m)
        out = (f"{head} · ‖x−x*‖ = {fmt(m['dist'])} · 違反量 {fmt(m['violation'])}"
               f" · 路徑上最大違反量 {fmt(m['max_violation'])}")
        return out + (f" · 完成 {m['outer']} 輪外圈" if m["outer"] else "")
    tol = _sci(reach_tol)
    reach = (f"f−f* < {tol} 於第 {m['reach']} 步" if m["reach"] is not None
             else f"未達 f−f* < {tol}（共 {m['steps']} 步）")
    out = f"{reach} · 折返 {m['zigzag']} 次 · f 上升 {m['rises']} 步"
    if "evals" in m:
        out += f" · {_stop(m)}，共 {m['evals']} 次 f 計算"
    return out


def validate(when: dict) -> None:
    """Raise ValueError on unknown keys, unknown statuses or non-numeric bounds."""
    if not isinstance(when, dict) or not when:
        raise ValueError(f"predicate must be a non-empty mapping, got {when!r}")
    unknown = set(when) - set(CHECKS)
    if unknown:
        raise ValueError(f"unknown predicate key(s): {sorted(unknown)}")
    for k, v in when.items():
        if k == "status":
            names = [v] if isinstance(v, str) else v
            if not isinstance(names, list) or not names or \
                    any(s not in STATUS_TEXT for s in names):
                raise ValueError(f"status must be one or more of {sorted(STATUS_TEXT)}, got {v!r}")
        elif k == "stays_feasible":
            if not isinstance(v, bool):
                raise ValueError(f"stays_feasible needs true or false, got {v!r}")
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{k} needs a number, got {v!r}")
        elif k in NEW_INT_KEYS and (not float(v).is_integer() or v < 0):
            raise ValueError(f"{k} needs a whole number >= 0, got {v!r}")
        elif k in POSITIVE_KEYS and not (math.isfinite(v) and v > 0):
            raise ValueError(f"{k} needs a positive number, got {v!r}")
    # Two bounds on one quantity that no run can satisfy together.
    empty = [(lo, hi) for lo, hi in EMPTY_IF_NOT_BELOW
             if lo in when and hi in when and not when[lo] < when[hi]]
    if when.get("outer_min", 0) > when.get("outer_max", math.inf):
        empty.append(("outer_min", "outer_max"))
    if empty:
        lo, hi = empty[0]
        raise ValueError(f"{lo} = {when[lo]} and {hi} = {when[hi]} leave no possible run")


def applicable(when: dict, m: Measure) -> None:
    """Raise ValueError when a key does not apply to this kind of run."""
    constrained = "dist" in m
    for k in when:
        if k in F_GAP_KEYS and constrained:
            raise ValueError(f"{k} compares f − f*, which the lab does not show on a "
                             "constrained problem; use final_dist_* instead")
        if k in CONSTRAINED_KEYS and not constrained:
            raise ValueError(f"{k} needs a constrained problem")
        if k in EVALS_KEYS and "evals" not in m:
            raise ValueError(f"{k} needs a method that counts f evaluations")


def check(when: dict, m: Measure) -> bool:
    """True when every condition of `when` holds for the measured trace."""
    validate(when)
    applicable(when, m)
    return all(CHECKS[k](m, v) for k, v in when.items())


def answer(options: list[dict], m: Measure) -> int:
    """Index of the single true option; ValueError unless exactly one holds."""
    hits = [i for i, o in enumerate(options) if check(o["when"], m)]
    if len(hits) != 1:
        labels = [options[i]["label"] for i in hits]
        raise ValueError(f"expected exactly one true option, got {len(hits)}: "
                         f"{labels} for measured {m}")
    return hits[0]

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
"""
from __future__ import annotations

import math

Measure = dict


def measure(trace: dict, fstar: float, reach_tol: float) -> Measure:
    """Numbers the predicates look at, also shown in the reveal."""
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
    return {"status": trace["status"], "steps": trace["n_iter"],
            "reach": reach, "zigzag": zigzag, "rises": rises,
            "final_gap": gap if math.isfinite(gap) else None}


def _reached(m: Measure, n: int) -> bool:
    return m["status"] != "diverged" and m["reach"] is not None and m["reach"] <= n


def _gap(m: Measure) -> float:
    return math.inf if m["final_gap"] is None or m["status"] == "diverged" \
        else m["final_gap"]


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
}

STATUS_TEXT = {"diverged": "發散", "converged": "梯度歸零而停止",
               "stopped": "演算法自行停止", "max_iter": "跑滿迭代數"}


def _sci(v: float) -> str:
    """1e-10 -> '10⁻¹⁰'; other numbers as %g."""
    e = math.log10(v) if v > 0 else 0.5
    if abs(e - round(e)) < 1e-9:
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "10" + str(int(round(e))).translate(sup)
    return f"{v:g}"


def criterion(when: dict, reach_tol: float) -> str:
    """Human-readable form of a predicate (generated, so it cannot drift)."""
    tol = _sci(reach_tol)
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
    }
    parts = [text[k](v) for k, v in when.items()]
    out = parts[0]
    for p in parts[1:]:  # a space between CJK and a leading digit
        out += ("，且 " if p[:1].isdigit() else "，且") + p
    return out


def summary(m: Measure, reach_tol: float) -> str:
    """What the run did, in the words of the lab's result line."""
    if m["status"] == "diverged":
        return f"第 {m['steps'] + 1} 步發散"
    tol = _sci(reach_tol)
    reach = (f"f−f* < {tol} 於第 {m['reach']} 步" if m["reach"] is not None
             else f"未達 f−f* < {tol}（共 {m['steps']} 步）")
    return f"{reach} · 折返 {m['zigzag']} 次 · f 上升 {m['rises']} 步"


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
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{k} needs a number, got {v!r}")


def check(when: dict, m: Measure) -> bool:
    """True when every condition of `when` holds for the measured trace."""
    validate(when)
    return all(CHECKS[k](m, v) for k, v in when.items())


def answer(options: list[dict], m: Measure) -> int:
    """Index of the single true option; ValueError unless exactly one holds."""
    hits = [i for i, o in enumerate(options) if check(o["when"], m)]
    if len(hits) != 1:
        labels = [options[i]["label"] for i in hits]
        raise ValueError(f"expected exactly one true option, got {len(hits)}: "
                         f"{labels} for measured {m}")
    return hits[0]

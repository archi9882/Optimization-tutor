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


# ---------------------------------------------------------------------------
# LP / integer / combinatorial widgets (lp.js, combo.js)
#
# Their questions measure the widget's whole result (api.lp_lab / combo_lab),
# not a 2D trace, so they have their own keys. A key that does not apply to
# the page's method fails the build. LP family (frontmatter lp:):
#
#     steps_min/max        int: animation steps (the widget's "第 k / N 步": N)
#     pivots_min/max       int: tableau pivots (simplex, dual simplex, cutting plane)
#     final_point          [x1, x2]: the method's answer lies within POINT_TOL of
#                          it (branch and bound: the best integer solution)
#     final_z_below/above  float: the answer's objective value z < X / >= X
#     z_trend              "up" | "down" | "mixed": z over the steps never
#                          falls (and rises) / never rises (and falls) / both
#                          (simplex, dual simplex, interior point, active set)
#     nodes_min/max        int: nodes solved by branch and bound
#     cuts_min/max         int: Gomory cuts added
#     cut_points_min/max   int: integer points of the original feasible region
#                          that some cut removes
#     stays_interior       bool: every interior-point iterate strictly inside
#
# Combinatorial family (frontmatter combo:):
#
#     steps_min/max        int: animation steps
#     total_below/above    float: DP total benefit, MST length, assignment hours
#     length_below/above   float: best tour length (2-opt, tabu)
#     reaches_optimum      bool: the result equals the brute-force optimum
#     distinct_tours_min/max  int: different tours visited (2-opt, tabu)
#     uphill_min/max       int: steps where the current tour got longer
#     rejected_min/max     int: Kruskal edges skipped because of a cycle
#     adjustments_min/max  int: Hungarian matrix adjustments
#     release_min/max      {month: int}: DP release of those months >= / <= r
# ---------------------------------------------------------------------------

POINT_TOL = 1e-3      # final_point: distance of the answer to the named point
OPT_TOL = 1e-9        # reaches_optimum: equal to the brute-force optimum up to this

TABLEAU_METHODS = {"simplex", "dual_simplex", "cutting_plane"}
TREND_METHODS = {"simplex", "dual_simplex", "barrier", "active_set"}
TOUR_METHODS = {"two_opt", "tabu"}
TOTAL_METHODS = {"dp", "kruskal", "hungarian"}

METHOD_NAMES = {"simplex": "單純形法", "dual_simplex": "對偶單純形法", "barrier": "內點法",
                "active_set": "有效集法", "branch_bound": "分支定界", "cutting_plane": "割平面法",
                "dp": "動態規劃", "kruskal": "Kruskal", "hungarian": "匈牙利法",
                "two_opt": "2-opt", "tabu": "禁忌搜尋"}

TOTAL_NAMES = {"dp": "總收益", "kruskal": "管線總長（km）", "hungarian": "總時數"}


def _g(v: float) -> str:
    """Integers plainly, other numbers with 3 significant digits (lp._g)."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.3g}"


def _pt(x) -> str:
    return f"({_g(x[0])}, {_g(x[1])})"


def _trend(zs: list[float]) -> str:
    d = [b - a for a, b in zip(zs, zs[1:])]
    up, down = any(v > 1e-9 for v in d), any(v < -1e-9 for v in d)
    return "mixed" if up and down else "up" if up else "down" if down else "flat"


def measure_lp(res: dict, prob) -> Measure:
    """Numbers of an LP-family widget run (res = api.lp_lab(spec), prob its LP)."""
    steps, method = res["steps"], res["method"]
    m: Measure = {"kind": "lp", "method": method, "status": res["status"],
                  "steps": len(steps) - 1}
    last = steps[-1]
    if method == "branch_bound":
        m["x"] = None if last["incumbent"] is None else [float(v) for v in last["incumbent"]]
        m["z"] = last["best_z"]
        m["nodes"] = len(last["tree"])
    else:
        m["x"], m["z"] = [float(v) for v in last["x"]], float(last["z"])
    if method in TABLEAU_METHODS:
        m["pivots"] = sum(1 for s in steps if s.get("pivot"))
    if method in TREND_METHODS:
        m["zs"] = [float(s["z"]) for s in steps]
        m["z_trend"] = _trend(m["zs"])
    if method == "cutting_plane":
        cuts = last["cuts"]
        m["cuts"] = len(cuts)
        pts = integer_points(prob)
        m["int_points"] = len(pts)
        m["cut_points"] = sum(1 for p in pts if any(
            c["a"][0] * p[0] + c["a"][1] * p[1] > c["b"] + 1e-9 for c in cuts))
    if method == "barrier":
        A, b, _ = prob.all_constraints()
        m["min_slack"] = min(bi - a[0] * s["x"][0] - a[1] * s["x"][1]
                             for s in steps for a, bi in zip(A, b))
    return m


def integer_points(prob) -> list[tuple[int, int]]:
    """Integer points of the original feasible region (inside the view box)."""
    A, b, _ = prob.all_constraints()
    x0, x1, y0, y1 = prob.bounds
    return [(i, j) for i in range(math.ceil(x0), math.floor(x1) + 1)
            for j in range(math.ceil(y0), math.floor(y1) + 1)
            if all(a[0] * i + a[1] * j <= bi + 1e-9 for a, bi in zip(A, b))]


def _canon_tour(t: list[int]) -> tuple:
    """A tour up to rotation and direction."""
    i = t.index(0)
    t = t[i:] + t[:i]
    return tuple(t if t[1] < t[-1] else [t[0]] + t[1:][::-1])


def combo_optimum(method: str, res: dict) -> float:
    """Brute-force optimum of the instance (independent of the method's code)."""
    import itertools

    from . import combo as C
    if method == "dp":
        R = C.RESERVOIR

        def best(t: int, s: int) -> float:
            if t == len(R["months"]):
                return 0.0
            avail = s + R["inflow"][t]
            return max(C.reservoir_value(t, r) + best(t + 1, min(avail - r, R["S"]))
                       for r in range(avail + 1))
        return best(0, R["s0"])
    if method == "hungarian":
        M = C.ASSIGN["C"]
        return min(sum(M[i][p[i]] for i in range(len(M)))
                   for p in itertools.permutations(range(len(M))))
    D = C.distances(res["points"])
    n = len(D)
    if method == "kruskal":            # Prim's algorithm
        inside, total = {0}, 0.0
        while len(inside) < n:
            d, j = min((D[i][j], j) for i in inside for j in range(n) if j not in inside)
            inside.add(j)
            total += d
        return total
    return min(sum(D[t[k]][t[(k + 1) % n]] for k in range(n))
               for t in ((0,) + p for p in itertools.permutations(range(1, n))))


def measure_combo(res: dict) -> Measure:
    """Numbers of a combinatorial widget run (res = api.combo_lab(spec))."""
    steps, method = res["steps"], res["method"]
    m: Measure = {"kind": "combo", "method": method, "status": res["status"],
                  "steps": len(steps) - 1, "optimum": combo_optimum(method, res)}
    if method in TOUR_METHODS:
        m["length"] = float(res["length"])
        lens = [s["length"] for s in steps]
        m["uphill"] = sum(b > a + 1e-9 for a, b in zip(lens, lens[1:]))
        m["distinct_tours"] = len({_canon_tour(s["tour"]) for s in steps})
        m["start_length"] = float(lens[0])
        m["optimal"] = m["length"] <= m["optimum"] + OPT_TOL
    else:
        m["total"] = float(res["total"])
        m["optimal"] = (m["total"] >= m["optimum"] - OPT_TOL if method == "dp"
                        else m["total"] <= m["optimum"] + OPT_TOL)
    if method == "kruskal":
        m["considered"] = len(steps) - 1
        m["rejected"] = sum(1 for s in steps[1:] if not s["accepted"])
    if method == "hungarian":
        m["adjustments"] = sum(1 for a, b in zip(steps[1:], steps[2:]) if a["M"] != b["M"])
        m["assignments"] = math.factorial(len(res["C"]))
    if method == "dp":
        m["releases"] = {res["months"][p["t"]]: p["r"] for p in steps[-1]["path"]
                         if p["r"] is not None}
    return m


def _int_bound(key: str, attr: str):
    lo = key.endswith("_min")
    return lambda m, v: m[attr] >= v if lo else m[attr] <= v


WIDGET_INT_KEYS = {
    # key prefix -> (measure field, methods it applies to or None for all, text)
    "steps": ("steps", None, "動畫步數"),
    "pivots": ("pivots", TABLEAU_METHODS, "樞紐次數"),
    "nodes": ("nodes", {"branch_bound"}, "解過的節點數"),
    "cuts": ("cuts", {"cutting_plane"}, "加入的割平面數"),
    "cut_points": ("cut_points", {"cutting_plane"}, "被割掉的可行整數點數"),
    "distinct_tours": ("distinct_tours", TOUR_METHODS, "走過的不同路線數"),
    "uphill": ("uphill", TOUR_METHODS, "路線變長的步數"),
    "rejected": ("rejected", {"kruskal"}, "因形成迴圈而跳過的邊數"),
    "adjustments": ("adjustments", {"hungarian"}, "調整矩陣的次數"),
}
LP_METHODS = {"simplex", "dual_simplex", "barrier", "active_set", "branch_bound", "cutting_plane"}
COMBO_METHODS = {"dp", "kruskal", "hungarian", "two_opt", "tabu"}
WIDGET_KINDS = {"lp": {"steps", "pivots", "nodes", "cuts", "cut_points", "final_point",
                       "final_z_below", "final_z_above", "z_trend", "stays_interior"},
                "combo": {"steps", "total_below", "total_above", "length_below", "length_above",
                          "reaches_optimum", "distinct_tours", "uphill", "rejected",
                          "adjustments", "release_min", "release_max"}}
WIDGET_FLOAT = {"final_z": ("z", LP_METHODS, "目標值 z"), "length": ("length", TOUR_METHODS, None),
                "total": ("total", TOTAL_METHODS, None)}
Z_TRENDS = {"up": "目標值一路不下降（且有上升）", "down": "目標值一路不上升（且有下降）",
            "mixed": "目標值有升有降"}


def _base(key: str) -> str:
    """'pivots_min' -> 'pivots'; keys without a _min/_max/_below/_above suffix as is."""
    for suf in ("_min", "_max", "_below", "_above"):
        if key.endswith(suf):
            return key[: -len(suf)]
    return key


def widget_keys(kind: str) -> set[str]:
    """Every predicate key allowed on a widget of this kind."""
    out = set()
    for k in WIDGET_KINDS[kind]:
        if k in WIDGET_INT_KEYS:
            out |= {f"{k}_min", f"{k}_max"}
        else:
            out.add(k)
    return out


def _applies(key: str) -> set[str] | None:
    """Methods a widget key applies to (None: every method of its kind)."""
    b = _base(key)
    if b in WIDGET_INT_KEYS:
        return WIDGET_INT_KEYS[b][1]
    if b in WIDGET_FLOAT:
        return WIDGET_FLOAT[b][1]
    return {"final_point": LP_METHODS, "z_trend": TREND_METHODS,
            "stays_interior": {"barrier"}, "reaches_optimum": None,
            "release": {"dp"}}.get(b)


def validate_widget(when: dict, kind: str, method: str) -> None:
    """Raise ValueError on keys unknown to this widget or method, or bad values."""
    from . import combo as C
    if not isinstance(when, dict) or not when:
        raise ValueError(f"predicate must be a non-empty mapping, got {when!r}")
    unknown = set(when) - widget_keys(kind)
    if unknown:
        raise ValueError(f"unknown predicate key(s) for the {kind} widget: {sorted(unknown)}; "
                         f"allowed: {sorted(widget_keys(kind))}")
    for k, v in when.items():
        ok = _applies(k)
        if ok is not None and method not in ok:
            raise ValueError(f"{k} does not apply to method {method!r}")
        b = _base(k)
        if b in WIDGET_INT_KEYS:
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{k} needs a whole number >= 0, got {v!r}")
        elif b in WIDGET_FLOAT:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{k} needs a number, got {v!r}")
        elif k == "final_point":
            if not (isinstance(v, list) and len(v) == 2 and all(
                    isinstance(c, (int, float)) and not isinstance(c, bool) for c in v)):
                raise ValueError(f"final_point needs [x1, x2], got {v!r}")
        elif k == "z_trend":
            if v not in Z_TRENDS:
                raise ValueError(f"z_trend must be one of {sorted(Z_TRENDS)}, got {v!r}")
        elif k in ("stays_interior", "reaches_optimum"):
            if not isinstance(v, bool):
                raise ValueError(f"{k} needs true or false, got {v!r}")
        elif b == "release":
            months = C.RESERVOIR["months"]
            if not isinstance(v, dict) or not v or any(
                    mo not in months or isinstance(r, bool) or not isinstance(r, int) or r < 0
                    for mo, r in v.items()):
                raise ValueError(f"{k} needs {{month: whole number}} with months {months}, got {v!r}")
    for b in WIDGET_INT_KEYS:
        lo, hi = when.get(f"{b}_min"), when.get(f"{b}_max")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"{b}_min = {lo} and {b}_max = {hi} leave no possible run")
    for b in WIDGET_FLOAT:
        lo, hi = when.get(f"{b}_above"), when.get(f"{b}_below")
        if lo is not None and hi is not None and not lo < hi:
            raise ValueError(f"{b}_above = {lo} and {b}_below = {hi} leave no possible run")
    for mo in set(when.get("release_min", {})) & set(when.get("release_max", {})):
        if when["release_min"][mo] > when["release_max"][mo]:
            raise ValueError(f"release_min and release_max of {mo} leave no possible run")


def _widget_check(k: str, v, m: Measure) -> bool:
    b = _base(k)
    if b in WIDGET_INT_KEYS:
        return _int_bound(k, WIDGET_INT_KEYS[b][0])(m, v)
    if b in WIDGET_FLOAT:
        val = m[WIDGET_FLOAT[b][0]]
        val = -math.inf if val is None else val      # B&B without an integer solution
        return val < v if k.endswith("_below") else val >= v
    if k == "final_point":
        return m["x"] is not None and math.dist(m["x"], v) < POINT_TOL
    if k == "z_trend":
        return m["z_trend"] == v
    if k == "stays_interior":
        return (m["min_slack"] > 0) == v
    if k == "reaches_optimum":
        return m["optimal"] == v
    if k == "release_min":
        return all(m["releases"][mo] >= r for mo, r in v.items())
    if k == "release_max":
        return all(m["releases"][mo] <= r for mo, r in v.items())
    raise ValueError(f"unknown predicate key {k!r}")


def check_widget(when: dict, m: Measure) -> bool:
    """True when every condition of `when` holds for the measured widget run."""
    validate_widget(when, m["kind"], m["method"])
    return all(_widget_check(k, v, m) for k, v in when.items())


def answer_widget(options: list[dict], m: Measure) -> int:
    """Index of the single true option; ValueError unless exactly one holds."""
    hits = [i for i, o in enumerate(options) if check_widget(o["when"], m)]
    if len(hits) != 1:
        labels = [options[i]["label"] for i in hits]
        shown = {k: v for k, v in m.items() if k != "zs"}
        raise ValueError(f"expected exactly one true option, got {len(hits)}: "
                         f"{labels} for measured {shown}")
    return hits[0]


def _num_text(v: float) -> str:
    """Bounds as written in the frontmatter: 35.999 stays 35.999."""
    if abs(v) < 0.01 and v != 0:
        return _sci(v)
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


STEP_NAMES = {"two_opt": "交換次數", "tabu": "回合數", "active_set": "步數"}


def criterion_widget(when: dict, method: str) -> str:
    """Human-readable form of a widget predicate (generated, so it cannot drift)."""
    parts, done = [], set()
    for k, v in when.items():
        if k in done:
            continue
        b = _base(k)
        if b in WIDGET_INT_KEYS:
            name = STEP_NAMES.get(method, WIDGET_INT_KEYS[b][2]) if b == "steps" \
                else WIDGET_INT_KEYS[b][2]
            lo, hi = when.get(f"{b}_min"), when.get(f"{b}_max")
            done |= {f"{b}_min", f"{b}_max"}
            if lo is not None and hi is not None:
                parts.append(f"{name} = {lo}" if lo == hi else f"{name} {lo}～{hi}")
            elif lo is not None:
                parts.append(f"{name} ≥ {lo}")
            else:
                parts.append(f"{name} = 0" if hi == 0 else f"{name} ≤ {hi}")
        elif b in WIDGET_FLOAT:
            name = WIDGET_FLOAT[b][2] or ("最佳路線長（km）" if b == "length" else TOTAL_NAMES[method])
            lo, hi = when.get(f"{b}_above"), when.get(f"{b}_below")
            done |= {f"{b}_above", f"{b}_below"}
            if lo is not None and hi is not None:
                parts.append(f"{_num_text(lo)} ≤ {name} < {_num_text(hi)}")
            elif lo is not None:
                parts.append(f"{name} ≥ {_num_text(lo)}")
            else:
                parts.append(f"{name} < {_num_text(hi)}")
        elif k == "final_point":
            parts.append(f"找到的整數最優解是 {_pt(v)}" if method == "branch_bound"
                         else f"最後的點在 {_pt(v)}（距離 < {_sci(POINT_TOL)}）")
        elif k == "z_trend":
            parts.append(Z_TRENDS[v])
        elif k == "stays_interior":
            parts.append("每個迭代點都嚴格在可行域內部" if v else "有迭代點碰到或越過邊界")
        elif k == "reaches_optimum":
            ref = "最小生成樹（Prim 獨立計算）" if method == "kruskal" else "窮舉找到的最優值"
            parts.append(f"等於{ref}" if v else f"比{ref}差")
        elif b == "release":
            sign = "≥" if k == "release_min" else "≤"
            parts.append("、".join(f"{mo}放水 {sign} {r}" for mo, r in v.items()))
        done.add(k)
    out = parts[0]
    for p in parts[1:]:
        out += ("，且 " if p[:1].isdigit() else "，且") + p
    return out


def summary_widget(m: Measure) -> str:
    """What the widget run did, in the words of its own notes and panels."""
    meth = m["method"]
    if m["kind"] == "lp":
        zname = "成本" if meth == "dual_simplex" else "目標值" if meth == "active_set" else "z"
        if meth == "branch_bound":
            ans = ("沒有可行的整數解" if m["x"] is None
                   else f"整數最優解 {_pt(m['x'])}，z = {_g(m['z'])}")
            return f"解了 {m['nodes']} 個節點，{ans}"
        if meth == "cutting_plane":
            return (f"加入 {m['cuts']} 條割平面、共 {m['pivots']} 次樞紐，整數最優解 {_pt(m['x'])}，"
                    f"z = {_g(m['z'])}；原可行域的 {m['int_points']} 個整數點被割掉 {m['cut_points']} 個")
        if meth == "barrier":
            where = "每個點都在可行域內部" if m["min_slack"] > 0 else "有點碰到或越過邊界"
            return (f"{m['steps']} 步牛頓法後停在 ({m['x'][0]:.4f}, {m['x'][1]:.4f})，"
                    f"z = {m['z']:.4f}；{where}")
        path = " → ".join(_g(z) for z in m["zs"])
        head = f"{m['pivots']} 次樞紐" if "pivots" in m else f"{m['steps']} 步"
        return f"{head}，停在 {_pt(m['x'])}，{zname}{' ' if zname == 'z' else ''}依序 {path}"
    opt = m["optimum"]
    if meth in TOUR_METHODS:
        unit = "回合" if meth == "tabu" else "次交換"
        out = (f"{m['steps']} {unit}，最佳路線 {m['length']:.3f} km（窮舉最短 {opt:.3f} km）；"
               f"路線變長 {m['uphill']} 次，共走過 {m['distinct_tours']} 條不同路線")
        return out
    if meth == "dp":
        plan = "、".join(f"{mo} {r}" for mo, r in m["releases"].items())
        return f"總收益 {_g(m['total'])}（窮舉最優 {_g(opt)}）；放水 {plan}"
    if meth == "kruskal":
        return (f"看了最短的 {m['considered']} 條邊、跳過 {m['rejected']} 條，"
                f"總長 {m['total']:.3f} km（Prim 獨立算出的最小生成樹 {opt:.3f} km）")
    return (f"調整矩陣 {m['adjustments']} 次，總時數 {_g(m['total'])}"
            f"（{m['assignments']} 種分派中最少為 {_g(opt)}）")

"""Integer programs in 2D: branch and bound, Gomory cutting planes.

Both runners reuse optlab.lp: the LP relaxations, the tableau and the
pivoting code. Each step records what the renderer draws on top of the
feasible region (the node's sub-region, cuts, the incumbent).
"""
from __future__ import annotations

import math

from . import lp as L
from .demos import branch_bound, cutting_plane, dual_simplex, simplex

INF = math.inf


def solve_relaxation(prob: L.LP, A_extra=(), b_extra=()):
    """LP relaxation by vertex enumeration (exact in 2D): (x, z, polygon)."""
    A, b, _ = prob.all_constraints()
    A, b = A + [list(a) for a in A_extra], b + list(b_extra)
    poly = L.feasible_polygon(A, b, prob.bounds)
    if not poly:
        return None, None, []
    x = max(poly, key=lambda p: (round(prob.objective(p), 9), -p[0]))
    return [_clean(v) for v in x], _clean(prob.objective(x)), poly


def _clean(v: float) -> float:
    """Snap values within 1e-9 of an integer (vertex intersections are inexact)."""
    r = round(v)
    return float(r) if abs(v - r) < 1e-9 else v


def _bound_text(prob: L.LP, bounds: list) -> str:
    names = ["x₁", "x₂"]
    return "、".join(f"{names[j]} {'≤' if sign > 0 else '≥'} {v}" for j, sign, v in bounds) or "原問題"


def run_branch_bound(prob: L.LP, decide=branch_bound.bb_decide, max_nodes: int = 40) -> dict:
    """Depth-first branch and bound; the '≤' child is explored first."""
    best_z, best_x = -INF, None
    stack = [(None, [])]                       # (parent id, [(j, ±1, value)])
    tree, steps = [], []
    status = "max_nodes"
    while stack and len(tree) < max_nodes:
        parent, bounds = stack.pop()
        nid = len(tree)                        # nodes are numbered in the order they are solved
        A_ex = [[sign * (1.0 if k == j else 0.0) for k in range(2)] for j, sign, _ in bounds]
        b_ex = [sign * v for _, sign, v in bounds]
        x, z, poly = solve_relaxation(prob, A_ex, b_ex)
        action = decide(x, z, best_z)
        where = _bound_text(prob, bounds)
        node = {"id": nid, "parent": parent, "label": where, "x": x, "z": z, "action": action[0]}
        if action[0] == "infeasible":
            note = f"節點 {nid}（{where}）：LP 鬆弛不可行，剪枝。"
        elif action[0] == "prune":
            note = (f"節點 {nid}（{where}）：LP 鬆弛最優 ({L._g(x[0])}, {L._g(x[1])})，z = {L._g(z)}，"
                    f"不比現有最佳 {L._g(best_z)} 好，剪枝。")
        elif action[0] == "incumbent":
            best_z, best_x = z, x
            note = (f"節點 {nid}（{where}）：LP 鬆弛最優 ({L._g(x[0])}, {L._g(x[1])}) 剛好是整數，"
                    f"z = {L._g(z)}，成為新的現有最佳解。")
        else:
            _, j, lo, hi = action
            name = ["x₁", "x₂"][j]
            note = (f"節點 {nid}（{where}）：LP 鬆弛最優 ({L._g(x[0])}, {L._g(x[1])})，z = {L._g(z)}，"
                    f"{name} = {L._g(x[j])} 不是整數。分成 {name} ≤ {lo} 與 {name} ≥ {hi} 兩個子問題，"
                    f"中間的長條被切掉。")
            node["branch"] = [j, lo, hi]
            stack.append((nid, bounds + [(j, -1, hi)]))
            stack.append((nid, bounds + [(j, 1, lo)]))
        tree.append(node)
        steps.append({"x": x, "z": z, "node": nid, "polygon": poly, "note": note,
                      "incumbent": best_x, "best_z": None if best_z == -INF else best_z,
                      "tree": [dict(n) for n in tree]})
    else:
        status = "optimal" if best_x is not None else "infeasible"
    if status == "optimal":
        steps[-1]["note"] += f"清單已空：整數最優解 ({L._g(best_x[0])}, {L._g(best_x[1])})，z = {L._g(best_z)}。"
    return {"steps": steps, "status": status}


def _cut_line(expr_cols, row_fracs, f0):
    """Cut Σ f_j col_j ≥ f0 written as a·x ≤ b, using each column's affine form in x."""
    a = [0.0, 0.0]
    const = 0.0
    for (a1, a2, c), f in zip(expr_cols, row_fracs):
        a[0] += f * a1
        a[1] += f * a2
        const += f * c
    # Σ f col = a·x + const ≥ f0  ⇔  −a·x ≤ const − f0
    return [-a[0], -a[1]], const - f0


def run_cutting_plane(prob: L.LP, cut=cutting_plane.gomory_cut,
                      choose_row=cutting_plane.choose_cut_row, max_cuts: int = 8) -> dict:
    """Simplex to the LP optimum, then Gomory cuts + dual simplex until integral."""
    T, basis, cols = L.initial_tableau(prob)
    # Each column as an affine function of x: value = a1 x1 + a2 x2 + c.
    expr = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)] + [(-a[0], -a[1], bi) for a, bi in zip(prob.A, prob.b)]
    cuts: list[dict] = []
    x = L.point_from(T, basis)
    steps = [_cp_snap(prob, T, basis, cols, cuts, "起點：原點，先用單純形法解 LP 鬆弛。", None, "simplex")]
    while (choice := simplex.choose_pivot(T)) is not None and choice[0] is not None:
        row, col = choice
        T = simplex.pivot(T, row, col)
        basis[row] = col
        x = L.point_from(T, basis)
        steps.append(_cp_snap(prob, T, basis, cols, cuts,
                              f"單純形法：{cols[col]} 進基。移到 ({L._g(x[0])}, {L._g(x[1])})，z = {L._g(prob.objective(x))}。",
                              (row, col), "simplex"))
    steps[-1]["note"] += "LP 鬆弛已最優。"
    status = "max_cuts"
    for k in range(max_cuts + 1):
        row = choose_row(T)
        if row is None:
            status = "optimal"
            steps[-1]["note"] += f"基變數全是整數：整數最優解 ({L._g(x[0])}, {L._g(x[1])})，z = {L._g(prob.objective(x))}。"
            break
        if k == max_cuts:
            break
        new = cut(T, row)
        n_old = len(T[0]) - 1
        fr = [-v for v in new[:n_old]]
        a_cut, b_cut = _cut_line(expr, fr, -new[-1])
        name = f"割{L.sub(k + 1)}"
        expr.append((-a_cut[0], -a_cut[1], b_cut))
        cuts.append({"a": a_cut, "b": b_cut, "label": name})
        T = [r[:-1] + [0.0, r[-1]] for r in T]
        T.insert(len(T) - 1, list(new))
        basis.append(len(cols) - 1)
        cols = cols[:-1] + [f"t{L.sub(k + 1)}", cols[-1]]
        steps.append(_cp_snap(prob, T, basis, cols, cuts,
                              f"第 {row + 1} 列（{cols[basis[row]] if basis[row] < len(cols) else ''}）右手邊 {L._g(T[row][-1])} 不是整數，"
                              f"產生 {name}：{_fmt_cut(a_cut, b_cut)}。目前的點違反它（新列右手邊為負）。", None, "cut"))
        while (choice := dual_simplex.dual_choose_pivot(T)) is not None and choice[1] is not None:
            r, c = choice
            T = simplex.pivot(T, r, c)
            basis[r] = c
            x = L.point_from(T, basis)
            steps.append(_cp_snap(prob, T, basis, cols, cuts,
                                  f"對偶單純形法：{cols[c]} 進基。移到 ({L._g(x[0])}, {L._g(x[1])})，"
                                  f"z = {L._g(prob.objective(x))}。"
                                  + ("右手邊仍有負值，這個點還違反某些限制。"
                                     if min(rw[-1] for rw in T[:-1]) < -1e-9 else ""),
                                  (r, c), "dual"))
    return {"steps": steps, "status": status}


def _fmt_cut(a, b) -> str:
    """Pretty a·x ≤ b with small integers when possible."""
    scale = next((d for d in range(1, 13) if all(abs(v * d - round(v * d)) < 1e-7 for v in (*a, b))), 1)
    a1, a2, bb = (round(v * scale, 6) for v in (*a, b))
    if all(float(v).is_integer() for v in (a1, a2, bb)):
        g = math.gcd(int(a1), int(a2), int(bb)) or 1
        a1, a2, bb = a1 / g, a2 / g, bb / g
    def term(c, name, first):
        c = L._g(c)
        if c == "0":
            return ""
        sign = "" if first else (" − " if c.startswith("-") else " + ")
        c = c.lstrip("-") if not first else c
        c = "" if c == "1" else ("−" if c == "-1" else c)
        return f"{sign}{c}{name}"
    left = term(a1, "x₁", True) + term(a2, "x₂", a1 == 0)
    return f"{left} ≤ {L._g(bb)}"


def _cp_snap(prob, T, basis, cols, cuts, note, pivot, phase) -> dict:
    x = L.point_from(T, basis)
    A, b, _ = prob.all_constraints()
    A = A + [c["a"] for c in cuts]
    b = b + [c["b"] for c in cuts]
    x = [_clean(v) for v in x]
    snap = L._snapshot(T, basis, cols, x, prob.objective(x), note, pivot)
    snap.update({"polygon": L.feasible_polygon(A, b, prob.bounds),
                 "cuts": [dict(c) for c in cuts], "phase": phase})
    return snap


RUNNERS = {"branch_bound": run_branch_bound, "cutting_plane": run_cutting_plane}

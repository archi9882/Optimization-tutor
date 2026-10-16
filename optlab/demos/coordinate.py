"""Cyclic coordinate descent demo (Gauss-Seidel / SOR on quadratics)."""
from __future__ import annotations

from ..problems import Problem, Vec, as_vec

PSEUDOCODE = """\
for k = 0, 1, 2, …:
    i ← k mod n                  # 輪流選一個座標
    g_i ← ∂f/∂x_i (x)
    h_i ← ∂²f/∂x_i² (x)          # 沿第 i 軸的曲率
    x_i ← x_i − ω·g_i / h_i       # 只動這個座標
"""


def coordinate_step(x, grad, curv, omega, i):
    """一步座標下降：只更新第 i 個座標，其他座標不動。"""
    x_new = list(x)
    x_new[i] = x[i] - omega * grad[i] / curv[i]
    return tuple(x_new)


def coordinate_curvature(problem: Problem, x: Vec, i: int) -> float:
    """|d^2 f / dx_i^2| floored at 1e-2, used as the coordinate step scale."""
    return max(abs(problem.hess(x)[i][i]), 1e-2)


def adapter(fn):
    """Lab loop: cycle i = 0, 1, 0, 1, ... and supply the curvatures."""
    def step(x, problem, state, p):
        k = state.get("k", 0)
        curv = (coordinate_curvature(problem, x, 0),
                coordinate_curvature(problem, x, 1))
        x_new = fn(x, problem.grad(x), curv, p["omega"], k % 2)
        return as_vec(x_new), {"k": k + 1}
    return step


FUNC = coordinate_step
HEADER = ""

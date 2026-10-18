"""Frank-Wolfe (conditional gradient) demo: linear oracle, then a convex step."""
from __future__ import annotations

from ..problems import as_vec
from ..sets import lmo_disk, lmo_polygon

PSEUDOCODE = """\
for k = 0, 1, 2, …:
    s_k ← argmin_{s ∈ C} ∇f(x_k)ᵀ s          # 線性子問題：多邊形上答案必在頂點
    gap_k ← ∇f(x_k)ᵀ (x_k − s_k)              # Frank–Wolfe 間隙 ≥ f(x_k) − f*（凸時）
    γ_k ← 2 / (k + 2)
    x_{k+1} ← x_k + γ_k (s_k − x_k)            # 凸組合：迭代點永遠可行
"""


def frank_wolfe_step(x, grad, k, lmo):
    """一步 Frank–Wolfe：呼叫線性最小化 lmo 找頂點 s，朝 s 走 γ = 2/(k+2)。
    回傳 (x_new, gap)。"""
    s = lmo(grad)
    gap = grad[0] * (x[0] - s[0]) + grad[1] * (x[1] - s[1])
    gamma = 2.0 / (k + 2)
    x_new = (x[0] + gamma * (s[0] - x[0]), x[1] + gamma * (s[1] - x[1]))
    return x_new, gap


GAP_TOL = 1e-8


def adapter(fn):
    """Lab loop: count k, draw the segment from each iterate to its oracle
    vertex s, stop (without moving) once the gap at x is below GAP_TOL."""
    def step(x, problem, state, p):
        if problem.lmo is None:
            return x, {"stop": True, "message": "這個問題沒有提供線性最小化子問題"}
        k = state.get("k", 0)
        x_new, gap = fn(x, problem.grad(x), k, problem.lmo)
        if gap < GAP_TOL:                    # x already optimal: do not move
            return x, {"k": k, "stop": True, "info": {"k": k, "gap": gap}}
        x_new = as_vec(x_new)
        s = problem.lmo(problem.grad(x))
        s_next = problem.lmo(problem.grad(x_new))
        info = {"k": k, "gap": gap, "γ": 2.0 / (k + 2), "頂點 s": list(s)}
        return x_new, {"k": k + 1, "info": info,
                       "poly_used": [list(x), list(s)], "poly": [list(x_new), list(s_next)],
                       "poly_kind": "open"}
    return step


FUNC = frank_wolfe_step
HELPERS = [lmo_polygon, lmo_disk]
HEADER = "import math\n\n\n"

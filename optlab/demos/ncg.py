"""Nonlinear conjugate gradient demo (Polak-Ribiere+, secant line search)."""
from __future__ import annotations

from ..problems import as_vec
from ._linalg import backtracking, dot

PSEUDOCODE = """\
d ← −∇f(x₀)
for k = 0, 1, 2, …:
    α ← 沿 d 的線搜尋（二次函數上可取精確步長）
    x ← x + α·d
    g_new ← ∇f(x)
    β ← max(0, g_newᵀ(g_new − g) / gᵀg)     # Polak–Ribière+
    d ← −g_new + β·d                         # 新方向與舊方向「共軛」
    若 d 不是下降方向：d ← −g_new           # 重新開始
"""


def secant_line_search(f, grad_fn, x, g, d):
    """沿 d 用兩次梯度估計曲率 dᵀHd，取二次模型的最佳步長，再用 Armijo 保底。"""
    slope = dot(g, d)
    t = 1e-6
    g_t = grad_fn((x[0] + t * d[0], x[1] + t * d[1]))
    curv = (dot(g_t, d) - slope) / t
    alpha = -slope / curv if curv > 0 else 1.0
    return backtracking(f, x, f(x), g, d, alpha)


def ncg_step(x, d_prev, g_prev, f, grad_fn):
    """一步非線性共軛梯度（Polak–Ribière+）：新方向 = 負梯度 + β·舊方向。"""
    g = grad_fn(x)
    if d_prev is None:
        d = (-g[0], -g[1])
    else:
        y = (g[0] - g_prev[0], g[1] - g_prev[1])
        beta = max(0.0, dot(g, y) / dot(g_prev, g_prev))
        d = (-g[0] + beta * d_prev[0], -g[1] + beta * d_prev[1])
        if dot(g, d) >= 0:              # 不是下降方向：重新從負梯度開始
            d = (-g[0], -g[1])
    x_new = secant_line_search(f, grad_fn, x, g, d)
    return x_new, d, g


def adapter(fn):
    """Lab loop: carry the previous direction and gradient."""
    def step(x, problem, state, p):
        x_new, d, g = fn(x, state.get("d"), state.get("g"), problem.f, problem.grad)
        return as_vec(x_new), {"d": as_vec(d), "g": as_vec(g)}
    return step


FUNC = ncg_step
HELPERS = [dot, backtracking, secant_line_search]
HEADER = ""

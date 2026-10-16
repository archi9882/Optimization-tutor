"""BFGS demo: inverse-Hessian approximation updated from gradient changes."""
from __future__ import annotations

from ..problems import as_vec
from ._linalg import backtracking, dot

PSEUDOCODE = """\
H ← I                               # Hessian 反矩陣的近似
for k = 0, 1, 2, …:
    d ← −H·∇f(x)
    x_new ← 沿 d 做線搜尋
    s ← x_new − x,   y ← ∇f(x_new) − ∇f(x)
    若 sᵀy > 0：
        ρ ← 1 / (sᵀy)
        H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
    x ← x_new
"""


def bfgs_step(x, H, f, grad_fn):
    """一步 BFGS：H 近似 Hessian 的反矩陣，只用梯度差逐步修正。"""
    g = grad_fn(x)
    d = (-(H[0][0] * g[0] + H[0][1] * g[1]), -(H[1][0] * g[0] + H[1][1] * g[1]))
    x_new = backtracking(f, x, f(x), g, d)
    s = (x_new[0] - x[0], x_new[1] - x[1])
    g_new = grad_fn(x_new)
    y = (g_new[0] - g[0], g_new[1] - g[1])
    sy = dot(s, y)
    if sy <= 1e-12:                     # 曲率條件不成立：不更新 H
        return x_new, H
    rho = 1.0 / sy
    V = [[(1.0 if i == j else 0.0) - rho * s[i] * y[j] for j in range(2)]
         for i in range(2)]
    VH = [[sum(V[i][k] * H[k][j] for k in range(2)) for j in range(2)]
          for i in range(2)]
    H_new = [[sum(VH[i][k] * V[j][k] for k in range(2)) + rho * s[i] * s[j]
              for j in range(2)] for i in range(2)]
    return x_new, H_new


def adapter(fn):
    """Lab loop: carry the inverse-Hessian approximation H (starts at I)."""
    def step(x, problem, state, p):
        H = state.get("H", [[1.0, 0.0], [0.0, 1.0]])
        x_new, H_new = fn(x, H, problem.f, problem.grad)
        return as_vec(x_new), {"H": H_new}
    return step


FUNC = bfgs_step
HELPERS = [dot, backtracking]
HEADER = ""

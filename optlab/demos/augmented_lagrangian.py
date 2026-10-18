"""Augmented Lagrangian (method of multipliers) demo: GD on L_A, then update λ."""
from __future__ import annotations

import math

from ..problems import as_vec
from ._linalg import backtracking, dot

PSEUDOCODE = """\
s_i = λ_i + ρ g_i(x)，不等式限制再取 max(0, ·)
L_A(x; λ) = f(x) + Σ_i (s_i² − λ_i²) / (2ρ)      # 等式時就是 f + λh + (ρ/2)h²
x ← x₀，λ ← 0，ρ 固定
重複（外迴圈）：
    重複（內迴圈）：對 L_A(·; λ) 做一步梯度下降（Armijo 回溯），直到 ‖∇L_A‖ 夠小
    λ_i ← s_i(x)                                   # 乘數更新：λ ← λ + ρ·g（不等式截在 0）
"""


def shifted(c, lam_i, rho, x):
    """s_i = λ_i + ρ g_i(x)；不等式限制截在 0 以上（限制不起作用時乘數歸零）。"""
    s = lam_i + rho * c.g(x)
    return s if c.kind == "eq" else max(0.0, s)


def augmented(x, lam, rho, f, cons):
    """增廣 Lagrangian L_A(x; λ) = f(x) + Σ (s_i² − λ_i²) / (2ρ)。"""
    return f(x) + sum(shifted(c, li, rho, x) ** 2 - li ** 2
                      for c, li in zip(cons, lam)) / (2 * rho)


def al_step(x, lam, rho, f, grad, cons, tol):
    """‖∇L_A‖ ≥ tol：對 L_A 做一步梯度下降；否則 x 不動、更新乘數 λ_i ← s_i。
    回傳 (x_new, lam_new, ‖∇L_A(x)‖)。"""
    g = list(grad(x))
    for c, li in zip(cons, lam):
        s, dg = shifted(c, li, rho, x), c.grad(x)
        g[0] += s * dg[0]                          # ∇L_A = ∇f + Σ s_i ∇g_i
        g[1] += s * dg[1]
    gnorm = math.sqrt(dot(g, g))
    if gnorm < tol:                                # 內圈收斂：更新乘數
        return x, [shifted(c, li, rho, x) for c, li in zip(cons, lam)], gnorm
    def L(p):
        return augmented(p, lam, rho, f, cons)
    x_new = backtracking(L, x, L(x), g, (-g[0], -g[1]))
    return x_new, list(lam), gnorm


INNER_TOL = 1e-6     # 內迴圈停止：‖∇L_A‖ < INNER_TOL
LAM_TOL = 1e-6       # 乘數幾乎不再變（= 限制已滿足）就停
MAX_OUTER = 50


def adapter(fn):
    """Lab loop: λ starts at 0 and ρ stays fixed. After `inner` inner steps the
    multiplier update is forced (tol = ∞). Stop when an update after a converged
    inner loop changes λ by less than LAM_TOL, or after MAX_OUTER updates."""
    def step(x, problem, state, p):
        cons = problem.constraints
        lam = state.get("lam", [0.0] * len(cons))
        n_in = state.get("n_in", 0) + 1
        tol = math.inf if n_in > int(p["inner"]) else INNER_TOL
        x_new, lam_new, gnorm = fn(x, lam, p["rho"], problem.f, problem.grad, cons, tol)
        x_new = as_vec(x_new)
        lam_new = [float(v) for v in lam_new]
        updated = gnorm < tol
        k = state.get("outer", 0) + (1 if updated else 0)
        info = {"外圈": k if updated else k + 1, "λ": lam_new, "內圈步": n_in, "‖∇L_A‖": gnorm}
        change = max([abs(a - b) for a, b in zip(lam_new, lam)] + [0.0])
        converged = updated and gnorm < INNER_TOL and change < LAM_TOL
        capped = updated and not converged and k >= MAX_OUTER
        state = {"lam": lam_new, "n_in": 0 if updated else n_in, "outer": k,
                 "info": info, "outer_end": updated, "stop": converged or capped}
        if capped:
            state["message"] = f"達到外圈上限 {MAX_OUTER} 次，乘數尚未收斂"
        return x_new, state
    return step


FUNC = al_step
HELPERS = [shifted, augmented, dot, backtracking]
HEADER = "import math\n\n\n"

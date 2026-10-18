"""Quadratic penalty demo: gradient descent on f + (rho/2)·violation², rho grows."""
from __future__ import annotations

import math

from ..problems import as_vec
from ._linalg import backtracking, dot

PSEUDOCODE = """\
P_ρ(x) = f(x) + (ρ/2) Σ_i v_i(x)²       # v_i = max(0, g_i(x))；等式限制 v_i = h_i(x)
x ← x₀，ρ ← ρ₀
重複（外迴圈）：
    重複（內迴圈）：對 P_ρ 做一步梯度下降（Armijo 回溯），直到 ‖∇P_ρ‖ 夠小或步數用完
    ρ ← γ·ρ                               # 懲罰加重，x 被推向可行域
"""


def violation(c, x):
    """限制違反量：不等式 g(x) ≤ 0 取 max(0, g)；等式 h(x) = 0 取 h 本身。"""
    v = c.g(x)
    return v if c.kind == "eq" else max(0.0, v)


def penalized(x, rho, f, cons):
    """懲罰函數 P_ρ(x) = f(x) + (ρ/2) Σ v_i(x)²。"""
    return f(x) + 0.5 * rho * sum(violation(c, x) ** 2 for c in cons)


def penalty_step(x, rho, f, grad, cons):
    """對 P_ρ 做一步梯度下降（Armijo 回溯）；回傳 (x_new, ‖∇P_ρ(x)‖)。"""
    g = list(grad(x))
    for c in cons:
        v, dg = violation(c, x), c.grad(x)
        g[0] += rho * v * dg[0]                    # ∇[(ρ/2) v²] = ρ·v·∇g
        g[1] += rho * v * dg[1]
    def P(p):
        return penalized(p, rho, f, cons)
    x_new = backtracking(P, x, P(x), g, (-g[0], -g[1]))
    return x_new, math.sqrt(dot(g, g))


INNER_TOL = 1e-3     # 內迴圈停止：‖∇P_ρ‖ < INNER_TOL
RHO_MAX = 1e4        # ρ 超過這個值就停


def adapter(fn):
    """Lab loop: inner steps until ‖∇P_ρ‖ < INNER_TOL or `inner` steps,
    then rho ← gamma·rho; stop after the outer iteration with rho ≥ RHO_MAX."""
    def step(x, problem, state, p):
        rho = state.get("rho", p["rho0"])
        n_in = state.get("n_in", 0) + 1
        x_new, gnorm = fn(x, rho, problem.f, problem.grad, problem.constraints)
        x_new = as_vec(x_new)
        done = gnorm < INNER_TOL or n_in >= int(p["inner"])
        k = state.get("outer", 0)
        viol = max([0.0] + [abs(violation(c, x_new)) for c in problem.constraints])
        info = {"外圈": k + 1, "ρ": rho, "內圈步": n_in, "‖∇P‖": gnorm, "違反量": viol}
        if done:
            k, n_in, rho_next = k + 1, 0, rho * p["gamma"]
        else:
            rho_next = rho
        return x_new, {"rho": rho_next, "n_in": n_in, "outer": k, "info": info,
                       "outer_end": done, "stop": done and rho >= RHO_MAX * (1 - 1e-9)}
    return step


FUNC = penalty_step
HELPERS = [violation, penalized, dot, backtracking]
HEADER = "import math\n\n\n"

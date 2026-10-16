"""Trust-region demo with the dogleg step."""
from __future__ import annotations

import math

from ..problems import as_vec
from ._linalg import dot, matvec, norm, solve2

PSEUDOCODE = """\
Δ ← Δ₀
for k = 0, 1, 2, …:
    m(p) = f(x) + gᵀp + ½ pᵀHp          # x 附近的二次模型
    p ← 在 ‖p‖ ≤ Δ 內近似最小化 m（dogleg 折線）
    ρ ← 實際下降 / 模型預測下降
    若 ρ < 0.25：Δ ← Δ/4                 # 模型不可信，縮小範圍
    若 ρ > 0.75 且 p 碰到邊界：Δ ← 2Δ    # 模型很準，放大範圍
    若 ρ > 0.1：x ← x + p                # 否則拒絕這一步
"""


def dogleg_step(x, radius, f, grad_fn, hess_fn):
    """一步信賴域（dogleg）：在半徑 Δ 內最小化二次模型，再依預測準不準調整 Δ。"""
    g, H = grad_fn(x), hess_fn(x)
    if norm(g) == 0:                                   # 已在駐點
        return x, radius
    gHg = dot(g, matvec(H, g))
    p_newton = solve2(H, (-g[0], -g[1]))
    steepest = (-radius * g[0] / norm(g), -radius * g[1] / norm(g))
    if gHg <= 0 or p_newton is None or dot(p_newton, g) >= 0:
        p = steepest                                   # H 非正定：沿最陡方向走到邊界
    elif norm(p_newton) <= radius:
        p = p_newton                                   # 牛頓步就在範圍內
    else:
        t = dot(g, g) / gHg
        p_u = (-t * g[0], -t * g[1])                   # 沿 −g 的模型最低點
        if norm(p_u) >= radius:
            p = steepest
        else:                                          # 沿折線 p_u → p_newton 走到邊界
            v = (p_newton[0] - p_u[0], p_newton[1] - p_u[1])
            a, b, c = dot(v, v), 2 * dot(p_u, v), dot(p_u, p_u) - radius ** 2
            tau = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
            p = (p_u[0] + tau * v[0], p_u[1] + tau * v[1])
    predicted = -(dot(g, p) + 0.5 * dot(p, matvec(H, p)))
    x_try = (x[0] + p[0], x[1] + p[1])
    ratio = (f(x) - f(x_try)) / predicted if predicted > 0 else -1.0
    if ratio < 0.25:
        radius = 0.25 * radius
    elif ratio > 0.75 and abs(norm(p) - radius) < 1e-9:
        radius = min(2.0 * radius, 10.0)
    return (x_try if ratio > 0.1 else x), radius


def adapter(fn):
    """Lab loop: carry the radius and report the radius used at each step."""
    def step(x, problem, state, p):
        radius = state.get("radius", p["radius0"])
        x_new, new_radius = fn(x, radius, problem.f, problem.grad, problem.hess)
        return as_vec(x_new), {"radius": float(new_radius), "radius_used": radius}
    return step


FUNC = dogleg_step
HELPERS = [dot, norm, matvec, solve2]
HEADER = "import math\n\n\n"

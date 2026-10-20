"""ADMM demo: min f(x) + g(z) s.t. x − z = 0, one x-update and one z-update per step."""
from __future__ import annotations

import math

from ..problems import as_vec
from ._linalg import solve2

PSEUDOCODE = """\
問題：min f(x) + g(z)  s.t.  x − z = 0      # g = 可行集合的指示函數，或 λ‖z‖₁
x ← x₀，z ← x₀，u ← 0                        # u = 縮放的對偶變數（乘數 / ρ）
重複：
    x ← argmin_x f(x) + (ρ/2)‖x − z + u‖²    # 只碰 f：二次函數時解一個線性方程組
    z ← prox_g(x + u)                         # 只碰 g：投影，或軟門檻
    u ← u + x − z                             # 乘數更新（和增廣 Lagrangian 相同）
    r = ‖x − z‖（原始殘差），s = ρ‖z − z_舊‖（對偶殘差）
    兩者都夠小 → 停止
"""


def soft_threshold(v, k):
    """λ‖z‖₁ 的 prox：每個座標往 0 收 k，越過 0 就停在 0。"""
    return tuple(math.copysign(max(abs(vi) - k, 0.0), vi) for vi in v)


def x_update(v, rho, grad, hess, x):
    """argmin f(x) + (ρ/2)‖x − v‖²：從 x 出發的牛頓法（f 是二次函數時一步就到）。"""
    for _ in range(20):
        g, H = grad(x), hess(x)
        r = (g[0] + rho * (x[0] - v[0]), g[1] + rho * (x[1] - v[1]))
        d = solve2(((H[0][0] + rho, H[0][1]), (H[1][0], H[1][1] + rho)), (-r[0], -r[1]))
        x = (x[0] + d[0], x[1] + d[1])
        if math.hypot(d[0], d[1]) < 1e-12:
            break
    return x


def admm_step(x, z, u, rho, grad, hess, project, l1):
    """ADMM 的一步（縮放形式）。project：可行集合的投影；沒有投影時 g = l1·‖z‖₁。

    回傳 (x, z, u, r, s)：r = ‖x − z‖ 原始殘差，s = ρ‖z − z_舊‖ 對偶殘差。
    """
    x = x_update((z[0] - u[0], z[1] - u[1]), rho, grad, hess, x)
    v = (x[0] + u[0], x[1] + u[1])
    z_old = z
    z = project(v) if project else soft_threshold(v, l1 / rho)
    u = (u[0] + x[0] - z[0], u[1] + x[1] - z[1])
    r = math.hypot(x[0] - z[0], x[1] - z[1])
    s = rho * math.hypot(z[0] - z_old[0], z[1] - z_old[1])
    return x, z, u, r, s


TOL = 1e-9          # r 與 s 都小於 TOL 時停止


def adapter(fn):
    """Lab loop: z starts at x0 and u at 0; stop when both residuals are below TOL.
    The problem supplies its projection, or an l1 weight (Problem.l1)."""
    def step(x, problem, state, p):
        if problem.project is None and not problem.l1:
            return x, {"stop": True, "message": "這個問題沒有提供投影或 ℓ1 項"}
        z = state.get("z", x)
        u = state.get("u", (0.0, 0.0))
        x_new, z_new, u_new, r, s = fn(x, z, u, p["rho"], problem.grad, problem.hess,
                                       problem.project, problem.l1)
        x_new, z_new, u_new = as_vec(x_new), as_vec(z_new), as_vec(u_new)
        info = {"z": list(z_new), "ρu": [p["rho"] * u_new[0], p["rho"] * u_new[1]],
                "原始殘差 r": r, "對偶殘差 s": s}
        return x_new, {"z": z_new, "u": u_new, "info": info,
                       "stop": r < TOL and s < TOL}
    return step


FUNC = admm_step
HELPERS = [soft_threshold, solve2, x_update]
HEADER = "import math\n\n\n"

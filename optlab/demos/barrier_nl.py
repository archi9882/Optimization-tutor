"""Log-barrier demo for g_i(x) <= 0: Newton steps on f − μ Σ log(−g_i), μ shrinks."""
from __future__ import annotations

import math

from ..problems import as_vec
from ._linalg import backtracking, dot, solve2

PSEUDOCODE = """\
B_μ(x) = f(x) − μ Σ_i log(−g_i(x))        # 只在嚴格可行 g_i(x) < 0 處有定義
x ← 嚴格可行的起點，μ ← μ₀
重複（外迴圈）：
    重複（內迴圈）：對 B_μ 做一步牛頓法，回溯到仍嚴格可行且 B_μ 下降
                    直到 ‖∇B_μ‖ 夠小或步數用完
    μ ← μ / σ                              # 牆變薄，x 沿中心路徑走向邊界上的最優解
"""


def barrier_value(x, mu, f, cons):
    """B_μ(x)；離開可行域內部就回傳 +∞，回溯線搜尋自然會把步長縮回來。"""
    if any(c.g(x) >= 0 for c in cons):
        return math.inf
    return f(x) - mu * sum(math.log(-c.g(x)) for c in cons)


def barrier_step(x, mu, f, grad, hess, cons):
    """對 B_μ 做一步牛頓法（回溯）；回傳 (x_new, ‖∇B_μ(x)‖)。x 必須嚴格可行。"""
    g = list(grad(x))
    H = [list(row) for row in hess(x)]
    for c in cons:
        gi, dg, Hc = c.g(x), c.grad(x), c.hess(x)   # gi < 0
        for j in range(2):
            g[j] += -mu * dg[j] / gi                 # ∇[−μ log(−g)] = −μ ∇g / g
            for k in range(2):
                H[j][k] += mu * (dg[j] * dg[k] / gi ** 2 - Hc[j][k] / gi)
    d = solve2(H, (-g[0], -g[1]))
    if d is None or dot(d, g) >= 0:                  # Hessian 非正定：改走負梯度
        d = (-g[0], -g[1])
    def B(p):
        return barrier_value(p, mu, f, cons)
    x_new = backtracking(B, x, B(x), g, d)
    return x_new, math.sqrt(dot(g, g))


INNER_TOL = 1e-6    # 內迴圈停止：‖∇B_μ‖ < INNER_TOL
MU_MIN = 1e-6       # μ 小於這個值就停


def adapter(fn):
    """Lab loop: inner Newton steps until ‖∇B_μ‖ < INNER_TOL or `inner` steps,
    then mu ← mu / sigma; stop after the outer iteration with mu ≤ MU_MIN.
    An infeasible start (or an equality constraint) stops at once."""
    def step(x, problem, state, p):
        cons = problem.constraints
        if any(c.kind == "eq" for c in cons):
            return x, {"stop": True, "message": "障礙法只處理不等式限制"}
        if not state and any(c.g(x) >= 0 for c in cons):
            return x, {"stop": True,
                       "message": "起點不在可行域內部：log(−g) 沒有定義，障礙法無法開始"}
        mu = state.get("mu", p["mu0"])
        n_in = state.get("n_in", 0) + 1
        x_new, gnorm = fn(x, mu, problem.f, problem.grad, problem.hess, cons)
        x_new = as_vec(x_new)
        done = gnorm < INNER_TOL or n_in >= int(p["inner"])
        k = state.get("outer", 0)
        slack = min(-c.g(x_new) for c in cons)
        info = {"外圈": k + 1, "μ": mu, "內圈步": n_in, "‖∇B‖": gnorm, "最小餘裕 −g": slack}
        if done:
            k, n_in, mu_next = k + 1, 0, mu / p["sigma"]
        else:
            mu_next = mu
        return x_new, {"mu": mu_next, "n_in": n_in, "outer": k, "info": info,
                       "outer_end": done, "stop": done and mu <= MU_MIN * (1 + 1e-9)}
    return step


FUNC = barrier_step
HELPERS = [barrier_value, dot, solve2, backtracking]
HEADER = "import math\n\n\n"

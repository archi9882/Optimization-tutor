"""SQP demo: solve a QP model of the problem, then a line search on an l1 merit."""
from __future__ import annotations

import math

from ..problems import as_vec
from ._linalg import dot, matvec

PSEUDOCODE = """\
x ← x₀，λ ← 0，μ ← 0，B ← I（或用 Lagrangian 的 Hessian）
重複：
    QP 子問題：min_d ½dᵀBd + ∇f(x)ᵀd                    # 二次模型
               s.t. g_i(x) + ∇g_i(x)ᵀd ≤ 0（等式為 = 0） # 線性化的限制
    QP 不可行 → 停止（線性化的限制互相矛盾）
    (d, λ) ← QP 的解與乘數；‖d‖ 夠小 → 停止（x 是 KKT 點）
    μ ← max(μ, 2·max|λ_i|)                              # 價值函數的懲罰係數
    φ(x) = f(x) + μ Σ 違反量_i(x)                       # ℓ1 價值函數
    t ← 1；φ(x + t·d) 下降不夠就 t ← t/2                # 回溯線搜尋
    x ← x + t·d
    B ← ∇²L(x, λ)（不正定就加 τI）或阻尼 BFGS 更新
"""


def solve_lin(M, b):
    """高斯消去法（部分選主元）解 M·z = b；M 奇異時回傳 None。"""
    n = len(b)
    A = [list(row) + [b[i]] for i, row in enumerate(M)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[p][k]) < 1e-12:
            return None
        A[k], A[p] = A[p], A[k]
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n + 1):
                A[i][j] -= m * A[k][j]
    z = [0.0] * n
    for i in reversed(range(n)):
        z[i] = (A[i][n] - sum(A[i][j] * z[j] for j in range(i + 1, n))) / A[i][i]
    return z


def qp_subproblem(B, gf, lin):
    """min ½dᵀBd + gfᵀd，限制 aᵀd + c ≤ 0（等式 = 0），lin = [(kind, a, c), ...]。

    二維時最多兩個限制同時起作用：逐一試工作集 W（等式一定在內），解 KKT 方程組
        [ B    A_Wᵀ ] [d]   [ −gf  ]
        [ A_W   0   ] [λ] = [ −c_W ]
    第一個「其他限制都滿足、不等式乘數 ≥ 0」的解就是答案（B 正定時唯一）。
    回傳 (d, λ)；每個工作集都不行時回傳 None（線性化的限制無解）。
    """
    eq = [i for i, (kind, a, c) in enumerate(lin) if kind == "eq"]
    ineq = [i for i, (kind, a, c) in enumerate(lin) if kind != "eq"]
    trials = [[]] + [[i] for i in ineq] + [[i, j] for i in ineq for j in ineq if i < j]
    for extra in trials:
        W = eq + extra
        if len(W) > 2:
            continue
        M = [[B[0][0], B[0][1]] + [lin[i][1][0] for i in W],
             [B[1][0], B[1][1]] + [lin[i][1][1] for i in W]]
        M += [[lin[i][1][0], lin[i][1][1]] + [0.0] * len(W) for i in W]
        sol = solve_lin(M, [-gf[0], -gf[1]] + [-lin[i][2] for i in W])
        if sol is None:
            continue
        d = (sol[0], sol[1])
        lam = [0.0] * len(lin)
        for i, m in zip(W, sol[2:]):
            lam[i] = m
        primal = all(dot(lin[i][1], d) + lin[i][2] <= 1e-9 for i in ineq)
        dual = all(lam[i] >= -1e-12 for i in ineq)
        if primal and dual:
            return d, lam
    return None


def merit(x, mu, f, cons):
    """ℓ1 價值函數 φ(x) = f(x) + μ Σ 違反量（不等式 max(0, g)，等式 |h|）。"""
    viol = sum(abs(c.g(x)) if c.kind == "eq" else max(0.0, c.g(x)) for c in cons)
    return f(x) + mu * viol


def hessian_lagrangian(x, lam, hess, cons, delta=1e-4):
    """B = ∇²f + Σ λ_i ∇²g_i；最小特徵值 < delta 時加 τI（Hessian 修正）。回傳 (B, τ)。"""
    H = [list(row) for row in hess(x)]
    for c, li in zip(cons, lam):
        Hc = c.hess(x)
        for i in range(2):
            for j in range(2):
                H[i][j] += li * Hc[i][j]
    a, b, d = H[0][0], H[0][1], H[1][1]
    low = (a + d) / 2 - math.sqrt(((a - d) / 2) ** 2 + b * b)   # 2×2 對稱矩陣的最小特徵值
    tau = max(0.0, delta - low)
    return [[a + tau, b], [b, d + tau]], tau


def damped_bfgs(B, s, y):
    """Powell 阻尼 BFGS：sᵀy 不夠大時把 y 換成 θy + (1−θ)Bs，B 保持正定。"""
    Bs = matvec(B, s)
    sBs, sy = dot(s, Bs), dot(s, y)
    if sBs < 1e-16:
        return B
    theta = 1.0 if sy >= 0.2 * sBs else 0.8 * sBs / (sBs - sy)
    r = [theta * y[i] + (1 - theta) * Bs[i] for i in range(2)]
    sr = dot(s, r)
    return [[B[i][j] - Bs[i] * Bs[j] / sBs + r[i] * r[j] / sr for j in range(2)]
            for i in range(2)]


def sqp_step(x, B, mu, f, grad, cons):
    """SQP 的一步：解 QP 子問題得到 d，再對 ℓ1 價值函數回溯線搜尋。

    回傳 (x_new, λ, μ, t, d)；線搜尋失敗時 t = 0、x 不動；QP 無解時回傳 None。
    """
    gf = grad(x)
    lin = [(c.kind, c.grad(x), c.g(x)) for c in cons]     # g(x) + ∇g(x)ᵀd
    sol = qp_subproblem(B, gf, lin)
    if sol is None:
        return None
    d, lam = sol
    mu = max(mu, 2 * max([abs(v) for v in lam] + [0.0]))   # μ > max|λ|：d 是 φ 的下降方向
    viol = sum(abs(c) if kind == "eq" else max(0.0, c) for kind, a, c in lin)
    slope = dot(gf, d) - mu * viol                          # φ 沿 d 的方向導數（上界）
    phi = merit(x, mu, f, cons)
    t = 1.0
    for _ in range(40):
        x_new = (x[0] + t * d[0], x[1] + t * d[1])
        if merit(x_new, mu, f, cons) <= phi + 1e-4 * t * slope:
            return x_new, lam, mu, t, d
        t /= 2
    return x, lam, mu, 0.0, d                               # 線搜尋失敗：x 不動，t = 0


STEP_TOL = 1e-10     # ‖d‖ 小於這個值：x 已是 KKT 點，停止


def grad_lagrangian(x, lam, grad, cons):
    """∇ₓL = ∇f + Σ λ_i ∇g_i（BFGS 的 y 用它）。"""
    g = list(grad(x))
    for c, li in zip(cons, lam):
        dg = c.grad(x)
        g[0] += li * dg[0]
        g[1] += li * dg[1]
    return g


def adapter(fn, bfgs=False, modify=True):
    """Lab loop: B is the Hessian of the Lagrangian at x with the last multipliers
    (modified unless modify=False), or a damped BFGS matrix starting from I. Stop when ‖d‖ < STEP_TOL
    or when the QP has no solution (x stays put)."""
    def step(x, problem, state, p):
        cons = problem.constraints
        lam = state.get("lam", [0.0] * len(cons))
        tau = 0.0
        if bfgs:
            B = state.get("B", [[1.0, 0.0], [0.0, 1.0]])
        else:
            delta = 1e-4 if modify else -math.inf      # 不修正時 B 可能不正定
            B, tau = hessian_lagrangian(x, lam, problem.hess, cons, delta)
        out = fn(x, B, state.get("mu", 0.0), problem.f, problem.grad, cons)
        if out is None:
            return x, {"stop": True, "dots": True, "lam": lam, "poly": [], "poly_used": [],
                       "info": {"λ": lam, "QP": "無解"},
                       "message": "QP 子問題無解（線性化的限制互相矛盾，或 B 不正定）"}
        x_new, lam_new, mu, t, d = out
        x_new = as_vec(x_new)
        lam_new = [float(v) for v in lam_new]
        dn = math.hypot(d[0], d[1])
        info = {"λ": lam_new, "μ": mu, "步長 t": t, "‖d‖": dn}
        stuck = t == 0 and dn >= STEP_TOL
        if not bfgs and modify:
            info["修正 τ"] = tau
        new = {"lam": lam_new, "mu": mu, "info": info, "dots": True,
               "poly": [list(x), [x[0] + d[0], x[1] + d[1]]], "poly_used": [],
               "poly_kind": "open", "stop": dn < STEP_TOL or stuck}
        if stuck:
            new["message"] = "線搜尋失敗：d 不是價值函數的下降方向（B 不正定？）"
        if bfgs:
            s = (x_new[0] - x[0], x_new[1] - x[1])
            g1 = grad_lagrangian(x_new, lam_new, problem.grad, cons)
            g0 = grad_lagrangian(x, lam_new, problem.grad, cons)
            new["B"] = damped_bfgs(B, s, (g1[0] - g0[0], g1[1] - g0[1]))
        return x_new, new
    return step


def bfgs_adapter(fn):
    """The same step with a damped BFGS matrix instead of the exact Hessian."""
    return adapter(fn, bfgs=True)


def raw_adapter(fn):
    """The exact Hessian of the Lagrangian without modification (may be indefinite)."""
    return adapter(fn, modify=False)


FUNC = sqp_step
HELPERS = [dot, matvec, solve_lin, qp_subproblem, merit, hessian_lagrangian, damped_bfgs,
           grad_lagrangian]
HEADER = "import math\n\n\n"

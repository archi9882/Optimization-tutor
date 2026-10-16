"""Levenberg-Marquardt demo: damped Gauss-Newton with adaptive lambda."""
from __future__ import annotations

from ..problems import as_vec
from ._linalg import solve2

PSEUDOCODE = """\
問題形式：f(x) = Σ r_i(x)²
λ ← λ₀
for k = 0, 1, 2, …:
    解 (JᵀJ + λI)·d = −Jᵀr
    若 f(x + d) < f(x)：接受，x ← x + d，λ ← λ/10   # 更像牛頓
    否則：拒絕，x 不動，λ ← 10·λ                    # 更像梯度下降
"""


def lm_step(x, lam, residual_fn, jac_fn):
    """一步 Levenberg-Marquardt：下降就接受並縮小 λ，否則拒絕並放大 λ。"""
    r, J = residual_fn(x), jac_fn(x)
    m = len(r)
    JtJ = [[sum(J[k][i] * J[k][j] for k in range(m)) for j in range(2)]
           for i in range(2)]
    Jtr = [sum(J[k][i] * r[k] for k in range(m)) for i in range(2)]
    A = [[JtJ[0][0] + lam, JtJ[0][1]], [JtJ[1][0], JtJ[1][1] + lam]]
    d = solve2(A, (-Jtr[0], -Jtr[1]))
    x_try = (x[0] + d[0], x[1] + d[1])
    if sum(ri ** 2 for ri in residual_fn(x_try)) < sum(ri ** 2 for ri in r):
        return x_try, lam / 10
    return x, lam * 10


def adapter(fn):
    """Lab loop: carry the damping lambda between iterations."""
    def step(x, problem, state, p):
        lam = state.get("lam", p["lam0"])
        x_new, lam_new = fn(x, lam, problem.residuals, problem.jac)
        return as_vec(x_new), {"lam": float(lam_new)}
    return step


FUNC = lm_step
HELPERS = [solve2]
HEADER = ""

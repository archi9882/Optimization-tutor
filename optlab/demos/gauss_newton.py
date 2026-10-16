"""Gauss-Newton demo for least squares f(x) = sum_i r_i(x)^2."""
from __future__ import annotations

from ..problems import as_vec
from ._linalg import solve2

PSEUDOCODE = """\
問題形式：f(x) = Σ r_i(x)²，J = 殘差的 Jacobian
for k = 0, 1, 2, …:
    r ← r(x),  J ← J(x)
    解 (JᵀJ)·d = −Jᵀr           # 用 JᵀJ 近似 Hessian，不需要二階導數
    x ← x + α·d
"""


def gauss_newton_step(x, r, J, alpha):
    """一步高斯-牛頓：用 JᵀJ 近似 Hessian，解 (JᵀJ)·d = −Jᵀr。"""
    m = len(r)
    JtJ = [[sum(J[k][i] * J[k][j] for k in range(m)) for j in range(2)]
           for i in range(2)]
    Jtr = [sum(J[k][i] * r[k] for k in range(m)) for i in range(2)]
    d = solve2(JtJ, (-Jtr[0], -Jtr[1]))
    if d is None:                       # J 不滿秩，方向不存在
        return x
    return (x[0] + alpha * d[0], x[1] + alpha * d[1])


def adapter(fn):
    """Lab loop: supply residuals and their Jacobian."""
    def step(x, problem, state, p):
        return as_vec(fn(x, problem.residuals(x), problem.jac(x), p["alpha"])), state
    return step


FUNC = gauss_newton_step
HELPERS = [solve2]
HEADER = ""

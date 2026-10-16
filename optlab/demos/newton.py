"""Newton's method demo (optionally damped by a fixed step alpha)."""
from __future__ import annotations

from ..problems import as_vec
from ._linalg import solve2

PSEUDOCODE = """\
for k = 0, 1, 2, …:
    g ← ∇f(x),  H ← ∇²f(x)
    解 H·d = −g                 # 牛頓方向：二次模型的最低點
    x ← x + α·d                 # α = 1 為純牛頓法
"""


def newton_step(x, grad, hess, alpha):
    """一步牛頓法：解 H·d = −g 得到牛頓方向，再走 α 倍（α = 1 為純牛頓）。"""
    d = solve2(hess, (-grad[0], -grad[1]))
    if d is None:                       # Hessian 奇異，牛頓方向不存在
        return x
    return (x[0] + alpha * d[0], x[1] + alpha * d[1])


def adapter(fn):
    """Lab loop: supply the exact gradient and Hessian."""
    def step(x, problem, state, p):
        return as_vec(fn(x, problem.grad(x), problem.hess(x), p["alpha"])), state
    return step


FUNC = newton_step
HELPERS = [solve2]
HEADER = ""

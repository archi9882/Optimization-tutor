"""Gradient descent demo: the code shown on the page and run by the lab."""
from __future__ import annotations

from ..problems import as_vec

PSEUDOCODE = """\
輸入：起點 x₀、學習率 η、步數 K
for k = 0, 1, …, K−1:
    g ← ∇f(x_k)              # 該點的梯度
    x_{k+1} ← x_k − η·g       # 沿負梯度走一步
"""


def gd_step(x, grad, lr):
    """一步梯度下降：沿負梯度方向走 lr 倍的距離。"""
    return tuple(xi - lr * gi for xi, gi in zip(x, grad))


def adapter(fn):
    """Lab loop: call the step once per iteration with the true gradient."""
    def step(x, problem, state, p):
        return as_vec(fn(x, problem.grad(x), p["lr"])), state
    return step


FUNC = gd_step
HEADER = ""

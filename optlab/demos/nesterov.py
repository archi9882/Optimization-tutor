"""Nesterov accelerated gradient demo (look-ahead form)."""
from __future__ import annotations

from ..problems import as_vec

PSEUDOCODE = """\
v ← 0
for k = 0, 1, 2, …:
    y ← x − η·β·v      # 先照慣性往前看
    g ← ∇f(y)          # 在前方的點算梯度
    v ← β·v + g
    x ← x − η·v
"""


def nesterov_step(x, v, grad_fn, lr, beta):
    """一步 Nesterov：在照慣性滑過去的點算梯度，其餘和動量法相同。"""
    look = tuple(xi - lr * beta * vi for xi, vi in zip(x, v))
    g = grad_fn(look)
    v_new = tuple(beta * vi + gi for vi, gi in zip(v, g))
    x_new = tuple(xi - lr * vi for xi, vi in zip(x, v_new))
    return x_new, v_new


def adapter(fn):
    """Lab loop: pass the gradient function so it can look ahead."""
    def step(x, problem, state, p):
        v = state.get("v", (0.0, 0.0))
        x_new, v_new = fn(x, v, problem.grad, p["lr"], p["beta"])
        return as_vec(x_new), {"v": as_vec(v_new)}
    return step


FUNC = nesterov_step
HEADER = ""

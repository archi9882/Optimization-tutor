"""Heavy-ball momentum demo (PyTorch convention)."""
from __future__ import annotations

from ..problems import as_vec

PSEUDOCODE = """\
v ← 0
for k = 0, 1, 2, …:
    g ← ∇f(x)
    v ← β·v + g        # 速度：過去梯度打折後累加
    x ← x − η·v        # 用新的速度移動
"""


def momentum_step(x, v, grad, lr, beta):
    """一步 heavy-ball 動量：先更新速度，再用新的速度移動。"""
    v_new = tuple(beta * vi + gi for vi, gi in zip(v, grad))
    x_new = tuple(xi - lr * vi for xi, vi in zip(x, v_new))
    return x_new, v_new


def adapter(fn):
    """Lab loop: carry the velocity v between iterations."""
    def step(x, problem, state, p):
        v = state.get("v", (0.0, 0.0))
        x_new, v_new = fn(x, v, problem.grad(x), p["lr"], p["beta"])
        return as_vec(x_new), {"v": as_vec(v_new)}
    return step


FUNC = momentum_step
HEADER = ""

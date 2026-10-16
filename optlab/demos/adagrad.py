"""AdaGrad demo: per-coordinate steps scaled by accumulated squared gradients."""
from __future__ import annotations

import math

from ..problems import as_vec

PSEUDOCODE = """\
s ← 0                      # 每個座標一個累加器
for k = 0, 1, 2, …:
    g ← ∇f(x)
    s ← s + g²               # 逐座標累加梯度平方
    x ← x − η·g / (√s + ε)   # 每個座標各自縮放步長
"""


def adagrad_step(x, s, grad, lr, eps=1e-8):
    """一步 AdaGrad：梯度一直很大的座標，步長被壓小。"""
    s_new = tuple(si + gi ** 2 for si, gi in zip(s, grad))
    x_new = tuple(xi - lr * gi / (math.sqrt(si) + eps)
                  for xi, gi, si in zip(x, grad, s_new))
    return x_new, s_new


def adapter(fn):
    """Lab loop: carry the accumulator s between iterations."""
    def step(x, problem, state, p):
        s = state.get("s", (0.0, 0.0))
        x_new, s_new = fn(x, s, problem.grad(x), p["lr"])
        return as_vec(x_new), {"s": as_vec(s_new)}
    return step


FUNC = adagrad_step
HEADER = "import math\n\n\n"

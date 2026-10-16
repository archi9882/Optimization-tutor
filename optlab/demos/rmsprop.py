"""RMSProp demo: exponential moving average of squared gradients."""
from __future__ import annotations

import math

from ..problems import as_vec

PSEUDOCODE = """\
s ← 0
for k = 0, 1, 2, …:
    g ← ∇f(x)
    s ← ρ·s + (1−ρ)·g²       # 移動平均，只記得最近約 1/(1−ρ) 步
    x ← x − η·g / (√s + ε)
"""


def rmsprop_step(x, s, grad, lr, rho, eps=1e-8):
    """一步 RMSProp：和 AdaGrad 相同，只是 s 改成指數移動平均。"""
    s_new = tuple(rho * si + (1 - rho) * gi ** 2 for si, gi in zip(s, grad))
    x_new = tuple(xi - lr * gi / (math.sqrt(si) + eps)
                  for xi, gi, si in zip(x, grad, s_new))
    return x_new, s_new


def adapter(fn):
    """Lab loop: carry the moving average s between iterations."""
    def step(x, problem, state, p):
        s = state.get("s", (0.0, 0.0))
        x_new, s_new = fn(x, s, problem.grad(x), p["lr"], p["rho"])
        return as_vec(x_new), {"s": as_vec(s_new)}
    return step


FUNC = rmsprop_step
HEADER = "import math\n\n\n"

"""Adam demo: momentum + RMSProp scaling + bias correction."""
from __future__ import annotations

import math

from ..problems import as_vec

PSEUDOCODE = """\
m ← 0, s ← 0
for t = 1, 2, 3, …:
    g ← ∇f(x)
    m ← β₁·m + (1−β₁)·g       # 一階動差（動量）
    s ← β₂·s + (1−β₂)·g²      # 二階動差（RMSProp）
    m̂ ← m / (1−β₁ᵗ)           # 偏差修正
    ŝ ← s / (1−β₂ᵗ)
    x ← x − η·m̂ / (√ŝ + ε)
"""


def adam_step(x, m, s, grad, t, lr, beta1, beta2, eps=1e-8):
    """一步 Adam（t 從 1 開始）。回傳未修正的 m、s 供下一步使用。"""
    m_new = tuple(beta1 * mi + (1 - beta1) * gi for mi, gi in zip(m, grad))
    s_new = tuple(beta2 * si + (1 - beta2) * gi ** 2 for si, gi in zip(s, grad))
    m_hat = [mi / (1 - beta1 ** t) for mi in m_new]
    s_hat = [si / (1 - beta2 ** t) for si in s_new]
    x_new = tuple(xi - lr * mh / (math.sqrt(sh) + eps)
                  for xi, mh, sh in zip(x, m_hat, s_hat))
    return x_new, m_new, s_new


def adapter(fn):
    """Lab loop: carry m, s and the step counter t."""
    def step(x, problem, state, p):
        t = state.get("t", 0) + 1
        m = state.get("m", (0.0, 0.0))
        s = state.get("s", (0.0, 0.0))
        x_new, m_new, s_new = fn(x, m, s, problem.grad(x), t, p["lr"],
                                 p["beta1"], p["beta2"])
        return as_vec(x_new), {"t": t, "m": as_vec(m_new), "s": as_vec(s_new)}
    return step


FUNC = adam_step
HEADER = "import math\n\n\n"

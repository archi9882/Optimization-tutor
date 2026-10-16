"""SGD demo: noisy gradient plus a decaying learning rate."""
from __future__ import annotations

import random

from ..problems import as_vec

PSEUDOCODE = """\
輸入：起點 x₀、初始學習率 η₀、衰減係數 c
for k = 0, 1, 2, …:
    抽一小批資料 B_k
    g̃ ← 平均_{i∈B_k} ∇f_i(x_k)   # 有雜訊、但不偏的梯度估計
    η_k ← η₀ / (1 + c·k)        # 學習率隨步數衰減
    x_{k+1} ← x_k − η_k·g̃
"""


def sgd_step(x, grad, lr0, k, decay):
    """一步 SGD：grad 是帶雜訊的梯度估計，學習率隨步數 k 衰減。"""
    lr = lr0 / (1 + decay * k)
    return tuple(xi - lr * gi for xi, gi in zip(x, grad))


def adapter(fn):
    """Lab loop: true gradient + seeded Gaussian noise stands in for a mini-batch."""
    def step(x, problem, state, p):
        rng = state.get("rng") or random.Random(int(p.get("seed", 0)))
        k = state.get("k", 0)
        g = problem.grad(x)
        g = (g[0] + p["sigma"] * rng.gauss(0.0, 1.0),
             g[1] + p["sigma"] * rng.gauss(0.0, 1.0))
        return as_vec(fn(x, g, p["lr"], k, p["decay"])), {"rng": rng, "k": k + 1}
    return step


FUNC = sgd_step
HEADER = ""

"""Exercise: an SGD step with a decaying learning rate."""
from __future__ import annotations

import random

from ..grading import Exercise, StepCase, as_vec

SKELETON = '''def sgd_step(x, grad, lr0, k, decay):
    """一步帶學習率衰減的 SGD。

    x, grad: 長度 2 的 tuple（位置、這一步「帶雜訊」的梯度估計）
    lr0:     初始學習率
    k:       目前是第幾步（從 0 開始）
    decay:   衰減係數 c，學習率 η_k = lr0 / (1 + c·k)
    回傳:    x_new
    """
    # TODO 1: 算出這一步的學習率 η_k
    # TODO 2: 和 GD 一樣沿負梯度走 η_k
    raise NotImplementedError
'''

SOLUTION = '''def sgd_step(x, grad, lr0, k, decay):
    lr = lr0 / (1 + decay * k)
    return (x[0] - lr * grad[0], x[1] - lr * grad[1])
'''


def reference(x, grad, lr0, k, decay):
    """Reference implementation in the learner's signature."""
    lr = lr0 / (1 + decay * k)
    return (x[0] - lr * grad[0], x[1] - lr * grad[1])


def adapter(fn):
    """Feed the learner the same seeded noisy gradient optlab's SGD uses."""
    def step(x, problem, state, p):
        rng = state.get("rng") or random.Random(int(p.get("seed", 0)))
        k = state.get("k", 0)
        g = problem.grad(x)
        g = (g[0] + p["sigma"] * rng.gauss(0.0, 1.0),
             g[1] + p["sigma"] * rng.gauss(0.0, 1.0))
        return as_vec(fn(x, g, p["lr"], k, p["decay"])), {"rng": rng, "k": k + 1}
    return step


EXERCISE = Exercise(
    id="sgd",
    func_name="sgd_step",
    skeleton=SKELETON,
    hints=[
        "更新公式和 GD 完全一樣，差別只在梯度有雜訊，以及學習率要隨步數變小。",
        "虛擬碼：η_k = lr0 / (1 + c·k)；x ← x − η_k · g",
        "k = 0 時 η_0 = lr0；k = 10、c = 0.1 時學習率剛好減半。",
    ],
    solution=SOLUTION,
    cases=[
        StepCase("k = 0 時就是 GD",
                 {"x": (1.0, 2.0), "grad": (0.5, -1.0), "lr0": 0.1, "k": 0,
                  "decay": 0.1},
                 ((0.95, 2.1),)),
        StepCase("k = 10、c = 0.1：學習率減半",
                 {"x": (1.0, 2.0), "grad": (0.5, -1.0), "lr0": 0.1, "k": 10,
                  "decay": 0.1},
                 ((0.975, 2.05),)),
        StepCase("c = 0：固定學習率",
                 {"x": (0.0, 0.0), "grad": (2.0, 4.0), "lr0": 0.5, "k": 99,
                  "decay": 0.0},
                 ((-1.0, -2.0),)),
    ],
    adapter=adapter,
    reference=reference,
    demo={"problem": "quadratic", "problem_params": {"kappa": 10},
          "params": {"lr": 0.15, "sigma": 1.0, "decay": 0.05, "seed": 0},
          "n_iter": 300},
    converge_below=1e-2,
)

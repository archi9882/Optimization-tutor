"""Exercise: one RMSProp step."""
from __future__ import annotations

import math

from ..grading import Exercise, StepCase, as_vec

SKELETON = '''import math

def rmsprop_step(x, s, grad, lr, rho, eps=1e-8):
    """一步 RMSProp。

    x, s, grad: 長度 2 的 tuple（位置、梯度平方的移動平均、該點梯度）
    lr, rho:    學習率 η、衰減係數 ρ
    回傳:       (x_new, s_new)
    每個座標的步長：lr * g_i / (sqrt(s_i) + eps)，s 用「更新後」的值
    """
    # TODO 1: s ← ρ·s + (1−ρ)·g²（逐分量）
    # TODO 2: 每個座標用自己的 sqrt(s_i) 縮放步長
    raise NotImplementedError
'''

SOLUTION = '''import math

def rmsprop_step(x, s, grad, lr, rho, eps=1e-8):
    s_new = (rho * s[0] + (1 - rho) * grad[0] ** 2,
             rho * s[1] + (1 - rho) * grad[1] ** 2)
    x_new = (x[0] - lr * grad[0] / (math.sqrt(s_new[0]) + eps),
             x[1] - lr * grad[1] / (math.sqrt(s_new[1]) + eps))
    return x_new, s_new
'''


def reference(x, s, grad, lr, rho, eps=1e-8):
    """Reference implementation in the learner's signature."""
    s_new = (rho * s[0] + (1 - rho) * grad[0] ** 2,
             rho * s[1] + (1 - rho) * grad[1] ** 2)
    return (x[0] - lr * grad[0] / (math.sqrt(s_new[0]) + eps),
            x[1] - lr * grad[1] / (math.sqrt(s_new[1]) + eps)), s_new


def adapter(fn):
    """Wrap a learner-style rmsprop_step into the shared StepFn signature."""
    def step(x, problem, state, p):
        s = state.get("s", (0.0, 0.0))
        x_new, s_new = fn(x, s, problem.grad(x), p["lr"], p["rho"])
        return as_vec(x_new), {"s": as_vec(s_new)}
    return step


EXERCISE = Exercise(
    id="rmsprop",
    func_name="rmsprop_step",
    skeleton=SKELETON,
    hints=[
        "和 AdaGrad 只差一行：s 不再無限累加，而是「舊的打折、新的補上」。",
        "虛擬碼：s ← ρ·s + (1−ρ)·g²；x ← x − η · g / (√s + ε)",
        "第一個分量：rho * s[0] + (1 - rho) * grad[0] ** 2",
    ],
    solution=SOLUTION,
    cases=[
        StepCase("從 s = 0 開始",
                 {"x": (1.0, 1.0), "s": (0.0, 0.0), "grad": (2.0, -1.0),
                  "lr": 0.01, "rho": 0.75},
                 ((0.98, 1.02), (1.0, 0.25))),
        StepCase("舊的平均被打折",
                 {"x": (0.0, 0.0), "s": (4.0, 4.0), "grad": (0.0, 4.0),
                  "lr": 0.1, "rho": 0.5},
                 ((0.0, -0.4 / math.sqrt(10.0)), (2.0, 10.0))),
        StepCase("ρ = 0：只看這一步的梯度",
                 {"x": (0.0, 0.0), "s": (100.0, 100.0), "grad": (3.0, -4.0),
                  "lr": 0.5, "rho": 0.0},
                 ((-0.5, 0.5), (9.0, 16.0))),
    ],
    adapter=adapter,
    reference=reference,
    demo={"problem": "quadratic", "problem_params": {"kappa": 10},
          "params": {"lr": 0.05, "rho": 0.99}, "n_iter": 300},
)

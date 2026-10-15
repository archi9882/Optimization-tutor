"""Exercise: one AdaGrad step."""
from __future__ import annotations

import math

from ..grading import Exercise, StepCase, as_vec

SKELETON = '''import math

def adagrad_step(x, s, grad, lr, eps=1e-8):
    """一步 AdaGrad。

    x, s, grad: 長度 2 的 tuple（位置、梯度平方累加、該點梯度）
    lr:         學習率 η
    回傳:       (x_new, s_new)
    每個座標的步長：lr * g_i / (sqrt(s_i) + eps)，s 用「更新後」的值
    """
    # TODO 1: s 的每個分量加上 g_i 的平方
    # TODO 2: 每個座標用自己的 sqrt(s_i) 縮放步長
    raise NotImplementedError
'''

SOLUTION = '''import math

def adagrad_step(x, s, grad, lr, eps=1e-8):
    s_new = (s[0] + grad[0] ** 2, s[1] + grad[1] ** 2)
    x_new = (x[0] - lr * grad[0] / (math.sqrt(s_new[0]) + eps),
             x[1] - lr * grad[1] / (math.sqrt(s_new[1]) + eps))
    return x_new, s_new
'''


def reference(x, s, grad, lr, eps=1e-8):
    """Reference implementation in the learner's signature."""
    s_new = (s[0] + grad[0] ** 2, s[1] + grad[1] ** 2)
    return (x[0] - lr * grad[0] / (math.sqrt(s_new[0]) + eps),
            x[1] - lr * grad[1] / (math.sqrt(s_new[1]) + eps)), s_new


def adapter(fn):
    """Wrap a learner-style adagrad_step into the shared StepFn signature."""
    def step(x, problem, state, p):
        s = state.get("s", (0.0, 0.0))
        x_new, s_new = fn(x, s, problem.grad(x), p["lr"])
        return as_vec(x_new), {"s": as_vec(s_new)}
    return step


EXERCISE = Exercise(
    id="adagrad",
    func_name="adagrad_step",
    skeleton=SKELETON,
    hints=[
        "s 記錄每個座標「到目前為止梯度有多大」。梯度一直很大的座標，步長就被壓小。",
        "虛擬碼：s ← s + g²（逐分量）；x ← x − η · g / (√s + ε)",
        "第一個分量：x[0] - lr * grad[0] / (math.sqrt(s_new[0]) + eps)",
    ],
    solution=SOLUTION,
    cases=[
        StepCase("第一步：每個座標都走 η",
                 {"x": (1.0, 1.0), "s": (0.0, 0.0), "grad": (2.0, -0.5),
                  "lr": 0.1},
                 ((0.9, 1.1), (4.0, 0.25))),
        StepCase("已有累積量",
                 {"x": (0.0, 0.0), "s": (9.0, 16.0), "grad": (4.0, 3.0),
                  "lr": 1.0},
                 ((-0.8, -0.6), (25.0, 25.0))),
        StepCase("梯度為 0 不動",
                 {"x": (1.0, 2.0), "s": (1.0, 1.0), "grad": (0.0, 0.0),
                  "lr": 0.5},
                 ((1.0, 2.0), (1.0, 1.0))),
    ],
    adapter=adapter,
    reference=reference,
    demo={"problem": "quadratic", "problem_params": {"kappa": 10},
          "params": {"lr": 0.5}, "n_iter": 300},
    converge_below=1e-6,
)

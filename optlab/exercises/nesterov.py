"""Exercise: one Nesterov step in look-ahead form."""
from __future__ import annotations

from .. import demos
from ..demos import nesterov as DEMO
from ..grading import Exercise, StepCase

SKELETON = '''def nesterov_step(x, v, grad_fn, lr, beta):
    """一步 Nesterov 加速梯度（look-ahead 形式）。

    x, v:    長度 2 的 tuple（位置、速度）
    grad_fn: 函式，grad_fn(p) 回傳點 p 的梯度
    lr, beta: 學習率 η、動量係數 β
    回傳:    (x_new, v_new)
    """
    # TODO 1: 先沿慣性往前看：look = x - lr * beta * v
    # TODO 2: 在 look 這個點算梯度（不是在 x！）
    # TODO 3: 和動量法一樣更新 v 與 x
    raise NotImplementedError
'''


def _bowl(p):
    """Gradient of f = 0.5 (x^2 + y^2)."""
    return (p[0], p[1])


def _stretched(p):
    """Gradient of f = x^2 + 2 y^2."""
    return (2 * p[0], 4 * p[1])


EXERCISE = Exercise(
    id="nesterov",
    func_name="nesterov_step",
    skeleton=SKELETON,
    hints=[
        "和動量法唯一的差別：梯度不是在 x 算，而是在「照慣性滑過去之後」的位置算。",
        "虛擬碼：look ← x − η·β·v；g ← ∇f(look)；v ← β·v + g；x ← x − η·v",
        "look 的第一個分量：x[0] - lr * beta * v[0]，然後呼叫 grad_fn(look)",
    ],
    solution=demos.source("nesterov"),
    blanks=("xi - lr * beta * vi", "grad_fn(look)", "beta * vi + gi"),
    cases=[
        StepCase("v = 0 時與 GD 相同",
                 {"x": (1.0, 2.0), "v": (0.0, 0.0), "grad_fn": _bowl,
                  "lr": 0.1, "beta": 0.9},
                 ((0.9, 1.8), (1.0, 2.0))),
        StepCase("帶著速度先往前看",
                 {"x": (1.0, 1.0), "v": (1.0, -1.0), "grad_fn": _bowl,
                  "lr": 0.5, "beta": 0.5},
                 ((0.375, 0.625), (1.25, 0.75))),
        StepCase("在拉長的碗上",
                 {"x": (1.0, 1.0), "v": (1.0, 1.0), "grad_fn": _stretched,
                  "lr": 0.1, "beta": 0.5},
                 ((0.76, 0.57), (2.4, 4.3))),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "quadratic", "problem_params": {"kappa": 10},
          "params": {"lr": 0.05, "beta": 0.8}, "n_iter": 300},
)

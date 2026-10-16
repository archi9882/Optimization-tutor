"""Exercise: one heavy-ball momentum step (PyTorch convention)."""
from __future__ import annotations

from .. import demos
from ..demos import momentum as DEMO
from ..grading import Exercise, StepCase

SKELETON = '''def momentum_step(x, v, grad, lr, beta):
    """一步 heavy-ball 動量（PyTorch 慣例）。

    x, v, grad: 長度 2 的 tuple（位置、速度、該點梯度）
    lr, beta:   學習率 η、動量係數 β
    回傳:       (x_new, v_new)
    """
    # TODO 1: 更新速度 v（過去速度衰減 + 新梯度）
    # TODO 2: 用「新的」速度更新位置 x
    raise NotImplementedError
'''


EXERCISE = Exercise(
    id="momentum",
    func_name="momentum_step",
    skeleton=SKELETON,
    hints=[
        "速度 v 是過去梯度的指數加權累積：先更新 v，再用新的 v 移動 x。",
        "虛擬碼：v ← β·v + ∇f(x)；x ← x − η·v",
        "第一個分量的速度：beta * v[0] + grad[0]",
    ],
    solution=demos.source("momentum"),
    blanks=("beta * vi + gi", "xi - lr * vi"),
    cases=[
        StepCase("第一步（v = 0）",
                 {"x": (1.0, 2.0), "v": (0.0, 0.0), "grad": (0.5, -1.0),
                  "lr": 0.1, "beta": 0.9},
                 ((0.95, 2.1), (0.5, -1.0))),
        StepCase("帶著既有速度",
                 {"x": (1.0, 1.0), "v": (1.0, -2.0), "grad": (0.2, 0.4),
                  "lr": 0.5, "beta": 0.5},
                 ((0.65, 1.3), (0.7, -0.6))),
        StepCase("β = 0 退化為 GD",
                 {"x": (2.0, -1.0), "v": (5.0, 5.0), "grad": (1.0, 1.0),
                  "lr": 0.1, "beta": 0.0},
                 ((1.9, -1.1), (1.0, 1.0))),
        StepCase("梯度為 0 仍慣性滑行",
                 {"x": (0.0, 0.0), "v": (1.0, 2.0), "grad": (0.0, 0.0),
                  "lr": 0.1, "beta": 0.9},
                 ((-0.09, -0.18), (0.9, 1.8))),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "quadratic", "problem_params": {"kappa": 10},
          "params": {"lr": 0.05, "beta": 0.8}, "n_iter": 300},
)

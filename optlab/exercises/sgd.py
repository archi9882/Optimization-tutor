"""Exercise: an SGD step with a decaying learning rate."""
from __future__ import annotations

from .. import demos
from ..demos import sgd as DEMO
from ..grading import Exercise, StepCase

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


EXERCISE = Exercise(
    id="sgd",
    func_name="sgd_step",
    skeleton=SKELETON,
    hints=[
        "更新公式和 GD 完全一樣，差別只在梯度有雜訊，以及學習率要隨步數變小。",
        "虛擬碼：η_k = lr0 / (1 + c·k)；x ← x − η_k · g",
        "k = 0 時 η_0 = lr0；k = 10、c = 0.1 時學習率剛好減半。",
    ],
    solution=demos.source("sgd"),
    blanks=("lr0 / (1 + decay * k)", "xi - lr * gi"),
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
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "quadratic", "problem_params": {"kappa": 10},
          "params": {"lr": 0.15, "sigma": 1.0, "decay": 0.05, "seed": 0},
          "n_iter": 300},
    converge_below=1e-2,
)

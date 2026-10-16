"""Exercise: one (damped) Newton step."""
from __future__ import annotations

from .. import demos
from ..demos import newton as DEMO
from ..grading import Exercise, StepCase

SKELETON = demos.helpers_source("newton") + '''def newton_step(x, grad, hess, alpha):
    """一步牛頓法。

    x, grad: 長度 2 的 tuple（位置、梯度）
    hess:    2×2 Hessian，((h11, h12), (h21, h22))
    alpha:   步長倍率 α（1 = 純牛頓法）
    回傳:    x_new；Hessian 奇異時原地不動
    可以使用上面的 solve2(A, b)。
    """
    # TODO 1: 解 H·d = −g 得到牛頓方向 d
    # TODO 2: d 為 None 時回傳 x；否則沿 d 走 α 倍
    raise NotImplementedError
'''

EXERCISE = Exercise(
    id="newton",
    func_name="newton_step",
    skeleton=SKELETON,
    hints=[
        "牛頓方向是二次模型的最低點：令模型梯度 g + H·d = 0，所以 H·d = −g。",
        "虛擬碼：d ← solve2(H, −g)；x_new ← x + α·d",
        "右手邊要寫成 tuple：(-grad[0], -grad[1])",
    ],
    solution=demos.source("newton"),
    blanks=("a11 * a22 - a12 * a21", "(-grad[0], -grad[1])", "x[0] + alpha * d[0]"),
    cases=[
        StepCase("對角 Hessian：一步到最低點",
                 {"x": (1.0, 1.0), "grad": (2.0, 4.0),
                  "hess": ((2.0, 0.0), (0.0, 4.0)), "alpha": 1.0},
                 ((0.0, 0.0),)),
        StepCase("α = 0.5 只走一半",
                 {"x": (1.0, 1.0), "grad": (2.0, 4.0),
                  "hess": ((2.0, 0.0), (0.0, 4.0)), "alpha": 0.5},
                 ((0.5, 0.5),)),
        StepCase("有耦合的 Hessian",
                 {"x": (0.0, 0.0), "grad": (3.0, 3.0),
                  "hess": ((2.0, 1.0), (1.0, 2.0)), "alpha": 1.0},
                 ((-1.0, -1.0),)),
        StepCase("Hessian 奇異時不動",
                 {"x": (2.0, 3.0), "grad": (1.0, 1.0),
                  "hess": ((1.0, 1.0), (1.0, 1.0)), "alpha": 1.0},
                 ((2.0, 3.0),)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "rosenbrock", "params": {"alpha": 1.0}, "n_iter": 100},
)

"""Exercise: one Gauss-Newton step for least squares."""
from __future__ import annotations

from .. import demos
from ..demos import gauss_newton as DEMO
from ..grading import Exercise, StepCase

SKELETON = demos.helpers_source("gauss_newton") + '''def gauss_newton_step(x, r, J, alpha):
    """一步高斯-牛頓（f = Σ r_i²）。

    x:     長度 2 的 tuple
    r:     殘差向量，長度 m
    J:     m×2 的 Jacobian，J[k][i] = ∂r_k/∂x_i
    alpha: 步長倍率 α
    回傳:  x_new；JᵀJ 奇異時原地不動
    """
    # TODO 1: 算出 2×2 的 JᵀJ 與長度 2 的 Jᵀr
    # TODO 2: 解 (JᵀJ)·d = −Jᵀr，再沿 d 走 α 倍
    raise NotImplementedError
'''

EXERCISE = Exercise(
    id="gauss_newton",
    func_name="gauss_newton_step",
    skeleton=SKELETON,
    hints=[
        "把殘差線性化：r(x + d) ≈ r + J·d。讓 ‖r + J·d‖² 最小，就得到正規方程 (JᵀJ)·d = −Jᵀr。",
        "(JᵀJ)[i][j] = Σ_k J[k][i]·J[k][j]；(Jᵀr)[i] = Σ_k J[k][i]·r[k]",
        "解完 d 之後和牛頓法一樣：x[0] + alpha * d[0]",
    ],
    solution=demos.source("gauss_newton"),
    blanks=("J[k][i] * J[k][j]", "J[k][i] * r[k]", "(-Jtr[0], -Jtr[1])"),
    cases=[
        StepCase("J = I：直接把殘差歸零",
                 {"x": (0.0, 0.0), "r": (1.0, 2.0),
                  "J": ((1.0, 0.0), (0.0, 1.0)), "alpha": 1.0},
                 ((-1.0, -2.0),)),
        StepCase("三個殘差、α = 0.5",
                 {"x": (1.0, 1.0), "r": (1.0, 1.0, 2.0),
                  "J": ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), "alpha": 0.5},
                 ((0.5, 0.5),)),
        StepCase("J 不滿秩時不動",
                 {"x": (5.0, 5.0), "r": (1.0, 1.0),
                  "J": ((1.0, 1.0), (1.0, 1.0)), "alpha": 1.0},
                 ((5.0, 5.0),)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "rosenbrock", "params": {"alpha": 1.0}, "n_iter": 100},
)

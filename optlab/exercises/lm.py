"""Exercise: one Levenberg-Marquardt step with lambda adaptation."""
from __future__ import annotations

from .. import demos
from ..demos import lm as DEMO
from ..grading import Exercise, StepCase

SKELETON = demos.helpers_source("lm") + '''def lm_step(x, lam, residual_fn, jac_fn):
    """一步 Levenberg-Marquardt。

    x:           長度 2 的 tuple
    lam:         目前的阻尼 λ
    residual_fn: residual_fn(p) 回傳殘差 tuple
    jac_fn:      jac_fn(p) 回傳 m×2 Jacobian
    回傳:        (x_new, lam_new)
    """
    # TODO 1: 和高斯-牛頓一樣算 JᵀJ、Jᵀr，但對角線加上 λ
    # TODO 2: 試走一步；f = Σ r² 下降就接受並 λ/10，否則不動並 λ·10
    raise NotImplementedError
'''


def _lin_r(p):
    return (p[0] - 1, p[1] - 2)


def _lin_J(p):
    return ((1.0, 0.0), (0.0, 1.0))


def _cub_r(p):
    return (p[0] ** 3 - 1, p[1])


def _cub_J(p):
    return ((3 * p[0] ** 2, 0.0), (0.0, 1.0))


EXERCISE = Exercise(
    id="lm",
    func_name="lm_step",
    skeleton=SKELETON,
    hints=[
        "λ 大時 (JᵀJ + λI)⁻¹ ≈ I/λ，等於小步的梯度下降；λ 小時就是高斯-牛頓。",
        "A = [[JtJ[0][0] + lam, JtJ[0][1]], [JtJ[1][0], JtJ[1][1] + lam]]，解 A·d = −Jᵀr",
        "比較 sum(ri ** 2 for ri in residual_fn(x_try)) 和目前的 Σ r²",
    ],
    solution=demos.source("lm"),
    blanks=("JtJ[0][0] + lam", "lam / 10", "lam * 10"),
    cases=[
        StepCase("λ = 1：接受並縮小 λ",
                 {"x": (0.0, 0.0), "lam": 1.0, "residual_fn": _lin_r,
                  "jac_fn": _lin_J},
                 ((0.5, 1.0), 0.1)),
        StepCase("λ 很小：幾乎就是高斯-牛頓",
                 {"x": (0.0, 0.0), "lam": 1e-8, "residual_fn": _lin_r,
                  "jac_fn": _lin_J},
                 ((1.0, 2.0), 1e-9)),
        StepCase("步子太大被拒絕：不動並放大 λ",
                 {"x": (0.1, 0.0), "lam": 1e-6, "residual_fn": _cub_r,
                  "jac_fn": _cub_J},
                 ((0.1, 0.0), 1e-5)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "rosenbrock", "params": {"lam0": 1.0}, "n_iter": 200},
)

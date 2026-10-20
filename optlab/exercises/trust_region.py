"""Exercise: one trust-region (dogleg) step."""
from __future__ import annotations

from .. import demos
from ..demos import trust_region as DEMO
from ..grading import Exercise, StepCase

SKELETON = demos.helpers_source("trust_region") + '''def dogleg_step(x, radius, f, grad_fn, hess_fn):
    """一步信賴域（dogleg）。

    x:      長度 2 的 tuple
    radius: 目前的信賴半徑 Δ
    f, grad_fn, hess_fn: 目標函數、梯度、Hessian
    回傳:   (x_new, radius_new)；被拒絕時 x_new = x
    """
    # TODO 1: 選步 p：牛頓步在半徑內就用它；否則沿 dogleg 折線走到邊界
    #         牛頓步不是下降方向（或 gᵀHg ≤ 0）時只走柯西點：沿 −g 走 min(Δ, ‖g‖³/gᵀHg)
    # TODO 2: ratio = 實際下降 / 模型預測下降
    # TODO 3: ratio < 0.25 → Δ/4；ratio > 0.75 且 p 在邊界 → 2Δ；ratio > 0.1 才接受
    raise NotImplementedError
'''


def _f(p):
    return p[0] ** 2 + 2 * p[1] ** 2


def _g(p):
    return (2 * p[0], 4 * p[1])


def _h(p):
    return ((2.0, 0.0), (0.0, 4.0))


def _sf(p):
    return 2 * p[0] ** 2 - p[1] ** 2


def _sg(p):
    return (4 * p[0], -2 * p[1])


def _sh(p):
    return ((4.0, 0.0), (0.0, -2.0))


def _rf(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


def _rg(p):
    r = p[1] - p[0] ** 2
    return (-2 * (1 - p[0]) - 400 * p[0] * r, 200 * r)


def _rh(p):
    return ((2 - 400 * (p[1] - 3 * p[0] ** 2), -400 * p[0]), (-400 * p[0], 200.0))


EXERCISE = Exercise(
    id="trust_region",
    func_name="dogleg_step",
    skeleton=SKELETON,
    hints=[
        "先決定「能走多遠」（半徑 Δ），再在範圍內找最好的點；線搜尋則是先決定方向再決定距離。",
        "ratio = (f(x) − f(x + p)) / (−(gᵀp + ½ pᵀHp))；分母是二次模型預測的下降量。",
        "半徑規則：ratio < 0.25 → 0.25·Δ；ratio > 0.75 且 ‖p‖ = Δ → min(2Δ, 10)。",
        "柯西點：沿 −g 的模型 m(−t·g/‖g‖) 在 t = ‖g‖³/gᵀHg 最低（gᵀHg > 0 時）；超過 Δ 就停在邊界。",
    ],
    solution=demos.source("trust_region"),
    blanks=("norm(p_newton) <= radius", "-(dot(g, p) + 0.5 * dot(p, matvec(H, p)))",
            "0.25 * radius"),
    cases=[
        StepCase("牛頓步在範圍內：直接走到底",
                 {"x": (0.1, 0.1), "radius": 1.0, "f": _f, "grad_fn": _g, "hess_fn": _h},
                 ((0.0, 0.0), 1.0)),
        StepCase("範圍太小：沿最陡方向走到邊界，半徑加倍",
                 {"x": (1.0, 1.0), "radius": 0.1, "f": _f, "grad_fn": _g, "hess_fn": _h},
                 ((0.9552786404500042, 0.9105572809000084), 0.2)),
        StepCase("鞍點型 Hessian、牛頓步往上：走柯西點（在圓內），半徑不變",
                 {"x": (1.0, 2.0), "radius": 8.0, "f": _sf, "grad_fn": _sg, "hess_fn": _sh},
                 ((-3.0, 6.0), 8.0)),
        StepCase("模型預測失準：拒絕並縮小半徑",
                 {"x": (0.0, 0.0), "radius": 1.0, "f": _rf, "grad_fn": _rg, "hess_fn": _rh},
                 ((0.0, 0.0), 0.25)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "rosenbrock", "params": {"radius0": 0.5}, "n_iter": 200},
)

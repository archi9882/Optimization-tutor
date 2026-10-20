"""Exercise: one SQP step (QP subproblem by working sets + l1 merit line search)."""
from __future__ import annotations

import inspect

from .. import demos
from ..demos import sqp as DEMO
from ..grading import Exercise, StepCase
from ..problems import Constraint, linear_constraint
from ._concheck import make_check

_GIVEN = [DEMO.dot, DEMO.matvec, DEMO.solve_lin, DEMO.hessian_lagrangian, DEMO.damped_bfgs,
          DEMO.grad_lagrangian]
SKELETON = DEMO.HEADER + "".join(inspect.getsource(h) + "\n\n" for h in _GIVEN) + '''def qp_subproblem(B, gf, lin):
    """min ½dᵀBd + gfᵀd，限制 aᵀd + c ≤ 0（等式 = 0），lin = [(kind, a, c), ...]。

    逐一試工作集 W（等式一定在內，不等式取 0、1 或 2 個，總數不超過 2），
    用 solve_lin 解 KKT 方程組 [[B, A_Wᵀ], [A_W, 0]]·[d; λ_W] = [−gf; −c_W]。
    接受條件：不在 W 裡的不等式滿足 aᵀd + c ≤ 1e-9，W 裡的不等式乘數 ≥ −1e-12。
    回傳 (d, λ)（λ 是長度 len(lin) 的 list，不在 W 的填 0）；都不行時回傳 None。
    """
    raise NotImplementedError


def merit(x, mu, f, cons):
    """ℓ1 價值函數 φ(x) = f(x) + μ Σ 違反量（不等式 max(0, g)，等式 |h|）。"""
    raise NotImplementedError


def sqp_step(x, B, mu, f, grad, cons):
    """SQP 的一步。

    x:     長度 2 的 tuple；B：QP 用的 2×2 矩陣（正定）
    mu:    目前的價值函數懲罰係數 μ（只增不減）
    f, grad: 目標函數與梯度；cons：限制的 list（c.kind、c.g、c.grad）
    回傳:  (x_new, λ, μ, t, d)
           QP 子問題無解時回傳 None；線搜尋 40 次都失敗時回傳 (x, λ, μ, 0.0, d)
    """
    # TODO 1: lin = [(c.kind, ∇g_i(x), g_i(x)) ...]，用 qp_subproblem 解出 (d, λ)
    # TODO 2: μ ← max(μ, 2·max|λ_i|)
    # TODO 3: 從 t = 1 開始回溯：merit(x + t·d) ≤ merit(x) + 1e-4·t·slope 就接受，
    #         slope = ∇f·d − μ·(x 的總違反量)
    raise NotImplementedError
'''


def _f(p):
    return (p[0] - 2) ** 2 + (p[1] - 2) ** 2


def _g(p):
    return (2 * (p[0] - 2), 2 * (p[1] - 2))


def _f0(p):
    return p[0] ** 2 + p[1] ** 2


def _g0(p):
    return (2 * p[0], 2 * p[1])


_SUM = linear_constraint((1.0, 1.0), 2.0, "x₁+x₂ ≤ 2")
_SUM_EQ = linear_constraint((1.0, 1.0), 2.0, "x₁+x₂ = 2", kind="eq")
_X1 = linear_constraint((1.0, 0.0), 0.5, "x₁ ≤ 0.5")
_X2 = linear_constraint((0.0, 1.0), 0.5, "x₂ ≤ 0.5")
_CIRCLE = Constraint("eq", lambda p: p[0] ** 2 + p[1] ** 2 - 1, lambda p: (2 * p[0], 2 * p[1]),
                     lambda p: ((2.0, 0.0), (0.0, 2.0)), "x₁²+x₂² = 1")
_I2 = [[2.0, 0.0], [0.0, 2.0]]

EXERCISE = Exercise(
    id="sqp",
    func_name="sqp_step",
    required=("qp_subproblem", "merit"),
    skeleton=SKELETON,
    hints=[
        "線性化的限制是 g_i(x) + ∇g_i(x)ᵀd ≤ 0：lin 的每一項放 (kind, ∇g_i(x), g_i(x))，"
        "qp_subproblem 會把 c = g_i(x) 移到右邊（−c_W）。",
        "qp_subproblem 的工作集解要通過兩個檢查：不在工作集裡的線性化限制要滿足，"
        "工作集裡的不等式乘數要 ≥ 0。少了任何一個，就可能把錯的工作集當成答案。",
        "μ 要大於所有 |λ_i|，ℓ1 價值函數沿 d 的方向導數 ∇f·d − μ·違反量 才保證是負的。",
        "回溯從 t = 1 開始：接近解時希望整步都被接受，才有二次收斂。",
    ],
    solution=demos.source("sqp"),
    blanks=("[-lin[i][2] for i in W]", "dot(lin[i][1], d) + lin[i][2]", "lam[i] >= -1e-12",
            "2 * max([abs(v) for v in lam] + [0.0])", "max(0.0, c.g(x))", "t * d[0]"),
    cases=[
        StepCase("預算限制起作用：B = 2I 就是 f 的 Hessian，一步到 (1, 1)，λ = 2、μ = 4",
                 {"x": (0.0, 0.0), "B": _I2, "mu": 0.0, "f": _f, "grad": _g, "cons": [_SUM]},
                 ((1.0, 1.0), [2.0], 4.0, 1.0, (1.0, 1.0))),
        StepCase("兩個限制同時起作用（角落）：d = (0.5, 0.5)，λ = (3, 3)",
                 {"x": (0.0, 0.0), "B": _I2, "mu": 0.0, "f": _f, "grad": _g,
                  "cons": [_X1, _X2]},
                 ((0.5, 0.5), [3.0, 3.0], 6.0, 1.0, (0.5, 0.5))),
        StepCase("等式限制：乘數可以是負的（λ = −2）",
                 {"x": (0.0, 0.0), "B": _I2, "mu": 0.0, "f": _f0, "grad": _g0,
                  "cons": [_SUM_EQ]},
                 ((1.0, 1.0), [-2.0], 4.0, 1.0, (1.0, 1.0))),
        StepCase("沒有限制、B = I/4 太小：d = (16, 16) 太長，回溯到 t = 1/8",
                 {"x": (0.0, 0.0), "B": [[0.25, 0.0], [0.0, 0.25]], "mu": 0.0, "f": _f,
                  "grad": _g, "cons": []},
                 ((2.0, 2.0), [], 0.0, 0.125, (16.0, 16.0))),
        StepCase("圓心：∇h = 0 而 h = −1，線性化的限制 −1 = 0 無解，回傳 None",
                 {"x": (0.0, 0.0), "B": _I2, "mu": 0.0, "f": _f, "grad": _g,
                  "cons": [_CIRCLE]},
                 (None,)),
        StepCase("價值函數：f(3, 1) = 2，違反 x₁+x₂ ≤ 2 共 2，μ = 2 → φ = 6",
                 {"x": (3.0, 1.0), "mu": 2.0, "f": _f, "cons": [_SUM]}, (6.0,), func="merit"),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "fert_disk", "params": {}, "n_iter": 50},
    check=make_check(DEMO, "sqp_step", [
        {"problem": "fert_disk", "params": {}, "n_iter": 50, "dist": 1e-10, "viol": 1e-10,
         "goal": "土壤圓盤：幾步內收斂到 x*"},
        {"problem": "reservoir", "params": {}, "n_iter": 50, "dist": 1e-10, "viol": 1e-10,
         "goal": "兩座水庫（箱型）：找出起作用的限制 x₂ ≤ 1"},
        {"problem": "rosen_disk", "params": {}, "n_iter": 50, "dist": 1e-10, "viol": 1e-10,
         "goal": "Rosenbrock + 圓盤：非凸、需要 Hessian 修正"},
        {"problem": "maratos", "params": {}, "n_iter": 50, "dist": 1e-10, "viol": 1e-10,
         "goal": "圓上等式：線搜尋會截短步長（Maratos 效應）"}]),
    loop_note="上方動畫每一步呼叫一次 sqp_step()：外層迴圈在呼叫前算好 B（Lagrangian 的 Hessian，"
              "用上一步的 λ，不正定時加 τI；或阻尼 BFGS），並把回傳的 λ、μ 留到下一步。"
              "‖d‖ < 10⁻¹⁰ 時停止；回傳 None（QP 無解）或 t = 0（線搜尋失敗）時也停止。"
              "細線段是 QP 給的整步 x → x + d，線搜尋接受的點在這條線段上。",
)

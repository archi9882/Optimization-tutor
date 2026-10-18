"""Exercise: one gradient step on the quadratic penalty function."""
from __future__ import annotations

from .. import demos
from ..demos import penalty as DEMO
from ..grading import Exercise, StepCase
from ..problems import linear_constraint
from ._concheck import make_check

SKELETON = demos.helpers_source("penalty") + '''def penalty_step(x, rho, f, grad, cons):
    """對 P_ρ(x) = f(x) + (ρ/2) Σ v_i(x)² 做一步梯度下降。

    x:     長度 2 的 tuple
    rho:   懲罰係數 ρ
    f, grad: 目標函數與梯度
    cons:  限制的 list；c.kind 是 "ineq"（g ≤ 0）或 "eq"（h = 0），c.g(x)、c.grad(x)
    回傳:  (x_new, ‖∇P_ρ(x)‖)；步長用上面的 backtracking()
    """
    # TODO 1: ∇P_ρ = ∇f + Σ ρ·v_i·∇g_i（v_i 用 violation(c, x)）
    # TODO 2: 沿 −∇P_ρ 用 backtracking(P, x, P(x), g, d) 決定步長
    raise NotImplementedError
'''


def _f(p):
    return (p[0] - 2) ** 2 + (p[1] - 2) ** 2


def _g(p):
    return (2 * (p[0] - 2), 2 * (p[1] - 2))


_LE = linear_constraint((1.0, 1.0), 2.0, "x₁+x₂ ≤ 2")
_EQ = linear_constraint((1.0, 1.0), 2.0, "x₁+x₂ = 2", kind="eq")

EXERCISE = Exercise(
    id="penalty",
    func_name="penalty_step",
    skeleton=SKELETON,
    hints=[
        "懲罰項只在違反時出現：不等式 g ≤ 0 的違反量是 max(0, g)，等式 h = 0 的違反量就是 h（正負都算違反）。",
        "(ρ/2)·v² 對 x 的梯度是 ρ·v·∇g（連鎖律）；把每個限制的這一項加到 ∇f 上。",
        "回溯線搜尋檢查的是 P_ρ 有沒有下降，所以傳給 backtracking 的函數要是 P_ρ，不是 f。",
    ],
    solution=demos.source("penalty"),
    blanks=("max(0.0, v)", "violation(c, x) ** 2", "rho * v * dg[0]"),
    cases=[
        StepCase("可行點：沒有懲罰，就是一步回溯 GD（t = 1/4）",
                 {"x": (0.0, 0.0), "rho": 10.0, "f": _f, "grad": _g, "cons": [_LE]},
                 ((1.0, 1.0), 5.656854249492381)),
        StepCase("違反 x₁+x₂ ≤ 2：∇f = 0，懲罰把點往回拉",
                 {"x": (2.0, 2.0), "rho": 1.0, "f": _f, "grad": _g, "cons": [_LE]},
                 ((1.5, 1.5), 2.8284271247461903)),
        StepCase("等式 x₁+x₂ = 2：在線的另一側也要懲罰",
                 {"x": (0.0, 0.0), "rho": 1.0, "f": _f, "grad": _g, "cons": [_EQ]},
                 ((1.5, 1.5), 8.48528137423857)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "fert_budget", "params": {"rho0": 1.0, "gamma": 10.0, "inner": 100},
          "n_iter": 400},
    check=make_check(DEMO, "penalty_step", [
        {"problem": "fert_budget", "params": {"rho0": 1.0, "gamma": 10.0, "inner": 100},
         "n_iter": 400, "dist": 1e-2, "viol": 1e-3,
         "goal": "施肥預算：ρ 放大到 10⁴ 後接近 x* = (1.5, 1.5)"}]),
    loop_note="上方動畫每一步呼叫一次 penalty_step()；‖∇P_ρ‖ < 10⁻³ 或內圈用完步數上限時，"
              "外層迴圈把 ρ 乘上 γ（實心大點標出每個外圈的終點），ρ 達到 10⁴ 的那一圈結束後停止。",
)

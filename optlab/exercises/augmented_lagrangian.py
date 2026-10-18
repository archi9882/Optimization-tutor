"""Exercise: one inner step / multiplier update of the augmented Lagrangian method."""
from __future__ import annotations

from .. import demos
from ..demos import augmented_lagrangian as DEMO
from ..grading import Exercise, StepCase
from ..problems import linear_constraint
from ._concheck import make_check

SKELETON = demos.helpers_source("augmented_lagrangian") + '''def al_step(x, lam, rho, f, grad, cons, tol):
    """增廣 Lagrangian 法的一步。

    x:     長度 2 的 tuple；lam：乘數的 list（每個限制一個）
    rho:   懲罰係數 ρ（固定）
    f, grad: 目標函數與梯度；cons：限制的 list（c.kind、c.g、c.grad）
    tol:   ‖∇L_A‖ < tol 時視為內圈收斂
    回傳:  (x_new, lam_new, ‖∇L_A(x)‖)
           內圈未收斂：對 L_A 做一步梯度下降，lam 不變
           內圈已收斂：x 不動，lam_i ← shifted(c, lam_i, rho, x)
    """
    # TODO 1: ∇L_A = ∇f + Σ s_i ∇g_i，其中 s_i = shifted(c, lam_i, rho, x)
    # TODO 2: ‖∇L_A‖ < tol → 回傳 (x, 新乘數, ‖∇L_A‖)
    # TODO 3: 否則用 backtracking 對 augmented(·, lam, rho, f, cons) 走一步
    raise NotImplementedError
'''


def _f(p):
    return (p[0] - 2) ** 2 + (p[1] - 2) ** 2


def _g(p):
    return (2 * (p[0] - 2), 2 * (p[1] - 2))


_LE = linear_constraint((1.0, 1.0), 2.0, "x₁+x₂ ≤ 2")
_EQ = linear_constraint((1.0, 1.0), 2.0, "x₁+x₂ = 2", kind="eq")

EXERCISE = Exercise(
    id="augmented_lagrangian",
    func_name="al_step",
    skeleton=SKELETON,
    hints=[
        "s_i = λ_i + ρ·g_i(x) 是「更新後的乘數」：∇L_A = ∇f + Σ s_i ∇g_i，和 Lagrangian 的梯度長得一樣。",
        "不等式限制的 s_i 要截在 0 以上：限制沒被碰到時乘數應該是 0（互補鬆弛）。",
        "L_A = f + Σ (s_i² − λ_i²)/(2ρ)。等式限制展開後就是 f + λh + (ρ/2)h²。",
    ],
    solution=demos.source("augmented_lagrangian"),
    blanks=("lam_i + rho * c.g(x)", "shifted(c, li, rho, x) ** 2 - li ** 2", "s * dg[0]",
            "[shifted(c, li, rho, x) for c, li in zip(cons, lam)]"),
    cases=[
        StepCase("內圈一步：λ = 0 時就像懲罰法，回溯到 t = 1/4",
                 {"x": (2.0, 2.0), "lam": [0.0], "rho": 1.0, "f": _f, "grad": _g,
                  "cons": [_LE], "tol": 1e-6},
                 ((1.5, 1.5), [0.0], 2.8284271247461903)),
        StepCase("內圈已收斂：λ ← λ + ρ·g = 1 + 2·1 = 3",
                 {"x": (1.5, 1.5), "lam": [1.0], "rho": 2.0, "f": _f, "grad": _g,
                  "cons": [_LE], "tol": 10.0},
                 ((1.5, 1.5), [3.0], 2.8284271247461903)),
        StepCase("限制沒被碰到：λ + ρ·g = 2 − 2 = 0，乘數歸零",
                 {"x": (0.0, 0.0), "lam": [2.0], "rho": 1.0, "f": _f, "grad": _g,
                  "cons": [_LE], "tol": 100.0},
                 ((0.0, 0.0), [0.0], 5.656854249492381)),
        StepCase("等式限制：s = λ + ρh 可以是負的",
                 {"x": (0.0, 0.0), "lam": [1.0], "rho": 1.0, "f": _f, "grad": _g,
                  "cons": [_EQ], "tol": 1e-6},
                 ((1.25, 1.25), [1.0], 7.0710678118654755)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "fert_budget", "params": {"rho": 10.0, "inner": 100}, "n_iter": 400},
    check=make_check(DEMO, "al_step", [
        {"problem": "fert_budget", "params": {"rho": 10.0, "inner": 100}, "n_iter": 400,
         "dist": 1e-5, "viol": 1e-5,
         "goal": "施肥預算：ρ = 10 固定，收斂到 x* = (1.5, 1.5)"},
        {"problem": "fert_eq", "params": {"rho": 10.0, "inner": 100}, "n_iter": 400,
         "dist": 1e-5, "viol": 1e-5,
         "goal": "等式版本 x₁ + x₂ = 3：同樣收斂"}]),
    loop_note="上方動畫每一步呼叫一次 al_step()，λ 從 0 開始、ρ 固定；內圈超過步數上限時，"
              "外層迴圈傳入 tol = ∞ 強制更新乘數（實心大點標出每次更新）。"
              "內圈收斂後的更新讓 λ 的變化小於 10⁻⁶ 時停止。",
)

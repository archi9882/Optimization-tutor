"""Exercise: one Newton step on the log-barrier function (nonlinear constraints)."""
from __future__ import annotations

from .. import demos
from ..demos import barrier_nl as DEMO
from ..grading import Exercise, StepCase
from ..problems import disk_constraint, linear_constraint
from ._concheck import make_check

SKELETON = demos.helpers_source("barrier_nl") + '''def barrier_step(x, mu, f, grad, hess, cons):
    """對 B_μ(x) = f(x) − μ Σ log(−g_i(x)) 做一步牛頓法。

    x:     長度 2 的 tuple，必須嚴格可行（每個 c.g(x) < 0）
    mu:    障礙參數 μ
    f, grad, hess: 目標函數、梯度、Hessian
    cons:  不等式限制的 list；c.g(x)、c.grad(x)、c.hess(x)
    回傳:  (x_new, ‖∇B_μ(x)‖)；x_new 仍嚴格可行
    """
    # TODO 1: ∇B = ∇f − μ Σ ∇g_i / g_i
    # TODO 2: ∇²B = ∇²f + μ Σ (∇g_i ∇g_iᵀ / g_i² − ∇²g_i / g_i)
    # TODO 3: 解 ∇²B d = −∇B（不是下降方向就改用 −∇B），再用 backtracking 搭配 barrier_value
    raise NotImplementedError
'''


def _f(p):
    return (p[0] - 2) ** 2 + (p[1] - 2) ** 2


def _g(p):
    return (2 * (p[0] - 2), 2 * (p[1] - 2))


def _h(p):
    return ((2.0, 0.0), (0.0, 2.0))


def _f1(p):
    return (p[0] - 2) ** 2 + p[1] ** 2


def _g1(p):
    return (2 * (p[0] - 2), 2 * p[1])


_BOX = [linear_constraint((1.0, 0.0), 1.0, "x₁ ≤ 1"), linear_constraint((0.0, 1.0), 1.0, "x₂ ≤ 1")]
_DISK = [disk_constraint((0.0, 0.0), 1.0, "單位圓盤")]

EXERCISE = Exercise(
    id="barrier_nl",
    func_name="barrier_step",
    skeleton=SKELETON,
    hints=[
        "−log(−g) 在 g → 0⁻ 時趨近 +∞：它是一道牆，μ 越小牆越薄。",
        "d/dx[−log(−g(x))] = −∇g / g。因為 g < 0，這一項指向遠離邊界的方向。",
        "二階導數有兩部分：∇g∇gᵀ/g²（靠近邊界時很大）和 −∇²g/g（限制本身彎曲時才有，例如圓盤 ∇²g = 2I）。",
    ],
    solution=demos.source("barrier_nl"),
    blanks=("c.g(x) >= 0", "-mu * dg[j] / gi", "dg[j] * dg[k] / gi ** 2 - Hc[j][k] / gi"),
    cases=[
        StepCase("μ = 1：牛頓步會撞牆，回溯兩次停在 (0.5, 0.5)",
                 {"x": (0.0, 0.0), "mu": 1.0, "f": _f, "grad": _g, "hess": _h, "cons": _BOX},
                 ((0.5, 0.5), 4.242640687119285)),
        StepCase("μ = 0.1：牆變薄，可以更靠近邊界",
                 {"x": (0.5, 0.5), "mu": 0.1, "f": _f, "grad": _g, "hess": _h, "cons": _BOX},
                 ((0.7916666666666667, 0.7916666666666667), 3.959797974644666)),
        StepCase("彎曲的限制：圓盤的 ∇²g = 2I 也進入 Hessian",
                 {"x": (0.0, 0.0), "mu": 0.5, "f": _f1, "grad": _g1, "hess": _h, "cons": _DISK},
                 ((2 / 3, 0.0), 4.0)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "fert_budget", "params": {"mu0": 1.0, "sigma": 10.0, "inner": 20},
          "n_iter": 200},
    check=make_check(DEMO, "barrier_step", [
        {"problem": "fert_budget", "params": {"mu0": 1.0, "sigma": 10.0, "inner": 20},
         "n_iter": 200, "dist": 1e-6, "viol": 0.0, "path": True,
         "goal": "施肥預算：μ 縮到 10⁻⁶ 以下，從內部逼近 x* = (1.5, 1.5)"},
        {"problem": "fert_disk", "params": {"mu0": 1.0, "sigma": 10.0, "inner": 20},
         "n_iter": 200, "dist": 1e-6, "viol": 0.0, "path": True,
         "goal": "土壤養分圓盤：彎曲的限制也要收斂"}]),
    loop_note="上方動畫每一步呼叫一次 barrier_step()；‖∇B_μ‖ < 10⁻⁶ 或內圈用完步數上限時，"
              "外層迴圈把 μ 除以 σ（實心大點標出每個外圈的終點），μ ≤ 10⁻⁶ 的那一圈結束後停止。"
              "起點不是嚴格可行時，外層迴圈直接停止。",
)

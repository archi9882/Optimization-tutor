"""Exercise: one ADMM step (x-update, prox z-update, scaled dual update)."""
from __future__ import annotations

import inspect

from .. import demos
from ..demos import admm as DEMO
from ..grading import Exercise, StepCase
from ..sets import project_polygon
from ._concheck import make_check

SKELETON = DEMO.HEADER + inspect.getsource(DEMO.solve2) + '''

def soft_threshold(v, k):
    """λ‖z‖₁ 的 prox：每個座標往 0 收 k，越過 0 就停在 0；回傳 tuple。"""
    raise NotImplementedError


def x_update(v, rho, grad, hess, x):
    """argmin f(x) + (ρ/2)‖x − v‖²：從 x 出發做牛頓法（最多 20 步，步長 < 1e-12 就停）。
    牛頓方程組：(∇²f + ρI)·d = −(∇f(x) + ρ(x − v))，可以用 solve2。"""
    raise NotImplementedError


def admm_step(x, z, u, rho, grad, hess, project, l1):
    """ADMM 的一步（縮放形式），問題 min f(x) + g(z) s.t. x − z = 0。

    x, z, u: 長度 2 的 tuple（u 是縮放的對偶變數，乘數 = ρ·u）
    rho:     懲罰係數 ρ；grad, hess：f 的梯度與 Hessian
    project: 可行集合的投影（g 是指示函數）；None 時 g(z) = l1·‖z‖₁
    回傳:    (x, z, u, r, s)，r = ‖x − z‖，s = ρ‖z − z_舊‖
    """
    # TODO 1: x ← x_update(z − u, ρ, grad, hess, x)
    # TODO 2: z ← project(x + u)，或 soft_threshold(x + u, l1/ρ)
    # TODO 3: u ← u + x − z，再算兩個殘差
    raise NotImplementedError
'''


def _g(p):
    return (2 * (p[0] - 2), 2 * (p[1] - 2))


def _h(p):
    return ((2.0, 0.0), (0.0, 2.0))


_BOX = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _box(y):
    return project_polygon(y, _BOX)


EXERCISE = Exercise(
    id="admm",
    func_name="admm_step",
    required=("soft_threshold", "x_update"),
    skeleton=SKELETON,
    hints=[
        "x 的子問題只看 f 和懲罰項 (ρ/2)‖x − (z − u)‖²：中心是 z − u，不是 z。",
        "z 的子問題只看 g：指示函數的 prox 是投影，λ‖z‖₁ 的 prox 是門檻為 λ/ρ 的軟門檻。",
        "u 累積每一步的 x − z（原始殘差），就是增廣 Lagrangian 的乘數更新 λ ← λ + ρ(x − z) 除以 ρ。",
        "對偶殘差看 z 動了多少：s = ρ‖z − z_舊‖。z 不再動、x 又等於 z，才是 KKT 點。",
    ],
    solution=demos.source("admm"),
    blanks=("max(abs(vi) - k, 0.0)", "rho * (x[0] - v[0])", "(z[0] - u[0], z[1] - u[1])",
            "l1 / rho", "u[0] + x[0] - z[0]", "rho * math.hypot(z[0] - z_old[0], z[1] - z_old[1])"),
    cases=[
        StepCase("箱型 [0, 1]²，從原點出發：x = (1, 1) 剛好在箱內，u 不變",
                 {"x": (0.0, 0.0), "z": (0.0, 0.0), "u": (0.0, 0.0), "rho": 2.0,
                  "grad": _g, "hess": _h, "project": _box, "l1": 0.0},
                 ((1.0, 1.0), (1.0, 1.0), (0.0, 0.0), 0.0, 2.8284271247461903)),
        StepCase("下一步：x 想到 (1.5, 1.5)，z 被投影回 (1, 1)，u 記下差距 0.5",
                 {"x": (1.0, 1.0), "z": (1.0, 1.0), "u": (0.0, 0.0), "rho": 2.0,
                  "grad": _g, "hess": _h, "project": _box, "l1": 0.0},
                 ((1.5, 1.5), (1.0, 1.0), (0.5, 0.5), 0.7071067811865476, 0.0)),
        StepCase("ℓ1（l1 = 1、ρ = 1）：v = (4/3, 0)，軟門檻後 z₂ 正好是 0",
                 {"x": (0.0, 0.0), "z": (0.0, 0.0), "u": (0.0, -2.0), "rho": 1.0,
                  "grad": _g, "hess": _h, "project": None, "l1": 1.0},
                 ((4 / 3, 2.0), (1 / 3, 0.0), (1.0, 0.0), 2.23606797749979, 1 / 3)),
        StepCase("軟門檻：(2.5, −1.5) 每個座標往 0 收 1",
                 {"v": (2.5, -1.5), "k": 1.0}, ((1.5, -0.5),), func="soft_threshold"),
        StepCase("軟門檻：|v_i| ≤ k 的座標變成 0",
                 {"v": (0.4, -3.0), "k": 1.0}, ((0.0, -2.0),), func="soft_threshold"),
        StepCase("x 的子問題：f 是二次函數，牛頓法從 (5, 5) 一步到 (1, 1)",
                 {"v": (0.0, 0.0), "rho": 2.0, "grad": _g, "hess": _h, "x": (5.0, 5.0)},
                 ((1.0, 1.0),), func="x_update"),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "fert_lasso", "params": {"rho": 1.0}, "n_iter": 150},
    check=make_check(DEMO, "admm_step", [
        {"problem": "fert_lasso", "params": {"rho": 1.0}, "n_iter": 150, "dist": 1e-8,
         "viol": 1e-8, "goal": "施肥稀疏（ℓ1）：收斂到 x* = (5/3, 0)"},
        {"problem": "fert_budget", "params": {"rho": 1.0}, "n_iter": 150, "dist": 1e-8,
         "viol": 1e-8, "goal": "施肥預算：投影到三角形，收斂到 x* = (1.5, 1.5)"},
        {"problem": "fert_disk", "params": {"rho": 3.0}, "n_iter": 150, "dist": 1e-8,
         "viol": 1e-8, "goal": "土壤圓盤：投影到圓盤"}]),
    loop_note="上方動畫每一步呼叫一次 admm_step()：z 從起點 x₀ 開始、u 從 0 開始，"
              "外層迴圈把回傳的 z、u 留到下一步，r 與 s 都小於 10⁻⁹ 時停止。"
              "實線是 x，同色虛線與空心方塊是 z。",
)

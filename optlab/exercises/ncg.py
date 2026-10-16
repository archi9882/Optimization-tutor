"""Exercise: one nonlinear conjugate-gradient step (Polak-Ribiere+)."""
from __future__ import annotations

from .. import demos
from ..demos import ncg as DEMO
from ..grading import Exercise, StepCase

SKELETON = demos.helpers_source("ncg") + '''def ncg_step(x, d_prev, g_prev, f, grad_fn):
    """一步非線性共軛梯度（Polak–Ribière+）。

    x:            長度 2 的 tuple
    d_prev, g_prev: 上一步的方向與梯度（第一步為 None）
    f, grad_fn:   目標函數與梯度函數
    回傳:         (x_new, d, g)：新位置、這一步用的方向、這一步的梯度
    可以使用上面的 dot 與 secant_line_search。
    """
    # TODO 1: 第一步 d = −g；之後 β = max(0, gᵀ(g − g_prev) / g_prevᵀg_prev)，d = −g + β·d_prev
    # TODO 2: d 不是下降方向（gᵀd ≥ 0）時改回 −g
    # TODO 3: 用 secant_line_search 沿 d 移動
    raise NotImplementedError
'''


def _f(p):
    return p[0] ** 2 + 2 * p[1] ** 2


def _g(p):
    return (2 * p[0], 4 * p[1])


EXERCISE = Exercise(
    id="ncg",
    func_name="ncg_step",
    skeleton=SKELETON,
    hints=[
        "共軛方向的意思是 dₖᵀ H dⱼ = 0：沿新方向前進時，不會破壞先前方向上已經做好的最小化。",
        "Polak–Ribière+：β = max(0, gᵀ(g − g_prev) / g_prevᵀg_prev)；取 max 是為了必要時自動重新開始。",
        "d = (-g[0] + beta * d_prev[0], -g[1] + beta * d_prev[1])",
    ],
    solution=demos.source("ncg"),
    blanks=("dot(g, y) / dot(g_prev, g_prev)", "-g[0] + beta * d_prev[0]", "-slope / curv"),
    cases=[
        StepCase("第一步：沿負梯度做精確線搜尋",
                 {"x": (1.0, 1.0), "d_prev": None, "g_prev": None, "f": _f, "grad_fn": _g},
                 ((4 / 9, -1 / 9), (-2.0, -4.0), (2.0, 4.0))),
        StepCase("第二步：二次函數上兩步到底",
                 {"x": (4 / 9, -1 / 9), "d_prev": (-2.0, -4.0), "g_prev": (2.0, 4.0),
                  "f": _f, "grad_fn": _g},
                 ((0.0, 0.0), (-0.9876543209876543, 0.24691358024691357),
                  (8 / 9, -4 / 9))),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "rosenbrock", "params": {}, "n_iter": 200},
)

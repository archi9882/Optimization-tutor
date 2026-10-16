"""Exercise: one BFGS step."""
from __future__ import annotations

from .. import demos
from ..demos import bfgs as DEMO
from ..grading import Exercise, StepCase

SKELETON = demos.helpers_source("bfgs") + '''def bfgs_step(x, H, f, grad_fn):
    """一步 BFGS。

    x:       長度 2 的 tuple
    H:       2×2 list，Hessian 反矩陣的近似（第一步為單位矩陣）
    f, grad_fn: 目標函數與梯度函數
    回傳:    (x_new, H_new)
    可以使用上面的 dot 與 backtracking。
    """
    # TODO 1: 方向 d = −H·g，沿 d 做 backtracking 線搜尋
    # TODO 2: s = x_new − x，y = g_new − g；sᵀy ≤ 1e-12 時不更新 H
    # TODO 3: ρ = 1/(sᵀy)，H_new = (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
    raise NotImplementedError
'''


def _f(p):
    return p[0] ** 2 + 2 * p[1] ** 2


def _g(p):
    return (2 * p[0], 4 * p[1])


def _fc(p):
    return -p[0] ** 2 + p[1] ** 2


def _gc(p):
    return (-2 * p[0], 2 * p[1])


_I = ((1.0, 0.0), (0.0, 1.0))

EXERCISE = Exercise(
    id="bfgs",
    func_name="bfgs_step",
    skeleton=SKELETON,
    hints=[
        "BFGS 的核心是割線條件：新的 H 必須滿足 H·y = s，也就是「梯度變了 y，位置就該變 s」。",
        "V = I − ρ s yᵀ，H_new = V·H·Vᵀ + ρ s sᵀ；在 2×2 時可以用兩層 list comprehension。",
        "ρ = 1.0 / dot(s, y)；最後一項是 rho * s[i] * s[j]",
    ],
    solution=demos.source("bfgs"),
    blanks=("f(x_new) <= fx + c1 * alpha * slope", "1.0 / sy", "rho * s[i] * s[j]"),
    cases=[
        StepCase("第一步（H = I）",
                 {"x": (1.0, 1.0), "H": _I, "f": _f, "grad_fn": _g},
                 ((0.0, -1.0),
                  ((1.0432098765432098, -0.1358024691358025),
                   (-0.1358024691358025, 0.2839506172839506)))),
        StepCase("第二步（用更新過的 H）",
                 {"x": (0.0, -1.0),
                  "H": ((1.0432098765432098, -0.1358024691358025),
                        (-0.1358024691358025, 0.2839506172839506)),
                  "f": _f, "grad_fn": _g},
                 ((-0.54320987654321, 0.1358024691358024),
                  ((0.8390828040323991, 0.08108501835557368),
                   (0.08108501835557368, 0.2693898956937241)))),
        StepCase("負曲率：sᵀy ≤ 0 時不更新 H",
                 {"x": (1.0, 0.0), "H": _I, "f": _fc, "grad_fn": _gc},
                 ((3.0, 0.0), _I)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "rosenbrock", "params": {}, "n_iter": 200},
)

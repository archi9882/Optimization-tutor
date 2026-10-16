"""Exercise: one L-BFGS step (two-loop recursion)."""
from __future__ import annotations

from .. import demos
from ..demos import lbfgs as DEMO
from ..grading import Exercise, StepCase

SKELETON = demos.helpers_source("lbfgs") + '''def lbfgs_step(x, memory, m, f, grad_fn):
    """一步 L-BFGS。

    x:       長度 2 的 tuple
    memory:  list of (s, y)，由舊到新，最多 m 組
    m:       記憶組數
    f, grad_fn: 目標函數與梯度函數
    回傳:    (x_new, memory_new)
    """
    # TODO 1: 兩迴圈遞推算出 r ≈ H·g（第一迴圈由新到舊，第二迴圈由舊到新）
    # TODO 2: d = −r，backtracking 線搜尋
    # TODO 3: sᵀy > 1e-12 才把 (s, y) 加入 memory，只保留最近 m 組
    raise NotImplementedError
'''


def _f(p):
    return p[0] ** 2 + 2 * p[1] ** 2


def _g(p):
    return (2 * p[0], 4 * p[1])


_PAIR1 = ((-1.0, -2.0), (-2.0, -8.0))
_PAIR2 = ((0.18300653594771238, 0.9542483660130718),
          (0.36601307189542476, 3.8169934640522873))
_X2 = (0.18300653594771238, -0.045751633986928164)

EXERCISE = Exercise(
    id="lbfgs",
    func_name="lbfgs_step",
    skeleton=SKELETON,
    hints=[
        "L-BFGS 從不存 H，只用最近幾組 (s, y) 算出 H·g 的結果，記憶體是 O(m·n)。",
        "第一迴圈：a = ρ·sᵀq，q ← q − a·y；中間乘 γ = sᵀy / yᵀy；第二迴圈：b = ρ·yᵀr，r ← r + (a − b)·s",
        "保留最近 m 組：memory = (memory + [(s, y)])[-int(m):]",
    ],
    solution=demos.source("lbfgs"),
    blanks=("rho * dot(s, q)", "dot(s, y) / dot(y, y)", "(a - b) * si"),
    cases=[
        StepCase("沒有記憶時就是梯度下降加線搜尋",
                 {"x": (1.0, 1.0), "memory": [], "m": 5, "f": _f, "grad_fn": _g},
                 ((0.0, -1.0), [_PAIR1])),
        StepCase("用一組記憶修正方向",
                 {"x": (0.0, -1.0), "memory": [_PAIR1], "m": 5, "f": _f, "grad_fn": _g},
                 (_X2, [_PAIR1, _PAIR2])),
        StepCase("m = 1：只留最新一組",
                 {"x": (0.0, -1.0), "memory": [_PAIR1], "m": 1, "f": _f, "grad_fn": _g},
                 (_X2, [_PAIR2])),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "rosenbrock", "params": {"m": 5}, "n_iter": 200},
)

"""Exercise: one step of the primal active-set method for a 2D QP."""
from __future__ import annotations

from .. import demos, lp
from ..grading import Exercise, StepCase
from ._lpcheck import as_trace, compare_runs

SKELETON = demos.helpers_source("active_set") + '''def active_set_step(x, W, Q, q, A, b):
    """有效集法一步，回傳 (x_new, W_new, done)。

    目標 min ½ xᵀQx + qᵀx，限制 A x ≤ b；W 是目前視為等式的限制編號。
    """
    # TODO 1: 依 |W| = 0、1、2 求出在 W 上的最佳方向 p
    # TODO 2: p ≈ 0 時算乘數 λ；全部 ≥ 0 就回傳 done=True，否則移除 λ 最負的限制
    # TODO 3: p ≠ 0 時沿 p 走，遇到第一條擋住的限制就停下並加入 W
    raise NotImplementedError
'''

_P = lp.farm_qp()
_A, _b, _ = _P.all_constraints()


def check(ns: dict) -> tuple:
    """Run the whole active-set method with the learner's step."""
    ref = lp.run_active_set(_P)
    user = lp.run_active_set(_P, ns["active_set_step"])
    results = compare_runs(user, ref, _P.objective, "求出最佳種植計畫 (3, 4.5)")
    return results, as_trace(user, _P.objective), as_trace(ref, _P.objective)


EXERCISE = Exercise(
    id="active_set",
    func_name="active_set_step",
    skeleton=SKELETON,
    hints=[
        "工作集 W 裡的限制被當成等式。|W| = 1 時只能沿著那條線移動，方向 t = (−a₂, a₁)。",
        "沿 t 的最佳步長 = −(tᵀg) / (tᵀQt)，其中 g = Qx + q；擋住的限制看 a_iᵀp > 0 的那些。",
        "乘數由 g + Σ λ_i a_i = 0 求出；λ < 0 表示那條限制其實在把 x 往可行域外推，應該放開。",
    ],
    solution=demos.source("active_set"),
    blanks=("solve2(Q, (-g[0], -g[1]))", "-dot(a, g) / dot(a, a)",
            "(bi - dot(ai, x)) / ap"),
    cases=[
        StepCase("W 為空：朝無限制最低點走，被灌溉水擋住",
                 {"x": (0.0, 0.0), "W": [], "Q": _P.Q, "q": _P.c, "A": _A, "b": _b},
                 ((3.103448275862069, 4.344827586206897), [2], False)),
        StepCase("沿灌溉水限制線走到最低點",
                 {"x": (3.103448275862069, 4.344827586206897), "W": [2], "Q": _P.Q,
                  "q": _P.c, "A": _A, "b": _b},
                 ((3.0, 4.5), [2], False)),
        StepCase("乘數 λ ≥ 0：已達最優",
                 {"x": (3.0, 4.5), "W": [2], "Q": _P.Q, "q": _P.c, "A": _A, "b": _b},
                 ((3.0, 4.5), [2], True)),
        StepCase("λ < 0：目標在限制內側，放開這條限制",
                 {"x": (4.0, 1.0), "W": [0], "Q": ((1.0, 0.0), (0.0, 1.0)),
                  "q": (-1.0, -1.0), "A": [(1.0, 0.0)], "b": [4.0]},
                 ((4.0, 1.0), [], False)),
        StepCase("頂點上乘數為負：放開 x₂ ≥ 0",
                 {"x": (4.0, 0.0), "W": [0, 4], "Q": _P.Q, "q": _P.c, "A": _A, "b": _b},
                 ((4.0, 0.0), [0], False)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    lp_preset="farm_qp",
    loop_note="上方動畫每一步呼叫一次 active_set_step()，把回傳的 x 與工作集 W 傳回下一步，直到回傳 done=True。",
)

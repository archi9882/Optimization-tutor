"""Exercise: dual simplex pivot selection."""
from __future__ import annotations

from .. import demos, lp
from ..grading import Exercise, StepCase
from ._lpcheck import as_trace, compare_runs

SKELETON = demos.helpers_source("dual_simplex") + '''def dual_choose_pivot(T):
    """對偶單純形選樞紐：回傳 (列, 欄)；已最優回傳 None；不可行回傳 (列, None)。

    T 的最後一列是目標列（檢驗數都 ≥ 0），每列最後一個數是右手邊。
    """
    # TODO 1: 出基列 = 右手邊（不含目標列）最負的那一列；若它 ≥ 0 回傳 None
    # TODO 2: 在該列係數 < 0 的欄中，取 |檢驗數 / 係數| 最小的欄
    # TODO 3: 沒有負係數時回傳 (列, None)
    raise NotImplementedError
'''

_D0 = [[-2.0, -1.0, 1.0, 0.0, -8.0], [-1.0, -2.0, 0.0, 1.0, -7.0], [2.0, 3.0, 0.0, 0.0, 0.0]]
_D1 = [[1.0, 0.5, -0.5, 0.0, 4.0], [0.0, -1.5, -0.5, 1.0, -3.0], [0.0, 2.0, 1.0, 0.0, -8.0]]
_D2 = [[1.0, 0.0, -2 / 3, 1 / 3, 3.0], [0.0, 1.0, 1 / 3, -2 / 3, 2.0],
       [0.0, 0.0, 1 / 3, 4 / 3, -12.0]]


def check(ns: dict) -> tuple:
    """Run the full dual simplex with the learner's selection rule."""
    prob = lp.fertilizer()
    ref = lp.run_dual_simplex(prob)
    user = lp.run_dual_simplex(prob, ns["dual_choose_pivot"], ns["pivot"])
    results = compare_runs(user, ref, prob.objective, "求出最便宜的施肥配方")
    return results, as_trace(user, prob.objective), as_trace(ref, prob.objective)


EXERCISE = Exercise(
    id="dual_simplex",
    func_name="dual_choose_pivot",
    required=("pivot",),
    skeleton=SKELETON,
    hints=[
        "原始單純形先選進基欄、再選出基列；對偶單純形反過來：先挑違反最嚴重的限制（列），再挑進基欄。",
        "對偶比值測試只看該列係數 < 0 的欄：用它們樞紐才能讓右手邊變正，而且檢驗數仍 ≥ 0。",
        "row = min(range(m), key=lambda i: T[i][-1])；比值是 abs(T[-1][j] / T[row][j])",
    ],
    solution=demos.source("dual_simplex"),
    blanks=("min(range(m), key=lambda i: T[i][-1])", "T[row][j] < -1e-12",
            "abs(T[-1][j] / T[row][j])"),
    cases=[
        StepCase("起始表：氮的列出基、x₁ 進基", {"T": _D0}, ((0, 0),)),
        StepCase("第二次：磷的列出基、x₂ 進基", {"T": _D1}, ((1, 1),)),
        StepCase("比值測試選到的不是第一個負係數欄",
                 {"T": [[-1.0, -2.0, 1.0, 0.0, -4.0], [1.0, 1.0, 0.0, 1.0, 6.0],
                        [3.0, 1.0, 0.0, 0.0, 0.0]]}, ((0, 1),)),
        StepCase("右手邊都 ≥ 0：已最優", {"T": _D2}, (None,)),
        StepCase("沒有負係數：原問題不可行",
                 {"T": [[1.0, 1.0, 1.0, -2.0], [1.0, 1.0, 0.0, 0.0]]}, ((0, None),)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    lp_preset="fertilizer",
    loop_note="上方動畫每一步呼叫一次 dual_choose_pivot() 選樞紐，再呼叫同一個 pivot() 更新表格；外層迴圈在右手邊全部 ≥ 0 或判定不可行時停止。",
)

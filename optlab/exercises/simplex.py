"""Exercise: the simplex pivot and pivot-selection rules."""
from __future__ import annotations

from .. import demos, lp
from ..grading import Exercise, StepCase
from ._lpcheck import as_trace, compare_runs

SKELETON = '''def choose_pivot(T):
    """選樞紐：回傳 (列, 欄)；已最優回傳 None；無界回傳 (None, 欄)。

    T 的最後一列是目標列（檢驗數），每列最後一個數是右手邊。
    """
    # TODO 1: 進基欄 = 目標列（不含右手邊）中最小（最負）的那一欄
    # TODO 2: 若它 ≥ 0，回傳 None
    # TODO 3: 對該欄係數 > 0 的列算比值 右手邊 / 係數，取最小比值的列
    raise NotImplementedError


def pivot(T, row, col):
    """以 T[row][col] 為樞紐做高斯消去，回傳新的表（list of lists）。"""
    # TODO 1: 樞紐列整列除以樞紐元
    # TODO 2: 其他每一列減去「該列在 col 的係數 × 新樞紐列」
    raise NotImplementedError
'''

_T0 = [[1.0, 0.0, 1.0, 0.0, 0.0, 4.0], [0.0, 2.0, 0.0, 1.0, 0.0, 12.0],
       [3.0, 2.0, 0.0, 0.0, 1.0, 18.0], [-3.0, -5.0, 0.0, 0.0, 0.0, 0.0]]
_T1 = [[1.0, 0.0, 1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 0.5, 0.0, 6.0],
       [3.0, 0.0, 0.0, -1.0, 1.0, 6.0], [-3.0, 0.0, 0.0, 2.5, 0.0, 30.0]]
_T2 = [[0.0, 0.0, 1.0, 1 / 3, -1 / 3, 2.0], [0.0, 1.0, 0.0, 0.5, 0.0, 6.0],
       [1.0, 0.0, 0.0, -1 / 3, 1 / 3, 2.0], [0.0, 0.0, 0.0, 1.5, 1.0, 36.0]]


def check(ns: dict) -> tuple:
    """Run the full simplex with the learner's functions on two objectives."""
    results, user_trace, ref_trace = [], None, None
    for c in [(3.0, 5.0), (5.0, 1.0)]:
        prob = lp.farm(*c)
        ref = lp.run_simplex(prob)
        user = lp.run_simplex(prob, ns["choose_pivot"], ns["pivot"])
        res = compare_runs(user, ref, prob.objective,
                           f"c = {c}：求出最優頂點與 z")
        results.extend(res if c == (3.0, 5.0) else res[:1])
        if user_trace is None:
            user_trace = as_trace(user, prob.objective)
            ref_trace = as_trace(ref, prob.objective)
    return results, user_trace, ref_trace


EXERCISE = Exercise(
    id="simplex",
    func_name="pivot",
    required=("choose_pivot",),
    skeleton=SKELETON,
    hints=[
        "樞紐運算就是解線性方程組時的高斯消去：讓進基欄變成單位向量，其他基變數的欄維持不變。",
        "比值測試保證走到相鄰頂點時仍可行：取 b_i / a_ij（a_ij > 0）最小的列，否則某個變數會變成負的。",
        "pivot：new_row = [v / T[row][col] for v in T[row]]；其他列 r 變成 [a - r[col] * b for a, b in zip(r, new_row)]",
    ],
    solution=demos.source("simplex"),
    blanks=("min(range(len(z) - 1), key=lambda j: z[j])", "T[i][-1] / T[i][col]",
            "v / T[row][col]", "a - factor * b"),
    cases=[
        StepCase("起始表選樞紐：x₂ 進基、s₂ 出基", {"T": _T0}, ((1, 1),), func="choose_pivot"),
        StepCase("第二次選樞紐：x₁ 進基、s₃ 出基", {"T": _T1}, ((2, 0),), func="choose_pivot"),
        StepCase("已最優時回傳 None", {"T": _T2}, (None,), func="choose_pivot"),
        StepCase("無界時回傳 (None, 欄)",
                 {"T": [[1.0, -1.0, 1.0, 2.0], [-1.0, -2.0, 0.0, 0.0]]}, ((None, 1),),
                 func="choose_pivot"),
        StepCase("第一次樞紐運算", {"T": _T0, "row": 1, "col": 1}, (_T1,)),
        StepCase("第二次樞紐運算", {"T": _T1, "row": 2, "col": 0}, (_T2,)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    lp_preset="farm",
    loop_note="上方動畫每一步呼叫一次 choose_pivot() 選樞紐，再呼叫 pivot() 更新單純形表；外層迴圈負責記錄每個頂點與表格，並在最優或無界時停止。",
)

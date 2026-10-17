"""Exercise: Gomory fractional cuts from a simplex tableau."""
from __future__ import annotations

from .. import demos, ip, lp
from ..grading import Exercise, StepCase
from ._combocheck import compare

SKELETON = '''import math


def frac(v):
    """小數部分，對負數也成立：frac(-0.25) = 0.75。接近整數時回傳 0。"""
    # TODO: v − ⌊v⌋，並把小於 1e-9 或大於 1 − 1e-9 的結果當成 0
    raise NotImplementedError


def choose_cut_row(T):
    """挑右手邊小數部分最大的列（不含最後的目標列）；全部是整數時回傳 None。"""
    raise NotImplementedError


def gomory_cut(T, row):
    """由第 row 列產生 Gomory 分數割，回傳要加進表格的新列。
    新列在原有各欄放 −frac(a_ij)，新 slack 欄放 1，右手邊放 −frac(b_i)。"""
    raise NotImplementedError
'''

# Optimal LP tableau of the greenhouse problem (columns x1 x2 s1 s2 | rhs).
_T = [[1.0, 0.0, 0.25, -0.5, 3.0], [0.0, 1.0, -0.125, 0.75, 1.5], [0.0, 0.0, 0.75, 0.5, 21.0]]


def check(ns: dict) -> tuple:
    """Run the full cutting-plane method with the learner's functions."""
    prob = lp.greenhouse()
    ref = ip.run_cutting_plane(prob)
    user = ip.run_cutting_plane(prob, ns["gomory_cut"], ns["choose_cut_row"])
    seq = lambda o: [s["x"] for s in o["steps"]]  # noqa: E731
    res = compare(user["steps"][-1]["x"], ref["steps"][-1]["x"], seq(user), seq(ref),
                  "加入割平面後得到整數最優解 (4, 0)")
    tr = lambda o: {"xs": seq(o), "fs": [s["z"] for s in o["steps"]], "status": o["status"]}  # noqa: E731
    return res, tr(user), tr(ref)


EXERCISE = Exercise(
    id="cutting_plane",
    func_name="gomory_cut",
    required=("frac", "choose_cut_row"),
    skeleton=SKELETON,
    hints=[
        "frac 要對負數也正確：−0.125 的小數部分是 0.875，所以要用 v − math.floor(v)，不能用 abs 或 int。",
        "基變數那一欄的係數是 0 或 1，小數部分是 0，所以割只牽涉非基變數。",
        "新列 = [−frac(a) for a in 該列除了右手邊的部分] + [1.0, −frac(右手邊)]。",
    ],
    solution=demos.source("cutting_plane"),
    blanks=("v - math.floor(v)", "frac(T[row][-1]) > 0", "-frac(r[-1])"),
    cases=[
        StepCase("正數的小數部分", {"v": 2.75}, (0.75,), func="frac"),
        StepCase("負數的小數部分", {"v": -0.125}, (0.875,), func="frac"),
        StepCase("整數", {"v": 3.0}, (0.0,), func="frac"),
        StepCase("挑右手邊 1.5 的那一列", {"T": _T}, (1,), func="choose_cut_row"),
        StepCase("整數解：不需要割", {"T": [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 9.0]]},
                 (None,), func="choose_cut_row"),
        StepCase("由第 2 列產生的割", {"T": _T, "row": 1},
                 ([0.0, 0.0, -0.875, -0.75, 1.0, -0.5],)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    lp_preset="greenhouse",
    loop_note="上方動畫先用單純形法解 LP 鬆弛，之後每一輪呼叫 choose_cut_row() 與 gomory_cut() 加一條割，再用對偶單純形法恢復可行；外層迴圈也負責把割換算成 x₁、x₂ 的直線畫在圖上。",
)

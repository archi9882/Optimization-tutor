"""Exercise: Hungarian method reduction and adjustment."""
from __future__ import annotations

from .. import combo, demos
from ..grading import Exercise, StepCase
from ._combocheck import compare

SKELETON = '''def reduce_matrix(C):
    """每列減去列最小值，再每欄減去欄最小值。"""
    raise NotImplementedError


def adjust(C, rows, cols):
    """rows、cols 是被直線蓋住的列與欄。沒被蓋住的元素減 k，交叉點加 k；回傳新矩陣。
    k 是沒被蓋住的元素中的最小值。"""
    # TODO 1: 找 k
    # TODO 2: 沒被蓋住的減 k；被列和欄同時蓋住的加 k；只被蓋一次的不變
    raise NotImplementedError
'''

_M = [[2, 0, 2], [1, 0, 5], [0, 0, 0]]


def check(ns: dict) -> tuple:
    """Run the whole method on the 5 × 5 machinery problem."""
    ref = combo.run_hungarian()
    user = combo.run_hungarian(ns["adjust"], ns["reduce_matrix"])
    seq = lambda o: [s["M"] for s in o["steps"]]  # noqa: E731
    fin = lambda o: [o.get("total"), o["steps"][-1]["assign"]]  # noqa: E731
    return compare(fin(user), fin(ref), seq(user), seq(ref), "最優指派（總時數 38）",
                   fmt=lambda v: f"總時數 {v[0]}，指派 {v[1]}"), None, None


EXERCISE = Exercise(
    id="hungarian",
    func_name="adjust",
    required=("reduce_matrix",),
    skeleton=SKELETON,
    hints=[
        "每列（或每欄）減去同一個常數，所有指派的總成本都減少同樣的量，最優指派不變。",
        "k 只從「列沒被蓋住而且欄也沒被蓋住」的元素裡找。",
        "交叉點要加 k：等於先把沒蓋住的列都減 k、再把蓋住的欄都加 k。",
    ],
    solution=demos.source("hungarian"),
    blanks=("v - min(row)", "v - k", "v + k"),
    cases=[
        StepCase("列與欄縮減", {"C": [[4, 1, 3], [2, 0, 5], [3, 2, 2]]}, (_M,), func="reduce_matrix"),
        StepCase("蓋住第 3 列與第 2 欄後調整", {"C": _M, "rows": [2], "cols": [1]},
                 ([[1, 0, 1], [0, 0, 4], [0, 1, 0]],)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    loop_note="上方動畫先呼叫 reduce_matrix()，之後每一輪由外層迴圈找最少的覆蓋線（本站的小矩陣直接暴力列舉），不夠 n 條就呼叫 adjust()。",
)

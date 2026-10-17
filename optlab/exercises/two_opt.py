"""Exercise: best-improvement 2-opt move."""
from __future__ import annotations

import math

from .. import combo, demos
from ..grading import Exercise, StepCase
from ._combocheck import compare

SKELETON = '''def tour_length(tour, D):
    """巡迴路線總長度（最後一點回到起點）。"""
    raise NotImplementedError


def two_opt_step(tour, D):
    """找增益最大的 2-opt 交換並執行；沒有能改善的交換時回傳 None。
    回傳 (新路線, i, j)：新路線是把 tour[i+1 … j] 反轉。"""
    # TODO 1: 對所有不相鄰的邊 (tour[i], tour[i+1]) 與 (tour[j], tour[j+1]) 算增益
    #         （i = 0 時 j 不能是 n − 1，因為最後一條邊回到 tour[0]，和第一條邊相鄰）
    # TODO 2: 增益 = 拿掉的兩條邊 − 換上的兩條邊；取最大且 > 1e-9 的
    # TODO 3: 反轉 tour[i+1 … j]
    raise NotImplementedError
'''

_SQ = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
_P = [(0, 0), (1, 0), (1, 1), (0, 1)]
_D = [[math.dist(p, q) for q in _P] for p in _P]


def check(ns: dict) -> tuple:
    """Run 2-opt on the 9 fields with the learner's step."""
    ref = combo.run_two_opt()
    user = combo.run_two_opt(ns["two_opt_step"])
    seq = lambda o: [s["tour"] for s in o["steps"]]  # noqa: E731
    return compare(round(user["length"], 6), round(ref["length"], 6), seq(user), seq(ref),
                   "停在局部最佳（18.643 km）", fmt=lambda v: f"{v:.3f} km"), None, None


EXERCISE = Exercise(
    id="two_opt",
    func_name="two_opt_step",
    required=("tour_length",),
    skeleton=SKELETON,
    hints=[
        "拿掉 (a→b)、(c→d)，換成 (a→c)、(b→d)；中間 b…c 那段要反過來走才接得上。",
        "增益 = D[a][b] + D[c][d] − D[a][c] − D[b][d]，大於 0 代表變短。",
        "Python 反轉一段：tour[i + 1:j + 1][::-1]。",
    ],
    solution=demos.source("two_opt"),
    blanks=("D[a][b] + D[c][d] - D[a][c] - D[b][d]", "gain > best_gain",
            "tour[i + 1:j + 1][::-1]"),
    cases=[
        StepCase("正方形周長", {"tour": [0, 1, 2, 3], "D": _D}, (4.0,), func="tour_length"),
        StepCase("交叉的路線：反轉中間兩站", {"tour": [0, 2, 1, 3], "D": _D}, (([0, 1, 2, 3], 0, 2),)),
        StepCase("已是最短：回傳 None", {"tour": [0, 1, 2, 3], "D": _D}, (None,)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    loop_note="上方動畫從 1 → 2 → … → 9 的路線出發，每一步呼叫一次 two_opt_step()，直到它回傳 None。",
)

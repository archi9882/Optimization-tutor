"""Exercise: one tabu-search move with aspiration."""
from __future__ import annotations

import math

from .. import combo, demos
from ..grading import Exercise, StepCase
from ._combocheck import compare

SKELETON = '''def tabu_step(tour, D, tabu, best_len):
    """做一步禁忌搜尋。tabu 是 {邊: 剩餘回合} 的字典，邊用 (小編號, 大編號) 表示。
    回傳 (新路線, i, j, 拿掉的兩條邊)；沒有可走的交換時回傳 None。"""
    # TODO 1: 對所有 2-opt 交換 (i, j) 算交換後的長度
    # TODO 2: 會加回禁忌邊的交換要跳過，除非交換後比 best_len 短（特赦）
    # TODO 3: 在允許的交換中取最短的（即使比現在長也照做）
    # TODO 4: 回傳新路線、i、j 與被拿掉的兩條邊
    raise NotImplementedError
'''

_P = [(0, 0), (1, 0), (1, 1), (0, 1)]
_D = [[math.dist(p, q) for q in _P] for p in _P]
_R2 = 2 * math.sqrt(2)


def check(ns: dict) -> tuple:
    """Run 20 rounds of tabu search from the 2-opt local optimum."""
    ref = combo.run_tabu()
    user = combo.run_tabu(ns["tabu_step"])
    seq = lambda o: [s["tour"] for s in o["steps"]]  # noqa: E731
    return compare(round(user["length"], 6), round(ref["length"], 6), seq(user), seq(ref),
                   "跳出局部最佳，找到 18.195 km", fmt=lambda v: f"{v:.3f} km"), None, None


EXERCISE = Exercise(
    id="tabu",
    func_name="tabu_step",
    skeleton=SKELETON,
    hints=[
        "和 2-opt 不同：不要求變短，只要是允許的交換中最短的就走。這才能離開局部最佳。",
        "一個交換會加上 (a, c) 與 (b, d) 兩條邊；只要其中一條在 tabu 裡，這個交換就是禁忌。",
        "特赦：禁忌的交換若能做出比歷史最佳 best_len 更短的路線，仍然允許。",
    ],
    solution=demos.source("tabu"),
    blanks=("cur - D[a][b] - D[c][d] + D[a][c] + D[b][d]", "any(e in tabu for e in added)",
            "new_len >= best_len - 1e-9"),
    cases=[
        StepCase("局部最佳也要走：選最不差的交換",
                 {"tour": [0, 1, 2, 3], "D": _D, "tabu": {}, "best_len": 4.0},
                 (([0, 2, 1, 3], 0, 2, [(0, 1), (2, 3)]),)),
        StepCase("所有交換都會加回禁忌邊：None",
                 {"tour": [0, 1, 2, 3], "D": _D, "tabu": {(0, 2): 3}, "best_len": 4.0}, (None,)),
        StepCase("特赦：禁忌但能打破歷史最佳",
                 {"tour": [0, 2, 1, 3], "D": _D, "tabu": {(0, 1): 2}, "best_len": 2 + _R2},
                 (([0, 1, 2, 3], 0, 2, [(0, 2), (1, 3)]),)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    loop_note="上方動畫從 2-opt 的局部最佳出發，每回合呼叫一次 tabu_step()；外層迴圈負責更新禁忌表（每回合剩餘次數減 1、加入剛拿掉的邊）與歷史最佳。",
)

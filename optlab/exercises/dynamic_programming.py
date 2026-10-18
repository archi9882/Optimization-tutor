"""Exercise: one backward Bellman stage."""
from __future__ import annotations

from .. import combo, demos
from ..grading import Exercise, StepCase
from ._combocheck import compare

SKELETON = '''def bellman_stage(V_next, t, S, inflow, value):
    """由下個月的價值表 V_next 算出這個月的 (V, 最優放水量)。
    蓄水量 s = 0…S；可放水量 r = 0…s + inflow；超過容量 S 的水會溢流。
    value(t, r) 是第 t 個月放 r 單位水的收益。"""
    # TODO 1: 對每個 s，試所有可放水量 r
    # TODO 2: 下個月初蓄水量 = s + inflow − r，但不能超過 S
    # TODO 3: 取「本月收益 + 下個月價值」最大的 r，記下最大值與 r
    raise NotImplementedError
'''


def _toy_value(t, r):
    return [0, 5, 6, 6.5][r]


def check(ns: dict) -> tuple:
    """Run the whole backward sweep and forward pass with the learner's stage."""
    ref = combo.run_dp()
    user = combo.run_dp(ns["bellman_stage"])
    seq = lambda o: [s["V"] for s in o["steps"][:len(o["months"]) + 1]]  # noqa: E731
    return compare(user["total"], ref["total"], seq(user), seq(ref),
                   "放水計畫的總收益（最優為 71）", fmt=lambda v: f"{v:g}"), None, None


EXERCISE = Exercise(
    id="dynamic_programming",
    func_name="bellman_stage",
    skeleton=SKELETON,
    hints=[
        "外層對 s 迴圈，內層對 r 迴圈；r 最多是這個月可用的水 s + inflow。",
        "下個月初的蓄水量是 min(s + inflow − r, S)：超過容量的水溢流掉。",
        "Bellman 方程：V(s) = max_r [ value(t, r) + V_next[下個月初蓄水量] ]。",
    ],
    solution=demos.source("dynamic_programming"),
    blanks=("range(s + inflow + 1)", "min(s + inflow - r, S)", "value(t, r) + V_next[nxt]"),
    cases=[
        StepCase("8 月（最後一個月）：把水全放掉最好",
                 {"V_next": [0.0] * 6, "t": 5, "S": 5, "inflow": 2, "value": combo.reservoir_value},
                 (([7, 9, 10, 10.5, 10.8, 11], [2, 3, 4, 5, 6, 7]),)),
        StepCase("小例子：容量 2 會溢流",
                 {"V_next": [0.0, 0.0, 10.0], "t": 0, "S": 2, "inflow": 1, "value": _toy_value},
                 (([5, 10, 15], [1, 0, 1]),)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    loop_note="上方動畫從 8 月往回，每個月呼叫一次 bellman_stage() 填一欄價值表；填完後外層迴圈從 3 月、蓄水 2 開始照著表中的最優放水量往前走。",
)

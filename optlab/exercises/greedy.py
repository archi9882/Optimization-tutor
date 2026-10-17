"""Exercise: Kruskal's greedy step with union-find."""
from __future__ import annotations

from .. import combo, demos
from ..grading import Exercise, StepCase
from ._combocheck import compare

SKELETON = '''def find(parent, i):
    """沿著 parent 找到 i 所在組的代表點（並把路徑壓平）。"""
    # TODO: 一直往上走直到 parent[i] == i；走的同時把 parent[i] 指向祖父，讓之後更快
    raise NotImplementedError


def kruskal_step(edge, parent):
    """考慮一條邊 (長度, u, v)：若 u、v 在不同組就合併並回傳 True，否則回傳 False。"""
    # TODO 1: 找 u、v 的代表點
    # TODO 2: 同一組：選它會形成迴圈，回傳 False
    # TODO 3: 不同組：把一組掛到另一組下面，回傳 True
    raise NotImplementedError
'''


def check(ns: dict) -> tuple:
    """Run Kruskal on the 9 fields with the learner's functions."""
    ref = combo.run_kruskal()
    user = combo.run_kruskal(ns["kruskal_step"])
    seq = lambda o: [s["edges"] for s in o["steps"]]  # noqa: E731
    return compare(round(user["total"], 6), round(ref["total"], 6), seq(user), seq(ref),
                   "最小生成樹總長（11.944 km）", fmt=lambda v: f"{v:.3f} km"), None, None


EXERCISE = Exercise(
    id="greedy",
    func_name="kruskal_step",
    required=("find",),
    skeleton=SKELETON,
    hints=[
        "每組用一棵樹表示，樹根就是代表點：parent[r] == r。",
        "路徑壓平：parent[i] = parent[parent[i]]，再 i = parent[i]。",
        "u、v 的代表點相同代表已經連通；不同時 parent[ru] = rv 把兩組合併。",
    ],
    solution=demos.source("greedy"),
    blanks=("parent[parent[i]]", "ru == rv", "parent[ru] = rv"),
    cases=[
        StepCase("找代表點", {"parent": [0, 0, 1, 3], "i": 2}, (0,), func="find"),
        StepCase("自己就是代表點", {"parent": [0, 0, 1, 3], "i": 3}, (3,), func="find"),
        StepCase("兩個單獨的點：選", {"edge": (1.0, 0, 1), "parent": [0, 1, 2]}, (True,)),
        StepCase("不同組：選", {"edge": (1.0, 0, 2), "parent": [1, 1, 2]}, (True,)),
        StepCase("同一組：會形成迴圈，不選", {"edge": (1.0, 0, 1), "parent": [1, 1, 2]}, (False,)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    loop_note="上方動畫先把所有候選管線依長度排序，再依序對每一條呼叫 kruskal_step()，選滿 n − 1 條為止。",
)

"""Branch-and-bound demo for a 2-variable integer program (maximisation)."""
from __future__ import annotations

import math

PSEUDOCODE = """\
best ← −∞，待處理清單 ← {原問題}
當清單不空：
    取出一個節點，解它的 LP 鬆弛，得到 x、z
    若不可行：剪枝                         # 這個分支沒有任何解
    若 z ≤ best：剪枝                      # 鬆弛的上界都不比現有解好
    若 x 全是整數：best ← z，記下 x         # 新的現有最佳解（incumbent）
    否則：挑一個非整數的 x_j，分成兩個子問題
          x_j ≤ ⌊x_j⌋  與  x_j ≥ ⌊x_j⌋ + 1   # 中間的分數區域被切掉
"""


def most_fractional(x):
    """回傳小數部分最接近 0.5 的變數編號；全部是整數時回傳 None。"""
    best, j_best = 1e-6, None
    for j, v in enumerate(x):
        dist = abs(v - round(v))            # 離最近整數多遠
        if dist > best:
            best, j_best = dist, j
    return j_best


def bb_decide(x, z, best_z):
    """處理一個節點的 LP 鬆弛解 (x, z)，回傳要做的事：
    ("infeasible",)、("prune",)、("incumbent",) 或 ("branch", j, ⌊x_j⌋, ⌊x_j⌋ + 1)。"""
    if x is None:
        return ("infeasible",)
    if z <= best_z + 1e-9:
        return ("prune",)
    j = most_fractional(x)
    if j is None:
        return ("incumbent",)
    lo = math.floor(x[j])
    return ("branch", j, lo, lo + 1)


FUNC = bb_decide
HELPERS = [most_fractional]
HEADER = "import math\n\n\n"

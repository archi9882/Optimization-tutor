"""2-opt local search for the travelling salesman problem."""
from __future__ import annotations

PSEUDOCODE = """\
tour ← 任一條巡迴路線
重複：
    對所有不相鄰的兩條邊 (a→b) 與 (c→d)：
        增益 = d(a,b) + d(c,d) − d(a,c) − d(b,d)   # 換成 (a→c)、(b→d) 能省多少
    若最大增益 ≤ 0：沒有任何 2-opt 交換能改善，停止（局部最佳）
    把 b…c 這一段反轉                             # 等於把兩條邊換掉
"""


def tour_length(tour, D):
    """巡迴路線總長度（最後一點回到起點）。"""
    return sum(D[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def two_opt_step(tour, D):
    """找增益最大的 2-opt 交換並執行；沒有能改善的交換時回傳 None。
    回傳 (新路線, i, j)：新路線是把 tour[i+1 … j] 反轉。"""
    n = len(tour)
    best_gain, best_ij = 1e-9, None
    for i in range(n - 1):
        for j in range(i + 2, n if i > 0 else n - 1):
            a, b = tour[i], tour[i + 1]
            c, d = tour[j], tour[(j + 1) % n]
            gain = D[a][b] + D[c][d] - D[a][c] - D[b][d]
            if gain > best_gain:
                best_gain, best_ij = gain, (i, j)
    if best_ij is None:
        return None
    i, j = best_ij
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:], i, j


FUNC = two_opt_step
HELPERS = [tour_length]
HEADER = ""

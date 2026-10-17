"""Backward dynamic programming for a small reservoir release schedule."""
from __future__ import annotations

PSEUDOCODE = """\
V_T(s) ← 0                                   # 最後一個月之後，剩下的水不再計價
對 t = T−1, T−2, …, 0（從最後一個月往回算）：
    對每個月初蓄水量 s：
        V_t(s) ← max_r [ 收益_t(r) + V_{t+1}(下個月初蓄水量) ]   # Bellman 方程
        記下讓它最大的放水量 r
從 s₀ 出發，照記下的放水量往前走一遍，得到最佳放水計畫
"""


def bellman_stage(V_next, t, S, inflow, value):
    """由下個月的價值表 V_next 算出這個月的 (V, 最佳放水量)。
    蓄水量 s = 0…S；可放水量 r = 0…s + inflow；超過容量 S 的水會溢流。"""
    V, policy = [], []
    for s in range(S + 1):
        best, best_r = None, None
        for r in range(s + inflow + 1):
            nxt = min(s + inflow - r, S)        # 下個月初的蓄水量
            total = value(t, r) + V_next[nxt]
            if best is None or total > best:
                best, best_r = total, r
        V.append(best)
        policy.append(best_r)
    return V, policy


FUNC = bellman_stage
HELPERS = []
HEADER = ""

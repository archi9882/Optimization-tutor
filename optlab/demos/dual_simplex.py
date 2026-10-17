"""Dual simplex demo: keep reduced costs >= 0, restore primal feasibility."""
from __future__ import annotations

from .simplex import pivot

PSEUDOCODE = """\
起點：檢驗數都 ≥ 0（對偶可行），但右手邊有負值（原始不可行）
重複：
    出基：右手邊最負的列 i                 # 違反最嚴重的限制
    若所有右手邊 ≥ 0：原始也可行，已達最優，停止
    進基：在 a_ij < 0 的欄中，取 |檢驗數_j / a_ij| 最小的欄 j
    若沒有 a_ij < 0：原問題不可行，停止
    以 (i, j) 為樞紐做高斯消去           # 和原始單純形用同一個 pivot
"""


def dual_choose_pivot(T):
    """先選右手邊最負的列出基，再用對偶比值測試選進基欄。最優回傳 None，不可行回傳 (列, None)。"""
    m = len(T) - 1
    row = min(range(m), key=lambda i: T[i][-1])
    if T[row][-1] >= -1e-12:
        return None
    ratios = [(abs(T[-1][j] / T[row][j]), j) for j in range(len(T[0]) - 1)
              if T[row][j] < -1e-12]
    if not ratios:
        return (row, None)
    return (row, min(ratios)[1])


FUNC = dual_choose_pivot
HELPERS = [pivot]
HEADER = ""

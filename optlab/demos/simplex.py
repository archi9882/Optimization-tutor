"""Primal simplex demo on a tableau (maximisation, Dantzig's rule)."""
from __future__ import annotations

PSEUDOCODE = """\
把 max cᵀx, Ax ≤ b, x ≥ 0 加上 slack 寫成單純形表
重複：
    進基：目標列中最負的檢驗數所在欄 j       # 增加 x_j 最能提高 z
    若所有檢驗數 ≥ 0：已達最優，停止
    出基：對 a_ij > 0 的列做比值測試 b_i / a_ij，取最小的列 i
    若沒有 a_ij > 0：問題無界，停止
    以 (i, j) 為樞紐做高斯消去           # 沿可行域的邊走到相鄰頂點
"""


def choose_pivot(T):
    """Dantzig 規則選進基欄，最小比值測試選出基列。最優回傳 None，無界回傳 (None, 欄)。"""
    z = T[-1]
    col = min(range(len(z) - 1), key=lambda j: z[j])
    if z[col] >= -1e-12:
        return None
    ratios = [(T[i][-1] / T[i][col], i) for i in range(len(T) - 1)
              if T[i][col] > 1e-12]
    if not ratios:
        return (None, col)
    return (min(ratios)[1], col)


def pivot(T, row, col):
    """以 T[row][col] 為樞紐：樞紐列除以樞紐元，其他列減去倍數讓該欄變成 0。"""
    new_row = [v / T[row][col] for v in T[row]]
    T_new = []
    for i, r in enumerate(T):
        if i == row:
            T_new.append(new_row)
        else:
            factor = r[col]
            T_new.append([a - factor * b for a, b in zip(r, new_row)])
    return T_new


FUNC = pivot
HELPERS = [choose_pivot]
HEADER = ""

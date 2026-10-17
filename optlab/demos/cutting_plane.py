"""Gomory fractional cutting-plane demo on a simplex tableau."""
from __future__ import annotations

import math

PSEUDOCODE = """\
用單純形法解 LP 鬆弛，得到最優表
重複：
    若基變數的值全是整數：已是整數最優解，停止
    挑右手邊小數部分最大的列 i
    由第 i 列產生 Gomory 割：Σ_j frac(a_ij)·x_j ≥ frac(b_i)   # 只對非基變數
    把割加進表格（新的 slack 欄），原本的解違反它
    用對偶單純形法恢復可行                    # 檢驗數仍 ≥ 0，正好用對偶單純形
"""


def frac(v):
    """小數部分，對負數也成立：frac(-0.25) = 0.75。接近整數時回傳 0。"""
    f = v - math.floor(v)
    return 0.0 if f < 1e-9 or f > 1 - 1e-9 else f


def choose_cut_row(T):
    """挑右手邊小數部分最大的列；全部是整數時回傳 None。"""
    m = len(T) - 1
    row = max(range(m), key=lambda i: frac(T[i][-1]))
    return row if frac(T[row][-1]) > 0 else None


def gomory_cut(T, row):
    """由第 row 列產生 Gomory 分數割，回傳要加進表格的新列。
    新列在原有各欄放 −frac(a_ij)，新 slack 欄放 1，右手邊放 −frac(b_i)。"""
    r = T[row]
    return [-frac(v) for v in r[:-1]] + [1.0, -frac(r[-1])]


FUNC = gomory_cut
HELPERS = [frac, choose_cut_row]
HEADER = "import math\n\n\n"

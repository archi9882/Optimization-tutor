"""Hungarian method: reduce, cover the zeros, adjust."""
from __future__ import annotations

PSEUDOCODE = """\
每列減去該列最小值；每欄減去該欄最小值       # 每列、每欄都至少有一個 0
重複：
    用最少的直線（整列或整欄）蓋住所有 0
    若需要 n 條線：可以在 0 上選出 n 個不同列不同欄的位置，就是最佳指派，停止
    否則：k ← 沒被蓋住的元素中的最小值
          沒被蓋住的元素都減 k，被蓋兩次（交叉點）的元素都加 k
"""


def reduce_matrix(C):
    """每列減去列最小值，再每欄減去欄最小值。"""
    R = [[v - min(row) for v in row] for row in C]
    col_min = [min(R[i][j] for i in range(len(R))) for j in range(len(R[0]))]
    return [[v - col_min[j] for j, v in enumerate(row)] for row in R]


def adjust(C, rows, cols):
    """rows、cols 是被直線蓋住的列與欄。沒被蓋住的元素減 k，交叉點加 k；回傳新矩陣。"""
    k = min(C[i][j] for i in range(len(C)) for j in range(len(C[0]))
            if i not in rows and j not in cols)
    out = []
    for i, row in enumerate(C):
        new = []
        for j, v in enumerate(row):
            if i not in rows and j not in cols:
                new.append(v - k)
            elif i in rows and j in cols:
                new.append(v + k)
            else:
                new.append(v)
        out.append(new)
    return out


FUNC = adjust
HELPERS = [reduce_matrix]
HEADER = ""

"""Tiny 2D linear-algebra helpers shared by the second-order demos.

They are shown on the pages together with each demo function, so they are
written to be read, not to be fast.
"""
from __future__ import annotations

import math


def dot(u, v):
    """內積 uᵀv。"""
    return sum(ui * vi for ui, vi in zip(u, v))


def norm(u):
    """歐氏長度 ‖u‖。"""
    return math.sqrt(dot(u, u))


def matvec(A, v):
    """矩陣乘向量 A·v。"""
    return tuple(dot(row, v) for row in A)


def solve2(A, b):
    """解 2×2 線性系統 A·d = b（Cramer 法則）；A 奇異時回傳 None。"""
    (a11, a12), (a21, a22) = A
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-12:
        return None
    return ((b[0] * a22 - a12 * b[1]) / det, (a11 * b[1] - a21 * b[0]) / det)


def backtracking(f, x, fx, g, d, alpha=1.0, c1=1e-4, shrink=0.5):
    """Armijo 回溯線搜尋：從 alpha 開始每次減半，直到函數值「下降得夠多」。"""
    slope = dot(g, d)                      # 沿 d 的方向導數，下降方向時為負
    x_new = x
    for _ in range(60):
        x_new = (x[0] + alpha * d[0], x[1] + alpha * d[1])
        if f(x_new) <= fx + c1 * alpha * slope:
            return x_new
        alpha *= shrink
    return x_new

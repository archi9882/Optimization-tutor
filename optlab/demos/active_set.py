"""Primal active-set demo for a 2D convex QP: min 1/2 x^T Q x + q^T x, A x <= b."""
from __future__ import annotations

from ._linalg import dot, solve2

PSEUDOCODE = """\
W ← 工作集（目前視為等式的限制），x ← 可行起點
重複：
    在 W 的等式限制下最小化 q(x + p)，得到方向 p
    若 p = 0：
        算 W 中每條限制的乘數 λ（Qx + q + Σ λ_i a_i = 0）
        若所有 λ ≥ 0：KKT 成立，已達最優，停止
        否則把 λ 最負的限制移出 W                # 它在把 x 往外拉，放開它
    否則：
        沿 p 前進，直到 1 或第一條擋住的限制為止
        若被擋住：把那條限制加入 W
"""


def active_set_step(x, W, Q, q, A, b):
    """有效集法一步：回傳 (x_new, W_new, done)。W 是限制編號的 list。"""
    g = [dot(Q[0], x) + q[0], dot(Q[1], x) + q[1]]          # 目前梯度 Qx + q
    if len(W) == 0:
        p = solve2(Q, (-g[0], -g[1]))                       # 無限制牛頓步
    elif len(W) == 1:
        a = A[W[0]]
        t = (-a[1], a[0])                                   # 沿限制線的方向
        p = tuple(-(dot(t, g) / dot(t, [dot(Q[0], t), dot(Q[1], t)])) * ti for ti in t)
    else:
        p = (0.0, 0.0)                                      # 兩條限制交於一點
    if abs(p[0]) + abs(p[1]) < 1e-10:
        if len(W) == 0:
            return x, W, True
        if len(W) == 1:
            a = A[W[0]]
            lams = [-dot(a, g) / dot(a, a)]
        else:
            M = [[A[W[0]][0], A[W[1]][0]], [A[W[0]][1], A[W[1]][1]]]
            lams = list(solve2(M, (-g[0], -g[1])))
        worst = min(range(len(W)), key=lambda i: lams[i])
        if lams[worst] >= -1e-10:
            return x, W, True
        return x, [w for i, w in enumerate(W) if i != worst], False
    alpha, block = 1.0, None
    for i, (ai, bi) in enumerate(zip(A, b)):
        ap = dot(ai, p)
        if i not in W and ap > 1e-12:
            step = (bi - dot(ai, x)) / ap
            if step < alpha:
                alpha, block = step, i
    x_new = (x[0] + alpha * p[0], x[1] + alpha * p[1])
    return x_new, (W + [block] if block is not None else W), False


FUNC = active_set_step
HELPERS = [dot, solve2]
HEADER = ""

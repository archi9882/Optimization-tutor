"""Simple feasible sets: Euclidean projections and linear minimisation oracles.

Projected gradient needs P_C(y) = argmin_{x in C} ||x - y||; Frank-Wolfe
needs the linear minimisation oracle LMO_C(g) = argmin_{s in C} g^T s.
Both are cheap for a disk and for a convex polygon given by its vertices.
These functions are shown on the pages, so they are written to be read.
"""
from __future__ import annotations

import math


def project_disk(y, center, r):
    """投影到圓盤 ‖x − c‖ ≤ r：在圓內就不動，在圓外就沿半徑拉回圓周。"""
    d = (y[0] - center[0], y[1] - center[1])
    dist = math.hypot(d[0], d[1])
    if dist <= r:
        return (y[0], y[1])
    return (center[0] + r * d[0] / dist, center[1] + r * d[1] / dist)


def project_segment(y, a, b):
    """y 到線段 ab 上最近的點。"""
    v = (b[0] - a[0], b[1] - a[1])
    t = ((y[0] - a[0]) * v[0] + (y[1] - a[1]) * v[1]) / (v[0] ** 2 + v[1] ** 2)
    t = min(1.0, max(0.0, t))
    return (a[0] + t * v[0], a[1] + t * v[1])


def project_polygon(y, vertices):
    """投影到凸多邊形（頂點逆時針排列）：在內部就不動，否則取各邊上最近點中最近的一個。"""
    n = len(vertices)
    inside = all((vertices[(i + 1) % n][0] - vertices[i][0]) * (y[1] - vertices[i][1])
                 - (vertices[(i + 1) % n][1] - vertices[i][1]) * (y[0] - vertices[i][0]) >= 0
                 for i in range(n))
    if inside:
        return (y[0], y[1])
    candidates = [project_segment(y, vertices[i], vertices[(i + 1) % n]) for i in range(n)]
    return min(candidates, key=lambda p: (p[0] - y[0]) ** 2 + (p[1] - y[1]) ** 2)


def lmo_polygon(g, vertices):
    """線性最小化：在多邊形上最小化 gᵀs，最優解一定在某個頂點，逐一比較即可。"""
    return min(vertices, key=lambda v: g[0] * v[0] + g[1] * v[1])


def lmo_disk(g, center, r):
    """線性最小化：在圓盤上最小化 gᵀs，答案是從圓心沿 −g 走到圓周。"""
    n = math.hypot(g[0], g[1])
    if n == 0:
        return (center[0], center[1])
    return (center[0] - r * g[0] / n, center[1] - r * g[1] / n)

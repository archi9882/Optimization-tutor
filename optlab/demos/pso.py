"""Particle swarm optimisation demo (inertia weight, global-best topology)."""
from __future__ import annotations

import random

from ..problems import as_vec

PSEUDOCODE = """\
輸入：粒子數 N、慣性 w、認知係數 c₁、社會係數 c₂、迭代數 K、範圍 [lo, hi]
初始化：x_i 在範圍內均勻撒點（粒子 0 放在起點 x₀），v_i ← 0，p_i ← x_i
for k = 0, 1, …, K−1:
    g ← 所有 p_i 中 f 最小的那個             # 群體最佳位置
    for 每個粒子 i、每個座標 d:
        r₁, r₂ ← U(0, 1) 各抽一個
        v ← w·v + c₁·r₁·(p_i − x_i) + c₂·r₂·(g − x_i)
        x ← x + v；超出範圍就停在邊界，該座標速度歸零
    算新的 f(x_i)；比 f(p_i) 小就更新 p_i      # 個體最佳位置
"""


def pso_step(xs, vs, pbest, pbest_f, f, rng, w, c1, c2, bounds):
    """一次 PSO 迭代：全部粒子同步更新速度與位置，再更新個體最佳位置。"""
    g = pbest[min(range(len(pbest)), key=lambda i: pbest_f[i])]   # 群體最佳位置
    new_xs, new_vs = [], []
    for x, v, p in zip(xs, vs, pbest):
        pos, vel = [], []
        for d in range(len(x)):
            r1, r2 = rng.random(), rng.random()
            vd = w * v[d] + c1 * r1 * (p[d] - x[d]) + c2 * r2 * (g[d] - x[d])
            xd = x[d] + vd
            lo, hi = bounds[d]
            if xd < lo or xd > hi:                  # 撞到邊界：停在邊界，速度歸零
                xd, vd = min(max(xd, lo), hi), 0.0
            pos.append(xd)
            vel.append(vd)
        new_xs.append(tuple(pos))
        new_vs.append(tuple(vel))
    new_p, new_pf = list(pbest), list(pbest_f)
    for i, x in enumerate(new_xs):
        fx = f(x)
        if fx < new_pf[i]:                          # 個體最佳位置只會變好
            new_p[i], new_pf[i] = x, fx
    return new_xs, new_vs, new_p, new_pf


def init_swarm(x0, n, bounds, rng):
    """粒子 0 放在 x0，其餘在範圍內均勻撒點；速度全為 0。"""
    xs = [tuple(x0)] + [tuple(lo + (hi - lo) * rng.random() for lo, hi in bounds)
                        for _ in range(n - 1)]
    return xs, [(0.0, 0.0)] * n


def adapter(fn):
    """Lab loop: one swarm iteration per step; the path is the swarm's best point.

    The seeded random.Random(seed) gives the same swarm in CPython and Pyodide.
    Every f evaluation (initial swarm included) is counted.
    """
    def setup(x0, problem, p):
        x_lo, x_hi, y_lo, y_hi = problem.bounds
        bounds = ((x_lo, x_hi), (y_lo, y_hi))
        rng = random.Random(int(p["seed"]))
        xs, vs = init_swarm(x0, int(p["n_pop"]), bounds, rng)
        fs = [problem.f(x) for x in xs]
        return _report(xs, vs, list(xs), fs, fs, rng, bounds, len(xs))

    def step(x, problem, state, p):
        calls = [0]

        def f(v):
            calls[0] += 1
            return problem.f(v)

        xs, vs, pb, pbf = fn(state["xs"], state["vs"], state["pbest"], state["pbest_f"], f,
                             state["rng"], p["w"], p["c1"], p["c2"], state["bounds"])
        xs = [as_vec(v) for v in xs]
        vs = [as_vec(v) for v in vs]
        pb = [as_vec(v) for v in pb]
        pbf = [float(v) for v in pbf]
        cur = [problem.f(v) for v in xs]        # for the mean-f curve only (not counted)
        return _report(xs, vs, pb, pbf, cur, state["rng"], state["bounds"],
                       state["evals"] + calls[0])

    step.init = setup
    return step


def _report(xs, vs, pb, pbf, cur, rng, bounds, evals):
    best = min(range(len(pbf)), key=lambda i: pbf[i])
    return pb[best], {"xs": xs, "vs": vs, "pbest": pb, "pbest_f": pbf, "rng": rng,
                      "bounds": bounds, "evals": evals, "pop": xs,
                      "fmean": sum(cur) / len(cur), "pop_trail": True}


FUNC = pso_step
HEADER = ""

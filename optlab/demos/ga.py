"""Real-coded genetic algorithm demo: tournament, SBX, polynomial mutation, elitism."""
from __future__ import annotations

import random

from ..problems import as_vec

PSEUDOCODE = """\
輸入：族群大小 N、菁英數 E、交配率 p_c、SBX 分布指數 η_c、
      每個座標的突變率 p_m、突變分布指數 η_m、範圍 [lo, hi]
初始化：N 個個體在範圍內均勻撒點（個體 0 放在起點 x₀）
每一代：
    下一代 ← 本代 f 最小的 E 個個體（菁英，原封不動）
    重複直到下一代有 N 個：
        a ← 二元錦標賽（隨機抽 2 個，取較好者）；b 同理
        以機率 p_c 做 SBX 交配得到兩個子代，否則子代 = 父母的複本
        每個子代、每個座標：以機率 p_m 做多項式突變
        超出範圍的座標截回邊界
    算新個體的 f
"""


def tournament(pop, fvals, rng):
    """二元錦標賽：隨機抽兩個（可重複），回傳 f 較小者。"""
    i = int(rng.random() * len(pop))
    j = int(rng.random() * len(pop))
    return pop[j] if fvals[j] < fvals[i] else pop[i]


def sbx(a, b, rng, eta):
    """模擬二進位交配（SBX）：每個座標抽一個 u，兩個子代對稱地落在父母兩側。"""
    c1, c2 = [], []
    for ad, bd in zip(a, b):
        u = rng.random()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c1.append(0.5 * ((1 + beta) * ad + (1 - beta) * bd))
        c2.append(0.5 * ((1 - beta) * ad + (1 + beta) * bd))
    return tuple(c1), tuple(c2)


def mutate(x, rng, pm, eta, bounds):
    """多項式突變：每個座標以機率 pm 加上 δ·(hi − lo)，δ ∈ [−1, 1] 集中在 0 附近。"""
    out = []
    for xd, (lo, hi) in zip(x, bounds):
        if rng.random() < pm:
            u = rng.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            xd = xd + delta * (hi - lo)
        out.append(min(max(xd, lo), hi))            # 截回範圍內
    return tuple(out)


def ga_generation(pop, fvals, f, rng, bounds, n_elite, pc, eta_c, pm, eta_m):
    """一代 GA：保留菁英，其餘由錦標賽選出的父母交配、突變產生。"""
    order = sorted(range(len(pop)), key=lambda i: fvals[i])
    elite = order[:n_elite]
    new_pop = [pop[i] for i in elite]
    new_f = [fvals[i] for i in elite]
    for _ in range((len(pop) - n_elite + 1) // 2):
        a = tournament(pop, fvals, rng)
        b = tournament(pop, fvals, rng)
        if rng.random() < pc:
            c1, c2 = sbx(a, b, rng, eta_c)
        else:
            c1, c2 = a, b
        for c in (c1, c2):
            if len(new_pop) < len(pop):
                child = mutate(c, rng, pm, eta_m, bounds)
                new_pop.append(child)
                new_f.append(f(child))
    return new_pop, new_f


def adapter(fn):
    """Lab loop: one generation per step; the path is the population's best point.

    Seeded random.Random(seed) (same in CPython and Pyodide). Every f
    evaluation, the initial population included, is counted.
    """
    def setup(x0, problem, p):
        x_lo, x_hi, y_lo, y_hi = problem.bounds
        bounds = ((x_lo, x_hi), (y_lo, y_hi))
        rng = random.Random(int(p["seed"]))
        n = int(p["n_pop"])
        pop = [tuple(x0)] + [tuple(lo + (hi - lo) * rng.random() for lo, hi in bounds)
                             for _ in range(n - 1)]
        fv = [problem.f(x) for x in pop]
        return _report(pop, fv, rng, bounds, n)

    def step(x, problem, state, p):
        calls = [0]

        def f(v):
            calls[0] += 1
            return problem.f(v)

        pop, fv = fn(state["pop"], state["fvals"], f, state["rng"], state["bounds"],
                     int(p["n_elite"]), p["pc"], p["eta_c"], p["pm"], p["eta_m"])
        pop = [as_vec(v) for v in pop]
        fv = [float(v) for v in fv]
        return _report(pop, fv, state["rng"], state["bounds"], state["evals"] + calls[0])

    step.init = setup
    return step


def _report(pop, fv, rng, bounds, evals):
    best = min(range(len(fv)), key=lambda i: fv[i])
    return pop[best], {"pop": pop, "fvals": fv, "rng": rng, "bounds": bounds,
                       "evals": evals, "fmean": sum(fv) / len(fv)}


FUNC = ga_generation
HELPERS = [tournament, sbx, mutate]
HEADER = ""

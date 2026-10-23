"""Exercise: one generation of a real-coded genetic algorithm."""
from __future__ import annotations

from .. import demos
from ..demos import ga as DEMO
from ..grading import Exercise, StepCase
from ._rng import Draws

SKELETON = '''def tournament(pop, fvals, rng):
    """二元錦標賽：i = int(rng.random() * N)，再抽 j；回傳 f 較小者（同分取 i）。"""
    raise NotImplementedError


def sbx(a, b, rng, eta):
    """SBX：每個座標抽一個 u；u ≤ 0.5 時 β = (2u)^(1/(η+1))，否則 β = (1/(2(1−u)))^(1/(η+1))。
    子代 c1 = ½[(1+β)a + (1−β)b]、c2 = ½[(1−β)a + (1+β)b]。回傳 (c1, c2) 兩個 tuple。"""
    raise NotImplementedError


def mutate(x, rng, pm, eta, bounds):
    """多項式突變：每個座標先抽 r，r < pm 才突變：再抽 u，
    u < 0.5 時 δ = (2u)^(1/(η+1)) − 1，否則 δ = 1 − (2(1−u))^(1/(η+1))；x ← x + δ·(hi − lo)。
    最後每個座標都截回 [lo, hi]。回傳 tuple。"""
    raise NotImplementedError


def ga_generation(pop, fvals, f, rng, bounds, n_elite, pc, eta_c, pm, eta_m):
    """一代 GA。

    pop, fvals: 本代個體（長度 2 的 tuple 的 list）與 f 值
    回傳:       (new_pop, new_f)：先放 f 最小的 n_elite 個菁英（由好到差），
                再放子代，直到和 pop 一樣多；只對子代呼叫 f

    亂數順序：每一對父母依序 tournament(a)、tournament(b)、一個 r（r < pc 才 SBX，
    否則子代是父母的複本）、SBX 的亂數，再依序突變 c1、c2（多出來的 c2 不突變也不抽亂數）。
    """
    # TODO 1: 依 fvals 排序，前 n_elite 個原封不動放進下一代
    # TODO 2: 重複 (N − n_elite + 1) // 2 次：選兩個父母、交配、突變、算 f
    raise NotImplementedError
'''


def _sq(p):
    return p[0] ** 2 + p[1] ** 2


_LINE = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]

EXERCISE = Exercise(
    id="ga",
    func_name="ga_generation",
    skeleton=SKELETON,
    required=("tournament", "sbx", "mutate"),
    hints=[
        "錦標賽：抽兩個索引 i、j，回傳 f 較小的那個個體；f 越小越好（我們在做最小化）。",
        "SBX 的兩個子代以父母的中點為中心對稱：c1 + c2 = a + b。β < 1 時子代落在父母之間，β > 1 時落在外側。",
        "菁英：order = sorted(range(N), key=lambda i: fvals[i])，前 n_elite 個直接複製（f 值也直接沿用，不重算）。",
    ],
    solution=demos.source("ga"),
    blanks=("fvals[j] < fvals[i]", "0.5 * ((1 + beta) * ad + (1 - beta) * bd)",
            "(2 * u) ** (1 / (eta + 1)) - 1", "xd + delta * (hi - lo)", "order[:n_elite]"),
    cases=[
        StepCase("錦標賽：抽到 1 號（f = 1）和 2 號（f = 2），1 號勝", {
            "pop": _LINE, "fvals": [3.0, 1.0, 2.0, 0.0], "rng": Draws([0.3, 0.6])},
            ((1.0, 0.0),), func="tournament"),
        StepCase("錦標賽：抽到 0 號（f = 3）和 3 號（f = 0），3 號勝", {
            "pop": _LINE, "fvals": [3.0, 1.0, 2.0, 0.0], "rng": Draws([0.1, 0.9])},
            ((3.0, 0.0),), func="tournament"),
        StepCase("SBX，η = 0：u = 0.25 → β = 0.5（內側），u = 0.75 → β = 2（外側）", {
            "a": (0.0, 0.0), "b": (4.0, 2.0), "rng": Draws([0.25, 0.75]), "eta": 0},
            ((1.0, -1.0), (3.0, 3.0)), func="sbx"),
        StepCase("SBX，η = 1：u = 0.125 → β = 0.5；u = 0.5 → β = 1（子代 = 父母）", {
            "a": (0.0, 0.0), "b": (4.0, 2.0), "rng": Draws([0.125, 0.5]), "eta": 1},
            ((1.0, 0.0), (3.0, 2.0)), func="sbx"),
        StepCase("突變，η = 0、p_m = 1：δ = −0.5 與 +0.5，第二個座標超出範圍被截回 2", {
            "x": (1.0, 1.0), "rng": Draws([0.0, 0.25, 0.5, 0.75]), "pm": 1.0, "eta": 0,
            "bounds": ((-5.0, 5.0), (0.0, 2.0))}, ((-4.0, 2.0),), func="mutate"),
        StepCase("突變，p_m = 0.5：第一個座標 r = 0.7 不突變，第二個 r = 0.2 突變", {
            "x": (1.0, 1.0), "rng": Draws([0.7, 0.2, 0.25]), "pm": 0.5, "eta": 0,
            "bounds": ((-5.0, 5.0), (-1.0, 3.0))}, ((1.0, -1.0),), func="mutate"),
        StepCase("一整代：2 個菁英 + 一對子代（η_c = 0、p_c = 1、p_m = 0）", {
            "pop": _LINE, "fvals": [0.0, 1.0, 4.0, 9.0], "f": _sq,
            "rng": Draws([0.6, 0.9, 0.3, 0.1, 0.5, 0.25, 0.5, 0.3, 0.3, 0.3, 0.3]),
            "bounds": ((-5.0, 5.0), (-5.0, 5.0)), "n_elite": 2, "pc": 1.0, "eta_c": 0,
            "pm": 0.0, "eta_m": 20},
            ([(0.0, 0.0), (1.0, 0.0), (1.5, 0.0), (0.5, 0.0)], [0.0, 1.0, 2.25, 0.25])),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "ackley",
          "params": {"n_pop": 20, "n_elite": 2, "pc": 0.9, "eta_c": 15, "pm": 0.5,
                     "eta_m": 20, "seed": 0}, "n_iter": 100},
    converge_below=1e-2,
    loop_note="上方動畫先建立族群：個體 0 放在起點，其餘 N − 1 個以 random.Random(種子) 在圖的範圍內均勻撒點；"
              "之後每一步呼叫一次 ga_generation()，並把這一代 f 最小的個體當作路徑上的點。"
              "外層迴圈也計數函數計算次數（起始族群的 N 次也算），跑滿迭代數才停。",
)

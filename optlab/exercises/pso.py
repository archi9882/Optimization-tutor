"""Exercise: one particle-swarm iteration."""
from __future__ import annotations

from .. import demos
from ..demos import pso as DEMO
from ..grading import Exercise, StepCase
from ._rng import Draws

SKELETON = '''def pso_step(xs, vs, pbest, pbest_f, f, rng, w, c1, c2, bounds):
    """一次 PSO 迭代（全部粒子同步更新）。

    xs, vs:          每個粒子的位置、速度（長度 2 的 tuple 的 list）
    pbest, pbest_f:  每個粒子的個體最佳位置與它的 f 值
    f:               目標函數
    rng:             亂數產生器，只用 rng.random()（[0, 1) 均勻分布）
    w, c1, c2:       慣性、認知係數、社會係數
    bounds:          ((lo₁, hi₁), (lo₂, hi₂))
    回傳:            (new_xs, new_vs, new_pbest, new_pbest_f)，不要修改傳進來的 list

    亂數順序（評分會用固定的亂數檢查）：依粒子順序、每個座標依序抽 r1 再抽 r2。
    """
    # TODO 1: g = pbest 中 f 最小的位置（迭代開始時決定，整次迭代都用它）
    # TODO 2: v ← w·v + c1·r1·(p − x) + c2·r2·(g − x)；x ← x + v
    #         超出 [lo, hi] 就把 x 截在邊界、這個座標的速度設為 0
    # TODO 3: 算新位置的 f，比個體最佳好就更新 pbest
    raise NotImplementedError
'''


def _sphere(p):
    return p[0] ** 2 + p[1] ** 2


_B = ((-5.0, 5.0), (-5.0, 5.0))

EXERCISE = Exercise(
    id="pso",
    func_name="pso_step",
    skeleton=SKELETON,
    hints=[
        "群體最佳 g 是 pbest 裡 f 最小的那一個，不是 xs 裡的；它在這一次迭代中固定不變。",
        "每個粒子、每個座標：r1, r2 = rng.random(), rng.random()；"
        "v = w*v[d] + c1*r1*(p[d]-x[d]) + c2*r2*(g[d]-x[d])，再 x = x[d] + v。",
        "個體最佳只在新位置更好時更新：先複製 pbest、pbest_f，再逐一比較 f(x_new) < pbest_f[i]。",
    ],
    solution=demos.source("pso"),
    blanks=("w * v[d]", "c1 * r1 * (p[d] - x[d])", "c2 * r2 * (g[d] - x[d])",
            "min(max(xd, lo), hi)", "fx < new_pf[i]"),
    cases=[
        StepCase("三項都有：慣性、個體、群體（r 全為 0.5，w = 0.5、c₁ = 1、c₂ = 2）",
                 {"xs": [(0.0, 0.0), (2.0, 0.0)], "vs": [(1.0, 0.0), (0.0, 0.0)],
                  "pbest": [(1.0, 1.0), (2.0, 0.0)], "pbest_f": [2.0, 4.0], "f": _sphere,
                  "rng": Draws([0.5] * 8), "w": 0.5, "c1": 1.0, "c2": 2.0, "bounds": _B},
                 ([(2.0, 1.5), (1.0, 1.0)], [(2.0, 1.5), (-1.0, 1.0)],
                  [(1.0, 1.0), (1.0, 1.0)], [2.0, 2.0])),
        StepCase("亂數順序：每個座標先 r₁ 再 r₂（w = 0、c₁ = c₂ = 1）",
                 {"xs": [(2.0, 0.0), (0.0, 0.0)], "vs": [(0.0, 0.0), (0.0, 0.0)],
                  "pbest": [(2.0, 0.0), (0.0, 1.0)], "pbest_f": [4.0, 1.0], "f": _sphere,
                  "rng": Draws([0.1, 0.9, 0.3, 0.7, 0.2, 0.6, 0.4, 0.8]),
                  "w": 0.0, "c1": 1.0, "c2": 1.0, "bounds": _B},
                 ([(0.2, 0.7), (0.0, 1.2)], [(-1.8, 0.7), (0.0, 1.2)],
                  [(0.2, 0.7), (0.0, 1.0)], [0.53, 1.0])),
        StepCase("撞到邊界：x 停在 5、速度歸零，新位置較差所以個體最佳不變",
                 {"xs": [(4.5, 0.0)], "vs": [(1.0, 0.0)], "pbest": [(4.5, 0.0)],
                  "pbest_f": [20.25], "f": _sphere, "rng": Draws([0.5] * 4),
                  "w": 1.0, "c1": 0.0, "c2": 0.0, "bounds": _B},
                 ([(5.0, 0.0)], [(0.0, 0.0)], [(4.5, 0.0)], [20.25])),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "rastrigin",
          "params": {"n_pop": 20, "w": 0.72, "c1": 1.5, "c2": 1.5, "seed": 0}, "n_iter": 100},
    converge_below=1e-6,
    loop_note="上方動畫先建立粒子群：粒子 0 放在起點，其餘 N − 1 個以 random.Random(種子) 在圖的範圍內均勻撒點，"
              "速度全為 0；之後每一步呼叫一次 pso_step()，並把目前的個體最佳中最好的一個當作路徑上的點。"
              "外層迴圈也計數函數計算次數（起始粒子群的 N 次也算），跑滿迭代數才停。",
)

"""Exercise: one Frank-Wolfe step and the vertex oracle."""
from __future__ import annotations

from .. import demos
from ..demos import frank_wolfe as DEMO
from ..grading import Exercise, StepCase
from ..sets import lmo_polygon
from ._concheck import make_check, with_helpers

SKELETON = '''import math


def lmo_polygon(g, vertices):
    """線性最小化：回傳多邊形頂點中讓 gᵀv 最小的那一個。"""
    # TODO: 逐一比較每個頂點的 g[0]·v[0] + g[1]·v[1]
    raise NotImplementedError


def frank_wolfe_step(x, grad, k, lmo):
    """一步 Frank–Wolfe。

    x, grad: 長度 2 的 tuple（目前位置、該點梯度）
    k:       迭代編號（從 0 開始），步長 γ = 2/(k+2)
    lmo:     線性最小化 lmo(g) → 可行集合中讓 gᵀs 最小的點 s
    回傳:    (x_new, gap)，gap = gᵀ(x − s)
    """
    # TODO 1: s = lmo(grad)；gap = gradᵀ(x − s)
    # TODO 2: x_new = x + γ (s − x)
    raise NotImplementedError
'''

_TRI = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)]
_SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]


def _wire(ns, prob):
    """Use the learner's lmo_polygon on the problem's polygon."""
    verts = _TRI if prob.id == "fert_budget" else _SQUARE
    return with_helpers(ns, prob, lmo=lambda n: lambda g: n["lmo_polygon"](g, verts))


EXERCISE = Exercise(
    id="frank_wolfe",
    func_name="frank_wolfe_step",
    skeleton=SKELETON,
    hints=[
        "線性函數在多邊形上的最小值一定出現在頂點，所以 LMO 只要把每個頂點代進 gᵀv 比大小。",
        "gap = gᵀ(x − s) ≥ 0；凸函數時它是 f(x) − f* 的上界，可以當停止條件。",
        "x_new = (1 − γ)x + γs 是 x 和 s 的凸組合，所以永遠留在可行集合裡，不需要投影。",
    ],
    solution=demos.source("frank_wolfe"),
    blanks=("g[0] * v[0] + g[1] * v[1]", "lmo(grad)", "grad[0] * (x[0] - s[0])", "2.0 / (k + 2)"),
    cases=[
        StepCase("LMO：g = (1, −1) 時正方形上 gᵀv 最小的頂點是 (0, 1)",
                 {"g": (1.0, -1.0), "vertices": _SQUARE}, ((0.0, 1.0),), func="lmo_polygon"),
        StepCase("k = 0：γ = 1，直接跳到頂點 (0, 0)，gap = 1 + 2 = 3",
                 {"x": (1.0, 1.0), "grad": (1.0, 2.0), "k": 0,
                  "lmo": lambda g: lmo_polygon(g, _TRI)}, ((0.0, 0.0), 3.0)),
        StepCase("k = 2：γ = 1/2，走到 (1, 1) 與頂點 (0, 3) 的中點",
                 {"x": (1.0, 1.0), "grad": (-1.0, -2.0), "k": 2,
                  "lmo": lambda g: lmo_polygon(g, _TRI)}, ((0.5, 2.0), 3.0)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    required=("lmo_polygon",),
    demo={"problem": "fert_budget", "params": {}, "n_iter": 300},
    check=make_check(DEMO, "frank_wolfe_step", [
        {"problem": "fert_budget", "params": {}, "n_iter": 300, "dist": 1e-2, "viol": 1e-12, "path": True,
         "stop": False, "goal": "施肥預算：300 步後接近 x* = (1.5, 1.5)，而且每一步都可行"},
        {"problem": "reservoir", "params": {}, "n_iter": 300, "dist": 1e-2, "viol": 1e-12, "path": True,
         "stop": False, "goal": "兩座水庫（箱型）"}], wire=_wire),
    loop_note="上方動畫每一步呼叫一次 frank_wolfe_step()，lmo 由問題提供（多邊形用 lmo_polygon）；"
              "細線連到這一步選中的頂點 s。gap < 10⁻⁸ 時不再移動並停止。",
)

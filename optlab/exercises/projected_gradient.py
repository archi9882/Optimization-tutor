"""Exercise: one projected-gradient step plus the projections it relies on."""
from __future__ import annotations

import inspect

from .. import demos
from ..demos import projected_gradient as DEMO
from ..grading import Exercise, StepCase
from ..sets import project_disk, project_segment
from ._concheck import make_check, with_helpers

SKELETON = "import math\n\n\n" + inspect.getsource(project_segment) + '''

def project_disk(y, center, r):
    """投影到圓盤 ‖x − center‖ ≤ r；回傳長度 2 的 tuple。"""
    # TODO: 在圓內就不動；在圓外就沿 center → y 的方向拉回圓周
    raise NotImplementedError


def project_polygon(y, vertices):
    """投影到凸多邊形（頂點逆時針排列）；可以使用上面的 project_segment。"""
    # TODO 1: y 在每條邊的左側（含邊上）→ 在內部，不動
    # TODO 2: 否則回傳各邊最近點中離 y 最近的一個
    raise NotImplementedError


def projected_gradient_step(x, grad, lr, project):
    """一步投影梯度。

    x, grad: 長度 2 的 tuple（目前位置、該點梯度）
    lr:      步長 η
    project: 投影函數 project(y) → 可行集合中離 y 最近的點
    回傳:    x_new（一定可行）
    """
    # TODO: 先走一步梯度下降得到 y，再回傳 project(y)
    raise NotImplementedError
'''

_SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
_TRI = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)]


def _wire(ns, prob):
    """Use the learner's projection helpers (disk or polygon) on this problem."""
    if prob.id == "fert_disk":
        return with_helpers(ns, prob, project=lambda n: lambda y: n["project_disk"](y, (1.0, 1.0), 1.0))
    verts = _TRI if prob.id == "fert_budget" else _SQUARE
    return with_helpers(ns, prob, project=lambda n: lambda y: n["project_polygon"](y, verts))


EXERCISE = Exercise(
    id="projected_gradient",
    func_name="projected_gradient_step",
    skeleton=SKELETON,
    hints=[
        "投影梯度就是「先照常走梯度步，走出去了再拉回來」。拉回來的方式是找可行集合中最近的點。",
        "圓盤：點在圓外時，沿圓心到該點的方向縮回半徑 r：c + r·(y − c)/‖y − c‖。",
        "多邊形：頂點逆時針排列時，y 在每條邊 a→b 的左側（外積 (b − a) × (y − a) ≥ 0）就在內部，原地不動；否則對每條邊呼叫 project_segment，取離 y 最近的結果。",
    ],
    solution=demos.source("projected_gradient"),
    blanks=("r * d[0] / dist", "min(1.0, max(0.0, t))", "x[0] - lr * grad[0]", "project(y)"),
    cases=[
        StepCase("圓盤投影：(3, 4) 拉回單位圓 → (0.6, 0.8)",
                 {"y": (3.0, 4.0), "center": (0.0, 0.0), "r": 1.0}, ((0.6, 0.8),),
                 func="project_disk"),
        StepCase("圓盤投影：圓內的點 (0.2, 0.3) 不動",
                 {"y": (0.2, 0.3), "center": (0.0, 0.0), "r": 1.0}, ((0.2, 0.3),),
                 func="project_disk"),
        StepCase("三角形投影：(3, 2) 落在斜邊 x₁ + x₂ = 3 外側 → 垂足 (2, 1)",
                 {"y": (3.0, 2.0), "vertices": _TRI}, ((2.0, 1.0),), func="project_polygon"),
        StepCase("三角形投影：內部的點 (0.5, 0.5) 不動",
                 {"y": (0.5, 0.5), "vertices": _TRI}, ((0.5, 0.5),), func="project_polygon"),
        StepCase("三角形投影：(4, −1) 在頂點外側 → 頂點 (3, 0)",
                 {"y": (4.0, -1.0), "vertices": _TRI}, ((3.0, 0.0),), func="project_polygon"),
        StepCase("多邊形投影：(3, 0.5) → 右邊 x₁ = 2 上的 (2, 0.5)",
                 {"y": (3.0, 0.5), "vertices": _SQUARE}, ((2.0, 0.5),), func="project_polygon"),
        StepCase("多邊形投影：(3, 3) → 角落 (2, 1)",
                 {"y": (3.0, 3.0), "vertices": _SQUARE}, ((2.0, 1.0),), func="project_polygon"),
        StepCase("一步：y = (1.5, 0) 走出單位圓，投影回 (1, 0)",
                 {"x": (0.5, 0.0), "grad": (-10.0, 0.0), "lr": 0.1,
                  "project": lambda y: project_disk(y, (0.0, 0.0), 1.0)}, ((1.0, 0.0),)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    required=("project_disk", "project_polygon"),
    demo={"problem": "fert_disk", "params": {"lr": 0.1}, "n_iter": 200},
    check=make_check(DEMO, "projected_gradient_step", [
        {"problem": "fert_disk", "params": {"lr": 0.1}, "n_iter": 200, "dist": 1e-8,
         "viol": 1e-12, "path": True, "goal": "土壤養分圓盤：停在圓周上的 x*"},
        {"problem": "fert_budget", "params": {"lr": 0.1}, "n_iter": 200, "dist": 1e-8,
         "viol": 1e-12, "path": True, "goal": "施肥預算（三角形）：停在預算線上的 x*"},
        {"problem": "reservoir", "params": {"lr": 0.1}, "n_iter": 200, "dist": 1e-8,
         "viol": 1e-12, "path": True, "goal": "兩座水庫（箱型）：停在 x₂ = 1 這條邊上的 x*"}], wire=_wire),
    loop_note="上方動畫每一步呼叫一次 projected_gradient_step()，project 由問題提供"
              "（圓盤用 project_disk，多邊形用 project_polygon）；x 不再移動時停止。",
)

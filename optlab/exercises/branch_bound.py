"""Exercise: the decision made at each branch-and-bound node."""
from __future__ import annotations

import math

from .. import demos, ip, lp
from ..grading import Exercise, StepCase
from ._combocheck import compare

SKELETON = '''import math


def most_fractional(x):
    """回傳小數部分最接近 0.5 的變數編號；全部是整數時回傳 None。"""
    # TODO: 對每個 x_j 算它離最近整數多遠，取最遠的（且要大於 1e-6）
    raise NotImplementedError


def bb_decide(x, z, best_z):
    """處理一個節點的 LP 鬆弛解 (x, z)，回傳要做的事：
    ("infeasible",)、("prune",)、("incumbent",) 或 ("branch", j, ⌊x_j⌋, ⌊x_j⌋ + 1)。"""
    # TODO 1: x 是 None：LP 不可行
    # TODO 2: z 不比現有最佳 best_z 好：剪枝（最大化問題）
    # TODO 3: 全是整數：新的現有最佳解；否則對最不整數的變數分支
    raise NotImplementedError
'''


def check(ns: dict) -> tuple:
    """Run branch and bound with the learner's bb_decide (and their most_fractional)."""
    fin = lambda o: [o["steps"][-1]["incumbent"], o["steps"][-1]["best_z"]]  # noqa: E731
    seq = lambda o: [[s["node"], s["tree"][-1]["action"]] for s in o["steps"]]  # noqa: E731
    tr = lambda o: {"xs": [s["x"] for s in o["steps"] if s["x"]],  # noqa: E731
                    "fs": [s["z"] for s in o["steps"] if s["x"]], "status": o["status"]}
    results, traces = [], None
    for c in [(5.0, 4.0), (3.0, 4.0)]:
        prob = lp.greenhouse(*c)
        ref = ip.run_branch_bound(prob)
        user = ip.run_branch_bound(prob, ns["bb_decide"])
        res = compare(fin(user), fin(ref), seq(user), seq(ref), f"c = {c}：找到整數最優解",
                      fmt=lambda v: f"x = {v[0]}，z = {v[1]}")
        results.extend(res if traces is None else res[:1])
        traces = traces or (tr(user), tr(ref))
    return results, traces[0], traces[1]


EXERCISE = Exercise(
    id="branch_bound",
    func_name="bb_decide",
    required=("most_fractional",),
    skeleton=SKELETON,
    hints=[
        "順序很重要：先檢查不可行，再檢查界限，最後才看是不是整數。",
        "最大化問題中，LP 鬆弛的 z 是這個節點內任何整數解的上界；上界 ≤ 現有最佳就不可能更好。",
        "abs(v - round(v)) 是離最近整數的距離；分支值用 math.floor(x[j]) 與它加 1。",
    ],
    solution=demos.source("branch_bound"),
    blanks=("abs(v - round(v))", "z <= best_z + 1e-9", "math.floor(x[j])"),
    cases=[
        StepCase("最不整數的變數", {"x": (3.4, 1.5)}, (1,), func="most_fractional"),
        StepCase("全部是整數", {"x": (2.0, 3.0)}, (None,), func="most_fractional"),
        StepCase("LP 不可行", {"x": None, "z": None, "best_z": -math.inf}, (("infeasible",),)),
        StepCase("根節點：對 x₂ = 1.5 分支", {"x": (3.0, 1.5), "z": 21.0, "best_z": -math.inf},
                 (("branch", 1, 1, 2),)),
        StepCase("上界 18 不比現有 20 好", {"x": (2.0, 2.0), "z": 18.0, "best_z": 20.0}, (("prune",),)),
        StepCase("整數解且更好", {"x": (4.0, 0.0), "z": 20.0, "best_z": 19.0}, (("incumbent",),)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    lp_preset="greenhouse",
    loop_note="上方動畫每處理一個節點，就用頂點列舉解它的 LP 鬆弛，再呼叫一次 bb_decide() 決定剪枝、更新最佳解或分支；外層迴圈用堆疊管理待處理的節點（深度優先）。",
)

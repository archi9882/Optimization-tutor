"""Exercise: one Newton step on the log-barrier function."""
from __future__ import annotations

from .. import demos, lp
from ..grading import Exercise, StepCase
from ._lpcheck import as_trace, compare_runs

SKELETON = demos.helpers_source("barrier") + '''def barrier_newton_step(x, mu, c, A, b):
    """對 φ_μ(x) = −cᵀx − μ Σ log(b_i − a_iᵀx) 做一步牛頓法。

    x:    長度 2 的 tuple（必須嚴格在可行域內）
    mu:   障礙參數 μ
    c:    要最大化的目標係數
    A, b: 限制 A x ≤ b（已包含 x ≥ 0）
    回傳: x_new（仍在內部，而且 φ 下降）
    """
    # TODO 1: slack s_i = b_i − a_iᵀx
    # TODO 2: 梯度 g_j = −c_j + μ Σ a_ij / s_i；Hessian H_jk = μ Σ a_ij a_ik / s_i²
    # TODO 3: 解 H d = −g；步長 t 從 1 開始減半，直到仍在內部且 φ 下降（Armijo）
    raise NotImplementedError
'''

_A = [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0], [-1.0, 0.0], [0.0, -1.0]]
_b = [4.0, 12.0, 18.0, 0.0, 0.0]


def check(ns: dict) -> tuple:
    """Run the barrier method with the learner's Newton step."""
    prob = lp.farm()
    ref = lp.run_barrier(prob)
    user = lp.run_barrier(prob, ns["barrier_newton_step"])
    results = compare_runs(user, ref, prob.objective, "沿中心路徑逼近最優頂點 (2, 6)")
    return results, as_trace(user, prob.objective), as_trace(ref, prob.objective)


EXERCISE = Exercise(
    id="barrier",
    func_name="barrier_newton_step",
    skeleton=SKELETON,
    hints=[
        "−log(s) 在 s → 0 時趨近 +∞，像一道牆把 x 擋在可行域內；μ 越小牆越薄。",
        "log(b_i − a_iᵀx) 對 x_j 的導數是 −a_ij / s_i，所以 −μ Σ log(...) 的梯度是 +μ Σ a_ij / s_i。",
        "Hessian 是 μ Σ a_i a_iᵀ / s_i²：靠近邊界的限制（s_i 小）權重最大。",
    ],
    solution=demos.source("barrier"),
    blanks=("-c[j] + mu * sum(ai[j] / si for ai, si in zip(A, s))",
            "ai[j] * ai[k] / si ** 2", "bi - dot(ai, x)"),
    cases=[
        StepCase("μ = 10：被障礙推回中央", {"x": (1.0, 1.0), "mu": 10.0, "c": [3.0, 5.0],
                                         "A": _A, "b": _b},
                 ((1.5997704374442034, 2.057528801598435),)),
        StepCase("μ = 1：往最優頂點靠近", {"x": (1.0, 1.0), "mu": 1.0, "c": [3.0, 5.0],
                                       "A": _A, "b": _b},
                 ((2.395931641372274, 3.6075011690685708),)),
        StepCase("μ = 0.1：貼近頂點但不越界", {"x": (2.0, 5.0), "mu": 0.1, "c": [3.0, 5.0],
                                           "A": _A, "b": _b},
                 ((1.8683035714285712, 5.8351934523809526),)),
    ],
    adapter=None, reference=None, demo=None,
    check=check,
    lp_preset="farm",
    loop_note="上方動畫每個 μ 呼叫 barrier_newton_step() 三次，再把 μ 除以 4，共 9 輪；外層迴圈只負責管理 μ 與記錄軌跡。",
)

"""Exercise: one Adam step with bias correction."""
from __future__ import annotations

from .. import demos
from ..demos import adam as DEMO
from ..grading import Exercise, StepCase

SKELETON = '''def adam_step(x, m, s, grad, t, lr, beta1, beta2, eps=1e-8):
    """一步 Adam。

    x, m, s, grad: 長度 2 的 tuple（位置、一階動差、二階動差、梯度）
    t:             這是第幾步（第一次呼叫時 t = 1）
    lr, beta1, beta2: 學習率與兩個衰減係數
    回傳:          (x_new, m_new, s_new)
    步長：lr * m_hat_i / (sqrt(s_hat_i) + eps)
    """
    # TODO 1: m ← β1·m + (1−β1)·g ；s ← β2·s + (1−β2)·g²
    # TODO 2: 偏差修正 m_hat = m / (1 − β1^t)，s_hat = s / (1 − β2^t)
    # TODO 3: 用 m_hat、s_hat 更新 x（回傳未修正的 m、s）
    raise NotImplementedError
'''


EXERCISE = Exercise(
    id="adam",
    func_name="adam_step",
    skeleton=SKELETON,
    hints=[
        "Adam = 動量（m）+ RMSProp（s）。兩個都從 0 起算，所以前幾步會偏小，要除以 (1 − β^t) 補回來。",
        "虛擬碼：m ← β1·m + (1−β1)·g；s ← β2·s + (1−β2)·g²；m̂ = m/(1−β1ᵗ)；ŝ = s/(1−β2ᵗ)；x ← x − η·m̂/(√ŝ + ε)",
        "回傳的是未修正的 m_new、s_new；m_hat、s_hat 只用來算這一步的位移。",
    ],
    solution=demos.source("adam"),
    blanks=("(1 - beta1) * gi", "(1 - beta2) * gi ** 2", "1 - beta1 ** t", "1 - beta2 ** t"),
    cases=[
        StepCase("第一步（t = 1）：每個座標走 η",
                 {"x": (1.0, 1.0), "m": (0.0, 0.0), "s": (0.0, 0.0),
                  "grad": (0.5, -2.0), "t": 1, "lr": 0.1, "beta1": 0.9,
                  "beta2": 0.999},
                 ((0.9, 1.1), (0.05, -0.2), (0.00025, 0.004))),
        StepCase("梯度不變的第二步仍走 η",
                 {"x": (0.9, 1.1), "m": (0.05, -0.2), "s": (0.00025, 0.004),
                  "grad": (0.5, -2.0), "t": 2, "lr": 0.1, "beta1": 0.9,
                  "beta2": 0.999},
                 ((0.8, 1.2), (0.095, -0.38), (0.00049975, 0.007996))),
        StepCase("梯度為 0 時仍靠動量前進",
                 {"x": (0.0, 0.0), "m": (0.2, 0.1), "s": (0.01, 0.04),
                  "grad": (0.0, 0.0), "t": 3, "lr": 0.1, "beta1": 0.9,
                  "beta2": 0.999},
                 ((-0.03638010194, -0.00909502551), (0.18, 0.09),
                  (0.00999, 0.03996))),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "quadratic", "problem_params": {"kappa": 10},
          "params": {"lr": 0.1, "beta1": 0.9, "beta2": 0.999}, "n_iter": 300},
    converge_below=1e-6,
)

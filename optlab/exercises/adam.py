"""Exercise: one Adam step with bias correction."""
from __future__ import annotations

import math

from ..grading import Exercise, StepCase, as_vec

SKELETON = '''import math

def adam_step(x, m, s, grad, t, lr, beta1, beta2, eps=1e-8):
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

SOLUTION = '''import math

def adam_step(x, m, s, grad, t, lr, beta1, beta2, eps=1e-8):
    m_new = tuple(beta1 * mi + (1 - beta1) * gi for mi, gi in zip(m, grad))
    s_new = tuple(beta2 * si + (1 - beta2) * gi ** 2 for si, gi in zip(s, grad))
    m_hat = [mi / (1 - beta1 ** t) for mi in m_new]
    s_hat = [si / (1 - beta2 ** t) for si in s_new]
    x_new = tuple(xi - lr * mh / (math.sqrt(sh) + eps)
                  for xi, mh, sh in zip(x, m_hat, s_hat))
    return x_new, m_new, s_new
'''


def reference(x, m, s, grad, t, lr, beta1, beta2, eps=1e-8):
    """Reference implementation in the learner's signature."""
    m_new = tuple(beta1 * mi + (1 - beta1) * gi for mi, gi in zip(m, grad))
    s_new = tuple(beta2 * si + (1 - beta2) * gi ** 2 for si, gi in zip(s, grad))
    m_hat = [mi / (1 - beta1 ** t) for mi in m_new]
    s_hat = [si / (1 - beta2 ** t) for si in s_new]
    x_new = tuple(xi - lr * mh / (math.sqrt(sh) + eps)
                  for xi, mh, sh in zip(x, m_hat, s_hat))
    return x_new, m_new, s_new


def adapter(fn):
    """Wrap a learner-style adam_step into the shared StepFn signature."""
    def step(x, problem, state, p):
        t = state.get("t", 0) + 1
        m = state.get("m", (0.0, 0.0))
        s = state.get("s", (0.0, 0.0))
        x_new, m_new, s_new = fn(x, m, s, problem.grad(x), t, p["lr"],
                                 p["beta1"], p["beta2"])
        return as_vec(x_new), {"t": t, "m": as_vec(m_new), "s": as_vec(s_new)}
    return step


EXERCISE = Exercise(
    id="adam",
    func_name="adam_step",
    skeleton=SKELETON,
    hints=[
        "Adam = 動量（m）+ RMSProp（s）。兩個都從 0 起算，所以前幾步會偏小，要除以 (1 − β^t) 補回來。",
        "虛擬碼：m ← β1·m + (1−β1)·g；s ← β2·s + (1−β2)·g²；m̂ = m/(1−β1ᵗ)；ŝ = s/(1−β2ᵗ)；x ← x − η·m̂/(√ŝ + ε)",
        "回傳的是未修正的 m_new、s_new；m_hat、s_hat 只用來算這一步的位移。",
    ],
    solution=SOLUTION,
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
    adapter=adapter,
    reference=reference,
    demo={"problem": "quadratic", "problem_params": {"kappa": 10},
          "params": {"lr": 0.1, "beta1": 0.9, "beta2": 0.999}, "n_iter": 300},
    converge_below=1e-6,
)

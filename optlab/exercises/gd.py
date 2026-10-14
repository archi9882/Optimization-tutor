"""Exercise: one gradient-descent step."""
from __future__ import annotations

from ..grading import Exercise, StepCase, as_vec

SKELETON = '''def gd_step(x, grad, lr):
    """一步梯度下降。

    x, grad: 長度 2 的 tuple（目前位置、該點梯度）
    lr:      學習率 η
    回傳:    新位置 x_new（長度 2 的 tuple）
    """
    # TODO: 沿負梯度方向走 lr 倍的距離
    raise NotImplementedError
'''

SOLUTION = '''def gd_step(x, grad, lr):
    return (x[0] - lr * grad[0], x[1] - lr * grad[1])
'''


def reference(x, grad, lr):
    """Reference implementation in the learner's signature."""
    return (x[0] - lr * grad[0], x[1] - lr * grad[1])


def adapter(fn):
    """Wrap a learner-style gd_step into the shared StepFn signature."""
    def step(x, problem, state, p):
        return as_vec(fn(x, problem.grad(x), p["lr"])), state
    return step


EXERCISE = Exercise(
    id="gd",
    func_name="gd_step",
    skeleton=SKELETON,
    hints=[
        "梯度指向「上坡最陡」的方向，所以要往它的反方向走。",
        "虛擬碼：x_new = x − η · ∇f(x)，兩個分量分開算。",
        "第一個分量：x[0] - lr * grad[0]",
    ],
    solution=SOLUTION,
    cases=[
        StepCase("一般情況", {"x": (1.0, 2.0), "grad": (0.5, -1.0),
                          "lr": 0.1}, ((0.95, 2.1),)),
        StepCase("η = 0 不動", {"x": (3.0, -4.0), "grad": (10.0, 10.0),
                             "lr": 0.0}, ((3.0, -4.0),)),
        StepCase("從原點出發", {"x": (0.0, 0.0), "grad": (2.0, 4.0),
                           "lr": 0.5}, ((-1.0, -2.0),)),
    ],
    adapter=adapter,
    reference=reference,
    demo={"problem": "quadratic", "problem_params": {"kappa": 10},
          "params": {"lr": 0.15}, "n_iter": 300},
)

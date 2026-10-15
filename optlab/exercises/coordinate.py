"""Exercise: one cyclic coordinate-descent (Gauss-Seidel / SOR) step."""
from __future__ import annotations

from ..first_order import coordinate_curvature
from ..grading import Exercise, StepCase, as_vec

SKELETON = '''def coordinate_step(x, grad, curv, omega, i):
    """一步循環座標下降：只更新第 i 個座標。

    x, grad: 長度 2 的 tuple（位置、該點梯度）
    curv:    長度 2 的 tuple，curv[i] = 沿第 i 軸的曲率 ∂²f/∂x_i²
    omega:   鬆弛係數 ω（1 = Gauss-Seidel，>1 = SOR 過鬆弛）
    i:       這一步要更新的座標（0 或 1）
    回傳:    x_new（另一個座標保持不變）
    """
    # TODO: x_i ← x_i − ω · g_i / curv_i，其他座標不動
    raise NotImplementedError
'''

SOLUTION = '''def coordinate_step(x, grad, curv, omega, i):
    x_new = list(x)
    x_new[i] = x[i] - omega * grad[i] / curv[i]
    return tuple(x_new)
'''


def reference(x, grad, curv, omega, i):
    """Reference implementation in the learner's signature."""
    x_new = list(x)
    x_new[i] = x[i] - omega * grad[i] / curv[i]
    return tuple(x_new)


def adapter(fn):
    """Cycle i = 0, 1, 0, 1, ... exactly like optlab's coordinate descent."""
    def step(x, problem, state, p):
        k = state.get("k", 0)
        curv = (coordinate_curvature(problem, x, 0),
                coordinate_curvature(problem, x, 1))
        x_new = fn(x, problem.grad(x), curv, p["omega"], k % 2)
        return as_vec(x_new), {"k": k + 1}
    return step


EXERCISE = Exercise(
    id="coordinate",
    func_name="coordinate_step",
    skeleton=SKELETON,
    hints=[
        "固定其他變數，沿第 i 軸看，f 是一條一維拋物線；ω = 1 時這一步正好跳到它的最低點。",
        "虛擬碼：x_i ← x_i − ω · g_i / curv_i，另一個分量原封不動。",
        "tuple 不能改值：先 x_new = list(x)，改完 x_new[i] 再 tuple(x_new)。",
    ],
    solution=SOLUTION,
    cases=[
        StepCase("更新第 0 個座標",
                 {"x": (2.0, 3.0), "grad": (4.0, 6.0), "curv": (2.0, 3.0),
                  "omega": 1.0, "i": 0},
                 ((0.0, 3.0),)),
        StepCase("更新第 1 個座標",
                 {"x": (2.0, 3.0), "grad": (4.0, 6.0), "curv": (2.0, 3.0),
                  "omega": 1.0, "i": 1},
                 ((2.0, 1.0),)),
        StepCase("ω = 1.5 過鬆弛",
                 {"x": (1.0, 1.0), "grad": (2.0, 0.0), "curv": (4.0, 1.0),
                  "omega": 1.5, "i": 0},
                 ((0.25, 1.0),)),
    ],
    adapter=adapter,
    reference=reference,
    demo={"problem": "quadratic",
          "problem_params": {"kappa": 10, "theta_deg": 30},
          "params": {"omega": 1.0}, "n_iter": 300},
)

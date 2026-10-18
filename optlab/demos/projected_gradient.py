"""Projected gradient demo: a gradient step, then the nearest feasible point."""
from __future__ import annotations

import math

from ..problems import as_vec
from ..sets import project_disk, project_polygon, project_segment

PSEUDOCODE = """\
P_C(y) = 可行集合 C 中離 y 最近的點          # 圓盤：拉回圓周；多邊形：最近的邊
for k = 0, 1, 2, …:
    y ← x_k − η ∇f(x_k)                       # 普通的梯度步，可能走出 C
    x_{k+1} ← P_C(y)                           # 投影回 C：迭代點永遠可行
停止：x_{k+1} = x_k（投影把梯度步完全抵銷，就是 KKT 點）
"""


def projected_gradient_step(x, grad, lr, project):
    """一步投影梯度：先沿負梯度走 lr 倍，再投影回可行集合。"""
    y = (x[0] - lr * grad[0], x[1] - lr * grad[1])
    return project(y)


STOP_MOVE = 1e-12


def adapter(fn):
    """Lab loop: the problem supplies its projection (disk or polygon, using the
    helpers shown above). Stop when the step no longer moves x."""
    def step(x, problem, state, p):
        if problem.project is None:
            return x, {"stop": True, "message": "這個問題沒有提供投影"}
        x_new = as_vec(fn(x, problem.grad(x), p["lr"], problem.project))
        move = math.dist(x, x_new)
        # Gradient mapping G = (x − x_new)/η: zero exactly at a KKT point.
        info = {"‖xₖ₊₁ − xₖ‖ / η": move / p["lr"]}
        return x_new, {"info": info, "stop": move < STOP_MOVE}
    return step


FUNC = projected_gradient_step
HELPERS = [project_disk, project_segment, project_polygon]
HEADER = "import math\n\n\n"

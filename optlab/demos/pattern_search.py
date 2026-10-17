"""Compass (coordinate pattern) search demo: poll +-h along each axis."""
from __future__ import annotations

from ..problems import as_vec

PSEUDOCODE = """\
h ← h₀
for k = 0, 1, 2, …:
    依序試 x + h·d，d ∈ {+e₁, −e₁, +e₂, −e₂}   # 輪詢（poll）
    第一個讓 f 變小的點 → 移過去，h 不變
    四個都沒有改善 → 留在原地，h ← h/2          # 網格加密
    h < 容許值 → 停止
"""

DIRECTIONS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))


def compass_step(x, fx, step, f):
    """一次輪詢：沿 ±x、±y 各試 step 遠，有改善就移過去；全部失敗就把 step 減半。"""
    for d in DIRECTIONS:
        y = (x[0] + step * d[0], x[1] + step * d[1])
        fy = f(y)
        if fy < fx:
            return y, fy, step              # 成功：移動，步長不變
    return x, fx, step / 2                  # 失敗：留在原地，步長減半


def stencil(x, step):
    """這一輪要試的四個點（畫在圖上的菱形）。"""
    return [[x[0] + step * d[0], x[1] + step * d[1]] for d in (
        DIRECTIONS[0], DIRECTIONS[2], DIRECTIONS[1], DIRECTIONS[3])]


STOP_STEP = 1e-8


def adapter(fn):
    """Lab loop: carry f(x) and the step, count evaluations, stop when
    the step is below STOP_STEP."""
    def step(x, problem, state, p):
        calls = [0]

        def f(v):
            calls[0] += 1
            return problem.f(v)

        h = state.get("step", p["step0"])
        fx = state["fx"] if "fx" in state else f(x)
        x_new, f_new, h_new = fn(x, fx, h, f)
        x_new = as_vec(x_new)
        return x_new, {
            "fx": float(f_new), "step": float(h_new),
            "evals": state.get("evals", 0) + calls[0],
            "poly_used": stencil(x, h), "poly": stencil(x_new, h_new),
            "stop": h_new < STOP_STEP}
    return step


FUNC = compass_step
HELPERS: list = []
HEADER = ("DIRECTIONS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))"
          "   # +e₁, −e₁, +e₂, −e₂\n\n\n")

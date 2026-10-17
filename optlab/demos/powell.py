"""Powell's conjugate-direction method with a golden-section line search."""
from __future__ import annotations

import math

from ..problems import as_vec

PSEUDOCODE = """\
方向集合 D ← {e₁, …, e_n}
for k = 0, 1, 2, …:                         # 一輪
    x_start ← x
    for 每個方向 d_i ∈ D:
        x ← argmin_t f(x + t·d_i)            # 一維搜尋（夾區間 + 黃金分割）
        記下這一次的下降量 Δ_i
    u ← (x − x_start) / ‖x − x_start‖        # 這一輪的總位移方向
    從 D 刪掉下降最多的 d_i，把 u 加到最後
    x ← argmin_t f(x + t·u)
"""


def bracket(phi, f0, h=0.1):
    """從 t = 0 往下坡走、步長每次加倍，直到函數值回升；回傳夾住極小值的區間。"""
    fh = phi(h)
    if fh >= f0:                         # +t 方向是上坡
        fm = phi(-h)
        if fm >= f0:
            return -h, h                 # 兩側都比較高：極小值在 [−h, h]
        h, fh = -h, fm                   # 改往 −t 方向找
    t_prev, t, ft = 0.0, h, fh
    for _ in range(60):
        t_next = 2 * t
        f_next = phi(t_next)
        if f_next >= ft:                 # 回升了：極小值在 t_prev 與 t_next 之間
            break
        t_prev, t, ft = t, t_next, f_next
    return min(t_prev, t_next), max(t_prev, t_next)


def golden_section(phi, a, b, tol=1e-8):
    """黃金分割搜尋：每次丟掉區間的 38.2%，保留夾住極小值的那一段。"""
    r = (math.sqrt(5) - 1) / 2           # 0.618…
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = phi(c), phi(d)
    while b - a > tol:
        if fc < fd:                      # 極小值在 [a, d]
            b, d, fd = d, c, fc
            c = b - r * (b - a)
            fc = phi(c)
        else:                            # 極小值在 [c, b]
            a, c, fc = c, d, fd
            d = a + r * (b - a)
            fd = phi(d)
    return (a + b) / 2


def line_min(f, x, fx, d):
    """沿方向 d 的一維最小化；找不到更低的點就留在原地。"""
    def phi(t):
        return f((x[0] + t * d[0], x[1] + t * d[1]))
    a, b = bracket(phi, fx)
    t = golden_section(phi, a, b)
    x_new = (x[0] + t * d[0], x[1] + t * d[1])
    f_new = f(x_new)
    return (x_new, f_new) if f_new < fx else (x, fx)


def powell_step(x, fx, dirs, f):
    """一輪 Powell：沿每個方向做一維最小化，再用這一輪的總位移取代下降最多的方向。"""
    x_start = x
    biggest, k_big = 0.0, 0
    for k, d in enumerate(dirs):
        x_new, f_new = line_min(f, x, fx, d)
        if fx - f_new > biggest:
            biggest, k_big = fx - f_new, k
        x, fx = x_new, f_new
    move = (x[0] - x_start[0], x[1] - x_start[1])
    length = math.hypot(move[0], move[1])
    if length > 0:
        u = (move[0] / length, move[1] / length)
        dirs = dirs[:k_big] + dirs[k_big + 1:] + [u]
        x, fx = line_min(f, x, fx, u)
    return x, fx, dirs


STOP_MOVE = 1e-8


def adapter(fn):
    """Lab loop: one call = one sweep. Carry f(x) and the direction set,
    count evaluations, stop once a sweep moves x less than STOP_MOVE."""
    def step(x, problem, state, p):
        calls = [0]

        def f(v):
            calls[0] += 1
            return problem.f(v)

        dirs = state.get("dirs", [(1.0, 0.0), (0.0, 1.0)])
        fx = state["fx"] if "fx" in state else f(x)
        x_new, f_new, new_dirs = fn(x, fx, list(dirs), f)
        x_new = as_vec(x_new)
        new_dirs = [as_vec(d) for d in new_dirs]
        arm = 0.12 * (problem.bounds[1] - problem.bounds[0])

        def fan(pt, ds):
            # Current directions drawn as two arms from the point.
            return [[pt[0] + arm * ds[0][0], pt[1] + arm * ds[0][1]], list(pt),
                    [pt[0] + arm * ds[1][0], pt[1] + arm * ds[1][1]]]

        return x_new, {
            "fx": float(f_new), "dirs": new_dirs,
            "evals": state.get("evals", 0) + calls[0],
            "poly_used": fan(x, dirs), "poly": fan(x_new, new_dirs),
            "poly_kind": "open",
            "stop": math.dist(x, x_new) < STOP_MOVE}
    return step


FUNC = powell_step
HELPERS = [bracket, golden_section, line_min]
HEADER = "import math\n\n\n"

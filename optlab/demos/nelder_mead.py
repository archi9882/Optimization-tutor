"""Nelder-Mead simplex demo: reflection, expansion, contraction and shrink."""
from __future__ import annotations

from ..problems import as_vec

PSEUDOCODE = """\
單純形：n+1 個頂點，依 f 排序 f(x₁) ≤ … ≤ f(x_{n+1})
c ← 除最差點 x_{n+1} 外的重心
x_r ← c + (c − x_{n+1})                          # 反射（α = 1）
若 f(x_r) < f(x₁)：
    x_e ← c + 2(c − x_{n+1})，x_e、x_r 取較好者  # 擴張（γ = 2）
否則若 f(x_r) < f(x_n)：用 x_r 換掉最差點
否則：                                            # 收縮（ρ = 0.5）
    若 f(x_r) < f(x_{n+1})：x_c ← c + 0.5(c − x_{n+1})，f(x_c) ≤ f(x_r) 才接受
    否則：               x_c ← c − 0.5(c − x_{n+1})，f(x_c) < f(x_{n+1}) 才接受
    都不接受：除 x₁ 外每點 x_i ← x₁ + 0.5(x_i − x₁)   # 縮小（σ = 0.5）
"""


def along(c, w, t):
    """從重心 c 沿「最差點 w → c」的方向走 t 倍：t = 1 反射、2 擴張、±0.5 收縮。"""
    return tuple(ci + t * (ci - wi) for ci, wi in zip(c, w))


def nelder_mead_step(simplex, fvals, f):
    """一次 Nelder–Mead 迭代：換掉最差頂點，或把整個單純形往最好點縮小。"""
    order = sorted(range(len(simplex)), key=lambda i: fvals[i])
    pts = [simplex[i] for i in order]                  # 由好到差
    fs = [fvals[i] for i in order]
    best, worst = pts[0], pts[-1]
    n = len(pts) - 1
    c = tuple(sum(p[j] for p in pts[:-1]) / n for j in range(len(best)))
    xr = along(c, worst, 1.0)                          # 反射
    fr = f(xr)
    if fr < fs[0]:                                     # 比最好點還好：試著走更遠
        xe = along(c, worst, 2.0)
        fe = f(xe)
        pts[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
    elif fr < fs[-2]:                                  # 比次差點好：接受反射
        pts[-1], fs[-1] = xr, fr
    else:
        if fr < fs[-1]:                                # 外側收縮
            xc = along(c, worst, 0.5)
            fc = f(xc)
            accept = fc <= fr
        else:                                          # 內側收縮
            xc = along(c, worst, -0.5)
            fc = f(xc)
            accept = fc < fs[-1]
        if accept:
            pts[-1], fs[-1] = xc, fc
        else:                                          # 縮小：全部往最好點靠近一半
            pts = [best] + [tuple(b + 0.5 * (p - b) for b, p in zip(best, q))
                            for q in pts[1:]]
            fs = [fs[0]] + [f(q) for q in pts[1:]]
    return pts, fs


def initial_simplex(x0, size):
    """起點 x0 加上沿兩個座標軸各走 size 的兩點。"""
    return [tuple(x0), (x0[0] + size, x0[1]), (x0[0], x0[1] + size)]


def diameter(simplex):
    """單純形最長的邊長（停止條件用）。"""
    return max(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
               for p in simplex for q in simplex)


STOP_SIZE = 1e-8


def adapter(fn):
    """Lab loop: carry the simplex and its f values, count evaluations,
    and stop once the simplex is smaller than STOP_SIZE."""
    def step(x, problem, state, p):
        calls = [0]

        def f(v):
            calls[0] += 1
            return problem.f(v)

        if "simplex" in state:
            simplex, fvals = state["simplex"], state["fvals"]
        else:
            simplex = initial_simplex(x, p["size"])
            fvals = [f(v) for v in simplex]
        new, new_f = fn(simplex, fvals, f)
        new = [as_vec(v) for v in new]
        new_f = [float(v) for v in new_f]
        best = min(range(len(new)), key=lambda i: new_f[i])
        return new[best], {
            "simplex": new, "fvals": new_f,
            "evals": state.get("evals", 0) + calls[0],
            "poly_used": [list(v) for v in simplex],
            "poly": [list(v) for v in new],
            "stop": diameter(new) < STOP_SIZE}
    return step


FUNC = nelder_mead_step
HELPERS = [along]
HEADER = ""

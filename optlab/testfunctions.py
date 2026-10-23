"""Eight classic 2D test functions for global (population) methods.

Standard textbook definitions, written out by hand; every listed minimum is
checked numerically in tests/test_testfunctions.py (dense grid plus local
refinement). Each problem has two boxes:

  bounds  the view drawn on the page; population methods also scatter their
          initial points in it and keep every point inside it
  domain  the usual search domain quoted with the function (larger than the
          view for Ackley, Bohachevsky and Griewank, whose ripples would be
          invisible at that scale)

`minima` lists every global minimum (f = fstar); `local_minima` lists a few
notable local minima nearest to it (location rounded to 6 decimals, refined
numerically). Gradients are analytic where f is differentiable; on a kink
(|.| at zero, Ackley's cone tip at the origin) the sign of zero is taken as
0. Hessians are central differences of the gradient (only Newton-type
methods use them, and none is offered on these problems).
"""
from __future__ import annotations

import math

from .problems import Problem, Vec

PI = math.pi
SQ2 = math.sqrt(2.0)


def _sign(v: float) -> float:
    return (v > 0) - (v < 0)


def _fd_hess(grad):
    """Hessian by central differences of the gradient (h = 1e-5)."""
    def hess(x: Vec) -> tuple[Vec, Vec]:
        h = 1e-5
        gx1, gx0 = grad((x[0] + h, x[1])), grad((x[0] - h, x[1]))
        gy1, gy0 = grad((x[0], x[1] + h)), grad((x[0], x[1] - h))
        a = (gx1[0] - gx0[0]) / (2 * h)
        b = 0.5 * ((gx1[1] - gx0[1]) + (gy1[0] - gy0[0])) / (2 * h)
        d = (gy1[1] - gy0[1]) / (2 * h)
        return ((a, b), (b, d))
    return hess


def _sym4(x: float, y: float) -> list[Vec]:
    """(±x, ±y) without duplicates (x or y may be 0)."""
    out: list[Vec] = []
    for sx in (1, -1):
        for sy in (1, -1):
            p = (sx * x + 0.0, sy * y + 0.0)
            if p not in out:
                out.append(p)
    return out


def sphere() -> Problem:
    """f = x1^2 + x2^2: the convex baseline every method should solve."""
    return Problem(
        id="sphere", name="Sphere 球面",
        f=lambda x: x[0] ** 2 + x[1] ** 2,
        grad=lambda x: (2 * x[0], 2 * x[1]),
        hess=lambda x: ((2.0, 0.0), (0.0, 2.0)),
        bounds=(-5.12, 5.12, -5.12, 5.12), domain=(-5.12, 5.12, -5.12, 5.12),
        x0=(3.5, 2.5), minima=[(0.0, 0.0)], fstar=0.0, test_function=True,
        note="凸、等向的碗：對照組，任何方法都該解得掉。")


def ackley() -> Problem:
    """Ackley (a = 20, b = 0.2, c = 2 pi): a funnel covered with small dimples."""
    def f(x: Vec) -> float:
        r = math.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
        cs = 0.5 * (math.cos(2 * PI * x[0]) + math.cos(2 * PI * x[1]))
        return -20 * math.exp(-0.2 * r) - math.exp(cs) + math.e + 20

    def grad(x: Vec) -> Vec:
        r = math.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
        a = 2 * math.exp(-0.2 * r) / r if r > 0 else 0.0      # cone tip: take 0
        e = math.exp(0.5 * (math.cos(2 * PI * x[0]) + math.cos(2 * PI * x[1])))
        return (a * x[0] + PI * math.sin(2 * PI * x[0]) * e,
                a * x[1] + PI * math.sin(2 * PI * x[1]) * e)

    return Problem(
        id="ackley", name="Ackley", f=f, grad=grad, hess=_fd_hess(grad),
        bounds=(-5.0, 5.0, -5.0, 5.0), domain=(-32.768, 32.768, -32.768, 32.768),
        x0=(3.3, 2.6), minima=[(0.0, 0.0)], fstar=0.0, test_function=True,
        local_minima=_sym4(0.952167, 0.0) + _sym4(0.0, 0.952167) + _sym4(0.968478, 0.968478),
        note="整體是往原點的漏斗，表面布滿小凹坑；原點是尖點（不可微）。")


def levy() -> Problem:
    """Levy function in 2D with w_i = 1 + (x_i - 1) / 4."""
    def f(x: Vec) -> float:
        w1, w2 = 1 + (x[0] - 1) / 4, 1 + (x[1] - 1) / 4
        return (math.sin(PI * w1) ** 2
                + (w1 - 1) ** 2 * (1 + 10 * math.sin(PI * w1 + 1) ** 2)
                + (w2 - 1) ** 2 * (1 + math.sin(2 * PI * w2) ** 2))

    def grad(x: Vec) -> Vec:
        w1, w2 = 1 + (x[0] - 1) / 4, 1 + (x[1] - 1) / 4
        s = math.sin(PI * w1 + 1)
        d1 = (PI * math.sin(2 * PI * w1) + 2 * (w1 - 1) * (1 + 10 * s * s)
              + (w1 - 1) ** 2 * 10 * PI * math.sin(2 * (PI * w1 + 1)))
        d2 = (2 * (w2 - 1) * (1 + math.sin(2 * PI * w2) ** 2)
              + (w2 - 1) ** 2 * 2 * PI * math.sin(4 * PI * w2))
        return (d1 / 4, d2 / 4)

    return Problem(
        id="levy", name="Levy", f=f, grad=grad, hess=_fd_hess(grad),
        bounds=(-10.0, 10.0, -10.0, 10.0), domain=(-10.0, 10.0, -10.0, 10.0),
        x0=(-6.0, 7.0), minima=[(1.0, 1.0)], fstar=0.0, test_function=True,
        local_minima=[(1.0, -2.89065), (1.0, 4.89065), (3.791082, 1.0), (-4.205435, 1.0)],
        note="沿兩個座標方向各有一排排的谷，谷底高度慢慢升高。")


CROSS_T = 1.349406537932761      # x1 = x2 on the diagonal, golden-section refined


def cross_in_tray() -> Problem:
    """Cross-in-tray: four global minima near (±1.3494, ±1.3494), f* ≈ -2.06261."""
    def f(x: Vec) -> float:
        e = math.exp(abs(100 - math.hypot(x[0], x[1]) / PI))
        return -1e-4 * (abs(math.sin(x[0]) * math.sin(x[1]) * e) + 1) ** 0.1

    def grad(x: Vec) -> Vec:
        r = math.hypot(x[0], x[1])
        e = math.exp(abs(100 - r / PI))
        s = math.sin(x[0]) * math.sin(x[1])
        a = abs(s) * e
        k = -1e-5 * (a + 1) ** -0.9
        de = -_sign(100 - r / PI) / (PI * r) if r > 0 else 0.0   # d|100 - r/pi| / dx_i = de * x_i
        sg = _sign(s)
        return (k * (sg * math.cos(x[0]) * math.sin(x[1]) * e + a * de * x[0]),
                k * (sg * math.sin(x[0]) * math.cos(x[1]) * e + a * de * x[1]))

    xs = _sym4(CROSS_T, CROSS_T)
    return Problem(
        id="cross_in_tray", name="Cross-in-tray 十字托盤", f=f, grad=grad, hess=_fd_hess(grad),
        bounds=(-10.0, 10.0, -10.0, 10.0), domain=(-10.0, 10.0, -10.0, 10.0),
        x0=(-6.0, 7.0), minima=xs, fstar=f(xs[0]), test_function=True,
        local_minima=_sym4(4.419076, 1.470622) + _sym4(1.470622, 4.419076),
        note="四個一樣深的全域最小值；兩條座標軸是 f 的「脊」（sin 為 0），把平面切成格子。")


def bukin6() -> Problem:
    """Bukin N.6: f = 100 sqrt(|x2 - 0.01 x1^2|) + 0.01 |x1 + 10|, min at (-10, 1)."""
    def f(x: Vec) -> float:
        return 100 * math.sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10)

    def grad(x: Vec) -> Vec:
        u = x[1] - 0.01 * x[0] ** 2
        du = 50 * _sign(u) / math.sqrt(abs(u)) if u != 0 else 0.0
        return (du * (-0.02 * x[0]) + 0.01 * _sign(x[0] + 10), du)

    return Problem(
        id="bukin6", name="Bukin N.6", f=f, grad=grad, hess=_fd_hess(grad),
        bounds=(-15.0, -5.0, -3.0, 3.0), domain=(-15.0, -5.0, -3.0, 3.0),
        x0=(-7.0, -2.0), minima=[(-10.0, 1.0)], fstar=0.0, test_function=True,
        note="谷底是拋物線 x₂ = 0.01x₁²，谷壁是 √|·| 的尖角（不可微）；沿谷底 f 只以 0.01|x₁+10| 的坡度變化。")


def rastrigin() -> Problem:
    """Rastrigin (A = 10): f = 20 + sum(x_i^2 - 10 cos(2 pi x_i))."""
    def f(x: Vec) -> float:
        return 20 + sum(v * v - 10 * math.cos(2 * PI * v) for v in x)

    def grad(x: Vec) -> Vec:
        return tuple(2 * v + 20 * PI * math.sin(2 * PI * v) for v in x)

    def hess(x: Vec) -> tuple[Vec, Vec]:
        h = [2 + 40 * PI * PI * math.cos(2 * PI * v) for v in x]
        return ((h[0], 0.0), (0.0, h[1]))

    return Problem(
        id="rastrigin", name="Rastrigin", f=f, grad=grad, hess=hess,
        bounds=(-5.12, 5.12, -5.12, 5.12), domain=(-5.12, 5.12, -5.12, 5.12),
        x0=(2.2, 2.3), minima=[(0.0, 0.0)], fstar=0.0, test_function=True,
        local_minima=_sym4(0.994959, 0.0) + _sym4(0.0, 0.994959) + _sym4(0.994959, 0.994959),
        note="碗上疊一層規則的蛋盒：每個整數格點附近都有一個局部最小值。")


def bohachevsky1() -> Problem:
    """Bohachevsky N.1: x1^2 + 2 x2^2 - 0.3 cos(3 pi x1) - 0.4 cos(4 pi x2) + 0.7."""
    def f(x: Vec) -> float:
        return (x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * math.cos(3 * PI * x[0])
                - 0.4 * math.cos(4 * PI * x[1]) + 0.7)

    def grad(x: Vec) -> Vec:
        return (2 * x[0] + 0.9 * PI * math.sin(3 * PI * x[0]),
                4 * x[1] + 1.6 * PI * math.sin(4 * PI * x[1]))

    def hess(x: Vec) -> tuple[Vec, Vec]:
        return ((2 + 2.7 * PI * PI * math.cos(3 * PI * x[0]), 0.0),
                (0.0, 4 + 6.4 * PI * PI * math.cos(4 * PI * x[1])))

    return Problem(
        id="bohachevsky1", name="Bohachevsky N.1", f=f, grad=grad, hess=hess,
        bounds=(-1.0, 1.0, -1.0, 1.0), domain=(-100.0, 100.0, -100.0, 100.0),
        x0=(0.8, 0.7), minima=[(0.0, 0.0)], fstar=0.0, test_function=True,
        local_minima=_sym4(0.618612, 0.0) + _sym4(0.0, 0.469528) + _sym4(0.618612, 0.469528),
        note="碗加上兩個方向頻率不同的漣漪；遠看是碗，近看才有局部最小值。")


def griewank() -> Problem:
    """Griewank: 1 + (x1^2 + x2^2) / 4000 - cos(x1) cos(x2 / sqrt 2)."""
    def f(x: Vec) -> float:
        return 1 + (x[0] ** 2 + x[1] ** 2) / 4000 - math.cos(x[0]) * math.cos(x[1] / SQ2)

    def grad(x: Vec) -> Vec:
        return (x[0] / 2000 + math.sin(x[0]) * math.cos(x[1] / SQ2),
                x[1] / 2000 + math.cos(x[0]) * math.sin(x[1] / SQ2) / SQ2)

    return Problem(
        id="griewank", name="Griewank", f=f, grad=grad, hess=_fd_hess(grad),
        bounds=(-10.0, 10.0, -10.0, 10.0), domain=(-600.0, 600.0, -600.0, 600.0),
        x0=(5.0, 6.0), minima=[(0.0, 0.0)], fstar=0.0, test_function=True,
        local_minima=_sym4(3.140023, 4.438444) + _sym4(6.280045, 0.0),
        note="極淺的碗（÷4000）上一片幾乎一樣深的谷：局部最小值彼此只差 0.01 左右。")


BUILDERS = {"sphere": sphere, "ackley": ackley, "levy": levy, "cross_in_tray": cross_in_tray,
            "bukin6": bukin6, "rastrigin": rastrigin, "bohachevsky1": bohachevsky1,
            "griewank": griewank}

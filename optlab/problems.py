"""2D test problems (the "problem zoo") shared by animations, exercises and tests.

Everything is pure Python on tuples so it runs in Pyodide without numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

Vec = tuple[float, float]


@dataclass(frozen=True)
class Problem:
    """A smooth 2D objective with its gradient and Hessian."""

    id: str
    name: str
    f: Callable[[Vec], float]
    grad: Callable[[Vec], Vec]
    hess: Callable[[Vec], tuple[Vec, Vec]]
    bounds: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    x0: Vec
    minima: list[Vec] = field(default_factory=list)
    fstar: float = 0.0
    note: str = ""
    # Least-squares form f(x) = sum_i r_i(x)^2, for Gauss-Newton / LM.
    residuals: Callable[[Vec], tuple[float, ...]] | None = None
    jac: Callable[[Vec], tuple[Vec, ...]] | None = None  # rows = dr_i/dx


def quadratic(kappa: float = 10.0, theta_deg: float = 0.0) -> Problem:
    """f(x) = 0.5 * x^T A x with eigenvalues (1, kappa), rotated by theta.

    The condition number of A is exactly kappa, L = kappa and mu = 1.
    """
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    # A = R diag(1, kappa) R^T
    a11 = c * c + kappa * s * s
    a22 = s * s + kappa * c * c
    a12 = (1.0 - kappa) * c * s

    def f(x: Vec) -> float:
        return 0.5 * (a11 * x[0] ** 2 + 2 * a12 * x[0] * x[1] + a22 * x[1] ** 2)

    def grad(x: Vec) -> Vec:
        return (a11 * x[0] + a12 * x[1], a12 * x[0] + a22 * x[1])

    def hess(x: Vec) -> tuple[Vec, Vec]:
        return ((a11, a12), (a12, a22))

    # Square root S = A^(1/2), so f = |S x|^2 / 2 = sum r_i^2 with r = S x / sqrt 2.
    rk = math.sqrt(kappa)
    s11 = (c * c + rk * s * s) / math.sqrt(2)
    s22 = (s * s + rk * c * c) / math.sqrt(2)
    s12 = (1.0 - rk) * c * s / math.sqrt(2)

    def residuals(x: Vec) -> tuple[float, float]:
        return (s11 * x[0] + s12 * x[1], s12 * x[0] + s22 * x[1])

    def jac(x: Vec) -> tuple[Vec, Vec]:
        return ((s11, s12), (s12, s22))

    return Problem(
        id="quadratic",
        name=f"二次碗 (κ={kappa:g})",
        f=f, grad=grad, hess=hess, residuals=residuals, jac=jac,
        bounds=(-5.0, 5.0, -3.0, 3.0),
        x0=(-4.5, 2.0),
        minima=[(0.0, 0.0)],
        fstar=0.0,
        note="條件數 κ = L/μ 決定峽谷有多窄。",
    )


def rosenbrock(a: float = 1.0, b: float = 100.0) -> Problem:
    """Rosenbrock banana valley, minimum at (a, a^2)."""

    def f(x: Vec) -> float:
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    def grad(x: Vec) -> Vec:
        r = x[1] - x[0] ** 2
        return (-2 * (a - x[0]) - 4 * b * x[0] * r, 2 * b * r)

    def hess(x: Vec) -> tuple[Vec, Vec]:
        h11 = 2 - 4 * b * (x[1] - 3 * x[0] ** 2)
        h12 = -4 * b * x[0]
        return ((h11, h12), (h12, 2 * b))

    sb = math.sqrt(b)

    def residuals(x: Vec) -> tuple[float, float]:
        return (a - x[0], sb * (x[1] - x[0] ** 2))

    def jac(x: Vec) -> tuple[Vec, Vec]:
        return ((-1.0, 0.0), (-2 * sb * x[0], sb))

    return Problem(
        id="rosenbrock",
        name="Rosenbrock 香蕉谷",
        f=f, grad=grad, hess=hess, residuals=residuals, jac=jac,
        bounds=(-2.0, 2.0, -1.0, 3.0),
        x0=(-1.5, 2.5),
        minima=[(a, a * a)],
        fstar=0.0,
        note="谷底彎曲又平坦：找到谷很容易，沿谷走到底很難。",
    )


def himmelblau() -> Problem:
    """Himmelblau's function: four global minima with f = 0."""

    def f(x: Vec) -> float:
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def grad(x: Vec) -> Vec:
        p = x[0] ** 2 + x[1] - 11
        q = x[0] + x[1] ** 2 - 7
        return (4 * x[0] * p + 2 * q, 2 * p + 4 * x[1] * q)

    def hess(x: Vec) -> tuple[Vec, Vec]:
        h11 = 12 * x[0] ** 2 + 4 * x[1] - 42
        h12 = 4 * (x[0] + x[1])
        h22 = 4 * x[0] + 12 * x[1] ** 2 - 26
        return ((h11, h12), (h12, h22))

    def residuals(x: Vec) -> tuple[float, float]:
        return (x[0] ** 2 + x[1] - 11, x[0] + x[1] ** 2 - 7)

    def jac(x: Vec) -> tuple[Vec, Vec]:
        return ((2 * x[0], 1.0), (1.0, 2 * x[1]))

    return Problem(
        id="himmelblau",
        name="Himmelblau 四谷",
        f=f, grad=grad, hess=hess, residuals=residuals, jac=jac,
        bounds=(-5.0, 5.0, -5.0, 5.0),
        x0=(-0.5, -0.5),
        minima=[(3.0, 2.0), (-2.805118, 3.131312),
                (-3.779310, -3.283186), (3.584428, -1.848126)],
        fstar=0.0,
        note="非凸，四個全域最小值：起點決定掉進哪個谷。",
    )


def as_vec(obj) -> Vec:
    """Convert a length-2 sequence (tuple, list, numpy array) to floats."""
    try:
        if len(obj) != 2:
            raise ValueError
        return (float(obj[0]), float(obj[1]))
    except (TypeError, ValueError):
        raise TypeError(f"需要長度 2 的向量，收到 {obj!r}") from None


def make_problem(problem_id: str, **params: float) -> Problem:
    """Build a problem from its id and optional parameters."""
    builders = {"quadratic": quadratic, "rosenbrock": rosenbrock,
                "himmelblau": himmelblau}
    if problem_id not in builders:
        raise KeyError(f"unknown problem: {problem_id}")
    return builders[problem_id](**params)


def grid(problem: Problem, nx: int = 120, ny: int = 100) -> dict:
    """Sample f on a regular grid over the problem bounds (for contour plots)."""
    xmin, xmax, ymin, ymax = problem.bounds
    xs = [xmin + (xmax - xmin) * i / (nx - 1) for i in range(nx)]
    ys = [ymin + (ymax - ymin) * j / (ny - 1) for j in range(ny)]
    z = [[problem.f((x, y)) for x in xs] for y in ys]
    return {"x": xs, "y": ys, "z": z, "fstar": problem.fstar,
            "minima": [list(m) for m in problem.minima],
            "bounds": list(problem.bounds), "name": problem.name,
            "note": problem.note}

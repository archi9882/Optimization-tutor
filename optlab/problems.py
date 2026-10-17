"""2D test problems (the "problem zoo") shared by animations, exercises and tests.

Everything is pure Python on tuples so it runs in Pyodide without numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .sets import lmo_disk, lmo_polygon, project_disk, project_polygon

Vec = tuple[float, float]


@dataclass(frozen=True)
class Constraint:
    """One constraint: kind "ineq" means g(x) <= 0, kind "eq" means g(x) = 0."""

    kind: str
    g: Callable[[Vec], float]
    grad: Callable[[Vec], Vec]
    hess: Callable[[Vec], tuple[Vec, Vec]]
    label: str = ""


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
    # Constrained problems: minima = [x*] (the constrained optimum), fstar = f(x*),
    # multipliers = KKT multipliers lambda* (one per constraint).
    constraints: tuple[Constraint, ...] = ()
    multipliers: tuple[float, ...] = ()
    project: Callable[[Vec], Vec] | None = None   # Euclidean projection onto the set
    lmo: Callable[[Vec], Vec] | None = None       # argmin_{s in set} g^T s
    unconstrained: Vec | None = None              # minimiser of f without constraints


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

# ---------------------------------------------------------------- constrained
# Constraints are written g(x) <= 0 (or g(x) = 0). The constrained optimum x*,
# f* = f(x*) and the KKT multipliers lambda* are derived by hand or solved
# numerically below, never typed in from memory.

_ZERO_H = ((0.0, 0.0), (0.0, 0.0))


def _linear(a: Vec, b: float, label: str, kind: str = "ineq") -> Constraint:
    """a^T x - b <= 0 (or = 0)."""
    return Constraint(kind, lambda x: a[0] * x[0] + a[1] * x[1] - b,
                      lambda x: (float(a[0]), float(a[1])), lambda x: _ZERO_H, label)


def _disk(c: Vec, r: float, label: str) -> Constraint:
    """(x - c)^2 - r^2 <= 0."""
    return Constraint("ineq", lambda x: (x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2 - r * r,
                      lambda x: (2 * (x[0] - c[0]), 2 * (x[1] - c[1])),
                      lambda x: ((2.0, 0.0), (0.0, 2.0)), label)


def _single_multiplier(grad: Callable[[Vec], Vec], c: Constraint, x: Vec) -> float:
    """lambda from grad f + lambda grad g = 0 when only constraint c is active."""
    gf, gc = grad(x), c.grad(x)
    return -(gf[0] * gc[0] + gf[1] * gc[1]) / (gc[0] ** 2 + gc[1] ** 2)


def _fert_objective():
    """Yield-loss proxy (x1 - 3)^2 + 2 (x2 - 2)^2 around the ideal dose (3, 2)."""
    def f(x: Vec) -> float:
        return (x[0] - 3) ** 2 + 2 * (x[1] - 2) ** 2

    def grad(x: Vec) -> Vec:
        return (2 * (x[0] - 3), 4 * (x[1] - 2))

    def hess(x: Vec) -> tuple[Vec, Vec]:
        return ((2.0, 0.0), (0.0, 4.0))
    return f, grad, hess


_FERT_NOTE = ("x₁ = 氮肥、x₂ = 磷肥施用量（每公頃百公斤）。目標以二次函數近似產量損失："
              "理想用量 (3, 2)，偏離越多損失越大（虛構的教學係數）。")
_FERT_BOUNDS = (-0.5, 4.0, -0.5, 3.5)


def fert_budget() -> Problem:
    """Fertiliser N/P with a budget N + P <= 3 and N, P >= 0 (a triangle).

    KKT by hand: 2(x1-3) + lam = 0, 4(x2-2) + lam = 0, x1 + x2 = 3
    -> lam* = 8/3, x* = (5/3, 4/3), f* = 8/3.
    """
    f, grad, hess = _fert_objective()
    verts = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)]
    return Problem(
        id="fert_budget", name="施肥預算（三角形）", f=f, grad=grad, hess=hess,
        bounds=_FERT_BOUNDS, x0=(0.5, 0.5), minima=[(5 / 3, 4 / 3)], fstar=8 / 3,
        note=_FERT_NOTE + "限制：預算 x₁ + x₂ ≤ 3，且 x₁, x₂ ≥ 0。",
        constraints=(_linear((1.0, 1.0), 3.0, "預算 x₁+x₂ ≤ 3"),
                     _linear((-1.0, 0.0), 0.0, "x₁ ≥ 0"),
                     _linear((0.0, -1.0), 0.0, "x₂ ≥ 0")),
        multipliers=(8 / 3, 0.0, 0.0),
        project=lambda y: project_polygon(y, verts),
        lmo=lambda g: lmo_polygon(g, verts),
        unconstrained=(3.0, 2.0))


def fert_equality() -> Problem:
    """Same objective with the budget spent exactly: N + P = 3 (same x*, lam*)."""
    f, grad, hess = _fert_objective()
    return Problem(
        id="fert_eq", name="預算全部用完（等式）", f=f, grad=grad, hess=hess,
        bounds=_FERT_BOUNDS, x0=(0.5, 0.5), minima=[(5 / 3, 4 / 3)], fstar=8 / 3,
        note=_FERT_NOTE + "限制：x₁ + x₂ = 3（等式，可行集是一條線）。",
        constraints=(_linear((1.0, 1.0), 3.0, "x₁+x₂ = 3", kind="eq"),),
        multipliers=(8 / 3,), unconstrained=(3.0, 2.0))


def fert_disk() -> Problem:
    """Same objective inside the disk (x1-1)^2 + (x2-1)^2 <= 1.

    KKT: x1 = (3+lam)/(1+lam), x2 = (4+lam)/(2+lam) and the disk equation
    (2/(1+lam))^2 + (2/(2+lam))^2 = 1, solved for lam by bisection.
    """
    f, grad, hess = _fert_objective()

    def resid(lam: float) -> float:
        return (2 / (1 + lam)) ** 2 + (2 / (2 + lam)) ** 2 - 1

    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if resid(mid) > 0 else (lo, mid)
    lam = 0.5 * (lo + hi)
    xs = ((3 + lam) / (1 + lam), (4 + lam) / (2 + lam))
    return Problem(
        id="fert_disk", name="土壤養分圓盤", f=f, grad=grad, hess=hess,
        bounds=_FERT_BOUNDS, x0=(0.6, 0.8), minima=[xs], fstar=f(xs),
        note=_FERT_NOTE + "限制：(x₁−1)² + (x₂−1)² ≤ 1，"
             "假想的土壤檢測建議：與建議用量 (1, 1) 的距離不超過 1。",
        constraints=(_disk((1.0, 1.0), 1.0, "土壤圓盤"),),
        multipliers=(lam,),
        project=lambda y: project_disk(y, (1.0, 1.0), 1.0),
        lmo=lambda g: lmo_disk(g, (1.0, 1.0), 1.0),
        unconstrained=(3.0, 2.0))


def reservoir() -> Problem:
    """Two reservoir releases with capacity bounds 0 <= r1 <= 2, 0 <= r2 <= 1.

    f = (r1 + r2 - 3)^2 + 0.5 (r1 - r2)^2: meet a demand of 3 while keeping the
    releases similar. Unconstrained optimum (1.5, 1.5); r2 <= 1 is active:
    min_r1 (r1 - 2)^2 + 0.5 (r1 - 1)^2 -> r1 = 5/3, lam(r2 <= 1) = 4/3, f* = 1/3.
    """
    def f(x: Vec) -> float:
        return (x[0] + x[1] - 3) ** 2 + 0.5 * (x[0] - x[1]) ** 2

    def grad(x: Vec) -> Vec:
        s, d = x[0] + x[1] - 3, x[0] - x[1]
        return (2 * s + d, 2 * s - d)

    def hess(x: Vec) -> tuple[Vec, Vec]:
        return ((3.0, 1.0), (1.0, 3.0))

    verts = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    return Problem(
        id="reservoir", name="兩座水庫放水（箱型）", f=f, grad=grad, hess=hess,
        bounds=(-0.5, 2.5, -0.5, 2.0), x0=(0.2, 0.2), minima=[(5 / 3, 1.0)], fstar=1 / 3,
        note="x₁, x₂ = 兩座水庫的日放水量（虛構單位）。下游需水 3，"
             "同時希望兩庫放水接近：f = (x₁+x₂−3)² + ½(x₁−x₂)²。容量：0 ≤ x₁ ≤ 2、0 ≤ x₂ ≤ 1。",
        constraints=(_linear((1.0, 0.0), 2.0, "x₁ ≤ 2"),
                     _linear((0.0, 1.0), 1.0, "x₂ ≤ 1"),
                     _linear((-1.0, 0.0), 0.0, "x₁ ≥ 0"),
                     _linear((0.0, -1.0), 0.0, "x₂ ≥ 0")),
        multipliers=(0.0, 4 / 3, 0.0, 0.0),
        project=lambda y: project_polygon(y, verts),
        lmo=lambda g: lmo_polygon(g, verts),
        unconstrained=(1.5, 1.5))


def rosenbrock_disk() -> Problem:
    """Rosenbrock (a=1, b=100) restricted to the unit disk x1^2 + x2^2 <= 1.

    Nonconvex objective, curved valley meeting the boundary. The optimum lies on
    the circle (the only stationary point of f, (1, 1), is outside), so it is
    found from d/dt f(cos t, sin t) = 0: a dense scan for the bracket, then bisection.
    """
    base = rosenbrock()

    def on_circle(t: float) -> float:
        return base.f((math.cos(t), math.sin(t)))

    def slope(t: float) -> float:
        g = base.grad((math.cos(t), math.sin(t)))
        return -g[0] * math.sin(t) + g[1] * math.cos(t)

    n = 3600
    k = min(range(n), key=lambda i: on_circle(-math.pi + 2 * math.pi * i / n))
    a, b = -math.pi + 2 * math.pi * (k - 1) / n, -math.pi + 2 * math.pi * (k + 1) / n
    for _ in range(100):                      # slope(a) < 0 < slope(b)
        m = 0.5 * (a + b)
        a, b = (m, b) if slope(m) < 0 else (a, m)
    t = 0.5 * (a + b)
    xs = (math.cos(t), math.sin(t))
    disk = _disk((0.0, 0.0), 1.0, "單位圓盤")
    return Problem(
        id="rosen_disk", name="Rosenbrock + 圓盤（非凸）", f=base.f, grad=base.grad,
        hess=base.hess, bounds=(-1.5, 1.5, -1.5, 1.5), x0=(-0.6, 0.6),
        minima=[xs], fstar=base.f(xs),
        note="非凸、病態的香蕉谷，加上限制 x₁² + x₂² ≤ 1；無限制最小值 (1, 1) 在圓外。",
        constraints=(disk,), multipliers=(_single_multiplier(base.grad, disk, xs),),
        project=lambda y: project_disk(y, (0.0, 0.0), 1.0),
        lmo=lambda g: lmo_disk(g, (0.0, 0.0), 1.0),
        unconstrained=(1.0, 1.0))


def violation(problem: Problem, x: Vec) -> float:
    """Largest constraint violation: max(g, 0) for g <= 0, |h| for h = 0."""
    worst = 0.0
    for c in problem.constraints:
        v = c.g(x)
        worst = max(worst, abs(v) if c.kind == "eq" else v)
    return worst


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
                "himmelblau": himmelblau, "fert_budget": fert_budget,
                "fert_eq": fert_equality, "fert_disk": fert_disk,
                "reservoir": reservoir, "rosen_disk": rosenbrock_disk}
    if problem_id not in builders:
        raise KeyError(f"unknown problem: {problem_id}")
    return builders[problem_id](**params)


def grid(problem: Problem, nx: int = 120, ny: int = 100) -> dict:
    """Sample f on a regular grid over the problem bounds (for contour plots)."""
    xmin, xmax, ymin, ymax = problem.bounds
    xs = [xmin + (xmax - xmin) * i / (nx - 1) for i in range(nx)]
    ys = [ymin + (ymax - ymin) * j / (ny - 1) for j in range(ny)]
    z = [[problem.f((x, y)) for x in xs] for y in ys]
    out = {"x": xs, "y": ys, "z": z, "fstar": problem.fstar,
           "minima": [list(m) for m in problem.minima],
           "bounds": list(problem.bounds), "name": problem.name,
           "note": problem.note}
    if problem.constraints:
        # Each constraint sampled on the same grid; the renderer shades g > 0
        # and draws g = 0 with marching squares (works for curved boundaries).
        out["constraints"] = [{"kind": c.kind, "label": c.label,
                               "z": [[round(c.g((x, y)), 6) for x in xs] for y in ys]}
                              for c in problem.constraints]
        if problem.unconstrained:
            out["unconstrained"] = list(problem.unconstrained)
    return out

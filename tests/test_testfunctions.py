"""The eight test functions: listed minima are verified numerically."""
from __future__ import annotations

import math
import random

import pytest

from optlab import make_problem
from optlab.testfunctions import BUILDERS

IDS = list(BUILDERS)


def _nm(f, x, size, iters=400):
    """Tiny Nelder-Mead for local refinement (independent of optlab.demos)."""
    S = [x, (x[0] + size, x[1]), (x[0], x[1] + size)]
    for _ in range(iters):
        S.sort(key=f)
        c = ((S[0][0] + S[1][0]) / 2, (S[0][1] + S[1][1]) / 2)
        w = S[2]
        r = (2 * c[0] - w[0], 2 * c[1] - w[1])
        if f(r) < f(S[0]):
            e = (3 * c[0] - 2 * w[0], 3 * c[1] - 2 * w[1])
            S[2] = e if f(e) < f(r) else r
        elif f(r) < f(S[1]):
            S[2] = r
        else:
            k = ((c[0] + w[0]) / 2, (c[1] + w[1]) / 2)
            if f(k) < f(w):
                S[2] = k
            else:
                S = [S[0]] + [((S[0][0] + p[0]) / 2, (S[0][1] + p[1]) / 2) for p in S[1:]]
    return min(S, key=f)


@pytest.mark.parametrize("pid", IDS)
def test_global_minimum_on_dense_grid_with_refinement(pid: str) -> None:
    """No grid point of the domain beats f*, and refining the best grid point
    lands on a listed global minimum."""
    p = make_problem(pid)
    x0, x1, y0, y1 = p.domain
    n = 301
    pts = [(x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * j / (n - 1))
           for i in range(n) for j in range(n)]
    if pid == "bukin6":            # the valley floor is a curve: sample it too
        pts += [(x, 0.01 * x * x) for x, _ in pts[::n]]
    best = min(pts, key=p.f)
    assert p.f(best) >= p.fstar - 1e-9
    # Also the view box (bounds), at a finer pitch where it is smaller than the domain.
    vx0, vx1, vy0, vy1 = p.bounds
    view = [(vx0 + (vx1 - vx0) * i / 200, vy0 + (vy1 - vy0) * j / 200)
            for i in range(201) for j in range(201)]
    assert min(map(p.f, view)) >= p.fstar - 1e-9
    if pid != "bukin6":
        x = _nm(p.f, best, (x1 - x0) / (n - 1) / 4)
        assert min(math.dist(x, m) for m in p.minima) < 1e-4
    for m in p.minima:
        assert p.f(m) == pytest.approx(p.fstar, abs=1e-12)


@pytest.mark.parametrize("pid", IDS)
def test_listed_local_minima_are_local_minima(pid: str) -> None:
    """Refining from each listed local minimum stays there, f is above f*, and
    every point on a small circle around it is higher."""
    p = make_problem(pid)
    for m in p.local_minima:
        x = _nm(p.f, m, 1e-3)
        assert math.dist(x, m) < 1e-5, (m, x)
        assert p.f(m) > p.fstar + 1e-3
        for k in range(16):
            t = 2 * math.pi * k / 16
            assert p.f((m[0] + 1e-2 * math.cos(t), m[1] + 1e-2 * math.sin(t))) > p.f(m)


@pytest.mark.parametrize("pid", IDS)
def test_gradient_matches_finite_difference(pid: str) -> None:
    """Analytic gradients agree with central differences away from kinks."""
    p = make_problem(pid)
    rng = random.Random(7)
    x0, x1, y0, y1 = p.bounds
    h = 1e-7
    for _ in range(30):
        x = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        if pid == "bukin6" and abs(x[1] - 0.01 * x[0] ** 2) < 1e-2:
            continue
        g = p.grad(x)
        fd = ((p.f((x[0] + h, x[1])) - p.f((x[0] - h, x[1]))) / (2 * h),
              (p.f((x[0], x[1] + h)) - p.f((x[0], x[1] - h))) / (2 * h))
        scale = 1 + max(abs(v) for v in fd)
        assert max(abs(a - b) for a, b in zip(g, fd)) < 1e-4 * scale, (x, g, fd)


def test_known_values() -> None:
    """Values that follow from the formulas by hand."""
    assert make_problem("rastrigin").f((1.0, 0.0)) == pytest.approx(1.0)
    assert make_problem("sphere").f((3.0, 4.0)) == 25.0
    assert make_problem("bukin6").f((-5.0, 0.25)) == pytest.approx(0.05)
    assert make_problem("griewank").f((0.0, 0.0)) == 0.0
    assert make_problem("cross_in_tray").fstar == pytest.approx(-2.06261, abs=1e-5)


def test_grid_carries_local_minima_and_domain_only_for_test_functions() -> None:
    from optlab.problems import grid
    g = grid(make_problem("rastrigin"), 20, 20)
    assert g["local_minima"] and g["domain"] == [-5.12, 5.12, -5.12, 5.12]
    assert "local_minima" not in grid(make_problem("rosenbrock"), 20, 20)


def _grid_local_minima(p, n=161):
    """Every strict local minimum of a grid over the view, refined by Nelder-Mead."""
    x0, x1, y0, y1 = p.bounds
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    ys = [y0 + (y1 - y0) * j / (n - 1) for j in range(n)]
    Z = [[p.f((x, y)) for x in xs] for y in ys]
    out: list = []
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            v = Z[j][i]
            if all(v < Z[j + b][i + a] for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b):
                q = _nm(p.f, (xs[i], ys[j]), (x1 - x0) / (n - 1) / 4)
                if p.f(q) > p.fstar + 1e-6 and not any(math.dist(q, r) < 1e-3 for r in out):
                    out.append(q)
    return out


@pytest.mark.parametrize("pid", [i for i in IDS if make_problem(i).local_minima])
def test_listed_local_minima_are_the_nearest_ones(pid: str) -> None:
    """The gallery says 'the N local minima nearest to the global minimum':
    no unlisted local minimum is as close as the farthest listed one."""
    p = make_problem(pid)

    def dist(q):
        return min(math.dist(q, m) for m in p.minima)
    far = max(dist(m) for m in p.local_minima)
    for q in _grid_local_minima(p):
        if dist(q) <= far + 1e-3:
            assert any(math.dist(q, m) < 1e-4 for m in p.local_minima), (q, dist(q), far)

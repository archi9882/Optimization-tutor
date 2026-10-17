"""Integer and combinatorial family: runners checked against brute force."""
from __future__ import annotations

import itertools

import pytest

from optlab import combo, ip, lp
from optlab.demos.two_opt import tour_length


def _integer_points(prob):
    A, b, _ = prob.all_constraints()
    return [(x, y) for x in range(10) for y in range(10)
            if all(a[0] * x + a[1] * y <= bi + 1e-9 for a, bi in zip(A, b))]


@pytest.mark.parametrize("c", [(5, 4), (3, 4), (4, 5), (6, 2), (1, 1), (2, 7)])
def test_integer_methods_match_enumeration(c) -> None:
    """Branch and bound and Gomory cuts reach the enumerated integer optimum."""
    prob = lp.greenhouse(*c)
    best = max(prob.objective(p) for p in _integer_points(prob))
    bb = ip.run_branch_bound(prob)
    assert bb["status"] == "optimal" and bb["steps"][-1]["best_z"] == pytest.approx(best)
    cp = ip.run_cutting_plane(prob)
    x = cp["steps"][-1]["x"]
    assert cp["status"] == "optimal" and all(abs(v - round(v)) < 1e-9 for v in x)
    assert prob.objective(x) == pytest.approx(best)


def test_rounding_the_lp_optimum_is_not_enough() -> None:
    """LP optimum (3, 1.5): rounding down gives 19, the integer optimum is 20."""
    prob = lp.greenhouse()
    x, z, _ = ip.solve_relaxation(prob)
    assert x == [3.0, 1.5] and z == 21.0
    assert prob.objective((3, 1)) == 19.0
    assert ip.run_branch_bound(prob)["steps"][-1]["incumbent"] == [4.0, 0.0]


def test_gomory_cuts_never_remove_integer_points() -> None:
    """Every cut is valid: all feasible integer points still satisfy it."""
    prob = lp.greenhouse()
    pts = _integer_points(prob)
    for cut in ip.run_cutting_plane(prob)["steps"][-1]["cuts"]:
        assert all(cut["a"][0] * x + cut["a"][1] * y <= cut["b"] + 1e-9 for x, y in pts)


def test_dp_matches_exhaustive_search() -> None:
    """Bellman recursion equals the best over every release sequence."""
    R = combo.RESERVOIR

    def best(t, s):
        if t == len(R["months"]):
            return 0.0
        return max(combo.reservoir_value(t, r) + best(t + 1, min(s + R["inflow"][t] - r, R["S"]))
                   for r in range(s + R["inflow"][t] + 1))
    out = combo.run_dp()
    assert out["total"] == pytest.approx(best(0, R["s0"])) == pytest.approx(71)
    assert out["steps"][-1]["V"][0][R["s0"]] == pytest.approx(out["total"])


def test_kruskal_matches_prim() -> None:
    """Greedy (Kruskal) total equals an independent Prim computation."""
    pts = combo.fields()
    D = combo.distances(pts)
    n, seen, total = len(pts), {0}, 0.0
    while len(seen) < n:
        d, j = min((D[i][j], j) for i in seen for j in range(n) if j not in seen)
        seen.add(j)
        total += d
    assert combo.run_kruskal()["total"] == pytest.approx(total)


def test_hungarian_matches_all_permutations() -> None:
    C = combo.ASSIGN["C"]
    n = len(C)
    best = min(sum(C[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
    out = combo.run_hungarian()
    assert out["status"] == "optimal" and out["total"] == best == 38


def test_two_opt_stops_at_a_local_optimum_that_tabu_escapes() -> None:
    """2-opt stalls above the brute-force optimum; tabu search reaches it."""
    D = combo.distances(combo.fields())
    n = len(D)
    opt = min(tour_length([0, *p], D) for p in itertools.permutations(range(1, n)) if p[0] < p[-1])
    two = combo.run_two_opt()
    assert two["status"] == "local_opt" and two["length"] > opt + 0.4
    assert combo.run_tabu()["length"] == pytest.approx(opt)


def _distinct_tours(out) -> int:
    def canon(t):
        rots = [t[i:] + t[:i] for i in range(len(t))]
        return min(min(rots), min(r[::-1][-1:] + r[::-1][:-1] for r in rots))
    return len({canon(tuple(s["tour"])) for s in out["steps"]})


def test_tabu_memory_prevents_cycling() -> None:
    """Without memory the search cycles among a few tours; tenure 3 explores more."""
    no_memory = combo.run_tabu(tenure=0)
    memory = combo.run_tabu(tenure=3)
    assert _distinct_tours(no_memory) == 3
    assert _distinct_tours(memory) > 3 * _distinct_tours(no_memory)


def test_tabu_tenure_zero_keeps_no_memory() -> None:
    """Tenure 0 must not forbid anything (regression: it used to act like 1)."""
    out = combo.run_tabu(tenure=0)
    assert all(s["tabu"] == [] for s in out["steps"])
    assert out["steps"] != combo.run_tabu(tenure=1)["steps"]

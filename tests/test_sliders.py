"""Lab sliders: every value a page sets or asks the reader to try is reachable."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from optlab import OPTIMIZERS
from optlab.optimizers import N_ITER, PROBLEM_PARAMS
from optlab.sliders import linear_step, slider_values, snap

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "content" / "algorithms"

# Values quoted in each page's 建議的實驗 (and nearby prose) that the reader is
# asked to set with a slider: (owner, param, value). owner is an optimizer id,
# "problem" (problem sliders) or "lab" (迭代數).
QUOTED: dict[str, list[tuple[str, str, float]]] = {
    "adagrad": [("problem", "theta_deg", 45), ("adagrad", "lr", 1.5), ("adagrad", "lr", 0.5),
                ("lab", "n_iter", 300)],
    "adam": [("problem", "theta_deg", 45), ("adam", "lr", 0.01), ("adam", "lr", 0.1),
             ("gd", "lr", 0.01), ("lab", "n_iter", 300)],
    "augmented-lagrangian": [("augmented_lagrangian", "rho", v) for v in (1, 10, 100, 1000)]
    + [("lab", "n_iter", 500)],
    "barrier": [("barrier_nl", "mu0", 1), ("barrier_nl", "inner", 20), ("barrier_nl", "inner", 1),
                ("barrier_nl", "inner", 2)] + [("barrier_nl", "sigma", v) for v in (2, 10, 100)],
    "bfgs": [("problem", "kappa", 10)],
    "coordinate-descent": [("problem", "theta_deg", 30), ("problem", "theta_deg", 0),
                           ("coordinate", "omega", 1), ("coordinate", "omega", 1.5),
                           ("coordinate", "omega", 1.95), ("lab", "n_iter", 300)],
    "frank-wolfe": [("projected_gradient", "lr", 0.1)],
    "gd": [("gd", "lr", v) for v in (0.05, 0.15, 0.21, 0.001, 0.002, 0.003)]
    + [("problem", "kappa", 50), ("problem", "kappa", 10)],
    "lbfgs": [("lbfgs", "m", v) for v in (1, 2, 5)] + [("lab", "n_iter", 300)],
    "levenberg-marquardt": [("lm", "lam0", 0.001), ("lm", "lam0", 1000)],
    "momentum": [("momentum", "beta", 0.97), ("momentum", "beta", 0), ("momentum", "lr", 0.035)],
    "nelder-mead": [("nelder_mead", "size", v) for v in (0.05, 0.5, 2)]
    + [("problem", "kappa", 100)] + [("problem", "theta_deg", v) for v in (0, 30, 45)],
    "nesterov": [("momentum", "beta", v) for v in (0.95, 0.8, 0.9)]
    + [("nesterov", "beta", v) for v in (0.95, 0.8, 0.9)]
    + [("momentum", "lr", 0.001), ("nesterov", "lr", 0.001), ("lab", "n_iter", 300),
       ("lab", "n_iter", 400)],
    "newton": [("newton", "alpha", 0.5)],
    "nonlinear-cg": [("problem", "kappa", 200)] + [("problem", "theta_deg", v) for v in (0, 60, 90)],
    "pattern-search": [("problem", "theta_deg", v) for v in (0, 30, 45)]
    + [("problem", "kappa", 100), ("pattern_search", "step0", 0.01)],
    "penalty": [("penalty", "inner", 300), ("lab", "n_iter", 500), ("penalty", "rho0", 1000),
                ("penalty", "gamma", 100)],
    "powell": [("problem", "kappa", 10), ("problem", "kappa", 100), ("problem", "theta_deg", 30),
               ("problem", "theta_deg", 0)],
    "projected-gradient": [("projected_gradient", "lr", v)
                           for v in (0.01, 0.1, 0.2, 0.35, 0.5, 0.002, 0.003, 0.004)],
    "rmsprop": [("rmsprop", "rho", 0.9), ("rmsprop", "rho", 0.99)],
    "sgd": [("sgd", "decay", 0.05), ("sgd", "sigma", 0), ("sgd", "lr", 0.05), ("sgd", "lr", 0.15),
            ("lab", "n_iter", 400)],
    "trust-region": [("trust_region", "radius0", v) for v in (0.5, 3, 0.05)],
}


def grid(owner: str, name: str) -> list[float]:
    if owner == "lab":
        assert name == "n_iter"
        return N_ITER.values()
    if owner == "problem":
        (p,) = [p for ps in PROBLEM_PARAMS.values() for p in ps if p.name == name]
        return p.values()
    (p,) = [p for p in OPTIMIZERS[owner].params if p.name == name]
    return p.values()


def labs() -> dict[str, dict]:
    out = {}
    for md in sorted(PAGES.glob("*.md")):
        text = md.read_text("utf-8")
        meta = yaml.safe_load(text.split("---", 2)[1]) if text.startswith("---") else {}
        if meta and "lab" in meta:
            out[md.stem] = meta["lab"]
    return out


def test_every_param_has_a_nice_grid() -> None:
    sliders = [p for o in OPTIMIZERS.values() for p in o.params]
    sliders += [p for ps in PROBLEM_PARAMS.values() for p in ps] + [N_ITER]
    for p in sliders:
        vals = p.values()
        assert vals == sorted(vals) and len(set(vals)) == len(vals), p.name
        assert vals[0] == p.lo and vals[-1] == p.hi and p.default in vals, p.name
        assert 10 <= len(vals) <= 1001, (p.name, len(vals))
        if p.integer:
            assert all(isinstance(v, int) for v in vals), p.name
        for v in vals:  # at most 2 significant figures on log sliders
            if p.log and v not in (p.lo, p.hi):
                assert float(f"{v:.2g}") == v, (p.name, v)


def test_grid_rules() -> None:
    assert slider_values(1e-4, 1.0, True, 0.05)[:3] == [1e-4, 1.1e-4, 1.2e-4]
    assert 0.21 in slider_values(1e-4, 1.0, True, 0.05)
    assert linear_step(0.5, 0.999, 0.99) == 0.001
    assert linear_step(20, 500, 150, integer=True) == 5
    assert slider_values(5, 300, True, 100, integer=True)[-3:] == [280, 290, 300]
    assert snap([0.1, 0.2, 0.5], 0.21) == 0.2


def test_frontmatter_values_are_on_the_grid() -> None:
    for page, cfg in labs().items():
        for algo, params in cfg.get("params", {}).items():
            for name, v in params.items():
                assert v in grid(algo, name), (page, algo, name, v)
        for name, v in cfg.get("problem_params", {}).items():
            assert v in grid("problem", name), (page, name, v)
        assert cfg.get("n_iter", 150) in N_ITER.values(), page
        tol = cfg.get("reach_tol", 1e-10)
        assert isinstance(tol, float) and tol in (1e-8, 1e-10), (page, tol)


@pytest.mark.parametrize("page", sorted(QUOTED))
def test_quoted_experiment_values_are_reachable(page: str) -> None:
    cfg = labs()[page]
    shown = set(cfg["algos"] + cfg.get("compare", []))
    for owner, name, v in QUOTED[page]:
        if owner not in ("lab", "problem"):
            assert owner in shown, (page, owner)
        assert v in grid(owner, name), (page, owner, name, v)

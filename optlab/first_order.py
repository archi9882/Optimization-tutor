"""First-order optimizers (the gradient-descent family).

Every optimizer shares one signature so a single runner and a single
renderer can animate all of them:

    step(x, problem, state, params) -> (x_new, state_new)

`state` is a plain dict carried between iterations (velocity, moments, ...).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .problems import Problem, Vec

StepFn = Callable[[Vec, Problem, dict, dict], tuple[Vec, dict]]


@dataclass(frozen=True)
class Param:
    """A tunable hyper-parameter with the slider range shown on the site."""

    name: str
    default: float
    lo: float
    hi: float
    log: bool = False
    label: str = ""


@dataclass(frozen=True)
class Optimizer:
    """An optimizer: id, display name, hyper-parameters and step function."""

    id: str
    name: str
    params: list[Param]
    step: StepFn
    init: Callable[[Vec], dict] = field(default=lambda x: {})

    def defaults(self) -> dict:
        """Return {param_name: default_value}."""
        return {p.name: p.default for p in self.params}


def gd_step(x: Vec, problem: Problem, state: dict, p: dict) -> tuple[Vec, dict]:
    """x <- x - lr * grad f(x)."""
    g = problem.grad(x)
    return (x[0] - p["lr"] * g[0], x[1] - p["lr"] * g[1]), state


def momentum_step(x: Vec, problem: Problem, state: dict,
                  p: dict) -> tuple[Vec, dict]:
    """Heavy ball, PyTorch convention: v <- beta*v + g ; x <- x - lr*v."""
    g = problem.grad(x)
    v = state.get("v", (0.0, 0.0))
    v = (p["beta"] * v[0] + g[0], p["beta"] * v[1] + g[1])
    return (x[0] - p["lr"] * v[0], x[1] - p["lr"] * v[1]), {"v": v}


def nesterov_step(x: Vec, problem: Problem, state: dict,
                  p: dict) -> tuple[Vec, dict]:
    """Nesterov momentum: evaluate the gradient at the look-ahead point."""
    v = state.get("v", (0.0, 0.0))
    look = (x[0] - p["lr"] * p["beta"] * v[0], x[1] - p["lr"] * p["beta"] * v[1])
    g = problem.grad(look)
    v = (p["beta"] * v[0] + g[0], p["beta"] * v[1] + g[1])
    return (x[0] - p["lr"] * v[0], x[1] - p["lr"] * v[1]), {"v": v}


def adagrad_step(x: Vec, problem: Problem, state: dict,
                 p: dict) -> tuple[Vec, dict]:
    """AdaGrad: per-coordinate step lr / sqrt(sum of squared gradients)."""
    g = problem.grad(x)
    s = state.get("s", (0.0, 0.0))
    s = (s[0] + g[0] ** 2, s[1] + g[1] ** 2)
    eps = 1e-8
    x_new = (x[0] - p["lr"] * g[0] / (math.sqrt(s[0]) + eps),
             x[1] - p["lr"] * g[1] / (math.sqrt(s[1]) + eps))
    return x_new, {"s": s}


def rmsprop_step(x: Vec, problem: Problem, state: dict,
                 p: dict) -> tuple[Vec, dict]:
    """RMSProp: like AdaGrad but with an exponential moving average."""
    g = problem.grad(x)
    s = state.get("s", (0.0, 0.0))
    r = p["rho"]
    s = (r * s[0] + (1 - r) * g[0] ** 2, r * s[1] + (1 - r) * g[1] ** 2)
    eps = 1e-8
    x_new = (x[0] - p["lr"] * g[0] / (math.sqrt(s[0]) + eps),
             x[1] - p["lr"] * g[1] / (math.sqrt(s[1]) + eps))
    return x_new, {"s": s}


def adam_step(x: Vec, problem: Problem, state: dict,
              p: dict) -> tuple[Vec, dict]:
    """Adam: momentum on g plus RMSProp scaling, both bias-corrected."""
    g = problem.grad(x)
    t = state.get("t", 0) + 1
    m = state.get("m", (0.0, 0.0))
    s = state.get("s", (0.0, 0.0))
    b1, b2, eps = p["beta1"], p["beta2"], 1e-8
    m = (b1 * m[0] + (1 - b1) * g[0], b1 * m[1] + (1 - b1) * g[1])
    s = (b2 * s[0] + (1 - b2) * g[0] ** 2, b2 * s[1] + (1 - b2) * g[1] ** 2)
    mh = (m[0] / (1 - b1 ** t), m[1] / (1 - b1 ** t))
    sh = (s[0] / (1 - b2 ** t), s[1] / (1 - b2 ** t))
    x_new = (x[0] - p["lr"] * mh[0] / (math.sqrt(sh[0]) + eps),
             x[1] - p["lr"] * mh[1] / (math.sqrt(sh[1]) + eps))
    return x_new, {"t": t, "m": m, "s": s}


_LR = "學習率 η"
OPTIMIZERS: dict[str, Optimizer] = {
    o.id: o for o in [
        Optimizer("gd", "GD", [Param("lr", 0.05, 1e-4, 1.0, True, _LR)],
                  gd_step),
        Optimizer("momentum", "Momentum",
                  [Param("lr", 0.05, 1e-4, 1.0, True, _LR),
                   Param("beta", 0.8, 0.0, 0.99, False, "動量 β")],
                  momentum_step),
        Optimizer("nesterov", "Nesterov",
                  [Param("lr", 0.05, 1e-4, 1.0, True, _LR),
                   Param("beta", 0.8, 0.0, 0.99, False, "動量 β")],
                  nesterov_step),
        Optimizer("adagrad", "AdaGrad",
                  [Param("lr", 0.5, 1e-3, 2.0, True, _LR)], adagrad_step),
        Optimizer("rmsprop", "RMSProp",
                  [Param("lr", 0.05, 1e-4, 1.0, True, _LR),
                   Param("rho", 0.9, 0.5, 0.999, False, "衰減 ρ")],
                  rmsprop_step),
        Optimizer("adam", "Adam",
                  [Param("lr", 0.1, 1e-4, 1.0, True, _LR),
                   Param("beta1", 0.9, 0.0, 0.999, False, "β₁"),
                   Param("beta2", 0.999, 0.9, 0.9999, False, "β₂")],
                  adam_step),
    ]
}

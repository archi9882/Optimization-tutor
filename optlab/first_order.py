"""First-order optimizers (the gradient-descent family).

Every optimizer shares one signature so a single runner and a single
renderer can animate all of them:

    step(x, problem, state, params) -> (x_new, state_new)

`state` is a plain dict carried between iterations (velocity, moments, ...).
"""
from __future__ import annotations

import math
import random
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


def sgd_step(x: Vec, problem: Problem, state: dict, p: dict) -> tuple[Vec, dict]:
    """GD on a noisy gradient g + sigma * N(0, I), with lr_k = lr / (1 + decay k).

    The additive Gaussian noise stands in for the error of a mini-batch
    gradient estimate. The generator is seeded so every run is reproducible.
    """
    rng = state.get("rng") or random.Random(int(p.get("seed", 0)))
    k = state.get("k", 0)
    g = problem.grad(x)
    g = (g[0] + p["sigma"] * rng.gauss(0.0, 1.0),
         g[1] + p["sigma"] * rng.gauss(0.0, 1.0))
    lr = p["lr"] / (1.0 + p["decay"] * k)
    return (x[0] - lr * g[0], x[1] - lr * g[1]), {"rng": rng, "k": k + 1}


def coordinate_curvature(problem: Problem, x: Vec, i: int) -> float:
    """|d^2 f / dx_i^2| floored at 1e-2, used as the coordinate step scale."""
    return max(abs(problem.hess(x)[i][i]), 1e-2)


def coordinate_step(x: Vec, problem: Problem, state: dict,
                    p: dict) -> tuple[Vec, dict]:
    """Cyclic coordinate descent, one coordinate per iteration.

    x_i <- x_i - omega * g_i / H_ii. On a quadratic this is exact line
    minimisation along e_i (omega = 1, Gauss-Seidel) or SOR (omega != 1).
    """
    k = state.get("k", 0)
    i = k % 2
    g = problem.grad(x)[i]
    step = p["omega"] * g / coordinate_curvature(problem, x, i)
    x_new = (x[0] - step, x[1]) if i == 0 else (x[0], x[1] - step)
    return x_new, {"k": k + 1}


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
        Optimizer("sgd", "SGD",
                  [Param("lr", 0.05, 1e-4, 1.0, True, _LR),
                   Param("sigma", 1.0, 0.0, 5.0, False, "梯度雜訊 σ"),
                   Param("decay", 0.0, 0.0, 0.2, False, "衰減 c")],
                  sgd_step),
        Optimizer("coordinate", "座標下降",
                  [Param("omega", 1.0, 0.05, 1.95, False, "鬆弛係數 ω")],
                  coordinate_step),
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
                   Param("rho", 0.99, 0.5, 0.999, False, "衰減 ρ")],
                  rmsprop_step),
        Optimizer("adam", "Adam",
                  [Param("lr", 0.1, 1e-4, 1.0, True, _LR),
                   Param("beta1", 0.9, 0.0, 0.999, False, "β₁"),
                   Param("beta2", 0.999, 0.9, 0.9999, False, "β₂")],
                  adam_step),
    ]
}

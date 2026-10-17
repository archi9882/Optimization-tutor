"""Registry of every optimizer the lab can animate.

Every optimizer shares one signature so a single runner and a single
renderer can animate all of them:

    step(x, problem, state, params) -> (x_new, state_new)

The step functions are not written here: each one is the readable demo
function in optlab.demos, wrapped by its adapter. The animation therefore
runs exactly the code shown in a page's "虛擬碼與示範程式" panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .demos import lab_step
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

    def defaults(self) -> dict:
        """Return {param_name: default_value}."""
        return {p.name: p.default for p in self.params}


_LR = "學習率 η"
OPTIMIZERS: dict[str, Optimizer] = {
    o.id: o for o in [
        Optimizer("gd", "GD", [Param("lr", 0.05, 1e-4, 1.0, True, _LR)],
                  lab_step("gd")),
        Optimizer("sgd", "SGD",
                  [Param("lr", 0.05, 1e-4, 1.0, True, _LR),
                   Param("sigma", 1.0, 0.0, 5.0, False, "梯度雜訊 σ"),
                   Param("decay", 0.0, 0.0, 0.2, False, "衰減 c")],
                  lab_step("sgd")),
        Optimizer("coordinate", "座標下降",
                  [Param("omega", 1.0, 0.05, 1.95, False, "鬆弛係數 ω")],
                  lab_step("coordinate")),
        Optimizer("momentum", "Momentum",
                  [Param("lr", 0.05, 1e-4, 1.0, True, _LR),
                   Param("beta", 0.8, 0.0, 0.99, False, "動量 β")],
                  lab_step("momentum")),
        Optimizer("nesterov", "Nesterov",
                  [Param("lr", 0.05, 1e-4, 1.0, True, _LR),
                   Param("beta", 0.8, 0.0, 0.99, False, "動量 β")],
                  lab_step("nesterov")),
        Optimizer("adagrad", "AdaGrad",
                  [Param("lr", 0.5, 1e-3, 2.0, True, _LR)], lab_step("adagrad")),
        Optimizer("rmsprop", "RMSProp",
                  [Param("lr", 0.05, 1e-4, 1.0, True, _LR),
                   Param("rho", 0.99, 0.5, 0.999, False, "衰減 ρ")],
                  lab_step("rmsprop")),
        Optimizer("adam", "Adam",
                  [Param("lr", 0.1, 1e-4, 1.0, True, _LR),
                   Param("beta1", 0.9, 0.0, 0.999, False, "β₁"),
                   Param("beta2", 0.999, 0.9, 0.9999, False, "β₂")],
                  lab_step("adam")),
        # ---- second-order and quasi-Newton ----
        Optimizer("newton", "Newton",
                  [Param("alpha", 1.0, 0.05, 1.0, False, "步長 α")],
                  lab_step("newton")),
        Optimizer("gauss_newton", "Gauss-Newton",
                  [Param("alpha", 1.0, 0.05, 1.0, False, "步長 α")],
                  lab_step("gauss_newton")),
        Optimizer("lm", "Levenberg-Marquardt",
                  [Param("lam0", 1.0, 1e-3, 1e3, True, "初始阻尼 λ₀")],
                  lab_step("lm")),
        Optimizer("bfgs", "BFGS", [], lab_step("bfgs")),
        Optimizer("lbfgs", "L-BFGS",
                  [Param("m", 5, 1, 10, False, "記憶組數 m")],
                  lab_step("lbfgs")),
        Optimizer("ncg", "非線性 CG", [], lab_step("ncg")),
        Optimizer("trust_region", "信賴域 (dogleg)",
                  [Param("radius0", 0.5, 0.05, 3.0, True, "初始半徑 Δ₀")],
                  lab_step("trust_region")),
        # ---- derivative-free ----
        Optimizer("nelder_mead", "Nelder-Mead",
                  [Param("size", 0.5, 0.05, 2.0, True, "初始單純形邊長")],
                  lab_step("nelder_mead")),
        Optimizer("powell", "Powell", [], lab_step("powell")),
        Optimizer("pattern_search", "模式搜尋",
                  [Param("step0", 0.5, 0.01, 2.0, True, "初始步長 h₀")],
                  lab_step("pattern_search")),
    ]
}

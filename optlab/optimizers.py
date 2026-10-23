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

from .demos import lab_step, sqp as _sqp
from .problems import Problem, Vec
from .sliders import slider_values

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
    integer: bool = False   # slider shows whole numbers only
    unit: str = ""

    def values(self) -> list[float]:
        """Every value the slider can take (see optlab.sliders)."""
        return slider_values(self.lo, self.hi, self.log, self.default, self.integer)


@dataclass(frozen=True)
class Optimizer:
    """An optimizer: id, display name, hyper-parameters and step function."""

    id: str
    name: str
    params: list[Param]
    step: StepFn
    handles_l1: bool = False   # may run on problems with an l1 term (Problem.l1)

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
                  [Param("m", 5, 1, 10, False, "記憶組數 m", integer=True)],
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
        # ---- constrained (problems with .constraints) ----
        Optimizer("penalty", "懲罰函數法",
                  [Param("rho0", 1.0, 0.1, 1000.0, True, "初始 ρ₀"),
                   Param("gamma", 10.0, 2.0, 100.0, True, "放大倍數 γ"),
                   Param("inner", 100, 5, 300, True, "內圈步數上限", integer=True)],
                  lab_step("penalty")),
        Optimizer("barrier_nl", "障礙函數法",
                  [Param("mu0", 1.0, 0.01, 10.0, True, "初始 μ₀"),
                   Param("sigma", 10.0, 2.0, 100.0, True, "縮小倍數 σ"),
                   Param("inner", 20, 1, 50, True, "內圈步數上限", integer=True)],
                  lab_step("barrier_nl")),
        Optimizer("augmented_lagrangian", "增廣 Lagrangian",
                  [Param("rho", 10.0, 0.1, 1000.0, True, "懲罰係數 ρ（固定）"),
                   Param("inner", 100, 5, 300, True, "內圈步數上限", integer=True)],
                  lab_step("augmented_lagrangian")),
        Optimizer("projected_gradient", "投影梯度",
                  [Param("lr", 0.1, 1e-3, 1.0, True, _LR)],
                  lab_step("projected_gradient")),
        Optimizer("frank_wolfe", "Frank-Wolfe", [], lab_step("frank_wolfe")),
        Optimizer("sqp", "SQP（Lagrangian Hessian）", [], lab_step("sqp")),
        Optimizer("sqp_raw", "SQP（Hessian 不修正）", [], _sqp.raw_adapter(_sqp.FUNC)),
        Optimizer("admm", "ADMM", [Param("rho", 1.0, 0.01, 100.0, True, "懲罰係數 ρ")],
                  lab_step("admm"), handles_l1=True),
        Optimizer("sqp_bfgs", "SQP（阻尼 BFGS）", [], _sqp.bfgs_adapter(_sqp.FUNC)),
        # ---- population-based (renderer 3: optlab.trace records "pop") ----
        Optimizer("pso", "PSO",
                  [Param("n_pop", 20, 10, 40, False, "粒子數 N", integer=True),
                   Param("w", 0.72, 0.0, 1.2, False, "慣性 w"),
                   Param("c1", 1.5, 0.0, 3.0, False, "認知 c₁"),
                   Param("c2", 1.5, 0.0, 3.0, False, "社會 c₂"),
                   Param("seed", 0, 0, 99, False, "亂數種子", integer=True)],
                  lab_step("pso")),
        Optimizer("ga", "GA",
                  [Param("n_pop", 20, 10, 40, False, "族群大小 N", integer=True),
                   Param("n_elite", 2, 0, 10, False, "菁英數 E", integer=True),
                   Param("pc", 0.9, 0.0, 1.0, False, "交配率 p_c"),
                   Param("eta_c", 15, 1, 50, False, "SBX η_c", integer=True),
                   Param("pm", 0.5, 0.0, 1.0, False, "突變率 p_m"),
                   Param("eta_m", 20, 1, 50, False, "突變 η_m", integer=True),
                   Param("seed", 0, 0, 99, False, "亂數種子", integer=True)],
                  lab_step("ga")),
    ]
}

# Sliders that belong to the lab rather than to one optimizer.
PROBLEM_PARAMS: dict[str, list[Param]] = {
    "quadratic": [Param("kappa", 10, 1, 200, True, "條件數 κ"),
                  Param("theta_deg", 0, 0, 90, False, "旋轉 θ", integer=True, unit="°")],
}
N_ITER = Param("n_iter", 150, 20, 500, False, "迭代數", integer=True)

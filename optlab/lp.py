"""2D linear and quadratic programs for the LP family's lab (renderer 2).

The runners below drive the demo functions in optlab.demos step by step and
record everything the page shows: the point, the tableau, the pivot, the
working set and a one-line explanation of each step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .demos import active_set, barrier, dual_simplex, simplex

TOL = 1e-9
SUB = "₀₁₂₃₄₅₆₇₈₉"


def sub(n: int) -> str:
    """Integer as unicode subscript digits."""
    return "".join(SUB[int(ch)] for ch in str(n))


@dataclass(frozen=True)
class LP:
    """A 2D program with constraints A x <= b and x >= 0.

    sense "max": maximise c^T x.  sense "min": minimise c^T x.
    For QPs, Q is set and the objective is min 1/2 x^T Q x + c^T x.
    """

    id: str
    name: str
    c: tuple[float, float]
    A: tuple[tuple[float, float], ...]
    b: tuple[float, ...]
    labels: tuple[str, ...]
    sense: str = "max"
    var_labels: tuple[str, str] = ("x₁", "x₂")
    bounds: tuple[float, float, float, float] = (-0.5, 8.0, -0.5, 8.0)
    x0: tuple[float, float] = (0.0, 0.0)
    Q: tuple[tuple[float, float], tuple[float, float]] | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)

    def objective(self, x) -> float:
        """Objective value (the quantity being maximised or minimised)."""
        lin = self.c[0] * x[0] + self.c[1] * x[1]
        if self.Q is None:
            return lin
        (q11, q12), (_, q22) = self.Q
        quad = 0.5 * (q11 * x[0] ** 2 + 2 * q12 * x[0] * x[1] + q22 * x[1] ** 2)
        return quad + lin + self.extra.get("const", 0.0)

    def all_constraints(self) -> tuple[list, list, list]:
        """A, b and labels including the sign constraints x1 >= 0, x2 >= 0."""
        A = [list(a) for a in self.A] + [[-1.0, 0.0], [0.0, -1.0]]
        b = list(self.b) + [0.0, 0.0]
        labels = list(self.labels) + [f"{self.var_labels[0]} ≥ 0",
                                      f"{self.var_labels[1]} ≥ 0"]
        return A, b, labels


def farm(c1: float = 3.0, c2: float = 5.0) -> LP:
    """Two crops on limited paddy land, greenhouse space and irrigation water."""
    return LP(
        id="farm", name="作物配置（最大化利潤）",
        c=(c1, c2),
        A=((1.0, 0.0), (0.0, 2.0), (3.0, 2.0)),
        b=(4.0, 12.0, 18.0),
        labels=("稻米田：x₁ ≤ 4", "溫室：2x₂ ≤ 12", "灌溉水：3x₁ + 2x₂ ≤ 18"),
        var_labels=("稻米 x₁", "蔬菜 x₂"),
        bounds=(-0.5, 7.0, -0.5, 7.5),
        x0=(1.0, 1.0),
        note="x₁、x₂ 為兩種作物的種植面積（公頃），c 為每公頃利潤（萬元）。",
    )


def fertilizer(c1: float = 2.0, c2: float = 3.0) -> LP:
    """Cheapest mix of two fertilisers meeting nitrogen and phosphorus needs."""
    return LP(
        id="fertilizer", name="施肥配方（最小化成本）",
        c=(c1, c2), sense="min",
        A=((-2.0, -1.0), (-1.0, -2.0)),
        b=(-8.0, -7.0),
        labels=("氮：2x₁ + x₂ ≥ 8", "磷：x₁ + 2x₂ ≥ 7"),
        var_labels=("肥料甲 x₁", "肥料乙 x₂"),
        bounds=(-0.5, 9.0, -0.5, 9.0),
        note="x₁、x₂ 為兩種肥料的用量（包），c 為每包價格（百元）。",
    )


def farm_qp() -> LP:
    """Same land as `farm`, but aim for a target plan with a quadratic penalty."""
    Q = ((1.0, 0.25), (0.25, 0.5))
    t = (5.0, 7.0)
    q = (-(Q[0][0] * t[0] + Q[0][1] * t[1]), -(Q[1][0] * t[0] + Q[1][1] * t[1]))
    return LP(
        id="farm_qp", name="作物配置（接近目標計畫，二次目標）",
        c=q, sense="min", Q=Q,
        A=((1.0, 0.0), (0.0, 2.0), (3.0, 2.0)),
        b=(4.0, 12.0, 18.0),
        labels=("稻米田：x₁ ≤ 4", "溫室：2x₂ ≤ 12", "灌溉水：3x₁ + 2x₂ ≤ 18"),
        var_labels=("稻米 x₁", "蔬菜 x₂"),
        bounds=(-0.5, 7.0, -0.5, 7.5),
        note="目標計畫 (5, 7) 超出可行範圍；求可行計畫中「加權距離」最近的一個。",
        # Constant 1/2 t^T Q t makes the objective the weighted distance itself.
        extra={"target": list(t),
               "const": 0.5 * (Q[0][0] * t[0] ** 2 + 2 * Q[0][1] * t[0] * t[1]
                               + Q[1][1] * t[1] ** 2)},
    )


def greenhouse(c1: float = 5.0, c2: float = 4.0) -> LP:
    """Integer program: how many large and small greenhouses to build."""
    return LP(
        id="greenhouse", name="溫室興建（整數規劃）",
        c=(c1, c2),
        A=((6.0, 4.0), (1.0, 2.0)),
        b=(24.0, 6.0),
        labels=("預算：6x₁ + 4x₂ ≤ 24", "土地：x₁ + 2x₂ ≤ 6"),
        var_labels=("大型 x₁", "小型 x₂"),
        bounds=(-0.5, 5.5, -0.5, 4.0),
        note="x₁、x₂ 為大型、小型溫室的棟數，必須是整數；c 為每棟年收益（十萬元）。",
        extra={"integer": True},
    )


PRESETS = {"farm": farm, "fertilizer": fertilizer, "farm_qp": farm_qp,
           "greenhouse": greenhouse}


def make_lp(preset: str, **params: float) -> LP:
    """Build a preset program by id."""
    return PRESETS[preset](**params)


# ---------- geometry ----------

def feasible_polygon(A, b, box) -> list[list[float]]:
    """Vertices (counter-clockwise) of {A x <= b} clipped to the view box."""
    x0, x1, y0, y1 = box
    rows = [list(a) for a in A] + [[1, 0], [-1, 0], [0, 1], [0, -1]]
    rhs = list(b) + [x1, -x0, y1, -y0]
    pts = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            (a, c), (d, e) = rows[i], rows[j]
            det = a * e - c * d
            if abs(det) < 1e-12:
                continue
            p = ((rhs[i] * e - c * rhs[j]) / det, (a * rhs[j] - rhs[i] * d) / det)
            if all(r[0] * p[0] + r[1] * p[1] <= h + 1e-7 for r, h in zip(rows, rhs)):
                if not any(math.dist(p, q) < 1e-7 for q in pts):
                    pts.append(p)
    if not pts:
        return []
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return [[p[0] + 0.0, p[1] + 0.0] for p in pts]


def region_data(lp: LP) -> dict:
    """Everything the renderer needs to draw the feasible region."""
    A, b, labels = lp.all_constraints()
    return {"polygon": feasible_polygon(A, b, lp.bounds),
            "constraints": [{"a": a, "b": bi, "label": lab}
                            for a, bi, lab in zip(A, b, labels)][:len(lp.A)],
            "c": list(lp.c), "sense": lp.sense, "var_labels": list(lp.var_labels),
            "target": lp.extra.get("target"), "lattice": lp.extra.get("integer", False)}


def objective_grid(lp: LP, nx: int = 90, ny: int = 90) -> dict:
    """Objective sampled on the view box (contours = iso-profit lines)."""
    xmin, xmax, ymin, ymax = lp.bounds
    xs = [xmin + (xmax - xmin) * i / (nx - 1) for i in range(nx)]
    ys = [ymin + (ymax - ymin) * j / (ny - 1) for j in range(ny)]
    sign = -1.0 if lp.sense == "max" else 1.0      # contours drawn for "lower is better"
    z = [[sign * lp.objective((x, y)) for x in xs] for y in ys]
    return {"x": xs, "y": ys, "z": z, "bounds": list(lp.bounds), "minima": [],
            "fstar": 0.0, "name": lp.name, "linear": lp.Q is None, "note": lp.note,
            "sign": sign, "flabel": "z" if lp.sense == "max" else "目標值"}


# ---------- tableau helpers ----------

def initial_tableau(lp: LP) -> tuple[list[list[float]], list[int], list[str]]:
    """Tableau [A | I | b] with objective row for max (-c) or min-as-max (+c)."""
    m = len(lp.A)
    cols = ["x₁", "x₂"] + [f"s{sub(i + 1)}" for i in range(m)] + ["右手邊"]
    T = []
    for i, (a, bi) in enumerate(zip(lp.A, lp.b)):
        T.append([float(a[0]), float(a[1])] + [1.0 if k == i else 0.0 for k in range(m)]
                 + [float(bi)])
    obj = [-lp.c[0], -lp.c[1]] if lp.sense == "max" else [lp.c[0], lp.c[1]]
    T.append([float(v) for v in obj] + [0.0] * m + [0.0])
    basis = [2 + i for i in range(m)]
    return T, basis, cols


def point_from(T, basis) -> list[float]:
    """Current basic solution restricted to (x1, x2)."""
    x = [0.0, 0.0]
    for r, var in enumerate(basis):
        if var < 2:
            x[var] = T[r][-1]
    return x


def _g(v: float) -> str:
    """Pretty number for explanations."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.3g}"


def _shadow_prices(lp: LP, T) -> str:
    """Dual values read from the slack columns of the objective row."""
    m = len(lp.A)
    parts = []
    for i in range(m):
        y = T[-1][2 + i]
        label = lp.labels[i].split("：")[0]
        parts.append(f"{label} {_g(abs(y))}")
    return "影子價格（每多一單位資源，最優目標值的變化）：" + "、".join(parts) + "。"


def _snapshot(T, basis, cols, x, value, note, pivot=None) -> dict:
    return {"x": [float(v) for v in x], "z": float(value),
            "tableau": [[float(v) for v in row] for row in T],
            "basis": [cols[b] for b in basis], "cols": cols,
            "pivot": list(pivot) if pivot else None, "note": note}


# ---------- runners ----------

def run_simplex(lp: LP, choose_pivot=simplex.choose_pivot, pivot=simplex.pivot,
                max_steps: int = 20) -> dict:
    """Primal simplex from the origin, using the given demo/learner functions."""
    T, basis, cols = initial_tableau(lp)
    x = point_from(T, basis)
    steps = [_snapshot(T, basis, cols, x, lp.objective(x),
                       "起點：原點 (0, 0)，基變數全是 slack，z = 0。")]
    status = "max_steps"
    for _ in range(max_steps):
        choice = choose_pivot(T)
        if choice is None:
            status = "optimal"
            steps[-1]["note"] += "目標列沒有負的檢驗數，已達最優。" + _shadow_prices(lp, T)
            break
        row, col = choice
        if row is None:
            status = "unbounded"
            steps[-1]["note"] += f"{cols[col]} 可以無限增加：問題無界。"
            break
        ratios = [f"{_g(T[i][-1])}/{_g(T[i][col])}={_g(T[i][-1] / T[i][col])}"
                  for i in range(len(T) - 1) if T[i][col] > 1e-12]
        entering, leaving = cols[col], cols[basis[row]]
        T = pivot(T, row, col)
        basis[row] = col
        x = point_from(T, basis)
        steps.append(_snapshot(
            T, basis, cols, x, lp.objective(x),
            f"{entering} 進基（檢驗數 {_g(steps[-1]['tableau'][-1][col])} 最負）；"
            f"比值測試 {'、'.join(ratios)}，最小者讓 {leaving} 出基。"
            f"移到頂點 ({_g(x[0])}, {_g(x[1])})，z = {_g(lp.objective(x))}。",
            pivot=(row, col)))
    return {"steps": steps, "status": status}


def dual_tableau(lp: LP):
    """Tableau for min c^T x, A x <= b with b < 0 allowed (dual feasible start)."""
    return initial_tableau(lp)


def run_dual_simplex(lp: LP, choose=dual_simplex.dual_choose_pivot,
                     pivot=simplex.pivot, max_steps: int = 20) -> dict:
    """Dual simplex from the origin (infeasible, dual feasible)."""
    T, basis, cols = dual_tableau(lp)
    x = point_from(T, basis)
    viol = [lp.labels[i].split("：")[0] for i in range(len(lp.b)) if lp.b[i] < 0]
    steps = [_snapshot(T, basis, cols, x, lp.objective(x),
                       f"起點：原點 (0, 0)。成本為 0 但違反 {'、'.join(viol)} 的需求"
                       "（右手邊為負）；檢驗數都 ≥ 0，所以是對偶可行。")]
    status = "max_steps"
    for _ in range(max_steps):
        choice = choose(T)
        if choice is None:
            status = "optimal"
            steps[-1]["note"] += "右手邊全部 ≥ 0：原始也可行，已達最優。"
            break
        row, col = choice
        if col is None:
            status = "infeasible"
            steps[-1]["note"] += f"第 {row + 1} 列沒有負係數：原問題不可行。"
            break
        ratios = [f"|{_g(T[-1][j])}/{_g(T[row][j])}|={_g(abs(T[-1][j] / T[row][j]))}"
                  for j in range(len(T[0]) - 1) if T[row][j] < -1e-12]
        leaving, entering = cols[basis[row]], cols[col]
        rhs = T[row][-1]
        T = pivot(T, row, col)
        basis[row] = col
        x = point_from(T, basis)
        steps.append(_snapshot(
            T, basis, cols, x, lp.objective(x),
            f"{leaving} 出基（右手邊 {_g(rhs)} 最負）；對偶比值 {'、'.join(ratios)}，"
            f"最小者讓 {entering} 進基。移到 ({_g(x[0])}, {_g(x[1])})，"
            f"成本 = {_g(lp.objective(x))}。",
            pivot=(row, col)))
    return {"steps": steps, "status": status}


def run_barrier(lp: LP, step=barrier.barrier_newton_step, mu0: float = 10.0,
                sigma: float = 4.0, inner: int = 3, outer: int = 9) -> dict:
    """Barrier method: a few Newton steps per mu, then shrink mu."""
    A, b, _ = lp.all_constraints()
    c = list(lp.c) if lp.sense == "max" else [-lp.c[0], -lp.c[1]]
    x = tuple(lp.x0)
    mu = mu0
    steps = [{"x": list(x), "z": lp.objective(x), "mu": mu, "central": True,
              "note": f"起點 ({_g(x[0])}, {_g(x[1])}) 在可行域內部，μ = {_g(mu)}。"}]
    for k in range(outer):
        for _ in range(inner):
            x = step(x, mu, c, A, b)
            steps.append({"x": list(x), "z": lp.objective(x), "mu": mu, "central": False,
                          "note": f"μ = {_g(mu)} 時的牛頓步：({x[0]:.3f}, {x[1]:.3f})，"
                                  f"z = {lp.objective(x):.6g}。"})
        steps[-1]["central"] = True
        gap = mu * len(b)
        steps[-1]["note"] += f"近似中心路徑上的點；精確中心點的對偶間隙上界 m·μ = {gap:.3g}。"
        mu /= sigma
    return {"steps": steps, "status": "optimal"}


def run_active_set(lp: LP, step=active_set.active_set_step, max_steps: int = 20) -> dict:
    """Primal active-set method for the QP preset, starting from x0 with W = []."""
    A, b, labels = lp.all_constraints()
    x, W = tuple(lp.x0), []
    steps = [{"x": list(x), "z": lp.objective(x), "W": [],
              "note": f"起點 ({_g(x[0])}, {_g(x[1])})，工作集 W 為空。"}]
    status = "max_steps"
    for _ in range(max_steps):
        x_new, W_new, done = step(x, list(W), lp.Q, lp.c, A, b)
        if done:
            status = "optimal"
            steps[-1]["note"] += "方向為 0 且乘數都 ≥ 0：KKT 成立，已達最優。"
            break
        added = [labels[i] for i in W_new if i not in W]
        dropped = [labels[i] for i in W if i not in W_new]
        if dropped:
            note = f"在 ({_g(x[0])}, {_g(x[1])}) 方向為 0，但「{dropped[0]}」的乘數為負：把它移出 W。"
        elif added:
            note = (f"沿方向前進，被「{added[0]}」擋住，停在 ({x_new[0]:.3f}, {x_new[1]:.3f})，"
                    "並把它加入 W。")
        else:
            note = f"不受阻擋，走到子問題的最低點 ({x_new[0]:.3f}, {x_new[1]:.3f})。"
        x, W = x_new, W_new
        steps.append({"x": list(x), "z": lp.objective(x), "W": [labels[i] for i in W],
                      "note": note})
    return {"steps": steps, "status": status}


RUNNERS = {"simplex": run_simplex, "dual_simplex": run_dual_simplex,
           "barrier": run_barrier, "active_set": run_active_set}


def lp_lab(spec: dict) -> dict:
    """Full payload for the LP lab: region, objective grid and the step list."""
    lp = make_lp(spec["preset"], **spec.get("preset_params", {}))
    method = spec["method"]
    kwargs = spec.get("params", {})
    if method in RUNNERS:
        out = RUNNERS[method](lp, **kwargs)
    else:
        from . import ip                      # integer methods build on this module
        out = ip.RUNNERS[method](lp, **kwargs)
    out.update({"region": region_data(lp), "grid": objective_grid(lp),
                "method": method, "preset": lp.id, "name": lp.name,
                "sense": lp.sense, "note": lp.note})
    return out

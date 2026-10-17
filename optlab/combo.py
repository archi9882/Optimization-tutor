"""Small combinatorial instances and step-by-step runners (renderers ④/⑤).

Every runner drives a demo function from optlab.demos (or a learner's
replacement) and records what the page draws at each step plus a one-line
explanation. Instances are tiny on purpose: tests can brute-force them.
"""
from __future__ import annotations

import math
import random

from .demos import dynamic_programming as dp_demo
from .demos import greedy as greedy_demo
from .demos import hungarian as hung_demo
from .demos import tabu as tabu_demo
from .demos import two_opt as two_opt_demo

# ---------- instances ----------

FIELD_SEED = 0


def fields(n: int = 9, seed: int = FIELD_SEED) -> list[list[float]]:
    """Field locations (km) on a 10 × 7 km area, fixed by `seed`."""
    rng = random.Random(seed)
    return [[round(rng.uniform(0.5, 9.5), 2), round(rng.uniform(0.5, 6.5), 2)] for _ in range(n)]


def distances(pts) -> list[list[float]]:
    """Euclidean distance matrix."""
    return [[math.dist(p, q) for q in pts] for p in pts]


RESERVOIR = {
    "months": ["3月", "4月", "5月", "6月", "7月", "8月"],
    "inflow": [3, 2, 1, 0, 0, 2],          # 每月入流（單位：百萬 m³）
    "weight": [1, 1, 2, 3, 3, 1],          # 每單位灌溉收益的月份權重（夏季缺水較值錢）
    "benefit": [0, 4, 7, 9, 10, 10.5, 10.8, 11],   # 放 r 單位的基本收益，邊際遞減
    "S": 5,                                 # 水庫容量
    "s0": 2,                                # 3 月初蓄水量
}


def reservoir_value(t: int, r: int) -> float:
    """Benefit of releasing r units in month t."""
    b = RESERVOIR["benefit"]
    return RESERVOIR["weight"][t] * b[min(r, len(b) - 1)]


ASSIGN = {
    "rows": ["曳引機 A", "曳引機 B", "插秧機 C", "收穫機 D", "噴藥機 E"],
    "cols": ["田區 1", "田區 2", "田區 3", "田區 4", "田區 5"],
    # 每台機具完成每塊田區作業所需的時數（含移動）
    "C": [[9, 11, 14, 11, 7],
          [6, 15, 13, 13, 10],
          [12, 13, 6, 8, 8],
          [11, 9, 10, 12, 9],
          [7, 12, 14, 10, 14]],
}


# ---------- dynamic programming ----------

def run_dp(stage=dp_demo.bellman_stage) -> dict:
    """Backward Bellman sweep, then the forward pass that reads off the plan."""
    R = RESERVOIR
    T, S = len(R["months"]), R["S"]
    V = [[None] * (S + 1) for _ in range(T + 1)]
    P = [[None] * (S + 1) for _ in range(T)]
    V[T] = [0.0] * (S + 1)
    steps = [{"V": [list(c) for c in V], "P": [list(c) for c in P], "col": T, "path": [],
              "note": "最後一個月之後剩下的水不計價：V(s) = 0。接著從 8 月往回算。"}]
    for t in range(T - 1, -1, -1):
        V[t], P[t] = stage(V[t + 1], t, S, R["inflow"][t], reservoir_value)
        V[t] = [float(v) for v in V[t]]
        steps.append({"V": [list(c) for c in V], "P": [list(c) for c in P], "col": t, "path": [],
                      "note": f"{R['months'][t]}：入流 {R['inflow'][t]}、收益權重 {R['weight'][t]}。"
                              f"對每個蓄水量試所有放水量，取「本月收益 + 下個月價值」最大者。"
                              f"蓄水 {R['s0']} 時的價值 = {V[t][R['s0']]:g}。"})
    s, path, total = R["s0"], [], 0.0
    for t in range(T):
        r = P[t][s]
        nxt = min(s + R["inflow"][t] - r, S)
        gain = reservoir_value(t, r)
        total += gain
        path.append({"t": t, "s": s, "r": r})
        spill = s + R["inflow"][t] - r - nxt
        prev, s = s, nxt
        steps.append({"V": [list(c) for c in V], "P": [list(c) for c in P], "col": t,
                      "path": [dict(p) for p in path],
                      "note": f"往前走：{R['months'][t]}初蓄水 {prev}，放水 {r}（收益 {gain:g}），"
                              f"下個月初剩 {nxt}" + (f"，溢流 {spill}" if spill else "") +
                              f"。累計收益 {total:g}。"})
    path.append({"t": T, "s": nxt, "r": None})
    steps[-1]["path"] = [dict(p) for p in path]
    steps[-1]["note"] += f"總收益 {total:g}，等於表中 3 月、蓄水 {R['s0']} 的值 {V[0][R['s0']]:g}。"
    return {"kind": "dp", "steps": steps, "status": "optimal", "total": total,
            "months": R["months"], "S": S, "inflow": R["inflow"], "weight": R["weight"]}


# ---------- greedy: Kruskal ----------

def run_kruskal(step=greedy_demo.kruskal_step, n: int = 9, seed: int = FIELD_SEED) -> dict:
    """Kruskal on the complete graph of field locations."""
    pts = fields(n, seed)
    D = distances(pts)
    edges = sorted((D[i][j], i, j) for i in range(n) for j in range(i + 1, n))
    parent = list(range(n))
    tree, total = [], 0.0
    steps = [{"edges": [], "consider": None, "note": f"{n} 塊田區、{len(edges)} 條候選管線。依長度由短到長排序。"}]
    for k, e in enumerate(edges):
        ok = step(e, parent)
        if ok:
            tree.append([e[1], e[2]])
            total += e[0]
        note = (f"第 {k + 1} 短的管線 {e[1] + 1}–{e[2] + 1}（{e[0]:.2f} km）："
                + ("連接兩個不同的組，選它。" if ok else "兩端已經連通，選了會形成迴圈，跳過。")
                + f"已選 {len(tree)} 條，總長 {total:.2f} km。")
        steps.append({"edges": [list(t) for t in tree], "consider": [e[1], e[2]], "accepted": ok,
                      "note": note})
        if len(tree) == n - 1:
            steps[-1]["note"] += f"選滿 {n - 1} 條，所有田區都連通了。"
            break
    return {"kind": "graph", "tour": False, "points": pts, "steps": steps, "status": "optimal",
            "total": total}


# ---------- Hungarian ----------

def min_cover(Z) -> tuple[list[int], list[int]]:
    """Fewest rows + columns covering every zero (brute force; n ≤ 6)."""
    n = len(Z)
    zeros = [(i, j) for i in range(n) for j in range(n) if abs(Z[i][j]) < 1e-9]
    best = None
    for mask in range(1 << (2 * n)):
        rows = [i for i in range(n) if mask >> i & 1]
        cols = [j for j in range(n) if mask >> (n + j) & 1]
        if best is not None and len(rows) + len(cols) >= len(best[0]) + len(best[1]):
            continue
        if all(i in rows or j in cols for i, j in zeros):
            best = (rows, cols)
    return best


def zero_assignment(Z) -> list[int] | None:
    """A perfect matching on zero entries (row i -> column), if one exists."""
    import itertools
    n = len(Z)
    for perm in itertools.permutations(range(n)):
        if all(abs(Z[i][perm[i]]) < 1e-9 for i in range(n)):
            return list(perm)
    return None


def run_hungarian(adjust=hung_demo.adjust, reduce=hung_demo.reduce_matrix) -> dict:
    """Reduce, cover, adjust until n lines are needed, then read the assignment."""
    C = ASSIGN["C"]
    n = len(C)
    steps = [{"M": [list(r) for r in C], "rows": [], "cols": [], "assign": None,
              "note": "原始成本矩陣（時數）。目標：每台機具負責一塊田區，總時數最少。"}]
    M = reduce(C)
    steps.append({"M": [list(r) for r in M], "rows": [], "cols": [], "assign": None,
                  "note": "每列減去列最小值，再每欄減去欄最小值。每列每欄都至少有一個 0；"
                          "減去常數不會改變哪個指派最好。"})
    for _ in range(n * n):
        rows, cols = min_cover(M)
        lines = len(rows) + len(cols)
        if lines == n:
            assign = zero_assignment(M)
            total = sum(C[i][assign[i]] for i in range(n))
            pairs = "、".join(f"{ASSIGN['rows'][i]}→{ASSIGN['cols'][assign[i]]}" for i in range(n))
            steps.append({"M": [list(r) for r in M], "rows": rows, "cols": cols, "assign": assign,
                          "note": f"最少需要 {n} 條線才能蓋住所有 0，所以能在 0 上選出 {n} 個不同列不同欄的位置："
                                  f"{pairs}。總時數 {total}。"})
            return {"kind": "matrix", "steps": steps, "status": "optimal", "total": total,
                    "row_labels": ASSIGN["rows"], "col_labels": ASSIGN["cols"], "C": C}
        uncovered = [M[i][j] for i in range(n) for j in range(n) if i not in rows and j not in cols]
        k = min(uncovered)
        steps.append({"M": [list(r) for r in M], "rows": rows, "cols": cols, "assign": None,
                      "note": f"最少 {lines} 條線就能蓋住所有 0，少於 {n} 條，還不能指派。"
                              f"沒被蓋住的最小值是 {k:g}。"})
        M = adjust(M, rows, cols)
        steps.append({"M": [list(r) for r in M], "rows": rows, "cols": cols, "assign": None,
                      "note": f"沒被蓋住的元素減 {k:g}、交叉點加 {k:g}：產生新的 0，"
                              f"而原本的 0 只要不在交叉點就保留。"})
    return {"kind": "matrix", "steps": steps, "status": "max_steps", "row_labels": ASSIGN["rows"],
            "col_labels": ASSIGN["cols"], "C": C}


# ---------- 2-opt and tabu ----------

def run_two_opt(step=two_opt_demo.two_opt_step, n: int = 9, seed: int = FIELD_SEED,
                max_steps: int = 50) -> dict:
    """Best-improvement 2-opt from the tour 1 → 2 → … → n."""
    pts = fields(n, seed)
    D = distances(pts)
    tour = list(range(n))
    L = two_opt_demo.tour_length
    steps = [{"tour": list(tour), "length": L(tour, D), "seg": None,
              "note": f"起始路線依編號巡田 1 → 2 → … → {n} → 1，總長 {L(tour, D):.2f} km。"}]
    status = "max_steps"
    for _ in range(max_steps):
        out = step(tour, D)
        if out is None:
            status = "local_opt"
            steps[-1]["note"] += "沒有任何 2-opt 交換能再縮短：到達局部最佳。"
            break
        new, i, j = out
        before = L(tour, D)
        tour = list(new)
        steps.append({"tour": list(tour), "length": L(tour, D), "seg": [i, j],
                      "note": f"把第 {i + 2}～{j + 1} 站這段反轉：拿掉邊 {_edge(steps[-1]['tour'], i)} 與 "
                              f"{_edge(steps[-1]['tour'], j)}，換成兩條較短的邊，"
                              f"總長 {before:.2f} → {L(tour, D):.2f} km。"})
    return {"kind": "graph", "tour": True, "points": pts, "steps": steps, "status": status,
            "length": L(tour, D)}


def _edge(tour, i) -> str:
    """Edge tour[i] → tour[i+1] as 1-based labels."""
    return f"{tour[i] + 1}–{tour[(i + 1) % len(tour)] + 1}"


def run_tabu(step=tabu_demo.tabu_step, n: int = 9, seed: int = FIELD_SEED,
             iters: int = 20, tenure: int = 5, start: list | None = None) -> dict:
    """Tabu search over 2-opt moves, starting from the 2-opt local optimum."""
    pts = fields(n, seed)
    D = distances(pts)
    L = two_opt_demo.tour_length
    if start is None:
        start = run_two_opt(n=n, seed=seed)["steps"][-1]["tour"]
    tour = list(start)
    best, best_len = list(tour), L(tour, D)
    tabu: dict = {}
    steps = [{"tour": list(tour), "length": best_len, "best": list(best), "best_len": best_len,
              "tabu": [], "seg": None,
              "note": f"從 2-opt 的局部最佳出發（{best_len:.3f} km）。2-opt 到這裡就停了。"}]
    for it in range(iters):
        out = step(tour, D, tabu, best_len)
        if out is None:
            break
        new, i, j, removed = out
        before = L(tour, D)
        tour = list(new)
        tabu = {e: k - 1 for e, k in tabu.items() if k > 1}
        for e in removed:
            if tenure > 0:                       # tenure 0 = no memory at all
                tabu[e] = tenure
        now = L(tour, D)
        improved = now < best_len - 1e-9
        if improved:
            best, best_len = list(tour), now
        trend = "變短" if now < before - 1e-9 else "變長（但仍是允許的最好選擇）"
        steps.append({"tour": list(tour), "length": now, "best": list(best), "best_len": best_len,
                      "tabu": [[list(e), k] for e, k in sorted(tabu.items())], "seg": [i, j],
                      "note": f"第 {it + 1} 回合：反轉第 {i + 2}～{j + 1} 站，{before:.3f} → {now:.3f} km，{trend}。"
                              + ("打破歷史最佳！" if improved else f"歷史最佳仍是 {best_len:.3f} km。")
                              + (f"拿掉的邊 {removed[0][0] + 1}–{removed[0][1] + 1}、{removed[1][0] + 1}–{removed[1][1] + 1}"
                                 f" 在 {tenure} 回合內不准加回。" if tenure > 0 else "禁忌期為 0：不記任何邊。")})
    return {"kind": "graph", "tour": True, "points": pts, "steps": steps, "status": "done",
            "length": best_len}


RUNNERS = {"dp": run_dp, "kruskal": run_kruskal, "hungarian": run_hungarian,
           "two_opt": run_two_opt, "tabu": run_tabu}


META = {
    "dp": ("水庫逐月放水（動態規劃）",
           "容量 5、3 月初蓄水 2（百萬 m³）。每月決定放多少水灌溉；夏季缺水，同樣的水收益較高，"
           "但先放掉就沒了。表格的值是「從這個月、這個蓄水量開始，之後最多能得到的收益」。"),
    "kruskal": ("田區灌溉管線（最小生成樹）",
                "用管線把 9 塊田區連通，總長度最短。管線長度取兩點直線距離（km）。"),
    "hungarian": ("機具分派（指派問題）",
                  "5 台機具、5 塊田區，每台負責一塊。表中是完成作業所需時數，求總時數最少的分派。"),
    "two_opt": ("巡田路線（旅行推銷員問題）",
                "從田區 1 出發走遍 9 塊田區再回來，求最短路線（km）。"),
    "tabu": ("巡田路線（旅行推銷員問題）",
             "和 2-opt 頁相同的 9 塊田區。從 2-opt 停下來的局部最佳出發。"),
}


def combo_lab(spec: dict) -> dict:
    """API entry: run one method with optional parameters."""
    out = RUNNERS[spec["method"]](**spec.get("params", {}))
    out["method"] = spec["method"]
    out["name"], out["note"] = META[spec["method"]]
    return out

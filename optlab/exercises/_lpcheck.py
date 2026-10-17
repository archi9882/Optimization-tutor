"""Layer 2-3 checks shared by the LP-family exercises."""
from __future__ import annotations

import math


def as_trace(run_out: dict, objective) -> dict:
    """Turn an LP runner result into the {xs, fs, status} shape plots expect."""
    xs = [s["x"] for s in run_out["steps"]]
    return {"xs": xs, "fs": [objective(x) for x in xs],
            "status": run_out["status"], "n_iter": len(xs) - 1}


def compare_runs(user: dict, ref: dict, objective, goal: str) -> list[dict]:
    """Layer 2: same final answer.  Layer 3: same sequence of points."""
    ux, rx = user["steps"][-1]["x"], ref["steps"][-1]["x"]
    ok2 = user["status"] == ref["status"] and math.dist(ux, rx) < 1e-6
    msg2 = (f"結束狀態 {user['status']}，終點 ({ux[0]:.4g}, {ux[1]:.4g})，"
            f"目標值 {objective(ux):.6g}；參考 ({rx[0]:.4g}, {rx[1]:.4g})，{objective(rx):.6g}")
    n = min(len(user["steps"]), len(ref["steps"]))
    dev = max((math.dist(user["steps"][i]["x"], ref["steps"][i]["x"]) for i in range(n)),
              default=math.inf)
    ok3 = len(user["steps"]) == len(ref["steps"]) and dev < 1e-6
    msg3 = f"步數 {len(user['steps']) - 1} / 參考 {len(ref['steps']) - 1}，最大偏差 {dev:.2e}"
    return [{"layer": 2, "name": goal, "passed": ok2, "message": msg2},
            {"layer": 3, "name": "每一步都和參考解相同", "passed": ok3, "message": msg3}]

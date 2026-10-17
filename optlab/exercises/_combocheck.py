"""Layer 2-3 checks for exercises graded by running a whole combinatorial method."""
from __future__ import annotations


def compare(user_final, ref_final, user_seq, ref_seq, goal: str, fmt=str) -> list[dict]:
    """Layer 2: same final answer.  Layer 3: same sequence of intermediate states."""
    from ..grading import same
    ok2 = same(user_final, ref_final, 1e-6)
    ok3 = len(user_seq) == len(ref_seq) and all(same(u, r, 1e-6) for u, r in zip(user_seq, ref_seq))
    first = next((k for k, (u, r) in enumerate(zip(user_seq, ref_seq)) if not same(u, r, 1e-6)), None)
    msg3 = (f"步數 {len(user_seq) - 1} / 參考 {len(ref_seq) - 1}"
            + ("，每一步都相同" if ok3 else f"，第 {first if first is not None else min(len(user_seq), len(ref_seq))} 步開始不同"))
    return [{"layer": 2, "name": goal, "passed": ok2,
             "message": f"你的結果 {fmt(user_final)}；參考 {fmt(ref_final)}"},
            {"layer": 3, "name": "每一步都和參考解相同", "passed": ok3, "message": msg3}]

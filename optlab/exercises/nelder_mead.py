"""Exercise: one Nelder-Mead iteration."""
from __future__ import annotations

from .. import demos
from ..demos import nelder_mead as DEMO
from ..grading import Exercise, StepCase

SKELETON = demos.helpers_source("nelder_mead") + '''def nelder_mead_step(simplex, fvals, f):
    """一次 Nelder–Mead 迭代。

    simplex: 3 個頂點的 list（每個是長度 2 的 tuple）
    fvals:   對應的函數值（已經算好，不要重算）
    f:       目標函數；每呼叫一次就算一次「昂貴」的函數計算
    回傳:    (新的頂點 list, 新的函數值 list)，頂點順序不限（但兩個 list 要一一對應）
    可以使用上面的 along(c, w, t)。
    """
    # TODO 1: 依 fvals 由好到差排序；c = 除最差點外的重心
    # TODO 2: 反射 t = 1；比最好點好 → 試擴張 t = 2；比次差點好 → 接受反射
    # TODO 3: 否則收縮（外側 t = 0.5 / 內側 t = −0.5）；收縮也失敗 → 全部往最好點縮一半
    raise NotImplementedError
'''


def _bowl(p):
    return p[0] ** 2 + p[1] ** 2


def _far(p):
    return (p[0] - 3) ** 2 + (p[1] + 3) ** 2


def _cap(p):
    return -3 * p[0] ** 2 - 3 * p[1] ** 2 + 2 * p[0] * p[1] + p[0] + 2 * p[1]


_S = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]

# A deliberately bumpy function (table values, bowl elsewhere): the reflected
# point is between second-worst and worst, the outside contraction point is
# worse than the reflected one, so the step must shrink.
_TABLE = {(0.0, 1.0): 4.0, (1.0, -1.0): 2.0, (0.75, -0.5): 3.0}


def _bumpy(p):
    return _TABLE.get((round(p[0], 9) + 0.0, round(p[1], 9) + 0.0), p[0] ** 2 + p[1] ** 2)


def _as_set(out):
    """Unordered simplex: sort (vertex, f) pairs so any vertex order compares equal."""
    pts, fs = out
    return sorted(zip([tuple(float(v) for v in p) for p in pts], [float(v) for v in fs]))

EXERCISE = Exercise(
    id="nelder_mead",
    func_name="nelder_mead_step",
    skeleton=SKELETON,
    hints=[
        "所有試探點都在同一條線上：從最差點 w 穿過重心 c 往外走。along(c, w, t) 的 t 決定走多遠，t = 1 是反射。",
        "判斷順序：f_r < f(最好) → 擴張；f_r < f(次差) → 接受反射；否則收縮。外側收縮要求 f_c ≤ f_r，內側收縮要求 f_c < f(最差)。",
        "縮小時最好點不動，其他點 x_i ← x_best + 0.5·(x_i − x_best)，再重算這些點的 f。",
    ],
    solution=demos.source("nelder_mead"),
    blanks=("ci + t * (ci - wi)", "fr < fs[-2]", "along(c, worst, -0.5)",
            "b + 0.5 * (p - b)"),
    cases=[
        StepCase("接受反射：f(x_r) = 4 介於最好與次差之間",
                 {"simplex": [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)],
                  "fvals": [2.0, 5.0, 5.0], "f": _bowl},
                 ([(1.0, 1.0), (2.0, 1.0), (2.0, 0.0)], [2.0, 5.0, 4.0]), normalize=_as_set),
        StepCase("擴張：反射點比最好點還好，再往前走一倍",
                 {"simplex": _S, "fvals": [18.0, 13.0, 25.0], "f": _far},
                 ([(1.0, 0.0), (0.0, 0.0), (1.5, -2.0)], [13.0, 18.0, 3.25]), normalize=_as_set),
        StepCase("內側收縮：反射點比最差點還差",
                 {"simplex": _S, "fvals": [0.0, 1.0, 1.0], "f": _bowl},
                 ([(0.0, 0.0), (1.0, 0.0), (0.25, 0.5)], [0.0, 1.0, 0.3125]), normalize=_as_set),
        StepCase("縮小：收縮點也不夠好，全部往最好點 (1, 0) 縮一半",
                 {"simplex": _S, "fvals": [0.0, -2.0, -1.0], "f": _cap},
                 ([(1.0, 0.0), (0.5, 0.5), (0.5, 0.0)], [-2.0, 0.5, -0.25]), normalize=_as_set),
        StepCase("外側收縮點比反射點還差：不接受，改成縮小",
                 {"simplex": _S, "fvals": [0.0, 1.0, 4.0], "f": _bumpy},
                 ([(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)], [0.0, 0.25, 0.25]), normalize=_as_set),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "rosenbrock", "params": {"size": 0.5}, "n_iter": 300},
    loop_note="上方動畫用起點加上沿兩軸各走「初始邊長」的兩點組成第一個單純形，"
              "每一步呼叫一次 nelder_mead_step()，畫出新的三角形並以最好的頂點當作路徑上的點；"
              "外層迴圈也負責計數函數計算次數，三角形最長邊小於 1e-8 時停止。",
)

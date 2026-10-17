"""Exercise: one sweep of Powell's conjugate-direction method."""
from __future__ import annotations

import math

from .. import demos
from ..demos import powell as DEMO
from ..grading import Exercise, StepCase

SKELETON = demos.helpers_source("powell") + '''def powell_step(x, fx, dirs, f):
    """一輪 Powell 法。

    x, fx: 目前位置與函數值
    dirs:  方向集合（list，每個方向是長度 2 的 tuple）
    f:     目標函數
    回傳:  (x_new, f_new, dirs_new)
    可以使用上面的 line_min(f, x, fx, d)，它回傳 (x_new, f_new)。
    """
    # TODO 1: 依序沿 dirs 的每個方向 line_min，記下下降量最大的方向編號
    # TODO 2: u = 這一輪的總位移（正規化成單位長度）；位移為 0 就不更新方向
    # TODO 3: 刪掉下降最多的方向、把 u 加到最後，再沿 u 做一次 line_min
    raise NotImplementedError
'''

_R = 1 / math.sqrt(2)
_E = [(1.0, 0.0), (0.0, 1.0)]


def _q12(p):
    return p[0] ** 2 + 2 * p[1] ** 2


def _q41(p):
    return 4 * p[0] ** 2 + p[1] ** 2


def _bowl(p):
    return p[0] ** 2 + p[1] ** 2


EXERCISE = Exercise(
    id="powell",
    func_name="powell_step",
    skeleton=SKELETON,
    hints=[
        "先做 n 次一維搜尋，「這一輪總共走了多少」x − x_start 就是新方向的候選。",
        "要丟掉的是「下降量 f_before − f_after 最大」的那個方向：它的貢獻已經被新方向 u 涵蓋。",
        "刪除第 k 個元素再加到最後：dirs[:k] + dirs[k + 1:] + [u]。",
    ],
    solution=demos.source("powell"),
    blanks=("fx - f_new > biggest", "x[0] - x_start[0]",
            "dirs[:k_big] + dirs[k_big + 1:] + [u]"),
    cases=[
        StepCase("f = x² + 2y²：沿 y 下降較多，丟掉 e₂",
                 {"x": (1.0, 1.0), "fx": 3.0, "dirs": _E, "f": _q12},
                 ((0.0, 0.0), 0.0, [(1.0, 0.0), (-_R, -_R)])),
        StepCase("f = 4x² + y²：沿 x 下降較多，丟掉 e₁",
                 {"x": (1.0, 1.0), "fx": 5.0, "dirs": _E, "f": _q41},
                 ((0.0, 0.0), 0.0, [(0.0, 1.0), (-_R, -_R)])),
        StepCase("已在最小值：不動，方向集合不變",
                 {"x": (0.0, 0.0), "fx": 0.0, "dirs": _E, "f": _bowl},
                 ((0.0, 0.0), 0.0, _E)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "rosenbrock", "params": {}, "n_iter": 100},
    loop_note="上方動畫一步＝呼叫一次 powell_step()（一輪 n + 1 次一維搜尋），"
              "路徑只連起每一輪的終點；兩條短線是目前的方向集合。"
              "外層迴圈也負責計數函數計算次數，一輪移動小於 1e-8 時停止。",
)

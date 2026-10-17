"""Exercise: one compass (coordinate pattern) search poll."""
from __future__ import annotations

from .. import demos
from ..demos import pattern_search as DEMO
from ..grading import Exercise, StepCase

SKELETON = demos.helpers_source("pattern_search") + '''def compass_step(x, fx, step, f):
    """一次羅盤搜尋輪詢。

    x, fx: 目前位置（長度 2 的 tuple）與它的函數值
    step:  目前步長 h
    f:     目標函數
    回傳:  (x_new, f_new, step_new)
    依 DIRECTIONS 的順序試 x + step·d，第一個讓 f 變小的點就接受（步長不變）；
    四個方向都失敗就留在原地，步長減半。
    """
    # TODO
    raise NotImplementedError
'''


def _tilt(p):
    """Improves toward −x and (more) toward +y: the polling order decides."""
    return 10.0 + p[0] - 2.0 * p[1]


def _flat(p):
    """A plateau: equal values are not an improvement."""
    return 0.0


def _bowl(p):
    return p[0] ** 2 + p[1] ** 2


def _low(p):
    return p[0] ** 2 + (p[1] + 1) ** 2


EXERCISE = Exercise(
    id="pattern_search",
    func_name="compass_step",
    skeleton=SKELETON,
    hints=[
        "這個方法只比較函數值大小，不估計梯度：試點只有 x ± h·e₁、x ± h·e₂ 四個。",
        "「第一個改善就走」：在 for 迴圈裡一找到 fy < fx 就 return，不必把四個都算完。",
        "迴圈跑完都沒有 return，代表目前的網格太粗：return x, fx, step / 2。",
    ],
    solution=demos.source("pattern_search"),
    blanks=("x[0] + step * d[0]", "fy < fx", "step / 2"),
    cases=[
        StepCase("第二個方向（−x）成功",
                 {"x": (1.0, 1.0), "fx": 2.0, "step": 0.5, "f": _bowl},
                 ((0.5, 1.0), 1.25, 0.5)),
        StepCase("四個方向都失敗：原地不動、步長減半",
                 {"x": (0.0, 0.0), "fx": 0.0, "step": 0.5, "f": _bowl},
                 ((0.0, 0.0), 0.0, 0.25)),
        StepCase("依 DIRECTIONS 的順序：−x 先成功就移動，不再看 +y",
                 {"x": (0.0, 0.0), "fx": 10.0, "step": 1.0, "f": _tilt},
                 ((-1.0, 0.0), 9.0, 1.0)),
        StepCase("平台上函數值相同不算改善：步長減半",
                 {"x": (0.0, 0.0), "fx": 0.0, "step": 0.5, "f": _flat},
                 ((0.0, 0.0), 0.0, 0.25)),
        StepCase("前三個方向失敗，最後的 −y 成功",
                 {"x": (0.0, 0.0), "fx": 1.0, "step": 1.0, "f": _low},
                 ((0.0, -1.0), 0.0, 1.0)),
    ],
    adapter=DEMO.adapter,
    reference=DEMO.FUNC,
    demo={"problem": "quadratic", "problem_params": {"kappa": 10, "theta_deg": 30},
          "params": {"step0": 0.5}, "n_iter": 300},
    loop_note="上方動畫每一步呼叫一次 compass_step()，把回傳的 f 值與步長傳回下一步；"
              "菱形是下一輪要試的四個點。外層迴圈也負責計數函數計算次數，步長小於 1e-8 時停止。",
)

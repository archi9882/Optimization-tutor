"""Log-barrier interior-point demo for max c^T x s.t. A x <= b."""
from __future__ import annotations

import math

from ._linalg import dot, solve2

PSEUDOCODE = """\
φ_μ(x) = −cᵀx − μ Σ_i log(b_i − a_iᵀx)       # 障礙函數：靠近邊界時趨近 +∞
x ← 可行域內部的一點，μ ← μ₀
重複（外迴圈）：
    重複幾次（內迴圈）：對 φ_μ 做一步牛頓法，步長縮到仍在內部且 φ 下降
    μ ← μ / σ                                 # 障礙越來越弱，x 沿中心路徑走向頂點
"""


def barrier_newton_step(x, mu, c, A, b):
    """對障礙函數 φ_μ 做一步牛頓法；回溯步長，確保仍在可行域內部且 φ 下降。"""
    s = [bi - dot(ai, x) for ai, bi in zip(A, b)]
    g = [-c[j] + mu * sum(ai[j] / si for ai, si in zip(A, s)) for j in range(2)]
    H = [[mu * sum(ai[j] * ai[k] / si ** 2 for ai, si in zip(A, s))
          for k in range(2)] for j in range(2)]
    d = solve2(H, (-g[0], -g[1]))

    def phi(p):
        slacks = [bi - dot(ai, p) for ai, bi in zip(A, b)]
        if min(slacks) <= 0:
            return math.inf
        return -dot(c, p) - mu * sum(math.log(v) for v in slacks)

    t, f0 = 1.0, phi(x)
    while phi((x[0] + t * d[0], x[1] + t * d[1])) > f0 + 1e-4 * t * dot(g, d):
        t *= 0.5
    return (x[0] + t * d[0], x[1] + t * d[1])


FUNC = barrier_newton_step
HELPERS = [dot, solve2]
HEADER = "import math\n\n\n"

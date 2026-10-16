"""L-BFGS demo: two-loop recursion over the last m (s, y) pairs."""
from __future__ import annotations

from ..problems import as_vec
from ._linalg import backtracking, dot

PSEUDOCODE = """\
memory ← []                         # 最近 m 組 (s, y)
for k = 0, 1, 2, …:
    q ← ∇f(x)
    由新到舊：a_i ← ρ_i s_iᵀq,  q ← q − a_i y_i
    r ← γ·q,   γ = sᵀy / yᵀy（最新一組）
    由舊到新：b ← ρ_i y_iᵀr,  r ← r + (a_i − b) s_i
    d ← −r                           # 等同 −H·∇f，但從不存 H
    x_new ← 沿 d 做線搜尋
    把 (x_new − x, ∇f(x_new) − ∇f(x)) 加入 memory，只留最近 m 組
"""


def lbfgs_step(x, memory, m, f, grad_fn):
    """一步 L-BFGS：只記住最近 m 組 (s, y)，用兩迴圈遞推算出 H·g。"""
    g = grad_fn(x)
    q = list(g)
    alphas = []
    for s, y in reversed(memory):                    # 第一個迴圈：由新到舊
        rho = 1.0 / dot(y, s)
        a = rho * dot(s, q)
        q = [qi - a * yi for qi, yi in zip(q, y)]
        alphas.append((rho, a))
    if memory:                                       # 初始縮放 γ
        s, y = memory[-1]
        gamma = dot(s, y) / dot(y, y)
    else:
        gamma = 1.0
    r = [gamma * qi for qi in q]
    for (s, y), (rho, a) in zip(memory, reversed(alphas)):   # 第二個迴圈：由舊到新
        b = rho * dot(y, r)
        r = [ri + (a - b) * si for ri, si in zip(r, s)]
    d = (-r[0], -r[1])
    x_new = backtracking(f, x, f(x), g, d)
    s = (x_new[0] - x[0], x_new[1] - x[1])
    g_new = grad_fn(x_new)
    y = (g_new[0] - g[0], g_new[1] - g[1])
    if dot(s, y) > 1e-12:
        memory = (memory + [(s, y)])[-int(m):]
    return x_new, memory


def adapter(fn):
    """Lab loop: carry the list of (s, y) pairs."""
    def step(x, problem, state, p):
        memory = state.get("memory", [])
        x_new, memory = fn(x, memory, p["m"], problem.f, problem.grad)
        return as_vec(x_new), {"memory": memory}
    return step


FUNC = lbfgs_step
HELPERS = [dot, backtracking]
HEADER = ""

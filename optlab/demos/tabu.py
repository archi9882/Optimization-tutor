"""Tabu search over 2-opt moves for the travelling salesman problem."""
from __future__ import annotations

PSEUDOCODE = """\
tour ← 起始路線；best ← tour；禁忌表 ← 空
重複 K 次：
    在所有 2-opt 交換中，挑「交換後最短」的一個，但
        跳過禁忌表中的交換，除非它能打破歷史最佳（特赦）
    即使它讓路線變長也照樣執行                  # 這是跳出局部最優的關鍵
    把被拿掉的兩條邊記入禁忌表 tenure 回合      # 短期內不准把它們加回來
    若比 best 短：best ← 新路線
"""


def tabu_step(tour, D, tabu, best_len):
    """做一步禁忌搜尋。tabu 是 {邊: 剩餘回合} 的字典，邊用 (小編號, 大編號) 表示。
    回傳 (新路線, i, j, 拿掉的兩條邊)；沒有可走的交換時回傳 None。"""
    n = len(tour)
    cur = sum(D[tour[k]][tour[(k + 1) % n]] for k in range(n))
    best_new, choice = None, None
    for i in range(n - 1):
        for j in range(i + 2, n if i > 0 else n - 1):
            a, b = tour[i], tour[i + 1]
            c, d = tour[j], tour[(j + 1) % n]
            new_len = cur - D[a][b] - D[c][d] + D[a][c] + D[b][d]
            added = [tuple(sorted(e)) for e in ((a, c), (b, d))]
            is_tabu = any(e in tabu for e in added)
            if is_tabu and new_len >= best_len - 1e-9:
                continue                        # 禁忌且無法特赦
            if best_new is None or new_len < best_new - 1e-12:
                best_new, choice = new_len, (i, j)
    if choice is None:
        return None
    i, j = choice
    removed = [tuple(sorted(e)) for e in ((tour[i], tour[i + 1]), (tour[j], tour[(j + 1) % n]))]
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:], i, j, removed


FUNC = tabu_step
HELPERS = []
HEADER = ""

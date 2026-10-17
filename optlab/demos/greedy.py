"""Kruskal's algorithm: the greedy method on a matroid (minimum spanning tree)."""
from __future__ import annotations

PSEUDOCODE = """\
把所有邊依長度由短到長排序
每個點自成一組
依序看每條邊 (u, v)：
    若 u、v 在不同組：選它，並把兩組合併        # 貪婪：當下最短且不形成迴圈
    否則：跳過                                 # 選了會形成迴圈
選滿 n − 1 條邊時停止
"""


def find(parent, i):
    """沿著 parent 找到 i 所在組的代表點（並把路徑壓平）。"""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def kruskal_step(edge, parent):
    """考慮一條邊 (長度, u, v)：若 u、v 在不同組就合併並回傳 True，否則回傳 False。"""
    _, u, v = edge
    ru, rv = find(parent, u), find(parent, v)
    if ru == rv:
        return False
    parent[ru] = rv
    return True


FUNC = kruskal_step
HELPERS = [find]
HEADER = ""

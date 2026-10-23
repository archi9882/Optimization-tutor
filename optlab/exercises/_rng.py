"""A scripted stand-in for random.Random in hand-checkable step cases."""
from __future__ import annotations


class Draws:
    """rng.random() returns the listed numbers in order (an error when they run out)."""

    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def random(self) -> float:
        if self.i >= len(self.values):
            raise RuntimeError(f"抽了超過 {len(self.values)} 個亂數：抽亂數的次數或順序和題目說明不同")
        self.i += 1
        return self.values[self.i - 1]

    def __repr__(self) -> str:
        return f"Draws({self.values})"

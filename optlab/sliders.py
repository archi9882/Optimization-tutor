"""Slider grids for the lab: every slider position is a "nice" number.

The browser slider moves over an index into the list returned by
`slider_values`, so a value quoted on a page (eta = 0.21, kappa = 50,
Delta0 = 0.5, ...) is reachable exactly instead of being approximated by
`toPrecision(3)` of a continuous position.

Rules (documented in README, section 用詞與數值):
  log scale    every number with two significant figures in [lo, hi]
               (1.0, 1.1, ..., 9.9 per decade), plus lo and hi
  linear scale a fixed step: the largest 1/2/5 x 10^k that is at most
               (hi - lo) / 50 and divides lo, hi and the default
  integer      values are rounded to integers and de-duplicated
               (linear steps are at least 1)
"""
from __future__ import annotations

import math

LINEAR_POSITIONS = 50   # a linear slider has at least this many steps


def _is_multiple(x: float, step: float) -> bool:
    q = x / step
    return abs(q - round(q)) < 1e-9 * max(1.0, abs(q))


def _decimals(step: float) -> int:
    return max(0, -math.floor(math.log10(step) + 1e-12))


def linear_step(lo: float, hi: float, default: float, integer: bool = False) -> float:
    """Largest 1/2/5 x 10^k <= (hi - lo) / LINEAR_POSITIONS dividing lo, hi, default."""
    target = (hi - lo) / LINEAR_POSITIONS
    k = math.floor(math.log10(target))
    for _ in range(12):
        for m in (5, 2, 1):
            s = float(f"{m}e{k}")
            if s <= target * (1 + 1e-12) and all(_is_multiple(v, s) for v in (lo, hi, default)):
                return max(s, 1.0) if integer else s
        k -= 1
    raise ValueError(f"no nice step for [{lo}, {hi}] with default {default}")


def slider_values(lo: float, hi: float, log: bool, default: float,
                  integer: bool = False) -> list[float]:
    """Sorted list of every value the slider can take."""
    if log:
        vals = {float(lo), float(hi)}
        for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
            for mant in range(10, 100):
                v = float(f"{mant}e{e - 1}")
                if lo <= v <= hi:
                    vals.add(v)
        if integer:
            vals = {float(round(v)) for v in vals}
    else:
        s = linear_step(lo, hi, default, integer)
        d = _decimals(s)
        n = round((hi - lo) / s)
        vals = {float(f"{lo + i * s:.{d}f}") for i in range(n + 1)}
    out = sorted(vals)
    if integer:
        return [int(v) for v in out]
    return out


def snap(values: list[float], v: float) -> float:
    """The grid value closest to v (relative distance on positive grids)."""
    if all(x > 0 for x in values) and v > 0:
        return min(values, key=lambda x: abs(math.log(x / v)))
    return min(values, key=lambda x: abs(x - v))

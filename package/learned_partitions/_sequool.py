"""SequOOL (Bartlett, Gabillon & Valko, ALT 2019) on a box, with trisection.

Maximizes ``value(z)`` over the box ``[lo, hi]`` in whatever coordinates the
caller chooses. The learned-partition method calls this in the coordinates of
the learned subspace; plain SequOOL calls it in the original coordinates.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


class BudgetExhausted(Exception):
    """Raised by the evaluation wrapper when the evaluation budget is used up."""


def max_evals_for_depth(h_max: int) -> int:
    """Exact number of evaluations SequOOL makes with depth ``h_max``.

    The root costs one evaluation; opening a cell costs two (the middle child
    keeps its parent's value); depth h >= 1 opens min(floor(h_max / h), cells
    available at depth h) cells, and each opened cell adds three children.
    """
    available, openings = 1, 0
    for h in range(h_max + 1):
        opened = min(1 if h == 0 else h_max // h, available)
        openings += opened
        available = 3 * opened
    return 1 + 2 * openings


def depth_for_budget(max_evals: int) -> int:
    """Largest depth whose full run fits in ``max_evals`` evaluations.

    SequOOL's original rule, h_max = floor(n / log-bar n), counts n as the
    number of openings and leaves roughly half of an evaluation budget unused;
    this picks the depth so that the whole budget is spent.
    """
    h = 1
    while max_evals_for_depth(h + 1) <= max_evals:
        h += 1
    return h


def sequool_maximize(
    value: Callable[[np.ndarray], float],
    lo: np.ndarray,
    hi: np.ndarray,
    max_evals: int,
) -> None:
    """Run SequOOL until ``max_evals`` evaluations of ``value`` have been made.

    ``value`` is responsible for tracking the best point; this routine only
    decides where to evaluate. Splits are trisections along coordinate
    ``h mod D`` at depth ``h``; at depth ``h >= 1`` the ``floor(h_max / h)``
    best cells are opened, as in SequOOL.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    dim = lo.size
    if max_evals < 1:
        return

    h_max = depth_for_budget(max_evals)

    center = (lo + hi) / 2.0
    # A cell is (value, lo, hi, center).
    levels: list[list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]] = [
        [(value(center), lo, hi, center)]
    ]

    try:
        for h in range(h_max + 1):
            cells = levels[h]
            if not cells:
                break
            k = 1 if h == 0 else math.floor(h_max / h)
            # Stable sort: ties keep insertion order, matching the paper's code.
            order = sorted(range(len(cells)), key=lambda i: -cells[i][0])[:k]
            axis = h % dim
            children: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
            for i in order:
                v, c_lo, c_hi, c_center = cells[i]
                a, b = c_lo[axis], c_hi[axis]
                third_lo, third_hi = (2 * a + b) / 3.0, (a + 2 * b) / 3.0

                left_hi = c_hi.copy()
                left_hi[axis] = third_lo
                left_center = c_center.copy()
                left_center[axis] = (5 * a + b) / 6.0

                right_lo = c_lo.copy()
                right_lo[axis] = third_hi
                right_center = c_center.copy()
                right_center[axis] = (a + 5 * b) / 6.0

                mid_lo, mid_hi = c_lo.copy(), c_hi.copy()
                mid_lo[axis], mid_hi[axis] = third_lo, third_hi

                children.append((value(left_center), c_lo, left_hi, left_center))
                children.append((value(right_center), right_lo, c_hi, right_center))
                children.append((v, mid_lo, mid_hi, c_center))
            levels.append(children)
    except BudgetExhausted:
        return

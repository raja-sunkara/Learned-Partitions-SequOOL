"""Public API: ``minimize`` and ``maximize``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.optimize import lsq_linear

from ._sequool import BudgetExhausted, sequool_maximize
from ._subspace import gradient_subspace, nn_subspace


@dataclass
class OptimizeResult:
    """Result of :func:`minimize` / :func:`maximize`.

    Attributes
    ----------
    x : best point found, in the original coordinates.
    fun : objective value at ``x``.
    nfev : number of objective evaluations used (never exceeds the budget).
    history : best objective value found after each evaluation.
    directions : (m, d) learned directions in normalized [-1, 1]^d coordinates,
        or ``None`` for ``method="sequool"``.
    method : the method that was run.
    """

    x: np.ndarray
    fun: float
    nfev: int
    history: np.ndarray
    directions: np.ndarray | None = None
    method: str = "learned"
    extra: dict = field(default_factory=dict)

    @property
    def m(self) -> int | None:
        return None if self.directions is None else int(self.directions.shape[0])


class _Objective:
    """Wraps the user's function: counts evaluations, tracks the best point."""

    def __init__(self, f, lower, upper, budget, sign):
        self.f, self.lower, self.upper = f, lower, upper
        self.half = (upper - lower) / 2.0
        self.budget, self.sign = budget, sign
        self.nfev = 0
        self.best_x: np.ndarray | None = None
        self.best = -np.inf  # best of sign * f, i.e. we always maximize internally
        self.history: list[float] = []

    def to_x(self, u: np.ndarray) -> np.ndarray:
        return self.lower + (np.clip(u, -1.0, 1.0) + 1.0) * self.half

    def __call__(self, u: np.ndarray) -> float:
        """Evaluate at normalized point ``u``; returns sign * f (to maximize)."""
        if self.nfev >= self.budget:
            raise BudgetExhausted
        x = self.to_x(u)
        val = self.sign * float(self.f(x))
        self.nfev += 1
        if not np.isfinite(val):
            val = -np.inf
        if val > self.best:
            self.best, self.best_x = val, x.copy()
        self.history.append(self.sign * self.best)
        return val


def _lift(directions: np.ndarray):
    """Map subspace coordinates z (length m) to a point u in [-1, 1]^d.

    Prefers u = A^T z. If that leaves the box, find the point in the box with
    A u = z that is closest to the subspace (adding only directions the
    function is insensitive to); if no such point exists, this degrades
    gracefully to the nearest point in the least-squares sense. This follows
    the procedure in Appendix 8 of the paper.
    """
    m, d = directions.shape
    proj_out = np.eye(d) - directions.T @ directions
    weight = 1e3
    system = np.vstack([weight * directions, proj_out])

    def lift(z: np.ndarray) -> np.ndarray:
        u = directions.T @ z
        if np.all(np.abs(u) <= 1.0):
            return u
        rhs = np.concatenate([weight * z, np.zeros(d)])
        return lsq_linear(system, rhs, bounds=(-1.0, 1.0), method="bvls").x

    return lift


def _extent(directions: np.ndarray, expand: float):
    # Range of a_i^T u over the box [-1, 1]^d is [-||a_i||_1, ||a_i||_1].
    radius = expand * np.abs(directions).sum(axis=1)
    return -radius, radius


def maximize(
    f: Callable[[np.ndarray], float],
    bounds: Sequence[Sequence[float]] | np.ndarray,
    budget: int = 1000,
    *,
    method: Literal["learned", "sequool"] = "learned",
    subspace: Literal["gradient", "nn"] = "gradient",
    m: int | None = None,
    n_learn: int | None = None,
    variance: float = 0.95,
    expand: float = 1.0,
    seed: int | None = 0,
    nn_kwargs: dict | None = None,
) -> OptimizeResult:
    """Maximize a black-box function ``f`` over a box. See :func:`minimize`."""
    return _run(f, bounds, budget, +1.0, method, subspace, m, n_learn, variance,
                expand, seed, nn_kwargs)


def minimize(
    f: Callable[[np.ndarray], float],
    bounds: Sequence[Sequence[float]] | np.ndarray,
    budget: int = 1000,
    *,
    method: Literal["learned", "sequool"] = "learned",
    subspace: Literal["gradient", "nn"] = "gradient",
    m: int | None = None,
    n_learn: int | None = None,
    variance: float = 0.95,
    expand: float = 1.0,
    seed: int | None = 0,
    nn_kwargs: dict | None = None,
) -> OptimizeResult:
    """Minimize a black-box function ``f`` over a box with a learned partition.

    Parameters
    ----------
    f : callable taking a 1-D array of length d and returning a float.
    bounds : sequence of (low, high) pairs, one per dimension, or a (d, 2) array.
    budget : total number of evaluations of ``f``, including those used to
        learn the subspace.
    method : ``"learned"`` (SequOOL on a learned subspace, Algorithm 1 of the
        paper) or ``"sequool"`` (plain SequOOL with axis-aligned trisection).
    subspace : how to learn the subspace. ``"gradient"`` (default) estimates
        gradients by finite differences at a few random points; it needs only
        numpy and is very accurate for noiseless objectives. ``"nn"`` fits a
        small neural network to random evaluations (Algorithm 1 of the paper;
        needs PyTorch) and is the better choice when ``f`` is noisy or not
        smooth enough for finite differences.
    m : subspace dimension. If ``None``, chosen to keep ``variance`` of the
        singular-value energy (95% by default, as in the paper).
    n_learn : evaluations spent learning the subspace. Defaults to
        ``min(budget // 3, 10 * (d + 1))`` for ``"gradient"`` (rounded down to
        a multiple of d + 1) and ``budget // 3`` for ``"nn"``.
    expand : factor by which to enlarge the search box in subspace
        coordinates. Values above 1 help when the estimated subspace is
        slightly off (see Lemma 4.5 of the paper).
    seed : random seed for sampling and network initialization.
    nn_kwargs : extra options for the network, e.g. ``{"readout": "weights"}``
        to read directions from the hidden-layer weights exactly as in the
        paper, or ``hidden``, ``epochs``, ``weight_decay``.

    Returns
    -------
    OptimizeResult with ``x``, ``fun``, ``nfev``, ``history`` and the learned
    ``directions``.
    """
    return _run(f, bounds, budget, -1.0, method, subspace, m, n_learn, variance,
                expand, seed, nn_kwargs)


def _run(f, bounds, budget, sign, method, subspace, m, n_learn, variance,
         expand, seed, nn_kwargs) -> OptimizeResult:
    b = np.asarray(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bounds must be a sequence of (low, high) pairs")
    lower, upper = b[:, 0], b[:, 1]
    if np.any(upper <= lower):
        raise ValueError("each bound needs low < high")
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be at least 1")
    d = lower.size

    obj = _Objective(f, lower, upper, budget, sign)
    rng = np.random.default_rng(seed)
    directions = None

    try:
        if method == "sequool":
            sequool_maximize(obj, -np.ones(d), np.ones(d), budget)
        elif method == "learned":
            directions = _learn(obj, d, budget, rng, subspace, m, n_learn,
                                variance, seed, nn_kwargs)
            lift = _lift(directions)
            lo, hi = _extent(directions, expand)
            remaining = budget - obj.nfev
            sequool_maximize(lambda z: obj(lift(z)), lo, hi, remaining)
        else:
            raise ValueError('method must be "learned" or "sequool"')
    except BudgetExhausted:
        pass

    return OptimizeResult(
        x=obj.best_x,
        fun=sign * obj.best,
        nfev=obj.nfev,
        history=np.asarray(obj.history),
        directions=directions,
        method=method,
    )


def _learn(obj, d, budget, rng, subspace, m, n_learn, variance, seed, nn_kwargs):
    if subspace == "nn":
        n = n_learn if n_learn is not None else max(2, budget // 3)
        n = min(n, budget - 1)
        u = rng.uniform(-1.0, 1.0, size=(n, d))
        y = np.array([obj(ui) for ui in u])
        finite = np.isfinite(y)
        directions, _ = nn_subspace(u[finite], y[finite], variance=variance, m=m,
                                    seed=0 if seed is None else int(seed),
                                    **(nn_kwargs or {}))
    elif subspace == "gradient":
        n = n_learn if n_learn is not None else max(d + 1, min(budget // 3, 10 * (d + 1)))
        n_points = max(1, min(n, budget - 1) // (d + 1))
        directions, _ = gradient_subspace(obj, n_points, d, rng, variance=variance, m=m)
    else:
        raise ValueError('subspace must be "nn" or "gradient"')
    return directions

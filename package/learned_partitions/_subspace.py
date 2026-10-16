"""Estimating the low-dimensional subspace a black-box function varies in.

Both estimators work in normalized coordinates u in [-1, 1]^d and return a
matrix with orthonormal rows (m x d).

* ``"nn"`` (Algorithm 1 in the paper): fit a one-hidden-layer ReLU network to
  function values at uniformly random points and take the top right singular
  vectors of its input gradients (default) or of its hidden-layer weight
  matrix (``readout="weights"``, the readout used in the paper).
* ``"gradient"``: estimate gradients by forward differences at random points
  and take the top right singular vectors of the stacked gradients (the
  idea behind Fornasier, Schnass & Vybiral, 2012, without the compressed-sensing
  step). Uses no PyTorch.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def choose_dim(singular_values: np.ndarray, variance: float, m: int | None) -> int:
    if m is not None:
        return int(min(max(m, 1), singular_values.size))
    energy = singular_values**2
    total = energy.sum()
    if total <= 0:
        return 1
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, variance) + 1)


def top_directions(matrix: np.ndarray, variance: float, m: int | None):
    """Top right singular vectors of ``matrix`` (rows are samples)."""
    _, s, vt = np.linalg.svd(matrix, full_matrices=False)
    k = choose_dim(s, variance, m)
    return vt[:k], s


def nn_subspace(
    u: np.ndarray,
    y: np.ndarray,
    *,
    variance: float = 0.95,
    m: int | None = None,
    readout: str = "gradients",
    hidden: int = 128,
    epochs: int = 2000,
    lr: float = 1e-2,
    weight_decay: float = 0.1,
    init_scale: float = 0.1,
    seed: int = 0,
):
    """Subspace from a one-hidden-layer ReLU network fitted to (u, y).

    ``readout="weights"`` takes the top right singular vectors of the
    hidden-layer weight matrix, as in Algorithm 1 of the paper.
    ``readout="gradients"`` (default) takes them from the network's input
    gradients at the training points. Both span the same space in the ideal
    case, but the gradient readout ignores hidden units the fit does not use,
    so the 95%-energy rule picks the right dimension far more reliably
    without per-problem tuning of the network.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            'subspace="nn" needs PyTorch. Install it with '
            '`pip install torch`, or use subspace="gradient".'
        ) from exc
    if readout not in ("gradients", "weights"):
        raise ValueError('readout must be "gradients" or "weights"')

    d = u.shape[1]
    torch.manual_seed(seed)
    x = torch.as_tensor(u, dtype=torch.float64)
    y = np.asarray(y, dtype=float)
    scale = y.std() or 1.0
    t = torch.as_tensor((y - y.mean()) / scale, dtype=torch.float64).reshape(-1, 1)

    first = torch.nn.Linear(d, hidden, dtype=torch.float64)
    second = torch.nn.Linear(hidden, 1, dtype=torch.float64)
    with torch.no_grad():
        # Small first-layer init + decoupled weight decay on that layer only:
        # weight the function never uses stays small, as in the analysis of
        # Mousavi-Hosseini et al. (2023) that the paper builds on.
        first.weight.mul_(init_scale)
    model = torch.nn.Sequential(first, torch.nn.ReLU(), second)
    opt = torch.optim.AdamW(
        [
            {"params": first.parameters(), "weight_decay": weight_decay},
            {"params": second.parameters(), "weight_decay": 0.0},
        ],
        lr=lr,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), t)
        loss.backward()
        opt.step()
        sched.step()

    if readout == "weights":
        matrix = first.weight.detach().numpy()  # hidden x d
    else:
        xg = x.clone().requires_grad_(True)
        model(xg).sum().backward()
        matrix = xg.grad.detach().numpy()  # n x d
    return top_directions(matrix, variance, m)


def gradient_subspace(
    f_u: Callable[[np.ndarray], float],
    n_points: int,
    d: int,
    rng: np.random.Generator,
    *,
    variance: float = 0.95,
    m: int | None = None,
    step: float = 1e-4,
):
    """Subspace from forward-difference gradients at ``n_points`` random points.

    Costs ``n_points * (d + 1)`` evaluations of ``f_u``.
    """
    grads = np.empty((n_points, d))
    eye = np.eye(d)
    for i in range(n_points):
        # Keep base points far enough inside the box that u + step*e_j stays inside.
        base = rng.uniform(-1 + step, 1 - step, size=d)
        f0 = f_u(base)
        grads[i] = [(f_u(base + step * eye[j]) - f0) / step for j in range(d)]
    # Drop gradients touched by non-finite values of f; if none survive, fall
    # back to the full space (plain SequOOL in a rotated frame).
    grads = grads[np.all(np.isfinite(grads), axis=1)]
    if grads.shape[0] == 0:
        return np.eye(d), np.ones(d)
    return top_directions(grads, variance, m)

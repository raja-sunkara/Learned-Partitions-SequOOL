"""Learned partitions for optimistic black-box optimization.

Reference implementation of Algorithm 1 from

    Raja Sunkara and Ardhendu Tripathy.
    "Adaptive Partitioning Schemes for Optimistic Optimization." ICML 2025.
    https://proceedings.mlr.press/v267/sunkara25a.html

Example
-------
A 50-dimensional function that only varies along 2 hidden directions:

>>> import numpy as np
>>> from learned_partitions import minimize
>>> rng = np.random.default_rng(0)
>>> A = np.linalg.qr(rng.standard_normal((50, 50)))[0][:2]
>>> f = lambda x: float(np.sum((A @ x - 0.3) ** 2))
>>> res = minimize(f, bounds=[(-1, 1)] * 50, budget=1000)
>>> res.m                      # learned subspace dimension
2
>>> res.fun < 1e-12
True
"""

from ._asktell import AskTellOptimizer
from ._minimize import OptimizeResult, maximize, minimize

__all__ = ["minimize", "maximize", "AskTellOptimizer", "OptimizeResult"]
__version__ = "0.2.0"

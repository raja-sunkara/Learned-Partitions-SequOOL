# learned-partitions

Black-box optimization that **learns the partition it searches over**.

Reference implementation of
*Adaptive Partitioning Schemes for Optimistic Optimization*,
Raja Sunkara and Ardhendu Tripathy, ICML 2025
([paper](https://proceedings.mlr.press/v267/sunkara25a.html) ·
[arXiv](https://arxiv.org/abs/2610.04039) ·
[experiment code](https://github.com/raja-sunkara/Learned-Partitions-SequOOL)).

Many real objectives in d dimensions only vary along a few hidden directions:
f(x) = g(Ax) with A of size m × d and m ≪ d. Partition-based optimizers like
SequOOL waste most of their budget splitting irrelevant axes. This package
spends a small part of the budget estimating those directions, then runs
SequOOL on partitions built in the m-dimensional subspace.

## Install

```bash
pip install "git+https://github.com/raja-sunkara/Learned-Partitions-SequOOL.git#subdirectory=package"
```

Only numpy and scipy are required. For the neural-network subspace estimator,
add PyTorch: `pip install torch`.

## Usage

```python
import numpy as np
from learned_partitions import minimize

# A 50-D function that only varies along 2 hidden directions
rng = np.random.default_rng(0)
A = np.linalg.qr(rng.standard_normal((50, 50)))[0][:2]
f = lambda x: float(np.sum((A @ x - 0.3) ** 2))

res = minimize(f, bounds=[(-1, 1)] * 50, budget=1000)
res.x          # best point found
res.fun        # its value (~1e-32 here; plain SequOOL reaches ~2e-6)
res.m          # learned subspace dimension (2)
res.history    # best value after each evaluation, for regret plots
```

`maximize` has the same signature. `method="sequool"` runs plain SequOOL
with axis-aligned trisection, as a baseline.

### Options

| Argument | Default | Meaning |
|---|---|---|
| `budget` | 1000 | Total evaluations of `f`, including subspace learning. Never exceeded. |
| `method` | `"learned"` | `"learned"` (this paper) or `"sequool"` (baseline). |
| `subspace` | `"gradient"` | How to find the directions: `"gradient"` (finite differences, numpy only) or `"nn"` (neural network surrogate, Algorithm 1 of the paper). |
| `m` | auto | Subspace dimension. By default, the smallest m keeping 95% of the singular-value energy. |
| `n_learn` | auto | Evaluations spent on learning the subspace. |
| `expand` | 1.0 | Enlarge the search box in subspace coordinates; >1 helps if the estimated subspace is slightly off. |
| `seed` | 0 | Random seed. |
| `nn_kwargs` | `None` | Network options: `readout` (`"gradients"` or `"weights"`), `hidden`, `epochs`, `weight_decay`, `lr`. |

### Which subspace estimator?

- **`"gradient"` (default).** Forward differences at about 10 random points,
  costing `10 * (d + 1)` evaluations. In our tests on noiseless 20-D rotated
  Rastrigin and power functions with a 2-D active subspace, it recovered the
  subspace exactly every time.
- **`"nn"`.** Fits a one-hidden-layer ReLU network to random evaluations. Use
  it when `f` is noisy or non-smooth, so finite differences are unreliable.
  It works well on smooth functions (subspace error about 2°), but it needs
  enough samples to fit `f`. On highly oscillatory functions such as 20-D
  Rastrigin with 500 samples, it overestimates m. `nn_kwargs={"readout":
  "weights"}` reads directions from the hidden-layer weights exactly as in the
  paper; the default reads them from the network's input gradients, which
  picks m more reliably without tuning.

## How it works

1. Map the box to [-1, 1]^d and estimate an orthonormal basis Â (m × d) of the
   directions `f` varies in.
2. Search z in the box {z : |z_i| ≤ ‖â_i‖₁}, which contains Âx for every x in
   [-1, 1]^d, with SequOOL: trisect cells along one subspace coordinate per
   depth, and at depth h open the ⌊h_max/h⌋ best cells.
3. Evaluate `f` at Âᵀz. If that point leaves the box, use the point in the box
   with Âx = z closest to the subspace (Appendix 8 of the paper).

The SequOOL depth is chosen so that the whole budget is used.

## Citation

```bibtex
@InProceedings{pmlr-v267-sunkara25a,
  title     = {Adaptive Partitioning Schemes for Optimistic Optimization},
  author    = {Sunkara, Raja and Tripathy, Ardhendu},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  pages     = {58029--58064},
  year      = {2025},
  volume    = {267},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v267/sunkara25a.html}
}
```

MIT License.

# Adaptive Partitioning Schemes for Optimistic Optimization

[![ICML 2025](https://img.shields.io/badge/ICML-2025-blue)](https://proceedings.mlr.press/v267/sunkara25a.html)
[![arXiv](https://img.shields.io/badge/arXiv-2610.04039-b31b1b)](https://arxiv.org/abs/2610.04039)
[![OpenReview](https://img.shields.io/badge/OpenReview-rA2P4H7Dep-8c1b13)](https://openreview.net/forum?id=rA2P4H7Dep)

Official code for **"Adaptive Partitioning Schemes for Optimistic Optimization"**
by [Raja Sunkara](mailto:rs5cq@mst.edu) and Ardhendu Tripathy, *ICML 2025*.

📄 [Paper (PMLR)](https://proceedings.mlr.press/v267/sunkara25a.html) ·
[PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/sunkara25a/sunkara25a.pdf) ·
[arXiv](https://arxiv.org/abs/2610.04039) ·
[ICML poster](https://icml.cc/virtual/2025/poster/43904) ·
[Cite](#citation)

---

## TL;DR

Partition-based black-box optimizers (DiRect, SOO, SequOOL) split the search space
into axis-aligned boxes. That wastes evaluations when the function only varies along a
few directions, and it scales badly with dimension.

We **learn the partition instead of fixing it**: fit a small neural network to a few
evaluations, read off the low-dimensional subspace that explains the function's
variation, and run SequOOL on partitions built in that subspace. An interleaved variant
keeps refitting the network and uses it to choose which direction to split next.

- **Theory.** For multi-index functions $f(\mathbf{x}) = g(\mathbf{A}\mathbf{x})$ with an
  $m$-dimensional subspace inside $d$ dimensions, the regret improves from
  $\tilde O(n^{-1/\eta})$ (SequOOL) to $\tilde O(n^{-\beta/\eta})$ with
  $\beta = 1 + \frac{d-m}{2m-1}$, even when the subspace is only estimated.
- **Practice.** Gradient-free, works with any black-box `f(x)`, and competitive with
  REMBO, HeSBO, RESOO, CMA-ES and DiRect on rotated BBOB-style benchmarks.
- **New application.** Recasts LLM weight quantization (AWQ scale search on OPT-1.3B)
  as a 72-dimensional black-box problem.

| Optimizer | Partitioning | Regret on multi-index $f$ |
|---|---|---|
| SOO | fixed, axis-aligned | $\tilde O(n^{-1/\eta})$ |
| SequOOL | fixed, axis-aligned | $\tilde O(n^{-1/\eta})$ |
| **Ours** | **learned subspace + adaptive splits** | $\tilde O(n^{-\beta/\eta})$, $\beta > 1$ |

---

## Installation

Requires **Python 3.10**. The HeSBO/REMBO baselines are pulled in as a git submodule,
so clone with `--recursive`:

```bash
git clone --recursive https://github.com/raja-sunkara/Learned-Partitions-SequOOL.git
cd Learned-Partitions-SequOOL

conda create -n bbopt python=3.10 -y
conda activate bbopt
pip install -e .
```

Already cloned without `--recursive`? Run `git submodule update --init`.

---

## Quick start

Every run is configured with [Hydra](https://hydra.cc); defaults live in `configs/run.yaml`.

```bash
# Our method (SequOOL on a learned subspace) on a 5-D Rastrigin with a 2-D active subspace
python3 scripts/main.py method=learned_subspace function_name=rastrigin_rotated n_dim=5 low_dim=2

# Baseline SequOOL on the same problem
python3 scripts/main.py method=sequool function_name=rastrigin_rotated n_dim=5 low_dim=2

# Several methods in parallel
python3 scripts/main.py --multirun method=learned_subspace,sequool,resoo,rembo function_name=branin
```

Results and logs are written to `experiments/hydra_outputs/` (single runs) or
`multirun/` (sweeps).

### Methods (`method=`)

| Key | Algorithm |
|---|---|
| `learned_subspace` | **Ours:** two-stage, learned subspace + SequOOL (Alg. 1) |
| `dimension_reduction_and_look_ahead` | **Ours:** interleaved learning + lookahead splits (Alg. 2) |
| `all_dimensions_look_ahead` | **Ours:** lookahead direction selection in the full space |
| `subspace_random_search` | Random search on the learned subspace |
| `sequool`, `soo`, `resoo` | Optimistic optimization baselines |
| `direct`, `dual_annealing`, `cma_es`, `random_search` | Classic derivative-free baselines |
| `bayesian_opt`, `rembo`, `hesbo` | Bayesian optimization baselines |

### Benchmark functions (`function_name=`)

`sphere_rotated`, `rastrigin_rotated`, `ellipsoid_rotated`, `different_powers`,
`rosenbrock`, `styblinski_tang`, `hartmann6`, `branin`, `sharp_ridge`,
`attractive_sector`, `step_ellipsoid`, `custom_f1`, and more in
`src/utils/coco_functions.py`. Each is embedded as a multi-index function
$f(\mathbf{x}) = g(\mathbf{A}\mathbf{x})$ with a random orthonormal $\mathbf{A}$;
set `n_dim` ($d$) and `low_dim` ($m$).

---

## Repository layout

```
configs/      Hydra configs (run.yaml, per-function NN hyperparameters)
scripts/      main.py entry point
src/
  sequool.py, soo.py, resoo.py           optimistic optimization algorithms
  dimension_reduction*.py                learned-subspace methods (ours)
  all_dimensions_look_ahead.py           lookahead direction selection (ours)
  benchmarks.py                          DiRect, CMA-ES, BO, REMBO, ... wrappers
  utils/                                 test functions, NN training, plotting
HesBO/        submodule: https://github.com/aminnayebi/HesBO
```

---

## Citation

If you use this code or build on the paper, please cite:

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

## Acknowledgements

Supported by the National Science Foundation under Grant No. 2246187.
The HeSBO and REMBO baselines use the code of
[Nayebi et al. (2019)](https://github.com/aminnayebi/HesBO).

Questions or issues: open a GitHub issue or email rs5cq@mst.edu.

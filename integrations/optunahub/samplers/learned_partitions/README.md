---
author: Raja Sunkara and Ardhendu Tripathy
title: Learned Partition Sampler (SequOOL on a learned subspace)
description: Optimistic optimization that learns the few directions an objective depends on, then runs SequOOL on a partition built in that subspace. Suited to expensive, deterministic objectives with many continuous parameters of which only a few combinations matter.
tags: [sampler, black-box optimization, optimistic optimization, high-dimensional optimization, subspace learning, SequOOL]
optuna_versions: [5.0.0]
license: MIT License
---

## Abstract

Partition-based optimistic optimizers such as DiRect, SOO and SequOOL search by repeatedly splitting the domain into boxes along the coordinate axes. When an objective over d parameters only depends on m \<< d hidden combinations of them, f(x) = g(Ax), most of those splits are wasted.

This sampler implements the method of [Adaptive Partitioning Schemes for Optimistic Optimization](https://proceedings.mlr.press/v267/sunkara25a.html) (Sunkara & Tripathy, ICML 2025). It spends part of the trial budget estimating the active subspace, by finite differences by default or with a small neural network, and then runs SequOOL on a partition built in that subspace. For such multi-index functions the paper proves that the simple regret improves from O(n^(-1/eta)) for SequOOL to O(n^(-beta/eta)), with beta = 1 + (d - m)/(2m - 1).

In a 20-parameter study whose objective depends on 2 hidden directions, with 600 trials and one seed, the best values were:

| Sampler                 | Best value |
| ----------------------- | ---------- |
| LearnedPartitionSampler | 1.9e-17    |
| TPESampler (default)    | 1.2e-04    |
| RandomSampler           | 5.4e-03    |

The advantage is largest when the hidden function is fairly smooth near its optimum. On highly multimodal objectives such as Rastrigin, or narrow curved valleys such as Rosenbrock, it can do worse than plain SequOOL. The [Colab demo](https://colab.research.google.com/github/raja-sunkara/Learned-Partitions-SequOOL/blob/main/notebooks/demo.ipynb) shows both cases.

## APIs

- `LearnedPartitionSampler(n_trials: int, search_space: dict[str, BaseDistribution] | None = None, seed: int | None = None, **kwargs)`
  - `n_trials`: The number of trials the study will run. The algorithm plans its search for exactly this many evaluations, so pass the same number you give to `study.optimize`. Trials beyond it are sampled randomly, with a warning.
  - `search_space`: Optional fixed search space. If omitted, it is inferred from the first trial, which is sampled randomly.
  - `seed`: Seed for the random number generators.
  - `**kwargs`: Options of the underlying optimizer, for example:
    - `m`: fix the subspace dimension (by default chosen to keep 95% of the singular-value energy).
    - `subspace="nn"`: estimate the subspace with a one-hidden-layer neural network (needs PyTorch) instead of finite differences. Use it for noisy or non-smooth objectives.
    - `n_learn`: number of trials spent learning the subspace.
    - `method="sequool"`: plain SequOOL with axis-aligned splits, for comparison.

Float and integer parameters are optimized; integers are rounded and log-scaled parameters are searched in log space. Categorical parameters are sampled randomly. Failed and pruned trials are treated as the worst possible value. The sampler supports single-objective, sequential studies (`n_jobs=1`).

## Installation

```shell
$ pip install -r https://hub.optuna.org/samplers/learned_partitions/requirements.txt
```

## Example

```python
import numpy as np
import optuna
import optunahub

# 20 parameters, but the objective only depends on 2 hidden combinations of them.
rng = np.random.default_rng(0)
A = np.linalg.qr(rng.standard_normal((20, 20)))[0][:2]


def objective(trial: optuna.Trial) -> float:
    x = np.array([trial.suggest_float(f"x{i}", -1, 1) for i in range(20)])
    return float(np.sum(np.abs(A @ x - 0.1) ** 1.5))


module = optunahub.load_module("samplers/learned_partitions")
sampler = module.LearnedPartitionSampler(n_trials=600, seed=0)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=600)
print(study.best_value)
```

## Others

- Paper: [PMLR](https://proceedings.mlr.press/v267/sunkara25a.html), [arXiv](https://arxiv.org/abs/2610.04039)
- Code, standalone package and experiments: [GitHub](https://github.com/raja-sunkara/Learned-Partitions-SequOOL)

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

from __future__ import annotations

import numpy as np
import optuna
import optunahub


# 20 parameters, but the objective only depends on 2 hidden combinations of them.
rng = np.random.default_rng(0)
A = np.linalg.qr(rng.standard_normal((20, 20)))[0][:2]


def objective(trial: optuna.Trial) -> float:
    x = np.array([trial.suggest_float(f"x{i}", -1, 1) for i in range(20)])
    return float(np.sum(np.abs(A @ x - 0.1) ** 1.5))


if __name__ == "__main__":
    module = optunahub.load_module("samplers/learned_partitions")
    sampler = module.LearnedPartitionSampler(n_trials=600, seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=600)
    print(study.best_value)

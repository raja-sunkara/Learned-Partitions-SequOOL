from __future__ import annotations

import math
from typing import Any
import warnings

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub


class LearnedPartitionSampler(optunahub.samplers.SimpleBaseSampler):
    """SequOOL on a learned low-dimensional partition (Sunkara & Tripathy, ICML 2025).

    The sampler spends part of the budget estimating the few directions the
    objective actually varies in, then runs the SequOOL optimistic optimizer on
    a partition built in that subspace. It suits expensive, deterministic
    objectives over many continuous parameters of which only a few
    combinations matter.

    Float and integer parameters are optimized; integers are rounded and
    log-scaled parameters are searched in log space. Categorical parameters,
    and parameters outside the first trial's search space, are sampled
    randomly. Only single-objective, sequential studies (``n_jobs=1``) are
    supported.

    Args:
        n_trials:
            The number of trials the study will run. The algorithm plans its
            search for exactly this many evaluations, so pass the same number
            you give to ``study.optimize``. Later trials fall back to random
            sampling.
        search_space:
            Optional fixed search space. If omitted, it is inferred from the
            first trial, which is sampled randomly.
        seed:
            Random seed.
        **kwargs:
            Passed to ``learned_partitions.AskTellOptimizer``, e.g. ``m`` (fix
            the subspace dimension), ``subspace="nn"`` (neural-network
            estimator, needs torch), ``n_learn`` or ``method="sequool"`` (plain
            SequOOL, for comparison).
    """

    def __init__(
        self,
        n_trials: int,
        search_space: dict[str, BaseDistribution] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(search_space, seed)
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        self._n_trials = int(n_trials)
        self._kwargs = kwargs
        self._optimizer: Any = None
        self._space: dict[str, BaseDistribution] | None = None
        self._pending: tuple[int, np.ndarray] | None = None
        self._warned = False

    def infer_relative_search_space(
        self, study: optuna.Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        if self._space is not None:
            return self._space
        space = super().infer_relative_search_space(study, trial)
        return {
            name: dist
            for name, dist in space.items()
            if isinstance(dist, (FloatDistribution, IntDistribution)) and not dist.single()
        }

    def sample_relative(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if not search_space:
            return {}
        if len(study.directions) > 1:
            raise ValueError("LearnedPartitionSampler supports single-objective studies only.")

        if self._optimizer is None:
            from learned_partitions import AskTellOptimizer

            self._space = dict(search_space)
            bounds = [self._internal_bounds(d) for d in self._space.values()]
            direction = "maximize" if study.direction == StudyDirection.MAXIMIZE else "minimize"
            budget = self._n_trials - sum(
                1 for t in study.get_trials(deepcopy=False) if t.number < trial.number
            )
            self._optimizer = AskTellOptimizer(
                bounds,
                budget=max(budget, 1),
                direction=direction,
                seed=self._seed,
                **self._kwargs,
            )

        if self._pending is not None:
            # Another trial is still running (n_jobs > 1): sample this one randomly.
            self._warn_once(
                "LearnedPartitionSampler is sequential; concurrent trials are sampled randomly."
            )
            return {}

        x = self._optimizer.ask()
        if x is None:
            self._warn_once(
                f"LearnedPartitionSampler used its budget of {self._n_trials} trials; "
                "further trials are sampled randomly."
            )
            return {}
        self._pending = (trial.number, x)
        assert self._space is not None
        return {name: self._to_param(v, d) for (name, d), v in zip(self._space.items(), x)}

    def after_trial(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        state: TrialState,
        values: list[float] | None,
    ) -> None:
        if self._pending is None or self._pending[0] != trial.number:
            return
        _, x = self._pending
        self._pending = None
        # Failed or pruned trials count as the worst possible value.
        value = values[0] if state == TrialState.COMPLETE and values else float("nan")
        self._optimizer.tell(x, value)

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            warnings.warn(message, stacklevel=3)
            self._warned = True

    @staticmethod
    def _internal_bounds(dist: BaseDistribution) -> tuple[float, float]:
        assert isinstance(dist, (FloatDistribution, IntDistribution))
        if dist.log:
            return math.log(dist.low), math.log(dist.high)
        if isinstance(dist, IntDistribution):
            # Widen by half a step so every integer gets an equal share of the interval.
            return dist.low - 0.5 * dist.step, dist.high + 0.5 * dist.step
        return float(dist.low), float(dist.high)

    @staticmethod
    def _to_param(value: float, dist: BaseDistribution) -> float | int:
        if dist.log:
            value = math.exp(value)
        if isinstance(dist, IntDistribution):
            step = dist.step
            v = dist.low + round((value - dist.low) / step) * step
            return int(min(max(v, dist.low), dist.high))
        assert isinstance(dist, FloatDistribution)
        if dist.step is not None:
            v = dist.low + round((value - dist.low) / dist.step) * dist.step
            return float(min(max(v, dist.low), dist.high))
        return float(min(max(value, dist.low), dist.high))

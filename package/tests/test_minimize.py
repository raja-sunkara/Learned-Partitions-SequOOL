import numpy as np
import pytest

from learned_partitions import maximize, minimize
from learned_partitions._sequool import depth_for_budget, max_evals_for_depth


def multi_index(d=10, m=2, seed=0):
    """f(x) = g(Ax) with a random orthonormal A (m x d); minimum 0 inside the box."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    a = q[:m]
    z_star = a @ rng.uniform(-0.5, 0.5, d)
    return (lambda x: float(np.sum((a @ x - z_star) ** 2))), a


def angle_deg(est, true):
    from scipy.linalg import subspace_angles
    return np.rad2deg(subspace_angles(est.T, true.T)).max()


@pytest.mark.parametrize("budget", [1, 2, 7, 50, 333, 1000])
@pytest.mark.parametrize("method", ["sequool", "learned"])
def test_never_exceeds_budget(budget, method):
    f, _ = multi_index()
    res = minimize(f, [(-1, 1)] * 10, budget=budget, method=method)
    assert 1 <= res.nfev <= budget
    assert len(res.history) == res.nfev


def test_sequool_eval_count_is_exact():
    calls = []
    f = lambda x: calls.append(1) or float(np.sum(x**2))
    for h in [1, 3, 10, 40]:
        calls.clear()
        budget = max_evals_for_depth(h)
        res = minimize(f, [(-1, 1)] * 3, budget=budget, method="sequool")
        assert res.nfev == len(calls) == budget
        assert depth_for_budget(budget) == h


def test_sequool_uses_most_of_budget():
    f = lambda x: float(np.sum(x**2))
    res = minimize(f, [(-1, 1)] * 20, budget=2000, method="sequool")
    assert res.nfev > 0.9 * 2000


def test_history_is_monotone_and_matches_result():
    f, _ = multi_index()
    res = minimize(f, [(-1, 1)] * 10, budget=400)
    assert np.all(np.diff(res.history) <= 0)
    assert res.history[-1] == pytest.approx(res.fun)
    assert f(res.x) == pytest.approx(res.fun)


def test_result_stays_in_bounds():
    f, _ = multi_index(d=6)
    bounds = [(-3, 2), (0, 5), (-1, 1), (10, 20), (-0.5, 0.5), (1, 2)]
    res = minimize(lambda x: f(np.clip(x, -1, 1)), bounds, budget=300)
    lo, hi = np.array(bounds).T
    assert np.all(res.x >= lo) and np.all(res.x <= hi)


def test_gradient_subspace_recovers_true_subspace():
    f, a = multi_index(d=20, m=2, seed=3)
    res = minimize(f, [(-1, 1)] * 20, budget=1000)
    assert res.m == 2
    assert angle_deg(res.directions, a) < 1.0
    assert res.fun < 1e-8


def test_learned_beats_sequool_on_multi_index():
    f, _ = multi_index(d=20, m=2, seed=4)
    b = [(-1, 1)] * 20
    learned = minimize(f, b, budget=1500)
    plain = minimize(f, b, budget=1500, method="sequool")
    assert learned.fun < plain.fun


def test_fixed_m_is_respected():
    f, _ = multi_index(d=8, m=2)
    res = minimize(f, [(-1, 1)] * 8, budget=400, m=3)
    assert res.directions.shape == (3, 8)
    assert np.allclose(res.directions @ res.directions.T, np.eye(3), atol=1e-8)


def test_maximize_is_minimize_of_negative():
    f, _ = multi_index(d=5)
    b = [(-1, 1)] * 5
    lo = minimize(f, b, budget=200, seed=1)
    hi = maximize(lambda x: -f(x), b, budget=200, seed=1)
    assert hi.fun == pytest.approx(-lo.fun)
    assert np.allclose(hi.x, lo.x)


@pytest.mark.parametrize("method", ["sequool", "learned"])
def test_nonfinite_values_are_skipped(method):
    f = lambda x: np.nan if x[0] > 0.5 else float(np.sum((x - 0.1) ** 2))
    res = minimize(f, [(-1, 1)] * 3, budget=300, method=method)
    assert np.isfinite(res.fun)


def test_all_nonfinite_gradients_fall_back_to_full_space():
    f = lambda x: np.inf if x[0] < 2 else 0.0  # never finite inside the box
    res = minimize(f, [(-1, 1)] * 3, budget=100)
    assert res.directions.shape == (3, 3)


def test_seed_makes_runs_reproducible():
    f, _ = multi_index()
    a = minimize(f, [(-1, 1)] * 10, budget=300, seed=7)
    b = minimize(f, [(-1, 1)] * 10, budget=300, seed=7)
    assert np.array_equal(a.x, b.x) and a.fun == b.fun


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(bounds=[(1, 0)]), "low < high"),
        (dict(bounds=[1, 2]), "pairs"),
        (dict(budget=0), "budget"),
        (dict(method="bogus"), "method"),
        (dict(subspace="bogus"), "subspace"),
    ],
)
def test_input_validation(kwargs, message):
    args = dict(f=lambda x: 0.0, bounds=[(-1, 1)] * 2, budget=10)
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        minimize(**args)


def test_nn_subspace_on_smooth_function():
    pytest.importorskip("torch")
    f, a = multi_index(d=10, m=2, seed=5)
    res = minimize(f, [(-1, 1)] * 10, budget=1200, subspace="nn",
                   nn_kwargs={"epochs": 800})
    assert res.m == 2
    assert angle_deg(res.directions, a) < 10.0

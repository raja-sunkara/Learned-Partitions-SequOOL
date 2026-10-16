import numpy as np
import pytest

from learned_partitions import AskTellOptimizer, maximize, minimize


def f(x):
    a = np.array([0.6, -0.8, 0.0, 0.0, 0.0])
    return float(abs(a @ x - 0.1) + 0.01 * np.sum(x**2))


def drive(opt, fn):
    xs = []
    while (x := opt.ask()) is not None:
        xs.append(x)
        opt.tell(x, fn(x))
    return xs


@pytest.mark.parametrize("kwargs", [dict(), dict(method="sequool"), dict(m=1, seed=3)])
def test_same_points_and_result_as_minimize(kwargs):
    seen = []
    ref = minimize(lambda x: seen.append(x.copy()) or f(x), [(-1, 1)] * 5, budget=150, **kwargs)
    opt = AskTellOptimizer([(-1, 1)] * 5, budget=150, **kwargs)
    xs = drive(opt, f)
    assert len(xs) == len(seen) == ref.nfev
    assert all(np.array_equal(a, b) for a, b in zip(xs, seen))
    assert opt.result.fun == ref.fun and np.array_equal(opt.result.x, ref.x)
    assert opt.finished and opt.ask() is None


def test_maximize_direction():
    ref = maximize(lambda x: -f(x), [(-1, 1)] * 5, budget=80)
    opt = AskTellOptimizer([(-1, 1)] * 5, budget=80, direction="maximize")
    drive(opt, lambda x: -f(x))
    assert opt.result.fun == ref.fun


def test_must_tell_before_asking_again():
    opt = AskTellOptimizer([(-1, 1)] * 2, budget=20)
    opt.ask()
    with pytest.raises(RuntimeError):
        opt.ask()
    opt.close()


def test_tell_checks_point():
    opt = AskTellOptimizer([(-1, 1)] * 2, budget=20)
    x = opt.ask()
    with pytest.raises(ValueError):
        opt.tell(x + 1.0, 0.0)
    opt.tell(x, 0.0)
    opt.close()


def test_tell_without_ask():
    opt = AskTellOptimizer([(-1, 1)] * 2, budget=20)
    with pytest.raises(RuntimeError):
        opt.tell(np.zeros(2), 0.0)
    opt.close()


@pytest.mark.parametrize("asked", [0, 1, 7])
def test_close_early_stops_worker(asked):
    opt = AskTellOptimizer([(-1, 1)] * 3, budget=500)
    for i in range(asked):
        x = opt.ask()
        if i < asked - 1:
            opt.tell(x, f(np.r_[x, 0, 0]))
    opt.close()
    assert opt.finished and opt.result is None and opt.ask() is None
    assert not opt._thread.is_alive()


def test_nan_values_are_worst():
    opt = AskTellOptimizer([(-1, 1)] * 2, budget=60)
    while (x := opt.ask()) is not None:
        opt.tell(x, np.nan if x[0] > 0 else float(np.sum((x + 0.3) ** 2)))
    assert np.isfinite(opt.result.fun) and opt.result.x[0] <= 0


def test_bad_arguments_fail_immediately():
    with pytest.raises(ValueError):
        AskTellOptimizer([(1, 0)], budget=10)
    with pytest.raises(ValueError):
        AskTellOptimizer([(0, 1)], budget=10, direction="up")
    with pytest.raises(TypeError):
        AskTellOptimizer([(0, 1)], budget=10, nonsense=1)


def test_worker_errors_surface_on_ask():
    opt = AskTellOptimizer([(-1, 1)] * 2, budget=20, subspace="bogus")
    with pytest.raises(ValueError, match="subspace"):
        while (x := opt.ask()) is not None:
            opt.tell(x, 0.0)

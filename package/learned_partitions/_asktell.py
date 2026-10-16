"""Ask/tell interface, for libraries that propose one point at a time.

Example
-------
>>> import numpy as np
>>> from learned_partitions import AskTellOptimizer
>>> opt = AskTellOptimizer(bounds=[(-1, 1)] * 3, budget=50)
>>> while (x := opt.ask()) is not None:
...     opt.tell(x, float(np.sum((x - 0.2) ** 2)))
>>> opt.result.nfev <= 50    # never more than the budget
True
"""

from __future__ import annotations

import queue
import threading
from typing import Sequence

import numpy as np

from ._minimize import OptimizeResult, _run


class _Stop(Exception):
    """Raised inside the worker when the optimizer is closed early."""


_DONE = object()


class AskTellOptimizer:
    """Run :func:`minimize` (or :func:`maximize`) one evaluation at a time.

    Call :meth:`ask` for the next point, evaluate it yourself, and report the
    value with :meth:`tell`. :meth:`ask` returns ``None`` once the budget is
    spent; :attr:`result` then holds the usual :class:`OptimizeResult`.
    Exactly one point is outstanding at a time.

    The algorithm runs in a background thread that waits for each told value,
    so the sequence of points is identical to calling :func:`minimize` with
    the same arguments. Call :meth:`close` to stop early.

    Parameters
    ----------
    bounds, budget : as in :func:`minimize`.
    direction : ``"minimize"`` (default) or ``"maximize"``.
    **kwargs : any other keyword of :func:`minimize` (``method``, ``subspace``,
        ``m``, ``n_learn``, ``seed``, ...).
    """

    def __init__(
        self,
        bounds: Sequence[Sequence[float]] | np.ndarray,
        budget: int = 1000,
        *,
        direction: str = "minimize",
        **kwargs,
    ) -> None:
        if direction not in ("minimize", "maximize"):
            raise ValueError('direction must be "minimize" or "maximize"')
        sign = -1.0 if direction == "minimize" else 1.0
        defaults = dict(method="learned", subspace="gradient", m=None, n_learn=None,
                        variance=0.95, expand=1.0, seed=0, nn_kwargs=None)
        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
        defaults.update(kwargs)

        self._asks: queue.Queue = queue.Queue(maxsize=1)
        self._tells: queue.Queue = queue.Queue(maxsize=1)
        self._pending: np.ndarray | None = None
        self._result: OptimizeResult | None = None
        self._error: BaseException | None = None
        self._finished = False

        def evaluate(x: np.ndarray) -> float:
            self._asks.put(np.array(x, dtype=float))
            value = self._tells.get()
            if value is _Stop:
                raise _Stop
            return value

        def work() -> None:
            try:
                self._result = _run(
                    evaluate, bounds, budget, sign, defaults["method"], defaults["subspace"],
                    defaults["m"], defaults["n_learn"], defaults["variance"],
                    defaults["expand"], defaults["seed"], defaults["nn_kwargs"],
                )
            except _Stop:
                pass
            except BaseException as exc:  # surfaced to the caller on the next ask()
                self._error = exc
            finally:
                self._asks.put(_DONE)

        # Validate arguments in the caller's thread so errors surface immediately.
        b = np.asarray(bounds, dtype=float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must be a sequence of (low, high) pairs")
        if np.any(b[:, 1] <= b[:, 0]):
            raise ValueError("each bound needs low < high")
        if int(budget) < 1:
            raise ValueError("budget must be at least 1")

        self._thread = threading.Thread(target=work, daemon=True, name="learned-partitions")
        self._thread.start()

    def ask(self) -> np.ndarray | None:
        """Next point to evaluate, or ``None`` when the run is over."""
        if self._pending is not None:
            raise RuntimeError("tell() the value of the previous point before asking again")
        if self._finished:
            return None
        item = self._asks.get()
        if item is _DONE:
            self._finished = True
            self._thread.join()
            if self._error is not None:
                raise self._error
            return None
        self._pending = item
        return item.copy()

    def tell(self, x: np.ndarray, value: float) -> None:
        """Report the objective value at ``x``, the point last returned by :meth:`ask`.

        Non-finite values (``nan``, ``inf``) are allowed and treated as the
        worst possible value.
        """
        if self._pending is None:
            raise RuntimeError("no point is waiting for a value; call ask() first")
        if not np.allclose(np.asarray(x, dtype=float), self._pending):
            raise ValueError("tell() must receive the point returned by the last ask()")
        self._pending = None
        self._tells.put(float(value))

    def close(self) -> None:
        """Stop the run early. :attr:`result` stays ``None``."""
        if self._finished:
            return
        if self._pending is not None:
            self._pending = None
            self._tells.put(_Stop)
        # Drain until the worker signals it is done.
        while True:
            item = self._asks.get()
            if item is _DONE:
                break
            self._tells.put(_Stop)
        self._finished = True
        self._thread.join()

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def result(self) -> OptimizeResult | None:
        """The final result once :meth:`ask` has returned ``None``."""
        return self._result

"""
Shared bootstrap confidence-interval utilities.

All functions accept a pre-seeded ``numpy.random.Generator`` rather than a
raw seed so callers control the random state explicitly and can chain calls
without seed aliasing.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

_NAN = float("nan")


def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Bootstrap CI for an arbitrary statistic of a 1-D sample.

    Parameters
    ----------
    values:
        1-D array of per-sample metric values.
    stat_fn:
        Callable mapping a 1-D array → scalar (e.g. ``np.mean``,
        ``lambda v: np.percentile(v, 5)``).
    n_boot:
        Number of bootstrap replicates.
    ci:
        Confidence level, e.g. ``0.95``.
    rng:
        NumPy Generator (``np.random.default_rng(seed)``).

    Returns
    -------
    ``(lo, hi)`` — the ``(1-ci)/2`` and ``(1+ci)/2`` percentiles of the
    bootstrap distribution.  Returns ``(NaN, NaN)`` when ``values`` is empty.
    """
    n = len(values)
    if n == 0:
        return _NAN, _NAN
    boot_stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[b] = stat_fn(sample)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_stats, 100.0 * alpha))
    hi = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))
    return lo, hi


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Bootstrap CI for the mean — convenience wrapper."""
    return bootstrap_ci(values, np.mean, n_boot, ci, rng)


def bootstrap_percentile_ci(
    values: np.ndarray,
    q: float,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Bootstrap CI for the *q*-th percentile.

    Parameters
    ----------
    q:
        Percentile in ``[0, 100]``.
    """
    return bootstrap_ci(values, lambda v: np.percentile(v, q), n_boot, ci, rng)


def bootstrap_proportion_ci(
    bool_values: np.ndarray,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Bootstrap CI for the proportion (mean) of a boolean array."""
    return bootstrap_mean_ci(bool_values.astype(np.float64), n_boot, ci, rng)

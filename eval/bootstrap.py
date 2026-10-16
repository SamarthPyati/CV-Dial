"""Seeded percentile bootstrap confidence intervals (PLAN §7: 95 % CI, 1,000 resamples on every MAE)."""

from __future__ import annotations

import zlib
from typing import Callable

import numpy as np


def bootstrap_ci(values, stat: Callable[[np.ndarray], float] = np.mean, resamples: int = 1000,
                 ci: float = 0.95, seed: int = 0, key: str = "") -> tuple[float, float]:
    """Percentile CI of `stat` over `values`. `key` decorrelates streams (e.g. 'S2|15-30') but stays deterministic."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    idx = rng.integers(0, len(v), size=(resamples, len(v)))
    stats = np.apply_along_axis(stat, 1, v[idx]) if stat is not np.mean else v[idx].mean(axis=1)
    a = (1 - ci) / 2
    lo, hi = np.quantile(stats, [a, 1 - a])
    return float(lo), float(hi)

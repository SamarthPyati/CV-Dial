import numpy as np

from eval.bootstrap import bootstrap_ci


def test_ci_brackets_mean_and_is_seeded():
    v = np.random.default_rng(0).exponential(1.0, 500)
    lo, hi = bootstrap_ci(v, seed=1, key="S2|all")
    assert lo < v.mean() < hi and hi - lo < 0.3
    assert (lo, hi) == bootstrap_ci(v, seed=1, key="S2|all")
    assert (lo, hi) != bootstrap_ci(v, seed=1, key="S1|all")


def test_ci_ignores_nan_and_handles_empty():
    assert bootstrap_ci([1.0, np.nan, 1.0]) == (1.0, 1.0)
    assert all(np.isnan(bootstrap_ci([np.nan])))


def test_median_stat():
    lo, hi = bootstrap_ci(np.arange(101.0), stat=np.median, resamples=200)
    assert lo < 50 < hi

import numpy as np
import pytest

from qfin.permutation import cross_pnl_matrix, run_permutations, permutation_pvalue


def test_cross_matrix_diagonal_is_real_pnl():
    rng = np.random.default_rng(0)
    weights = rng.random((20, 5))
    returns = rng.normal(0, 0.02, (20, 5))
    cross = cross_pnl_matrix(weights, returns)
    real_pnl = (weights * returns).sum(axis=1)
    #cross[t, t] = weights[t] . returns[t] = the strategy's actual return on day t
    assert np.allclose(np.diag(cross), real_pnl)


def test_run_permutations_is_seed_reproducible():
    cross = np.random.default_rng(1).normal(0, 0.01, (30, 30))
    a = run_permutations(cross, n_trials=50, seed=42)
    b = run_permutations(cross, n_trials=50, seed=42)
    assert np.array_equal(a, b)
    assert a.shape == (50, 30)


def test_pvalue_not_significant_under_null():
    #random weights on random returns = no real edge. the real (unshuffled) ordering shouldn't look
    #special against the random reshufflings, so its p-value should sit well away from 0.
    rng = np.random.default_rng(7)
    weights = rng.random((60, 8))
    returns = rng.normal(0, 0.02, (60, 8))
    cross = cross_pnl_matrix(weights, returns)
    real_final = np.cumprod(1.0 + np.diag(cross))[-1]
    curves = run_permutations(cross, n_trials=2000, seed=7)
    p = permutation_pvalue(curves, real_final)
    assert 0.05 < p < 0.95


def test_cpp_backend_matches_python_distribution():
    #the pybind11 backend and the numpy reference sample the same permutation distribution, so over
    #many trials their mean final equity should agree. skip cleanly if the extension isn't built.
    gaka_core = pytest.importorskip('gaka_core')
    rng = np.random.default_rng(11)
    cross = np.ascontiguousarray(rng.normal(0, 0.01, (40, 40)))
    n = 20000
    cpp = gaka_core.run_permutations_fast(cross, n)
    py = run_permutations(cross, n_trials=n, seed=11)
    assert cpp.shape == py.shape
    #both are noisy samples of the same process, compare means with a generous tolerance
    assert np.isclose(cpp[:, -1].mean(), py[:, -1].mean(), atol=0.02)

import numpy as np

#the statistical edge test. we shuffle the DAYS our real weights get applied to and ask: out of
#n random reshufflings, how often does luck beat the real ordering? that fraction is the p-value.
#
#why shuffle weight->day pairings and not the portfolio returns directly: compounding is
#commutative, (1+r1)(1+r2)... is order-independent, so permuting the final return series gives
#every trial the identical endpoint and a meaningless p-value. permuting which day each set of
#weights lands on instead tests whether the TIMING of the signals carries information.
#
#run_permutations here is a readable numpy reference. the heavy lifting in production is the
#pybind11 gaka_core backend, and tests/test_permutation.py checks the two agree.


def cross_pnl_matrix(weights_effective, returns):
    #precompute weights[i] . returns[j] for every (day i, day j) pair once. an (n*assets)@(assets*n)
    #matmul, so the inner per-trial loop is just O(1) lookups instead of redoing the dot product.
    #this is the change that cut the workload ~99.9% and let 500k trials run in under a second.
    w = np.ascontiguousarray(np.asarray(weights_effective, dtype=np.float64))
    r = np.ascontiguousarray(np.asarray(returns, dtype=np.float64))
    return np.ascontiguousarray(w @ r.T)


def run_permutations(cross_pnl, n_trials, seed=None):
    #pure-numpy reference: for each trial, shuffle the day order, walk the diagonal pick
    #cross_pnl[shuffled_day, t] and compound it into an equity curve.
    rng = np.random.default_rng(seed)
    n_days = cross_pnl.shape[0]
    out = np.empty((n_trials, n_days), dtype=np.float64)
    order = np.arange(n_days)
    for trial in range(n_trials):
        rng.shuffle(order)
        daily = cross_pnl[order, np.arange(n_days)]
        out[trial] = np.cumprod(1.0 + daily)
    return out


def permutation_pvalue(permutation_curves, real_final_value):
    #fraction of random trials whose final equity matched or beat the real strategy
    final_random = permutation_curves[:, -1]
    return np.sum(final_random >= real_final_value) / len(final_random)

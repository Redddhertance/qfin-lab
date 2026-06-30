import numpy as np
import pandas as pd

from qfin.backtest import lag_weights, run_backtest


def _frame(values):
    return pd.DataFrame(values, columns=['A'])


def test_lag_pushes_weights_one_day_forward():
    weights = _frame([1.0, 0.0, 0.0])
    effective = lag_weights(weights)
    #day 0 starts flat (nothing known yet), the day-0 target is only live on day 1
    assert effective['A'].tolist() == [0.0, 1.0, 0.0]


def test_no_lookahead_bias():
    #the whole point of the lag. put a huge return on day 2 and let the strategy "want" to be fully
    #invested on day 2. because weights are lagged, deciding weight=1 on day 2 captures day 3's
    #return, not day 2's, so the spike must NOT show up in pnl. a strategy that could see the future
    #would catch it. this test fails the instant someone removes the shift.
    returns = _frame([0.0, 0.0, 0.50, 0.0])
    weights = _frame([0.0, 0.0, 1.0, 0.0])  #fully invested on the spike day
    pnl, equity, _ = run_backtest(weights, returns, bps_per_turnover=0.0)
    assert np.isclose(pnl.iloc[2], 0.0)  #spike not captured
    assert np.allclose(equity, 1.0)      #no profit anywhere


def test_lagged_weight_captures_following_day():
    #the mirror of the above: a weight set on day 1 correctly earns day 2's return
    returns = _frame([0.0, 0.0, 0.10, 0.0])
    weights = _frame([0.0, 1.0, 0.0, 0.0])  #invested on day 1 -> live on day 2
    pnl, _, _ = run_backtest(weights, returns, bps_per_turnover=0.0)
    assert np.isclose(pnl.iloc[2], 0.10)


def test_flat_book_is_flat():
    returns = _frame([0.01, -0.02, 0.03, 0.0])
    weights = _frame([0.0, 0.0, 0.0, 0.0])
    pnl, equity, metrics = run_backtest(weights, returns)
    assert np.allclose(pnl, 0.0)
    assert np.allclose(equity, 1.0)
    assert metrics['pct_time_in_market'] == 0.0


def test_costs_reduce_return():
    #identical book run with and without costs, the costed one must end lower
    returns = _frame([0.0, 0.05, 0.05, 0.05])
    weights = _frame([1.0, 0.0, 1.0, 0.0])  #plenty of turnover
    _, eq_free, _ = run_backtest(weights, returns, bps_per_turnover=0.0)
    _, eq_costed, _ = run_backtest(weights, returns, bps_per_turnover=10.0)
    assert eq_costed.iloc[-1] < eq_free.iloc[-1]

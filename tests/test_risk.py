import numpy as np
import pandas as pd

from qfin.risk import (
    annualised_volatility,
    sharpe,
    max_drawdown,
    value_at_risk,
    conditional_var,
    beta,
)


def test_max_drawdown_monotonic_is_zero():
    equity = pd.Series([1.0, 1.1, 1.2, 1.5])
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_known_value():
    #peak of 2 then back to 1 is a 50% drawdown
    equity = pd.Series([1.0, 2.0, 1.0, 1.5])
    assert np.isclose(max_drawdown(equity), -0.5)


def test_sharpe_zero_vol_returns_zero():
    flat = pd.Series([0.0] * 100)
    assert sharpe(flat) == 0.0


def test_cvar_is_at_least_as_extreme_as_var():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0, 0.02, 5000))
    var = value_at_risk(returns, 0.05)
    cvar = conditional_var(returns, 0.05)
    #the average of the worst tail can't be milder than its cutoff
    assert cvar <= var


def test_beta_recovers_known_slope():
    rng = np.random.default_rng(3)
    market = pd.Series(rng.normal(0, 0.01, 2000))
    asset = 2.0 * market  #perfectly 2x the market
    assert np.isclose(beta(asset, market), 2.0)


def test_annualised_vol_scales_with_sqrt_time():
    rng = np.random.default_rng(4)
    daily = pd.Series(rng.normal(0, 0.01, 10000))
    out = annualised_volatility(daily)
    assert np.isclose(out, daily.std(ddof=0) * np.sqrt(252))

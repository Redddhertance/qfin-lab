import numpy as np
import pandas as pd

from qfin.features import sma, rsi, atr, atr_percent


def test_sma_of_constant_is_constant():
    prices = pd.Series([5.0] * 250)
    out = sma(prices, window=200)
    #first 199 are warmup NaN, everything after should just be the constant back again
    assert out.iloc[:199].isna().all()
    assert np.allclose(out.iloc[199:], 5.0)


def test_rsi_stays_in_bounds():
    rng = np.random.default_rng(0)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))
    out = rsi(prices, window=14).dropna()
    assert (out >= 0).all() and (out <= 100).all()


def test_rsi_pegs_high_when_only_gains():
    #a series that only ever goes up has no losses, so rs -> inf and rsi -> 100
    prices = pd.Series(np.arange(1, 100, dtype=float))
    out = rsi(prices, window=14).dropna()
    assert (out > 99.9).all()


def test_atr_of_flat_prices_is_zero():
    flat = pd.Series([10.0] * 100)
    out = atr(flat, flat, flat, window=14).dropna()
    assert np.allclose(out, 0.0)


def test_atr_percent_scales_by_price():
    rng = np.random.default_rng(1)
    close = pd.Series(50 + np.cumsum(rng.normal(0, 0.5, 200)))
    high = close + 1.0
    low = close - 1.0
    pct = atr_percent(high, low, close, window=14).dropna()
    assert (pct >= 0).all()
    assert (pct < 1).all()  #a 1-point range on a ~50 price is nowhere near 100%

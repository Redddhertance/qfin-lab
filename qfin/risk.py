import numpy as np
import pandas as pd

def realised_volatility(pnl: pd.Series, lookback: int = 60, freq: int = 252):
    r = pnl.rolling(lookback).std(ddof=0) * np.sqrt(freq)
    return r

def volatility_targeted_scale(pnl_proxy: pd.Series, target: float = 0.12, lookback: int = 60, cap: float = 1.5):
    realised_vol = realised_volatility(pnl_proxy, lookback=lookback)
    scale = target / realised_vol
    #clip already caps the div-by-zero inf at the leverage cap and vol can never be
    #negative, so the old trailing replace([inf, -inf]) was dead and pandas 2 deprecates
    #calling it without a value anyway. fillna covers the lookback warmup, no position until
    #there's enough history to measure vol
    return scale.clip(upper=cap).fillna(0.0)
import numpy as np
import pandas as pd

def realised_volatility(pnl: pd.Series, lookback: int = 60, freq: int = 252):
    r = pnl.rolling(lookback).std(ddof=0) * np.sqrt(freq)
    return r

def volatility_targeted_scale(pnl_proxy: pd.Series, target: float = 0.12, lookback: int = 60, cap: float = 1.5):
    realised_vol = realised_volatility(pnl_proxy, lookback=lookback)
    scale = target / realised_vol
    return scale.clip(upper=cap). fillna(0.0).replace([np.inf, -np.inf])
import pandas as pd
import numpy as np  

trading_days = 252

def to_series(x):
    return x if isinstance(x, pd.Series) else pd.Series(x)



class Summary:
    annual_ret: float
    annual_vol: float
    sharpe: float
    max_dd: float
    turnover: float
    n_days: int
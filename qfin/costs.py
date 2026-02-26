import pandas as pd
def linear_costs(weights: pd.DataFrame, bps_per_turnover: float = 3.0):
    #Cost (returned) = turnover * (bps / 1e4) (decimalisation)
    tw = weights.diff().abs().sum(axis=1).fillna(0.0)
    return tw * (bps_per_turnover / 1e4)
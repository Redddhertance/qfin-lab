import numpy as np
import pandas as pd

#portfolio/asset risk metrics. pulled out of the scanner so the backtest can reuse the exact
#same definitions, otherwise sharpe in one file silently disagrees with sharpe in the other.

TRADING_DAYS = 252


def annualised_volatility(returns, trading_days=TRADING_DAYS):
    #ddof=0 (population) on purpose, daily samples are plentiful so the small-sample correction
    #just adds noise and makes results harder to reconcile against other tools
    return returns.std(ddof=0) * np.sqrt(trading_days)


def sharpe(returns, risk_free=0.0, trading_days=TRADING_DAYS):
    vol = annualised_volatility(returns, trading_days)
    if vol == 0:
        return 0.0
    excess = returns.mean() * trading_days - risk_free
    return excess / vol


def max_drawdown(equity):
    #how far we fell from the running peak. cummax tracks the high-water mark, the rest is the
    #percentage gap below it, min() is the worst of those
    return (equity / equity.cummax() - 1.0).min()


def value_at_risk(returns, level=0.05):
    #historical var, the level-quantile of the daily return distribution (a negative number)
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    return np.percentile(clean, level * 100)


def conditional_var(returns, level=0.05):
    #cvar / expected shortfall, the average loss in the tail beyond var. captures how bad the bad
    #days actually are, which plain var hides
    clean = returns.dropna()
    var = value_at_risk(clean, level)
    tail = clean[clean <= var]
    if len(tail) == 0:
        return var
    return tail.mean()


def beta(asset_returns, market_returns):
    #regression beta via the covariance matrix, cov(asset, market) / var(market)
    aligned = pd.DataFrame({'a': asset_returns, 'm': market_returns}).dropna()
    if len(aligned) < 2:
        return np.nan
    matrix = np.cov(aligned['a'], aligned['m'])
    if matrix[1, 1] == 0:
        return np.nan
    return matrix[0, 1] / matrix[1, 1]

import numpy as np
import pandas as pd

trading_days = 252

def to_series(x):
    #coerce whatever gets passed in into a float series, squeezing single column frames
    if isinstance(x, pd.Series):
        return x.astype(float)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f'expected a single column, got {x.shape[1]}')
        return x.iloc[:, 0].astype(float)
    return pd.Series(x, dtype=float)

def clean(pnl):
    #scrubs the inf values that come out of zero-price glitches in the lowcap feed
    return to_series(pnl).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def equity_curve(pnl, start_value: float = 1.0):
    return start_value * (1.0 + clean(pnl)).cumprod()

def cagr(pnl, freq: int = trading_days):
    r = clean(pnl)
    if len(r) == 0:
        return 0.0
    total = float((1.0 + r).prod())
    if total <= 0.0:
        return -1.0 #wiped out, annualising a negative number is meaningless
    return total ** (freq / len(r)) - 1.0

def annual_volatility(pnl, freq: int = trading_days):
    r = clean(pnl)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(freq))

def sharpe(pnl, risk_free: float = 0.0, freq: int = trading_days):
    r = clean(pnl)
    vol = annual_volatility(r, freq)
    if vol == 0.0:
        return 0.0
    excess = r.mean() * freq - risk_free #excess over cash, not raw return
    return float(excess / vol)

def sortino(pnl, risk_free: float = 0.0, freq: int = trading_days):
    #same as sharpe but only penalises deviation below the daily hurdle
    r = clean(pnl)
    if len(r) < 2:
        return 0.0
    hurdle = risk_free / freq
    downside = np.minimum(r - hurdle, 0.0)
    dd = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(freq))
    if dd == 0.0:
        return 0.0
    return float((r.mean() * freq - risk_free) / dd)

def drawdown_series(pnl):
    eq = equity_curve(pnl)
    return eq / eq.cummax() - 1.0

def max_drawdown(pnl):
    dd = drawdown_series(pnl)
    return float(dd.min()) if len(dd) else 0.0

def drawdown_detail(pnl):
    #depth of the worst drawdown plus the dates it ran between, for the headline card
    dd = drawdown_series(pnl)
    if len(dd) == 0 or dd.min() == 0.0:
        return {'depth': 0.0, 'peak': None, 'trough': None, 'recovered': None, 'length_days': 0}
    trough = dd.idxmin()
    pre = dd.loc[:trough]
    peaks = pre[pre >= 0.0]
    peak = peaks.index[-1] if len(peaks) else pre.index[0] #last time we were at a high water mark
    post = dd.loc[trough:]
    recovered_at = post[post >= 0.0]
    recovered = recovered_at.index[0] if len(recovered_at) else None
    end = recovered if recovered is not None else dd.index[-1]
    return {
        'depth': float(dd.min()),
        'peak': peak,
        'trough': trough,
        'recovered': recovered,
        'length_days': int(len(dd.loc[peak:end]))
    }

def calmar(pnl, freq: int = trading_days):
    mdd = abs(max_drawdown(pnl))
    if mdd == 0.0:
        return 0.0
    return float(cagr(pnl, freq) / mdd)

def value_at_risk(pnl, level: float = 0.05):
    #historical var, just the 5th percentile of daily returns
    r = clean(pnl)
    if len(r) == 0:
        return 0.0
    return float(np.percentile(r, level * 100.0))

def conditional_var(pnl, level: float = 0.05):
    #mean of the tail past var, tells you how bad it gets once you're already in the tail
    r = clean(pnl)
    if len(r) == 0:
        return 0.0
    var = value_at_risk(r, level)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float(var)

def hit_rate(pnl):
    #ignores flat days so cash periods don't drag the number down
    r = clean(pnl)
    traded = r[r != 0.0]
    return float((traded > 0.0).mean()) if len(traded) else 0.0

def skewness(pnl):
    r = clean(pnl)
    sd = r.std(ddof=0)
    if len(r) < 3 or sd == 0.0:
        return 0.0
    return float((((r - r.mean()) / sd) ** 3).mean())

def excess_kurtosis(pnl):
    r = clean(pnl)
    sd = r.std(ddof=0)
    if len(r) < 4 or sd == 0.0:
        return 0.0
    return float((((r - r.mean()) / sd) ** 4).mean() - 3.0) #minus 3 so a normal reads 0

def rolling_sharpe(pnl, window: int = trading_days, risk_free: float = 0.0, freq: int = trading_days):
    r = clean(pnl)
    mean = r.rolling(window).mean() * freq - risk_free
    vol = r.rolling(window).std(ddof=0) * np.sqrt(freq)
    return (mean / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

def monthly_returns(pnl):
    #year by month grid for the heatmap
    r = clean(pnl)
    if len(r) == 0 or not isinstance(r.index, pd.DatetimeIndex):
        return pd.DataFrame()
    monthly = r.resample('ME').apply(lambda x: float((1.0 + x).prod() - 1.0))
    grid = pd.DataFrame({'year': monthly.index.year, 'month': monthly.index.month, 'ret': monthly.to_numpy()})
    return grid.pivot(index='year', columns='month', values='ret').sort_index()

def align(pnl, benchmark):
    #inner join on dates so a benchmark with a different calendar doesn't shift everything
    r, b = clean(pnl), clean(benchmark)
    joined = pd.concat([r.rename('r'), b.rename('b')], axis=1).dropna()
    return joined['r'].to_numpy(), joined['b'].to_numpy()

def beta_alpha(pnl, benchmark, risk_free: float = 0.0, freq: int = trading_days):
    #ols beta and the annualised residual, ie the bit the benchmark doesn't explain
    r, b = align(pnl, benchmark)
    if len(r) < 2:
        return 0.0, 0.0
    var_b = float(np.var(b, ddof=0))
    if var_b == 0.0:
        return 0.0, 0.0
    beta = float(np.cov(r, b, ddof=0)[0, 1] / var_b)
    hurdle = risk_free / freq
    alpha_daily = float(np.mean(r - hurdle) - beta * np.mean(b - hurdle))
    return beta, alpha_daily * freq

def tracking_error(pnl, benchmark, freq: int = trading_days):
    r, b = align(pnl, benchmark)
    if len(r) < 2:
        return 0.0
    return float(np.std(r - b, ddof=0) * np.sqrt(freq))

def information_ratio(pnl, benchmark, freq: int = trading_days):
    r, b = align(pnl, benchmark)
    te = tracking_error(r, b, freq)
    if te == 0.0:
        return 0.0
    return float(np.mean(r - b) * freq / te)

def capture_ratios(pnl, benchmark, period: str = 'ME'):
    #done on monthly returns like morningstar and every factsheet does it.
    #on daily returns you end up compounding hundreds of up-days into a massive
    #denominator and every ratio collapses towards zero, which says nothing about the strategy
    joined = pd.concat([clean(pnl).rename('r'), clean(benchmark).rename('b')], axis=1).dropna()
    if joined.empty:
        return 0.0, 0.0
    if period and isinstance(joined.index, pd.DatetimeIndex):
        joined = joined.resample(period).agg(lambda x: float((1.0 + x).prod() - 1.0))
    r, b = joined['r'].to_numpy(), joined['b'].to_numpy()

    def capture(mask):
        if mask.sum() == 0:
            return 0.0
        bench = float(np.mean(b[mask]))
        if bench == 0.0:
            return 0.0
        return float(np.mean(r[mask]) / bench)

    return capture(b > 0.0), capture(b < 0.0)

def annualised_turnover(weights, freq: int = trading_days):
    #one way turnover per year as a multiple of portfolio value
    if weights is None or len(weights) == 0:
        return 0.0
    daily = weights.diff().abs().sum(axis=1).fillna(0.0)
    return float(daily.mean() * freq)

def summarise(pnl, weights=None, benchmark=None, costs=None, risk_free: float = 0.0, freq: int = trading_days):
    #everything the tearsheet needs, in one pass
    r = clean(pnl)
    s = {
        'n_days': len(r),
        'start': None,
        'end': None,
        'risk_free': risk_free,
        'total_return': float((1.0 + r).prod() - 1.0) if len(r) else 0.0,
        'annual_return': cagr(r, freq),
        'annual_vol': annual_volatility(r, freq),
        'sharpe': sharpe(r, risk_free, freq),
        'sortino': sortino(r, risk_free, freq),
        'calmar': calmar(r, freq),
        'max_dd': max_drawdown(r),
        'max_dd_days': drawdown_detail(r)['length_days'],
        'var_95': value_at_risk(r),
        'cvar_95': conditional_var(r),
        'hit_rate': hit_rate(r),
        'skew': skewness(r),
        'excess_kurtosis': excess_kurtosis(r),
        'best_day': float(r.max()) if len(r) else 0.0,
        'worst_day': float(r.min()) if len(r) else 0.0,
        'annual_turnover': 0.0,
        'cost_drag': 0.0,
        'avg_positions': 0.0,
        'pct_time_invested': 0.0,
        'bench_annual_return': 0.0,
        'bench_annual_vol': 0.0,
        'bench_sharpe': 0.0,
        'bench_max_dd': 0.0,
        'excess_return': 0.0,
        'beta': 0.0,
        'alpha': 0.0,
        'tracking_error': 0.0,
        'information_ratio': 0.0,
        'up_capture': 0.0,
        'down_capture': 0.0
    }
    if len(r) and isinstance(r.index, pd.DatetimeIndex):
        s['start'], s['end'] = r.index[0], r.index[-1]

    if weights is not None and len(weights):
        s['annual_turnover'] = annualised_turnover(weights, freq)
        s['avg_positions'] = float((weights != 0.0).sum(axis=1).mean())
        s['pct_time_invested'] = float((weights.abs().sum(axis=1) > 0.0).mean())

    if costs is not None:
        s['cost_drag'] = float(clean(costs).mean() * freq)

    if benchmark is not None:
        b = clean(benchmark)
        s['bench_annual_return'] = cagr(b, freq)
        s['bench_annual_vol'] = annual_volatility(b, freq)
        s['bench_sharpe'] = sharpe(b, risk_free, freq)
        s['bench_max_dd'] = max_drawdown(b)
        s['excess_return'] = s['annual_return'] - s['bench_annual_return']
        s['beta'], s['alpha'] = beta_alpha(r, b, risk_free, freq)
        s['tracking_error'] = tracking_error(r, b, freq)
        s['information_ratio'] = information_ratio(r, b, freq)
        s['up_capture'], s['down_capture'] = capture_ratios(r, b)

    return s

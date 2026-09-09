import os
import sys

import gaka_core # type: ignore
import numpy as np
import pandas as pd
import yfinance as yf

#pathing fix due to recent vscode updates, ensures csv files are found correctly regardless of where script is run from. not sure why this is suddenly occuring more often, implemented in both scanner and backtest scripts
scriptdirectory = os.path.dirname(os.path.abspath(__file__))
projectroot = os.path.dirname(scriptdirectory)
sys.path.insert(0, projectroot) #lets the shared qfin package import when this is run as a script rather than a module

from qfin import costs as C
from qfin import data as D
from qfin import metrics as M
from qfin import universe as U
from qfin.tearsheet import build_tearsheet

START = '2022-01-01'
END = '2026-01-01'
#cash rate the sharpe is measured against. 2022-2026 had rates around 4-5%, so reporting
#raw return over vol rather than excess over cash was flattering the number by roughly 0.35
RISK_FREE = 0.04
#impact and the participation cap both scale with how big you are, so the backtest needs an
#assumed book size. 10m is small enough that the cap never binds on s&p names, raise it to
#see where capacity runs out
CAPITAL = 1e7
MAX_PARTICIPATION = 0.10 #never build a position needing more than 10% of a day's volume
MAX_WEIGHT = 0.05 #risk management, cap max alloc per asset at 5%
#the C++ backend hands back a full (trials x days) curve matrix, so 500k trials over a 4 year
#sample wants about 4gb of ram. drop it with GAKA_N_TRIALS=20000 for a quick run
N_TRIALS = int(os.environ.get('GAKA_N_TRIALS', 500_000))
REPORT_PATH = os.path.join(projectroot, 'reports', 'gaka_technicals_tearsheet.html')
PRICE_CACHE = os.path.join(projectroot, 'data', 'cache', 'sp500_prices.pkl')

#point in time universe. taking the index as it stood on each past date puts back the names
#that have since been acquired or dropped, which a current holdings file silently omits
membership = U.build_membership(START, END)
tickers = U.all_tickers(membership)
raw = D.download(tickers, START, END, cache_path=PRICE_CACHE)

#no bfill anywhere. a name is untradeable outside its real listed window rather than being
#given an invented flat price history that a 200 day trend filter would happily accept
prices = D.clean_prices(raw)
close, valid = prices['close'], prices['valid']
high, low, volume = prices['high'], prices['low'], prices['volume']
returns = D.to_returns(close, valid)
adv = D.dollar_volume(close, volume)
spread = C.liquidity_spread(adv)
daily_vol = returns.rolling(21, min_periods=5).std()
in_index = U.membership_mask(membership, close.index, close.columns)

priced = int(valid.any().sum())
print(f'Universe: {len(tickers)} names ever in the index, {priced} priceable, '
      f'{len(tickers) - priced} unavailable on yfinance')

spy_data = yf.download('SPY', start=START, end=END, auto_adjust=True, progress=False) #type: ignore
spy_close = pd.Series(spy_data['Close'].squeeze()) #type: ignore
spy_returns = spy_close.pct_change().fillna(0.0)

def atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = np.maximum(high_low, high_close)
    tr = np.maximum(tr, low_close) #true range, max of the three ranges for each stock
    #uses wilders smoothing method (exponential moving average with alpha = 1/window), which gives more weight to recent values, more reliable than SMA approximation
    return tr.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

def calculate_sma(prices, window=200):
    return prices.rolling(window=window, min_periods=window).mean()

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss))

validsma = close > calculate_sma(close)
validrsi = calculate_rsi(close) > 55
validatr = (atr(high, low, close) / close) < 0.05
spy_sma200 = spy_close.rolling(window=200, min_periods=200).mean()
market_regime = pd.DataFrame(
    np.tile((spy_close > spy_sma200).to_numpy().reshape(-1, 1), (1, close.shape[1])),
    index=close.index, columns=close.columns)

#a name has to pass the signal, be priceable that day, and actually be in the index that day
eligible = validsma & validrsi & validatr & market_regime & valid & in_index
rebalance_dates = pd.date_range(start=START, end=END, freq='ME')

weights_today = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
snap_dates = [eligible.index[eligible.index.get_indexer([d], method='nearest')[0]] for d in rebalance_dates]
for i, date in enumerate(snap_dates):
    eligible_stocks = eligible.loc[date][eligible.loc[date]].index.tolist()
    if not eligible_stocks:
        continue #nothing passes, the book sits in cash until the next rebalance
    weight = min(1.0 / len(eligible_stocks), MAX_WEIGHT)
    end_date = snap_dates[i + 1] if i + 1 < len(snap_dates) else eligible.index[-1]
    start_i, end_i = eligible.index.get_loc(date), eligible.index.get_loc(end_date)
    weights_today.iloc[start_i:max(end_i, start_i + 1),
                       weights_today.columns.get_indexer(eligible_stocks)] = weight

#hold through the month, but drop anything that stops being tradeable (delisting, halt)
weights_today = weights_today.where(valid, 0.0)
#refuses to hold more than the book could actually fill at the participation limit
weights_today = C.participation_cap(weights_today, adv, capital=CAPITAL,
                                    max_participation=MAX_PARTICIPATION)

weights_effective = weights_today.shift(1).fillna(0.0) #lag weights by one day to avoid lookahead bias (weights determined on t for t+1)
gross_p_return = (weights_effective * returns).sum(axis=1)
#spread on every trade plus square root impact that grows with participation, replacing the
#flat 2bp which was an institutional large cap number doing far too much work
costs, spread_cost, impact_cost = C.total_costs(weights_effective, spread, adv, daily_vol,
                                                capital=CAPITAL)
pnl = gross_p_return - costs

summary = M.summarise(pnl, weights=weights_effective, benchmark=spy_returns, costs=costs,
                      risk_free=RISK_FREE)
bench = M.summarise(spy_returns, risk_free=RISK_FREE)

print('\n=== GAKA Strategy Performance ===')
print(f"Days: {summary['n_days']}")
print(f"Annual Return: {summary['annual_return']:.2%}")
print(f"Annual Volatility: {summary['annual_vol']:.2%}")
print(f"Sharpe (excess of {RISK_FREE:.1%}): {summary['sharpe']:.2f}")
print(f"Max Drawdown: {summary['max_dd']:.2%}")
print(f"Annualised turnover: {summary['annual_turnover']:.1f}x")
print(f"Cost drag: {summary['cost_drag']:.2%} "
      f"(spread {spread_cost.mean() * 252:.2%}, impact {impact_cost.mean() * 252:.2%})")
print(f"\n=== Benchmark (SPY) ===")
print(f"Annual Return: {bench['annual_return']:.2%}")
print(f"Sharpe (excess of {RISK_FREE:.1%}): {bench['sharpe']:.2f}")
print(f"Max Drawdown: {bench['max_dd']:.2%}")
print(f"\n=== Alpha ===")
print(f"Outperformance: {summary['excess_return']:.2%}")
print(f"Average positions held: {summary['avg_positions']:.1f}")
print(f"Percent of days invested: {summary['pct_time_invested']:.1%}")

#permutation test: shuffle signal rows, not portfolio returns.
#shuffling portfolio returns is commutative under compounding — (1+r1)(1+r2)...(1+rT) is order-independent, so every permutation trial ends at the identical final value.
#shuffling weight rows instead tests whether the TIMING of our signals has predictive power:
#each trial applies the real allocations to randomly chosen days.
def permutation_test_fixed(returns, weights_effective, n_trials=N_TRIALS):
    real_pnl = (weights_effective * returns).sum(axis=1)
    real_equity = (1.0 + real_pnl).cumprod()

    #(days * assets) * (assets * days) = (days * days), transpose to align dimensions
    print(f'\nPrecomputing cross-PnL matrix for {len(returns)} days...')
    cross_pnl_matrix = np.ascontiguousarray(
        weights_effective.to_numpy(dtype=np.float64) @ returns.to_numpy(dtype=np.float64).T)
    print(f'Running {n_trials} permutation trials via C++ backend...')
    permutation_array = gaka_core.run_permutations_fast(cross_pnl_matrix, n_trials)

    final_returns_random = permutation_array[:, -1]
    final_return_real = float(real_equity.iloc[-1])
    p_value = float(np.sum(final_returns_random >= final_return_real) / n_trials)

    print(f'\n=== Permutation Test Results ===')
    print(f'Real Strategy Final Value: ${final_return_real:.2f}')
    print(f'Random Mean: ${np.mean(final_returns_random):.2f}')
    print(f'P-value: {p_value:.6f}')
    return p_value, final_returns_random, final_return_real

p_value, perm_finals, real_final = permutation_test_fixed(returns, weights_effective)

#tearsheet for the weekly IC, one self contained html file per run
report = build_tearsheet(
    pnl,
    weights=weights_effective,
    benchmark=spy_returns,
    costs=costs,
    title='GAKA Technicals - S&P 500 Momentum',
    subtitle=(f'Point-in-time S&P 500, above 200d SMA, RSI &gt; 55, ATR &lt; 5%, SPY regime '
              f'filter, monthly rebalance, {MAX_WEIGHT:.0%} position cap'),
    bench_label='SPY',
    risk_free=RISK_FREE,
    permutation={'final_values': perm_finals, 'real_final': real_final, 'p_value': p_value},
    notes=(f'Universe rebuilt point-in-time from index membership, so names that left the '
           f'index are included while they were members. {len(tickers) - priced} of '
           f'{len(tickers)} members could not be priced on yfinance, mostly acquisitions and '
           f'renames, so a residual survivorship gap remains. Costs are a liquidity-scaled '
           f'spread plus square-root impact on an assumed ${CAPITAL/1e6:.0f}m book.'),
    out_path=REPORT_PATH)
print(f'\nTearsheet written to {report}')

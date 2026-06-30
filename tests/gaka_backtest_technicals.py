import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import gaka_core  # type: ignore

from qfin.data import data_path, clean_close, to_returns
from qfin.features import sma, rsi, atr_percent
from qfin.backtest import run_backtest
from qfin.permutation import cross_pnl_matrix, permutation_pvalue
from qfin.utils import nearest_index, broadcast_series

#universe
TICKERS = pd.read_csv(data_path('lowcaptickers.csv'))['ticker'].tolist()[:1000]
#testing with lowcap universe
START = '2022-01-01'
END = '2026-01-01'

#data
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True)  # type: ignore
close = clean_close(raw['Close'].copy())  # type: ignore
returns = to_returns(close)

spy_data = yf.download('SPY', start=START, end=END, auto_adjust=True)  # type: ignore
spy_close = pd.Series(spy_data['Close'].squeeze())  # type: ignore
spy_returns = spy_close.pct_change().fillna(0.0)
spy_equity = (1.0 + spy_returns).cumprod()

#signal masks: above 200sma, rsi momentum, and a volatility cap via atr%
validsma = close > sma(close, window=200)
validrsi = rsi(close, window=14) > 55
validatr = atr_percent(raw['High'], raw['Low'], raw['Close'], window=14) < 0.05  # type: ignore

#market regime filter, no positions at all while spy is below its own 200sma
spy_sma200 = spy_close.rolling(window=200, min_periods=200).mean()
market_regime = spy_close > spy_sma200
market_regime_df = broadcast_series(market_regime, close.columns, close.index)
eligible = validsma & validrsi & validatr & market_regime_df

rebalance_dates = pd.date_range(start=START, end=END, freq='ME')
weights_today = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)  # type: ignore
for date in rebalance_dates:
    date = nearest_index(eligible.index, date)
    eligible_stocks = eligible.loc[date][eligible.loc[date] == True].index.tolist()
    if len(eligible_stocks) > 0:
        #risk management, cap max alloc per asset at 5%. if <20 names qualify the rest sits in cash
        MAX_WEIGHT = 0.05
        weight = min(1.0 / len(eligible_stocks), MAX_WEIGHT)

        current_idx = eligible.index.get_loc(date)
        if current_idx < len(eligible) - 1:
            future_rebalances = [d for d in rebalance_dates if d > date]
            if future_rebalances:
                next_idx = eligible.index.get_loc(nearest_index(eligible.index, future_rebalances[0]))
            else:
                next_idx = len(eligible)
            #hold these weights from this rebalance up to the day before the next one
            weights_today.loc[eligible.index[current_idx]:eligible.index[min(next_idx - 1, len(eligible) - 1)], eligible_stocks] = weight

#the engine applies the one-day lag and the turnover costs internally
pnl, equity, metrics = run_backtest(weights_today, returns)

trading_days = 252
spy_final_value = float(spy_equity.iloc[-1])
spy_annual = spy_final_value ** (trading_days / len(spy_returns)) - 1.0

print('=== GAKA Strategy Performance ===')
print(f"Days: {metrics['days']}")
print(f"Annual Return: {metrics['annual_return']:.2%}")
print(f"Annual Volatility: {metrics['annual_volatility']:.2%}")
print(f"Sharpe: {metrics['sharpe']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f'\n=== Benchmark (SPY) ===')
print(f'Annual Return: {spy_annual:.2%}')
print(f'\n=== Alpha ===')
print(f"Outperformance: {metrics['annual_return'] - spy_annual:.2%}")
print(f"Average positions held: {metrics['avg_positions']:.1f}")
print(f"Percent of days invested: {metrics['pct_time_in_market']:.1%}")

#permutation test: shuffle which DAYS the real weights land on, not the portfolio returns.
#shuffling portfolio returns is commutative under compounding so every trial ends identically.
#shuffling weight->day pairings instead tests whether the TIMING of the signals has predictive power.
def permutation_test_fixed(returns, weights_today, n_trials=500000):
    #re-derive the lagged book the same way the engine does, so the test matches the reported pnl
    weights_effective = weights_today.shift(1).fillna(0.0)
    real_pnl = (weights_effective * returns).sum(axis=1)
    real_equity = (1.0 + real_pnl).cumprod()

    print(f"Precomputing cross-PnL matrix for {len(returns)} days...")
    cross = cross_pnl_matrix(weights_effective, returns)

    print(f"\nRunning {n_trials} permutation trials via C++ backend...")
    permutation_array = gaka_core.run_permutations_fast(cross, n_trials)

    fig, ax = plt.subplots(figsize=(14, 8))
    #only draw the first 200 random curves, matplotlib chokes well before 500k lines
    for i in range(min(n_trials, 200)):
        ax.plot(real_equity.index, permutation_array[i], color='gray', alpha=0.3, linewidth=0.8)
    ax.plot(real_equity.index, real_equity.values, color='red', linewidth=2.5, label='Real Strategy', zorder=10)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (Log Scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f'Permutation Test (Corrected): Real Strategy vs {n_trials} Random Trials', fontsize=14, color='green')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('permutation_test_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()

    final_return_real = real_equity.iloc[-1]
    p_value = permutation_pvalue(permutation_array, final_return_real)

    print(f"\n=== Permutation Test Results ===")
    print(f"Real Strategy Final Value: ${final_return_real:.2f}")
    print(f"Random Mean: ${np.mean(permutation_array[:, -1]):.2f}")
    print(f"P-value: {p_value:.6f}")
    return p_value, permutation_array, real_equity

p_value, perm_curves, real_curve = permutation_test_fixed(returns, weights_today, n_trials=500000)

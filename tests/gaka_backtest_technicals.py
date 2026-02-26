import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
#universe
#TICKERS = pd.read_csv('tests/scanner/scticker.csv')['ticker'].tolist()[:300] 
TICKERS = pd.read_csv('data/lowcaptickers.csv')['ticker'].tolist()[:1000]
#testing with lowcap universe
START = '2022-01-01'
END = '2026-01-01'
#data
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True) #get adjusted prices via autoadjust
close = raw['Close'].copy() #get only closing prices
close = close.fillna(method='ffill').fillna(method='bfill').fillna(0) #drop rows without prices
returns = close.pct_change().fillna(0.0) #compute returns
#pct_change = percentage change between daily returns ((today - yesterday) / yesterday)
#fillna(0.0) fills vacant values with 0.0

spy_data = yf.download('SPY', start=START, end=END, auto_adjust=True)
spy = spy_data['Close']
spy_close = spy_data['Close'].squeeze()
spy_returns = spy.pct_change().fillna(0.0)
spy_equity = (1.0 + spy_returns).cumprod()

def atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = np.maximum(high_low, high_close)
    tr = np.maximum(tr, low_close) #true range, max of the three ranges for each stock
    atr = tr.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    #uses wilders smoothing method (exponential moving average with alpha = 1/window) to calculate ATR, which gives more weight to recent values, more reliable than SMA approximation
    return atr

def calculate_sma(prices, window=200):
    #price > 200 sma
    return prices.rolling(window=window, min_periods=window).mean()
def calcuate_rsi(close, window=14):
    period = 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

sma200 = calculate_sma(close)
validsma = close > sma200
rsi = calcuate_rsi(close)
validrsi = rsi > 55

atr_values = atr(raw['High'], raw['Low'], raw['Close'])
atrpercent = atr_values / close
validatr = atrpercent < 0.05
rebalance_dates = pd.date_range(start=START, end=END, freq='M')

weights_today = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
spy_sma200 = spy_close.rolling(200).mean()
market_regime = (spy_close > spy_sma200) #series
market_regime_df = pd.DataFrame(
    np.tile(market_regime.values.reshape(-1, 1), (1, len(close.columns))),
    index=close.index,
    columns=close.columns
)
eligible = validsma & validrsi & validatr & market_regime_df
for date in rebalance_dates:
    if date not in eligible.index:
        date = eligible.index[eligible.index.get_indexer([date], method='nearest')[0]]
    eligible_stocks = eligible.loc[date][eligible.loc[date] == True].index.tolist()
    #finds nearest day for rebalance, then puts eligible stocks on the day
    if len(eligible_stocks) > 0:
        #eligible stocks = equal weight
        weight = 1.0 / len(eligible_stocks)
        
        next_rebalance_idx = eligible.index.get_loc(date)
        if next_rebalance_idx < len(eligible) - 1:
            future_rebalances = [d for d in rebalance_dates if d > date]
            if future_rebalances:
                next_date = future_rebalances[0]
                if next_date not in eligible.index:
                    next_date = eligible.index[eligible.index.get_indexer([next_date], method='nearest')[0]]
                next_idx = eligible.index.get_loc(next_date)
            else:
                next_idx = len(eligible)
            
            #weight-setting
            weights_today.loc[eligible.index[next_rebalance_idx]:eligible.index[min(next_idx-1, len(eligible)-1)], eligible_stocks] = weight

#######n = returns.shape[1]
#shape provides df dimensions, shape[1] is for column count
#n = number of assets (SPY + QQQ, 2), allocates equal weights (n=2) every day each gets 0.5, creates df full of 0.5 with same shape as returns
########weights_today = pd.DataFrame(1.0 / n, index=returns.index, columns=returns.columns)#establishes equal weights
#create df of weights that have same shape as returns and gives each asset a weight of 50%
#weights rebalanced daily
weights_effective = weights_today.shift(1).fillna(0.0) #lag weights by one day to avoid lookahead bias (weights determined on t for t+1)
#fillna(0.0) fills first day with 0 as it will be null after shift(1)
#means that weights determined today will only be used tomorrow
gross_p_return = (weights_effective * returns).sum(axis=1) #portfolio return (gross)
#weight * return daily, then sums across columns for total daily return)
bps_per_turnover = 2.0 #2 bp per 100% turnover, for buying/selling the whole portfolio, 0.02% is lsot in cost
daily_turnover = weights_effective.diff().abs().sum(axis=1).fillna(0.0) #daily turnover, .diff calculates difference w previous day, abs value (direction irrelevant), sums across columns (sum(axis-1)), fills na w 0.0
costs = daily_turnover * (bps_per_turnover / 1e4) #transaction costs
#divide by 1e4 to convert bps into decimal form (0.0002), multiply by daily_turnover to get daily transaction costs
#If yesterday weights = [0.5, 0.5], today = [0.4, 0.6], turnover = |0.4–0.5| + |0.6–0.5| = 0.2 (20% of portfolio traded).
pnl = gross_p_return - costs #net pnl after costs
equity = (1.0 + pnl).cumprod()
#adds 1 to each return then multiplies them all together (cumprod). This gives cumulative portfolio value assuming it starts with $1
#data
trading_days = 252
annual_returnn = equity.iloc[-1] ** (trading_days / len(pnl)) - 1.0
#iloc[-1] gets last value of series, final portfolio value. (increases to 252/number of days to annualise growth (total growth into annual growth))
annual_volatility = pnl.std(ddof=0) * np.sqrt(trading_days)
#ddof=0 provides population standard deviation rather than small sample bias
#stnadard deviation of daily returns, scales by sqrt252 to annualise volatility
sharpe = (pnl.mean() * trading_days) / annual_volatility
max_dd = (equity / equity.cummax() - 1.0).min()
#dd = how much portfolio fell from previous high. equity cummax keeps track of maximum throughout year to maintain highest (cummax)
#dividing equity by cummax and subtracting 1 provides percentage drop
#min() takes lowest drop as max drawdown

spy_final_value = float(spy_equity.iloc[-1])
spy_annual = spy_final_value ** (trading_days / len(spy_returns)) - 1.0
#ensures value rather than series
avg_positions = (weights_effective > 0).sum(axis=1).mean()
pct_time_in_market = (weights_effective.sum(axis=1) > 0).mean()


print('=== GAKA Strategy Performance ===')
print(f'Days: {len(pnl)}')
print(f'Annual Return: {annual_returnn:.2%}')
print(f'Annual Volatility: {annual_volatility:.2%}')
print(f'Sharpe: {sharpe:.2f}')
print(f'Max Drawdown: {max_dd:.2%}')
print(f'\n=== Benchmark (SPY) ===')
print(f'Annual Return: {spy_annual:.2%}')
print(f'\n=== Alpha ===')
print(f'Outperformance: {annual_returnn - spy_annual:.2%}')
print(f"Average positions held: {avg_positions:.1f}")
print(f"Percent of days invested: {pct_time_in_market:.1%}")

def permutation_test_fixed(returns, weights_effective, n_trials=200):
    """
    Corrected Permutation Test: Shuffles strategy timing (weights) 
    against actual market returns.
    """
    
    real_pnl = (weights_effective * returns).sum(axis=1)
    real_equity = (1.0 + real_pnl).cumprod()
    
    permutation_curves = []
    
    print(f"Running {n_trials} permutation trials (Signal Shuffle)...")
    
    for trial in range(n_trials):
        #shuffling weights breaks link between strategy signal and return, randomness simulation
        shuffled_weights = weights_effective.sample(frac=1).reset_index(drop=True)
        
        #adjust index alignment
        shuffled_weights.index = returns.index
        
        #recalculate pnl
        trial_pnl = (shuffled_weights * returns).sum(axis=1)
        trial_equity = (1.0 + trial_pnl).cumprod()
        
        permutation_curves.append(trial_equity.values)
    
    permutation_array = np.array(permutation_curves)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i in range(n_trials):
        ax.plot(real_equity.index, permutation_array[i], 
                color='gray', alpha=0.3, linewidth=0.8)
    
    ax.plot(real_equity.index, real_equity.values, 
            color='red', linewidth=2.5, label='Real Strategy', zorder=10)
            
    ax.set_yscale('log')
    ax.set_title(f'Permutation Test (Signal Shuffle): Real Strategy vs {n_trials} Random Trials')
    ax.legend()
    plt.show()

    final_returns_random = permutation_array[:, -1]
    final_return_real = real_equity.iloc[-1]
    
    p_value = np.sum(final_returns_random >= final_return_real) / n_trials
    
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("strong (<0.05)")
    elif p_value < 0.10:
        print("marginal (<0.10)")
    else:
        print("insignificant")
    return p_value, permutation_array, real_equity

p_value, perm_curves, real_curve = permutation_test_fixed(returns, weights_effective, n_trials=200)
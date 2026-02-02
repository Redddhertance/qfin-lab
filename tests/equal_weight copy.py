import pandas as pd
import numpy as np
import yfinance as yf
TICKERS = ['SPY', 'QQQ']
START = '2018-01-01'
END = None
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True) #get adjusted prices via autoadjust
close = raw['Close'].copy() #get only closing prices
close = close.dropna(how='any') #drop rows without prices
returns = close.pct_change().fillna(0.0) #compute returns
#pct_change = percentage change between daily returns]
#fillna(0.0) fills vacant values with 0.0
l = 252
g = 21
signal = (close.shift(g) / close.shift(l)) - 1.0

def rank_to_longshort_weights(sig_row: pd.Series, long_franc=0.5, short_franc=0.5):
    s=sig_row.dropna()
    if s.empty:
        return pd.Series(0.0, index=sig_row.index)
    n=len(s)
    k_long = max(1, int(np.floor(n * long_franc)))
    k_short = max(1, int(np.floor(n * short_franc)))
    longs= s.nlargest(k_long).index
    shorts = s.nsmallest(k_short).index
    w= pd.Series(0.0, index=sig_row.index)
    w.loc[longs] = 1.0 / k_long
    w.loc[shorts] = -1.0 / k_short
    return w
weights_today = signal.apply(rank_to_longshort_weights, axis=1)
#create df of weights that have same shape as returns and gives each asset a weight of 50


weights_effective = weights_today.shift(1).fillna(0.0) #lag weights by one day to avoid lookahead bias (weights determined on t for t+1)
#fillna(0.0) fills first day with 0 as it will be null after shift(1)
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
print('SPY & QQQ Equal Weight Portfolio Performance:')
print(f'Days: {len(pnl)}')
print(f'Annual Return: {annual_returnn:.2%}')
print(f'Annual Volatility: {annual_volatility:.2%}')
print(f'Sharpe: {sharpe:.2f}')
print(f'Max Drawdown: {max_dd:.2%}')
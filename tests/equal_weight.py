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
n = returns.shape[1]
#shape provides df dimensions, shape[1] is for column count
#n = number of assets (SPY + QQQ, 2), allocates equal weights
weights_today = pd.DataFrame(1.0 / n, index=returns.index, columns=returns.columns)#establishes equal weights
#create df of weights that have same shape as returns and gives each asset a weight of 50%
#weights rebalanced daily
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
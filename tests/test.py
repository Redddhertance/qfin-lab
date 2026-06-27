import pandas as pd
import yfinance as yf
print('Enter ticker:')
ticker = input().strip().upper()
print(f'You entered ticker: {ticker}')
raw = yf.download(ticker, period='1d', interval='1m', auto_adjust=True)
if raw.empty:
    print(f'No data found for ticker: {ticker}')
    exit(1)
else:
    print(f'Data for {ticker} downloaded successfully.')
close = raw['Close'].copy()
returns = close.pct_change().fillna(0.0)
print(f'Returns for {ticker}:')
print(returns)
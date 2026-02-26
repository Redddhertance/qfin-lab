import numpy as np
import pandas as pd
import yfinance as yf

data = pd.read_csv('data/tickers.csv', index_col=0, parse_dates=True)
def get_ticker_data(ticker: str) -> pd.DataFrame:
    if ticker not in data.index:
        raise ValueError('Ticker{} not found in dataset'.format(ticker))
    return data.loc[ticker]

print(get_ticker_data(data.Tickers))
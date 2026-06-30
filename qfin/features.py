import numpy as np
import pandas as pd

#shared technical indicators. these used to live copy-pasted in both the scanner and the
#backtest, which meant a fix in one never made it to the other. single source of truth now.


def sma(prices, window=200):
    #price > 200 sma is the core trend filter everywhere in this repo
    return prices.rolling(window=window, min_periods=window).mean()


def rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    #wilders smoothing (ewm alpha=1/window) rather than a plain rolling mean, matches how
    #charting platforms actually compute rsi so the numbers line up with what i see on tradingview
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = np.maximum(high_low, high_close)
    tr = np.maximum(tr, low_close)  #true range = max of the three candle ranges
    #same wilders smoothing as rsi, more responsive than an sma of true range
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def atr_percent(high, low, close, window=14):
    #atr as a fraction of price so it's comparable across a universe of different priced stocks
    return atr(high, low, close, window) / close

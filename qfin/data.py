import os
import pickle

import numpy as np
import pandas as pd
import yfinance as yf

#the old loader did close.ffill().bfill().fillna(0), which is two separate problems.
#bfill invents a flat price history for anything that listed mid sample, so a 2024 ipo
#looks like it traded quietly since 2022 and can pass a 200 day trend filter on data that
#never existed. fillna(0) then leaves genuine zero prices, and a zero denominator in
#pct_change is where the infinities came from. this module carries an explicit validity
#mask instead, so a name is simply untradeable outside its real listed window

def download(tickers, start, end, cache_path=None, refresh=False):
    #cache to disk because a thousand tickers is slow and the backtest gets rerun a lot
    if cache_path and os.path.exists(cache_path) and not refresh:
        with open(cache_path, 'rb') as fh:
            return pickle.load(fh)
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'wb') as fh:
            pickle.dump(raw, fh)
    return raw

def listed_window(close: pd.DataFrame):
    #true only between a name's first and last real print. cummax forwards finds the first,
    #cummax on the reversed frame finds the last, so a delisting closes the window too
    present = close.notna() & (close > 0)
    started = present.cummax()
    ending = present[::-1].cummax()[::-1]
    return started & ending

def clean_prices(raw, max_gap: int = 5):
    #forward fill only inside the listed window and only across short gaps, which covers
    #halts and missing prints without manufacturing history at either end
    close = raw['Close'].copy()
    valid = listed_window(close)
    close = close.where(close > 0)
    filled = close.ffill(limit=max_gap).where(valid)
    out = {'close': filled, 'valid': valid & filled.notna()}
    for field in ('High', 'Low', 'Volume'):
        if field in raw.columns.get_level_values(0):
            frame = raw[field].copy().where(valid)
            out[field.lower()] = frame.ffill(limit=max_gap) if field != 'Volume' else frame
    return out

def to_returns(close: pd.DataFrame, valid: pd.DataFrame):
    #returns only count on days where both today and yesterday are real prints, so the
    #first day of a listing and the day after a delisting don't book a phantom move
    prev = valid.shift(1).astype('boolean').fillna(False).astype(bool)
    tradeable = valid & prev
    r = close.pct_change(fill_method=None).where(tradeable)
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def dollar_volume(close: pd.DataFrame, volume: pd.DataFrame, window: int = 21):
    #average daily traded value, what the participation cap in costs.py is measured against
    return (close * volume).rolling(window, min_periods=5).mean()

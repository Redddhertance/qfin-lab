import os

import numpy as np
import pandas as pd

#data loading + cleaning helpers. the path logic kept getting re-derived in every script (and
#breaking when run from a different working directory), so it lives here once now.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_path(*parts):
    #absolute path into the data/ folder regardless of where the script was launched from
    return os.path.join(PROJECT_ROOT, 'data', *parts)


def load_tickers(filename, column='ticker', limit=None):
    tickers = pd.read_csv(data_path(filename))[column].tolist()
    return tickers[:limit] if limit else tickers


def clean_close(close):
    #forward then back fill gaps, then 0 the rest. the bfill matters for names that ipo'd partway
    #through the window, otherwise their early NaNs poison everything downstream.
    return close.ffill().bfill().fillna(0)


def to_returns(close):
    #daily simple returns. the inf scrub fixes a real yfinance glitch: an occasional 0 price makes
    #pct_change blow up to inf and that single value breaks the whole permutation equity curve.
    return close.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

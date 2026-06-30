import numpy as np
import pandas as pd

from .risk import annualised_volatility, sharpe, max_drawdown, TRADING_DAYS

#the core long-only book given target weights and asset returns. the lookahead-bias guard (the
#shift) lives in here rather than in the script so it can actually be tested, see tests/test_backtest.py.

BPS_PER_TURNOVER = 2.0  #2bp of cost per 100% of the book turned over


def lag_weights(weights):
    #weights decided using data up to and including day t can only be traded into at t+1. shifting
    #by one day is the single line that stops the whole backtest from cheating.
    return weights.shift(1).fillna(0.0)


def transaction_costs(effective_weights, bps_per_turnover=BPS_PER_TURNOVER):
    #turnover = how much of the book changed vs yesterday (direction doesn't matter, hence abs)
    daily_turnover = effective_weights.diff().abs().sum(axis=1).fillna(0.0)
    return daily_turnover * (bps_per_turnover / 1e4)


def run_backtest(weights, returns, bps_per_turnover=BPS_PER_TURNOVER):
    #returns the net pnl series, the equity curve and a small metrics dict. weights are the targets
    #for each day BEFORE lagging, this function applies the lag itself.
    effective = lag_weights(weights)
    gross = (effective * returns).sum(axis=1)
    costs = transaction_costs(effective, bps_per_turnover)
    pnl = gross - costs
    equity = (1.0 + pnl).cumprod()

    metrics = {
        'days': len(pnl),
        'annual_return': equity.iloc[-1] ** (TRADING_DAYS / len(pnl)) - 1.0 if len(pnl) else 0.0,
        'annual_volatility': annualised_volatility(pnl),
        'sharpe': sharpe(pnl),
        'max_drawdown': max_drawdown(equity),
        'avg_positions': (effective > 0).sum(axis=1).mean(),
        'pct_time_in_market': (effective.sum(axis=1) > 0).mean(),
    }
    return pnl, equity, metrics

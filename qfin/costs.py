import numpy as np
import pandas as pd

#a flat 2bp per unit of turnover is an institutional large cap number and it was carrying
#the whole backtest. real cost is a spread you always pay plus an impact term that grows
#with how much of the day's volume you're trying to take. both are modelled here

def linear_costs(weights: pd.DataFrame, bps_per_turnover: float = 3.0):
    #the old flat model, kept so the new one can be benchmarked against it
    tw = weights.diff().abs().sum(axis=1).fillna(0.0)
    return tw * (bps_per_turnover / 1e4)

def corwin_schultz_spread(high: pd.DataFrame, low: pd.DataFrame, smooth: int = 21):
    #NOT the default, see liquidity_spread. kept because it's the right tool for genuinely
    #illiquid names, but on liquid ones it is badly biased upward: the true spread sits so
    #close to zero that the estimator's noise straddles it, and clipping the negative half
    #to zero before averaging leaves pure bias. it prices apple at ~40bp against a real ~1bp.
    #treat its output as a loose upper bound on thin names, never as a large cap estimate.
    #estimates the bid-ask spread from daily high/low ranges (corwin & schultz 2012).
    #the insight is that a single day's range reflects both volatility and the spread, but
    #a two day range holds twice the volatility and the same spread, so the two separate.
    #free, and it uses bars we already download rather than tick data we don't have
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)
    high2 = pd.concat([high, high.shift(1)]).groupby(level=0).max()
    low2 = pd.concat([low, low.shift(1)]).groupby(level=0).min()
    gamma = np.log(high2 / low2) ** 2

    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    #the estimator is noisy day to day and a fair share of estimates come out negative,
    #which is meaningless. corwin & schultz set those to zero and average rather than
    #discarding them, because dropping only the negatives keeps the high tail and biases
    #the level badly upward. averaging zeros in is the published treatment
    spread = spread.where(np.isfinite(spread)).clip(lower=0.0)
    return spread.rolling(smooth, min_periods=3).mean()

def liquidity_spread(adv: pd.DataFrame, floor_bps: float = 1.0, cap_bps: float = 200.0):
    #spread as a power law in dollar volume, which is the shape the empirical microstructure
    #literature reports: roughly 1-2bp on a mega cap turning over a billion a day, ~5bp at
    #a hundred million, ~20bp at ten million, ~70bp at a million. calibrated to hit those
    #anchor points, so it degrades sensibly from an s&p name to a micro cap
    a, b = 151000.0, 0.556
    bps = a * np.power(adv.where(adv > 0.0), -b)
    return (bps.clip(lower=floor_bps, upper=cap_bps) / 1e4).fillna(cap_bps / 1e4)

def spread_costs(weights: pd.DataFrame, spread: pd.DataFrame, floor_bps: float = 5.0,
                 cap_bps: float = 300.0):
    #you cross half the spread on the way in and half on the way out, so per unit of
    #one way turnover the cost is half the spread. floored because the estimator
    #occasionally returns implausibly tight values, capped so one bad print can't dominate
    traded = weights.diff().abs().fillna(0.0)
    half = (spread.reindex_like(traded) / 2.0).clip(lower=floor_bps / 1e4, upper=cap_bps / 1e4)
    return (traded * half.fillna(floor_bps / 1e4)).sum(axis=1)

def impact_costs(weights: pd.DataFrame, adv: pd.DataFrame, volatility: pd.DataFrame,
                 capital: float = 1e7, coefficient: float = 0.6):
    #square root market impact, the standard practitioner form: moving a position that is
    #a fraction p of average daily volume costs roughly coeff * daily vol * sqrt(p).
    #impact scales with how big you are, so it needs an assumed capital base
    traded_value = weights.diff().abs().fillna(0.0) * capital
    participation = (traded_value / adv.reindex_like(traded_value)).replace([np.inf, -np.inf], np.nan)
    participation = participation.clip(upper=1.0).fillna(0.0)
    impact = coefficient * volatility.reindex_like(traded_value).fillna(0.0) * np.sqrt(participation)
    return (weights.diff().abs().fillna(0.0) * impact).sum(axis=1)

def participation_cap(weights: pd.DataFrame, adv: pd.DataFrame, capital: float = 1e7,
                      max_participation: float = 0.10):
    #refuses to hold more than you could actually trade. a position is capped so that
    #building it takes no more than max_participation of one day's dollar volume, which is
    #what stops a backtest quietly allocating into names it could never fill
    ceiling = (adv.reindex_like(weights) * max_participation / capital)
    ceiling = ceiling.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    capped = weights.clip(upper=ceiling)
    #put the capital freed by the cap back across the names that still have room, once,
    #rather than iterating to convergence
    shortfall = (weights.sum(axis=1) - capped.sum(axis=1)).clip(lower=0.0)
    room = (ceiling - capped).clip(lower=0.0)
    room_total = room.sum(axis=1).replace(0.0, np.nan)
    redistributed = capped + room.mul(shortfall / room_total, axis=0).fillna(0.0)
    return redistributed.clip(upper=ceiling).fillna(0.0)

def total_costs(weights: pd.DataFrame, spread: pd.DataFrame, adv: pd.DataFrame,
                volatility: pd.DataFrame, capital: float = 1e7, coefficient: float = 0.6,
                floor_bps: float = 5.0):
    #what the backtest charges itself: spread on every trade plus size dependent impact
    s = spread_costs(weights, spread, floor_bps=floor_bps)
    i = impact_costs(weights, adv, volatility, capital=capital, coefficient=coefficient)
    return s + i, s, i

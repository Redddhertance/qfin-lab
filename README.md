# qfin-lab
A quantitative equity research framework built for GAKA Labs, combining systematic technical screening with risk analytics and backtesting. Designed to support fundamental analysts by filtering a large stock universe down to actionable candidates.

Overview

The framework has two main components: a scanner that screens stocks on fundamental and technical criteria, and a backtesting engine that tests momentum-based strategies against historical data with rigorous validation.

Components

Scanner (tests/scanner/scanner.py)
Screens a CSV-defined ticker universe through a two-stage pipeline:
Stage 1 — Bulk Filter: Downloads 1-year price data and removes stocks trading below their 200-day SMA, reducing the universe before detailed analysis.
Stage 2 — Screener: For surviving stocks, evaluates fundamental and technical criteria and generates buy/watch signals. Buy signal requires ROE > 15%, debt-to-equity < 100, P/E < 35, and bullish trend confirmation.
Outputs risk metrics including volatility, Sharpe ratio, max drawdown, VaR (95%), CVaR (95%), sector beta, and stress test scenarios to CSV.

Backtesting Engine (tests/gaka_backtest_technicals.py)
Tests a momentum-based long strategy across a configurable ticker universe and date range.
Signal logic: Stocks must be above 200-day SMA, RSI > 55, and ATR% < 5% (volatility filter). A market regime filter using SPY's 200-day SMA prevents any positions during broad market downtrends.
Portfolio construction: Monthly rebalancing with equal weighting across all eligible stocks. Weights are lagged by one day to prevent lookahead bias.
Transaction costs: 2 basis points per unit of turnover modelled explicitly.

Performance metrics reported:

Annualised return and volatility
Sharpe ratio
Maximum drawdown
Outperformance vs SPY benchmark

Validation — Permutation Test: Runs 200 trials shuffling strategy weights against actual returns to test whether performance is distinguishable from random timing. Reports p-value with significance thresholds at 0.05 and 0.10. Results plotted on log-scale equity curve chart.

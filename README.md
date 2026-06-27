qfin-lab
A quantitative equity research framework built for GAKA Labs, combining systematic technical screening with risk analytics and backtesting. Designed to support fundamental analysts by filtering a large stock universe down to actionable candidates.

Overview
The framework has two main components: a scanner that screens stocks on fundamental and technical criteria, and a backtesting engine that tests momentum-based strategies against historical data with rigorous validation.

Feature Summary:

Algorithmic Screener (scanner.py): Filters large equities universes (e.g., S&P 500, IWM 1000) using a combination of technical indicators (SMA) and dynamically retrieved market data.

Technical Backtesting Engine (gaka_backtest_technicals.py): A vector-optimized backtester evaluating custom technical strategies (RSI, ATR, SMA) and broader market regime filters. Rebalances monthly, enforces concentration limits, and calculates explicit transaction costs (slippage/turnover).

Risk Metrics Suite: Computes detailed portfolio risk profiles including annualized volatility, Sharpe Ratio, Maximum Drawdown, Value at Risk (VaR 95), Conditional VaR (CVaR 95), and Sector Beta exposure.

High-Performance Statistical Edge Validation: Utilizes Monte Carlo-style permutation testing (Random Asset Selection / Monkey Dartboard) to decouple strategy signals from underlying asset returns. Driven by a custom C++ backend to scale up to 500,000 trials, definitively proving whether outperformance is statistically significant (p-value) or just luck.

Components
Scanner (tests/scanner/scanner.py)
Screens a CSV-defined ticker universe through a two-stage pipeline:

Stage 1 — Bulk Filter: Downloads 1-year price data and removes stocks trading below their 200-day SMA, reducing the universe before detailed analysis.

Stage 2 — Screener: For surviving stocks, evaluates fundamental and technical criteria and generates buy/watch signals. Buy signal requires ROE > 15%, debt-to-equity < 100, P/E < 35, and bullish trend confirmation.

Outputs risk metrics including volatility, Sharpe ratio, max drawdown, VaR (95%), CVaR (95%), sector beta, and stress test scenarios to CSV.

Backtesting Engine (tests/gaka_backtest_technicals.py)
Tests a momentum-based long strategy across a configurable ticker universe and date range.

Signal logic: Stocks must be above 200-day SMA, RSI > 55, and ATR% < 5% (volatility filter). A market regime filter using SPY's 200-day SMA prevents any positions during broad market downtrends.

Portfolio construction & Risk Management: Monthly rebalancing with equal weighting across all eligible stocks. Crucially, exposure is capped at a 5% maximum allocation per asset. This prevents catastrophic drawdowns from micro-cap delistings/bankruptcies and dynamically builds a cash buffer when the regime filter restricts eligible setups. Weights are lagged by one day to prevent lookahead bias.

Transaction costs: 2 basis points per unit of turnover modelled explicitly.

Data Sanitization: Explicit handling and scrubbing of np.inf values caused by zero-price data glitches in raw low-cap data feeds.

Performance metrics reported:

Annualised return and volatility

Sharpe ratio

Maximum drawdown

Performance vs SPY benchmark

Validation — C++ Permutation Test: Shuffles strategy weights against actual returns to test whether the timing of signals has predictive power. To bypass Python's GIL and scaling limitations, the permutation engine was rewritten in C++ (gaka_core.cpp) via pybind11. It runs 500,000 permutations and reports the p-value with significance thresholds at 0.05 and 0.10. Results are plotted on a log-scale equity curve chart.

Recent architecture updates:
C++ Backend Integration (pybind11): Ported the permutation test's heavy mathematical loops to C++ to strip out latency.

Cross-Matrix Optimization: Replaced the O(N³) inner asset loop with pre-computed NumPy cross-PnL matrices before passing memory pointers to C++. This reduced the permutation workload by 99.9% (from 500 billion operations to 500 million), allowing 500,000 trials to execute in under a second.

Eliminated Lookahead Bias: Deprecated fundamental data backtesting in favor of pure technical OHLC data to ensure historical accuracy.

Dynamic I/O Pathing: Built path-agnostic data loading (os.path.abspath), ensuring scripts run reliably regardless of the terminal's working directory.

Pandas 2.0 Compliance: Migrated to direct ffill()/bfill() chaining and updated timeseries frequencies for forward-compatibility.

Strict Type Safety: Resolved complex matrix shaping and Pylance linting warnings for seamless IDE integration.
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qfin import metrics as M
from qfin.tearsheet import build_tearsheet

#everything here runs offline, no yfinance calls, so it's safe to run in ci

@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2022-01-03', periods=600)
    assets = [f'A{i}' for i in range(10)]
    weights = pd.DataFrame(0.0, index=idx, columns=assets)
    weights.iloc[:, :5] = 0.05
    weights.iloc[200:260, :] = 0.0 #cash period, like the regime filter produces
    rets = pd.DataFrame(rng.normal(0.0005, 0.015, (len(idx), len(assets))), index=idx, columns=assets)
    costs = weights.diff().abs().sum(axis=1).fillna(0.0) * 2e-4
    pnl = (weights * rets).sum(axis=1) - costs
    bench = pd.Series(rng.normal(0.0004, 0.011, len(idx)), index=idx)
    return pnl, weights, bench, costs

def test_equity_curve_compounds():
    pnl = pd.Series([0.1, 0.1, -0.1])
    assert np.allclose(M.equity_curve(pnl).to_numpy(), [1.1, 1.21, 1.089])

def test_cagr_recovers_a_known_annual_rate():
    #exactly 252 days of a constant rate has to annualise back to that rate
    daily = 10 ** (1 / 252) - 1
    assert M.cagr(pd.Series([daily] * 252)) == pytest.approx(9.0, rel=1e-6)

def test_sharpe_subtracts_the_risk_free_rate():
    rng = np.random.default_rng(1)
    pnl = pd.Series(rng.normal(0.0006, 0.01, 2000))
    gross = M.sharpe(pnl, risk_free=0.0)
    net = M.sharpe(pnl, risk_free=0.04)
    assert net < gross
    assert gross - net == pytest.approx(0.04 / M.annual_volatility(pnl), rel=1e-9)

def test_max_drawdown_matches_a_hand_worked_path():
    #1.0 -> 1.5 -> 0.75, worst peak to trough fall is 50%
    assert M.max_drawdown(pd.Series([0.5, -0.5, 0.2])) == pytest.approx(-0.5)

def test_drawdown_detail_finds_peak_trough_and_recovery():
    idx = pd.bdate_range('2023-01-02', periods=5)
    detail = M.drawdown_detail(pd.Series([0.0, -0.2, -0.1, 0.5, 0.0], index=idx))
    assert detail['peak'] == idx[0]
    assert detail['trough'] == idx[2]
    assert detail['recovered'] == idx[3]

def test_var_and_cvar_order_correctly():
    rng = np.random.default_rng(2)
    pnl = pd.Series(rng.normal(0.0, 0.01, 5000))
    assert M.conditional_var(pnl) <= M.value_at_risk(pnl) < 0

def test_beta_of_a_scaled_benchmark_is_the_scale():
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0.0003, 0.012, 1500))
    beta, alpha = M.beta_alpha(bench * 1.5, bench)
    assert beta == pytest.approx(1.5, rel=1e-9)
    assert alpha == pytest.approx(0.0, abs=1e-9) #a pure multiple leaves no residual

def test_capture_ratios_are_one_against_itself():
    rng = np.random.default_rng(4)
    bench = pd.Series(rng.normal(0.0003, 0.012, 800))
    up, down = M.capture_ratios(bench, bench)
    assert up == pytest.approx(1.0)
    assert down == pytest.approx(1.0)

def test_capture_ratios_are_not_distorted_by_daily_compounding():
    #half the benchmark should read ~50% both ways. compounding daily up-days over
    #four years instead collapses it towards zero, which was the original bug
    rng = np.random.default_rng(11)
    idx = pd.bdate_range('2022-01-03', periods=1000)
    bench = pd.Series(rng.normal(0.0005, 0.011, len(idx)), index=idx)
    up, down = M.capture_ratios(bench * 0.5, bench)
    assert up == pytest.approx(0.5, rel=0.05)
    assert down == pytest.approx(0.5, rel=0.05)

def test_monthly_returns_grid_compounds_within_each_month():
    idx = pd.bdate_range('2023-01-02', '2023-02-28')
    grid = M.monthly_returns(pd.Series(0.001, index=idx))
    jan = (idx.month == 1).sum()
    assert grid.loc[2023, 1] == pytest.approx(1.001 ** jan - 1)

def test_metrics_survive_an_all_cash_run():
    s = M.summarise(pd.Series(0.0, index=pd.bdate_range('2023-01-02', periods=100)))
    assert s['sharpe'] == 0.0 and s['max_dd'] == 0.0 and s['annual_vol'] == 0.0

def test_metrics_survive_an_empty_series():
    s = M.summarise(pd.Series(dtype=float))
    assert s['n_days'] == 0 and s['annual_return'] == 0.0

def test_infinities_are_scrubbed():
    #zero-price glitches in the lowcap feed produce inf returns
    assert np.isfinite(M.equity_curve(pd.Series([0.01, np.inf, -0.01, np.nan]))).all()

def test_annualised_turnover_of_a_daily_full_rotation():
    idx = pd.bdate_range('2023-01-02', periods=253)
    w = pd.DataFrame({'a': [1.0, 0.0] * 126 + [1.0], 'b': 0.0}, index=idx)
    #one asset flipping between 1.0 and 0.0 is 1.0 of one way turnover per day,
    #over 252 diffs across a 253 row frame
    assert M.annualised_turnover(w) == pytest.approx(252 * (252 / 253), rel=1e-6)

def test_build_tearsheet_writes_a_self_contained_file(tmp_path, sample):
    pnl, weights, bench, costs = sample
    out = build_tearsheet(pnl, weights=weights, benchmark=bench, costs=costs,
                          title='Test Run', bench_label='SPY', risk_free=0.04,
                          out_path=str(tmp_path / 't.html'))
    html = open(out, encoding='utf-8').read()
    assert html.startswith('<!doctype html>')
    assert '<title>Test Run</title>' in html
    #every image has to be inlined or the report stops being one portable file
    assert 'data:image/png;base64,' in html
    assert 'src="http' not in html and 'src="./' not in html

def test_tearsheet_includes_the_permutation_verdict(tmp_path, sample):
    pnl, weights, bench, costs = sample
    rng = np.random.default_rng(5)
    out = build_tearsheet(pnl, weights=weights, benchmark=bench, costs=costs,
                          permutation={'final_values': rng.lognormal(0.2, 0.4, 500),
                                       'real_final': 2.5, 'p_value': 0.012},
                          out_path=str(tmp_path / 'p.html'))
    html = open(out, encoding='utf-8').read()
    assert 'p-value is 0.0120' in html
    assert 'significant at the 5% level' in html

def test_tearsheet_renders_without_a_benchmark_or_weights(tmp_path, sample):
    pnl, _, _, _ = sample
    html = open(build_tearsheet(pnl, out_path=str(tmp_path / 'bare.html')), encoding='utf-8').read()
    assert 'Versus' not in html
    assert len(html) > 5000

def test_tearsheet_creates_missing_directories(tmp_path, sample):
    pnl, _, _, _ = sample
    assert os.path.exists(build_tearsheet(pnl, out_path=str(tmp_path / 'a' / 'b' / 't.html')))

def test_importing_the_tearsheet_does_not_hijack_the_matplotlib_backend():
    #if this module ever calls matplotlib.use() it kills plt.show() in the backtest scripts
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ('import sys; sys.path.insert(0, %r); import qfin.tearsheet; '
            "print('matplotlib.pyplot' in sys.modules)" % root)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert out.stdout.strip() == 'False', out.stderr

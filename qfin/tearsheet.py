import base64
import datetime
import io
import os

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import metrics as M

#one palette across every figure so the report doesn't look like six separate charts
INK = '#12161c'
MUTED = '#6b7482'
GRID = '#dfe3e8'
ACCENT = '#c0392b' #strategy
BENCH = '#4a6fa5' #benchmark
POSITIVE = '#2e7d5b'
NEGATIVE = '#b03a2e'
FIG_BG = '#ffffff'

def new_fig(width: float = 10.0, height: float = 3.4):
    #Figure() rather than plt.subplots() on purpose. pyplot needs a backend picked at
    #import time and calling matplotlib.use() in here would silently kill plt.show()
    #for the backtest scripts that import this module
    fig = Figure(figsize=(width, height), dpi=140)
    ax = fig.subplots()
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    return fig, ax

def date_axis(ax, max_ticks: int = 8):
    #thins the date ticks, otherwise the labels collide on the narrow half width charts
    locator = mdates.AutoDateLocator(minticks=2, maxticks=max_ticks)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

def encode(fig):
    #base64 the png straight into the html so the report is one portable file
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

def chart_equity(pnl, benchmark, bench_label):
    eq = M.equity_curve(pnl)
    fig, ax = new_fig(height=3.8)
    ax.plot(eq.index, eq.to_numpy(), color=ACCENT, linewidth=1.9, label='Strategy')
    if benchmark is not None:
        beq = M.equity_curve(benchmark).reindex(eq.index).ffill()
        ax.plot(beq.index, beq.to_numpy(), color=BENCH, linewidth=1.5, label=bench_label, alpha=0.9)
    ax.set_yscale('log') #log scale so a 10% move reads the same early and late in the sample
    ax.set_ylabel('Growth of 1.00 (log)', color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    date_axis(ax)
    return encode(fig)

def chart_drawdown(pnl):
    dd = M.drawdown_series(pnl)
    fig, ax = new_fig(height=2.4)
    ax.fill_between(dd.index, dd.to_numpy() * 100.0, 0.0, color=NEGATIVE, alpha=0.28, linewidth=0)
    ax.plot(dd.index, dd.to_numpy() * 100.0, color=NEGATIVE, linewidth=1.2)
    ax.set_ylabel('Drawdown (%)', color=MUTED, fontsize=9)
    date_axis(ax)
    return encode(fig)

def chart_rolling_sharpe(pnl, risk_free, window):
    rs = M.rolling_sharpe(pnl, window=window, risk_free=risk_free).dropna()
    if rs.empty:
        return '' #not enough history to fill the window yet
    fig, ax = new_fig(height=2.4)
    ax.axhline(0.0, color=MUTED, linewidth=0.9)
    ax.plot(rs.index, rs.to_numpy(), color=INK, linewidth=1.3)
    ax.fill_between(rs.index, rs.to_numpy(), 0.0, where=rs.to_numpy() >= 0, color=POSITIVE, alpha=0.18, linewidth=0)
    ax.fill_between(rs.index, rs.to_numpy(), 0.0, where=rs.to_numpy() < 0, color=NEGATIVE, alpha=0.18, linewidth=0)
    ax.set_ylabel(f'Rolling {window}d Sharpe', color=MUTED, fontsize=9)
    date_axis(ax)
    return encode(fig)

def chart_monthly_heatmap(pnl):
    grid = M.monthly_returns(pnl)
    if grid.empty:
        return ''
    values = grid.to_numpy(dtype=float) * 100.0
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    limit = max(limit, 0.5) #symmetric colour scale so green and red are comparable

    height = max(1.5, 0.42 * len(grid) + 1.1) #grows with the number of years
    fig = Figure(figsize=(10.0, height), dpi=140)
    ax = fig.subplots()
    fig.patch.set_facecolor(FIG_BG)
    im = ax.imshow(values, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')

    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([labels[int(c) - 1] for c in grid.columns], fontsize=8.5) #int() as the column labels come back typed as Hashable
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([str(y) for y in grid.index], fontsize=8.5)
    ax.tick_params(colors=MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                continue #month outside the sample
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7.5,
                    color=INK if abs(v) < limit * 0.6 else '#ffffff')
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.015).ax.tick_params(labelsize=8, colors=MUTED, length=0)
    return encode(fig)

def chart_distribution(pnl):
    r = M.to_series(pnl).replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r != 0.0] #drop the cash days, they'd pile up a fake spike at zero
    if len(r) < 10:
        return ''
    var = M.value_at_risk(r)
    cvar = M.conditional_var(r)
    fig, ax = new_fig(width=5.0, height=2.9)
    ax.hist(np.asarray(r, dtype=float) * 100.0, bins=60, color=BENCH, alpha=0.75, linewidth=0)
    ax.axvline(var * 100.0, color=ACCENT, linewidth=1.4, linestyle='--', label=f'VaR 95: {var:.2%}')
    ax.axvline(cvar * 100.0, color=NEGATIVE, linewidth=1.4, linestyle=':', label=f'CVaR 95: {cvar:.2%}')
    ax.set_xlabel('Daily return (%)', color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return encode(fig)

def chart_exposure(weights):
    if weights is None or len(weights) == 0:
        return ''
    positions = (weights != 0.0).sum(axis=1)
    gross = weights.abs().sum(axis=1) * 100.0
    fig, ax = new_fig(width=5.0, height=2.9)
    ax.fill_between(gross.index, gross.to_numpy(), 0.0, color=BENCH, alpha=0.25, linewidth=0)
    ax.plot(gross.index, gross.to_numpy(), color=BENCH, linewidth=1.2, label='Gross exposure (%)')
    ax.set_ylabel('Gross exposure (%)', color=MUTED, fontsize=9)
    twin = ax.twinx() #position count is a different unit, needs its own axis
    twin.plot(positions.index, positions.to_numpy(), color=INK, linewidth=1.1, alpha=0.8, label='Positions')
    twin.set_ylabel('Positions held', color=MUTED, fontsize=9)
    twin.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ('top', 'right', 'left'):
        twin.spines[side].set_visible(False)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [str(l.get_label()) for l in lines], frameon=False, fontsize=8, loc='upper left') #get_label is typed as object
    date_axis(ax, max_ticks=4)
    return encode(fig)

def chart_permutation(final_values, real_final, p_value):
    vals = np.asarray(final_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return ''
    fig, ax = new_fig(width=10.0, height=3.0)
    ax.hist(vals, bins=120, color=MUTED, alpha=0.55, linewidth=0, label=f'Permuted trials (n={len(vals):,})')
    ax.axvline(real_final, color=ACCENT, linewidth=2.0, label=f'Real strategy: {real_final:.2f}  (p={p_value:.4f})')
    ax.set_xlabel('Final equity multiple', color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    return encode(fig)

def pct(x):
    return 'n/a' if x is None else f'{x:.2%}'

def num(x, dp: int = 2):
    return 'n/a' if x is None else f'{x:.{dp}f}'

def tone(x, good_when_high: bool = True):
    #green or red class for the metric cards
    if x is None or x == 0:
        return ''
    positive = x > 0 if good_when_high else x < 0
    return 'pos' if positive else 'neg'

def card(label, value, klass='', note=''):
    note_html = f'<div class="note">{note}</div>' if note else ''
    return f'<div class="card"><div class="label">{label}</div><div class="value {klass}">{value}</div>{note_html}</div>'

def rows(pairs):
    return ''.join(f'<tr><th>{k}</th><td class="{t}">{v}</td></tr>' for k, v, t in pairs)

def section(title, body):
    return f'<section><h2>{title}</h2>{body}</section>' if body else ''

def img(src, caption=''):
    if not src:
        return ''
    cap = f'<figcaption>{caption}</figcaption>' if caption else ''
    return f'<figure><img src="{src}" alt="{caption}">{cap}</figure>'

CSS = """
:root{--ink:#12161c;--muted:#6b7482;--rule:#e3e7ec;--bg:#f6f7f9;--card:#fff;
--pos:#2e7d5b;--neg:#b03a2e;--accent:#c0392b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px 72px}
header{background:var(--ink);color:#fff;padding:30px 0 26px;margin-bottom:30px}
header .wrap{padding-bottom:0}
.brand{font-size:11px;letter-spacing:.18em;text-transform:uppercase;
color:#9aa4b1;margin-bottom:9px}
header h1{margin:0 0 8px;font-size:25px;font-weight:600;letter-spacing:-.01em}
.meta{color:#9aa4b1;font-size:12.5px}
.meta span+span:before{content:"\\00b7";margin:0 9px;color:#5b6472}
section{margin:34px 0}
h2{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);
margin:0 0 14px;padding-bottom:9px;border-bottom:1px solid var(--rule);font-weight:600}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:7px;padding:13px 15px}
.card .label{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.card .value{font-size:23px;font-weight:600;margin-top:5px;
font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.card .note{font-size:11px;color:var(--muted);margin-top:3px}
.pos{color:var(--pos)}.neg{color:var(--neg)}
figure{margin:0 0 18px;background:var(--card);border:1px solid var(--rule);
border-radius:7px;padding:15px;overflow-x:auto}
figure img{display:block;width:100%;height:auto;min-width:520px}
figcaption{font-size:11.5px;color:var(--muted);margin-top:9px}
.split{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.split figure{margin:0}
table{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--rule);border-radius:7px;overflow:hidden}
th,td{padding:9px 15px;text-align:left;border-bottom:1px solid var(--rule);font-size:13px}
tr:last-child th,tr:last-child td{border-bottom:0}
th{font-weight:500;color:var(--muted);width:56%}
td{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.callout{background:var(--card);border:1px solid var(--rule);border-left:3px solid var(--accent);
border-radius:7px;padding:14px 17px;font-size:13px;color:#3c4250}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--rule);
font-size:11.5px;color:var(--muted)}
@media(max-width:760px){.split{grid-template-columns:1fr}.wrap{padding:0 16px 48px}}
@media print{body{background:#fff}header{background:#fff;color:var(--ink);
border-bottom:2px solid var(--ink)}.brand,.meta{color:var(--muted)}
figure,.card,table{break-inside:avoid}}
"""

def build_tearsheet(pnl, weights=None, benchmark=None, costs=None, title='Strategy Tearsheet',
                    subtitle='', bench_label='Benchmark', risk_free: float = 0.0,
                    freq: int = M.trading_days, rolling_window: int = M.trading_days,
                    permutation=None, notes='', out_path='reports/tearsheet.html'):
    #permutation takes the output of the c++ validation run:
    #{'final_values': one final equity multiple per trial, 'real_final': float, 'p_value': float}
    s = M.summarise(pnl, weights=weights, benchmark=benchmark, costs=costs,
                    risk_free=risk_free, freq=freq)

    period = 'n/a'
    if s['start'] is not None and s['end'] is not None:
        period = f"{s['start']:%d %b %Y} to {s['end']:%d %b %Y}"

    headline = ''.join([
        card('Annual return', pct(s['annual_return']), tone(s['annual_return'])),
        card('Annual volatility', pct(s['annual_vol'])),
        card('Sharpe', num(s['sharpe']), tone(s['sharpe']), f'excess of {risk_free:.1%} rf'),
        card('Sortino', num(s['sortino']), tone(s['sortino'])),
        card('Max drawdown', pct(s['max_dd']), 'neg' if s['max_dd'] < 0 else '',
             f"{s['max_dd_days']} trading days"),
        card('Calmar', num(s['calmar']), tone(s['calmar']))
    ])

    risk_rows = rows([
        ('Total return', pct(s['total_return']), tone(s['total_return'])),
        ('Annualised return', pct(s['annual_return']), tone(s['annual_return'])),
        ('Annualised volatility', pct(s['annual_vol']), ''),
        ('Daily VaR (95%)', pct(s['var_95']), 'neg'),
        ('Daily CVaR (95%)', pct(s['cvar_95']), 'neg'),
        ('Best day', pct(s['best_day']), 'pos'),
        ('Worst day', pct(s['worst_day']), 'neg'),
        ('Hit rate', pct(s['hit_rate']), ''),
        ('Skew', num(s['skew']), ''),
        ('Excess kurtosis', num(s['excess_kurtosis']), ''),
        ('Trading days', f"{s['n_days']:,}", '')
    ])

    impl_rows = rows([
        ('Annualised turnover', f"{s['annual_turnover']:.2f}x", ''),
        ('Cost drag (annual)', pct(-abs(s['cost_drag'])) if s['cost_drag'] else 'n/a',
         'neg' if s['cost_drag'] else ''),
        ('Average positions', num(s['avg_positions'], 1), ''),
        ('Days invested', pct(s['pct_time_invested']), '')
    ])

    bench_block = ''
    if benchmark is not None:
        bench_cards = ''.join([
            card('Excess return', pct(s['excess_return']), tone(s['excess_return']), f'vs {bench_label}'),
            card('Alpha (annual)', pct(s['alpha']), tone(s['alpha'])),
            card('Beta', num(s['beta'])),
            card('Information ratio', num(s['information_ratio']), tone(s['information_ratio'])),
            card('Up capture', pct(s['up_capture'])),
            card('Down capture', pct(s['down_capture']), 'pos' if 0 < s['down_capture'] < 1 else '')
        ])
        bench_table = rows([
            (f'{bench_label} annual return', pct(s['bench_annual_return']), ''),
            (f'{bench_label} annual volatility', pct(s['bench_annual_vol']), ''),
            (f'{bench_label} Sharpe', num(s['bench_sharpe']), ''),
            (f'{bench_label} max drawdown', pct(s['bench_max_dd']), 'neg'),
            ('Tracking error', pct(s['tracking_error']), '')
        ])
        bench_block = section(f'Versus {bench_label}',
                              f'<div class="cards">{bench_cards}</div>'
                              f'<div style="margin-top:16px"><table>{bench_table}</table></div>')

    perm_block = ''
    if permutation:
        perm_img = chart_permutation(permutation.get('final_values', []),
                                     float(permutation.get('real_final', 0.0)),
                                     float(permutation.get('p_value', 1.0)))
        p = float(permutation.get('p_value', 1.0))
        verdict = ('significant at the 5% level' if p < 0.05
                   else 'significant at the 10% level' if p < 0.10
                   else 'not statistically significant')
        perm_block = section('Signal validation',
                             img(perm_img, 'Final equity multiple across permuted trials. Weight rows are '
                                           'shuffled against real return days, so the null tests whether '
                                           'signal timing carries information.')
                             + f'<div class="callout">The permutation p-value is {p:.4f}, which is {verdict}. '
                               f'It tests signal timing only and does not correct for parameters searched '
                               f'during development.</div>')

    notes_block = section('Notes', f'<div class="callout">{notes}</div>') if notes else ''
    generated = datetime.datetime.now().strftime('%d %b %Y, %H:%M')
    subtitle_html = f'<div class="meta">{subtitle}</div>' if subtitle else ''

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}</style></head><body>
<header><div class="wrap">
  <div class="brand">GAKA Labs &middot; Research</div>
  <h1>{title}</h1>
  {subtitle_html}
  <div class="meta"><span>{period}</span><span>{s['n_days']:,} trading days</span>
  <span>Generated {generated}</span></div>
</div></header>
<div class="wrap">
  {section('Headline', f'<div class="cards">{headline}</div>')}
  {section('Performance', img(chart_equity(pnl, benchmark, bench_label))
           + img(chart_drawdown(pnl))
           + img(chart_rolling_sharpe(pnl, risk_free, rolling_window)))}
  {section('Monthly returns (%)', img(chart_monthly_heatmap(pnl)))}
  {section('Risk and implementation',
           '<div class="split">' + img(chart_distribution(pnl)) + img(chart_exposure(weights)) + '</div>'
           + f'<div class="split"><table>{risk_rows}</table><table>{impl_rows}</table></div>')}
  {bench_block}
  {perm_block}
  {notes_block}
</div></body></html>"""

    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    return out_path

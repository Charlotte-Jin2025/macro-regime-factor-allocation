"""Build figures from this study's computed results, without refitting models."""
from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'utsg7394_final_mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'utsg7394_final_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

DESIGN = 'Release_corrected_current_vintage'
PRIMARY = 'LS5_Variance36_Cap40_Shrink50'
FACTORS = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
STRATEGIES = [
    ('Historical_Mean', 'Historical mean', '#252525', '-'),
    ('Macro', 'Macro forecast average', '#14648A', '-'),
    ('Macro_M1_YoY', 'Fixed state', '#A66531', '--'),
    ('Macro_M1_ExpandingYoY', 'Expanding state', '#7C6594', ':'),
]
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 11,
         'axes.spines.top': False, 'axes.spines.right': False,
         'figure.dpi': 180, 'savefig.dpi': 180}


def time_axis(ax, dates, every=3):
    """Show the actual first and last month as well as interior year ticks."""
    lo, hi = pd.Timestamp(min(dates)), pd.Timestamp(max(dates))
    middle = [pd.Timestamp(f'{year}-01-01') for year in range(lo.year + every, hi.year, every)]
    ticks = [lo] + middle + [hi]
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_xticklabels([lo.strftime('%b %Y')] + [str(t.year) for t in middle] + [hi.strftime('%b %Y')])
    ax.grid(axis='y', alpha=.18)


def save(fig, out, number, title):
    path = out / f'figure{number}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return {'number': number, 'file': path.name, 'title': title}


def us_figures(base, out):
    r = pd.read_csv(base / 'results/us/returns.csv', parse_dates=['Return_Date'])
    r = r[(r.Design == DESIGN) & (r.Cohort == 'Extended_175') &
          (r.Allocation == PRIMARY) & (r.Cost_bps == 10)]
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    series = {}
    for key, label, color, line in STRATEGIES:
        z = r[r.Strategy == key].sort_values('Return_Date').set_index('Return_Date')
        if len(z) != 175 or z.index.duplicated().any():
            raise ValueError(f'Expected 175 unique return months for {key}.')
        series[key] = z.Net_Return_pct
        axes[0].plot(z.index, (1 + z.Net_Return_pct / 100).cumprod(),
                     label=label, color=color, ls=line, lw=1.8)
    axes[0].axhline(1, color='#888888', lw=.6)
    axes[0].set_ylabel('Compounded index\nInitial value = 1')
    axes[0].set_title('Five factor allocations and the incremental macro return', loc='left', weight='bold')
    axes[0].legend(ncol=2, loc='upper left', frameon=False, fontsize=10)
    delta = (series['Macro'] - series['Historical_Mean']).cumsum()
    axes[1].plot(delta.index, delta, color='#14648A', lw=1.8)
    axes[1].axhline(0, color='#555555', lw=.7)
    axes[1].set_ylabel('Cumulative arithmetic\nMacro minus mean (pp)')
    for ax in axes:
        ax.axvline(pd.Timestamp('2019-04-01'), color='#777777', ls='--', lw=.8)
        time_axis(ax, delta.index)
    axes[1].set_xlabel('Return month')
    fig.tight_layout(h_pad=1.4)
    records = [save(fig, out, 1, 'Allocation paths and cumulative macro return increment')]

    w = pd.read_csv(base / 'results/us/weights.csv', parse_dates=['Return_Date'])
    w = w[(w.Design == DESIGN) & (w.Allocation == PRIMARY) & (w.Strategy == 'Macro')].sort_values('Return_Date')
    f = ['SMB', 'HML', 'RMW', 'CMA', 'MOM']
    if len(w) != 175 or not np.allclose(w[f].sum(axis=1), 1) or not np.allclose(w.MKT_RF, 0):
        raise ValueError('The primary allocation must contain 175 fully invested five-factor weights.')
    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    ax.stackplot(w.Return_Date, w[f].to_numpy().T, labels=f,
                 colors=['#688EA8', '#AECAD2', '#748B70', '#D2B16B', '#AC8594'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Factor weight')
    ax.set_title('Macro target allocation across five long-short factors', loc='left', weight='bold')
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(.5, -.16), frameon=False)
    time_axis(ax, w.Return_Date)
    fig.tight_layout()
    records.append(save(fig, out, 2, 'Five factor target weights'))

    c = pd.read_csv(base / 'results/states/model_calendar.csv', parse_dates=['Signal_Date'])
    c = c[c.Design == DESIGN].sort_values('Signal_Date')
    names = ['Expansion', 'Overheating', 'Slowdown', 'Stagflation']
    colors = ['#79A68B', '#D2A069', '#7196B2', '#A488A1', '#EEEEEE']
    matrix = np.asarray([[names.index(v) if v in names else 4 for v in c[col]]
                         for col in ['Regime_YoY', 'Regime_ExpandingYoY']])
    x = mdates.date2num(c.Signal_Date)
    edges = np.r_[x[0] - 15, (x[:-1] + x[1:]) / 2, x[-1] + 15]
    fig, ax = plt.subplots(figsize=(10.8, 4.3))
    ax.pcolormesh(edges, np.arange(3), matrix, cmap=ListedColormap(colors),
                  vmin=-.5, vmax=4.5, shading='flat')
    ax.set_yticks([.5, 1.5], ['Fixed median', 'Expanding median'])
    ax.set_ylim(2, 0)
    for boundary in ['2011-12-01', '2019-03-01']:
        ax.axvline(pd.Timestamp(boundary), color='#333333', lw=1)
    ax.xaxis_date()
    time_axis(ax, c.Signal_Date, every=6)
    ax.set_title('Growth and inflation states at forecast origins', loc='left', weight='bold')
    ax.set_xlabel('Signal month')
    ax.legend(handles=[Patch(facecolor=color, label=name) for color, name in zip(colors, names + ['Unavailable'])],
              ncol=3, loc='upper center', bbox_to_anchor=(.5, -.35), frameon=False, fontsize=10)
    fig.tight_layout()
    records.append(save(fig, out, 3, 'Fixed and expanding growth inflation states'))
    return records


def diagnostic_figures(base, out):
    p = pd.read_csv(base / 'results/diagnostics/cumulative_portfolio_difference.csv', parse_dates=['Return_Date'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, mapping, title in zip(axes, [PRIMARY, 'Raw6_Sensitivity'], ['Five factor variance mapping', 'Six factor positive forecast mapping']):
        for cohort, label, color, ls in [('Extended_175', '175 months', '#14648A', '-'), ('Original_88', '88 months', '#A66531', '--')]:
            z = p[(p.Allocation == mapping) & (p.Cohort == cohort) & (p.Sample == 'Full') &
                  (p.Comparison == 'Macro_minus_Historical_Mean')].sort_values('Return_Date')
            ax.plot(z.Return_Date, z.Cumulative_Arithmetic_Difference_pp, label=label, color=color, ls=ls, lw=1.6)
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color='#666666', lw=.7)
        ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'), color='#AAAAAA', alpha=.2)
        ax.set_ylabel('Cumulative arithmetic difference (pp)')
        time_axis(ax, p[p.Cohort == 'Extended_175'].Return_Date, every=5)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    records = [save(fig, out, 4, 'Macro increment under two allocation mappings')]

    d = pd.read_csv(base / 'results/international/model_shrinkage_diagnostics.csv').sort_values('prediction_minus_mean_rms_bp')
    if len(d) != 20:
        raise ValueError('Expected all 20 international forecast models.')
    fig, ax = plt.subplots(figsize=(11, 7.5))
    labels = [f'{r.country} | {r.theme} | {"Macro + state" if r.model == "Macro+Regime" else r.model}' for r in d.itertuples()]
    ax.barh(np.arange(len(d)), d.prediction_minus_mean_rms_bp,
            color=np.where(d.alpha_at_upper_boundary, '#14648A', '#A66531'))
    ax.set_yticks(np.arange(len(d)), labels, fontsize=10)
    ax.set_xscale('log')
    ax.axvline(1, color='#333333', ls='--', lw=.8)
    ax.set_xlabel('RMS deviation from matched mean (bp, logarithmic scale)')
    ax.set_title('International forecast shrinkage', loc='left', weight='bold')
    ax.legend(handles=[Patch(facecolor='#14648A', label='Penalty at upper grid boundary'),
                       Patch(facecolor='#A66531', label='Penalty below boundary')], loc='lower right', frameon=False, fontsize=10)
    fig.tight_layout()
    records.append(save(fig, out, 5, 'International forecast deviation from historical means'))

    d = pd.read_csv(base / 'results/diagnostics/cumulative_forecast_loss.csv', parse_dates=['Return_Date'])
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)
    for ax, factor in zip(axes.flat, FACTORS):
        for estimator, label, color, ls in [('OLS', 'OLS vs matched mean (172)', '#14648A', '-'),
                                           ('Ensemble', 'Forecast average vs allocation mean (175)', '#A66531', '--')]:
            z = d[(d.Cohort == 'Extended_175') & (d.Factor == factor) & (d.Estimator == estimator)].sort_values('Return_Date')
            ax.plot(z.Return_Date, z.Cumulative_Raw_SSE_Gain_pp2, label=label, color=color, ls=ls, lw=1.4)
        ax.set_title(factor)
        ax.axhline(0, color='#555555', lw=.6)
        ax.set_ylabel('Cumulative SSE gain (pp²)')
        time_axis(ax, d[d.Cohort == 'Extended_175'].Return_Date, every=5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .95))
    records.append(save(fig, out, 6, 'Raw forecast loss accounting with explicitly distinct benchmarks'))
    return records



def annual_figure(base, out):
    d = pd.read_csv(base / 'results/robustness/annual_incremental_returns.csv')
    d = d[d.Cohort == 'Extended_175'].sort_values('Year').set_index('Year')
    if len(d) != 15 or d.N_Months.sum() != 175:
        raise ValueError('Annual figure requires all 175 primary target months')
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1.3]})
    x = np.arange(len(d)); width = .38
    axes[0].bar(x - width/2, d.Historical_Mean_Compounded_Return_pct, width,
                color='#707070', label='Historical mean')
    axes[0].bar(x + width/2, d.Macro_Compounded_Return_pct, width,
                color='#14648A', label='Macro forecast average')
    axes[0].set_ylabel('Compounded return (%)')
    axes[0].set_title('Absolute returns and relative allocation gains', loc='left', weight='bold')
    axes[0].legend(frameon=False, ncol=2, loc='upper left', fontsize=10)
    j2022=list(d.index).index(2022)
    axes[0].annotate(f'{d.loc[2022,"Macro_Compounded_Return_pct"]:.1f}%',
                    (j2022+width/2, d.loc[2022,'Macro_Compounded_Return_pct']),
                    xytext=(0,6), textcoords='offset points', ha='center', fontsize=10)
    gains=d.Macro_minus_HM_Arithmetic_Sum_pp
    axes[1].bar(x, gains, width=.65,
                color=np.where(gains >= 0, '#14648A', '#A66531'))
    axes[1].set_ylabel('Summed paired\nMacro minus mean (pp)')
    axes[1].set_title('Annual contributions to the arithmetic increment', loc='left', fontsize=11)
    for year in [2022,2024]:
        j=list(d.index).index(year)
        axes[1].annotate(f'{gains.loc[year]:.2f}', (j,gains.loc[year]),
                        xytext=(0,5), textcoords='offset points', ha='center', fontsize=9)
    axes[1].set_xticks(x, [str(y) if d.loc[y,'N_Months']==12 else f'{y}\nJan-Jul' for y in d.index])
    axes[1].tick_params(axis='x', labelsize=9)
    axes[1].set_xlabel('Return year; 2026 is seven observed months, not annualized')
    for ax in axes:
        ax.axhline(0, color='#333333', lw=.8)
        ax.grid(axis='y', alpha=.18)
    fig.tight_layout(h_pad=1.8)
    return [save(fig, out, 7, 'Annual absolute returns and additive relative gains')]


def cross_market_figure(base, out):
    """Show paired conditional uncertainty and actual target-weight distance."""
    d = pd.read_csv(base / 'results/cross_market/allocation_summary.csv').set_index('Country')
    us = pd.read_csv(base / 'results/us/studentized_intervals.csv')
    us = us[(us.Design == DESIGN) & (us.Cohort == 'Extended_175') &
            (us.Allocation == PRIMARY) & (us.Cost_bps == 10) &
            (us.Sample == 'Full') & (us.Comparison == 'Macro_minus_Historical_Mean') &
            (us.Block_Length == 6) & (us.HAC_Lags == 3)]
    if len(us) != 1:
        raise ValueError('Expected one US primary Sharpe interval')
    w = pd.read_csv(base / 'results/us/weights.csv')
    w = w[(w.Design == DESIGN) & (w.Allocation == PRIMARY)]
    factors = FACTORS[1:]
    a = w[w.Strategy == 'Macro'].sort_values('Return_Date')[factors].to_numpy()
    b = w[w.Strategy == 'Historical_Mean'].sort_values('Return_Date')[factors].to_numpy()
    rows = [('US', 'Jan 2012–Jul 2026; N=175', float(us.iloc[0].Sharpe_Difference),
             float(us.iloc[0].CI_Low), float(us.iloc[0].CI_High), float(.5 * np.abs(a-b).sum(axis=1).mean()))]
    for country, period in [('UK', 'Dec 2014–Dec 2025; N=133'), ('DE', 'Nov 2015–Dec 2025; N=122')]:
        r = d.loc[country]
        rows.append((country, period, r.Macro_vs_HM_Sharpe_Difference,
                     r.Macro_vs_HM_CI_Low, r.Macro_vs_HM_CI_High,
                     r.Macro_vs_HM_Mean_Half_L1_Weight_Difference))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.75, 1]})
    y = np.arange(3)[::-1]
    for pos, (country, period, point, low, high, distance) in zip(y, rows):
        axes[0].hlines(pos, low, high, color='#14648A', lw=2)
        axes[0].plot(point, pos, 'o', color='#14648A', ms=6)
        axes[0].text(.99, pos+.19, f'ΔSR {point:+.5f}' if country == 'DE' else f'ΔSR {point:+.3f}',
                     transform=axes[0].get_yaxis_transform(), ha='right', fontsize=10)
        axes[1].barh(pos, distance*100, height=.38, color='#688EA8')
        axes[1].text(distance*100+.15, pos, f'{distance*100:.2f}%', va='center', fontsize=10)
    axes[0].set_yticks(y, [country+'\n'+period for country, period, *_ in rows], fontsize=10)
    axes[0].set_xlim(-.12, .30); axes[0].set_ylim(-.5, 2.55)
    axes[0].axvline(0, color='#555555', lw=.8)
    axes[0].set_title('Macro minus historical mean', loc='left', fontsize=11, weight='bold')
    axes[0].set_xlabel('Annualized Sharpe difference\n95% conditional paired interval')
    axes[1].set_yticks(y, [r[0] for r in rows]); axes[1].set_xlim(0, 10)
    axes[1].set_ylim(-.5, 2.55)
    axes[1].set_title('Distance between allocations', loc='left', fontsize=11, weight='bold')
    axes[1].set_xlabel('Average half-L1 weight difference (%)')
    for ax in axes: ax.grid(axis='x', alpha=.18)
    fig.suptitle('Frozen allocation rule across three inspected markets', x=.02, ha='left', fontsize=13, weight='bold')
    fig.text(.02, .015, '10 bp sleeve charge; 3,000 paired moving-block draws, block 6, HAC3. Different forecast designs and periods.\nGerman near-identical paths compress both the effect and its conditional interval; this is not a stronger information test.', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .92), w_pad=2)
    return [save(fig, out, 8, 'Cross-market paired Sharpe increments and allocation distance')]


def run_analysis(base_path=None):
    base = Path(base_path).resolve() if base_path is not None else Path(__file__).resolve().parents[1]
    out = base / 'results/figures'
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        records = us_figures(base, out) + diagnostic_figures(base, out) + annual_figure(base, out) + cross_market_figure(base, out)
    (out / 'figure_manifest.json').write_text(json.dumps({'figures': records,
        'source': 'Computed study results', 'model_refits': False,
        'matplotlib': matplotlib.__version__}, indent=2) + '\n', encoding='utf-8')
    return records


if __name__ == '__main__':
    run_analysis()

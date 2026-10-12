"""Conditional conditional diagnostics; never replace earlier fitted forecasts.

Inputs are fixed fixed-specification current-vintage paths and initial training data.
Writes only results/diagnostics. No downloads, rolling-model redesign or MCS.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from . import us_core as u
from . import us_design as prior

BASE = Path(__file__).resolve().parents[1]
DESIGN = 'Release_corrected_current_vintage'
PRIMARY = 'LS5_Variance36_Cap40_Shrink50'
RAW = 'Raw6_Sensitivity'
COHORTS = {'Extended_175': ('2012-01-31', '2026-07-31'),
           'Original_88': ('2019-04-30', '2026-07-31')}
PAIRS = [('Macro', 'Historical_Mean'), ('Macro_M1_YoY', 'Macro'),
         ('Macro_M1_ExpandingYoY', 'Macro')]
MOM_COSTS = [0., 35., 48.4, 65.]
MOM_COST_SOURCE = ('Historical sensitivity anchors from Novy-Marx and Velikov, August 2015, 64-page author manuscript; '
    '48.4bps UMD, while 35/65bps refer to different decile momentum portfolios. '
    'Hypothetical constant monthly charges, not reconstructed current factor costs or a verified final-published table.')
MOM_COST_SOURCE_URL = 'https://mysimon.rochester.edu/novy-marx/research/ToAatTC.pdf'
MOM_COST_SOURCE_SHA256 = 'af18adac3aec357d7ce7c323e020e5e1d002c3cb44413b4f27b18f434d39b76f'
SEED = 20261011


def sample_frames(frame):
    yield 'Full', frame
    yield 'Exclude_2020_Mar_Dec', frame[~frame.Return_Date.between('2020-03-01', '2020-12-31')]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(base):
    paths = ['results/us/' + x + '.csv' for x in
             ['returns', 'weights', 'forecasts', 'studentized_intervals']]
    paths += ['results/design/' + DESIGN + x for x in
              ['_model_calendar.csv', '_tuning.csv']]
    paths += ['results/training/training_block_placebo.csv']
    paths += ['data/us/' + x + '.csv' for x in
              ['VIXCLS', 'CPIAUCSL', 'BAA10Y', 'T10Y2Y', 'UNRATE', 'DGS2',
               'INDPRO', 'F-F_Research_Data_5_Factors_2x3', 'F-F_Momentum_Factor']]
    manifest = pd.DataFrame([{'Path': p, 'SHA256': sha256(base / p)} for p in paths])
    root = base / 'results/us'
    returns = pd.read_csv(root / 'returns.csv', parse_dates=['Return_Date'])
    weights = pd.read_csv(root / 'weights.csv', parse_dates=['Signal_Date', 'Return_Date'])
    forecasts = pd.read_csv(root / 'forecasts.csv', parse_dates=['Signal_Date', 'Return_Date'])
    intervals = pd.read_csv(root / 'studentized_intervals.csv')
    selected = lambda frame: frame[frame.Design.eq(DESIGN)].copy()
    return selected(returns), selected(weights), selected(forecasts), selected(intervals), manifest


def sharpe(r):
    r = np.asarray(r, float)
    return np.sqrt(12) * r.mean() / r.std(ddof=1)


def cer(r, gamma):
    """Annual mean-variance certainty-equivalent approximation, decimal units."""
    r = np.asarray(r, float)
    return 12 * (r.mean() - gamma / 2 * r.var(ddof=1))


def path_pairs(returns, cost=10):
    subset = returns[returns.Cost_bps.eq(cost)]
    for (cohort, allocation), group in subset.groupby(['Cohort', 'Allocation']):
        for sample, frame in sample_frames(group):
            panel = frame.pivot(index='Return_Date', columns='Strategy', values='Net_Return_pct').sort_index() / 100
            for a, b in PAIRS:
                key = {'Design': DESIGN, 'Cohort': cohort, 'Allocation': allocation,
                       'Sample': sample, 'Comparison': a + '_minus_' + b, 'Cost_bps': cost}
                yield key, panel, a, b


def block_intervals(returns, archived, draws):
    rows = []
    for key, panel, a, b in path_pairs(returns):
        old = archived.copy()
        for name, value in key.items():
            old = old[old[name].eq(value)]
        assert len(old) == 1
        previous = old.iloc[0]
        for block in [3, 6, 12]:
            if block == 6 and int(previous.Requested_Draws) == draws:
                result = previous.to_dict()
                result['Source'] = 'Primary-inference block6 result reused within this run'
            else:
                result = key | prior.studentized_sharpe_interval(
                    panel[a], panel[b], panel.index, B=draws, block=block, hac=3,
                    seed=int(previous.Seed) + 100000 * block)
                result['Source'] = 'Conditional additional block sensitivity, same frozen paths'
            result['Inference_Scope'] = 'Pointwise conditional path uncertainty; no fitting, selection or stability uncertainty'
            rows.append(result)
    return pd.DataFrame(rows)


def leave_one_month(returns):
    rows = []
    for key, panel, a, b in path_pairs(returns):
        ra, rb = panel[a].to_numpy(), panel[b].to_numpy()
        theta = sharpe(ra) - sharpe(rb)
        full_mean = 1200 * (ra - rb).mean()
        full_cer = {gamma: 100 * (cer(ra, gamma) - cer(rb, gamma)) for gamma in [3, 5]}
        for j, date in enumerate(panel.index):
            keep = np.arange(len(ra)) != j
            delta = sharpe(ra[keep]) - sharpe(rb[keep])
            rows.append(key | {'Deleted_Return_Date': date, 'N_Full': len(ra), 'N_Remaining': int(keep.sum()),
                'Full_Sharpe_Difference': theta, 'Deleted_Sharpe_Difference': delta,
                'Deletion_Change_in_Sharpe_Difference': delta - theta,
                'Full_Annual_Mean_Gain_pp': full_mean,
                'Deleted_Annual_Mean_Gain_pp': 1200 * (ra[keep] - rb[keep]).mean(),
                'Deleted_Month_Arithmetic_Gain_pp': 100 * (ra[j] - rb[j]),
                **{f'Deleted_CER_Gain_Gamma{gamma}_pp': 100 * (cer(ra[keep], gamma) - cer(rb[keep], gamma)) for gamma in [3, 5]},
                **{f'Full_CER_Gain_Gamma{gamma}_pp': full_cer[gamma] for gamma in [3, 5]},
                'Path_Rule': 'Delete one outcome only; all retained weights and original-time allocation charges unchanged'})
    frame = pd.DataFrame(rows)
    keys = ['Cohort', 'Allocation', 'Sample', 'Comparison']
    frame['Absolute_Influence_Rank'] = frame.groupby(keys).Deletion_Change_in_Sharpe_Difference.transform(
        lambda x: x.abs().rank(ascending=False, method='first')).astype(int)
    return frame


def calendar_blocks(returns):
    rows = []
    periods = [('2012_2015', '2012-01-01', '2015-12-31'),
               ('2016_2018', '2016-01-01', '2018-12-31'),
               ('2019_2021', '2019-01-01', '2021-12-31'),
               ('2022_2026_Jul', '2022-01-01', '2026-07-31')]
    periods += [(str(year), f'{year}-01-01', f'{year}-12-31') for year in range(2012, 2027)]
    for (cohort, allocation), group in returns[returns.Cost_bps.eq(10)].groupby(['Cohort', 'Allocation']):
        panel = group.pivot(index='Return_Date', columns='Strategy', values='Net_Return_pct').sort_index() / 100
        for label, lo, hi in periods:
            p = panel.loc[lo:hi]
            if len(p) < 2:
                continue
            for a, b in PAIRS:
                rows.append({'Design': DESIGN, 'Cohort': cohort, 'Allocation': allocation,
                    'Comparison': a + '_minus_' + b, 'Period': label,
                    'Grouping': 'Calendar_year' if len(label) == 4 else 'Calendar_block',
                    'First_Return': p.index.min(), 'Last_Return': p.index.max(), 'N': len(p),
                    'Annual_Mean_A_pp': 1200 * p[a].mean(), 'Annual_Mean_B_pp': 1200 * p[b].mean(),
                    'Annual_Mean_Gain_pp': 1200 * (p[a] - p[b]).mean(),
                    'Sharpe_A': sharpe(p[a]), 'Sharpe_B': sharpe(p[b]),
                    'Sharpe_Difference': sharpe(p[a]) - sharpe(p[b]),
                    'CER_Gain_Gamma3_pp': 100 * (cer(p[a], 3) - cer(p[b], 3)),
                    'CER_Gain_Gamma5_pp': 100 * (cer(p[a], 5) - cer(p[b], 5)),
                    'Cost_bps': 10, 'Rule': 'Descriptive short subperiod; original-time charges retained; no new free entry'})
    return pd.DataFrame(rows)


def economic_values(returns):
    strategy_rows, comparisons = [], []
    for (cohort, allocation, cost, strategy), group in returns.groupby(['Cohort', 'Allocation', 'Cost_bps', 'Strategy']):
        for sample, frame in sample_frames(group):
            r = frame.Net_Return_pct.to_numpy() / 100
            for gamma in [3, 5]:
                strategy_rows.append({'Design': DESIGN, 'Cohort': cohort, 'Allocation': allocation, 'Sample': sample,
                    'Strategy': strategy, 'Cost_bps': cost, 'Gamma': gamma, 'N': len(frame),
                    'Annual_Mean_pp': 1200 * r.mean(), 'Annual_Variance_Decimal': 12 * r.var(ddof=1),
                    'Annual_Volatility_pct': 100 * np.sqrt(12) * r.std(ddof=1), 'Sharpe': sharpe(r),
                    'CER_Annual_pp': 100 * cer(r, gamma), 'Average_Monthly_One_Way_Turnover': frame.Turnover.mean(),
                    'Annual_Sum_Rate_Turnover': 12 * frame.Turnover.mean(),
                    'Annual_Allocation_Cost_pp': 12 * frame.Allocation_Cost_pct.mean()})
    for cost in [0, 10, 25, 50]:
        for key, panel, a, b in path_pairs(returns, cost):
            for gamma in [3, 5]:
                gain = cer(panel[a], gamma) - cer(panel[b], gamma)
                comparisons.append(key | {'Gamma': gamma, 'N': len(panel),
                    'Annual_Mean_Gain_pp': 1200 * (panel[a] - panel[b]).mean(),
                    'CER_Gain_Annual_pp': 100 * gain, 'Mean_Variance_Fee_Equivalent_bp_per_year': 10000 * gain,
                    'Interpretation': 'Mean-variance approximation on synthetic excess returns, not exact CRRA welfare or investable management fee'})
    return pd.DataFrame(strategy_rows), pd.DataFrame(comparisons)


def crossing(ra, rb, exposure_a, exposure_b, gamma):
    """Solve the frozen-path annual CER difference as monthly sleeve drag changes.

    d is decimal return drag per month; cost in bps is d*10,000.
    For allocation cost, exposure is half target-weight turnover.
    """
    ra, rb, wa, wb = map(lambda x: np.asarray(x, float), (ra, rb, exposure_a, exposure_b))
    a = ra.mean() - rb.mean() - gamma / 2 * (ra.var(ddof=1) - rb.var(ddof=1))
    b = -(wa.mean() - wb.mean()) + gamma * (np.cov(ra, wa, ddof=1)[0, 1] - np.cov(rb, wb, ddof=1)[0, 1])
    c = -gamma / 2 * (wa.var(ddof=1) - wb.var(ddof=1))
    raw_roots = np.roots([c, b, a]) if abs(c) > 1e-15 else ([-a / b] if abs(b) > 1e-15 else [])
    positive = sorted(float(np.real(v)) for v in raw_roots if abs(np.imag(v)) < 1e-9 and np.real(v) > 1e-12)
    loss_roots = [v for v in positive if b + 2 * c * v < 0]
    first_loss = loss_roots[0] * 10000 if a > 0 and loss_roots else np.nan
    for value in positive:
        assert np.isclose(cer(ra - value * wa, gamma) - cer(rb - value * wb, gamma), 0, atol=1e-10)
    return {'Gamma': gamma, 'Baseline_Annual_Gain_pp': 1200 * a,
        'Monthly_Polynomial_Constant': a, 'Monthly_Polynomial_Linear': b, 'Monthly_Polynomial_Quadratic': c,
        'All_Positive_Root_bps': '|'.join(f'{v * 10000:.10g}' for v in positive),
        'First_Loss_of_Positive_Advantage_bps': first_loss,
        'Status': 'Already_nonpositive_at_zero' if a <= 0 else ('Positive_to_nonpositive_crossing' if np.isfinite(first_loss) else 'No_positive_to_nonpositive_root'),
        'Caution': 'Counterfactual constant charge on frozen weights; extrapolated crossings are not feasible-cost estimates'}


def costs(returns, weights):
    rows, comparisons, crossings = [], [], []
    for (cohort, allocation), group in returns[returns.Cost_bps.eq(10)].groupby(['Cohort', 'Allocation']):
        selected_weights = weights[weights.Allocation.eq(allocation)]
        for sample, frame in sample_frames(group):
            net = frame.pivot(index='Return_Date', columns='Strategy', values='Net_Return_pct').sort_index() / 100
            gross = frame.pivot(index='Return_Date', columns='Strategy', values='Gross_Return_pct').loc[net.index] / 100
            turns = frame.pivot(index='Return_Date', columns='Strategy', values='Turnover').loc[net.index]
            mom = selected_weights.pivot(index='Return_Date', columns='Strategy', values='MOM').loc[net.index, net.columns]
            key = {'Design': DESIGN, 'Cohort': cohort, 'Allocation': allocation, 'Sample': sample, 'N': len(net)}
            for cost in MOM_COSTS:
                charged = net - mom * cost / 10000
                for strategy in charged:
                    r = charged[strategy]
                    rows.append(key | {'Strategy': strategy, 'Allocation_Cost_bps': 10, 'MOM_Internal_Cost_bps_per_month': cost,
                        'Mean_MOM_Weight': mom[strategy].mean(), 'Annual_Internal_Cost_pp': .12 * cost * mom[strategy].mean(),
                        'Annual_Mean_pp': 1200 * r.mean(), 'Sharpe': sharpe(r),
                        'CER_Gamma3_Annual_pp': 100 * cer(r, 3), 'CER_Gamma5_Annual_pp': 100 * cer(r, 5),
                        'Scenario': MOM_COST_SOURCE})
                for a, b in PAIRS:
                    difference = charged[a] - charged[b]
                    drag = .12 * cost * (mom[a].mean() - mom[b].mean())
                    assert np.isclose(1200 * difference.mean(), 1200 * (net[a] - net[b]).mean() - drag)
                    comparisons.append(key | {'Comparison': a + '_minus_' + b,
                        'Allocation_Cost_bps': 10, 'MOM_Internal_Cost_bps_per_month': cost,
                        'Mean_MOM_Weight_A': mom[a].mean(), 'Mean_MOM_Weight_B': mom[b].mean(),
                        'Relative_Annual_Internal_Cost_pp': drag,
                        'Annual_Mean_Gain_pp': 1200 * difference.mean(), 'Sharpe_Difference': sharpe(charged[a]) - sharpe(charged[b]),
                        'CER_Gain_Gamma3_pp': 100 * (cer(charged[a], 3) - cer(charged[b], 3)),
                        'CER_Gain_Gamma5_pp': 100 * (cer(charged[a], 5) - cer(charged[b], 5))})
            for a, b in PAIRS:
                for gamma in [0, 3, 5]:
                    for kind, ra, rb, wa, wb, condition in [
                        ('Allocation_only', gross[a], gross[b], turns[a], turns[b], 'No internal drag; variable bps per one-way sleeve turnover'),
                        ('MOM_internal', net[a], net[b], mom[a], mom[b], 'Allocation charge held at10bps; variable monthly bps per unit MOM weight')]:
                        crossings.append(key | {'Comparison': a + '_minus_' + b, 'Cost_Type': kind,
                            'Condition': condition, **crossing(ra, rb, wa, wb, gamma)})
    return pd.DataFrame(rows), pd.DataFrame(comparisons), pd.DataFrame(crossings)


def cumulative_paths(returns, forecasts):
    portfolio, loss = [], []
    for key, panel, a, b in path_pairs(returns):
        delta = 100 * (panel[a] - panel[b])
        portfolio.append(pd.DataFrame(key | {'Return_Date': panel.index, 'Monthly_Difference_pp': delta.to_numpy(),
                                            'Cumulative_Arithmetic_Difference_pp': delta.cumsum().to_numpy()}))
    for cohort, (lo, hi) in COHORTS.items():
        fc = forecasts[forecasts.Return_Date.between(lo, hi)]
        for factor, group in fc.groupby('Factor'):
            group = group.sort_values('Return_Date')
            for estimator in ['OLS', 'Ensemble']:
                use = ~group.Macro_Fallback.astype(bool) if estimator == 'OLS' else pd.Series(True, index=group.index)
                model = group.Macro_OLS if estimator == 'OLS' else group.Macro
                benchmark = group.Nested_Mean_OLS if estimator == 'OLS' else group.Historical_Mean
                bm_sq = ((group.Actual - benchmark) ** 2).where(use)
                model_sq = ((group.Actual - model) ** 2).where(use)
                gain = bm_sq - model_sq
                loss.append(pd.DataFrame({'Design': DESIGN, 'Cohort': cohort, 'Factor': factor, 'Estimator': estimator,
                    'Return_Date': group.Return_Date.to_numpy(), 'Included_In_Comparison': use.to_numpy(),
                    'Benchmark_Squared_Error_pp2': bm_sq.to_numpy(), 'Model_Squared_Error_pp2': model_sq.to_numpy(),
                    'Raw_Squared_Error_Gain_pp2': gain.to_numpy(),
                    'Cumulative_Raw_SSE_Gain_pp2': gain.fillna(0).cumsum().to_numpy(),
                    'Cumulative_Compared_Observations': use.cumsum().to_numpy(),
                    'Benchmark': 'Matched training-label intercept mean' if estimator == 'OLS' else 'Fuller-history allocation mean',
                    'Fallback_Rule': 'Exclude fallback; cumulative line holds level across excluded dates' if estimator == 'OLS' else 'Include fallback; forecast equals benchmark so gain is zero',
                    'Interpretation': 'Raw SSE accounting, not Clark-West adjusted loss or a calibrated significance test'}))
    return pd.concat(portfolio, ignore_index=True), pd.concat(loss, ignore_index=True)


def penalty_diagnostics(base):
    data, _ = u.load_data(base)
    calendar, _, _ = prior.yoy_features(base, data, DESIGN)
    train = calendar.loc[:u.TRAIN_END].dropna(subset=u.PREDICTORS + [f + '_next' for f in u.FACTORS])
    assert len(train) == 261
    params, old = prior.frozen_parameters(base, DESIGN)
    scaler = StandardScaler().fit(train[u.PREDICTORS]);z = scaler.transform(train[u.PREDICTORS])
    singular2 = np.linalg.svd(z, compute_uv=False) ** 2
    df = lambda alpha: float(np.sum(singular2 / (singular2 + alpha)))
    initial, curves, selections, folds = [], [], [], []
    original_grid = np.logspace(-4, 6, 41)
    added = np.logspace(6.25, 10, 16)
    grid = np.r_[original_grid, added]
    splits = list(TimeSeriesSplit(n_splits=5).split(train))
    for f in u.FACTORS:
        alpha = params[f]['Ridge']['alpha'];en = ElasticNet(**params[f]['Elastic_Net'], max_iter=30000).fit(z, train[f + '_next'])
        active = np.abs(en.coef_) > 1e-10
        initial.append({'Design': DESIGN, 'Factor': f, 'Model': 'Ridge', 'Alpha': alpha,
            'N_Train': 261, 'Slope_Effective_DF': df(alpha), 'Total_Effective_DF': 1 + df(alpha),
            'Active_Predictor_Count': np.nan, 'L1_Ratio': np.nan,
            'Definition': 'trace[Z(ZtransZ+alpha I)^(-1)Ztrans], centered standardized design; add1 unpenalized intercept; conditional on fixed alpha'})
        initial.append({'Design': DESIGN, 'Factor': f, 'Model': 'Elastic_Net', 'Alpha': params[f]['Elastic_Net']['alpha'],
            'N_Train': 261, 'Slope_Effective_DF': np.nan, 'Total_Effective_DF': np.nan,
            'Active_Predictor_Count': int(active.sum()), 'L1_Ratio': params[f]['Elastic_Net']['l1_ratio'],
            'Active_Predictor_Names': '|'.join(np.asarray(u.PREDICTORS)[active]),
            'Definition': 'Active coefficient count at1e-10 is a complexity proxy only; not exact Elastic Net effective DF; intercept excluded'})
        y = train[f + '_next']
        for alpha_candidate in grid:
            fold_losses, fold_means = [], []
            for fold, (fit, val) in enumerate(splits, 1):
                pipe = Pipeline([('scale', StandardScaler()), ('model', Ridge(alpha=float(alpha_candidate)))])
                pipe.fit(train.iloc[fit][u.PREDICTORS], y.iloc[fit])
                pred = pipe.predict(train.iloc[val][u.PREDICTORS])
                loss = float(np.mean((y.iloc[val] - pred) ** 2))
                baseline = float(np.mean((y.iloc[val] - y.iloc[fit].mean()) ** 2))
                fold_losses.append(loss);fold_means.append(baseline)
                folds.append({'Factor': f, 'Alpha': alpha_candidate, 'Fold': fold, 'Training_N': len(fit),
                    'Validation_N': len(val), 'Training_Last_Label': train.index[fit[-1]] + pd.offsets.MonthEnd(),
                    'Validation_First_Signal': train.index[val[0]], 'MSE': loss, 'Matched_Mean_MSE': baseline})
                assert train.index[fit[-1]] + pd.offsets.MonthEnd() <= train.index[val[0]]
            curves.append({'Factor': f, 'Alpha': alpha_candidate, 'CV_MSE': np.mean(fold_losses),
                'CV_Matched_Mean_MSE': np.mean(fold_means), 'Slope_Effective_DF_Initial261': df(alpha_candidate),
                'CV_Relative_MSE_Gain_vs_Matched_Mean': 1 - np.mean(fold_losses) / np.mean(fold_means)})
        fcurves = pd.DataFrame(curves);fcurves = fcurves[fcurves.Factor.eq(f)]
        old_selected = old[(old.Factor == f) & (old.Model == 'Ridge')].iloc[0]
        for top in [1e6, 1e8, 1e10]:
            subset = fcurves[fcurves.Alpha.le(top * (1 + 1e-12))]
            selected = subset.loc[subset.CV_MSE.idxmin()]
            if top == 1e6:
                assert np.isclose(selected.Alpha, old_selected.alpha)
                assert np.isclose(selected.CV_MSE, old_selected.CV_MSE, atol=1e-9, rtol=1e-10)
            baseline = subset[subset.Alpha.eq(original_grid[-1])].iloc[0]
            selections.append(selected.to_dict() | {'Design': DESIGN, 'N_Train': 261, 'Fold_Count': 5,
                'Grid_Upper_Bound': top, 'Selected_At_Upper_Bound': bool(np.isclose(selected.Alpha, top)),
                'Grid_Candidates': len(subset), 'Original_Frozen_Alpha': old_selected.alpha,
                'CV_MSE_Change_vs_Original_Chosen': float(selected.CV_MSE - old_selected.CV_MSE),
                'CV_MSE_Change_vs_Alpha1e6': float(selected.CV_MSE - baseline.CV_MSE),
                'Total_Effective_DF_Initial261': 1 + selected.Slope_Effective_DF_Initial261,
                'Used_To_Replace_Primary_Penalty': False,
                'Interpretation': 'Initial-fold grid-boundary sensitivity only; no OOS selection or forecast replacement'})
    return pd.DataFrame(initial), pd.DataFrame(curves), pd.DataFrame(selections), pd.DataFrame(folds)


def placebo_adjustment(base):
    x = pd.read_csv(base / 'results/training/training_block_placebo.csv')
    assert len(x) == 6
    x['Holm_p_Six_Factors'] = multipletests(x.Block_Permutation_Share_At_Least_Observed, method='holm')[1]
    x['BH_q_Six_Factors_Exploratory'] = multipletests(x.Block_Permutation_Share_At_Least_Observed, method='fdr_bh')[1]
    x['Scope'] = 'Training-only block permutation tail shares; dependent-block exchangeability remains an assumption; not an OOS skill test'
    return x




def verify(base, returns, forecasts, intervals, loss, initial, selections, mom_comparison, crossings, manifest):
    checks = 0
    assert len(intervals) == 72 and intervals.Finite_Draws.eq(intervals.Requested_Draws).all();checks += 1
    assert set(intervals.Block_Length) == {3, 6, 12};checks += 1
    for row in manifest.itertuples():
        assert sha256(base / row.Path) == row.SHA256;checks += 1
    archived = pd.read_csv(base / 'results/us/studentized_intervals.csv')
    keys = ['Design', 'Cohort', 'Allocation', 'Sample', 'Comparison', 'Cost_bps']
    new6 = intervals[intervals.Block_Length.eq(6)].set_index(keys)
    old6 = archived[archived.Design.eq(DESIGN)].set_index(keys)
    same_draw_count = new6.loc[old6.index, 'Requested_Draws'].eq(old6.Requested_Draws)
    for col in ['Sharpe_Difference', 'CI_Low', 'CI_High', 'Original_HAC_SE']:
        compared = old6.index if col in ['Sharpe_Difference', 'Original_HAC_SE'] else old6.index[same_draw_count]
        assert np.allclose(new6.loc[compared, col].to_numpy(), old6.loc[compared, col].to_numpy(), rtol=1e-9, atol=1e-10, equal_nan=True);checks += 1
    h1 = pd.read_csv(base / 'results/us/direct_h1_forecast_loss.csv')
    for (cohort, factor, estimator), z in loss.groupby(['Cohort', 'Factor', 'Estimator']):
        match = h1[(h1.Design == DESIGN) & (h1.Cohort == cohort) & (h1.Factor == factor) &
            (h1.Estimator == estimator) & (h1.Method == 'DM_Loss_HAC') & (h1.HAC_Lags == 3)].iloc[0]
        assert z.Included_In_Comparison.sum() == match.N;checks += 1
        assert np.isclose(z.Raw_Squared_Error_Gain_pp2.mean(), match.Mean, atol=1e-10);checks += 1
        assert np.isclose(z.Cumulative_Raw_SSE_Gain_pp2.iloc[-1], z.Raw_Squared_Error_Gain_pp2.sum(), atol=1e-10);checks += 1
    assert len(initial) == 12 and len(selections) == 18 and not selections.Used_To_Replace_Primary_Penalty.any();checks += 1
    for (cohort, allocation), z in returns.groupby(['Cohort', 'Allocation']):
        zero = z[z.Cost_bps.eq(0)]
        ten = z[z.Cost_bps.eq(10)]
        assert np.allclose(ten.Net_Return_pct, ten.Gross_Return_pct - .1 * ten.Turnover);checks += 1
        assert np.allclose(zero.Net_Return_pct, zero.Gross_Return_pct);checks += 1
    baseline = mom_comparison[mom_comparison.MOM_Internal_Cost_bps_per_month.eq(0)]
    assert np.allclose(baseline.Relative_Annual_Internal_Cost_pp, 0);checks += 1
    raw_cross = crossings[(crossings.Cohort == 'Original_88') & (crossings.Allocation == RAW) &
        (crossings.Sample == 'Full') & (crossings.Comparison == 'Macro_minus_Historical_Mean') &
        (crossings.Cost_Type == 'Allocation_only') & (crossings.Gamma == 0)].iloc[0]
    assert np.isclose(raw_cross.First_Loss_of_Positive_Advantage_bps, 155.74350139650744, atol=1e-7);checks += 1
    return {'checks': checks, 'all_passed': True, 'source_inputs_unchanged': True,
        'block6_archived_point_and_interval_values_identical': bool(same_draw_count.all()),
        'forecast_loss_reconciles_with_us_analysis': True, 'bootstrap_rows': len(intervals),
        'finite_draws_min': int(intervals.Finite_Draws.min()),
        'scope': 'Numerical consistency checks; no claim to validate asymptotic assumptions or executable costs'}




def run_analysis(base_path=None, bootstrap_n=3000):
    base = Path(base_path).resolve() if base_path else BASE
    out = base / 'results/diagnostics';out.mkdir(parents=True, exist_ok=True)
    returns, weights, forecasts, archived, manifest = load_inputs(base)
    print('Diagnostics: unchanged current-vintage paths loaded', flush=True)
    intervals = block_intervals(returns, archived, bootstrap_n)
    influence = leave_one_month(returns)
    calendars = calendar_blocks(returns)
    values, gains = economic_values(returns)
    mom, mom_gains, crossings = costs(returns, weights)
    portfolio, loss = cumulative_paths(returns, forecasts)
    print('Diagnostics: path uncertainty, economic values and cost stresses complete', flush=True)
    initial, curves, selection, folds = penalty_diagnostics(base)
    placebo = placebo_adjustment(base)
    outputs = {'sharpe_block_sensitivity': intervals, 'leave_one_month_influence': influence,
        'leave_one_month_top5': influence[influence.Absolute_Influence_Rank.le(5)],
        'calendar_block_summary': calendars, 'cer_and_turnover': values, 'economic_value_comparisons': gains,
        'mom_internal_cost_sensitivity': mom, 'mom_internal_cost_comparisons': mom_gains,
        'cost_crossings': crossings, 'cumulative_portfolio_difference': portfolio,
        'cumulative_forecast_loss': loss, 'initial_effective_df': initial,
        'expanded_grid_cv': curves, 'expanded_grid_selection': selection, 'expanded_grid_folds': folds,
        'placebo_holm': placebo, 'input_manifest': manifest}
    for name, frame in outputs.items():frame.to_csv(out / (name + '.csv'), index=False)
    validation = verify(base, returns, forecasts, intervals, loss, initial, selection, mom_gains, crossings, manifest)
    (out / 'verification.json').write_text(json.dumps(validation, indent=2))
    summary = {'output': str(out), 'design': DESIGN, 'allocations': [PRIMARY, RAW],
        'primary_predictions_changed': False, 'cohorts': COHORTS,
        'bootstrap_draws': bootstrap_n, 'blocks': [3, 6, 12], 'hac': 3,
        'mom_costs_monthly_bps': MOM_COSTS, 'mom_cost_source': MOM_COST_SOURCE,
        'mom_cost_source_url': MOM_COST_SOURCE_URL, 'mom_cost_source_sha256': MOM_COST_SOURCE_SHA256, 'gammas': [3, 5],
        'initial_training_rows': 261, 'outputs': {k: len(v) for k, v in outputs.items()},
        'verification': validation, 'figures': 'Generated centrally by src/plot_figures.py in results/figures'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'output': str(out), 'checks': validation['checks'], 'source_inputs_unchanged': True}), flush=True)
    return summary



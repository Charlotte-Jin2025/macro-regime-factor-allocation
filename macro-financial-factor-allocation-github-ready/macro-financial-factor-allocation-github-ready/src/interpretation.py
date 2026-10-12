"""Read-only diagnostic summaries for the fixed academic specification.

No models, thresholds, penalties, predictions or allocation weights are fitted
or changed. All outputs go to results/interpretation. Input hashes record the
generated tables used in each execution. Decimal R², annual Sharpe and percentage-point
returns have separate columns; no mismatched-cohort R² products are used.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[1]
DESIGN = 'Release_corrected_current_vintage'
PRIMARY = 'LS5_Variance36_Cap40_Shrink50'
FACTORS = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
STATES = ['Expansion', 'Overheating', 'Slowdown', 'Stagflation']
COHORTS = {'Extended_175': ('2012-01-31', '2026-07-31'),
           'Original_88': ('2019-04-30', '2026-07-31')}
SPECIFICATIONS = [
    ('FixedYoY', 'Macro_M1_YoY', 'H2_YoY_OLS_Eligible',
     'Macro_OLS_Matched_YoY_Labels', 'Nested_Mean_M1_YoY', 'Regime_FixedYoY'),
    ('ExpandingYoY', 'Macro_M1_ExpandingYoY', 'H2_ExpandingYoY_OLS_Eligible',
     'Macro_OLS_Matched_ExpandingYoY_Labels', 'Nested_Mean_M1_ExpandingYoY', 'Regime_ExpandingYoY')]
INPUT_FILES = ['results/us/forecasts.csv', 'results/us/returns.csv', 'results/us/weights.csv', 'results/us/mde.csv', 'results/us/direct_h2_forecast_loss.csv', 'results/states/threshold_audit.csv', 'results/diagnostics/mom_internal_cost_sensitivity.csv', 'results/diagnostics/mom_internal_cost_comparisons.csv', 'results/diagnostics/sharpe_block_sensitivity.csv']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_upstream_csv(base):
    return {str(p.relative_to(base)): sha256(p) for p in sorted((base / 'results').rglob('*.csv'))
            if 'interpretation' not in p.relative_to(base).parts}


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(0, 1)
    result = np.empty(len(p)); result[order] = adjusted
    return result


def hac_se(scores, dates, lag):
    scores = np.asarray(scores, float)
    scores = scores - scores.mean()
    months = pd.DatetimeIndex(dates).to_period('M').asi8
    lookup = {m: i for i, m in enumerate(months)}
    meat = np.dot(scores, scores)
    for k in range(1, lag + 1):
        pairs = [(i, lookup[m - k]) for i, m in enumerate(months) if m - k in lookup]
        if pairs:
            a, b = np.asarray(pairs).T
            meat += 2 * (1 - k / (lag + 1)) * np.dot(scores[a], scores[b])
    return np.sqrt(max(meat, 0)) / len(scores)


def sharpe_influence(pair):
    pair = np.asarray(pair, float)
    n = len(pair); means = pair.mean(axis=0); centered = pair - means
    variance = (centered ** 2).mean(axis=0); sd = np.sqrt(variance)
    correction = np.sqrt((n - 1) / n)
    sr = np.sqrt(12) * means / pair.std(axis=0, ddof=1)
    scores = np.sqrt(12) * correction * (centered / sd -
             means * (centered ** 2 - variance) / (2 * sd ** 3))
    return sr[0] - sr[1], scores[:, 0] - scores[:, 1]


def mse(actual, prediction):
    return float(np.mean((np.asarray(actual) - np.asarray(prediction)) ** 2))


def forecast_summaries(fc, thresholds):
    keys = ['Design', 'Signal_Date', 'Return_Date']
    labels = thresholds[keys + ['Regime_FixedYoY', 'Regime_ExpandingYoY']]
    assert not labels.duplicated(keys).any()
    merged = fc.merge(labels, on=keys, validate='many_to_one', how='left', indicator=True,
                      suffixes=('', '_Audit'))
    assert merged['_merge'].eq('both').all()
    for label in ['Regime_FixedYoY', 'Regime_ExpandingYoY']:
        if label + '_Audit' in merged:
            assert merged[label].fillna('Missing').eq(merged[label + '_Audit'].fillna('Missing')).all()
            merged = merged.drop(columns=[label + '_Audit'])
    assert not merged.duplicated(keys + ['Factor']).any()
    rows, tests, coverage = [], [], []
    for design, design_fc in merged.groupby('Design'):
        for cohort, (lo, hi) in COHORTS.items():
            sample = design_fc[design_fc.Return_Date.between(lo, hi)]
            expected = 175 if cohort == 'Extended_175' else 88
            assert len(sample) == 6 * expected
            for variant, model, eligible, reduced, matched, label in SPECIFICATIONS:
                unique = sample.drop_duplicates('Return_Date')
                for state in STATES + ['Missing']:
                    mask = unique[label].isna() if state == 'Missing' else unique[label].eq(state)
                    coverage.append({'Design': design, 'Cohort': cohort, 'State_Variant': variant,
                                     'State': state, 'N_All_Signals': int(mask.sum()),
                                     'N_Conditional_Eligible': int((mask & unique[eligible]).sum()),
                                     'N_All_Cohort': expected,
                                     'Count_Unit': 'unique signal months, not six factor observations'})
                for factor, group in sample.groupby('Factor'):
                    group = group.sort_values('Return_Date')
                    z = group[group[eligible].astype(bool)].dropna(subset=[reduced, model + '_OLS', matched])
                    assert len(z) == expected - 3
                    assert (~z[model + '_Fallback'].astype(bool)).all()
                    for scope, frame in [('Conditional_eligible_only', z),
                                         ('All_issued_including_HM_fallback', group)]:
                        scopes = [('All', frame)]
                        if scope == 'Conditional_eligible_only':
                            scopes += [(state, frame[frame[label].eq(state)]) for state in STATES]
                        for state, q in scopes:
                            record = {'Design': design, 'Cohort': cohort, 'State_Variant': variant,
                                      'State_Model': model, 'Factor': factor, 'Scope': scope,
                                      'Signal_State': state, 'N': len(q),
                                      'N_Eligible': int(q[eligible].sum()), 'N_Fallback': int((~q[eligible]).sum()),
                                      'MSE_Units': 'squared percentage points',
                                      'Allocation_HM': 'All available factor-return history to signal date',
                                      'Matched_HM': 'Intercept-only mean on exactly the state OLS training labels',
                                      'Reduced_Macro': 'Seven-feature OLS on exactly the state OLS training labels',
                                      'Fallback_Convention': 'State OLS and reduced Macro use fuller-history allocation HM on unavailable signals; matched HM remains a diagnostic training mean'}
                            if len(q):
                                actual = q.Actual
                                reduced_prediction = q[reduced].where(q[eligible], q.Historical_Mean)
                                losses = {'Allocation_HM': mse(actual, q.Historical_Mean),
                                          'Matched_HM': mse(actual, q[matched]),
                                          'Reduced_Macro_OLS': mse(actual, reduced_prediction),
                                          'State_OLS': mse(actual, q[model + '_OLS'])}
                                record.update({'MSE_' + k: v for k, v in losses.items()})
                                for benchmark in ['Allocation_HM', 'Matched_HM', 'Reduced_Macro_OLS']:
                                    record['State_OLS_R2_vs_' + benchmark] = 1 - losses['State_OLS'] / losses[benchmark]
                                record['Reduced_Macro_R2_vs_Allocation_HM'] = 1 - losses['Reduced_Macro_OLS'] / losses['Allocation_HM']
                            rows.append(record)
                    raw = (z.Actual - z[reduced]) ** 2 - (z.Actual - z[model + '_OLS']) ** 2
                    cw = raw + (z[reduced] - z[model + '_OLS']) ** 2
                    for lag in [3, 6, 12]:
                        for method, loss in [('DM_Loss_HAC', raw), ('Clark_West', cw)]:
                            se = hac_se(loss, z.Return_Date, lag)
                            stat = loss.mean() / se
                            tests.append({'Design': design, 'Cohort': cohort, 'State_Model': model,
                                          'State_Variant': variant, 'Factor': factor, 'Estimator': 'OLS',
                                          'Method': method, 'HAC_Lags': lag, 'N': len(z),
                                          'Benchmark_MSE': mse(z.Actual, z[reduced]),
                                          'Model_MSE': mse(z.Actual, z[model + '_OLS']),
                                          'Mean': loss.mean(), 'HAC_SE': se, 'Statistic': stat,
                                          'p_Two_Sided': 2 * norm.sf(abs(stat)),
                                          'p_One_Sided_Positive': norm.sf(stat)})
    tests = pd.DataFrame(tests)
    family = ['Design', 'Cohort', 'State_Model', 'Estimator', 'Method', 'HAC_Lags']
    for key, ix in tests.groupby(family).groups.items():
        assert len(ix) == 6
        pcol = 'p_One_Sided_Positive' if key[-2] == 'Clark_West' else 'p_Two_Sided'
        tests.loc[ix, 'Holm_p_Six_Factors'] = holm(tests.loc[ix, pcol])
    accuracy = pd.DataFrame(rows)
    pcols = ['DM_Holm_p_HAC3', 'CW_Holm_p_HAC3']
    for i, method in enumerate(['DM_Loss_HAC', 'Clark_West']):
        z = tests[(tests.Method == method) & (tests.HAC_Lags == 3)]
        z = z[['Design', 'Cohort', 'State_Model', 'Factor', 'Holm_p_Six_Factors']].rename(columns={'Holm_p_Six_Factors': pcols[i]})
        accuracy = accuracy.merge(z, on=['Design', 'Cohort', 'State_Model', 'Factor'], validate='many_to_one')
    # Inferential p-values apply only to aggregate eligible samples, not strata or fallback summaries.
    ineligible = ~(accuracy.Scope.eq('Conditional_eligible_only') & accuracy.Signal_State.eq('All'))
    accuracy.loc[ineligible, pcols] = np.nan
    return accuracy, tests, pd.DataFrame(coverage)


def event_summaries(returns):
    rows = []
    events = pd.to_datetime(['2020-03-31', '2020-04-30', '2020-11-30'])
    selected = returns[(returns.Cost_bps.isin([0, 10])) & returns.Strategy.isin(['Macro', 'Historical_Mean'])]
    keys = ['Design', 'Cohort', 'Allocation', 'Cost_bps']
    for key, group in selected.groupby(keys):
        panel = group.pivot(index='Return_Date', columns='Strategy', values='Net_Return_pct').sort_index()
        for period, mask in [('2020_calendar_year', panel.index.year == 2020),
                             ('2020_Mar_Dec', (panel.index >= '2020-03-31') & (panel.index <= '2020-12-31')),
                             ('2020_Mar_Apr_Nov_selected', panel.index.isin(events))]:
            q = panel.loc[mask]
            assert len(q) == (12 if period == '2020_calendar_year' else 10 if period == '2020_Mar_Dec' else 3)
            compounded = 100 * ((1 + q / 100).prod() - 1)
            rows.append(dict(zip(keys, key)) | {'Period': period, 'N': len(q),
                        'Macro_Arithmetic_Sum_pp': q.Macro.sum(), 'HM_Arithmetic_Sum_pp': q.Historical_Mean.sum(),
                        'Macro_minus_HM_Arithmetic_pp': (q.Macro - q.Historical_Mean).sum(),
                        'Macro_Compounded_Return_pct': compounded.Macro,
                        'HM_Compounded_Return_pct': compounded.Historical_Mean,
                        'Macro_minus_HM_Compounded_pp': compounded.Macro - compounded.Historical_Mean,
                        'Interpretation': 'Additive arithmetic PnL differences are distinct from separately compounded portfolio-return differences'})
    return pd.DataFrame(rows)


def power_summaries(returns):
    rows = []
    selected = returns[(returns.Cost_bps == 10) & returns.Strategy.isin(['Macro', 'Historical_Mean'])]
    keys = ['Design', 'Cohort', 'Allocation']
    critical = norm.ppf(.975) + norm.ppf(.8)
    for key, group in selected.groupby(keys):
        panel = group.pivot(index='Return_Date', columns='Strategy', values='Net_Return_pct').sort_index()
        pair = panel[['Macro', 'Historical_Mean']].to_numpy() / 100
        delta, scores = sharpe_influence(pair)
        n = len(panel); monthly_sr = pair.mean(axis=0) / pair.std(axis=0, ddof=1)
        rho = np.corrcoef(pair.T)[0, 1]
        iid_var = (2 * (1 - rho) + .5 * (monthly_sr[0] ** 2 + monthly_sr[1] ** 2 -
                   2 * rho ** 2 * np.prod(monthly_sr))) / n
        iid_se = np.sqrt(12 * iid_var)
        for lag in [3, 6, 12]:
            se = hac_se(scores, panel.index, lag)
            mde = critical * se; iid_mde = critical * iid_se
            rows.append(dict(zip(keys, key)) | {'N': n, 'Cost_bps': 10, 'HAC_Lags': lag,
                         'Observed_Annual_Sharpe_Difference': delta, 'IF_HAC_SE_Annual': se,
                         'HAC_MDE_Annual_Sharpe': mde, 'IID_JK_Memmel_SE_Annual': iid_se,
                         'IID_MDE_Annual_Sharpe': iid_mde, 'Return_Correlation': rho,
                         'HAC_Years_For_Observed_Delta_SquareRootScaling': n / 12 * (mde / abs(delta)) ** 2,
                         'IID_Years_For_Observed_Delta_SquareRootScaling': n / 12 * (iid_mde / abs(delta)) ** 2,
                         'Observed_Delta_to_HAC_MDE': abs(delta) / mde,
                         'Alpha_TwoSided': .05, 'Power': .8,
                         'Caution': 'Conditional square-root-N extrapolation of this frozen paired path; assumes unchanged long-run variance and effect. Not a forecast of feasible calendar waiting time or a design selected before inspecting outcomes.'})
    return pd.DataFrame(rows)


def weight_ranges(weights):
    rows = []
    for cohort, (lo, hi) in COHORTS.items():
        selected = weights[weights.Return_Date.between(lo, hi)]
        for key, group in selected.groupby(['Design', 'Allocation', 'Strategy']):
            sleeve = FACTORS[1:] if key[1] == PRIMARY else FACTORS
            values = group[sleeve].to_numpy()
            rows.append(dict(zip(['Design', 'Allocation', 'Strategy'], key)) |
                        {'Cohort': cohort, 'N': len(group), 'Weight_Min': values.min(),
                         'Weight_Max': values.max(), 'Max_Sum_Weight_Error': np.max(np.abs(values.sum(axis=1) - 1)),
                         'Eligible_Factors': '|'.join(sleeve),
                         'Cap_Shrink_Applicable': key[1] == PRIMARY and key[2] in
                         ['Macro', 'Historical_Mean', 'Macro_M1_YoY', 'Macro_M1_ExpandingYoY']})
    return pd.DataFrame(rows)


def cost_summaries(returns, weights, old_cost):
    rows = []
    selected = returns[(returns.Cost_bps == 10) & returns.Strategy.isin(['Macro', 'Historical_Mean', 'Equal_Weight'])]
    keys = ['Design', 'Cohort', 'Allocation', 'Strategy']
    for key, group in selected.groupby(keys):
        group = group.sort_values('Return_Date')
        w = weights[(weights.Design == key[0]) & (weights.Allocation == key[2]) & (weights.Strategy == key[3])]
        q = group.merge(w[['Return_Date', 'MOM']], on='Return_Date', validate='one_to_one')
        assert len(q) == len(group)
        for cost in [0., 65.]:
            r = q.Net_Return_pct.to_numpy() / 100 - cost / 10000 * q.MOM.to_numpy()
            rows.append(dict(zip(keys, key)) | {'N': len(r), 'Allocation_Cost_bps': 10,
                         'MOM_Internal_Cost_bps_per_month': cost, 'Annual_Mean_pp': 1200 * r.mean(),
                         'Sharpe': np.sqrt(12) * r.mean() / r.std(ddof=1),
                         'CER_Gamma3_Annual_pp': 1200 * (r.mean() - 3 * r.var(ddof=1) / 2),
                         'CER_Gamma5_Annual_pp': 1200 * (r.mean() - 5 * r.var(ddof=1) / 2),
                         'Annual_Internal_Cost_pp': 12 * cost / 100 * q.MOM.mean(),
                         'Cost_Status': 'Hypothetical constant charge; historical portfolio-specific cost anchor, not estimated current tradable-factor cost'})
    result = pd.DataFrame(rows)
    # The old diagnostic charge was exported only for the main design.
    old = old_cost[(old_cost.Sample == 'Full') & old_cost.Strategy.isin(['Macro', 'Historical_Mean', 'Equal_Weight']) &
                   old_cost.MOM_Internal_Cost_bps_per_month.isin([0, 65])]
    on = keys + ['MOM_Internal_Cost_bps_per_month']
    joined = result.merge(old, on=on, suffixes=('_new', '_old'), validate='one_to_one')
    assert len(joined) == len(old)
    for col in ['Annual_Mean_pp', 'Sharpe', 'CER_Gamma3_Annual_pp', 'CER_Gamma5_Annual_pp', 'Annual_Internal_Cost_pp']:
        assert np.allclose(joined[col + '_new'], joined[col + '_old'], atol=2e-12, rtol=2e-12)
    return result


def run_analysis(base_path=None):
    base = Path(base_path).resolve() if base_path else BASE
    out = base / 'results/interpretation'; out.mkdir(parents=True, exist_ok=True)
    before = snapshot_upstream_csv(base)
    read = lambda relative, dates=None: pd.read_csv(base / relative, parse_dates=dates)
    fc = read('results/us/forecasts.csv', ['Signal_Date', 'Return_Date'])
    returns = read('results/us/returns.csv', ['Return_Date'])
    weights = read('results/us/weights.csv', ['Signal_Date', 'Return_Date'])
    thresholds = read('results/states/threshold_audit.csv', ['Signal_Date', 'Return_Date'])
    old_tests = read('results/us/direct_h2_forecast_loss.csv')
    old_mde = read('results/us/mde.csv')
    old_cost = read('results/diagnostics/mom_internal_cost_sensitivity.csv')
    intervals = read('results/diagnostics/sharpe_block_sensitivity.csv')
    accuracy, tests, coverage = forecast_summaries(fc, thresholds)
    test_keys = ['Design', 'Cohort', 'State_Model', 'Factor', 'Estimator', 'Method', 'HAC_Lags']
    old_ols = old_tests[old_tests.Estimator == 'OLS']
    joined = tests.merge(old_ols, on=test_keys, suffixes=('_new', '_old'), validate='one_to_one')
    assert len(joined) == len(old_ols) == 288
    for col in ['Benchmark_MSE', 'Model_MSE', 'Mean', 'HAC_SE', 'Statistic', 'p_Two_Sided', 'p_One_Sided_Positive', 'Holm_p_Six_Factors']:
        assert np.allclose(joined[col + '_new'], joined[col + '_old'], atol=2e-10, rtol=2e-10)
    power = power_summaries(returns)
    mde_key = ['Design', 'Cohort', 'Allocation', 'HAC_Lags']
    old_h1 = old_mde[(old_mde.Sample == 'Full') & (old_mde.Comparison == 'Macro_minus_Historical_Mean')]
    joined = power.merge(old_h1, on=mde_key, suffixes=('_new', '_old'), validate='one_to_one')
    assert len(joined) == len(old_h1) == 24
    for new, old in [('Observed_Annual_Sharpe_Difference_new', 'Observed_Annual_Sharpe_Difference_old'),
                     ('IF_HAC_SE_Annual', 'Actual_Paired_IF_HAC_SE_Annual'),
                     ('HAC_MDE_Annual_Sharpe', 'Approx_MDE_Annual_Sharpe_80pct'),
                     ('IID_JK_Memmel_SE_Annual', 'IID_JK_Memmel_SE_Annual_Secondary')]:
        assert np.allclose(joined[new], joined[old], atol=2e-12, rtol=2e-12)
    events = event_summaries(returns)
    ranges = weight_ranges(weights)
    capped = ranges[ranges.Cap_Shrink_Applicable]
    assert capped.Weight_Min.ge(.1 - 1e-12).all() and capped.Weight_Max.le(.4 + 1e-12).all()
    assert ranges.Max_Sum_Weight_Error.le(1e-12).all()
    costs = cost_summaries(returns, weights, old_cost)
    negative = intervals[intervals.CI_High < 0].copy()
    negative['Claim_Scope'] = 'Pointwise conditional-path interval, no simultaneous multiple-comparison coverage; short excluded segment remains fixed when block exceeds its length'
    primary = accuracy[(accuracy.Design == DESIGN) & (accuracy.Cohort == 'Extended_175') &
                       (accuracy.Scope == 'Conditional_eligible_only') & (accuracy.Signal_State == 'All')]
    assert len(primary) == 12
    assert primary.N.eq(172).all()
    main = primary.pivot(index='Factor', columns='State_Variant', values='State_OLS_R2_vs_Allocation_HM').reindex(FACTORS)
    main.columns = ['Expanding_State_OLS_R2_vs_Allocation_HM', 'Fixed_State_OLS_R2_vs_Allocation_HM']
    main = main.reset_index(); main['N_Eligible'] = 172
    main['R2_Unit'] = 'decimal; multiply by 100 to display percent'
    tables = {'state_ols_accuracy': accuracy, 'state_ols_main_comparison': main, 'h2_inference_recheck': tests,
              'state_coverage': coverage, 'event_2020_mapping_comparison': events, 'power_frozen_paths': power,
              'weight_ranges': ranges, 'absolute_cost_and_cer': costs, 'negative_pointwise_state_intervals': negative}
    for name, frame in tables.items():
        frame.to_csv(out / (name + '.csv'), index=False)
    after = snapshot_upstream_csv(base)
    assert before == after, 'Upstream scientific results were modified'
    pd.DataFrame([{'Path': path, 'SHA256': digest, 'Frozen_Explicit_Read_Input': path in INPUT_FILES}
                  for path, digest in before.items()]).to_csv(out / 'upstream_integrity.csv', index=False)
    rmw = primary[primary.Factor == 'RMW']
    main_power = power[(power.Design == DESIGN) & (power.Cohort == 'Extended_175') & (power.Allocation == PRIMARY)]
    main_events = events[(events.Design == DESIGN) & (events.Cohort == 'Extended_175')]
    summary = {'scope': 'Frozen diagnostic summaries; no refitting or new variants',
               'RMW_175_eligible': rmw.to_dict(orient='records'),
               'main_power': main_power.to_dict(orient='records'),
               'main_2020_mapping': main_events.to_dict(orient='records'),
               'output_rows': {name: len(frame) for name, frame in tables.items()},
               'all_upstream_csv_unchanged': True, 'upstream_csv_count': len(before)}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    checks = {'passed': True, 'frozen_explicit_inputs': len(INPUT_FILES), 'unchanged_upstream_csv': len(before),
              'eligible_accuracy_rows': int((accuracy.Scope == 'Conditional_eligible_only').sum()),
              'H2_OLS_inference_reproduced_rows': len(old_ols), 'H1_MDE_reproduced_rows': len(old_h1),
              'cost_recheck': True, 'capped_weight_range': [.1, .4], 'all_weight_sums_one': True,
              'no_model_refits': True, 'no_added_model_variants': True,
              'output_sha256': {str(p.name): sha256(p) for p in sorted(out.glob('*.csv'))}}
    (out / 'verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    return summary




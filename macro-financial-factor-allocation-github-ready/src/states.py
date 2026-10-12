"""Retrospective frozen/expanding YoY state diagnostics and 175-origin forecasts.

Writes only results/states. Initial 261 states use the old frozen
training medians. Later expanding states are generated once with medians of
previously observed signal features, then retained unchanged in later fits.
No downloads, imputation, hyperparameter search, or performance-driven choice.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import us_core as u
from . import us_design as prior

BASE = Path(__file__).resolve().parents[1]
MODEL = 'Macro_M1_ExpandingYoY'
STATE = 'Regime_ExpandingYoY'
STATE_INPUTS = ['IP_YoY_lag', 'CPI_YoY_lag']
DUMMIES = ['State_ExpandingYoY_' + r for r in u.REGIMES[1:]]
KEYS = ['Design', 'Signal_Date', 'Return_Date', 'Factor']
SPLITS = [('Training', u.START, u.TRAIN_END),
          ('Validation', u.TRAIN_END + pd.offsets.MonthEnd(), u.VALID_END),
          ('Evaluation', u.VALID_END + pd.offsets.MonthEnd(), u.SIGNAL_END)]


def date_string(value):
    return pd.Timestamp(value).date().isoformat() if pd.notna(value) else None


def stable_hash(series):
    """Audit hash of dated values, not a substitute for the exported observations."""
    payload = '\n'.join(f'{date_string(k)}={v}' for k, v in series.items())
    return hashlib.sha256(payload.encode()).hexdigest()


def split_for(date):
    return 'Training' if date <= u.TRAIN_END else ('Validation' if date <= u.VALID_END else 'Evaluation')


def expanding_state_calendar(calendar, design):
    x = calendar.copy()
    targets = [f + '_next' for f in u.FACTORS]
    seed = x.loc[:u.TRAIN_END].dropna(subset=u.PREDICTORS + targets)
    assert len(seed) == 261 and seed.index.equals(x.loc[:u.TRAIN_END].index)
    fixed = seed[STATE_INPUTS].median()
    history = {v: [(date, float(value)) for date, value in seed[v].dropna().items()]
               for v in STATE_INPUTS}
    cuts = pd.DataFrame(index=x.index, columns=STATE_INPUTS, dtype=float)
    threshold_rows = []
    for date, row in x.iterrows():
        initial = date <= u.TRAIN_END
        record = {'Design': design, 'Signal_Date': date,
                  'Return_Date': date + pd.offsets.MonthEnd(), 'Split': split_for(date),
                  'Threshold_Mode': 'Initial_training_fixed' if initial else 'Previous_signals_expanding',
                  'Initial_Seed_N': len(seed)}
        for variable in STATE_INPUTS:
            observed = history[variable]
            threshold = float(fixed[variable]) if initial else float(np.median([v for _, v in observed]))
            cuts.loc[date, variable] = threshold
            max_date = observed[-1][0]
            if not initial:
                assert max_date < date
            record.update({variable: row[variable], variable + '_Fixed_Median': float(fixed[variable]),
                           variable + '_Expanding_Median': threshold,
                           variable + '_Threshold_N': len(observed),
                           variable + '_Threshold_Max_Signal_Date': max_date,
                           variable + '_Threshold_Max_Reference_Month': max_date - pd.offsets.MonthEnd()})
        threshold_rows.append(record)
        # Current signal enters only future thresholds, independently for each
        # state variable; unavailable other macro features do not suppress it.
        if not initial:
            for variable in STATE_INPUTS:
                if pd.notna(row[variable]):
                    history[variable].append((date, float(row[variable])))
    x[STATE] = u.regimes_from(x.IP_YoY_lag, x.CPI_YoY_lag, cuts.IP_YoY_lag, cuts.CPI_YoY_lag)
    for regime, column in zip(u.REGIMES[1:], DUMMIES):
        x[column] = (x[STATE] == regime).fillna(False).astype(float)
    assert x.loc[:u.TRAIN_END, STATE].equals(x.loc[:u.TRAIN_END, 'Regime_YoY'])
    # Independent vectorized construction verifies the sequential threshold
    # rule; it is an audit only, not a relabeling step in the model fits.
    expected = x[STATE_INPUTS].expanding().median().shift(1)
    assert np.allclose(cuts.loc[u.TRAIN_END + pd.offsets.MonthEnd():],
                       expected.loc[u.TRAIN_END + pd.offsets.MonthEnd():], equal_nan=True)
    for variable in STATE_INPUTS:
        x[variable + '_Expanding_Median'] = cuts[variable]
        x[variable + '_Fixed_Median'] = fixed[variable]
    threshold_audit = pd.DataFrame(threshold_rows)
    threshold_audit['Regime_FixedYoY'] = x.Regime_YoY.values
    threshold_audit[STATE] = x[STATE].values
    threshold_audit['Continuous_Predictors_Available'] = x[u.PREDICTORS].notna().all(axis=1).values
    threshold_audit['State_Available'] = x[STATE].notna().values
    threshold_audit['Conditional_Forecast_Eligible'] = (
        x[u.PREDICTORS].notna().all(axis=1) & x[STATE].notna()).values
    return x, threshold_audit


def state_diagnostics(x, design):
    frequency, diagnostics, persistence, autocorrelation, cross_tabs = [], [], [], [], []
    periods = SPLITS + [('2019_2020_signal_months', pd.Timestamp('2019-03-31'), pd.Timestamp('2020-12-31'))]
    for split, start, end in periods:
        frame = x.loc[start:end]
        for variant, column in [('FrozenYoY', 'Regime_YoY'), ('ExpandingYoY', STATE)]:
            states = frame[column].astype('string')
            n_valid, n_total = int(states.notna().sum()), len(frame)
            eligible = states.notna() & frame[u.PREDICTORS].notna().all(axis=1)
            high_growth = int(states.isin(['Expansion', 'Overheating']).sum())
            high_inflation = int(states.isin(['Overheating', 'Stagflation']).sum())
            for regime in u.REGIMES:
                count = int(states.eq(regime).sum())
                frequency.append({'Design': design, 'Variant': variant, 'Split': split,
                                  'Signal_Start': start, 'Signal_End': end, 'Regime': regime, 'N': count,
                                  'Percent_Of_Valid_States': 100 * count / n_valid,
                                  'Percent_Of_All_Signals': 100 * count / n_total,
                                  'N_Valid_States': n_valid, 'N_All_Signals': n_total})
            diagnostics.append({'Design': design, 'Variant': variant, 'Split': split,
                                'Signal_Start': start, 'Signal_End': end, 'N_All_Signals': n_total,
                                'N_Valid_States': n_valid, 'N_Missing_States': n_total - n_valid,
                                'N_Conditional_Model_Eligible': int(eligible.sum()),
                                'N_Issued_Forecasts': 0 if split == 'Training' else n_total,
                                'N_NonFallback_Forecasts': 0 if split == 'Training' else int(eligible.sum()),
                                'High_Growth_N_Among_Valid_States': high_growth,
                                'High_Growth_Percent_Of_Valid_States': 100 * high_growth / n_valid,
                                'High_Growth_Percent_Of_All_Signals': 100 * high_growth / n_total,
                                'High_Inflation_Percent_Of_Valid_States': 100 * high_inflation / n_valid,
                                'Count_Unit': 'unique signal months, not six factor observations'})
            adjacent = states.notna() & states.shift().notna()
            switches = int((states[adjacent] != states.shift()[adjacent]).sum())
            spell_ids = ((states != states.shift()).fillna(True) | ~adjacent).cumsum()
            spells = states.dropna().groupby(spell_ids[states.notna()]).size()
            persistence.append({'Design': design, 'Variant': variant, 'Split': split,
                                'N_Valid': n_valid, 'Adjacent_Pairs': int(adjacent.sum()),
                                'Switches': switches, 'Switch_Rate': switches / adjacent.sum(),
                                'N_Spells': len(spells), 'Mean_Spell_Months': spells.mean(),
                                'Median_Spell_Months': spells.median()})
        for variable in ['IP_YoY_lag', 'CPI_YoY_lag', 'IP_Growth_lag', 'Inflation_lag']:
            series = frame[variable]
            for lag in [1, 3, 12]:
                good = series.notna() & series.shift(lag).notna()
                autocorrelation.append({'Design': design, 'Split': split, 'Variable': variable,
                                        'Calendar_Lag': lag, 'N_Pairs': int(good.sum()),
                                        'Pearson_Autocorrelation': series.corr(series.shift(lag)),
                                        'Interpretation': 'descriptive; overlapping YoY windows and structural persistence are not separated'})
        for fixed_state in u.REGIMES:
            for expanding_state in u.REGIMES:
                cross_tabs.append({'Design': design, 'Split': split, 'Frozen_State': fixed_state,
                                   'Expanding_State': expanding_state,
                                   'N': int(((frame.Regime_YoY == fixed_state) & (frame[STATE] == expanding_state)).sum())})
    return {'state_frequencies': pd.DataFrame(frequency), 'state_availability': pd.DataFrame(diagnostics),
            'state_persistence': pd.DataFrame(persistence), 'overlap_autocorrelation': pd.DataFrame(autocorrelation),
            'state_cross_tab': pd.DataFrame(cross_tabs)}


def causality_audit(original, fitted_calendar, design):
    """Perturb current/future state features; past labels/current cuts must hold."""
    rows = []
    for date in pd.to_datetime(['2012-12-31', '2019-03-31', '2025-12-31']):
        changed = original.copy()
        changed.loc[date:, STATE_INPUTS] += 777.0
        alternate, _ = expanding_state_calendar(changed, design)
        cut_columns = [v + '_Expanding_Median' for v in STATE_INPUTS]
        current_and_past_cuts_unchanged = np.allclose(
            alternate.loc[:date, cut_columns], fitted_calendar.loc[:date, cut_columns], equal_nan=True)
        past_states_unchanged = alternate.loc[alternate.index < date, STATE].equals(
            fitted_calendar.loc[fitted_calendar.index < date, STATE])
        assert current_and_past_cuts_unchanged and past_states_unchanged
        rows.append({'Design': design, 'Perturbation_Start_Signal': date,
                     'Perturbation': 'add 777 pp to available current and future YoY state features',
                     'Thresholds_At_And_Before_Perturbation_Unchanged': current_and_past_cuts_unchanged,
                     'States_Strictly_Before_Perturbation_Unchanged': past_states_unchanged})
    return pd.DataFrame(rows)


def fit_expanding_forecasts(data, calendar, parameters, design):
    targets = [f + '_next' for f in u.FACTORS]
    columns = u.PREDICTORS + DUMMIES
    rows, audit = [], []
    state_hash_before = stable_hash(calendar[STATE])
    dates = u.monthly_index(u.TRAIN_END + pd.offsets.MonthEnd(), u.SIGNAL_END)
    assert len(dates) == 175
    for date in dates:
        current = calendar.loc[[date]]
        macro_train = calendar.loc[calendar.index < date].dropna(subset=u.PREDICTORS + targets)
        train = macro_train.dropna(subset=[STATE])
        history = data.loc[u.START + pd.offsets.MonthEnd():date, u.FACTORS]
        hmean = history.mean()
        ret_date = date + pd.offsets.MonthEnd()
        max_label = train.index.max() + pd.offsets.MonthEnd()
        assert max_label <= date and history.index.max() <= date
        missing = bool(current[u.PREDICTORS].isna().any(axis=None)) or pd.isna(current[STATE].iloc[0])
        expected_mean = train[columns].mean().to_numpy()
        expected_var = train[columns].var(ddof=0).to_numpy()
        max_scaler_mean_error = max_scaler_variance_error = 0.0
        label_hash = stable_hash(pd.Series((train.index + pd.offsets.MonthEnd()).astype(str), index=train.index))
        for factor in u.FACTORS:
            row = {'Design': design, 'Signal_Date': date, 'Return_Date': ret_date, 'Factor': factor,
                   'Actual': float(data.loc[ret_date, factor]), 'Historical_Mean': float(hmean[factor]),
                   MODEL + '_Fallback': missing, 'M1_ExpandingYoY_OLS_Eligible': not missing,
                   'H2_ExpandingYoY_OLS_Eligible': not missing,
                   'Nested_Mean_M1_ExpandingYoY': float(train[factor + '_next'].mean()),
                   'N_Train_ExpandingYoY': len(train), 'Max_ExpandingYoY_Training_Label_Date': max_label,
                   'Max_Historical_Return_Date': history.index.max(), STATE: current[STATE].iloc[0],
                   'ExpandingYoY_Training_Labels_Match_Macro': train.index.equals(macro_train.index),
                   'ExpandingYoY_Training_Label_Hash': label_hash}
            predictions = []
            for model, estimator in [('OLS', LinearRegression()), ('Ridge', Ridge(**parameters[factor]['Ridge'])),
                                     ('Elastic_Net', ElasticNet(**parameters[factor]['Elastic_Net'], max_iter=30000))]:
                if missing:
                    prediction = float(hmean[factor])
                else:
                    pipeline = Pipeline([('scale', StandardScaler()), ('model', estimator)])
                    pipeline.fit(train[columns], train[factor + '_next'])
                    scaler = pipeline.named_steps['scale']
                    max_scaler_mean_error = max(max_scaler_mean_error, float(np.max(np.abs(scaler.mean_ - expected_mean))))
                    max_scaler_variance_error = max(max_scaler_variance_error, float(np.max(np.abs(scaler.var_ - expected_var))))
                    assert np.allclose(scaler.mean_, expected_mean) and np.allclose(scaler.var_, expected_var)
                    prediction = float(pipeline.predict(current[columns])[0])
                row[MODEL + '_' + model] = prediction
                predictions.append(prediction)
            row[MODEL] = float(np.mean(predictions))
            # Explicit reduced OLS on exactly the augmented model's labels;
            # the allocation historical mean is a different benchmark.
            if missing:
                row['Macro_OLS_Matched_ExpandingYoY_Labels'] = float(hmean[factor])
            else:
                reduced = Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])
                reduced.fit(train[u.PREDICTORS], train[factor + '_next'])
                row['Macro_OLS_Matched_ExpandingYoY_Labels'] = float(reduced.predict(current[u.PREDICTORS])[0])
            rows.append(row)
        audit.append({'Design': design, 'Signal_Date': date, 'Return_Date': ret_date,
                      'N_Train_Macro_Complete': len(macro_train), 'N_Train_ExpandingYoY': len(train),
                      'Training_Labels_Match_Macro': train.index.equals(macro_train.index),
                      'Training_Signal_Start': train.index.min(), 'Training_Signal_End': train.index.max(),
                      'Max_Training_Label_Date': max_label, 'Max_Historical_Return_Date': history.index.max(),
                      'Training_Label_Hash': label_hash, 'Frozen_Training_State_Hash': stable_hash(train[STATE]),
                      'Fallback': missing, 'Scaler_Fit_Observations': 0 if missing else len(train),
                      'Scaler_Max_Mean_Error': max_scaler_mean_error,
                      'Scaler_Max_Variance_Error': max_scaler_variance_error,
                      **{'Train_State_' + r: int(train[STATE].eq(r).sum()) for r in u.REGIMES},
                      **{'Scaler_Mean_' + c: float(v) for c, v in zip(columns, expected_mean)},
                      **{'Scaler_Variance_' + c: float(v) for c, v in zip(columns, expected_var)}})
    assert stable_hash(calendar[STATE]) == state_hash_before
    forecasts = pd.DataFrame(rows)
    assert len(forecasts) == 175 * 6 and not forecasts.duplicated(KEYS).any()
    assert np.isfinite(forecasts[[MODEL, 'Actual', 'Historical_Mean']].to_numpy()).all()
    for component in ['OLS', 'Ridge', 'Elastic_Net']:
        assert np.array_equal(forecasts.loc[forecasts[MODEL + '_Fallback'], MODEL + '_' + component],
                              forecasts.loc[forecasts[MODEL + '_Fallback'], 'Historical_Mean'])
    assert np.allclose(forecasts[MODEL], forecasts[[MODEL + '_' + m for m in ['OLS', 'Ridge', 'Elastic_Net']]].mean(axis=1))
    return forecasts, pd.DataFrame(audit)




def run_analysis(base_path=None, selected_designs=None):
    base = Path(base_path).resolve() if base_path else BASE
    out = base / 'results/states'; out.mkdir(parents=True, exist_ok=True)
    designs = selected_designs or prior.DESIGNS
    data, manifest = u.load_data(base)
    frames, audits, calendars, thresholds, tunings, checks, causal_checks = [], [], [], [], [], [], []
    all_diagnostics = {}
    for design in designs:
        parameters, tuning = prior.frozen_parameters(base, design)
        original, old_cuts, _ = prior.yoy_features(base, data, design)
        x, threshold = expanding_state_calendar(original, design)
        causal_checks.append(causality_audit(original, x, design))
        fc, audit = fit_expanding_forecasts(data, x, parameters, design)
        frames.append(fc); audits.append(audit); thresholds.append(threshold)
        export = x.reset_index(); export['Design'] = design; calendars.append(export)
        tuning['Design'] = design; tunings.append(tuning)
        for key, frame in state_diagnostics(x, design).items(): all_diagnostics.setdefault(key, []).append(frame)
        fixed_states = u.regimes_from(x.IP_YoY_lag, x.CPI_YoY_lag,
                                     old_cuts[0]['Training_Median_Percent'], old_cuts[1]['Training_Median_Percent'])
        assert fixed_states.equals(x.Regime_YoY)
        assert fc.ExpandingYoY_Training_Labels_Match_Macro.all()
        checks.append({'Design': design, 'Initial_Seed_States_Identical': True,
                       'Independent_Sequential_Threshold_Check': True, 'No_State_Relabeling_During_Fits': True,
                       'Forecast_Origins': int(fc.Signal_Date.nunique()), 'Factor_Forecast_Rows': len(fc),
                       'All_Training_Labels_Match_Macro': True, 'All_Max_Label_Dates_Not_After_Signal': True,
                       'Current_Observation_Excluded_From_Expanding_Threshold': True,
                       'Fallback_Origins': int(fc.loc[fc[MODEL + '_Fallback'], 'Signal_Date'].nunique())})
        print(design + ': 175 expanding-state forecast origins and state audit complete', flush=True)
    forecasts = pd.concat(frames, ignore_index=True)
    calendar = pd.concat(calendars, ignore_index=True)
    exports = {'expanding_forecasts': forecasts, 'training_audit': pd.concat(audits, ignore_index=True),
               'model_calendar': calendar, 'threshold_audit': pd.concat(thresholds, ignore_index=True),
               'frozen_tuning': pd.concat(tunings, ignore_index=True), 'input_manifest': manifest,
               'causality_audit': pd.concat(causal_checks, ignore_index=True)}
    old_forecasts = pd.read_csv(base / 'reference_results/us/forecasts.csv', parse_dates=['Signal_Date', 'Return_Date'])
    old_forecasts = old_forecasts[old_forecasts.Return_Date >= '2019-04-30']
    old_forecasts = old_forecasts[KEYS + ['Historical_Mean', 'Actual', 'Macro_OLS']]
    matched = forecasts.merge(old_forecasts, on=KEYS, suffixes=('_new', '_old'), validate='one_to_one')
    reconciliation = []
    for design, frame in matched.groupby('Design'):
        assert len(frame) == 88 * 6
        for new_col, old_col in [('Historical_Mean_new', 'Historical_Mean_old'), ('Actual_new', 'Actual_old'),
                                 ('Macro_OLS_Matched_ExpandingYoY_Labels', 'Macro_OLS')]:
            difference = float((frame[new_col] - frame[old_col]).abs().max())
            assert np.allclose(frame[new_col], frame[old_col], atol=1e-11, rtol=1e-11)
            reconciliation.append({'Design': design, 'New_Column': new_col, 'Old_Column': old_col,
                                   'N_Matched_Original_Rows': len(frame), 'Maximum_Absolute_Difference': difference,
                                   'Units': 'monthly percentage points', 'Passed': True})
    exports['original_88_reconciliation'] = pd.DataFrame(reconciliation)
    exports.update({key: pd.concat(value, ignore_index=True) for key, value in all_diagnostics.items()})
    for name, frame in exports.items(): frame.to_csv(out / (name + '.csv'), index=False)
    result = {'scope': 'Retrospective H2 sensitivity after inspecting original evaluation; not a new untouched test',
              'forecast_model': MODEL, 'forecast_signal_start': '2011-12-31', 'forecast_signal_end': '2026-06-30',
              'forecast_return_start': '2012-01-31', 'forecast_return_end': '2026-07-31',
              'state_rule': 'High strictly exceeds threshold; equality low; no imputation',
              'units': 'All factor returns, forecasts, state growth rates and inflation rates are monthly-data percentage points; medians of YoY variables retain YoY percentage units',
              'initial_seed': '261 initial states and medians identical to original fixed rule',
              'post_seed_threshold': 'Each variable independently uses its previous observed signal values only; current signal enters future thresholds',
              'history_rule': 'Post-seed state labels are generated once, retained in later training, and never relabeled using a later threshold',
              'model_rule': 'Same seven continuous features; OLS/Ridge/ElasticNet equal mean; original per-design Macro penalties; training-only scalers',
              'data_scope': 'Archived revised-history inputs; separate IP-only vintage sensitivity; no full real-time claim',
              'limitations': 'Reduced switching can reflect overlapping 12-month changes and threshold/distribution shifts; persistence alone is not interpretation or forecasting validation',
              'checks': checks, 'forecast_rows': len(forecasts)}
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(out), 'checks': checks}, ensure_ascii=False), flush=True)
    return result


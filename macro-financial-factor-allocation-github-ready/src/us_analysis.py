"""US forecast and allocation analysis: longer pseudo-OOS, exposure/timing and model evidence.

Writes results/us only. The primary information model is Macro;
fixed/expanding YoY states are H2 diagnostics. No inspected sample is relabelled
an untouched test. Initial penalty selection uses the original 261 rows only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from . import us_core as u
from . import us_design as prev
from .us_design import mean_hac

BASE = Path(__file__).resolve().parents[1]
DESIGNS = prev.DESIGNS
COHORTS = {'Original_88': (pd.Timestamp('2019-04-30'), pd.Timestamp('2026-07-31')),
           'Extended_175': (pd.Timestamp('2012-01-31'), pd.Timestamp('2026-07-31'))}
ALLOCATIONS = ['LS5_Variance36_Cap40_Shrink50', 'Raw6_Sensitivity']
STATE_MODELS = ['Macro_M1_YoY', 'Macro_M1_ExpandingYoY']
EVENTS = pd.to_datetime(['2020-03-31', '2020-04-30', '2020-11-30'])
SEED = 20261009


def training_hash(index):
    return hashlib.sha256('|'.join(pd.DatetimeIndex(index).strftime('%Y-%m-%d')).encode()).hexdigest()


def fit_forecasts(base):
    data, _ = u.load_data(base)
    forecasts, model_stats, contributions, timing = [], [], [], []
    for design in DESIGNS:
        calendar, _, _ = prev.yoy_features(base, data, design)
        parameters, _ = prev.frozen_parameters(base, design)
        target_cols = [f + '_next' for f in u.FACTORS]
        dates = pd.date_range(u.TRAIN_END + pd.offsets.MonthEnd(), u.SIGNAL_END, freq='ME')
        for date in dates:
            target_date = date + pd.offsets.MonthEnd()
            current = calendar.loc[[date]]
            macro_train = calendar.loc[calendar.index < date].dropna(subset=u.PREDICTORS + target_cols)
            state_train = macro_train.dropna(subset=['Regime_YoY'])
            history = data.loc[u.START + pd.offsets.MonthEnd():date, u.FACTORS]
            assert macro_train.index.max() + pd.offsets.MonthEnd() <= date
            timing.append({'Design': design, 'Signal_Date': date, 'Return_Date': target_date,
                'Macro_N_Train': len(macro_train), 'YoY_N_Train': len(state_train),
                'Macro_Training_Hash': training_hash(macro_train.index),
                'YoY_Training_Hash': training_hash(state_train.index),
                'Training_Labels_Identical': macro_train.index.equals(state_train.index),
                'Max_Macro_Training_Label_Date': macro_train.index.max() + pd.offsets.MonthEnd(),
                'Max_YoY_Training_Label_Date': state_train.index.max() + pd.offsets.MonthEnd(),
                'Historical_Return_End': history.index.max()})
            for f in u.FACTORS:
                row = {'Design': design, 'Signal_Date': date, 'Return_Date': target_date, 'Factor': f,
                    'Actual': float(data.loc[target_date, f]), 'Historical_Mean': float(history[f].mean()),
                    'Segment': 'Validation_87' if date <= u.VALID_END else 'Original_88',
                    'N_Train': len(macro_train)}
                for spec, train, cols in [('Macro', macro_train, u.PREDICTORS),
                         ('Macro_M1_YoY', state_train, u.PREDICTORS + ['State_YoY_' + r for r in u.REGIMES[1:]])]:
                    missing = bool(current[u.PREDICTORS].isna().any(axis=None)) or (spec != 'Macro' and pd.isna(current.Regime_YoY.iloc[0]))
                    row[spec + '_Fallback'] = missing
                    row['Nested_Mean_OLS' if spec == 'Macro' else 'Nested_Mean_M1_YoY'] = train[f + '_next'].mean()
                    row[spec + '_N_Train'] = len(train)
                    preds = []
                    for model, est in [('OLS', LinearRegression()), ('Ridge', Ridge(**parameters[f]['Ridge'])),
                                      ('Elastic_Net', ElasticNet(**parameters[f]['Elastic_Net'], max_iter=30000))]:
                        pipe = Pipeline([('scale', StandardScaler()), ('model', est)])
                        # Coefficient norms remain observable on fallback dates; current X is never imputed.
                        pipe.fit(train[cols], train[f + '_next'])
                        estimator, scaler = pipe.named_steps['model'], pipe.named_steps['scale']
                        coef = np.asarray(estimator.coef_)
                        intercept = float(estimator.intercept_)
                        alpha = np.nan if model == 'OLS' else parameters[f][model]['alpha']
                        max_alpha = np.nan if model == 'OLS' else (1e6 if model == 'Ridge' else 10.)
                        prediction = row['Historical_Mean'] if missing else float(pipe.predict(current[cols])[0])
                        row[spec + '_' + model] = prediction
                        preds.append(prediction)
                        model_stats.append({'Design': design, 'Signal_Date': date, 'Return_Date': target_date,
                            'Factor': f, 'Specification': spec, 'Model': model, 'N_Train': len(train),
                            'Intercept_pct': intercept, 'Coefficient_L2_Standardized': np.linalg.norm(coef),
                            'Coefficient_L1_Standardized': np.abs(coef).sum(),
                            'Nonzero_Coefficients_1e_10': int((np.abs(coef) > 1e-10).sum()),
                            'Frozen_Alpha': alpha, 'Alpha_At_Upper_Grid': bool(model != 'OLS' and np.isclose(alpha, max_alpha)),
                            'Fallback': missing, 'Prediction_pct': prediction,
                            'Matched_Training_Mean_pct': train[f + '_next'].mean()})
                        if not missing:
                            z = scaler.transform(current[cols])[0]
                            values = coef * z
                            assert np.isclose(intercept + values.sum(), prediction, atol=1e-10)
                            # Exact predictor decomposition is retained for all months, not just selected events.
                            common = {'Design': design, 'Signal_Date': date, 'Return_Date': target_date,
                                      'Factor': f, 'Specification': spec, 'Model': model}
                            contributions.append(common | {'Component': 'Intercept', 'Raw_Feature': np.nan,
                                'Training_Feature_Mean': np.nan, 'Training_Feature_SD': np.nan,
                                'Standardized_Feature': np.nan, 'Coefficient_Standardized': np.nan,
                                'Prediction_Contribution_pct': intercept})
                            for j, name in enumerate(cols):
                                contributions.append(common | {'Component': name, 'Raw_Feature': current[name].iloc[0],
                                    'Training_Feature_Mean': scaler.mean_[j], 'Training_Feature_SD': scaler.scale_[j],
                                    'Standardized_Feature': z[j], 'Coefficient_Standardized': coef[j],
                                    'Prediction_Contribution_pct': values[j]})
                    row[spec] = np.mean(preds)
                eligible = not row['Macro_M1_YoY_Fallback']
                row['H2_YoY_OLS_Eligible'] = eligible
                if eligible:
                    reduced = Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])
                    reduced.fit(state_train[u.PREDICTORS], state_train[f + '_next'])
                    row['Macro_OLS_Matched_YoY_Labels'] = float(reduced.predict(current[u.PREDICTORS])[0])
                else:
                    row['Macro_OLS_Matched_YoY_Labels'] = np.nan
                forecasts.append(row)
        print(design + ': 175-month component forecasts complete', flush=True)
    return data, pd.DataFrame(forecasts), pd.DataFrame(model_stats), pd.DataFrame(contributions), pd.DataFrame(timing)


def add_expanding_forecasts(base, fc):
    folder = base / 'results/states'
    candidates = [folder / 'forecasts.csv', folder / 'expanding_forecasts.csv']
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return fc, False
    other = pd.read_csv(path, parse_dates=['Signal_Date', 'Return_Date'])
    keys = ['Design', 'Signal_Date', 'Return_Date', 'Factor']
    wanted = [c for c in other if c not in fc.columns or c in keys]
    assert not other.duplicated(keys).any()
    joined = fc.merge(other[wanted], on=keys, how='left', validate='one_to_one')
    if 'Macro_M1_ExpandingYoY' not in joined or joined.Macro_M1_ExpandingYoY.isna().any():
        raise ValueError('Expanding-state file exists but does not cover all 175 monthly forecasts in both designs')
    return joined, True


def allocation_outputs(data, fc):
    states = [s for s in STATE_MODELS if s in fc]
    strategies = ['Equal_Weight', 'Inverse_Volatility', 'Historical_Mean', 'Macro'] + states
    weight_rows, return_rows, metrics, risks = [], [], [], []
    for design, ds in fc.groupby('Design'):
        actual = ds.pivot(index='Return_Date', columns='Factor', values='Actual')[u.FACTORS]
        for date, group in ds.groupby('Signal_Date'):
            g = group.set_index('Factor').loc[u.FACTORS]
            history = data.loc[:date, u.FACTORS].tail(36) / 100
            assert len(history) == 36 and history.notna().all(axis=None)
            variance = history.var(ddof=1).to_numpy()
            risks.append({'Design': design, 'Signal_Date': date, 'History_Start': history.index.min(),
                          'History_End': history.index.max(), 'N': 36})
            for allocation in ALLOCATIONS:
                selected = np.arange(1, 6) if allocation.startswith('LS5') else np.arange(6)
                for strategy in strategies:
                    if strategy == 'Equal_Weight':
                        w = np.ones(len(selected)) / len(selected)
                    elif strategy == 'Inverse_Volatility':
                        w = 1 / np.sqrt(variance[selected]); w /= w.sum()
                    else:
                        scores = g[strategy].to_numpy()[selected] / 100
                        if allocation.startswith('LS5'):
                            raw = u.scores_to_weights(scores / variance[selected])[0]
                            w = u.scores_to_weights(.5 * raw + .5 / 5, cap=.4)[0]
                        else:
                            w = u.scores_to_weights(scores)[0]
                    full = np.zeros(6); full[selected] = w
                    weight_rows.append({'Design': design, 'Allocation': allocation, 'Strategy': strategy,
                        'Signal_Date': date, 'Return_Date': date + pd.offsets.MonthEnd(), **dict(zip(u.FACTORS, full))})
        design_weights = pd.DataFrame(weight_rows)
        design_weights = design_weights[design_weights.Design == design]
        for cohort, (lo, hi) in COHORTS.items():
            for (allocation, strategy), w in design_weights[design_weights.Return_Date.between(lo, hi)].groupby(['Allocation', 'Strategy']):
                w = w.sort_values('Return_Date')
                matrix = w[u.FACTORS].to_numpy()
                gross = (matrix * actual.loc[w.Return_Date].to_numpy()).sum(axis=1)
                turnover = np.r_[0., .5 * np.abs(np.diff(matrix, axis=0)).sum(axis=1)]
                for cost in [0, 10, 25, 50]:
                    frame = pd.DataFrame({'Design': design, 'Cohort': cohort, 'Allocation': allocation,
                        'Strategy': strategy, 'Cost_bps': cost, 'Return_Date': w.Return_Date.to_numpy(),
                        'Gross_Return_pct': gross, 'Turnover': turnover,
                        'Allocation_Cost_pct': turnover * cost / 100, 'Net_Return_pct': gross - turnover * cost / 100})
                    return_rows.append(frame)
                    for sample, z in samples(frame):
                        metrics.append({'Design': design, 'Cohort': cohort, 'Allocation': allocation, 'Strategy': strategy,
                            'Cost_bps': cost, 'Sample': sample, **u.performance(z.Net_Return_pct)})
    return pd.DataFrame(weight_rows), pd.concat(return_rows, ignore_index=True), pd.DataFrame(metrics), pd.DataFrame(risks)


def samples(frame):
    return [('Full', frame), ('Exclude_2020_Mar_Dec', frame.loc[~frame.Return_Date.between('2020-03-01', '2020-12-31')])]


def exposure_timing(fc, weights, returns):
    paths, averages, summary, differences, events = [], [], [], [], []
    for key, frame in returns[returns.Cost_bps == 10].groupby(['Design', 'Cohort', 'Allocation', 'Strategy']):
        design, cohort, allocation, strategy = key
        w = weights[(weights.Design == design) & (weights.Allocation == allocation) & (weights.Strategy == strategy)].set_index('Return_Date').loc[frame.Return_Date, u.FACTORS]
        actual = fc[fc.Design == design].pivot(index='Return_Date', columns='Factor', values='Actual')[u.FACTORS].loc[frame.Return_Date]
        avg = w.mean().to_numpy()
        static = actual.to_numpy() @ avg
        timing = ((w.to_numpy() - avg) * actual.to_numpy()).sum(axis=1)
        assert np.allclose(static + timing, frame.Gross_Return_pct)
        assert np.allclose(static + timing - frame.Allocation_Cost_pct, frame.Net_Return_pct)
        rec = frame.copy(); rec['Static_Exposure_Return_pct'] = static; rec['Dynamic_Timing_Return_pct'] = timing
        paths.append(rec)
        common = dict(zip(['Design', 'Cohort', 'Allocation', 'Strategy'], key))
        averages.append(common | dict(zip(u.FACTORS, avg)) | {'Weight_Observations': len(w), 'Basis': 'Retrospective complete cohort; includes future weights'})
        for sample, z in samples(rec):
            summary.append(common | {'Sample': sample, 'N': len(z), 'Annual_Gross_Dynamic_pp': 12*z.Gross_Return_pct.mean(),
                'Annual_Static_Exposure_pp': 12*z.Static_Exposure_Return_pct.mean(),
                'Annual_Dynamic_Timing_pp': 12*z.Dynamic_Timing_Return_pct.mean(),
                'Annual_Allocation_Cost_pp': 12*z.Allocation_Cost_pct.mean(), 'Annual_Net_pp': 12*z.Net_Return_pct.mean(),
                'Average_Weight_Basis': 'Full cohort, unchanged when COVID months are removed'})
    paths = pd.concat(paths, ignore_index=True)
    for key, ds in paths.groupby(['Design', 'Cohort', 'Allocation']):
        design, cohort, allocation = key
        comparisons = [('Macro', 'Historical_Mean')] + [(s, 'Macro') for s in STATE_MODELS if s in ds.Strategy.unique()]
        for first, second in comparisons:
            a = ds[ds.Strategy == first].set_index('Return_Date').sort_index()
            b = ds[ds.Strategy == second].set_index('Return_Date').sort_index()
            cols = ['Gross_Return_pct', 'Static_Exposure_Return_pct', 'Dynamic_Timing_Return_pct', 'Allocation_Cost_pct', 'Net_Return_pct']
            delta = a[cols] - b[cols]; delta['Return_Date'] = delta.index
            for sample, z in samples(delta):
                differences.append(dict(zip(['Design', 'Cohort', 'Allocation'], key)) | {'Comparison': first+'_minus_'+second,
                    'Sample': sample, 'N': len(z), **{'Annual_'+col.replace('_pct', '_pp'): 12*z[col].mean() for col in cols}})
            wa = weights[(weights.Design == design)&(weights.Allocation == allocation)&(weights.Strategy == first)].set_index('Return_Date')
            wb = weights[(weights.Design == design)&(weights.Allocation == allocation)&(weights.Strategy == second)].set_index('Return_Date')
            actual = fc[fc.Design == design].pivot(index='Return_Date',columns='Factor',values='Actual')
            for date in EVENTS:
                for factor in u.FACTORS:
                    events.append(dict(zip(['Design','Cohort','Allocation'],key)) | {'Comparison': first+'_minus_'+second,
                        'Return_Date': date, 'Signal_Date': date-pd.offsets.MonthEnd(), 'Factor': factor,
                        'First_Weight': wa.loc[date,factor], 'Second_Weight': wb.loc[date,factor],
                        'Weight_Difference': wa.loc[date,factor]-wb.loc[date,factor], 'Actual_Factor_Return_pct': actual.loc[date,factor],
                        'Gross_PnL_Contribution_pp': (wa.loc[date,factor]-wb.loc[date,factor])*actual.loc[date,factor],
                        'Portfolio_Cost_Difference_pp': a.loc[date,'Allocation_Cost_pct']-b.loc[date,'Allocation_Cost_pct'],
                        'Portfolio_Net_Difference_pp': a.loc[date,'Net_Return_pct']-b.loc[date,'Net_Return_pct']})
    return paths, pd.DataFrame(averages), pd.DataFrame(summary), pd.DataFrame(differences), pd.DataFrame(events)


def spanning_hac(y, x, dates, lag):
    """OLS spanning alpha/beta with natural-calendar Bartlett score covariance."""
    matrix = np.column_stack([np.ones(len(x)), x])
    inv = np.linalg.inv(matrix.T @ matrix)
    beta = inv @ matrix.T @ y
    residual = y - matrix @ beta
    scores = matrix * residual[:, None]
    meat = scores.T @ scores
    months = pd.DatetimeIndex(dates).to_period('M').asi8
    lookup = {month:i for i,month in enumerate(months)}
    for k in range(1,lag+1):
        pairs = [(i,lookup[month-k]) for i,month in enumerate(months) if month-k in lookup]
        if pairs:
            a,b = np.asarray(pairs).T
            cross = scores[a].T @ scores[b]
            meat += (1-k/(lag+1))*(cross+cross.T)
    cov = inv @ meat @ inv
    se = np.sqrt(np.maximum(np.diag(cov),0))
    return {'N':len(y),'HAC_Lags':lag,'Alpha_Monthly_pp':beta[0],'Alpha_Annual_pp':12*beta[0],
        'Beta':beta[1],'Alpha_HAC_SE_Monthly_pp':se[0],'Alpha_HAC_SE_Annual_pp':12*se[0],
        'Beta_HAC_SE':se[1],'Alpha_p_Two_Sided':2*norm.sf(abs(beta[0]/se[0])),
        'Alpha_Annual_CI_Low_pp':12*(beta[0]-norm.ppf(.975)*se[0]),
        'Alpha_Annual_CI_High_pp':12*(beta[0]+norm.ppf(.975)*se[0]),
        'Interpretation':'Conditional spanning intercept; not isolated causal timing or guaranteed investable alpha'}


def comparative_inference(returns, draws):
    intervals, spanning, mde, means = [], [], [], []
    for key, ds in returns[returns.Cost_bps == 10].groupby(['Design','Cohort','Allocation']):
        common = dict(zip(['Design','Cohort','Allocation'],key))
        for sample, sub in samples(ds):
            panel = sub.pivot(index='Return_Date',columns='Strategy',values='Net_Return_pct').sort_index()
            pairs = [('Macro','Historical_Mean')] + [(s,'Macro') for s in STATE_MODELS if s in panel]
            for a,b in pairs:
                tag = common | {'Sample':sample,'Comparison':a+'_minus_'+b,'Cost_bps':10}
                seed = SEED + (100 if sample!='Full' else 0) + (1000 if key[1]=='Extended_175' else 0)
                result = prev.studentized_sharpe_interval(panel[a]/100,panel[b]/100,panel.index,B=draws,block=6,hac=3,seed=seed)
                intervals.append(tag | result)
                values = panel[[a,b]].to_numpy()/100
                delta,score = prev.sharpe_if(values)
                monthly_sr = values.mean(axis=0)/values.std(axis=0,ddof=1)
                rho = np.corrcoef(values.T)[0,1]
                iid_variance = (2*(1-rho)+.5*(monthly_sr[0]**2+monthly_sr[1]**2-2*rho**2*np.prod(monthly_sr)))/len(values)
                iid_se = np.sqrt(12*iid_variance)
                for lag in [3,6,12]:
                    se = float(prev.calendar_hac_se(score,panel.index,lag))
                    critical = norm.ppf(.975)+norm.ppf(.8)
                    mde.append(tag | {'N':len(values),'HAC_Lags':lag,'Observed_Annual_Sharpe_Difference':delta,
                        'Actual_Paired_IF_HAC_SE_Annual':se,'Approx_MDE_Annual_Sharpe_80pct':critical*se,
                        'IID_JK_Memmel_SE_Annual_Secondary':iid_se,'IID_MDE_Annual_Sharpe_80pct_Secondary':critical*iid_se,
                        'Return_Correlation':rho,'Two_Sided_Alpha':.05,'Target_Power':.8,'Normal_Critical_Sum':critical,
                        'IID_Years_For_Delta0_1_AssumptionsOnly':len(values)/12*(critical*iid_se/.1)**2,
                        'Caution':'Approximate normal local-alternative MDE; conditional IID years illustration is not a calendar-time forecast'})
                    spanning.append(tag | spanning_hac(panel[a].to_numpy(),panel[b].to_numpy(),panel.index,lag))
                    means.append(tag | mean_hac(panel[a]-panel[b],lag,panel.index))
    means = pd.DataFrame(means)
    for _,ix in means.groupby(['Design','Cohort','Allocation','Sample','HAC_Lags']).groups.items():
        means.loc[ix,'Holm_p_Within_Displayed_Comparison_Family'] = multipletests(means.loc[ix,'p_Two_Sided'],method='holm')[1]
    return pd.DataFrame(intervals),pd.DataFrame(spanning),pd.DataFrame(mde),means


def direct_h2_tests(fc):
    rows=[]
    specifications=[('Macro_M1_YoY','Macro_OLS_Matched_YoY_Labels','H2_YoY_OLS_Eligible'),
                    ('Macro_M1_ExpandingYoY','Macro_OLS_Matched_ExpandingYoY_Labels','H2_ExpandingYoY_OLS_Eligible')]
    for design,design_fc in fc.groupby('Design'):
        for cohort,(lo,hi) in COHORTS.items():
            sample=design_fc[design_fc.Return_Date.between(lo,hi)]
            for state,matched,eligible in specifications:
                if state not in sample:continue
                if matched not in sample or eligible not in sample:raise ValueError('Missing direct H2 matched-OLS columns: '+state)
                for factor,g in sample.groupby('Factor'):
                    g=g.sort_values('Return_Date')
                    for model in ['OLS','Ensemble']:
                        if model=='OLS':
                            z=g[g[eligible].fillna(False).astype(bool)].dropna(subset=[matched,state+'_OLS'])
                            smaller=z[matched];larger=z[state+'_OLS']
                            validity='Nested OLS state versus reduced Macro on exactly identical training labels; observed signals only'
                        else:
                            z=g;smaller=z.Macro;larger=z[state]
                            validity='Approximate ensemble state-versus-Macro CW; shrinkage invalidates the standard nested OLS derivation'
                        raw=(z.Actual-smaller)**2-(z.Actual-larger)**2
                        adjusted=raw+(smaller-larger)**2
                        for lag in [3,6,12]:
                            for method,loss in [('DM_Loss_HAC',raw),('Clark_West',adjusted)]:
                                rows.append({'Design':design,'Cohort':cohort,'State_Model':state,'Factor':factor,
                                    'Estimator':model,'Method':method,'HAC_Lags':lag,'Validity':validity,
                                    'Comparison':'State versus Macro directly','Units':'squared percentage points',
                                    'Benchmark_MSE':((z.Actual-smaller)**2).mean(),
                                    'Model_MSE':((z.Actual-larger)**2).mean(),
                                    'OOS_R2_vs_Direct_Benchmark':1-((z.Actual-larger)**2).sum()/((z.Actual-smaller)**2).sum(),
                                    **mean_hac(loss,lag,z.Return_Date)})
    out=pd.DataFrame(rows)
    for key,ix in out.groupby(['Design','Cohort','State_Model','Estimator','Method','HAC_Lags']).groups.items():
        p='p_One_Sided_Positive' if key[-2]=='Clark_West' else 'p_Two_Sided'
        assert len(ix)==6
        out.loc[ix,'Holm_p_Six_Factors']=multipletests(out.loc[ix,p],method='holm')[1]
    return out


def direct_h1_tests(fc):
    rows=[]
    for design,design_fc in fc.groupby('Design'):
        for cohort,(lo,hi) in COHORTS.items():
            sample=design_fc[design_fc.Return_Date.between(lo,hi)]
            for factor,g in sample.groupby('Factor'):
                g=g.sort_values('Return_Date')
                for estimator in ['OLS','Ensemble']:
                    if estimator=='OLS':
                        z=g[~g.Macro_Fallback.astype(bool)].dropna(subset=['Macro_OLS','Nested_Mean_OLS'])
                        smaller=z.Nested_Mean_OLS;larger=z.Macro_OLS
                        validity='Macro OLS versus nested intercept with identical eligible training labels; no fallback observations'
                    else:
                        z=g;smaller=z.Historical_Mean;larger=z.Macro
                        validity='Approximate ensemble CW versus fuller-history historical mean; not the standard OLS nested case'
                    raw=(z.Actual-smaller)**2-(z.Actual-larger)**2
                    cw=raw+(smaller-larger)**2
                    for lag in [3,6,12]:
                        for method,loss in [('DM_Loss_HAC',raw),('Clark_West',cw)]:
                            rows.append({'Design':design,'Cohort':cohort,'Factor':factor,'Estimator':estimator,
                                'Method':method,'HAC_Lags':lag,'Validity':validity,'Comparison':'Macro versus historical mean',
                                'Units':'squared percentage points','Benchmark_MSE':((z.Actual-smaller)**2).mean(),
                                'Model_MSE':((z.Actual-larger)**2).mean(),
                                'OOS_R2_vs_Direct_Benchmark':1-((z.Actual-larger)**2).sum()/((z.Actual-smaller)**2).sum(),
                                **mean_hac(loss,lag,z.Return_Date)})
    out=pd.DataFrame(rows)
    for key,ix in out.groupby(['Design','Cohort','Estimator','Method','HAC_Lags']).groups.items():
        p='p_One_Sided_Positive' if key[-2]=='Clark_West' else 'p_Two_Sided'
        assert len(ix)==6
        out.loc[ix,'Holm_p_Six_Factors']=multipletests(out.loc[ix,p],method='holm')[1]
    return out


def refresh_forecast_tests(base_path=None):
    """Refresh H1/H2 loss evidence from saved predictions, without refitting/bootstrap."""
    base=Path(base_path).resolve() if base_path else BASE
    out=base/'results/us'
    fc=pd.read_csv(out/'forecasts.csv',parse_dates=['Signal_Date','Return_Date'])
    for name,frame in [('direct_h1_forecast_loss',direct_h1_tests(fc)),('direct_h2_forecast_loss',direct_h2_tests(fc))]:
        frame.to_csv(out/(name+'.csv'),index=False)


def event_summary_tables(events, contributions):
    """Avoid double-counting portfolio totals repeated on the six factor rows."""
    keys=['Design','Cohort','Allocation','Comparison','Return_Date','Signal_Date']
    grouped=events.groupby(keys,dropna=False)
    for col in ['Portfolio_Cost_Difference_pp','Portfolio_Net_Difference_pp']:
        assert grouped[col].nunique().eq(1).all()
    totals=grouped.agg(Gross_PnL_Difference_pp=('Gross_PnL_Contribution_pp','sum'),
        Allocation_Cost_Difference_pp=('Portfolio_Cost_Difference_pp','first'),
        Net_PnL_Difference_pp=('Portfolio_Net_Difference_pp','first')).reset_index()
    assert np.allclose(totals.Gross_PnL_Difference_pp-totals.Allocation_Cost_Difference_pp,
                       totals.Net_PnL_Difference_pp)
    keys=['Design','Signal_Date','Return_Date','Factor','Specification','Component']
    selected=contributions[contributions.Return_Date.isin(EVENTS)]
    grouped=selected.groupby(keys,dropna=False)
    assert grouped.Model.nunique().eq(3).all()
    ensemble=grouped.Prediction_Contribution_pct.mean().reset_index()
    ensemble['Attribution']='Prediction only; not a causal feature attribution of nonlinear portfolio PnL'
    return totals,ensemble


def refresh_event_summaries(base_path=None):
    """Export reporting aggregates using saved paths; no fitting or simulation."""
    base=Path(base_path).resolve() if base_path else BASE
    out=base/'results/us'
    events=pd.read_csv(out/'event_factor_pnl.csv',parse_dates=['Signal_Date','Return_Date'])
    contributions=pd.read_csv(out/'event_predictor_contributions.csv',parse_dates=['Signal_Date','Return_Date'])
    totals,ensemble=event_summary_tables(events,contributions)
    totals.to_csv(out/'event_portfolio_totals.csv',index=False)
    ensemble.to_csv(out/'event_ensemble_predictor_contributions.csv',index=False)


def component_diagnostics(fc, model_stats):
    results=[]
    for (design,factor),ds in fc.groupby(['Design','Factor']):
        for segment,g in [('Validation_87',ds[ds.Segment=='Validation_87']),('Original_88',ds[ds.Segment=='Original_88']),('Extended_175',ds)]:
            for spec in ['Macro','Macro_M1_YoY']:
                matched=g['Nested_Mean_OLS' if spec=='Macro' else 'Nested_Mean_M1_YoY']
                for model in ['OLS','Ridge','Elastic_Net','Ensemble']:
                    name=spec if model=='Ensemble' else spec+'_'+model
                    prediction=g[name]
                    error=(g.Actual-prediction)**2
                    match_error=(g.Actual-matched)**2
                    stats=model_stats[(model_stats.Design==design)&(model_stats.Factor==factor)&
                        (model_stats.Specification==spec)&(model_stats.Model==model)&model_stats.Return_Date.isin(g.Return_Date)]
                    results.append({'Design':design,'Factor':factor,'Segment':segment,'Specification':spec,'Model':model,
                        'N':len(g),'RMSE_pct':np.sqrt(error.mean()),
                        'OOS_R2_vs_Historical_Mean':1-error.sum()/((g.Actual-g.Historical_Mean)**2).sum(),
                        'OOS_R2_vs_Matched_Mean':1-error.sum()/match_error.sum(),
                        'Prediction_SD_pct':prediction.std(),
                        'RMS_Deviation_From_Matched_Mean_pct':np.sqrt(((prediction-matched)**2).mean()),
                        'Mean_Absolute_Deviation_From_Matched_Mean_pct':(prediction-matched).abs().mean(),
                        'Median_Coefficient_L2_Standardized':stats.Coefficient_L2_Standardized.median() if len(stats) else np.nan,
                        'Frozen_Alpha':stats.Frozen_Alpha.iloc[0] if len(stats) else np.nan,
                        'Alpha_At_Upper_Grid':bool(stats.Alpha_At_Upper_Grid.iloc[0]) if len(stats) else False,
                        'Refit_Rule':'Expanding refit each signal; not the old once-fit validation diagnostic'})
    return pd.DataFrame(results)


def frozen_initial_validation(base, data):
    """Separate old-style 87-month diagnostic: fit initial261 once, never refit."""
    forecasts, rows = [], []
    for design in DESIGNS:
        calendar, _, _ = prev.yoy_features(base, data, design)
        parameters, old_tuning = prev.frozen_parameters(base, design)
        targets = [f+'_next' for f in u.FACTORS]
        initial = calendar.loc[:u.TRAIN_END].dropna(subset=u.PREDICTORS+targets)
        validation = calendar.loc[u.TRAIN_END+pd.offsets.MonthEnd():u.VALID_END]
        assert len(initial)==261 and len(validation)==87
        for spec in ['Macro','Macro_M1_YoY']:
            train=initial if spec=='Macro' else initial.dropna(subset=['Regime_YoY'])
            cols=u.PREDICTORS+([] if spec=='Macro' else ['State_YoY_'+r for r in u.REGIMES[1:]])
            assert validation[cols].notna().all(axis=None)
            for f in u.FACTORS:
                fixed_mean=train[f+'_next'].mean()
                actual=validation[f+'_next'].to_numpy()
                predictions=[]
                for model,est in [('OLS',LinearRegression()),('Ridge',Ridge(**parameters[f]['Ridge'])),
                                  ('Elastic_Net',ElasticNet(**parameters[f]['Elastic_Net'],max_iter=30000))]:
                    pipe=Pipeline([('scale',StandardScaler()),('model',est)])
                    pipe.fit(train[cols],train[f+'_next'])
                    prediction=pipe.predict(validation[cols]);predictions.append(prediction)
                    coef=pipe.named_steps['model'].coef_
                    alpha=np.nan if model=='OLS' else parameters[f][model]['alpha']
                    rmse=np.sqrt(np.mean((actual-prediction)**2))
                    if spec=='Macro' and model!='OLS':
                        previous=old_tuning[(old_tuning.Factor==f)&(old_tuning.Model==model)].iloc[0]
                        assert np.isclose(rmse,previous.Validation_RMSE,atol=1e-8)
                    rows.append({'Design':design,'Specification':spec,'Factor':f,'Model':model,'N_Train':len(train),'N_Validation':87,
                        'RMSE_pct':rmse,'R2_vs_Fixed_Initial_Training_Mean':1-np.sum((actual-prediction)**2)/np.sum((actual-fixed_mean)**2),
                        'RMS_Deviation_From_Fixed_Mean_pct':np.sqrt(np.mean((prediction-fixed_mean)**2)),
                        'Coefficient_L2_Standardized':np.linalg.norm(coef),'Frozen_Alpha':alpha,
                        'Alpha_At_Upper_Grid':bool(model!='OLS' and np.isclose(alpha,1e6 if model=='Ridge' else 10)),
                        'Evaluation_Rule':'Initial261 fit once;87 validation signals predicted with frozen coefficients and scaler; no selection feedback'})
                    for date,pred,act in zip(validation.index,prediction,actual):
                        forecasts.append({'Design':design,'Specification':spec,'Factor':f,'Model':model,
                            'Signal_Date':date,'Return_Date':date+pd.offsets.MonthEnd(), 'Actual_pct':act,
                            'Prediction_pct':pred,'Fixed_Initial_Training_Mean_pct':fixed_mean})
                ensemble=np.mean(predictions,axis=0)
                rows.append({'Design':design,'Specification':spec,'Factor':f,'Model':'Ensemble','N_Train':len(train),'N_Validation':87,
                    'RMSE_pct':np.sqrt(np.mean((actual-ensemble)**2)),
                    'R2_vs_Fixed_Initial_Training_Mean':1-np.sum((actual-ensemble)**2)/np.sum((actual-fixed_mean)**2),
                    'RMS_Deviation_From_Fixed_Mean_pct':np.sqrt(np.mean((ensemble-fixed_mean)**2)),
                    'Coefficient_L2_Standardized':np.nan,'Frozen_Alpha':np.nan,'Alpha_At_Upper_Grid':False,
                    'Evaluation_Rule':'Average of initial261-frozen OLS,Ridge,ElasticNet; no selection feedback'})
                for date,pred,act in zip(validation.index,ensemble,actual):
                    forecasts.append({'Design':design,'Specification':spec,'Factor':f,'Model':'Ensemble',
                        'Signal_Date':date,'Return_Date':date+pd.offsets.MonthEnd(),'Actual_pct':act,
                        'Prediction_pct':pred,'Fixed_Initial_Training_Mean_pct':fixed_mean})
    return pd.DataFrame(forecasts),pd.DataFrame(rows)


def verify(base,fc,weights,returns,paths,intervals,timing,contributions):
    checks=0
    for design,group in fc.groupby('Design'):
        assert group.Signal_Date.nunique()==175 and len(group)==1050;checks+=1
        prior=pd.read_csv(base/'reference_results/us/forecasts.csv',parse_dates=['Signal_Date','Return_Date'])
        prior=prior[prior.Design==design].set_index(['Signal_Date','Factor'])
        current=group.set_index(['Signal_Date','Factor']).loc[prior.index]
        for col in ['Actual','Historical_Mean','Macro','Macro_OLS','Macro_Ridge','Macro_Elastic_Net','Macro_M1_YoY','Macro_M1_YoY_OLS']:
            assert np.allclose(current[col],prior[col],atol=1e-9);checks+=1
    assert (timing.Max_Macro_Training_Label_Date<=timing.Signal_Date).all()
    assert (timing.Max_YoY_Training_Label_Date<=timing.Signal_Date).all();checks+=2
    for key,g in weights.groupby(['Design','Allocation','Strategy']):
        assert len(g)==175 and np.allclose(g[u.FACTORS].sum(axis=1),1)
        assert (g[u.FACTORS]>=0).all(axis=None)
        if key[1].startswith('LS5'):
            assert (g.MKT_RF==0).all()
            if key[2] not in ['Equal_Weight','Inverse_Volatility']:assert g[u.FACTORS[1:]].max(axis=None)<=.4+1e-10
        checks+=1
    for key,g in returns.groupby(['Design','Cohort','Allocation','Strategy','Cost_bps']):
        assert len(g)==(88 if key[1]=='Original_88' else 175)
        assert g.iloc[0].Turnover==0
        assert np.allclose(g.Net_Return_pct,g.Gross_Return_pct-g.Allocation_Cost_pct);checks+=3
    assert np.allclose(paths.Gross_Return_pct,paths.Static_Exposure_Return_pct+paths.Dynamic_Timing_Return_pct)
    assert np.allclose(paths.Net_Return_pct,paths.Static_Exposure_Return_pct+paths.Dynamic_Timing_Return_pct-paths.Allocation_Cost_pct);checks+=2
    summed=contributions.groupby(['Design','Signal_Date','Factor','Specification','Model']).Prediction_Contribution_pct.sum()
    lookup=fc.set_index(['Design','Signal_Date','Factor'])
    for key,value in summed.items():
        assert np.isclose(value,lookup.loc[key[:3],key[3]+'_'+key[4]],atol=1e-9);checks+=1
    assert (intervals.Finite_Draws==intervals.Requested_Draws).all();checks+=1
    return checks




def run_analysis(base_path=None,bootstrap_n=3000,reuse_forecasts=False):
    base=Path(base_path).resolve() if base_path else BASE
    out=base/'results/us';out.mkdir(parents=True,exist_ok=True)
    if reuse_forecasts:
        data,_=u.load_data(base)
        fc=pd.read_csv(out/'fixed_forecasts.csv',parse_dates=['Signal_Date','Return_Date'])
        stats=pd.read_csv(out/'model_component_fit_diagnostics.csv',parse_dates=['Signal_Date','Return_Date'])
        contributions=pd.read_csv(out/'predictor_contributions.csv',parse_dates=['Signal_Date','Return_Date'])
        timing=pd.read_csv(out/'training_timing_audit.csv',parse_dates=['Signal_Date','Return_Date','Max_Macro_Training_Label_Date','Max_YoY_Training_Label_Date'])
    else:
        data,fc,stats,contributions,timing=fit_forecasts(base)
        for name,frame in [('fixed_forecasts',fc),('model_component_fit_diagnostics',stats),
                           ('predictor_contributions',contributions),('training_timing_audit',timing)]:frame.to_csv(out/(name+'.csv'),index=False)
    fc,has_expanding=add_expanding_forecasts(base,fc)
    fc.to_csv(out/'forecasts.csv',index=False)
    print('Forecasts saved; expanding-state joined: '+str(has_expanding),flush=True)
    weights,returns,metrics,risks=allocation_outputs(data,fc)
    paths,averages,decomp,differences,events=exposure_timing(fc,weights,returns)
    intervals,spanning,mde,means=comparative_inference(returns,bootstrap_n)
    h2=direct_h2_tests(fc)
    h1=direct_h1_tests(fc)
    diagnostics=component_diagnostics(fc,stats)
    frozen_forecasts,frozen_diagnostics=frozen_initial_validation(base,data)
    event_totals,event_ensemble=event_summary_tables(events,contributions)
    exports={'weights':weights,'returns':returns,'metrics':metrics,'risk_history_audit':risks,
        'exposure_timing_monthly_paths':paths,'retrospective_average_weights':averages,
        'exposure_timing_decomposition':decomp,'exposure_timing_comparison':differences,
        'event_factor_pnl':events,'studentized_intervals':intervals,'spanning_hac':spanning,
        'event_portfolio_totals':event_totals,'event_ensemble_predictor_contributions':event_ensemble,
        'mde':mde,'mean_difference_hac':means,'direct_h2_forecast_loss':h2,'direct_h1_forecast_loss':h1,
        'validation_component_diagnostics':diagnostics,
        'initial_fit_frozen_validation_forecasts':frozen_forecasts,
        'initial_fit_frozen_validation_diagnostics':frozen_diagnostics,
        'event_predictor_contributions':contributions[contributions.Return_Date.isin(EVENTS)]}
    for name,frame in exports.items():frame.to_csv(out/(name+'.csv'),index=False)
    checks=verify(base,fc,weights,returns,paths,intervals,timing,contributions)
    audit={'parameter_selection':'us_core.fit_tuning calls GridSearchCV.fit only on261 initial train rows; validation predict is reporting-only',
        'validation_feedback':'No validation error is used in best_params selection; subsequent researcher specification choices remain retrospective',
        'training_signals':'1990-03 through2011-11;261;labels1990-04 through2011-12',
        'validation_signals':'2011-12 through2019-02;87;targets2012-01 through2019-03',
        'extended_returns':'2012-01 through2026-07;175',
        'original_returns':'2019-04 through2026-07;88',
        'tuning_source_sha256':hashlib.sha256((base/'src/us_core.py').read_bytes()).hexdigest()}
    (out/'tuning_validation_audit.json').write_text(json.dumps(audit,indent=2))
    summary={'primary_model':'Macro','primary_allocation':ALLOCATIONS[0],'sensitivity_allocation':ALLOCATIONS[1],
        'state_role':'Fixed and expanding YoY are H2 diagnostics; no claim to restore four-state identification',
        'designs':DESIGNS,'expanding_state_joined':has_expanding,'cohorts':{k:[str(v[0].date()),str(v[1].date())] for k,v in COHORTS.items()},
        'costs_bps':[0,10,25,50],'entry':'Each88/175cohort starts at zero turnover independently; deleted-event paths keep original charges',
        'decomposition':'Gross dynamic=own full-cohort average-weight static exposure+dynamic weight deviations; net subtracts allocation cost',
        'static_benchmark_warning':'Average weights include future choices; retrospective attribution only, never an investable ex ante benchmark',
        'covid_rule':'Remove Mar-Dec2020 outcomes, preserve full-cohort average weights and decisions; N78/165',
        'spanning':'Macro versus HM and state versus Macro, intercept+beta, Bartlett HAC3/6/12 preserving gaps; alpha is not pure timing',
        'MDE':'Annual Sharpe delta approximately (z.975+z.8)*actual paired IF-HAC SE; IID JK-Memmel secondary; years not a time forecast',
        'event_attribution':'Model predictions decompose intercept plus coefficient*standardized feature. Portfolio PnL decomposes weight differences*actual factor returns. Nonlinear score mapping does not identify feature-specific PnL.',
        'direct_H2':'OLS state vs reduced Macro fitted on exactly identical eligible labels; ensemble CW approximate; Holm6 per design/cohort/state/estimator/method/HAC',
        'direct_H1':'Macro OLS vs matched nested intercept mean; ensemble vs fuller-history historical mean CW approximate; DM and CW with HAC3/6/12 and Holm6',
        'bootstrap':'Conditional paired moving-block percentile-t,3000default,block6,HAC3; original and each draw studentized; no selection/refit uncertainty',
        'validation_diagnostics':'validation_component_diagnostics uses expanding refits; initial_fit_frozen_validation_diagnostics fits once on initial261 and evaluates87 with frozen coefficients/scaler; archived tuning validation errors independently reconciled',
        'verification_checks':checks,'forecast_rows':len(fc),'inference_rows':len(intervals),'finite_draws_min':int(intervals.Finite_Draws.min())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({'output':str(out),'verification_checks':checks,'expanding_state_joined':has_expanding},ensure_ascii=False),flush=True)
    return summary

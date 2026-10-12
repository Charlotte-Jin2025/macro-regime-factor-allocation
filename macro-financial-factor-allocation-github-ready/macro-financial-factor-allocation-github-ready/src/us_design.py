"""Release-aware feature calendars and natural-calendar inference utilities."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from . import us_core as u
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'results/design'
DESIGNS=['Release_corrected_current_vintage','Partial_realtime_IP']
SEED=20261007
def features(data, realtime=False):
    x=u.feature_data(data)
    growth=data.IP_Growth.rolling(3).mean().shift(1).reindex(x.index)
    inflation=data.Inflation.rolling(3).mean().shift(1).reindex(x.index)
    monthly_g=data.IP_Growth.shift(1).reindex(x.index)
    if realtime:
        vint=pd.read_csv(BASE/'data/realtime/ipt.csv',index_col=0,na_values=['#N/A'])
        vint.index=pd.to_datetime(vint.index.str.replace(':','-')+'-01').to_period('M').to_timestamp('M')
        audit=[]
        for date in x.index:
            col=f'IPT{date.year%100:02d}M{date.month}'
            ref=date-pd.offsets.MonthEnd()
            series=pd.to_numeric(vint[col],errors='coerce')
            changes=series.pct_change(fill_method=None)*100
            x.loc[date,'IP_Growth_lag']=changes.get(ref,np.nan)
            monthly_g.loc[date]=changes.get(ref,np.nan)
            growth.loc[date]=changes.rolling(3).mean().get(ref,np.nan)
            last=series.dropna().index.max()
            assert last<=ref
            audit.append({'Signal_Date':date,'Vintage_Column':col,'Reference_Month':ref,
                          'Latest_Available_Observation':last,'IP_Growth_lag':x.loc[date,'IP_Growth_lag'],
                          'IP_Growth_3M_lag':growth.loc[date],
                          'Current_Vintage_IP_Growth_lag':data.IP_Growth.shift(1).get(date,np.nan)})
        pd.DataFrame(audit).to_csv(OUT/'ip_vintage_feature_audit.csv',index=False)
    # Keep the exact M-1 reference rule. No stale carry-forward or nowcast substitution.
    for date in [pd.Timestamp('2025-10-31'),pd.Timestamp('2025-11-30')]:
        x.loc[date,'IP_Growth_lag']=np.nan
        growth.loc[date]=np.nan
        monthly_g.loc[date]=np.nan
    x.loc[pd.Timestamp('2025-10-31'),'Unemployment_lag']=np.nan
    train=x.loc[:u.TRAIN_END].dropna(subset=u.PREDICTORS+[f+'_next' for f in u.FACTORS])
    for key,g,i in [('3M',growth,inflation),('Monthly',monthly_g,data.Inflation.shift(1).reindex(x.index))]:
        cg,ci=g.reindex(train.index).median(),i.reindex(train.index).median()
        x['Regime_'+key]=u.regimes_from(g,i,cg,ci).reindex(x.index)
        for regime in u.REGIMES[1:]:x['State_'+key+'_'+regime]=(x['Regime_'+key]==regime).fillna(False).astype(float)
    x['Growth_3M']=growth;x['Inflation_3M']=inflation
    return x

def frozen_parameters(base, design):
    frame = pd.read_csv(base / 'results/design' / (design + '_tuning.csv'))
    params = {}
    for f in u.FACTORS:
        params[f] = {}
        for name in ['Ridge', 'Elastic_Net']:
            row = frame[(frame.Factor == f) & (frame.Model == name)].iloc[0]
            params[f][name] = {'alpha': float(row.alpha)}
            if name == 'Elastic_Net':
                params[f][name]['l1_ratio'] = float(row.l1_ratio)
    return params, frame


def yoy_features(base, data, design):
    calendar = pd.read_csv(base / 'results/design' /
                           (design + '_model_calendar.csv'), index_col=0, parse_dates=True)
    growth = data.IP_Growth_YoY.shift(1).reindex(calendar.index)
    inflation = data.Inflation_YoY.shift(1).reindex(calendar.index)
    audit = []
    if design == 'Partial_realtime_IP':
        vint = pd.read_csv(base / 'data/realtime/ipt.csv', index_col=0, na_values=['#N/A'])
        vint.index = pd.to_datetime(vint.index.str.replace(':', '-') + '-01').to_period('M').to_timestamp('M')
        for date in calendar.index:
            col = f'IPT{date.year % 100:02d}M{date.month}'
            ref = date - pd.offsets.MonthEnd()
            denominator_ref = ref - pd.offsets.MonthEnd(12)
            series = pd.to_numeric(vint[col], errors='coerce')
            numerator = series.get(ref, np.nan)
            denominator = series.get(denominator_ref, np.nan)
            value = 100 * (numerator / denominator - 1)
            assert series.dropna().index.max() <= ref
            growth.loc[date] = value
            audit.append({'Signal_Date': date, 'Vintage_Column': col, 'Reference_Month': ref,
                          'Denominator_Reference_Month': denominator_ref,
                          'Numerator_Same_Vintage': numerator, 'Denominator_Same_Vintage': denominator,
                          'YoY_Before_Release_Mask': value})
    # September and October 2025 IP were not released by the prescribed signals.
    growth.loc[pd.to_datetime(['2025-10-31', '2025-11-30'])] = np.nan
    calendar['IP_YoY_lag'] = growth
    calendar['CPI_YoY_lag'] = inflation
    train = calendar.loc[:u.TRAIN_END].dropna(subset=u.PREDICTORS + [f + '_next' for f in u.FACTORS])
    cuts = train[['IP_YoY_lag', 'CPI_YoY_lag']].median()
    calendar['Regime_YoY'] = u.regimes_from(growth, inflation, cuts.IP_YoY_lag, cuts.CPI_YoY_lag)
    for state in u.REGIMES[1:]:
        calendar['State_YoY_' + state] = (calendar.Regime_YoY == state).fillna(False).astype(float)
    thresholds = [{'Design': design, 'Variant': 'YoY', 'Variable': key,
                   'Training_Median_Percent': float(cuts[key]),
                   'Training_Signal_Start': u.START, 'Training_Signal_End': u.TRAIN_END,
                   'Threshold_Observations': int(train[key].notna().sum()),
                   'Rule': 'High strictly exceeds median; equality is low; frozen after initial training'}
                  for key in cuts.index]
    return calendar, thresholds, audit


def sharpe_if(values):
    """Annual sample-SD Sharpe and its empirical delta-method influence scores.

    values shape (..., time, two strategies). Use population moments to derive
    IF, multiplied by sqrt((N-1)/N) to match the sample-SD point estimator.
    """
    values = np.asarray(values, float)
    n = values.shape[-2]
    mean = values.mean(axis=-2, keepdims=True)
    centered = values - mean
    variance = (centered ** 2).mean(axis=-2, keepdims=True)
    sd = np.sqrt(variance)
    correction = np.sqrt((n - 1) / n)
    with np.errstate(divide='ignore', invalid='ignore'):
        sr = np.sqrt(12) * correction * (mean / sd).squeeze(-2)
        influence = np.sqrt(12) * correction * (centered / sd -
                    mean * (centered ** 2 - variance) / (2 * sd ** 3))
    return sr[..., 0] - sr[..., 1], influence[..., 0] - influence[..., 1]


def calendar_hac_se(scores, dates, lag):
    """Bartlett IF-HAC standard error, keeping actual missing-month gaps."""
    scores = np.asarray(scores, float)
    months = pd.DatetimeIndex(dates).to_period('M').asi8
    n = len(months)
    scores = scores - scores.mean(axis=-1, keepdims=True)
    meat = np.sum(scores ** 2, axis=-1)
    lookup = {m: i for i, m in enumerate(months)}
    for k in range(1, lag + 1):
        pairs = [(i, lookup[m-k]) for i, m in enumerate(months) if m-k in lookup]
        if pairs:
            a, b = np.asarray(pairs).T
            meat += 2 * (1 - k / (lag + 1)) * np.sum(scores[..., a] * scores[..., b], axis=-1)
    return np.sqrt(np.maximum(meat, 0)) / n


def calendar_segments(dates):
    months = pd.DatetimeIndex(dates).to_period('M').asi8
    return np.split(np.arange(len(months)), np.where(np.diff(months) != 1)[0] + 1)


def paired_block_indices(dates, block, draws, rng):
    """Non-circular overlapping blocks, separately within retained segments.

    Preserves the Covid-excluded 11+67 cohort and never joins source blocks
    across its missing event. The length-12 draw fixes the short 11-month
    segment, which is explicitly reported as a finite-sample boundary.
    """
    pieces, lengths, candidates = [], [], []
    for segment in calendar_segments(dates):
        n = len(segment)
        length = min(block, n)
        starts = rng.integers(0, n - length + 1, size=(draws, int(np.ceil(n / length))))
        indexes = (starts[..., None] + np.arange(length)).reshape(draws, -1)[:, :n]
        pieces.append(segment[indexes])
        lengths.append(length)
        candidates.append(n - length + 1)
    return np.concatenate(pieces, axis=1), lengths, candidates


def studentized_sharpe_interval(r_a, r_b, dates, B=3000, block=6, hac=3, seed=SEED):
    """Public API: paired decimal monthly returns, sorted month-end dates.

    Returns the A-minus-B annual Sharpe difference and 95% HAC-studentized
    moving-block percentile-t interval. Units may both be percent instead of
    decimal without affecting results. Missing episodes are represented by
    absent dates, not NaNs. Segment sizes/gaps are preserved as described above.
    """
    pair = np.column_stack([r_a, r_b]).astype(float)
    dates = pd.DatetimeIndex(dates)
    if len(pair) != len(dates) or len(pair) < 12 or not np.isfinite(pair).all():
        raise ValueError('Require at least 12 aligned finite monthly paired observations')
    if dates.has_duplicates or not dates.is_monotonic_increasing or not dates.is_month_end.all():
        raise ValueError('Dates must be unique, sorted month-end observations')
    theta, scores = sharpe_if(pair)
    se = float(calendar_hac_se(scores, dates, hac))
    indices, lengths, candidates = paired_block_indices(dates, block, B, np.random.default_rng(seed))
    boot_theta, boot_scores = sharpe_if(pair[indices])
    boot_se = calendar_hac_se(boot_scores, dates, hac)
    finite = np.isfinite(boot_theta) & np.isfinite(boot_se) & (boot_se > 1e-12)
    if finite.sum() < .95 * B or not np.isfinite(se) or se <= 1e-12:
        raise ValueError('Degenerate paired Sharpe studentizer or >5% unusable draws')
    low, high = np.quantile((boot_theta[finite] - theta) / boot_se[finite], [.025, .975])
    return {'Method': 'HAC-studentized moving-block percentile-t', 'N_Months': len(pair),
            'Block_Length': block, 'HAC_Lags': hac, 'Sharpe_Difference': float(theta),
            'Original_HAC_SE': se, 'CI_Low': float(theta - high * se), 'CI_High': float(theta - low * se),
            'Pivot_Q025': float(low), 'Pivot_Q975': float(high), 'Requested_Draws': B,
            'Finite_Draws': int(finite.sum()), 'Invalid_Draws': int((~finite).sum()),
            'Bootstrap_SE_Min': float(boot_se[finite].min()), 'Bootstrap_SE_Max': float(boot_se[finite].max()),
            'Retained_Segment_Lengths': '|'.join(map(str, map(len, calendar_segments(dates)))),
            'Effective_Segment_Block_Lengths': '|'.join(map(str, lengths)),
            'Candidate_Blocks_Per_Segment': '|'.join(map(str, candidates)),
            'Short_Segment_Fixed': any(x == 1 for x in candidates), 'Seed': seed}

def mean_hac(x, lag, dates=None):
    x = np.asarray(x, float)
    if dates is not None:
        # Excluded fallback months remain zero estimating-equation scores on
        # the month calendar; lags must not jump across omitted observations.
        residual = pd.Series(x - x.mean(), index=pd.DatetimeIndex(dates))
        calendar = pd.date_range(residual.index.min(), residual.index.max(), freq="ME")
        residual = residual.reindex(calendar, fill_value=0.).to_numpy()
        meat = np.sum(residual ** 2)
        for k in range(1, lag + 1):
            meat += 2 * (1 - k / (lag + 1)) * np.dot(residual[k:], residual[:-k])
        se = np.sqrt(max(meat, 0.)) / len(x)
        statistic = x.mean() / se if se else np.nan
        return {"Mean": x.mean(), "HAC_SE": se, "Statistic": statistic,
                "p_Two_Sided": 2 * norm.sf(abs(statistic)), "p_One_Sided_Positive": norm.sf(statistic),
                "CI_Low": x.mean() - norm.ppf(.975) * se, "CI_High": x.mean() + norm.ppf(.975) * se,
                "N": len(x), "HAC_Lags": lag}
    fit = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    lo, hi = fit.conf_int()[0]
    return {"Mean": x.mean(), "HAC_SE": fit.bse[0], "Statistic": fit.tvalues[0],
            "p_Two_Sided": fit.pvalues[0], "p_One_Sided_Positive": norm.sf(fit.tvalues[0]),
            "CI_Low": lo, "CI_High": hi, "N": len(x), "HAC_Lags": lag}


def run_analysis(base_path):
    global BASE, OUT
    BASE=Path(base_path).resolve();OUT=BASE/'results/design';OUT.mkdir(parents=True,exist_ok=True)
    data,manifest=u.load_data(BASE)
    data.to_csv(OUT/'calendar_data.csv')
    manifest.to_csv(OUT/'input_manifest.csv',index=False)
    specification=json.loads((BASE/'specification.json').read_text())
    for design,realtime in [(DESIGNS[0],False),(DESIGNS[1],True)]:
        calendar=features(data,realtime)
        params,tuning=u.fit_tuning(calendar, fixed_parameters=specification['us_parameters'][design])
        calendar.to_csv(OUT/(design+'_model_calendar.csv'))
        tuning.to_csv(OUT/(design+'_tuning.csv'),index=False)
        print(design+': initial-training cross-validation complete',flush=True)
    events=[('CPI','2025-08','2025-09-11'),('CPI','2025-09','2025-10-24'),('CPI','2025-10',None),('CPI','2025-11','2025-12-18'),
        ('Unemployment','2025-08','2025-09-05'),('Unemployment','2025-09','2025-11-20'),('Unemployment','2025-10',None),('Unemployment','2025-11','2025-12-16'),
        ('IP','2025-08','2025-09-16'),('IP','2025-09','2025-12-03'),('IP','2025-10','2025-12-23'),('IP','2025-11','2025-12-23')]
    pd.DataFrame(events,columns=['Series','Reference_Month','First_Release_Date']).to_csv(OUT/'release_calendar.csv',index=False)
    rows=[]
    for signal in pd.date_range('2025-09-30','2025-12-31',freq='ME'):
        ref=(signal-pd.offsets.MonthEnd()).strftime('%Y-%m')
        for series in ['CPI','Unemployment','IP']:
            release=next(e[2] for e in events if e[0]==series and e[1]==ref)
            rows.append(dict(Signal_Date=signal,Series=series,Reference_Month=ref,First_Release_Date=release,
                Level_Released_By_Signal=release is not None and pd.Timestamp(release)<=signal))
    pd.DataFrame(rows).to_csv(OUT/'release_availability_audit.csv',index=False)

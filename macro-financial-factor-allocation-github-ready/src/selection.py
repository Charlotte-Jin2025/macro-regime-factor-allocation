"""Reconstruct documented allocation mappings and a bounded-family Reality Check.

All inputs are this run's expanding forecasts and raw factor returns. No
mapping is selected or substituted. The bootstrap is conditional on fixed
forecast/weight paths and cannot reconstruct the unrecorded research search.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from . import us_core as u
from .us_analysis import COHORTS
from .us_design import paired_block_indices

DESIGN = 'Release_corrected_current_vintage'
PRIMARY = 'LS5_Variance36_Cap40_Shrink50'
STRATEGIES = ['Historical_Mean','Macro','Macro_M1_YoY','Macro_M1_ExpandingYoY']
MAPPINGS = ['Raw6_Sensitivity', 'Raw6_VolCeiling5',
            'Risk6_Cap40_Shrink50', 'Risk6_Cap40_Shrink50_VolCeiling5',
            'Rank6_Cap40_Shrink50', 'Rank6_Cap40_Shrink50_VolCeiling5',
            'LS5_PositiveRaw', PRIMARY,
            'FixedMarket1of6_LS5_PositiveRaw',
            'FixedMarket1of6_LS5_Variance36_Cap40_Shrink50']
SEED = 20261008


def allocation(scores, covariance, mapping):
    """Return six risky weights and zero-return cash weight for a known mapping."""
    variance = np.maximum(np.diag(covariance), 1e-12)
    cash = 0.0
    if mapping in MAPPINGS[:6]:
        if mapping.startswith('Raw6'):
            w = u.scores_to_weights(scores)[0]
        else:
            values = scores / variance if mapping.startswith('Risk6') else rankdata(scores, method='average')
            raw = u.scores_to_weights(values)[0]
            w = u.scores_to_weights(.5 * raw + .5 / 6, cap=.4)[0]
        if mapping.endswith('VolCeiling5'):
            sigma = np.sqrt(max(float(w @ covariance @ w), 0.0) * 12)
            exposure = min(1.0, .05 / sigma) if sigma > 0 else 1.0
            w *= exposure
            cash = 1.0 - exposure
    else:
        market = 1 / 6 if mapping.startswith('FixedMarket') else 0.0
        raw = u.scores_to_weights(scores[1:] / variance[1:] if 'Variance36' in mapping else scores[1:])[0]
        ls = u.scores_to_weights(.5 * raw + .5 / 5, cap=.4)[0] if 'Variance36' in mapping else raw
        w = np.r_[market, (1 - market) * ls]
    return w, cash


def reality_check(differences, dates, block, draws=9999, seed=SEED):
    """White-style least-favourable null: common block indices over centred d_t.

    Each column is a matched Macro/state-minus-HM return contrast, in percent
    per month. The statistic is max(0,max_j sqrt(n)*mean(d_j)). Centring all
    candidates imposes the boundary joint null. Negative-mean candidates are
    retained, making this the conservative unstudentized Reality Check rather
    than Hansen SPA. Monte Carlo tail uses (1+count)/(B+1).
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(dates) or not np.isfinite(values).all():
        raise ValueError('Reality Check requires an aligned finite T by K contrast matrix')
    dates = pd.DatetimeIndex(dates)
    if dates.has_duplicates or not dates.is_monotonic_increasing or not dates.is_month_end.all():
        raise ValueError("Dates must be unique sorted month-end observations")
    n = len(dates)
    if n < block or block < 1 or draws < 1:
        raise ValueError('Invalid block length or bootstrap draws')
    means = values.mean(axis=0)
    centered = values - means
    indices, lengths, candidates = paired_block_indices(dates, block, draws, np.random.default_rng(seed))
    diagnostics = {"effective_block_lengths":lengths,"candidate_blocks":candidates}
    bootstrap_means = centered[indices].mean(axis=1)
    maxima = np.maximum(0, np.sqrt(n) * bootstrap_means.max(axis=1))
    observed = max(0.0, float(np.sqrt(n) * means.max()))
    p = (1 + int(np.sum(maxima >= observed))) / (draws + 1)
    candidate_p = np.asarray([(1 + np.sum(maxima >= max(0.0, np.sqrt(n)*v))) / (draws + 1) for v in means])
    return {'p':float(p), 'means':means, 'candidate_p':candidate_p,
            'best':int(np.argmax(means)), 'observed':observed, 'block_diagnostics':diagnostics}


def run_analysis(base_path=None, bootstrap_n=9999):
    base = Path(base_path).resolve() if base_path is not None else Path(__file__).resolve().parents[1]
    out = base / 'results/selection'; out.mkdir(parents=True, exist_ok=True)
    data, _ = u.load_data(base)
    fc = pd.read_csv(base / 'results/us/forecasts.csv', parse_dates=['Signal_Date','Return_Date'])
    fc = fc[fc.Design.eq(DESIGN)]
    actual = fc.pivot(index='Return_Date',columns='Factor',values='Actual')[u.FACTORS]
    weights = []
    for date, group in fc.groupby('Signal_Date',sort=True):
        g = group.set_index('Factor').loc[u.FACTORS]
        history = data.loc[:date,u.FACTORS].tail(36) / 100
        if len(history) != 36 or history.isna().any(axis=None):
            raise AssertionError('Complete preceding 36-month covariance required')
        covariance = history.cov().to_numpy()
        for mapping in MAPPINGS:
            for strategy in STRATEGIES:
                w, cash = allocation(g[strategy].to_numpy() / 100, covariance, mapping)
                weights.append({'Signal_Date':date,'Return_Date':g.Return_Date.iloc[0],
                    'Mapping':mapping,'Strategy':strategy,'Cash_Weight':cash, **dict(zip(u.FACTORS,w))})
    weights = pd.DataFrame(weights)
    if not np.allclose(weights[u.FACTORS].sum(axis=1) + weights.Cash_Weight, 1):
        raise AssertionError('Total risky and cash weights must sum to one')
    rows, metrics, comparisons, checks = [], [], [], []
    for cohort,(lo,hi) in COHORTS.items():
        for (mapping,strategy),w in weights[weights.Return_Date.between(lo,hi)].groupby(['Mapping','Strategy']):
            w = w.sort_values('Return_Date')
            matrix = w[u.FACTORS + ['Cash_Weight']].to_numpy()
            turnover = np.r_[0., .5 * np.abs(np.diff(matrix,axis=0)).sum(axis=1)]
            gross = (w[u.FACTORS].to_numpy() * actual.loc[w.Return_Date].to_numpy()).sum(axis=1)
            frame = pd.DataFrame({'Cohort':cohort,'Mapping':mapping,'Strategy':strategy,
                'Return_Date':w.Return_Date.to_numpy(),'Gross_Return_pct':gross,'Turnover':turnover,
                'Allocation_Cost_pct':turnover*.1,'Net_Return_pct':gross-turnover*.1})
            rows.append(frame)
            metrics.append({'Cohort':cohort,'Mapping':mapping,'Strategy':strategy,
                **u.performance(frame.Net_Return_pct),'Mean_Turnover':turnover.mean()})
    returns = pd.concat(rows,ignore_index=True); metrics = pd.DataFrame(metrics)
    original = pd.read_csv(base / 'results/us/returns.csv',parse_dates=['Return_Date'])
    original = original[original.Design.eq(DESIGN) & original.Cost_bps.eq(10)]
    for mapping in [PRIMARY,'Raw6_Sensitivity']:
        a = returns[returns.Mapping.eq(mapping)].rename(columns={'Mapping':'Allocation'})
        b = original[original.Allocation.eq(mapping) & original.Strategy.isin(STRATEGIES)]
        keys=['Cohort','Allocation','Strategy','Return_Date']
        joined=a.merge(b,on=keys,validate='one_to_one',suffixes=('_new','_existing'))
        err = np.max(np.abs(joined.Net_Return_pct_new - joined.Net_Return_pct_existing))
        if len(joined) != len(a) or not np.allclose(joined.Net_Return_pct_new,joined.Net_Return_pct_existing,rtol=1e-9,atol=1e-10):
            raise AssertionError('Reconstructed primary/raw mapping differs from the main analysis')
        checks.append({'mapping':mapping,'max_abs_return_pct_difference':float(err),'passed':True})
    rc, individual = [], []
    for cohort, group in returns.groupby('Cohort'):
        contrasts, names = [], []
        for mapping in MAPPINGS:
            panel=group[group.Mapping.eq(mapping)].pivot(index='Return_Date',columns='Strategy',values='Net_Return_pct').sort_index()
            m=metrics[metrics.Cohort.eq(cohort)&metrics.Mapping.eq(mapping)].set_index('Strategy')
            comparisons.append({'Cohort':cohort,'Mapping':mapping,'N_Months':len(panel),
                'Macro_Sharpe':m.loc['Macro','Sharpe'],'Historical_Mean_Sharpe':m.loc['Historical_Mean','Sharpe'],
                'Sharpe_Difference':m.loc['Macro','Sharpe']-m.loc['Historical_Mean','Sharpe'],
                'Annual_Mean_Gain_pp':12*(panel.Macro-panel.Historical_Mean).mean()})
            for strategy in STRATEGIES[1:]:
                contrasts.append((panel[strategy]-panel.Historical_Mean).to_numpy())
                names.append(mapping+' | '+strategy)
        matrix=np.asarray(contrasts).T
        dates=panel.index
        for family, use in [('Macro_10_Mappings',[i for i,name in enumerate(names) if name.endswith(' | Macro')]),
                            ('Information_Models_30',[*range(len(names))])]:
            values=matrix[:,use]; labels=[names[i] for i in use]
            primary_index=labels.index(PRIMARY+' | Macro')
            for block in [3,6,12]:
                result=reality_check(values,dates,block,bootstrap_n,SEED+len(dates)+block)
                rc.append({'Cohort':cohort,'Family':family,'Block_Length':block,'N_Months':len(dates),
                    'N_Candidates':len(labels),'Bootstrap_Draws':bootstrap_n,
                    'Observed_Max_Annual_Mean_Gain_pp':12*result['means'][result['best']],
                    'Best_Candidate':labels[result['best']],'Reality_Check_p':result['p'],
                    'Primary_Unadjusted_Annual_Mean_Gain_pp':12*result['means'][primary_index],
                    'Primary_Max_Adjusted_p':result['candidate_p'][primary_index]})
                for i,name in enumerate(labels):
                    individual.append({'Cohort':cohort,'Family':family,'Block_Length':block,'Candidate':name,
                        'Annual_Mean_Gain_pp':12*result['means'][i],'Max_Adjusted_p':result['candidate_p'][i]})
    # Calendar-year outcomes are descriptive, and 2026 contains January-July only.
    primary = original[(original.Allocation.eq(PRIMARY)) & original.Cohort.eq('Extended_175') &
                       original.Strategy.isin(['Macro','Historical_Mean'])]
    annual=[]
    for (year,strategy),g in primary.groupby([primary.Return_Date.dt.year,'Strategy']):
        annual.append({'Year':int(year),'Strategy':strategy,'N_Months':len(g),
            'Compounded_Return_pct':100*((1+g.Net_Return_pct/100).prod()-1),
            'Arithmetic_Return_pp':g.Net_Return_pct.sum()})
    remaining = primary[primary.Strategy.eq('Macro') & primary.Return_Date.dt.year.ne(2022)]
    summary={'design':DESIGN,'mappings':MAPPINGS,'strategies':STRATEGIES,'allocation_cost_bps':10,
        'bootstrap_draws':bootstrap_n,'seed':SEED,'matching_main_paths':checks,
        'non_2022_macro_compounded_return_pct':float(100*((1+remaining.Net_Return_pct/100).prod()-1)),
        'scope':'White-style unstudentized maximum of matched mean-return increments; ten documented mappings and thirty information-model/mapping combinations, with common paired non-circular moving blocks. Conditional on fixed fitted paths. Reconstructed on identical release-corrected forecasts. No selection or primary-specification replacement. A complete historical search log is unavailable; this bounded family does not correct undisclosed experiments, model refitting, state-rule choice, or sample-window selection.',
        'null':'All candidate expected matched mean-return gains are nonpositive. All series centered at their own observed means to impose the least-favourable boundary null.',
        'funding':'Volatility-ceiling cash earns zero excess return. Factor and funding costs remain outside the synthetic exercise.',
        'cohort_entry':'Each 175/88 cohort independently starts with zero target-change turnover.'}
    for name,frame in [('mapping_weights',weights),('mapping_returns',returns),('mapping_metrics',metrics),
                       ('mapping_comparisons',pd.DataFrame(comparisons)),('reality_check',pd.DataFrame(rc)),
                       ('candidate_adjusted_p',pd.DataFrame(individual)),('annual_returns',pd.DataFrame(annual))]:
        frame.to_csv(out/(name+'.csv'),index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Documented mapping family and Reality Check complete',flush=True)
    return summary

if __name__=='__main__':
    run_analysis()

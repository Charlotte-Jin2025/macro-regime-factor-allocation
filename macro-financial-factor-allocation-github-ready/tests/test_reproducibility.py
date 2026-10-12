"""Meaningful portability, timing, accounting and joint-test invariance checks."""
from pathlib import Path
import json
import platform
import sys
# Make direct execution independent of the terminal working directory.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.us_core import FACTORS, match_grid_parameters
from src.international_core import unique_alpha_index
from src.selection import reality_check


def run_checks(base_path=None):
    base=Path(base_path).resolve() if base_path is not None else Path(__file__).resolve().parents[1]
    checks=[]
    def record(name, action):
        action();checks.append({'name':name,'passed':True})
    def check(condition):
        if not condition:raise AssertionError('Scientific/portability check failed')
    def shifted_grid():
        alpha=316.22776601683796
        shifted=np.nextafter(alpha,-np.inf)
        check(shifted != alpha)
        check(match_grid_parameters([{'model__alpha':shifted}],{'alpha':alpha})==0)
        check(unique_alpha_index(alpha,np.array([shifted]))==0)
    def outside_grid():
        try:match_grid_parameters([{'model__alpha':100.}],{'alpha':316.22776601683796})
        except RuntimeError:return
        raise AssertionError('Out-of-grid penalty was accepted')
    def ambiguous_grid():
        try:match_grid_parameters([{'model__alpha':1.},{'model__alpha':np.nextafter(1.,2.)}],{'alpha':1.})
        except RuntimeError:return
        raise AssertionError('Ambiguous penalty grid was accepted')
    record('US and international accept one-ULP grid representation changes',shifted_grid)
    record('Out-of-grid penalty is rejected',outside_grid)
    record('Ambiguous grid matches are rejected',ambiguous_grid)
    dates=pd.date_range('2012-01-31',periods=48,freq='ME')
    rng=np.random.default_rng(214)
    series=rng.normal(0,.3,size=48)+.035
    def duplicate_candidates():
        a=reality_check(series[:,None],dates,6,499,12)
        b=reality_check(np.tile(series[:,None],(1,3)),dates,6,499,12)
        check(a['p']==b['p'])
    def joint_null():
        x=np.c_[series-series.mean()-.2,-series+series.mean()-.15]
        check(reality_check(x,dates,6,499,12)['p']==1.0)
    def fixed_positive():
        check(reality_check(np.ones((48,2))*.2,dates,6,499,12)['p']==1/500)
    record('Reality Check invariant to duplicate perfectly correlated candidates',duplicate_candidates)
    record('Reality Check returns p=1 when all observed candidate means are negative',joint_null)
    record('Positive deterministic contrast has the Monte Carlo minimum p',fixed_positive)
    def invalid_dates():
        try:reality_check(series[:,None],dates[::-1],6,499,12)
        except ValueError:return
        raise AssertionError('Unsorted date sequence accepted')
    record('Reality Check rejects an unsorted calendar',invalid_dates)
    timing=pd.read_csv(base/'results/us/training_timing_audit.csv')
    def causal_labels():
        signal=pd.to_datetime(timing.Signal_Date)
        for key in ['Max_Macro_Training_Label_Date','Max_YoY_Training_Label_Date','Historical_Return_End']:
            check((pd.to_datetime(timing[key])<=signal).all())
    record('All fitted target labels and historical returns precede or equal their signal',causal_labels)
    weights=pd.read_csv(base/'results/us/weights.csv')
    def weight_constraints():
        w=weights[weights.Allocation.eq('LS5_Variance36_Cap40_Shrink50') & weights.Strategy.eq('Macro')]
        check(np.allclose(w[FACTORS].sum(axis=1),1))
        check(np.allclose(w.MKT_RF,0))
        check(w[FACTORS[1:]].min(axis=None)>=.1-1e-10)
        check(w[FACTORS[1:]].max(axis=None)<=.4+1e-10)
    record('Primary Macro weights obey five-sleeve total and 10-40 percent bounds',weight_constraints)
    returns=pd.read_csv(base/'results/us/returns.csv')
    record('Gross minus recorded allocation costs equals every net return',
           lambda:check(np.allclose(returns.Gross_Return_pct-returns.Allocation_Cost_pct,returns.Net_Return_pct,rtol=1e-10,atol=1e-12)))
    annual=pd.read_csv(base/'results/selection/annual_returns.csv')
    def annual_accounting():
        r=returns[returns.Design.eq('Release_corrected_current_vintage') & returns.Allocation.eq('LS5_Variance36_Cap40_Shrink50') & returns.Cohort.eq('Extended_175') & returns.Cost_bps.eq(10)]
        for strategy in ['Macro','Historical_Mean']:
            original=(1+r[r.Strategy.eq(strategy)].Net_Return_pct.to_numpy()/100).prod()
            reconstructed=(1+annual[annual.Strategy.eq(strategy)].Compounded_Return_pct.to_numpy()/100).prod()
            check(np.isclose(original,reconstructed,rtol=1e-12,atol=1e-12))
        check((annual[annual.Year.eq(2026)].N_Months==7).all())
    record('Calendar-year compounds multiply back to the complete path and 2026 stays partial',annual_accounting)
    selection=json.loads((base/'results/selection/summary.json').read_text())
    record('Reconstructed primary and raw mapping paths reconcile with main analysis',
           lambda:check(len(selection['matching_main_paths'])==2 and all(x['passed'] for x in selection['matching_main_paths'])))
    report={'all_passed':True,'tests_run':len(checks),'platform':platform.platform(),
            'python':platform.python_version(),'checks':checks,
            'scope':'Semantic and floating-representation checks on the platform named above. One-ULP checks simulate rounding differences; they do not establish execution on untested platforms.'}
    (base/'results/scientific_tests.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Scientific and portability checks: {len(checks)} passed.',flush=True)
    return report

if __name__=='__main__':
    if len(sys.argv)>1:run_checks(sys.argv[1])
    else:run_checks()

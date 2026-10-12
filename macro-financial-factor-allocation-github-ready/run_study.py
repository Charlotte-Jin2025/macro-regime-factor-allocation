"""Rebuild the fixed study, validate its tables and generate figures; no Word report."""
from pathlib import Path
import argparse
import importlib
import json
import os
import platform
import sys
import tempfile
import time

BASE=Path(__file__).resolve().parent

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-inputs',action='store_true',help='Check exact hashes of original source files only')
    mode.add_argument('--figures-only',action='store_true',help='Draw figures from existing generated results')
    mode.add_argument('--report-only',action='store_true',help='Disabled: this project does not generate Word reports')
    mode.add_argument('--validate-only',action='store_true',help='Compare existing generated tables with numerical references')
    mode.add_argument('--check-environment',action='store_true',help='Check dependencies and project location without calculation')
    parser.add_argument('--skip-report',action='store_true',help='Compatibility option; Word generation is always disabled')
    args=parser.parse_args(argv)
    if args.report_only:
        parser.error('Word report generation is disabled. Run python run_study.py for analysis and figures.')
    from start_jupyter import require_environment
    require_environment(include_notebook=False)
    if args.check_environment:return 0
    # One thread avoids expensive oversubscription on small expanding samples.
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]='1'
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'utsg7394_final_mpl'))
    os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'utsg7394_final_cache'))
    os.environ['LOKY_MAX_CPU_COUNT']='1'
    from src.validation import verify_inputs,compare_results
    checked=verify_inputs(BASE)
    print(f'Raw source integrity: {checked} files passed.',flush=True)
    if args.verify_inputs:return 0
    if args.validate_only:
        result=compare_results(BASE);print(f"Numerical comparison: {result['files']} tables passed.",flush=True);return 0
    if args.figures_only:
        from src.plot_figures import run_analysis
        run_analysis(BASE);return 0
    started=time.monotonic();timings=[]
    stages=[('Raw calendar and initial-training CV','us_design',{}),
        ('Training diagnostics and block permutation','training',{}),
        ('Causal expanding state labels and forecasts','states',{}),
        ('175/88-month forecasts, allocation and inference','us_analysis',{'bootstrap_n':3000}),
        ('International initial CV and expanding Ridge fits','international_analysis',{}),
        ('Cost, power and event diagnostics','diagnostics',{'bootstrap_n':3000}),
        ('Evidence summaries and cross-checks','interpretation',{}),
        ('Documented mappings and bounded-family Reality Check','selection',{'bootstrap_n':9999}),
        ('Controlled IP sensitivity and annual incremental accounting','robustness',{'bootstrap_n':3000}),
        ('Frozen UK and German allocation transfer and paired uncertainty','cross_market',{'bootstrap_n':3000})]
    for number,(label,module,kwargs) in enumerate(stages,1):
        print(f'[{number}/{len(stages)}] {label}',flush=True)
        begin=time.monotonic()
        importlib.import_module('src.'+module).run_analysis(BASE,**kwargs)
        timings.append(dict(stage=label,seconds=round(time.monotonic()-begin,3)))
    result=compare_results(BASE)
    print(f"Numerical comparison: {result['files']} tables passed.",flush=True)
    from src.plot_figures import run_analysis as plot
    plot(BASE)
    from tests.test_reproducibility import run_checks
    run_checks(BASE)
    from tests.test_supplement import run_checks as supplement_checks
    supplement_checks(BASE)
    from tests.test_cross_market import run_checks as cross_market_checks
    cross_market_checks(BASE)
    import numpy,pandas,scipy,sklearn,statsmodels,matplotlib
    details=dict(completed=True,analyses_complete=True,report_generated=False,
        total_seconds=round(time.monotonic()-started,3),stages=timings,
        bootstrap_draws=3000,reality_check_draws=9999,raw_inputs_verified=checked,numeric_reference_tables=result['files'],
        scientific_checks=12,supplement_checks=8,cross_market_checks=11,
        versions={name:module.__version__ for name,module in [('numpy',numpy),('pandas',pandas),('scipy',scipy),('scikit-learn',sklearn),('statsmodels',statsmodels),('matplotlib',matplotlib)]},
        python=platform.python_version(),threads_per_library=1,
        execution='Offline; no models or result paths loaded from reference_results for fitting or portfolio calculations.')
    (BASE/'results/run_metadata.json').write_text(json.dumps(details,indent=2))
    print(f"Study rebuilt and validated in {details['total_seconds']/60:.1f} minutes. Results: {BASE/'results'}",flush=True)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (RuntimeError,FileNotFoundError,AssertionError,ModuleNotFoundError) as exc:
        print('Stopped: '+str(exc),file=sys.stderr);sys.exit(1)

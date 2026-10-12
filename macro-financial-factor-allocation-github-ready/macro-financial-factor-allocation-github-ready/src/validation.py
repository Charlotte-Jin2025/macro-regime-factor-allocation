"""Exact raw-input integrity and tolerance-based numerical-result validation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

RTOL = 1e-9
ATOL = 1e-10
NUMERIC_KEYS = {'Cost_bps', 'HAC_Lags', 'Block_Length', 'Seed', 'Gamma',
                'MOM_Internal_Cost_bps_per_month'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def verify_inputs(base):
    base = Path(base)
    manifest = json.loads((base/'data/input_manifest.json').read_text())
    for record in manifest['files'] + manifest.get('configuration_files',[]):
        path = base/record['path']
        if not path.is_file() or digest(path) != record['sha256']:
            raise RuntimeError('Raw source missing or changed: '+record['path'])
    return len(manifest['files'])

def compare_csv(expected, actual):
    left, right = pd.read_csv(expected), pd.read_csv(actual)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        raise AssertionError(f'CSV layout differs: {actual}')
    maximum = 0.0
    for column in left:
        a, b = left[column], right[column]
        if column in NUMERIC_KEYS or (pd.api.types.is_integer_dtype(a) and pd.api.types.is_integer_dtype(b)):
            if not a.equals(b):
                raise AssertionError(f'Numeric key/count mismatch: {actual.name}, {column}')
        elif pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            if not np.allclose(av, bv, rtol=RTOL, atol=ATOL, equal_nan=True):
                mask = ~np.isclose(av,bv,rtol=RTOL,atol=ATOL,equal_nan=True)
                ix = int(np.flatnonzero(mask)[0])
                raise AssertionError(f'Numeric mismatch: {actual.name}, {column}, row {ix}: {av[ix]} vs {bv[ix]}')
            finite = np.isfinite(av)&np.isfinite(bv)
            if finite.any(): maximum=max(maximum,float(np.abs(av[finite]-bv[finite]).max()))
        else:
            aa, bb = a.astype('string').fillna('<MISSING>'), b.astype('string').fillna('<MISSING>')
            if not aa.equals(bb):
                raise AssertionError(f'String/boolean/key mismatch: {actual.name}, {column}')
    return dict(rows=len(left),columns=len(left.columns),maximum_absolute_numeric_difference=maximum)

def compare_results(base):
    base = Path(base)
    records=[]
    auxiliary={'international/forecast_metrics.csv','international/model_selection.csv','international/forecasts.csv'}
    for reference in sorted((base/'reference_results').rglob('*.csv')):
        relative=reference.relative_to(base/'reference_results').as_posix()
        if relative in auxiliary:continue
        actual=base/'results'/relative
        if not actual.is_file():raise FileNotFoundError('Required regenerated result missing: '+relative)
        result=compare_csv(reference,actual)
        records.append(dict(result=relative,passed=True,**result))
    report=dict(all_passed=True,rtol=RTOL,atol=ATOL,files=len(records),checks=records,
                note='Reference tables are validation targets only; all forecasts and strategy returns are rebuilt from raw data.')
    (base/'results/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report

"""Raw US data loading, causal transformations, training-only tuning and allocation primitives."""
from pathlib import Path
import hashlib, io, re, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller
FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
PREDICTORS = ["VIX", "Credit_Spread", "Term_Spread", "Treasury_2Y_change", "Inflation_lag", "Unemployment_lag", "IP_Growth_lag"]
REGIMES = ["Expansion", "Overheating", "Slowdown", "Stagflation"]
START = pd.Timestamp("1990-03-31")
TRAIN_END = pd.Timestamp("2011-11-30")
VALID_END = pd.Timestamp("2019-02-28")
SIGNAL_END = pd.Timestamp("2026-06-30")
RETURN_END = pd.Timestamp("2026-07-31")
SEED = 20260920

def match_grid_parameters(candidates, fixed):
    """Return the unique declared CV grid point matching recorded parameters.

    A tolerance of rtol=1e-12, atol=0 only absorbs final-bit float drift.
    It is independent of loss tolerances and cannot select a new grid point.
    """
    matches = [i for i, candidate in enumerate(candidates)
               if all('model__'+key in candidate and
                      np.isclose(candidate['model__'+key], value,
                                 rtol=1e-12, atol=0.0)
                      for key, value in fixed.items())]
    if not matches:
        raise RuntimeError(f'Recorded parameter is outside the declared CV grid: {fixed}')
    if len(matches) != 1:
        raise RuntimeError(f'Recorded parameter matches multiple CV grid points: {fixed}')
    return matches[0]

def monthly_index(start, end):
    return pd.period_range(start, end, freq="M").to_timestamp("M")


def load_french(path, momentum=False):
    """Read only YYYYMM records; exclude annual summary rows and metadata."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    rows = [line for line in lines if re.match(r"^\s*\d{6}\s*,", line)]
    names = ["Date", "MOM"] if momentum else ["Date"] + FACTORS[:5] + ["RF"]
    data = pd.read_csv(io.StringIO("\n".join(rows)), names=names)
    data.index = pd.to_datetime(data.pop("Date").astype(str), format="%Y%m").dt.to_period("M").dt.to_timestamp("M")
    data.index.name = "Date"
    data = data.apply(pd.to_numeric, errors="coerce").replace([-99.99, -999.0], np.nan)
    assert not data.index.duplicated().any()
    return data


def load_data(base):
    src = base / "data" / "us"
    mapping = {"VIXCLS": "VIX", "CPIAUCSL": "CPI", "BAA10Y": "Credit_Spread",
               "T10Y2Y": "Term_Spread", "UNRATE": "Unemployment", "DGS2": "Treasury_2Y",
               "INDPRO": "Industrial_Production"}
    frames, manifest = [], []
    for stem, name in mapping.items():
        path = src / f"{stem}.csv"
        raw = pd.read_csv(path)
        raw[raw.columns[0]] = pd.to_datetime(raw.iloc[:, 0])
        series = pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").values,
                           index=pd.DatetimeIndex(raw.iloc[:, 0]), name=name)
        # Monthly source values retain missing months; daily finance is monthly mean.
        monthly = series.groupby(series.index.to_period("M")).mean()
        monthly.index = monthly.index.to_timestamp("M")
        frames.append(monthly)
        manifest.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "source_rows": len(raw), "monthly_start": str(monthly.index.min().date()),
                         "monthly_end": str(monthly.index.max().date()), "aggregation": "monthly mean"})
    for filename, mom in [("F-F_Research_Data_5_Factors_2x3.csv", False), ("F-F_Momentum_Factor.csv", True)]:
        path = src / filename
        f = load_french(path, momentum=mom)
        frames.append(f)
        manifest.append({"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "source_rows": len(f), "monthly_start": str(f.index.min().date()),
                         "monthly_end": str(f.index.max().date()), "aggregation": "original monthly percent returns"})
    data = pd.concat(frames, axis=1).sort_index()
    data = data.reindex(monthly_index(data.index.min(), RETURN_END))
    data.index.name = "Signal_Date"
    data["Inflation"] = data.CPI.pct_change(fill_method=None) * 100
    data["IP_Growth"] = data.Industrial_Production.pct_change(fill_method=None) * 100
    data["Inflation_YoY"] = data.CPI.pct_change(12, fill_method=None) * 100
    data["IP_Growth_YoY"] = data.Industrial_Production.pct_change(12, fill_method=None) * 100
    data["Treasury_2Y_change"] = data.Treasury_2Y.diff()
    for f in FACTORS:
        data[f + "_next"] = data[f].shift(-1)
    return data, pd.DataFrame(manifest)


def feature_data(data, lag=1):
    x = data.copy()
    for col in ["Inflation", "Unemployment", "IP_Growth"]:
        x[col + "_lag"] = x[col].shift(lag)
    return x.loc[START:SIGNAL_END].copy()


def regimes_from(growth, inflation, cut_growth, cut_inflation):
    good = growth.notna() & inflation.notna()
    high_g, high_i = growth > cut_growth, inflation > cut_inflation
    out = pd.Series(pd.NA, index=growth.index, dtype="string")
    out.loc[good & high_g & ~high_i] = "Expansion"
    out.loc[good & high_g & high_i] = "Overheating"
    out.loc[good & ~high_g & ~high_i] = "Slowdown"
    out.loc[good & ~high_g & high_i] = "Stagflation"
    return out


def fit_tuning(x, fixed_parameters=None):
    train = x.loc[:TRAIN_END].dropna(subset=PREDICTORS + [f + "_next" for f in FACTORS])
    validation = x.loc[TRAIN_END + pd.offsets.MonthEnd():VALID_END].dropna(subset=PREDICTORS + [f + "_next" for f in FACTORS])
    cv = TimeSeriesSplit(n_splits=5)
    params, rows = {}, []
    for factor in FACTORS:
        y = train[factor + "_next"]
        hp = {}
        definitions = [
            ("Ridge", Ridge(), {"model__alpha": np.logspace(-4, 6, 41)}),
            ("Elastic_Net", ElasticNet(max_iter=30000), {"model__alpha": np.logspace(-4, 1, 26),
                                                        "model__l1_ratio": [.1, .5, .9]})]
        for name, estimator, grid in definitions:
            pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
            search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=1)
            search.fit(train[PREDICTORS], y)
            hp[name] = {k.replace("model__", ""): float(v) for k, v in search.best_params_.items()}
            if fixed_parameters is not None:
                # Keep the declared specification when numerically tied CV
                # candidates change argmin order across BLAS/library builds.
                # Selection is checked using initial-training folds only.
                fixed = fixed_parameters[factor][name]
                candidates = search.cv_results_["params"]
                # JSON parsing and np.logspace may differ in the last binary
                # digit across builds. Only parameter matching uses this narrow
                # tolerance; the declared grid and CV-loss check stay fixed.
                selected_index = match_grid_parameters(candidates, fixed)
                selected_mse = -float(search.cv_results_["mean_test_score"][selected_index])
                best_mse = -float(search.best_score_)
                if not np.isclose(selected_mse,best_mse,rtol=1e-9,atol=1e-10):
                    raise RuntimeError(f'Fixed training choice is not CV-equivalent: {factor} {name}; fixed MSE={selected_mse:.17g}, minimum MSE={best_mse:.17g}')
                hp[name] = dict(fixed)
                pipe.set_params(**{'model__'+k:v for k,v in fixed.items()})
                pipe.fit(train[PREDICTORS], y)
                pred = pipe.predict(validation[PREDICTORS])
            else:
                pred = search.predict(validation[PREDICTORS])
            actual = validation[factor + "_next"].values
            base_error = np.mean((actual - y.mean()) ** 2)
            rows.append({"Factor": factor, "Model": name, **hp[name],
                         "CV_MSE": -search.best_score_, "Validation_RMSE": np.sqrt(np.mean((actual - pred) ** 2)),
                         "Validation_OOS_R2_vs_fixed_training_mean": 1 - np.mean((actual - pred) ** 2) / base_error,
                         "N_Train": len(train), "N_Validation": len(validation)})
        params[factor] = hp
    return params, pd.DataFrame(rows)


def scores_to_weights(scores, cap=1.):
    """Nonnegative factor-sleeve scores; redistribute excess proportionally.

    Where positive-score sleeves cannot absorb all mass under the cap,
    redistribute to uncapped zero-score sleeves equally. This is predeclared.
    """
    arr = np.maximum(np.asarray(scores, float), 0.)
    if arr.ndim == 1:
        arr = arr[None, :]
    out = np.zeros_like(arr)
    for k, row in enumerate(arr):
        left, free = 1., np.ones(len(row), dtype=bool)
        while free.any():
            raw = row[free]
            provisional = left * (raw / raw.sum() if raw.sum() > 0 else np.ones(free.sum()) / free.sum())
            over = provisional > cap + 1e-12
            indexes = np.flatnonzero(free)
            if not over.any():
                out[k, indexes] = provisional
                break
            out[k, indexes[over]] = cap
            left -= cap * over.sum()
            free[indexes[over]] = False
    assert np.allclose(out.sum(axis=1), 1)
    assert out.max() <= cap + 1e-9
    return out


def performance(r, gamma=3):
    r = np.asarray(r, float) / 100
    assert np.isfinite(r).all()
    vol = r.std(ddof=1) * np.sqrt(12)
    wealth = np.r_[1., np.cumprod(1 + r)]
    return {"N_Months": len(r), "Annualized_Mean": r.mean() * 12,
            "Synthetic_CAGR": wealth[-1] ** (12 / len(r)) - 1,
            "Annualized_Volatility": vol, "Sharpe": r.mean() * 12 / vol if vol else np.nan,
            "CER_Gamma3": r.mean() * 12 - gamma / 2 * r.var(ddof=1) * 12,
            "Max_Drawdown": np.min(wealth / np.maximum.accumulate(wealth) - 1),
            "Synthetic_Cumulative_Return": wealth[-1] - 1}


def training_diagnostics(x):
    targets = [f + "_next" for f in FACTORS]
    train = x.loc[:TRAIN_END].dropna(subset=PREDICTORS + targets)
    rows = []
    for f in FACTORS:
        fit = sm.OLS(train[f + "_next"], sm.add_constant(train[PREDICTORS])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        for p in PREDICTORS:
            rows.append({"Factor": f, "Predictor": p, "Coefficient": fit.params[p], "HAC_p": fit.pvalues[p],
                         "R_squared": fit.rsquared, "N": len(train)})
    rows = pd.DataFrame(rows)
    rows["Holm_p_42_coefficients"] = multipletests(rows.HAC_p, method="holm")[1]
    adf = []
    for p in PREDICTORS:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='adfuller currently returns a plain tuple.*', category=FutureWarning)
            stat, prob, *_ = adfuller(train[p], autolag="AIC")
        adf.append({"Predictor": p, "ADF_Statistic": stat, "ADF_p": prob, "Train_N": len(train)})
    return rows, pd.DataFrame(adf), train[PREDICTORS].corr()

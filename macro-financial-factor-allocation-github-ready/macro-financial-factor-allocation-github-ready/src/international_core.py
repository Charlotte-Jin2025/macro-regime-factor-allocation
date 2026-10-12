"""International calendar parsing and leakage-free Ridge feature construction."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .us_core import match_grid_parameters
THEMES=["size","value","investment","profitability","momentum"]
MACRO=["growth","inflation","stress"]
DESIGN=MACRO+["state_growth","state_inflation"]
ALPHAS=[.01,.1,1.,10.,100.,1000.,10000.,100000.,1000000.]
MODELS=["Macro","Macro+Regime"]
FILES=["UK_CPI_ONS_D7BT.csv","UK_IP_ONS.csv","UK_CLIFS_ECB.csv","JKP_GBD_monthly_vw.csv", "DE_CPI_Destatis.csv","DE_IP_Destatis.csv","DE_CLIFS_ECB.csv","JKP_DEU_monthly_vw.csv"]

def unique_alpha_index(value, grid=None):
    """Match a recorded alpha to one declared grid point despite final-bit drift."""
    grid = ALPHAS if grid is None else grid
    candidates = [{"model__alpha": alpha} for alpha in grid]
    return match_grid_parameters(candidates, {"alpha": float(value)})

def date(x):
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def monthly(dates, values, name):
    idx = pd.to_datetime(dates, errors="coerce").dt.to_period("M")
    s = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=idx, name=name)
    s = s.loc[~s.index.isna()].sort_index()
    if s.index.has_duplicates:
        raise ValueError(f"Duplicate months in {name}")
    s.index = s.index.to_timestamp("M")
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="ME"))


def load_calendar(inputs, country):
    if country == "UK":
        items = []
        for fn, col, name in [("UK_CPI_ONS_D7BT.csv", "CPI INDEX 00: ALL ITEMS 2015=100", "cpi"),
                              ("UK_IP_ONS.csv", "IOP: B-E: PRODUCTION: CVMSA", "ip")]:
            d = pd.read_csv(inputs / fn, usecols=["Title", col], low_memory=False)
            d = d.loc[d.Title.astype(str).str.fullmatch(r"\d{4} [A-Z]{3}")]
            items.append(monthly(pd.to_datetime(d.Title, format="%Y %b"), d[col], name))
        cpi, ip = items
        jkp = "JKP_GBD_monthly_vw.csv"
    else:
        d = pd.read_csv(inputs / "DE_CPI_Destatis.csv", sep=";", skiprows=6, header=None, engine="python")
        dates = pd.to_datetime(d[0].astype(str) + " " + d[1].astype(str).str.strip(), format="%Y %B", errors="coerce")
        cpi = monthly(dates, d[2], "cpi")
        d = pd.read_csv(inputs / "DE_IP_Destatis.csv", sep=";", skiprows=9, header=None, engine="python")
        d = d.loc[d[0].astype(str).str.strip().eq("Industry")]
        years = pd.to_numeric(d[1], errors="coerce").astype("Int64").astype(str)
        dates = pd.to_datetime(years + " " + d[2].astype(str).str.strip(), format="%Y %B", errors="coerce")
        ip = monthly(dates, d[7], "ip")
        jkp = "JKP_DEU_monthly_vw.csv"
    d = pd.read_csv(inputs / f"{country}_CLIFS_ECB.csv")
    stress_col = next(c for c in d if "Country Level Index" in c)
    stress = monthly(pd.to_datetime(d.DATE), d[stress_col], "stress_level")
    d = pd.read_csv(inputs / jkp)
    d = d.loc[d.name.isin(THEMES)].copy()
    assert d.freq.eq("monthly").all() and d.weighting.eq("vw").all()
    d["date"] = pd.to_datetime(d.date).dt.to_period("M").dt.to_timestamp("M")
    assert not d.duplicated(["date", "name"]).any()
    factors = d.pivot(index="date", columns="name", values="ret")[THEMES]
    calendar = pd.concat([cpi, ip, stress, factors], axis=1, sort=True)
    calendar = calendar.reindex(pd.date_range(calendar.index.min(), calendar.index.max(), freq="ME"))
    calendar.index.name = "signal_date"
    calendar["growth_yoy"] = calendar.ip.pct_change(12, fill_method=None) * 100
    calendar["inflation_yoy"] = calendar.cpi.pct_change(12, fill_method=None) * 100
    for theme in THEMES:
        calendar[theme + "_next"] = calendar[theme].shift(-1)
    calendar["target_date"] = calendar.index + pd.offsets.MonthEnd(1)
    return calendar


def make_design(calendar, lag, smooth=1):
    d = calendar.copy()
    d["growth"] = d.growth_yoy.shift(lag)
    d["inflation"] = d.inflation_yoy.shift(1)
    d["stress"] = d.stress_level.shift(lag)
    d["state_growth"] = d.growth.rolling(smooth, min_periods=smooth).mean()
    d["state_inflation"] = d.inflation.rolling(smooth, min_periods=smooth).mean()
    d["ip_reference_date"] = d.index - pd.offsets.MonthEnd(lag)
    d["stress_reference_date"] = d.index - pd.offsets.MonthEnd(lag)
    d["cpi_reference_date"] = d.index - pd.offsets.MonthEnd(1)
    d["oldest_state_ip_reference"] = d.index - pd.offsets.MonthEnd(lag + smooth - 1)
    d["oldest_state_cpi_reference"] = d.index - pd.offsets.MonthEnd(smooth)
    d["lag_ip_clifs"] = lag
    d["state_smoothing_months"] = smooth
    return d.dropna(subset=DESIGN + [x + "_next" for x in THEMES]).copy()


class DesignFeatures(BaseEstimator, TransformerMixin):
    """Keep macro levels fixed; fit state thresholds inside each training fold."""
    def __init__(self, with_regime=False, thresholds=None):
        self.with_regime = with_regime
        self.thresholds = thresholds

    def fit(self, X, y=None):
        a = np.asarray(X, float)
        self.thresholds_ = np.median(a[:, 3:5], axis=0) if self.thresholds is None else np.asarray(self.thresholds)
        return self

    def transform(self, X):
        a = np.asarray(X, float)
        if not self.with_regime:
            return a[:, :3]
        state = 2 * (a[:, 3] > self.thresholds_[0]).astype(int) + (a[:, 4] > self.thresholds_[1]).astype(int)
        return np.column_stack([a[:, :3], np.eye(4)[state, 1:]])


def pipeline(model, alpha=1.0, thresholds=None):
    return Pipeline([("design", DesignFeatures(model == "Macro+Regime", thresholds)),
                     ("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])

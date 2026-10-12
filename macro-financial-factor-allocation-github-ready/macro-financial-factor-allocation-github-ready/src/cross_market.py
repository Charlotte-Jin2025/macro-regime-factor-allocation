"""Post-US transfer of the frozen allocation mapping to inspected UK/DE forecasts.

The three-predictor Ridge forecasting models were fitted and inspected before
this allocation supplement. They are not the US seven-predictor ensemble, and
JKP factor themes are not the Fama--French factors. Only the allocation mapping,
risk window, target-weight turnover convention and hypothetical sleeve charge
are transferred unchanged. No model, predictor or allocation parameter is
selected here. All outputs are conditional on the fixed forecast paths.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from . import international_core as ic
from . import us_core as uc
from . import us_design as inference

THEMES = ic.THEMES
STRATEGIES = ["Equal_Weight", "Historical_Mean", "Macro", "Macro+Regime"]
PAIRS = [("Macro", "Historical_Mean"), ("Macro+Regime", "Macro")]
ALLOCATION = "LS5_Variance36_Cap40_Shrink50"
COST_BPS = 10.
SEED = 20261009
DEGENERACY_CORRELATION = .9999  # Diagnostic threshold, not a model-selection rule.


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def risk_window(calendar, signal):
    """Exactly 36 natural months through signal; no forward fill/gap compression."""
    signal = pd.Timestamp(signal)
    required = pd.date_range(signal - pd.offsets.MonthEnd(35), signal, freq="ME")
    frame = calendar.reindex(required)[THEMES]
    if len(frame) != 36 or frame.isna().any(axis=None):
        raise ValueError(f"Need 36 complete natural months of risk history at {signal:%Y-%m-%d}")
    variance = frame.var(ddof=1).to_numpy()
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError(f"Nonpositive or nonfinite theme variance at {signal:%Y-%m-%d}")
    return frame, variance


def _weight_rows(calendar, forecasts, country):
    rows, risk_rows = [], []
    source = forecasts.loc[forecasts.country.eq(country)].copy()
    for signal, frame in source.groupby("signal_date", sort=True):
        target = pd.Timestamp(signal) + pd.offsets.MonthEnd(1)
        if not frame.target_date.eq(target).all():
            raise ValueError("All theme/model targets must be exactly the next natural month")
        if len(frame) != 10 or frame.duplicated(["model", "theme"]).any():
            raise ValueError("Each allocation signal needs two models and all five themes")
        macro = frame.loc[frame.model.eq("Macro")].set_index("theme").loc[THEMES]
        state = frame.loc[frame.model.eq("Macro+Regime")].set_index("theme").loc[THEMES]
        if not np.allclose(macro.matched_training_mean_decimal, state.matched_training_mean_decimal,
                           rtol=0., atol=1e-14):
            raise ValueError("Macro and state fits must use identical training-label means")
        if not np.allclose(macro.actual_decimal.to_numpy(float), calendar.loc[target, THEMES].to_numpy(float), rtol=1e-12, atol=1e-14):
            raise ValueError("Forecast actuals differ from raw target-month theme returns")
        history, variance = risk_window(calendar, signal)
        risk_rows.append({"Country": country, "Signal_Date": signal, "Return_Date": target,
            "History_Start": history.index.min(), "History_End": history.index.max(),
            "N_Months": len(history), "Variance_ddof": 1,
            **{theme + "_variance_decimal": float(variance[j]) for j, theme in enumerate(THEMES)}})
        forecasts_by_strategy = {"Historical_Mean": macro.matched_training_mean_decimal.to_numpy(),
                                 "Macro": macro.prediction_decimal.to_numpy(),
                                 "Macro+Regime": state.prediction_decimal.to_numpy()}
        for strategy in STRATEGIES:
            if strategy == "Equal_Weight":
                weights = np.full(5, .2)
            else:
                raw = uc.scores_to_weights(forecasts_by_strategy[strategy] / variance)[0]
                weights = uc.scores_to_weights(.5 * raw + .1, cap=.4)[0]
            rows.append({"Country": country, "Allocation": ALLOCATION, "Strategy": strategy,
                         "Signal_Date": signal, "Return_Date": target,
                         **dict(zip(THEMES, weights))})
    return rows, risk_rows


def _return_rows(calendar, weights, country):
    rows = []
    for strategy, frame in weights.loc[weights.Country.eq(country)].groupby("Strategy", sort=False):
        frame = frame.sort_values("Return_Date")
        dates = pd.DatetimeIndex(frame.Return_Date)
        if not dates.equals(pd.date_range(dates.min(), dates.max(), freq="ME")):
            raise ValueError("Allocation targets must retain every natural month; no gap compression")
        matrix = frame[THEMES].to_numpy()
        actual = calendar.loc[dates, THEMES].to_numpy()
        if not np.isfinite(actual).all():
            raise ValueError("All five target-month theme returns must be available")
        gross = np.sum(matrix * actual, axis=1)
        turnover = np.r_[0., .5 * np.abs(np.diff(matrix, axis=0)).sum(axis=1)]
        charge = turnover * COST_BPS / 10000
        rows.append(pd.DataFrame({"Country": country, "Allocation": ALLOCATION,
            "Strategy": strategy, "Cost_bps": COST_BPS, "Return_Date": dates,
            "Gross_Return_decimal": gross, "Turnover": turnover,
            "Allocation_Cost_decimal": charge, "Net_Return_decimal": gross - charge}))
    return rows


def _pair_statistics(returns, weights, bootstrap_n):
    means, intervals, mdes, path_rows, diagnostics = [], [], [], [], []
    for country, frame in returns.groupby("Country", sort=True):
        panel = frame.pivot(index="Return_Date", columns="Strategy", values="Net_Return_decimal").sort_index()
        wpanel = {strategy: group.set_index("Return_Date")[THEMES].sort_index()
                  for strategy, group in weights.loc[weights.Country.eq(country)].groupby("Strategy")}
        for i, (a, b) in enumerate(PAIRS):
            comparison = a + "_minus_" + b
            tag = {"Country": country, "Comparison": comparison, "Strategy_A": a,
                   "Strategy_B": b, "Allocation": ALLOCATION, "Cost_bps": COST_BPS}
            pair = panel[[a, b]].to_numpy()
            difference = pair[:, 0] - pair[:, 1]
            delta, score = inference.sharpe_if(pair)
            weight_difference = wpanel[a].to_numpy() - wpanel[b].to_numpy()
            half_l1 = .5 * np.abs(weight_difference).sum(axis=1)
            correlation = float(np.corrcoef(pair.T)[0, 1])
            delta_rms = float(np.sqrt(np.mean(difference ** 2)))
            rms_scale = delta_rms / float(np.sqrt(np.mean(pair[:, 1] ** 2)))
            # Report numerical proximity on its actual scale. These descriptive
            # flags are never used to filter data, choose models or alter tests.
            near_degenerate = bool(correlation >= DEGENERACY_CORRELATION)
            diagnostic = tag | {"N_Months": len(panel), "Return_Correlation": correlation,
                "Mean_Half_L1_Weight_Difference": float(half_l1.mean()),
                "Max_Half_L1_Weight_Difference": float(half_l1.max()),
                "Weight_Difference_RMS": float(np.sqrt(np.mean(weight_difference ** 2))),
                "Paired_Return_Difference_RMS_bp": 10000 * delta_rms,
                "Paired_Return_Difference_Max_Abs_bp": 10000 * float(np.max(np.abs(difference))),
                "Paired_Return_RMS_Relative_To_Benchmark": rms_scale,
                "Numerically_Identical_Paths_1e12": bool(np.max(np.abs(difference)) <= 1e-12),
                "Near_Degenerate_Paired_Paths": near_degenerate,
                "Diagnostic_Correlation_Threshold": DEGENERACY_CORRELATION,
                "Caution": "Conditional on inspected forecast paths; near-identical paths compress the effect and its estimated SE/MDE, not evidence of a more powerful economic test"}
            diagnostics.append(diagnostic)
            path_rows.append(pd.DataFrame(tag | {"Return_Date": panel.index,
                "Return_A_decimal": pair[:, 0], "Return_B_decimal": pair[:, 1],
                "Return_Difference_decimal": difference, "Half_L1_Weight_Difference": half_l1}))
            for lag in [3, 6, 12]:
                result = inference.mean_hac(difference, lag, panel.index)
                means.append(tag | result | {"Annualized_Mean_Difference_pp": 1200 * result["Mean"],
                    "Annualized_HAC_SE_pp": 1200 * result["HAC_SE"],
                    "Annualized_CI_Low_pp": 1200 * result["CI_Low"],
                    "Annualized_CI_High_pp": 1200 * result["CI_High"]})
                se = float(inference.calendar_hac_se(score, panel.index, lag))
                critical = float(norm.ppf(.975) + norm.ppf(.8))
                mdes.append(tag | {"N_Months": len(panel), "HAC_Lags": lag,
                    "Observed_Annual_Sharpe_Difference": float(delta),
                    "Actual_Paired_IF_HAC_SE_Annual": se,
                    "Approx_MDE_Annual_Sharpe_80pct": critical * se,
                    "Return_Correlation": correlation, "Near_Degenerate_Paired_Paths": near_degenerate,
                    "Two_Sided_Alpha": .05, "Target_Power": .8, "Normal_Critical_Sum": critical,
                    "Caution": "Approximate normal local-alternative MDE on fixed paths. Very small effects and MDE for near-identical paths do not establish economic sensitivity."})
            for block in [3, 6, 12]:
                seed = SEED + (1000 if country == "DE" else 0) + 100 * i + block
                interval = inference.studentized_sharpe_interval(pair[:, 0], pair[:, 1], panel.index,
                    B=bootstrap_n, block=block, hac=3, seed=seed)
                intervals.append(tag | interval | {"Primary_Block": block == 6,
                    "Near_Degenerate_Paired_Paths": near_degenerate,
                    "Inference_Scope": "Paired fixed-path uncertainty, no forecast refitting or post-selection uncertainty"})
    means = pd.DataFrame(means)
    for _, ix in means.groupby("HAC_Lags").groups.items():
        if len(ix) != 4:
            raise ValueError("Four prespecified international mean comparisons required per HAC lag")
        means.loc[ix, "Holm_p_Four_International_Comparisons"] = multipletests(
            means.loc[ix, "p_Two_Sided"], method="holm")[1]
        means.loc[ix, "Holm_Family"] = "UK and DE x Macro-HM and state-Macro; separately at each HAC lag; excludes US/forecast tests and does not adjust across lags or the prior design search"
    return means, pd.DataFrame(intervals), pd.DataFrame(mdes), pd.concat(path_rows, ignore_index=True), pd.DataFrame(diagnostics)


def run_analysis(base_path=None, bootstrap_n=3000):
    base = Path(base_path or Path(__file__).resolve().parents[1]).resolve()
    if int(bootstrap_n) != 3000:
        raise ValueError("The archived cross-market supplement fixes the bootstrap at 3,000 draws")
    out = base / "results/cross_market"
    out.mkdir(parents=True, exist_ok=True)
    path_file = base / "results/international/forecast_and_coefficient_path.csv"
    forecasts = pd.read_csv(path_file, parse_dates=["signal_date", "target_date"])
    specification = json.loads((base / "specification.json").read_text())
    transfer_file = base / "cross_market_specification.json"
    transfer = json.loads(transfer_file.read_text())
    declared = specification["primary_allocation"]
    if (declared["variance_months"], declared["shrinkage_to_equal_weight"], declared["weight_cap"]) != (36, .5, .4):
        raise ValueError("US frozen allocation declaration differs from the transfer rule")
    if set(declared["factors"]) != {"SMB", "HML", "RMW", "CMA", "MOM"}:
        raise ValueError("US primary declaration is not the five-long-short-factor allocation")
    expected = {"risk_variance_months": 36, "risk_variance_ddof": 1,
        "shrinkage_to_equal_weight": .5, "weight_cap": .4, "allocation_cost_bps": COST_BPS,
        "bootstrap_draws": 3000, "bootstrap_blocks": [3, 6, 12], "primary_bootstrap_block": 6,
        "studentizer_hac_lag": 3, "mean_hac_lags": [3, 6, 12], "seed_base": SEED,
        "diagnostic_near_degeneracy_return_correlation": DEGENERACY_CORRELATION,
        "parameter_reselection_for_transfer": False, "countries": ["UK", "DE"], "themes": THEMES,
        "models": ["Macro", "Macro+Regime"], "forecast_predictor_lags": {"ip_yoy": 2, "clifs": 2, "cpi_yoy": 1}}
    if any(transfer.get(key) != value for key, value in expected.items()):
        raise ValueError("Archived cross-market transfer configuration differs from its frozen implementation")
    weights, risks, return_rows, metrics, audit_rows, state_counts = [], [], [], [], [], []
    for country in ["UK", "DE"]:
        calendar = ic.load_calendar(base / "data/international", country)
        initial_design = ic.make_design(calendar, 2)
        first_signal = forecasts.loc[forecasts.country.eq(country), "signal_date"].min()
        initial_training = initial_design.loc[initial_design.index < first_signal]
        thresholds = initial_training[["state_growth", "state_inflation"]].median()
        selected_signals = forecasts.loc[forecasts.country.eq(country), "signal_date"].drop_duplicates().sort_values()
        for phase, frame in [("Initial_Training", initial_training), ("Allocation_Evaluation", initial_design.loc[selected_signals])]:
            codes = 2 * frame.state_growth.gt(thresholds.state_growth).astype(int) + frame.state_inflation.gt(thresholds.state_inflation).astype(int)
            for code, label in enumerate(["Slowdown", "Stagflation", "Expansion", "Overheating"]):
                state_counts.append({"Country": country, "Phase": phase, "State": label,
                    "State_Code": code, "Count": int(codes.eq(code).sum()), "N_Signals": len(frame),
                    "Initial_Growth_Threshold_pct": float(thresholds.state_growth),
                    "Initial_Inflation_Threshold_pct": float(thresholds.state_inflation),
                    "Reference_Category": code == 0,
                    "Coverage_Scope": "Inherited international fixed-median Ridge state coding; not the US state schedule"})
        calendar[THEMES].to_csv(out / (country.lower() + "_theme_return_calendar.csv"))
        rows, audit = _weight_rows(calendar, forecasts, country)
        country_weights = pd.DataFrame(rows)
        weights.extend(rows); risks.extend(audit)
        returned = _return_rows(calendar, country_weights, country)
        return_rows.extend(returned)
        for frame in returned:
            strategy = frame.Strategy.iloc[0]
            estimates = uc.performance(100 * frame.Net_Return_decimal.to_numpy())
            metrics.append({"Country": country, "Allocation": ALLOCATION, "Strategy": strategy,
                "Cost_bps": COST_BPS, "First_Target": frame.Return_Date.min(),
                "Last_Target": frame.Return_Date.max(), "Mean_Turnover": float(frame.Turnover.mean()),
                **estimates})
        source = forecasts.loc[forecasts.country.eq(country)]
        for model, group in source.groupby("model", sort=False):
            for theme, part in group.groupby("theme", sort=False):
                alpha = part.selected_alpha.unique()
                if len(alpha) != 1:
                    raise ValueError("Historical international penalty must stay fixed through allocation supplement")
                audit_rows.append({"Country": country, "Model": model, "Theme": theme,
                    "Selected_Alpha": alpha[0], "Forecast_Algorithm": "StandardScaler + Ridge",
                    "IP_YoY_Lag_Months": 2, "CLIFS_Lag_Months": 2, "CPI_YoY_Lag_Months": 1,
                    "State_Thresholds": "Initial training medians; fixed OOS",
                    "State_Growth_Threshold_pct": float(thresholds.state_growth),
                    "State_Inflation_Threshold_pct": float(thresholds.state_inflation),
                    "State_Reference_Category": "Low growth / low inflation (Slowdown)",
                    "Penalty_Selection": "Each Macro and Macro+Regime theme model separately selected in initial-training CV; fixed thereafter",
                    "First_Signal": part.signal_date.min(), "Last_Signal": part.signal_date.max(),
                    "First_Target": part.target_date.min(), "Last_Target": part.target_date.max(),
                    "N_Targets": len(part), "Initial_Training_Labels": int(part.sort_values("signal_date").train_n.iloc[0]),
                    "Allocation_Parameter_Reselection": False,
                    "Data_and_Design_Limit": "Inspected current-vintage Ridge forecasts and USD JKP theme returns; no held-out cross-market discovery, no identical US forecasting model"})
        print(f"{country}: transferred frozen allocation to {len(country_weights)//4} targets", flush=True)
    weights = pd.DataFrame(weights)
    returns = pd.concat(return_rows, ignore_index=True)
    metrics = pd.DataFrame(metrics)
    means, intervals, mde, differences, diagnostics = _pair_statistics(returns, weights, int(bootstrap_n))
    summary_rows = []
    for country in ["UK", "DE"]:
        estimates = metrics.loc[metrics.Country.eq(country)].set_index("Strategy")
        row = {"Country": country, "N_Months": int(estimates.loc["Macro", "N_Months"]),
               "First_Target": estimates.loc["Macro", "First_Target"],
               "Last_Target": estimates.loc["Macro", "Last_Target"]}
        for strategy, prefix in [("Equal_Weight", "EW"), ("Historical_Mean", "HM"), ("Macro", "Macro"), ("Macro+Regime", "State")]:
            for col, label in [("Annualized_Mean", "Annual_Mean_pct"), ("Sharpe", "Sharpe"), ("Mean_Turnover", "Mean_Turnover")]:
                row[prefix + "_" + label] = float(estimates.loc[strategy, col]) * (100 if col == "Annualized_Mean" else 1)
        for a, b in PAIRS:
            comp = a + "_minus_" + b
            prefix = "Macro_vs_HM" if a == "Macro" else "State_vs_Macro"
            ci = intervals.loc[intervals.Country.eq(country) & intervals.Comparison.eq(comp) & intervals.Block_Length.eq(6)].iloc[0]
            mean = means.loc[means.Country.eq(country) & means.Comparison.eq(comp) & means.HAC_Lags.eq(3)].iloc[0]
            power = mde.loc[mde.Country.eq(country) & mde.Comparison.eq(comp) & mde.HAC_Lags.eq(3)].iloc[0]
            diag = diagnostics.loc[diagnostics.Country.eq(country) & diagnostics.Comparison.eq(comp)].iloc[0]
            for key in ["Sharpe_Difference", "CI_Low", "CI_High", "Original_HAC_SE"]:
                row[prefix + "_" + key] = float(ci[key])
            for key in ["Annualized_Mean_Difference_pp", "p_Two_Sided", "Holm_p_Four_International_Comparisons"]:
                row[prefix + "_" + key] = float(mean[key])
            row[prefix + "_Approx_MDE"] = float(power.Approx_MDE_Annual_Sharpe_80pct)
            for key in ["Mean_Half_L1_Weight_Difference", "Paired_Return_Difference_RMS_bp", "Return_Correlation", "Near_Degenerate_Paired_Paths"]:
                row[prefix + "_" + key] = diag[key]
        summary_rows.append(row)
    compact = pd.DataFrame(summary_rows)
    input_paths = [base / "data/international" / name for name in ic.FILES] + [path_file, base / "specification.json", transfer_file]
    input_hashes = pd.DataFrame({"Path": [str(path.relative_to(base)) for path in input_paths],
                                "SHA256": [_sha(path) for path in input_paths]})
    tables = {"weights": weights, "returns": returns, "metrics": metrics,
        "risk_timing_audit": pd.DataFrame(risks), "mean_difference_hac": means,
        "sharpe_intervals": intervals, "mde": mde, "paired_path_differences": differences,
        "path_proximity_diagnostics": diagnostics, "forecast_design_audit": pd.DataFrame(audit_rows),
        "allocation_summary": compact, "input_sha256": input_hashes, "state_coverage": pd.DataFrame(state_counts)}
    for name, frame in tables.items():
        frame.to_csv(out / (name + ".csv"), index=False)
    summary = {"Study_Status": "Post-US exploratory allocation transfer using already inspected international forecast models",
        "Frozen_Allocation": {"variance_months": 36, "variance_ddof": 1,
            "variance_units": "Decimal original JKP theme returns", "variance_end": "Signal month, not next-month outcome",
            "negative_scores": "Truncate to zero before normalization", "all_nonpositive": "Equal raw weights",
            "shrinkage_to_equal_weight": .5, "weight_cap": .4, "redistribution": "Same proportional capped simplex as US",
            "allocation_charge_bps": COST_BPS, "turnover": "Half L1 change in target weights, no drift adjustment",
            "initial_entry": "Free at start of each country's observed cohort", "internal_theme_cost": "Not modeled; 10 bp sleeve charge is hypothetical"},
        "Themes": THEMES, "Strategies": STRATEGIES, "Bootstrap_Draws": int(bootstrap_n),
        "Bootstrap_Method": "Same US non-circular overlapping moving-block HAC-studentized percentile-t, blocks 3/6/12; block6 main, HAC3 studentizer",
        "Bootstrap_Seeds": {country: {a + "_minus_" + b: {str(block): SEED + (1000 if country == "DE" else 0) + 100 * i + block for block in [3, 6, 12]}
            for i, (a, b) in enumerate(PAIRS)} for country in ["UK", "DE"]},
        "Holm_Family": "Four international paired mean-return comparisons (2 countries x 2 pairs), separately at each HAC lag; does not include US tests or adjust across lags/full prior search",
        "Sharpe_Intervals": "Marginal conditional fixed-path intervals; no multiple-comparison or forecast/refit/selection uncertainty adjustment",
        "MDE": "(z.975+z.8) x actual paired Sharpe IF-HAC SE; approximate local normal alternative, no calendar-time extrapolation",
        "Data_Limits": ["JKP factor themes differ from US Fama-French and MOM factor constructions.",
            "Country returns are USD denominated; no currency hedge. Different weights may change currency exposure; cancellation is not guaranteed.",
            "International forecasts use three predictors and Ridge; US uses seven predictors and OLS/Ridge/Elastic Net combination.",
            "Macro+Regime and Macro use separately selected initial-training penalties; their difference is an inherited model-design comparison, not an isolated causal effect of state indicators. International state dummies omit Slowdown, whereas US omits Expansion.",
            "Current-vintage inputs and fixed reference lags are pseudo-OOS, not full release-vintage portfolios.",
            "Reported countries, models and windows were previously inspected; extension transfers an inspected US allocation rule and is not a preregistered independent external validation.",
            "No risk-free subtraction or within-theme trading-cost calibration; these are synthetic theme allocations.",
            "Near-identical paths mechanically compress effect, uncertainty and MDE; small MDE is not stronger economic detection."],
        "Output_Rows": {name: len(frame) for name, frame in tables.items()},
        "Results": json.loads(compact.to_json(orient="records", date_format="iso")),
        "Input_SHA256": input_hashes.to_dict("records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(compact[["Country", "N_Months", "Macro_Sharpe", "HM_Sharpe", "Macro_vs_HM_Sharpe_Difference", "State_vs_Macro_Sharpe_Difference"]].to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=None)
    args = parser.parse_args()
    run_analysis(args.base)

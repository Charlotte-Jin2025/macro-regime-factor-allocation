"""Bounded numerical supplements requested by the three report reviews.

Only results/robustness is written. The existing primary forecasts, portfolio
mapping, costs, samples and parameter choices remain untouched. The one new
path substitutes historical IP inputs while retaining the current-vintage
initial-training penalties. It decomposes an existing sensitivity; it is not
a newly selected primary strategy, a causal experiment or a real-time panel.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from . import us_core as u
from . import us_design as design_tools
from . import us_analysis as us


CURRENT = "Release_corrected_current_vintage"
IP_RETUNED = "Partial_realtime_IP"
IP_FIXED = "Partial_realtime_IP_Current_Penalties"
PRIMARY = "LS5_Variance36_Cap40_Shrink50"
DESIGNS = (CURRENT, IP_FIXED, IP_RETUNED)
DESIGN_LABELS = {
    CURRENT: "Current IP / current-training penalties",
    IP_FIXED: "Historical IP / current-training penalties",
    IP_RETUNED: "Historical IP / separately retuned penalties",
}
MEAN_FAMILY = ("Three Macro-minus-historical-mean design contrasts: current IP, "
               "historical IP with current penalties, historical IP retuned; "
               "separate family within each cohort and HAC lag; no search-wide correction")
ARCHIVED_MEAN_FAMILY = ("Archived three portfolio contrasts within the same design, "
                        "cohort and HAC lag: Macro-minus-historical-mean, "
                        "fixed-state-minus-Macro, expanding-state-minus-Macro")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pair(returns, design, cohort, cost=10):
    """Pair actual observed returns by date, never by row position."""
    subset = returns.loc[(returns.Design == design) &
                         (returns.Cohort == cohort) &
                         (returns.Allocation == PRIMARY) &
                         (returns.Cost_bps == cost) &
                         returns.Strategy.isin(["Macro", "Historical_Mean"])]
    if subset.duplicated(["Return_Date", "Strategy"]).any():
        raise ValueError("Duplicate return dates within a strategy")
    pair = subset.pivot(index="Return_Date", columns="Strategy",
                        values="Net_Return_pct").sort_index()
    if len(pair) != int(cohort.split("_")[-1]) or pair.isna().any(axis=None):
        raise ValueError("The paired primary cohort must contain every target month")
    return pair[["Macro", "Historical_Mean"]]


def _annual_accounting(returns):
    annual, summary = [], []
    for cohort in us.COHORTS:
        pair = _pair(returns, CURRENT, cohort)
        for year, frame in pair.groupby(pair.index.year):
            macro = np.prod(1 + frame.Macro.to_numpy() / 100) - 1
            mean = np.prod(1 + frame.Historical_Mean.to_numpy() / 100) - 1
            annual.append({"Design": CURRENT, "Cohort": cohort,
                "Allocation": PRIMARY, "Cost_bps": 10, "Year": int(year),
                "N_Months": len(frame), "Target_Start": frame.index.min(),
                "Target_End": frame.index.max(), "Full_Calendar_Year": len(frame) == 12,
                "Macro_Compounded_Return_pct": 100 * macro,
                "Historical_Mean_Compounded_Return_pct": 100 * mean,
                "Compound_Return_Difference_pp": 100 * (macro - mean),
                "Macro_Arithmetic_Sum_pp": frame.Macro.sum(),
                "Historical_Mean_Arithmetic_Sum_pp": frame.Historical_Mean.sum(),
                "Macro_minus_HM_Arithmetic_Sum_pp": (frame.Macro - frame.Historical_Mean).sum(),
                "Increment_Unit": "Sum of paired monthly arithmetic return differences in pp; not a compound-wealth contribution"})
        masks = {"Full": np.ones(len(pair), dtype=bool),
                 "Only_2022": pair.index.year == 2022,
                 "Exclude_2022": pair.index.year != 2022}
        full_increment = (pair.Macro - pair.Historical_Mean).sum()
        for name, mask in masks.items():
            frame = pair.loc[mask]
            incremental = (frame.Macro - frame.Historical_Mean).sum()
            summary.append({"Design": CURRENT, "Cohort": cohort,
                "Allocation": PRIMARY, "Cost_bps": 10, "Sample": name,
                "N_Months": len(frame),
                "Macro_Compounded_Return_pct": 100 * (np.prod(1 + frame.Macro / 100) - 1),
                "Historical_Mean_Compounded_Return_pct": 100 * (np.prod(1 + frame.Historical_Mean / 100) - 1),
                "Macro_Annual_Arithmetic_Mean_pp": 12 * frame.Macro.mean(),
                "Historical_Mean_Annual_Arithmetic_Mean_pp": 12 * frame.Historical_Mean.mean(),
                "Macro_minus_HM_Arithmetic_Sum_pp": incremental,
                "Macro_minus_HM_Annual_Arithmetic_Mean_pp": 12 * incremental / len(frame),
                "Share_of_Full_Arithmetic_Increment_pct": 100 * incremental / full_increment if full_increment else np.nan,
                "Path_Rule": "Remove outcomes only; retained original weights and allocation charges unchanged; separately compound the retained months"})
        record = [x for x in annual if x["Cohort"] == cohort]
        for strategy, key in [("Macro", "Macro_Compounded_Return_pct"),
                              ("Historical_Mean", "Historical_Mean_Compounded_Return_pct")]:
            reconstructed = np.prod([1 + x[key] / 100 for x in record])
            direct = np.prod(1 + pair[strategy].to_numpy() / 100)
            if not np.isclose(reconstructed, direct, rtol=1e-12, atol=1e-12):
                raise AssertionError("Calendar-year compound accounting does not reconcile")
        if not np.isclose(sum(x["Macro_minus_HM_Arithmetic_Sum_pp"] for x in record),
                          full_increment, atol=1e-12):
            raise AssertionError("Annual incremental accounting does not reconcile")
    return pd.DataFrame(annual), pd.DataFrame(summary)


def _fixed_ip_forecasts(base, data, original_forecasts):
    """Fit continuous models only; no new tuning and no shared API mutation."""
    calendar = pd.read_csv(base / "results/design" /
                          (IP_RETUNED + "_model_calendar.csv"),
                          index_col=0, parse_dates=True)
    parameters, _ = design_tools.frozen_parameters(base, CURRENT)
    retuned_parameters, _ = design_tools.frozen_parameters(base, IP_RETUNED)
    target_cols = [factor + "_next" for factor in u.FACTORS]
    component_rows, training_rows, parameter_rows = [], [], []
    for factor in u.FACTORS:
        for model in ["Ridge", "Elastic_Net"]:
            frozen = parameters[factor][model]
            retuned = retuned_parameters[factor][model]
            parameter_rows.append({"Design": IP_FIXED, "Factor": factor, "Model": model,
                "Applied_Alpha": frozen["alpha"],
                "Applied_l1_ratio": frozen.get("l1_ratio", np.nan),
                "Historical_IP_Retuned_Alpha": retuned["alpha"],
                "Historical_IP_Retuned_l1_ratio": retuned.get("l1_ratio", np.nan),
                "Retuning_Changes_Parameters": frozen != retuned,
                "Parameter_Source": "results/design/" + CURRENT + "_tuning.csv",
                "Selection_Sample": "Initial 261 current-vintage training signals only; no supplementary cross-validation"})
    for date in pd.date_range(u.TRAIN_END + pd.offsets.MonthEnd(),
                              u.SIGNAL_END, freq="ME"):
        target_date = date + pd.offsets.MonthEnd()
        current = calendar.loc[[date]]
        train = calendar.loc[calendar.index < date].dropna(subset=u.PREDICTORS + target_cols)
        history = data.loc[u.START + pd.offsets.MonthEnd():date, u.FACTORS]
        max_label_date = train.index.max() + pd.offsets.MonthEnd()
        if max_label_date > date or history.index.max() > date:
            raise AssertionError("Training returns must have completed by the signal")
        missing = bool(current[u.PREDICTORS].isna().any(axis=None))
        training_rows.append({"Design": IP_FIXED, "Signal_Date": date,
            "Return_Date": target_date, "N_Train": len(train),
            "Training_Label_Hash": us.training_hash(train.index),
            "Max_Training_Label_Date": max_label_date,
            "Historical_Return_End": history.index.max(),
            "Macro_Fallback": missing,
            "Calendar_Source": "results/design/" + IP_RETUNED + "_model_calendar.csv"})
        for factor in u.FACTORS:
            row = {"Design": IP_FIXED, "Signal_Date": date,
                "Return_Date": target_date, "Factor": factor,
                "Actual": float(data.loc[target_date, factor]),
                "Historical_Mean": float(history[factor].mean()),
                "Segment": "Validation_87" if date <= u.VALID_END else "Original_88",
                "N_Train": len(train), "Macro_N_Train": len(train),
                "Macro_Fallback": missing,
                "Nested_Mean_OLS": float(train[factor + "_next"].mean())}
            for name, estimator in [
                ("OLS", LinearRegression()),
                ("Ridge", Ridge(**parameters[factor]["Ridge"])),
                ("Elastic_Net", ElasticNet(**parameters[factor]["Elastic_Net"], max_iter=30000)),
            ]:
                pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
                pipe.fit(train[u.PREDICTORS], train[factor + "_next"])
                row["Macro_" + name] = (row["Historical_Mean"] if missing else
                    float(pipe.predict(current[u.PREDICTORS])[0]))
            row["Macro"] = np.mean([row["Macro_" + name]
                                      for name in ["OLS", "Ridge", "Elastic_Net"]])
            component_rows.append(row)
    forecasts = pd.DataFrame(component_rows)
    keys = ["Signal_Date", "Return_Date", "Factor"]
    original = original_forecasts.loc[original_forecasts.Design == IP_RETUNED]
    merged = forecasts.merge(original[keys + ["Actual", "Historical_Mean", "N_Train",
        "Macro_Fallback", "Macro_OLS", "Nested_Mean_OLS"]], on=keys,
        suffixes=("_Fixed", "_Retuned"), validate="one_to_one")
    for col in ["Actual", "Historical_Mean", "N_Train", "Macro_OLS", "Nested_Mean_OLS"]:
        if not np.allclose(merged[col + "_Fixed"], merged[col + "_Retuned"],
                           rtol=1e-10, atol=1e-11):
            raise AssertionError("Fixed-IP path changed a non-penalty component: " + col)
    if not merged.Macro_Fallback_Fixed.equals(merged.Macro_Fallback_Retuned):
        raise AssertionError("IP fixed-penalty path changed fallback dates")
    audit = pd.DataFrame(training_rows)
    original_timing = pd.read_csv(base / "results/us/training_timing_audit.csv",
        parse_dates=["Signal_Date", "Return_Date"])
    original_timing = original_timing.loc[original_timing.Design == IP_RETUNED]
    labels = audit.merge(original_timing[["Signal_Date", "Return_Date", "Macro_Training_Hash"]],
                         on=["Signal_Date", "Return_Date"], validate="one_to_one")
    if not labels.Training_Label_Hash.equals(labels.Macro_Training_Hash):
        raise AssertionError("Fixed-IP supplement changed eligible training labels")
    audit["Matches_Retuned_IP_Training_Label_Hash"] = True
    # Same eligible data imply identical OLS forecasts; Ridge/EN alone may change.
    return forecasts, audit, pd.DataFrame(parameter_rows)


def _primary_allocation(data, forecasts):
    weight_rows = []
    for date, group in forecasts.groupby("Signal_Date"):
        frame = group.set_index("Factor").loc[u.FACTORS]
        variance = (data.loc[:date, u.FACTORS].tail(36) / 100).var(ddof=1).to_numpy()
        for strategy in ["Historical_Mean", "Macro"]:
            score = frame[strategy].to_numpy()[1:] / 100 / variance[1:]
            raw = u.scores_to_weights(score)[0]
            weights = u.scores_to_weights(.5 * raw + .5 / 5, cap=.4)[0]
            full = np.r_[0., weights]
            if not np.isclose(full.sum(), 1.) or weights.min() < .1 - 1e-10:
                raise AssertionError("Primary sleeve weights violate the mapping")
            weight_rows.append({"Design": IP_FIXED, "Allocation": PRIMARY,
                "Strategy": strategy, "Signal_Date": date,
                "Return_Date": date + pd.offsets.MonthEnd(),
                **dict(zip(u.FACTORS, full))})
    weights = pd.DataFrame(weight_rows)
    actual = forecasts.pivot(index="Return_Date", columns="Factor",
                             values="Actual")[u.FACTORS]
    returns = []
    for cohort, (start, end) in us.COHORTS.items():
        for strategy in ["Historical_Mean", "Macro"]:
            frame = weights.loc[(weights.Strategy == strategy) &
                                weights.Return_Date.between(start, end)].sort_values("Return_Date")
            matrix = frame[u.FACTORS].to_numpy()
            gross = (matrix * actual.loc[frame.Return_Date].to_numpy()).sum(axis=1)
            turnover = np.r_[0., .5 * np.abs(np.diff(matrix, axis=0)).sum(axis=1)]
            for cost in [0, 10, 25, 50]:
                returns.append(pd.DataFrame({"Design": IP_FIXED, "Cohort": cohort,
                    "Allocation": PRIMARY, "Strategy": strategy, "Cost_bps": cost,
                    "Return_Date": frame.Return_Date.to_numpy(), "Gross_Return_pct": gross,
                    "Turnover": turnover, "Allocation_Cost_pct": turnover * cost / 100,
                    "Net_Return_pct": gross - turnover * cost / 100}))
    return weights, pd.concat(returns, ignore_index=True)


def _independent_component_check(base, forecasts):
    """Check selected component fits with direct matrix algebra, no sklearn."""
    calendar = pd.read_csv(base / "results/design" /
        (IP_RETUNED + "_model_calendar.csv"), index_col=0, parse_dates=True)
    tuning = pd.read_csv(base / "results/design" / (CURRENT + "_tuning.csv"))
    dates = [u.TRAIN_END + pd.offsets.MonthEnd(), pd.Timestamp("2020-07-31"), u.SIGNAL_END]
    targets = [factor + "_next" for factor in u.FACTORS]
    max_ols, max_ridge = 0., 0.
    for date in dates:
        train = calendar.loc[calendar.index < date].dropna(subset=u.PREDICTORS + targets)
        raw = train[u.PREDICTORS].to_numpy()
        center, scale = raw.mean(axis=0), raw.std(axis=0, ddof=0)
        z = (raw - center) / scale
        current = (calendar.loc[date, u.PREDICTORS].to_numpy(float) - center) / scale
        augmented = np.column_stack([np.ones(len(z)), z])
        for factor in u.FACTORS:
            y = train[factor + "_next"].to_numpy()
            reference = forecasts.loc[(forecasts.Signal_Date == date) &
                                      (forecasts.Factor == factor)].iloc[0]
            alpha = float(tuning.loc[(tuning.Factor == factor) &
                                    (tuning.Model == "Ridge"), "alpha"].iloc[0])
            ridge = y.mean() + current @ np.linalg.solve(
                z.T @ z + alpha * np.eye(len(u.PREDICTORS)), z.T @ (y - y.mean()))
            coefficient = np.linalg.lstsq(augmented, y, rcond=None)[0]
            ols = coefficient[0] + current @ coefficient[1:]
            max_ols = max(max_ols, abs(ols - reference.Macro_OLS))
            max_ridge = max(max_ridge, abs(ridge - reference.Macro_Ridge))
    if max_ols > 1e-10 or max_ridge > 1e-10:
        raise AssertionError("Direct algebra does not reproduce fixed-IP OLS/Ridge fits")
    fallback = forecasts.loc[forecasts.Macro_Fallback]
    expected = np.repeat(fallback.Historical_Mean.to_numpy()[:, None], 4, axis=1)
    if not np.allclose(fallback[["Macro_OLS", "Macro_Ridge", "Macro_Elastic_Net", "Macro"]], expected):
        raise AssertionError("Missing predictors do not generate the declared mean fallback")
    return {"method": "Direct augmented-design least squares and centered Ridge normal equations; no sklearn fit or shared fitting helper",
            "signal_dates": [date.date().isoformat() for date in dates],
            "factor_pairs_checked": len(dates) * len(u.FACTORS),
            "OLS_max_abs_prediction_error_pp": float(max_ols),
            "Ridge_max_abs_prediction_error_pp": float(max_ridge),
            "fallback_factor_rows": len(fallback),
            "fallback_target_dates": sorted(fallback.Return_Date.dt.strftime("%Y-%m-%d").unique().tolist())}


def _ip_design_comparisons(returns, archived_hac, bootstrap_n):
    metrics, mean_tests, intervals, sequential = [], [], [], []
    for cohort in us.COHORTS:
        pairs = {}
        for design in DESIGNS:
            pairs[design] = _pair(returns, design, cohort)
            for cost in [0, 10, 25, 50]:
                pair = _pair(returns, design, cohort, cost)
                for strategy in ["Historical_Mean", "Macro"]:
                    metrics.append({"Design": design, "Design_Label": DESIGN_LABELS[design],
                        "Cohort": cohort, "Allocation": PRIMARY, "Cost_bps": cost,
                        "Strategy": strategy, "Sample": "Full", **u.performance(pair[strategy])})
            pair = pairs[design]
            delta = pair.Macro - pair.Historical_Mean
            for lag in [3, 6, 12]:
                result = design_tools.mean_hac(delta.to_numpy(), lag, pair.index)
                source = archived_hac.loc[(archived_hac.Design == design) &
                    (archived_hac.Cohort == cohort) & (archived_hac.Allocation == PRIMARY) &
                    (archived_hac.Sample == "Full") & (archived_hac.Cost_bps == 10) &
                    (archived_hac.Comparison == "Macro_minus_Historical_Mean") &
                    (archived_hac.HAC_Lags == lag)]
                source_holm = (float(source.iloc[0].Holm_p_Within_Displayed_Comparison_Family)
                               if len(source) == 1 else np.nan)
                if design != IP_FIXED:
                    if len(source) != 1 or not np.isclose(result["p_Two_Sided"],
                        float(source.iloc[0].p_Two_Sided), rtol=1e-9, atol=1e-10):
                        raise AssertionError("Supplement mean test does not reconcile with archived main path")
                mean_tests.append({"Design": design, "Design_Label": DESIGN_LABELS[design],
                    "Cohort": cohort, "Allocation": PRIMARY, "Cost_bps": 10,
                    "Comparison": "Macro_minus_Historical_Mean", **result,
                    "Annual_Mean_Gain_pp": 12 * result["Mean"],
                    "Annual_CI_Low_pp": 12 * result["CI_Low"],
                    "Annual_CI_High_pp": 12 * result["CI_High"],
                    "Archived_Within_Design_Three_Comparison_Holm_p": source_holm,
                    "Archived_Family_Definition": ARCHIVED_MEAN_FAMILY if len(source) else "Not defined for the continuous-only counterfactual",
                    "Supplement_Family_Definition": MEAN_FAMILY})
            seed = us.SEED + (1000 if cohort == "Extended_175" else 0)
            interval = design_tools.studentized_sharpe_interval(
                pair.Macro.to_numpy() / 100, pair.Historical_Mean.to_numpy() / 100,
                pair.index, B=bootstrap_n, block=6, hac=3, seed=seed)
            intervals.append({"Design": design, "Design_Label": DESIGN_LABELS[design],
                "Cohort": cohort, "Allocation": PRIMARY, "Cost_bps": 10,
                "Comparison": "Macro_minus_Historical_Mean", **interval,
                "Inference_Scope": "Pointwise conditional paired fixed-path interval; no retuning, selection or refit uncertainty"})
        for left, right, label in [
            (IP_FIXED, CURRENT, "Historical_IP_substitution_with_current_penalties"),
            (IP_RETUNED, IP_FIXED, "Retuning_on_historical_IP_given_the_substitution"),
            (IP_RETUNED, CURRENT, "Combined_historical_IP_and_retuning"),
        ]:
            a, b = pairs[left], pairs[right]
            if not a.index.equals(b.index) or not np.allclose(a.Historical_Mean, b.Historical_Mean,
                rtol=1e-10, atol=1e-11):
                raise AssertionError("IP decomposition requires identical paired targets and historical means")
            sequential.append({"Cohort": cohort, "Step": label,
                "First_Design": left, "Second_Design": right, "N_Months": len(a),
                "Annual_Arithmetic_Increment_Change_pp": 12 * (a.Macro - b.Macro).mean(),
                "Sharpe_Increment_Change": (u.performance(a.Macro)["Sharpe"] - u.performance(a.Historical_Mean)["Sharpe"]) -
                                           (u.performance(b.Macro)["Sharpe"] - u.performance(b.Historical_Mean)["Sharpe"]),
                "Interpretation": "Sequential computational accounting on fixed data and settings; order-specific, not causal attribution or a new independent test"})
    tests = pd.DataFrame(mean_tests)
    tests["Holm_p_Three_Design_Contrasts"] = np.nan
    for _, indexes in tests.groupby(["Cohort", "HAC_Lags"]).groups.items():
        tests.loc[indexes, "Holm_p_Three_Design_Contrasts"] = multipletests(
            tests.loc[indexes, "p_Two_Sided"], method="holm")[1]
    return pd.DataFrame(metrics), tests, pd.DataFrame(intervals), pd.DataFrame(sequential)


def _training_slopes(base):
    slopes = pd.read_csv(base / "results/training/macro_regression.csv")
    if len(slopes) != 42 or slopes.duplicated(["Factor", "Predictor"]).any():
        raise ValueError("Expected exactly six factors times seven initial-training slopes")
    correction = multipletests(slopes.HAC_p, method="holm")[1]
    if not np.allclose(correction, slopes.Holm_p_42_coefficients, atol=1e-12):
        raise AssertionError("Training-slope Holm family does not reconcile")
    slopes["Training_Signal_Start"] = u.START
    slopes["Training_Signal_End"] = u.TRAIN_END
    slopes["HAC_Lags"] = 3
    slopes["Family_Definition"] = "All 42 OLS slope coefficients: six factors times seven predictors, initial 261 signals, two-sided HAC3 p values; intercepts excluded"
    slopes["Coefficient_Unit"] = slopes.Predictor.map({
        "VIX": "Next-month factor return pp per one VIX index point",
        "Credit_Spread": "Next-month factor return pp per spread pp",
        "Term_Spread": "Next-month factor return pp per spread pp",
        "Treasury_2Y_change": "Next-month factor return pp per yield-change pp",
        "Inflation_lag": "Next-month factor return pp per monthly inflation pp",
        "Unemployment_lag": "Next-month factor return pp per unemployment-rate pp",
        "IP_Growth_lag": "Next-month factor return pp per monthly IP-growth pp",
    })
    if slopes.Coefficient_Unit.isna().any():
        raise ValueError("Unrecognized predictor unit")
    slopes["Interpretation"] = "Initial-sample conditional predictive association, not causal channel evidence or out-of-sample skill; coefficient p values do not determine CV penalty selection"
    return slopes


def _rmw_state_means(base):
    calendar = pd.read_csv(base / "results/states/model_calendar.csv", parse_dates=["Signal_Date"])
    calendar = calendar.loc[calendar.Design == CURRENT].set_index("Signal_Date").sort_index()
    splits = [("Training_261", u.START, u.TRAIN_END),
              ("Former_validation_87", u.TRAIN_END + pd.offsets.MonthEnd(), u.VALID_END),
              ("Original_88", u.VALID_END + pd.offsets.MonthEnd(), u.SIGNAL_END),
              ("Extended_175", u.TRAIN_END + pd.offsets.MonthEnd(), u.SIGNAL_END)]
    rows = []
    for variant, column in [("FixedYoY", "Regime_YoY"),
                            ("ExpandingYoY", "Regime_ExpandingYoY")]:
        for split, start, end in splits:
            frame = calendar.loc[start:end]
            n_valid = int(frame[column].notna().sum())
            for state in u.REGIMES:
                group = frame.loc[frame[column].eq(state)]
                outcomes = group.RMW_next.dropna()
                rows.append({"Design": CURRENT, "Variant": variant, "Split": split,
                    "Regime": state, "Signal_Start": start, "Signal_End": end,
                    "Target_Start": start + pd.offsets.MonthEnd(),
                    "Target_End": end + pd.offsets.MonthEnd(),
                    "N_All_Signals": len(frame), "N_Valid_States": n_valid,
                    "N_Missing_States": len(frame) - n_valid,
                    "N_Observed_State": len(group), "N_RMW_Outcomes": len(outcomes),
                    "N_Continuous_Model_Eligible_In_State": int(group[u.PREDICTORS].notna().all(axis=1).sum()),
                    "RMW_Next_Month_Mean_pct": outcomes.mean(),
                    "RMW_Next_Month_SD_pct": outcomes.std(ddof=1),
                    "RMW_Next_Month_Median_pct": outcomes.median(),
                    "Count_Unit": "Unique monthly signal labels; includes a known state even if another continuous predictor is unavailable",
                    "Interpretation": "Observed subsequent operating-profitability factor returns; descriptive, not a causal defensive mechanism or a balanced original four-state comparison"})
            if sum(row["N_Observed_State"] for row in rows if row["Variant"] == variant and
                   row["Split"] == split) != n_valid:
                raise AssertionError("State count accounting does not reconcile")
    return pd.DataFrame(rows)


def _existing_summaries(base):
    mapping = pd.read_csv(base / "results/selection/mapping_metrics.csv")
    comparisons = pd.read_csv(base / "results/selection/mapping_comparisons.csv")
    metric_columns = ["Cohort", "Mapping", "Strategy", "Mean_Turnover"]
    turnover = mapping.loc[mapping.Strategy.isin(["Macro", "Historical_Mean"]), metric_columns]
    wide = turnover.pivot(index=["Cohort", "Mapping"], columns="Strategy", values="Mean_Turnover")
    wide = wide.rename(columns={"Macro": "Macro_Mean_Monthly_One_Way_Turnover",
                               "Historical_Mean": "HM_Mean_Monthly_One_Way_Turnover"}).reset_index()
    mapping_summary = comparisons.merge(wide, on=["Cohort", "Mapping"], validate="one_to_one")
    mapping_summary["Mapping_Comparison_Scope"] = "Inspected documented mappings; scaling, shrinkage and cap can change together; no attribution to an individual constraint"
    crossings = pd.read_csv(base / "results/diagnostics/cost_crossings.csv")
    crossings = crossings.loc[(crossings.Design == CURRENT) &
        (crossings.Allocation == PRIMARY) & (crossings.Sample == "Full") &
        (crossings.Comparison == "Macro_minus_Historical_Mean") & (crossings.Gamma == 0)].copy()
    crossings["Estimand"] = "Zero crossing of frozen-path arithmetic mean increment; not a Sharpe crossing, significance threshold or calibrated actual cost"
    influence = pd.read_csv(base / "results/diagnostics/leave_one_month_influence.csv")
    influence = influence.loc[(influence.Design == CURRENT) &
        (influence.Allocation == PRIMARY) & (influence.Sample == "Full") &
        (influence.Comparison == "Macro_minus_Historical_Mean") & (influence.Cost_bps == 10)]
    extremes = []
    for cohort, group in influence.groupby("Cohort"):
        # Report both directions, as well as the two review-referenced months.
        selections = [group.loc[[group.Deleted_Sharpe_Difference.idxmin()]].assign(Selection="Minimum_deleted_delta_Sharpe"),
                      group.loc[[group.Deleted_Sharpe_Difference.idxmax()]].assign(Selection="Maximum_deleted_delta_Sharpe")]
        for date in ["2020-08-31", "2026-07-31"]:
            selections.append(group.loc[group.Deleted_Return_Date == date].assign(Selection="Review_referenced_month"))
        extremes.extend(selections)
    return mapping_summary, crossings, pd.concat(extremes, ignore_index=True)


def run_analysis(base_path, bootstrap_n=3000):
    """Build the supplement after the existing US/state/diagnostic stages."""
    base = Path(base_path).resolve()
    out = base / "results/robustness"
    out.mkdir(parents=True, exist_ok=True)
    bootstrap_n = int(bootstrap_n)
    if bootstrap_n != 3000:
        raise ValueError("This archived supplement fixes the Sharpe bootstrap at 3,000 draws")
    data, _ = u.load_data(base)
    original_forecasts = pd.read_csv(base / "results/us/forecasts.csv", parse_dates=["Signal_Date", "Return_Date"])
    original_returns = pd.read_csv(base / "results/us/returns.csv", parse_dates=["Return_Date"])
    original_weights = pd.read_csv(base / "results/us/weights.csv", parse_dates=["Signal_Date", "Return_Date"])
    archived_hac = pd.read_csv(base / "results/us/mean_difference_hac.csv")
    annual, concentration = _annual_accounting(original_returns)
    fixed_forecasts, training_audit, parameter_audit = _fixed_ip_forecasts(base, data, original_forecasts)
    independent_components = _independent_component_check(base, fixed_forecasts)
    fixed_weights, fixed_returns = _primary_allocation(data, fixed_forecasts)
    # Historical means, risks, costs and initial-cohort turnover are unchanged.
    for cohort in us.COHORTS:
        for cost in [0, 10, 25, 50]:
            original_pair = _pair(original_returns, IP_RETUNED, cohort, cost)
            fixed_pair = _pair(fixed_returns, IP_FIXED, cohort, cost)
            if not np.allclose(original_pair.Historical_Mean, fixed_pair.Historical_Mean,
                               rtol=1e-10, atol=1e-11):
                raise AssertionError("Fixed-IP supplement changed the historical-mean return path")
    hm_original = original_weights.loc[(original_weights.Design == IP_RETUNED) &
        (original_weights.Allocation == PRIMARY) & (original_weights.Strategy == "Historical_Mean")].sort_values("Return_Date")
    hm_fixed = fixed_weights.loc[fixed_weights.Strategy == "Historical_Mean"].sort_values("Return_Date")
    if not np.allclose(hm_original[u.FACTORS], hm_fixed[u.FACTORS], rtol=1e-10, atol=1e-11):
        raise AssertionError("Fixed-IP supplement changed historical-mean weights")
    returns = pd.concat([original_returns, fixed_returns], ignore_index=True)
    metrics, mean_tests, intervals, sequential = _ip_design_comparisons(returns, archived_hac, bootstrap_n)
    original_intervals = pd.read_csv(base / "results/us/studentized_intervals.csv")
    for _, interval in intervals.loc[intervals.Design != IP_FIXED].iterrows():
        reference = original_intervals.loc[(original_intervals.Design == interval.Design) &
            (original_intervals.Cohort == interval.Cohort) & (original_intervals.Allocation == PRIMARY) &
            (original_intervals.Sample == "Full") & (original_intervals.Comparison == "Macro_minus_Historical_Mean") &
            (original_intervals.Cost_bps == 10) & (original_intervals.Block_Length == 6) &
            (original_intervals.HAC_Lags == 3)]
        if len(reference) != 1 or not np.allclose(
            interval[["Sharpe_Difference", "CI_Low", "CI_High"]].astype(float),
            reference.iloc[0][["Sharpe_Difference", "CI_Low", "CI_High"]].astype(float),
            rtol=1e-9, atol=1e-10):
            raise AssertionError("Supplement Sharpe interval does not reconcile with archived main path")
    slopes = _training_slopes(base)
    state_means = _rmw_state_means(base)
    mapping, crossings, influence = _existing_summaries(base)
    frames = {
        "annual_incremental_returns.csv": annual,
        "annual_concentration_summary.csv": concentration,
        "ip_fixed_penalty_forecasts.csv": fixed_forecasts,
        "ip_fixed_penalty_training_audit.csv": training_audit,
        "ip_parameter_audit.csv": parameter_audit,
        "ip_fixed_penalty_weights.csv": fixed_weights,
        "ip_fixed_penalty_returns.csv": fixed_returns,
        "ip_design_metrics.csv": metrics,
        "ip_design_mean_hac.csv": mean_tests,
        "ip_design_sharpe_intervals.csv": intervals,
        "ip_sequential_decomposition.csv": sequential,
        "training_slopes.csv": slopes,
        "rmw_state_means.csv": state_means,
        "mapping_turnover.csv": mapping,
        "cost_crossings.csv": crossings,
        "month_influence_extremes.csv": influence,
    }
    for name, frame in frames.items():
        frame.to_csv(out / name, index=False)
    input_paths = ["results/us/forecasts.csv", "results/us/returns.csv", "results/us/weights.csv",
        "results/us/training_timing_audit.csv", "results/us/studentized_intervals.csv",
        "results/us/mean_difference_hac.csv", "results/design/" + CURRENT + "_tuning.csv",
        "results/design/" + IP_RETUNED + "_tuning.csv",
        "results/design/" + IP_RETUNED + "_model_calendar.csv", "results/states/model_calendar.csv",
        "results/training/macro_regression.csv", "results/selection/mapping_metrics.csv",
        "results/selection/mapping_comparisons.csv", "results/diagnostics/cost_crossings.csv",
        "results/diagnostics/leave_one_month_influence.csv"]
    summary = {"scope": "Bounded numerical supplements; original primary outputs unchanged",
        "new_primary_or_tuning_search": False, "ip_counterfactual_design": IP_FIXED,
        "ip_counterfactual_scope": "Historical-IP features and eligible labels from the existing IP calendar; current-vintage initial-training penalties and mixing fixed; refit scalers and coefficients recursively; other current vintages, factor returns, fallback, mean benchmark, risks, mapping and costs unchanged",
        "causal_or_fully_realtime_claim": False,
        "mean_holm_family": MEAN_FAMILY,
        "archived_mean_holm_family": ARCHIVED_MEAN_FAMILY,
        "sharpe_inference": "3,000 paired six-month moving-block percentile-t draws with HAC3 studentization; generated paths fixed; pointwise conditional intervals",
        "bootstrap_n": bootstrap_n,
        "bootstrap_seeds": {"Original_88": us.SEED, "Extended_175": us.SEED + 1000},
        "csv_count": len(frames),
        "csv_schema": {name: {"rows": len(frame), "columns": list(frame.columns)} for name, frame in frames.items()},
        "input_sha256": {name: _sha256(base / name) for name in input_paths},
        "checks": {"paired_months_complete": True, "annual_compound_and_increment_accounting": True,
            "fixed_ip_actuals_means_labels_OLS_fallback_unchanged": True,
            "historical_mean_weights_and_returns_unchanged": True,
            "primary_weights_sum_one_and_10_40_bounds": True,
            "training_slopes_Holm42_reconciles": True, "state_count_partitions_reconcile": True,
            "archived_mean_HAC_p_reconciles": True,
            "archived_main_Sharpe_intervals_reconcile": True},
        "independent_component_algebra": independent_components,
        "training_slope_counts": {"raw_p_below_005": int((slopes.HAC_p < .05).sum()),
                                  "Holm42_p_below_005": int((slopes.Holm_p_42_coefficients < .05).sum())}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Robustness supplement: fixed-penalty IP decomposition and " + str(len(frames)) + " CSVs complete", flush=True)
    return summary

"""Semantic checks for the fixed-penalty IP and reporting supplements.

These checks read generated paths, raw French returns and declared parameters.
They do not invoke the supplement's fitting or allocation functions, repeat
cross-validation, or treat stored check flags as evidence of correctness.
"""
from pathlib import Path
import io
import json
import platform
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
CURRENT = "Release_corrected_current_vintage"
RETUNED = "Partial_realtime_IP"
FIXED = "Partial_realtime_IP_Current_Penalties"
PRIMARY = "LS5_Variance36_Cap40_Shrink50"
COHORTS = {"Original_88": ("2019-04-30", "2026-07-31"),
           "Extended_175": ("2012-01-31", "2026-07-31")}


def raw_french(base):
    """Read the original monthly records, excluding annual summaries."""
    frames = []
    for name, columns in [
        ("F-F_Research_Data_5_Factors_2x3.csv", FACTORS[:5] + ["RF"]),
        ("F-F_Momentum_Factor.csv", ["MOM"]),
    ]:
        text = (base / "data/us" / name).read_text(encoding="utf-8-sig")
        rows = [line for line in text.splitlines() if re.match(r"^\s*\d{6}\s*,", line)]
        frame = pd.read_csv(io.StringIO("\n".join(rows)), names=["Date"] + columns)
        frame.index = pd.to_datetime(frame.pop("Date").astype(str), format="%Y%m").dt.to_period("M").dt.to_timestamp("M")
        frames.append(frame.replace([-99.99, -999.0], np.nan))
    return pd.concat(frames, axis=1)[FACTORS].sort_index()


def _holm(p):
    """Step-down Holm correction independently of the generating utility."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    answer = np.empty_like(p)
    answer[order] = adjusted
    return answer


def _capped_primary(scores):
    positive = np.maximum(np.asarray(scores, float), 0.)
    raw = positive / positive.sum() if positive.sum() else np.ones(5) / 5
    proposed = .5 * raw + .1
    # Proportional redistribution on the five sleeve simplex, with 40% cap.
    free = np.ones(5, dtype=bool)
    result = np.zeros(5)
    remaining = 1.
    while free.any():
        candidates = remaining * proposed[free] / proposed[free].sum()
        indexes = np.flatnonzero(free)
        capped = candidates > .4 + 1e-12
        if not capped.any():
            result[indexes] = candidates
            break
        result[indexes[capped]] = .4
        remaining -= .4 * capped.sum()
        free[indexes[capped]] = False
    return np.r_[0., result]


def run_checks(base_path=None):
    base = Path(base_path).resolve() if base_path else Path(__file__).resolve().parents[1]
    supplement = base / "results/robustness"
    read = lambda name, dates=None: pd.read_csv(supplement / name, parse_dates=dates or [])
    forecasts = read("ip_fixed_penalty_forecasts.csv", ["Signal_Date", "Return_Date"])
    audit = read("ip_fixed_penalty_training_audit.csv", ["Signal_Date", "Return_Date", "Max_Training_Label_Date", "Historical_Return_End"])
    weights = read("ip_fixed_penalty_weights.csv", ["Signal_Date", "Return_Date"])
    returns = read("ip_fixed_penalty_returns.csv", ["Return_Date"])
    main_forecasts = pd.read_csv(base / "results/us/forecasts.csv", parse_dates=["Signal_Date", "Return_Date"])
    main_returns = pd.read_csv(base / "results/us/returns.csv", parse_dates=["Return_Date"])
    main_weights = pd.read_csv(base / "results/us/weights.csv", parse_dates=["Signal_Date", "Return_Date"])
    factors = raw_french(base)
    checks = []

    def check(condition, message):
        if not condition:
            raise AssertionError(message)

    def close(actual, expected, message):
        check(np.allclose(actual, expected, rtol=1e-10, atol=1e-11, equal_nan=True), message)

    def record(name, action):
        action()
        checks.append({"name": name, "passed": True})

    def frozen_parameters():
        declared = json.loads((base / "specification.json").read_text())["us_parameters"]
        parameters = read("ip_parameter_audit.csv")
        check(len(parameters) == 12 and not parameters.duplicated(["Factor", "Model"]).any(), "Counterfactual parameters need six factors and two penalties")
        for row in parameters.itertuples():
            current = declared[CURRENT][row.Factor][row.Model]
            retuned = declared[RETUNED][row.Factor][row.Model]
            close(row.Applied_Alpha, current["alpha"], "Fixed-IP penalty changed from declared current design")
            close(row.Applied_l1_ratio, current.get("l1_ratio", np.nan), "Fixed-IP EN mixing changed")
            close(row.Historical_IP_Retuned_Alpha, retuned["alpha"], "Retuned comparison penalty is mislabeled")
            close(row.Historical_IP_Retuned_l1_ratio, retuned.get("l1_ratio", np.nan), "Retuned EN mixing is mislabeled")
            check(row.Retuning_Changes_Parameters == (current != retuned), "Parameter-change flag does not match the declared designs")
            check(row.Design == FIXED and CURRENT + "_tuning.csv" in row.Parameter_Source, "Counterfactual penalty source is not the current design")

    record("Counterfactual uses all declared current penalties and labels retuning differences", frozen_parameters)

    def nonpenalty_invariance():
        keys = ["Signal_Date", "Return_Date", "Factor"]
        old = main_forecasts[main_forecasts.Design.eq(RETUNED)]
        joined = forecasts.merge(old, on=keys, validate="one_to_one", suffixes=("_fixed", "_retuned"))
        check(len(joined) == len(old) == len(forecasts) == 1050, "Counterfactual target-factor panel changed")
        for column in ["Actual", "Historical_Mean", "Nested_Mean_OLS", "N_Train", "Macro_N_Train", "Macro_OLS"]:
            close(joined[column + "_fixed"], joined[column + "_retuned"], "Non-penalty component changed: " + column)
        check(joined.Macro_Fallback_fixed.equals(joined.Macro_Fallback_retuned), "Fallback eligibility changed")
        expected_dates = pd.date_range("2011-12-31", "2026-06-30", freq="ME")
        check(audit.Signal_Date.equals(pd.Series(expected_dates)), "Counterfactual signal calendar changed")
        check((audit.Return_Date == audit.Signal_Date + pd.offsets.MonthEnd()).all(), "Forecast target is not the next calendar month")
        check((audit.Max_Training_Label_Date <= audit.Signal_Date).all() and (audit.Historical_Return_End <= audit.Signal_Date).all(), "Future labels entered the counterfactual fit")
        timing = pd.read_csv(base / "results/us/training_timing_audit.csv", parse_dates=["Signal_Date", "Return_Date"])
        joined = audit.merge(timing[timing.Design.eq(RETUNED)], on=["Signal_Date", "Return_Date"], validate="one_to_one")
        check(joined.Training_Label_Hash.equals(joined.Macro_Training_Hash), "Counterfactual changed eligible training labels")
        close(joined.N_Train, joined.Macro_N_Train, "Counterfactual training counts changed")

    record("OLS, observations, means, labels and causal calendar are unchanged from retuned IP", nonpenalty_invariance)

    def actual_average_and_fallback():
        component_names = ["Macro_OLS", "Macro_Ridge", "Macro_Elastic_Net"]
        close(forecasts.Macro, forecasts[component_names].mean(axis=1), "Ensemble is not the average of its three actual fitted components")
        fallback = forecasts[forecasts.Macro_Fallback]
        check(set(fallback.Return_Date.dt.strftime("%Y-%m-%d")) == {"2025-11-30", "2025-12-31", "2026-01-31"}, "Unexpected missing-predictor fallback months")
        close(fallback[component_names + ["Macro"]], np.repeat(fallback.Historical_Mean.to_numpy()[:, None], 4, axis=1), "Fallback did not use the full-history mean")
        nonmissing = forecasts[~forecasts.Macro_Fallback]
        check((np.abs(nonmissing.Macro_Ridge - nonmissing.Nested_Mean_OLS) > 1e-4).any(), "Ridge path was replaced by an intercept shortcut")

    record("Ensemble averages actual components and all three missing targets use the declared fallback", actual_average_and_fallback)

    def mapping_and_portfolio_accounting():
        check(len(weights) == 350 and not weights.duplicated(["Return_Date", "Strategy"]).any(), "Counterfactual weights are incomplete or duplicated")
        close(weights[FACTORS].sum(axis=1), 1., "Counterfactual weights do not sum to one")
        close(weights.MKT_RF, 0., "Counterfactual includes unintended market exposure")
        check(weights[FACTORS[1:]].min(axis=None) >= .1 - 1e-10 and weights[FACTORS[1:]].max(axis=None) <= .4 + 1e-10, "Counterfactual violates sleeve bounds")
        for date, group in forecasts.groupby("Signal_Date"):
            variance = (factors.loc[:date].tail(36) / 100).var(ddof=1).to_numpy()
            ordered = group.set_index("Factor").loc[FACTORS]
            for strategy in ["Macro", "Historical_Mean"]:
                score = ordered[strategy].to_numpy()[1:] / 100 / variance[1:]
                expected = _capped_primary(score)
                actual = weights[weights.Signal_Date.eq(date) & weights.Strategy.eq(strategy)][FACTORS].iloc[0]
                close(actual, expected, "Counterfactual primary variance scaling/shrinkage/cap mapping changed")
        check(len(returns) == 2104, "Counterfactual return panel is incomplete")
        for (cohort, strategy, cost), group in returns.groupby(["Cohort", "Strategy", "Cost_bps"]):
            group = group.sort_values("Return_Date")
            matrix = weights[weights.Strategy.eq(strategy)].set_index("Return_Date").loc[group.Return_Date, FACTORS].to_numpy()
            expected_gross = (matrix * factors.loc[group.Return_Date].to_numpy()).sum(axis=1)
            expected_turnover = np.r_[0., .5 * np.abs(np.diff(matrix, axis=0)).sum(axis=1)]
            close(group.Gross_Return_pct, expected_gross, "Raw factor PnL does not reconcile")
            close(group.Turnover, expected_turnover, "Within-cohort turnover or first-entry reset changed")
            close(group.Allocation_Cost_pct, expected_turnover * cost / 100, "Basis-point allocation charge is wrong")
            close(group.Net_Return_pct, expected_gross - expected_turnover * cost / 100, "Net portfolio accounting is wrong")
            if strategy == "Historical_Mean":
                old = main_returns[main_returns.Design.eq(RETUNED) & main_returns.Cohort.eq(cohort) & main_returns.Allocation.eq(PRIMARY) & main_returns.Strategy.eq(strategy) & main_returns.Cost_bps.eq(cost)].sort_values("Return_Date")
                close(group.Net_Return_pct, old.Net_Return_pct, "Counterfactual historical-mean return path changed")
        old = main_weights[main_weights.Design.eq(RETUNED) & main_weights.Allocation.eq(PRIMARY) & main_weights.Strategy.eq("Historical_Mean")].sort_values("Return_Date")
        close(weights[weights.Strategy.eq("Historical_Mean")].sort_values("Return_Date")[FACTORS], old[FACTORS], "Counterfactual historical-mean weights changed")

    record("Primary weights, raw factor PnL, cohort-entry costs and historical-mean paths reconcile", mapping_and_portfolio_accounting)

    def annual_partition():
        annual = read("annual_incremental_returns.csv")
        summary = read("annual_concentration_summary.csv")
        for cohort in COHORTS:
            subset = main_returns[main_returns.Design.eq(CURRENT) & main_returns.Cohort.eq(cohort) & main_returns.Allocation.eq(PRIMARY) & main_returns.Cost_bps.eq(10)]
            pair = subset.pivot(index="Return_Date", columns="Strategy", values="Net_Return_pct").sort_index()
            table = annual[annual.Cohort.eq(cohort)].set_index("Year")
            for year, group in pair.groupby(pair.index.year):
                row = table.loc[year]
                check(row.N_Months == len(group) and row.Full_Calendar_Year == (len(group) == 12), "Partial calendar year is mislabeled")
                close(row.Macro_minus_HM_Arithmetic_Sum_pp, (group.Macro - group.Historical_Mean).sum(), "Calendar-year incremental arithmetic sum is wrong")
                for strategy, column in [("Macro", "Macro_Compounded_Return_pct"), ("Historical_Mean", "Historical_Mean_Compounded_Return_pct")]:
                    close(row[column], 100 * (np.prod(1 + group[strategy] / 100) - 1), "Calendar-year compound is wrong")
            rows = summary[summary.Cohort.eq(cohort)].set_index("Sample")
            close(rows.loc["Full", "Macro_minus_HM_Arithmetic_Sum_pp"], rows.loc["Only_2022", "Macro_minus_HM_Arithmetic_Sum_pp"] + rows.loc["Exclude_2022", "Macro_minus_HM_Arithmetic_Sum_pp"], "2022 increment partition does not sum to full path")
            for sample, mask in [("Full", np.ones(len(pair), bool)), ("Only_2022", pair.index.year == 2022), ("Exclude_2022", pair.index.year != 2022)]:
                group = pair.loc[mask]
                check(rows.loc[sample, "N_Months"] == len(group), "Concentration denominator is wrong")
                close(rows.loc[sample, "Macro_minus_HM_Annual_Arithmetic_Mean_pp"], 12 * (group.Macro - group.Historical_Mean).mean(), "Concentration annual arithmetic increment is wrong")
                for strategy in ["Macro", "Historical_Mean"]:
                    close(rows.loc[sample, strategy + "_Compounded_Return_pct"], 100 * (np.prod(1 + group[strategy] / 100) - 1), "Concentration compound is wrong")

    record("Annual increments and the 2022 partition use paired arithmetic gains and actual partial years", annual_partition)

    def cost_zero_crossings():
        crossing = read("cost_crossings.csv")
        check(len(crossing) == 4 and (crossing.Gamma == 0).all(), "Expected two mean-return cost crossings in each cohort")
        for row in crossing.itertuples():
            selected = main_returns[main_returns.Design.eq(CURRENT) & main_returns.Cohort.eq(row.Cohort) & main_returns.Allocation.eq(PRIMARY)]
            selected = selected[selected.Cost_bps.eq(0 if row.Cost_Type == "Allocation_only" else 10)]
            pair = selected.pivot(index="Return_Date", columns="Strategy", values="Net_Return_pct").sort_index()
            if row.Cost_Type == "Allocation_only":
                exposure = selected.pivot(index="Return_Date", columns="Strategy", values="Turnover").sort_index()
            elif row.Cost_Type == "MOM_internal":
                exposure = main_weights[main_weights.Design.eq(CURRENT) & main_weights.Allocation.eq(PRIMARY)].pivot(index="Return_Date", columns="Strategy", values="MOM").reindex(pair.index)
            else:
                raise AssertionError("Unknown zero-crossing cost type")
            advantage = (pair.Macro - pair.Historical_Mean).mean()
            differential_exposure = (exposure.Macro - exposure.Historical_Mean).mean()
            expected = 100 * advantage / differential_exposure
            close(row.First_Loss_of_Positive_Advantage_bps, expected, "Counterfactual arithmetic cost crossing is wrong")
            close(advantage - expected / 100 * differential_exposure, 0., "Reported cost crossing does not remove the mean advantage")

    record("Counterfactual cost crossings independently zero the arithmetic mean increment", cost_zero_crossings)

    def state_means_target_alignment():
        states = read("rmw_state_means.csv", ["Signal_Start", "Signal_End"])
        calendar = pd.read_csv(base / "results/states/model_calendar.csv", parse_dates=["Signal_Date"])
        calendar = calendar[calendar.Design.eq(CURRENT)].set_index("Signal_Date")
        check(len(states) == 32, "Expected four states in four windows under both label rules")
        predictors = ["VIX", "Credit_Spread", "Term_Spread", "Treasury_2Y_change", "Inflation_lag", "Unemployment_lag", "IP_Growth_lag"]
        for row in states.itertuples():
            column = "Regime_YoY" if row.Variant == "FixedYoY" else "Regime_ExpandingYoY"
            frame = calendar.loc[row.Signal_Start:row.Signal_End]
            group = frame[frame[column].eq(row.Regime)]
            subsequent_dates = group.index + pd.offsets.MonthEnd()
            observed = factors.RMW.reindex(subsequent_dates).dropna()
            check(row.N_All_Signals == len(frame) and row.N_Valid_States == frame[column].notna().sum() and row.N_Missing_States == frame[column].isna().sum(), "State availability denominator is wrong")
            check(row.N_Observed_State == len(group) and row.N_RMW_Outcomes == len(observed), "State counts or next-month target counts are wrong")
            check(row.N_Continuous_Model_Eligible_In_State == group[predictors].notna().all(axis=1).sum(), "Continuous eligibility was conflated with known state coverage")
            close(row.RMW_Next_Month_Mean_pct, observed.mean(), "State mean uses wrong factor return or signal-target alignment")
            close(row.RMW_Next_Month_SD_pct, observed.std(ddof=1), "State SD is wrong")
            close(row.RMW_Next_Month_Median_pct, observed.median(), "State median is wrong")
            if not len(observed):
                check(pd.isna(row.RMW_Next_Month_Mean_pct) and pd.isna(row.RMW_Next_Month_SD_pct) and pd.isna(row.RMW_Next_Month_Median_pct), "Absent state was represented by a false zero return")
        absent = states[states.Split.eq("Original_88") & states.Regime.eq("Expansion")]
        check(len(absent) == 2 and absent.N_Observed_State.eq(0).all(), "Original four-state coverage limitation disappeared")

    record("Conditional state moments join raw RMW to the next target month and absent Expansion stays undefined", state_means_target_alignment)

    def mean_tests_and_families():
        tests = read("ip_design_mean_hac.csv")
        combined = pd.concat([main_returns, returns], ignore_index=True)
        check(len(tests) == 18, "Expected three designs, two cohorts and three HAC lags")
        for row in tests.itertuples():
            selected = combined[combined.Design.eq(row.Design) & combined.Cohort.eq(row.Cohort) & combined.Allocation.eq(PRIMARY) & combined.Cost_bps.eq(10)]
            pair = selected.pivot(index="Return_Date", columns="Strategy", values="Net_Return_pct").sort_index()
            values = (pair.Macro - pair.Historical_Mean).to_numpy()
            centered = values - values.mean()
            meat = np.dot(centered, centered)
            for lag in range(1, row.HAC_Lags + 1):
                meat += 2 * (1 - lag / (row.HAC_Lags + 1)) * np.dot(centered[lag:], centered[:-lag])
            se = np.sqrt(meat) / len(values)
            close(row.Mean, values.mean(), "Mean test uses the wrong paired path")
            close(row.HAC_SE, se, "HAC mean standard error is wrong")
            close(row.p_Two_Sided, 2 * norm.sf(abs(values.mean() / se)), "Mean-test p-value is wrong")
            close(row.Annual_Mean_Gain_pp, 12 * values.mean(), "Mean gain units are wrong")
            if row.Design == FIXED:
                check(pd.isna(row.Archived_Within_Design_Three_Comparison_Holm_p), "Continuous-only counterfactual was assigned the archived state-comparison family")
        for _, group in tests.groupby(["Cohort", "HAC_Lags"]):
            check(set(group.Design) == {CURRENT, FIXED, RETUNED}, "Supplement Holm family is not exactly the three declared design contrasts")
            close(group.Holm_p_Three_Design_Contrasts, _holm(group.p_Two_Sided), "Supplement Holm correction uses the wrong family")
            check(group.Supplement_Family_Definition.str.contains("separate family within each cohort and HAC lag", regex=False).all(), "Within-lag family scope is not disclosed")

    record("HAC mean inference and Holm correction use the declared three-design within-lag family", mean_tests_and_families)
    report = {"all_passed": True, "tests_run": len(checks), "platform": platform.platform(),
              "python": platform.python_version(), "checks": checks,
              "scope": "Semantic counterfactual checks from raw French returns and declared settings on this platform. No cross-validation, supplement refitting, selection-uncertainty validation or execution on another operating system."}
    (base / "results/supplement_tests.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Supplement semantic checks: {len(checks)} passed.", flush=True)
    return report


if __name__ == "__main__":
    run_checks(sys.argv[1] if len(sys.argv) > 1 else None)

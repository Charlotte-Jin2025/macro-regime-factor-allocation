"""Independent semantic audits of the frozen cross-market allocation transfer.

Raw JKP monthly returns are read independently of the allocation generator.
The checks reconstruct forecast training-label means, variances, simplex
weights, costs, paired HAC moments and bootstrap pivots from their definitions.
They do not accept stored pass flags or trial figures as validation evidence.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

THEMES = ["size", "value", "investment", "profitability", "momentum"]
STRATEGIES = ["Equal_Weight", "Historical_Mean", "Macro", "Macro+Regime"]
PAIRS = [("Macro", "Historical_Mean"), ("Macro+Regime", "Macro")]


def _raw_returns(base, country):
    name = "JKP_GBD_monthly_vw.csv" if country == "UK" else "JKP_DEU_monthly_vw.csv"
    raw = pd.read_csv(base / "data/international" / name)
    raw = raw.loc[raw.name.isin(THEMES)].copy()
    assert raw.freq.eq("monthly").all() and raw.weighting.eq("vw").all()
    raw["Date"] = pd.to_datetime(raw.date).dt.to_period("M").dt.to_timestamp("M")
    assert not raw.duplicated(["Date", "name"]).any()
    return raw.pivot(index="Date", columns="name", values="ret")[THEMES].sort_index()


def _weights(score):
    """Explicit five-theme positive-score/shrink/cap simplex definition."""
    positive = np.maximum(np.asarray(score, float), 0)
    normalized = positive / positive.sum() if positive.sum() else np.full(5, .2)
    shrunk = .5 * normalized + .1
    result = np.zeros(5)
    free = np.ones(5, bool)
    remaining = 1.
    while free.any():
        proposed = remaining * shrunk[free] / shrunk[free].sum()
        indexes = np.flatnonzero(free)
        over = proposed > .4 + 1e-12
        if not over.any():
            result[indexes] = proposed
            break
        result[indexes[over]] = .4
        remaining -= .4 * over.sum()
        free[indexes[over]] = False
    return result


def _hac_se(x, lag):
    """Mean estimating-equation SE for a complete natural-month sequence."""
    centered = np.asarray(x, float) - np.mean(x, axis=-1, keepdims=True)
    n = centered.shape[-1]
    meat = np.sum(centered ** 2, axis=-1)
    for k in range(1, lag + 1):
        meat += 2 * (1 - k / (lag + 1)) * np.sum(centered[..., k:] * centered[..., :-k], axis=-1)
    return np.sqrt(np.maximum(meat, 0)) / n


def _sharpe_if(pair):
    pair = np.asarray(pair, float)
    n = pair.shape[-2]
    mu = pair.mean(axis=-2, keepdims=True)
    centered = pair - mu
    variance = (centered ** 2).mean(axis=-2, keepdims=True)
    sd = np.sqrt(variance)
    factor = np.sqrt(12 * (n - 1) / n)
    sr = (factor * mu / sd).squeeze(-2)
    score = factor * (centered / sd - mu * (centered ** 2 - variance) / (2 * sd ** 3))
    return sr[..., 0] - sr[..., 1], score[..., 0] - score[..., 1]


def _holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    answer = np.empty_like(values)
    answer[order] = np.minimum(1., np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order]))
    return answer


def run_checks(base_path=None):
    base = Path(base_path or Path(__file__).resolve().parents[1]).resolve()
    folder = base / "results/cross_market"
    read = lambda name, dates=None: pd.read_csv(folder / name, parse_dates=dates or [])
    weights = read("weights.csv", ["Signal_Date", "Return_Date"])
    returns = read("returns.csv", ["Return_Date"])
    risk = read("risk_timing_audit.csv", ["Signal_Date", "Return_Date", "History_Start", "History_End"])
    metrics = read("metrics.csv", ["First_Target", "Last_Target"])
    hac = read("mean_difference_hac.csv")
    intervals = read("sharpe_intervals.csv")
    mde = read("mde.csv")
    proximity = read("path_proximity_diagnostics.csv")
    design = read("forecast_design_audit.csv", ["First_Signal", "First_Target", "Last_Signal", "Last_Target"])
    coverage = read("state_coverage.csv")
    source = pd.read_csv(base / "results/international/forecast_and_coefficient_path.csv",
                         parse_dates=["signal_date", "target_date"])
    raw = {country: _raw_returns(base, country) for country in ["UK", "DE"]}
    panels = {country: frame.pivot(index="Return_Date", columns="Strategy", values="Net_Return_decimal").sort_index()
              for country, frame in returns.groupby("Country")}
    checks = []

    def check(condition, message):
        if not condition:
            raise AssertionError(message)

    def close(actual, expected, message):
        check(np.allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True), message)

    def record(name, callback):
        callback()
        checks.append({"name": name, "passed": True})

    def windows_and_units():
        for country, first, count in [("UK", "2014-12-31", 133), ("DE", "2015-11-30", 122)]:
            required = pd.date_range(first, "2025-12-31", freq="ME")
            panel = panels[country]
            check(len(panel) == count and panel.index.equals(required), "International target calendar changed or lost a month")
            check(set(panel.columns) == set(STRATEGIES), "Missing or unplanned international strategy")
            country_weights = weights.loc[weights.Country.eq(country)]
            check(country_weights.Return_Date.eq(country_weights.Signal_Date + pd.offsets.MonthEnd(1)).all(), "Weights must precede next-month return")
            close(country_weights[THEMES].sum(axis=1), 1., "Theme sleeve weights must sum to one")
            check(country_weights[THEMES].ge(.1 - 1e-12).all(axis=None), "50% EW shrinkage floor violated")
            check(country_weights[THEMES].le(.4 + 1e-12).all(axis=None), "Frozen 40% cap violated")
        check(len(weights) == 1020 and len(returns) == 1020, "Unexpected international path count")
        check(returns.Cost_bps.eq(10).all(), "Frozen 10 bp charge changed")
    record("complete_target_calendar_decimal_units_and_frozen_simplex", windows_and_units)

    def matched_training_means():
        from src import international_core as parsing
        for country in ["UK", "DE"]:
            sample = parsing.make_design(parsing.load_calendar(base / "data/international", country), 2)
            for signal, forecast in source.loc[source.country.eq(country)].groupby("signal_date"):
                training = sample.loc[(sample.index < signal) & (sample.target_date <= signal)]
                check(training.target_date.max() <= signal, "Training uses unavailable outcome label")
                expected = training[[theme + "_next" for theme in THEMES]].mean().to_numpy()
                for model in ["Macro", "Macro+Regime"]:
                    ordered = forecast.loc[forecast.model.eq(model)].set_index("theme").loc[THEMES]
                    close(ordered.matched_training_mean_decimal, expected, "Historical mean does not match exact model training labels")
                    close(ordered.actual_decimal, raw[country].loc[signal + pd.offsets.MonthEnd(1)], "Model actuals differ from raw JKP returns")
                    check(ordered.train_n.eq(len(training)).all(), "Forecast sample count differs from reconstructed training labels")
    record("matched_training_label_means_and_raw_JKP_actuals", matched_training_means)

    def risk_and_weights():
        for audit in risk.itertuples():
            dates = pd.date_range(audit.Signal_Date - pd.offsets.MonthEnd(35), audit.Signal_Date, freq="ME")
            actual = raw[audit.Country].reindex(dates)[THEMES]
            check(len(actual) == 36 and actual.notna().all(axis=None), "Risk window compresses gaps or lacks 36 months")
            check(audit.History_Start == dates.min() and audit.History_End == dates.max(), "Risk audit has incorrect endpoints")
            check(audit.History_End < audit.Return_Date and audit.N_Months == 36 and audit.Variance_ddof == 1, "Risk window sees target return")
            variance = actual.var(ddof=1).to_numpy()
            close([getattr(audit, theme + "_variance_decimal") for theme in THEMES], variance, "Risk variance is not decimal sample variance")
            forecast = source.loc[source.country.eq(audit.Country) & source.signal_date.eq(audit.Signal_Date)]
            for strategy in STRATEGIES:
                ordered = forecast.loc[forecast.model.eq("Macro" if strategy != "Macro+Regime" else strategy)].set_index("theme").loc[THEMES]
                if strategy == "Equal_Weight":
                    expected = np.full(5, .2)
                else:
                    score = ordered.matched_training_mean_decimal.to_numpy() if strategy == "Historical_Mean" else ordered.prediction_decimal.to_numpy()
                    expected = _weights(score / variance)
                stored = weights.loc[weights.Country.eq(audit.Country) & weights.Signal_Date.eq(audit.Signal_Date) & weights.Strategy.eq(strategy), THEMES]
                check(len(stored) == 1, "Duplicate or missing allocation weight")
                close(stored.iloc[0], expected, "Allocation differs from independently defined frozen US mapping")
    record("raw_36_calendar_month_risks_and_independent_weight_mapping", risk_and_weights)

    def causal_risk_guard():
        from src.cross_market import risk_window
        calendar = raw["UK"].copy()
        signal = pd.Timestamp("2014-11-30")
        _, initial = risk_window(calendar, signal)
        perturbed = calendar.copy()
        perturbed.loc[signal + pd.offsets.MonthEnd(1), THEMES] = 10000.
        _, unchanged = risk_window(perturbed, signal)
        close(initial, unchanged, "Next-month outcome perturbation changes risk estimate")
        missing = calendar.drop(index=signal - pd.offsets.MonthEnd(8))
        try:
            risk_window(missing, signal)
        except ValueError:
            pass
        else:
            raise AssertionError("Risk helper silently compresses missing calendar month")
    record("future_outcome_invariance_and_missing_month_rejection", causal_risk_guard)

    def return_accounting():
        for (country, strategy), frame in returns.groupby(["Country", "Strategy"]):
            frame = frame.sort_values("Return_Date")
            matched = weights.loc[weights.Country.eq(country) & weights.Strategy.eq(strategy)].set_index("Return_Date").loc[frame.Return_Date]
            matrix = matched[THEMES].to_numpy()
            actual = raw[country].loc[frame.Return_Date, THEMES].to_numpy()
            gross = np.sum(matrix * actual, axis=1)
            turnover = np.r_[0., .5 * np.abs(np.diff(matrix, axis=0)).sum(axis=1)]
            close(frame.Gross_Return_decimal, gross, "Gross allocation return uses different JKP target values")
            close(frame.Turnover, turnover, "Turnover differs from target-to-target half-L1 convention")
            close(frame.Allocation_Cost_decimal, .001 * turnover, "10 bp decimal sleeve charge wrong")
            close(frame.Net_Return_decimal, gross - .001 * turnover, "Net return accounting wrong")
            check(frame.Turnover.iloc[0] == 0 and frame.Allocation_Cost_decimal.iloc[0] == 0, "Initial entry not free")
    record("raw_return_dot_products_target_turnover_costs_and_free_entry", return_accounting)

    def metric_accounting():
        for row in metrics.itertuples():
            net = panels[row.Country][row.Strategy].to_numpy()
            sd = net.std(ddof=1)
            wealth = np.r_[1., np.cumprod(1 + net)]
            close(row.Annualized_Mean, 12 * net.mean(), "Annual mean units wrong")
            close(row.Annualized_Volatility, np.sqrt(12) * sd, "Annual volatility units wrong")
            close(row.Sharpe, np.sqrt(12) * net.mean() / sd, "Annual sample-SD Sharpe wrong")
            close(row.Synthetic_CAGR, wealth[-1] ** (12 / len(net)) - 1, "Synthetic CAGR wrong")
            close(row.Max_Drawdown, np.min(wealth / np.maximum.accumulate(wealth) - 1), "Synthetic drawdown wrong")
            check(row.N_Months == len(net), "Metric count wrong")
    record("independent_annual_moments_synthetic_compounding_and_drawdowns", metric_accounting)

    def pair_hac():
        for row in hac.itertuples():
            panel = panels[row.Country]
            pair = panel[[row.Strategy_A, row.Strategy_B]].to_numpy()
            difference = pair[:, 0] - pair[:, 1]
            se = float(_hac_se(difference, row.HAC_Lags))
            statistic = difference.mean() / se
            close(row.Mean, difference.mean(), "Paired monthly mean wrong")
            close(row.HAC_SE, se, "Paired calendar HAC SE wrong")
            close(row.p_Two_Sided, 2 * norm.sf(abs(statistic)), "HAC two-sided normal p wrong")
            close(row.Annualized_Mean_Difference_pp, 1200 * difference.mean(), "Annual mean difference pp conversion wrong")
            close(row.Annualized_CI_Low_pp, 1200 * (difference.mean() - norm.ppf(.975) * se), "Annual HAC CI pp wrong")
        for lag, group in hac.groupby("HAC_Lags"):
            check(len(group) == 4 and set(group.Country) == {"UK", "DE"}, "International Holm family must contain four tests")
            close(group.Holm_p_Four_International_Comparisons, _holm(group.p_Two_Sided), "International-only Holm adjustment wrong")
            check(group.Holm_Family.str.contains("excludes US/forecast").all(), "Holm family scope missing")
    record("paired_calendar_HAC_units_and_four_international_Holm_family", pair_hac)

    def sharpe_power():
        for row in mde.itertuples():
            pair = panels[row.Country][[row.Strategy_A, row.Strategy_B]].to_numpy()
            delta, score = _sharpe_if(pair)
            se = float(_hac_se(score, row.HAC_Lags))
            close(row.Observed_Annual_Sharpe_Difference, delta, "Paired Sharpe difference wrong")
            close(row.Actual_Paired_IF_HAC_SE_Annual, se, "Paired influence-function Sharpe SE wrong")
            close(row.Approx_MDE_Annual_Sharpe_80pct, (norm.ppf(.975) + norm.ppf(.8)) * se, "MDE method differs from US local-normal approximation")
            close(row.Return_Correlation, np.corrcoef(pair.T)[0, 1], "Paired path correlation wrong")
        check(len(mde) == 12 and set(mde.HAC_Lags) == {3, 6, 12}, "Missing MDE lag sensitivities")
    record("independent_paired_Sharpe_IF_HAC_and_indicative_MDE", sharpe_power)

    def bootstrap_pivots():
        for row in intervals.itertuples():
            panel = panels[row.Country]
            pair = panel[[row.Strategy_A, row.Strategy_B]].to_numpy()
            delta, score = _sharpe_if(pair)
            se = float(_hac_se(score, 3))
            n = len(pair)
            rng = np.random.default_rng(row.Seed)
            starts = rng.integers(0, n - row.Block_Length + 1,
                                  size=(3000, int(np.ceil(n / row.Block_Length))))
            indexes = (starts[..., None] + np.arange(row.Block_Length)).reshape(3000, -1)[:, :n]
            bootstrap_delta, bootstrap_score = _sharpe_if(pair[indexes])
            bootstrap_se = _hac_se(bootstrap_score, 3)
            finite = np.isfinite(bootstrap_delta) & np.isfinite(bootstrap_se) & (bootstrap_se > 1e-12)
            q_low, q_high = np.quantile((bootstrap_delta[finite] - delta) / bootstrap_se[finite], [.025, .975])
            close(row.Pivot_Q025, q_low, "Lower moving-block studentized pivot wrong")
            close(row.Pivot_Q975, q_high, "Upper moving-block studentized pivot wrong")
            close(row.CI_Low, delta - q_high * se, "Lower paired percentile-t interval wrong")
            close(row.CI_High, delta - q_low * se, "Upper paired percentile-t interval wrong")
            check(row.Finite_Draws == int(finite.sum()) and row.Requested_Draws == 3000, "Bootstrap finite-draw audit wrong")
            check(row.Primary_Block == (row.Block_Length == 6) and row.HAC_Lags == 3, "Main bootstrap/studentizer changed")
        check(len(intervals) == 12 and set(intervals.Block_Length) == {3, 6, 12}, "Missing block sensitivities")
    record("independent_3000_draw_moving_block_percentile_t_pivots", bootstrap_pivots)

    def proximity_and_design():
        for row in proximity.itertuples():
            panel = panels[row.Country]
            a = weights.loc[weights.Country.eq(row.Country) & weights.Strategy.eq(row.Strategy_A)].set_index("Return_Date")[THEMES].loc[panel.index].to_numpy()
            b = weights.loc[weights.Country.eq(row.Country) & weights.Strategy.eq(row.Strategy_B)].set_index("Return_Date")[THEMES].loc[panel.index].to_numpy()
            difference = panel[row.Strategy_A].to_numpy() - panel[row.Strategy_B].to_numpy()
            close(row.Mean_Half_L1_Weight_Difference, .5 * np.abs(a - b).sum(axis=1).mean(), "Average paired half-L1 weights wrong")
            close(row.Paired_Return_Difference_RMS_bp, 10000 * np.sqrt(np.mean(difference ** 2)), "Paired RMS basis points wrong")
            check(row.Near_Degenerate_Paired_Paths == (row.Return_Correlation >= .9999), "Near-degeneracy diagnostic flag wrong")
            check("not evidence of a more powerful" in row.Caution, "Near-degeneracy economic caveat missing")
        check(len(design) == 20, "Need all country/model/theme inherited designs")
        check(design.Forecast_Algorithm.eq("StandardScaler + Ridge").all(), "International algorithm mislabeled as US ensemble")
        check(design.State_Reference_Category.eq("Low growth / low inflation (Slowdown)").all(), "International state reference mislabeled")
        check(design.Allocation_Parameter_Reselection.eq(False).all(), "Transfer reports parameter reselection")
        for row in design.itertuples():
            forecast = source.loc[source.country.eq(row.Country) & source.model.eq(row.Model) & source.theme.eq(row.Theme)]
            close(forecast.selected_alpha, row.Selected_Alpha, "Frozen historical CV penalty changed")
        from src import international_core as parsing
        for country in ["UK", "DE"]:
            sample = parsing.make_design(parsing.load_calendar(base / "data/international", country), 2)
            dates = source.loc[source.country.eq(country), "signal_date"].drop_duplicates().sort_values()
            initial = sample.loc[sample.index < dates.min()]
            cuts = initial[["state_growth", "state_inflation"]].median()
            for phase, frame in [("Initial_Training", initial), ("Allocation_Evaluation", sample.loc[dates])]:
                state_code = 2 * frame.state_growth.gt(cuts.state_growth).astype(int) + frame.state_inflation.gt(cuts.state_inflation).astype(int)
                group = coverage.loc[coverage.Country.eq(country) & coverage.Phase.eq(phase)].sort_values("State_Code")
                close(group.Count, [int(state_code.eq(code).sum()) for code in range(4)], "Fixed-state support counts wrong")
                check(group.Count.sum() == len(frame), "State counts do not exhaust eligible signals")
        summary = json.loads((folder / "summary.json").read_text())
        check("already inspected" in summary["Study_Status"], "Extension mislabeled independent untouched holdout")
        check("cancellation is not guaranteed" in " ".join(summary["Data_Limits"]), "Incorrect guarantee of FX cancellation")
    record("path_proximity_and_inherited_Ridge_design_caveats", proximity_and_design)

    def hashes():
        inputs = read("input_sha256.csv")
        check(len(inputs) == 11, "Expected eight raw international inputs plus forecast path, US and transfer specifications")
        for row in inputs.itertuples():
            close_hash = hashlib.sha256((base / row.Path).read_bytes()).hexdigest()
            check(row.SHA256 == close_hash, "Cross-market input hash mismatch")
    record("all_raw_and_upstream_input_SHA256_hashes", hashes)

    result = {"all_passed": True, "tests_run": len(checks), "passed": len(checks), "checks": checks,
              "scope": "Raw JKP returns, exact training means, pre-target risks, independent weights/returns/costs/HAC/MDE/bootstrap; no claims about untested operating systems"}
    (folder / "semantic_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"cross_market_checks_passed": len(checks)}), flush=True)
    return result


if __name__ == "__main__":
    base = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if base:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run_checks(base)

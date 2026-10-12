"""Report bindings for the computed cross-market allocation supplement."""
import csv
from pathlib import Path


def read_csv(base, path):
    with (Path(base) / path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def one(rows, **filters):
    selected = [r for r in rows if all(str(r[k]) == str(v) for k, v in filters.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one cross-market result for {filters}")
    return selected[0]


def fmt(x, digits=3, scale=1):
    return f"{float(x) * scale:.{digits}f}".replace("-", "−")


def load_extension(base, data, values):
    data["cross_market"] = read_csv(base, "results/cross_market/allocation_summary.csv")
    data["cross_intervals"] = read_csv(base, "results/cross_market/sharpe_intervals.csv")
    data["cross_means"] = read_csv(base, "results/cross_market/mean_difference_hac.csv")
    data["cross_states"] = read_csv(base, "results/cross_market/state_coverage.csv")
    data["cross_sources"] = read_csv(base, "documentation/data_source_decisions.csv")
    for country, prefix in [("UK", "uk"), ("DE", "de")]:
        r = one(data["cross_market"], Country=country)
        precision = 5 if country == "DE" else 3
        values.update({
            prefix + "_delta_sr": fmt(r["Macro_vs_HM_Sharpe_Difference"], precision),
            prefix + "_ci_low": fmt(r["Macro_vs_HM_CI_Low"], precision),
            prefix + "_ci_high": fmt(r["Macro_vs_HM_CI_High"], precision),
            prefix + "_mean_gain": fmt(r["Macro_vs_HM_Annualized_Mean_Difference_pp"], 4 if country == "DE" else 3),
            prefix + "_mde": fmt(r["Macro_vs_HM_Approx_MDE"], 5 if country == "DE" else 3),
            prefix + "_ew_sr": fmt(r["EW_Sharpe"]),
            prefix + "_macro_sr": fmt(r["Macro_Sharpe"]),
            prefix + "_state_delta_sr": fmt(r["State_vs_Macro_Sharpe_Difference"]),
            prefix + "_weight_distance": fmt(r["Macro_vs_HM_Mean_Half_L1_Weight_Difference"], 2, 100),
        })
    weights = read_csv(base, "results/us/weights.csv")
    filters = {"Design": "Release_corrected_current_vintage", "Allocation": "LS5_Variance36_Cap40_Shrink50"}
    w = [r for r in weights if all(r[k] == v for k, v in filters.items())]
    hm = {r["Return_Date"]: r for r in w if r["Strategy"] == "Historical_Mean"}
    macro = [r for r in w if r["Strategy"] == "Macro"]
    distance = sum(.5 * sum(abs(float(r[f]) - float(hm[r["Return_Date"]][f]))
                       for f in ["SMB", "HML", "RMW", "CMA", "MOM"]) for r in macro) / len(macro)
    values["us_weight_distance"] = fmt(distance, 2, 100)


def extension_table(kind, data):
    if kind == "cross_market_allocation":
        rows = []
        for country in ["UK", "DE"]:
            r = one(data["cross_market"], Country=country)
            d = 5 if country == "DE" else 3
            rows.append([f'{country} / {r["N_Months"]}', fmt(r["EW_Sharpe"]), fmt(r["HM_Sharpe"]),
                fmt(r["Macro_Sharpe"]), fmt(r["Macro_vs_HM_Sharpe_Difference"], d),
                f'[{fmt(r["Macro_vs_HM_CI_Low"], d)}, {fmt(r["Macro_vs_HM_CI_High"], d)}]',
                fmt(r["Macro_vs_HM_Annualized_Mean_Difference_pp"], 4 if country == "DE" else 3),
                fmt(r["Macro_vs_HM_p_Two_Sided"])])
        return ("Table 9 UK and German allocations under the frozen rule",
            ["Market / N", "Equal weight SR", "Mean SR", "Macro SR", "Δ SR", "95% interval", "Mean gain pp", "HAC3 p"],
            rows, [.95, .55, .55, .55, .80, 1.60, .80, .70],
            "UK: December 2014–December 2025; Germany: November 2015–December 2025. Each market uses five JKP themes, 36 prior calendar months of variance, 50% equal-weight shrinkage, a 40% cap and a hypothetical 10 bp sleeve charge. Differences compare Macro with its identically mapped, matched-label historical mean and are computed before rounding strategy levels; mean gain is annual pp. Sharpe intervals use 3,000 paired six-month moving-block percentile-t draws, HAC3 studentization, and condition on fitted paths. Mean p-values are two-sided; four-comparison international Holm p=0.951 for both Macro–mean contrasts. Equal weight is a separate benchmark. No within-theme cost calibration is imposed.",
            ["results/cross_market/allocation_summary.csv", "results/cross_market/sharpe_intervals.csv", "results/cross_market/mean_difference_hac.csv"])
    if kind == "international_sources":
        return ("Table B2 International data sources and inclusion decisions",
            ["Input", "Source", "Use and documented constraint"], [
                ["UK CPI and IP", "ONS D7BT / K222", "All-items CPI and seasonally adjusted production levels; year-on-year changes; CPI reference lag one and IP lag two."],
                ["German CPI and IP", "Destatis 61111 / 42153", "National CPI and industry production indices; the same calendar transformations and reference lags."],
                ["UK and German stress", "ECB CLIFS", "Country equity, bond and FX stress composites at reference lag two; not equivalent to a Treasury yield, VIX or Baa spread."],
                ["Five country themes", "JKP Global Factor Data", "Monthly ret in decimal USD; value-weighted characteristic portfolios averaged within signed themes. Theme memberships vary; these are conceptual counterparts to US factors."],
                ["UK three-month bills", "Bank of England IUMAAJNB", "Series discontinued in July 2017. An earlier rate file establishes the coverage constraint; it is excluded from this three-predictor experiment and not redistributed."],
                ["Other planned candidates", "UK unemployment / AQR", "Listed in the original source plan but no retained raw file or executable analysis. Their listing documents a plan, not completed data collection or evidence."],
            ], [1.35, 1.50, 3.65],
            "Eight international raw files supply the four retained input groups; exact file names, columns, monthly coverage, hashes and primary URLs are in documentation/data_source_decisions.csv. Reference lags are transparent timing proxies, not verified historical release dates. Original download timestamps and decision-month vintages are unavailable. The bill-series constraint does not establish that CLIFS was chosen as its replacement. Data descriptions are supported by ONS, Destatis, ECB, JKP and Bank of England sources cited in the report.",
            ["documentation/data_source_decisions.csv", "documentation/cross_market_sources.json", "data/input_manifest.json"])
    if kind == "international_allocation_uncertainty":
        rows = []
        for country in ["UK", "DE"]:
            for comparison, label in [("Macro_minus_Historical_Mean", "Macro − mean"), ("Macro+Regime_minus_Macro", "State − Macro")]:
                intervals = [one(data["cross_intervals"], Country=country, Comparison=comparison, Block_Length=b) for b in [3, 6, 12]]
                d = 5 if country == "DE" and comparison.startswith("Macro_minus") else 4
                rows.append([f"{country} {label}", fmt(intervals[0]["Sharpe_Difference"], d),
                    *[f'[{fmt(r["CI_Low"], d)}, {fmt(r["CI_High"], d)}]' for r in intervals]])
        return ("Table B3 International Sharpe differences by block length",
            ["Paired comparison", "Δ SR", "3 months", "6 months", "12 months"], rows,
            [1.50, .80, 1.40, 1.40, 1.40],
            "95% marginal fixed-path moving-block percentile-t intervals, 3,000 paired draws per setting, HAC3 studentizer. UK N=133; Germany N=122. Six-month blocks are primary. UK State−Macro intervals exclude zero at blocks three and six but include it at twelve; Sharpe intervals have no multiplicity or prior-selection correction. International two-sided mean tests use a separate four-comparison Holm family at each HAC lag and are not equivalent to these Sharpe intervals. Evaluation state counts, in Slowdown / Stagflation / Expansion / Overheating order, are UK 38 / 53 / 30 / 12 and Germany 31 / 61 / 18 / 12. The inherited international model omits Slowdown and selects separate Macro and state penalties. Full parameter, risk-date and state audits are in the mapped CSVs.",
            ["results/cross_market/sharpe_intervals.csv", "results/cross_market/mean_difference_hac.csv", "results/cross_market/state_coverage.csv", "results/cross_market/forecast_design_audit.csv", "results/cross_market/risk_timing_audit.csv"])
    raise ValueError(kind)

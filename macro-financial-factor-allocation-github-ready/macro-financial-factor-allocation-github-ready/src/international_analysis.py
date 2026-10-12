"""Quantify international Ridge shrinkage without changing the fitted design.

Only writes results/international. All macro data and returns are
the packaged fixed snapshots. A small forecast deviation is a descriptive
property of this fitted pipeline, not proof that macro information is absent.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

FIRST_OOS = {"UK": pd.Timestamp("2014-11-30"), "DE": pd.Timestamp("2015-10-31")}
TOLERANCE_RMS_BP = 1.0
TOLERANCE_MAX_BP = 5.0


def run_analysis(base_path=None):
    base = Path(base_path or Path(__file__).resolve().parents[1]).resolve()
    sys.path.insert(0, str(base))
    from . import international_core as old
    out = base/"results/international"
    out.mkdir(parents=True, exist_ok=True)
    original_selection = pd.read_csv(base/"reference_results/international/model_selection.csv")
    
    reference = pd.read_csv(base/"reference_results/international/forecast_metrics.csv")
    
    reference_predictions = pd.read_csv(base/"reference_results/international/forecasts.csv")
    
    selections, cvrows, paths, coefs, timing, inputs = [], [], [], [], [], []
    for country in ["UK", "DE"]:
        # Reconstruct the calendar before shifting; archived sample dates are a check.
        sample = old.make_design(old.load_calendar(base/"data/international", country), 2)
        sample.to_csv(out/f"{country.lower()}_model_sample.csv")
        first = FIRST_OOS[country]
        train0 = sample.loc[sample.index < first]
        thresholds = tuple(float(train0[c].median()) for c in ["state_growth", "state_inflation"])
        folds = list(TimeSeriesSplit(n_splits=4).split(train0))
        selected = {}
        for model in old.MODELS:
            for theme in old.THEMES:
                search = GridSearchCV(old.pipeline(model), {"ridge__alpha": old.ALPHAS},
                                      scoring="neg_mean_squared_error", cv=folds, n_jobs=1, error_score="raise")
                search.fit(train0[old.DESIGN], train0[theme+"_next"])
                alpha = float(search.best_params_["ridge__alpha"])
                best = -float(search.best_score_)
                original = original_selection.loc[(original_selection.country==country) &
                    (original_selection.model==model) & (original_selection.theme==theme)].iloc[0]
                assert old.unique_alpha_index(alpha) == old.unique_alpha_index(original.alpha), \
                    f'International CV selected a different grid point: {country} {model} {theme}'
                assert np.isclose(best,original.cv_mse,rtol=1e-9,atol=1e-10), \
                    f'International CV loss differs: {country} {model} {theme}'
                selected[model, theme] = alpha
                baseline_losses, model_losses = [], []
                for fold, (fit, val) in enumerate(folds, 1):
                    training = train0.iloc[fit]
                    validation = train0.iloc[val]
                    assert training.target_date.max() <= validation.index.min()
                    benchmark = float(training[theme+"_next"].mean())
                    fit_pipe = old.pipeline(model, alpha)
                    fit_pipe.fit(training[old.DESIGN], training[theme+"_next"])
                    prediction = fit_pipe.predict(validation[old.DESIGN])
                    actual = validation[theme+"_next"].to_numpy()
                    bm_mse = float(np.mean((actual-benchmark)**2))
                    model_mse = float(np.mean((actual-prediction)**2))
                    baseline_losses.append(bm_mse); model_losses.append(model_mse)
                    cvrows.append(dict(country=country, model=model, theme=theme, fold=fold,
                        fit_n=len(training), validation_n=len(validation), alpha=alpha,
                        first_validation_signal=old.date(validation.index.min()),
                        latest_training_target=old.date(training.target_date.max()),
                        fold_training_mean_decimal=benchmark, baseline_mse=bm_mse, selected_model_mse=model_mse,
                        baseline_kind="Intercept-only mean fitted once on this CV training fold",
                        state_thresholds_fitted_within_fold=True, scaler_fitted_within_fold=True))
                cv_bm = float(np.mean(baseline_losses))
                assert abs(best-np.mean(model_losses)) < 1e-14
                full_initial = old.pipeline(model, alpha, thresholds)
                full_initial.fit(train0[old.DESIGN], train0[theme+"_next"])
                selections.append(dict(country=country, model=model, theme=theme, selected_alpha=alpha,
                    alpha_grid_min=min(old.ALPHAS), alpha_grid_max=max(old.ALPHAS),
                    alpha_at_upper_boundary=alpha==max(old.ALPHAS), initial_train_n=len(train0),
                    initial_cv_baseline_mse=cv_bm, initial_cv_best_mse=best,
                    initial_cv_mse_improvement=cv_bm-best,
                    initial_cv_relative_mse_improvement=1-best/cv_bm,
                    cv_beats_mean=best<cv_bm,
                    initial_standardized_coef_l2=float(np.linalg.norm(full_initial.named_steps["ridge"].coef_)),
                    initial_coef_signal_cutoff=old.date(train0.index.max()),
                    initial_latest_training_target=old.date(train0.target_date.max())))
        print(country+": reproduced all initial CV selections", flush=True)
        for date in sample.index[sample.index>=first]:
            training = sample.loc[(sample.index<date)&(sample.target_date<=date)]
            test = sample.loc[[date]]
            timing.append(dict(country=country, signal_date=old.date(date),
                target_date=old.date(test.target_date.iloc[0]), train_n=len(training),
                latest_training_target=old.date(training.target_date.max()),
                ip_lag=2, clifs_lag=2, cpi_lag=1))
            assert training.target_date.max()<=date
            assert test.target_date.iloc[0] == date+pd.offsets.MonthEnd(1)
            for model in old.MODELS:
                names = old.MACRO + ([] if model=="Macro" else ["state1_low_growth_high_inflation",
                    "state2_high_growth_low_inflation", "state3_high_growth_high_inflation"])
                for theme in old.THEMES:
                    pipe = old.pipeline(model, selected[model, theme], thresholds)
                    pipe.fit(training[old.DESIGN], training[theme+"_next"])
                    ridge = pipe.named_steps["ridge"]
                    coef = np.asarray(ridge.coef_)
                    benchmark = float(training[theme+"_next"].mean())
                    prediction = float(pipe.predict(test[old.DESIGN])[0])
                    deviation = prediction-benchmark
                    assert abs(float(ridge.intercept_)-benchmark)<1e-12
                    paths.append(dict(country=country, model=model, theme=theme,
                        signal_date=old.date(date), target_date=old.date(test.target_date.iloc[0]),
                        train_n=len(training), selected_alpha=selected[model, theme],
                        actual_decimal=float(test[theme+"_next"].iloc[0]), prediction_decimal=prediction,
                        matched_training_mean_decimal=benchmark, deviation_decimal=deviation,
                        deviation_bp=10000*deviation, standardized_coef_l2=float(np.linalg.norm(coef)),
                        standardized_coef_max_abs=float(np.max(np.abs(coef))),
                        intercept_decimal=float(ridge.intercept_),
                        intercept_minus_training_mean=float(ridge.intercept_)-benchmark))
                    scale = pipe.named_steps["scale"]
                    for j, name in enumerate(names):
                        coefs.append(dict(country=country, model=model, theme=theme,
                            signal_date=old.date(date), target_date=old.date(test.target_date.iloc[0]),
                            feature=name, standardized_coefficient_decimal=float(coef[j]),
                            coefficient_bp_per_training_sd=float(coef[j]*10000),
                            training_feature_mean=float(scale.mean_[j]), training_feature_scale=float(scale.scale_[j]),
                            scale_one_can_indicate_constant_column=bool(scale.var_[j]==0)))
    path = pd.DataFrame(paths)
    selection = pd.DataFrame(selections)
    rows = []
    for (country, model, theme), d in path.groupby(["country", "model", "theme"], sort=False):
        d = d.sort_values("signal_date")
        bm = float(np.mean((d.actual_decimal-d.matched_training_mean_decimal)**2))
        mse = float(np.mean((d.actual_decimal-d.prediction_decimal)**2))
        rms = float(np.sqrt(np.mean(d.deviation_decimal**2)))
        mx = float(d.deviation_decimal.abs().max())
        rows.append(dict(country=country, model=model, theme=theme, oos_n=len(d),
            first_oos_target=d.target_date.iloc[0], last_oos_target=d.target_date.iloc[-1],
            oos_r2=1-mse/bm, oos_model_mse=mse, oos_baseline_mse=bm,
            prediction_minus_mean_rms_decimal=rms, prediction_minus_mean_rms_bp=10000*rms,
            prediction_minus_mean_max_abs_decimal=mx, prediction_minus_mean_max_abs_bp=10000*mx,
            rms_deviation_divided_by_baseline_rmse=rms/np.sqrt(bm),
            fraction_predictions_within_1bp=float(d.deviation_bp.abs().le(1).mean()),
            close_to_mean_rms1bp_max5bp=bool(rms*10000<=TOLERANCE_RMS_BP and mx*10000<=TOLERANCE_MAX_BP),
            numerically_equal_to_mean_1e_12=bool(mx<=1e-12),
            standardized_coef_l2_median=float(d.standardized_coef_l2.median()),
            standardized_coef_l2_max=float(d.standardized_coef_l2.max()),
            standardized_coef_l2_last=float(d.standardized_coef_l2.iloc[-1]),
            coefficient_last_signal=d.signal_date.iloc[-1],
            max_abs_intercept_minus_training_mean=float(d.intercept_minus_training_mean.abs().max())))
    diagnostics = selection.merge(pd.DataFrame(rows), on=["country", "model", "theme"], validate="one_to_one")
    check = diagnostics.merge(reference[["country", "model", "theme", "oos_r2"]],
        on=["country", "model", "theme"], suffixes=("_new", "_reference"), validate="one_to_one")
    check["r2_abs_difference"] = (check.oos_r2_new-check.oos_r2_reference).abs()
    assert len(check)==20 and np.allclose(check.oos_r2_new,check.oos_r2_reference,rtol=1e-9,atol=1e-10)
    compare_keys = ["country", "model", "theme", "signal_date", "target_date"]
    pc = path.merge(reference_predictions, on=compare_keys, suffixes=("_new", "_reference"), validate="one_to_one")
    pred_error = float((pc.prediction_decimal_new-pc.prediction_decimal_reference).abs().max())
    mean_error = float((pc.matched_training_mean_decimal-pc.historical_mean_decimal).abs().max())
    assert len(pc)==len(path) and pred_error<1e-10 and mean_error<1e-10
    tolerances = []
    for rms, maximum in [(.1,.5),(1.,5.),(5.,25.)]:
        flag = (diagnostics.prediction_minus_mean_rms_bp<=rms)&(diagnostics.prediction_minus_mean_max_abs_bp<=maximum)
        tolerances.append(dict(rms_tolerance_bp=rms, maximum_tolerance_bp=maximum,
            close_models=int(flag.sum()), all_models=len(flag),
            upper_boundary_close_models=int((flag&diagnostics.alpha_at_upper_boundary).sum()),
            tolerance_role="Declared descriptive scale; not a significance test or optimized cutoff"))
    for filename in old.FILES:
        f = base/"data/international"/filename
        inputs.append(dict(path=str(f.relative_to(base)), sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    tables = {"model_shrinkage_diagnostics": diagnostics, "cv_fold_baselines": pd.DataFrame(cvrows),
        "forecast_and_coefficient_path": path, "standardized_coefficients": pd.DataFrame(coefs),
        "timing_audit": pd.DataFrame(timing), "reproduction_checks": check,
        "tolerance_sensitivity": pd.DataFrame(tolerances), "input_sha256": pd.DataFrame(inputs)}
    for name, table in tables.items(): table.to_csv(out/(name+".csv"), index=False)
    summary = dict(specification="Fixed IP YoY lag2 / CPI YoY lag1 / CLIFS lag2; fixed 20 country/theme/model pipelines",
        n_models=len(diagnostics), alpha_grid=old.ALPHAS,
        alpha_at_upper_boundary_count=int(diagnostics.alpha_at_upper_boundary.sum()),
        close_mean_count=int(diagnostics.close_to_mean_rms1bp_max5bp.sum()),
        numerical_mean_count=int(diagnostics.numerically_equal_to_mean_1e_12.sum()),
        close_mean_rms_tolerance_bp=TOLERANCE_RMS_BP, close_mean_max_tolerance_bp=TOLERANCE_MAX_BP,
        cv_beats_fold_mean_count=int(diagnostics.cv_beats_mean.sum()),
        positive_oos_r2_count=int(diagnostics.oos_r2.gt(0).sum()),
        cv_baseline="Each validation fold compared with an intercept-only mean fitted on exactly that fold's training labels; average fold MSE matches GridSearchCV scoring",
        coefficient_scale="Ridge coefficients after fold/expanding training StandardScaler; y remains decimal return. L2 excludes intercept.",
        coefficient_time="Every OOS fit exported; last signal 2025-11-30 predicts target 2025-12-31",
        thresholds="Fold-fitted for CV; full-initial-training medians frozen for OOS",
        forecast_path_rows=len(path), coefficient_rows=len(coefs), timing_rows=len(timing),
        max_r2_reproduction_error=float(check.r2_abs_difference.max()),
        max_forecast_reproduction_error=pred_error, max_mean_reproduction_error=mean_error,
        tolerance_sensitivity=tolerances, results=diagnostics.to_dict("records"),
        interpretation="Large penalties and near-mean forecasts identify little retained variation in this selected pipeline. Boundary choices are not mathematical equality; near-zero R2 and Holm=1 alone are not evidence that all macro information is absent.",
        limitations=["Descriptive cutoffs are declared for this audit after the original evaluation was observed.",
            "CV comparisons reuse the data that selected alpha and are not independent tests of predictive skill.",
            "Coefficients on standardized correlated predictors are not structural effects.",
            "Current-vintage macro and factor inputs remain unchanged; no new historical-release information is introduced."])
    (out/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["n_models", "alpha_at_upper_boundary_count", "close_mean_count", "cv_beats_fold_mean_count", "max_r2_reproduction_error"]}), flush=True)
    return summary


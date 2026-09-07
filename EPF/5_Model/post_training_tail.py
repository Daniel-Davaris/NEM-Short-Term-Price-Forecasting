"""Leakage-safe post-training models for spike detection and tail-aware prices.

This module only combines predictions from already-trained component models.  It
does not read or modify the component model artefacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def _rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def spike_classification_metrics(labels, alerts, beta=2.0):
    labels = np.asarray(labels, dtype=bool)
    alerts = np.asarray(alerts, dtype=bool)
    true_positive = int(np.count_nonzero(labels & alerts))
    predicted_positive = int(np.count_nonzero(alerts))
    actual_positive = int(np.count_nonzero(labels))
    precision = true_positive / predicted_positive if predicted_positive else np.nan
    recall = true_positive / actual_positive if actual_positive else np.nan
    beta_squared = beta**2
    if np.isfinite(precision) and np.isfinite(recall) and precision + recall:
        f_beta = (1.0 + beta_squared) * precision * recall / (
            beta_squared * precision + recall
        )
    else:
        f_beta = np.nan
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f_beta": float(f_beta),
        "alerts": predicted_positive,
        "actual_spikes": actual_positive,
    }


def price_metrics(actual, predicted, spike_threshold, spike_alerts=None, beta=2.0):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = np.abs(predicted - actual)
    spikes = actual > spike_threshold
    non_spikes = ~spikes
    if spike_alerts is None:
        spike_alerts = predicted > spike_threshold
    classification = spike_classification_metrics(spikes, spike_alerts, beta=beta)
    return {
        "mae": float(np.mean(error)),
        "rmse": _rmse(actual, predicted),
        "spike_mae": float(np.mean(error[spikes])) if spikes.any() else np.nan,
        "non_spike_mae": float(np.mean(error[non_spikes])) if non_spikes.any() else np.nan,
        "spike_precision": classification["precision"],
        "spike_recall": classification["recall"],
        "spike_f_beta": classification["f_beta"],
        "predicted_spikes": classification["alerts"],
        "actual_spikes": classification["actual_spikes"],
    }


def choose_spike_threshold(
    labels, raw_scores, beta=2.0, min_precision=0.60, base_alerts=None
):
    """Choose a stable raw-score threshold on a chronological tuning block."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(raw_scores, dtype=np.float64)
    if base_alerts is None:
        base_alerts = np.zeros(len(labels), dtype=bool)
    else:
        base_alerts = np.asarray(base_alerts, dtype=bool)
    quantiles = np.linspace(0.0, 1.0, 1001)
    thresholds = np.unique(np.quantile(scores, quantiles))
    best = None
    fallback = None
    for threshold in thresholds:
        metrics = spike_classification_metrics(
            labels, (scores >= threshold) | base_alerts, beta=beta
        )
        if not np.isfinite(metrics["f_beta"]):
            continue
        rank = (metrics["f_beta"], metrics["recall"], metrics["precision"])
        if fallback is None or rank > fallback[0]:
            fallback = (rank, float(threshold), metrics)
        if metrics["precision"] >= min_precision and (best is None or rank > best[0]):
            best = (rank, float(threshold), metrics)
    selected = best or fallback
    if selected is None:
        raise ValueError("Cannot select a spike threshold from an empty score vector")
    return selected[1], selected[2]


def apply_spike_router(
    central_prediction,
    component_predictions,
    router,
    spike_threshold,
):
    """Return tail-aware price, calibrated probability, and spike alert."""
    central = np.asarray(central_prediction, dtype=np.float32)
    raw_probability = np.asarray(
        component_predictions["spike_probability"], dtype=np.float32
    )
    calibrated_probability = router["probability_calibrator"].predict(
        raw_probability
    ).astype(np.float32)
    spike_alert = (
        (raw_probability >= router["raw_probability_threshold"])
        | (central > spike_threshold)
    )

    candidate = np.asarray(
        component_predictions[router["candidate_key"]], dtype=np.float32
    )
    candidate = np.maximum(candidate, central)
    routed = central + np.float32(router["candidate_weight"]) * (candidate - central)
    tail_prediction = central.copy()
    tail_prediction[spike_alert] = np.maximum(
        routed[spike_alert], np.float32(spike_threshold + 0.01)
    )
    return tail_prediction, calibrated_probability, spike_alert


def _choose_tail_configuration(
    actual,
    central,
    component_predictions,
    raw_probability,
    raw_threshold,
    spike_threshold,
    beta,
    spike_error_weight,
    max_overall_mae_degradation,
    max_non_spike_mae_degradation,
):
    labels = actual > spike_threshold
    alerts = (
        (raw_probability >= raw_threshold)
        | (central > spike_threshold)
    )
    central_metrics = price_metrics(
        actual, central, spike_threshold, beta=beta
    )
    weights = np.where(labels, spike_error_weight, 1.0)
    configurations = []
    for candidate_key, candidate_weights in {
        "spike_mae": (0.0, 0.25, 0.50, 0.75, 1.0),
        "spike_q90": (0.10, 0.25, 0.50),
    }.items():
        candidate = np.maximum(
            np.asarray(component_predictions[candidate_key], dtype=np.float32),
            central,
        )
        for candidate_weight in candidate_weights:
            routed = central + np.float32(candidate_weight) * (candidate - central)
            prediction = central.copy()
            prediction[alerts] = np.maximum(
                routed[alerts], np.float32(spike_threshold + 0.01)
            )
            metrics = price_metrics(
                actual,
                prediction,
                spike_threshold,
                spike_alerts=alerts,
                beta=beta,
            )
            weighted_mae = float(
                np.average(np.abs(prediction - actual), weights=weights)
            )
            feasible = (
                metrics["mae"]
                <= central_metrics["mae"] * (1.0 + max_overall_mae_degradation)
                and metrics["non_spike_mae"]
                <= central_metrics["non_spike_mae"]
                * (1.0 + max_non_spike_mae_degradation)
            )
            configurations.append(
                {
                    "candidate_key": candidate_key,
                    "candidate_weight": float(candidate_weight),
                    "weighted_mae": weighted_mae,
                    "feasible": bool(feasible),
                    **metrics,
                }
            )

    feasible = [item for item in configurations if item["feasible"]]
    pool = feasible or configurations
    selected = min(
        pool,
        key=lambda item: (
            item["weighted_mae"],
            item["spike_mae"],
            item["mae"],
        ),
    )
    return selected


def fit_horizon_postprocessor(
    *,
    horizon,
    meta,
    anchor,
    actual,
    index,
    component_predictions,
    test_start,
    horizon_granularity_minutes,
    feature_granularity_minutes,
    stacker_params,
    early_stopping_rounds=60,
    train_fraction=0.60,
    tune_fraction=0.20,
    spike_threshold=150.0,
    spike_beta=2.0,
    minimum_spike_precision=0.60,
    spike_error_weight=4.0,
    max_overall_mae_degradation=0.05,
    max_non_spike_mae_degradation=0.10,
):
    """Fit a central residual stacker and a calibrated tail-risk router.

    Validation is divided chronologically into train, tune, and untouched audit
    blocks. Horizon-sized gaps prevent overlapping target windows from crossing
    either split. The final models are then refit on every target known by the
    test boundary.
    """
    lead = pd.Timedelta(minutes=horizon * horizon_granularity_minutes)
    eligible = np.asarray(index + lead <= test_start)
    eligible &= np.isfinite(actual) & np.isfinite(anchor)
    eligible &= np.isfinite(meta).all(axis=1)

    X = np.asarray(meta[eligible], dtype=np.float32)
    y = np.asarray(actual[eligible], dtype=np.float32)
    base = np.asarray(anchor[eligible], dtype=np.float32)
    dates = index[eligible]
    residual = (y - base).astype(np.float32)
    components = {
        key: np.asarray(values[eligible], dtype=np.float32)
        for key, values in component_predictions.items()
    }
    raw_probability = np.clip(
        components["spike_probability"], np.float32(1e-6), np.float32(1.0 - 1e-6)
    )
    labels = y > spike_threshold

    tune_start = int(round(len(y) * train_fraction))
    audit_start = int(round(len(y) * (train_fraction + tune_fraction)))
    purge_rows = int(
        np.ceil(
            horizon
            * horizon_granularity_minutes
            / feature_granularity_minutes
        )
    )
    train_stop = tune_start - purge_rows
    tune_stop = audit_start - purge_rows
    if train_stop < 10_000 or tune_stop - tune_start < 2_000 or len(y) - audit_start < 2_000:
        raise ValueError(
            f"Insufficient post-training data for h{horizon}: train={train_stop}, "
            f"tune={tune_stop - tune_start}, audit={len(y) - audit_start}, "
            f"purge={purge_rows}"
        )

    selector = lgb.LGBMRegressor(**stacker_params)
    selector.fit(
        X[:train_stop],
        residual[:train_stop],
        eval_set=[(X[tune_start:tune_stop], residual[tune_start:tune_stop])],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    best_iteration = int(selector.best_iteration_ or stacker_params["n_estimators"])
    selector_prediction = (
        base + selector.predict(X, num_iteration=best_iteration)
    ).astype(np.float32)

    tune_slice = slice(tune_start, tune_stop)
    audit_slice = slice(audit_start, len(y))
    raw_threshold, threshold_tune_metrics = choose_spike_threshold(
        labels[tune_slice],
        raw_probability[tune_slice],
        beta=spike_beta,
        min_precision=minimum_spike_precision,
        base_alerts=selector_prediction[tune_slice] > spike_threshold,
    )
    tune_components = {
        key: values[tune_slice] for key, values in components.items()
    }
    route = _choose_tail_configuration(
        y[tune_slice],
        selector_prediction[tune_slice],
        tune_components,
        raw_probability[tune_slice],
        raw_threshold,
        spike_threshold,
        spike_beta,
        spike_error_weight,
        max_overall_mae_degradation,
        max_non_spike_mae_degradation,
    )

    calibration_selector = IsotonicRegression(
        out_of_bounds="clip", y_min=0.0, y_max=1.0
    )
    calibration_selector.fit(
        raw_probability[:train_stop], labels[:train_stop].astype(np.float32)
    )
    audit_probability = calibration_selector.predict(raw_probability[audit_slice])

    final_params = dict(stacker_params)
    final_params["n_estimators"] = best_iteration
    stacker = lgb.LGBMRegressor(**final_params)
    stacker.fit(X, residual)

    probability_calibrator = IsotonicRegression(
        out_of_bounds="clip", y_min=0.0, y_max=1.0
    )
    probability_calibrator.fit(raw_probability, labels.astype(np.float32))
    router = {
        "raw_probability_threshold": raw_threshold,
        "candidate_key": route["candidate_key"],
        "candidate_weight": route["candidate_weight"],
        "probability_calibrator": probability_calibrator,
    }

    audit_components = {
        key: values[audit_slice] for key, values in components.items()
    }
    audit_tail, _, audit_alert = apply_spike_router(
        selector_prediction[audit_slice],
        audit_components,
        router,
        spike_threshold,
    )
    audit_actual = y[audit_slice]
    audit_base = base[audit_slice]
    central_metrics = price_metrics(
        audit_actual,
        selector_prediction[audit_slice],
        spike_threshold,
        beta=spike_beta,
    )
    tail_metrics = price_metrics(
        audit_actual,
        audit_tail,
        spike_threshold,
        spike_alerts=audit_alert,
        beta=spike_beta,
    )
    audit_labels = labels[audit_slice]
    diagnostics = {
        "horizon": horizon,
        "lead_hours": horizon * horizon_granularity_minutes / 60.0,
        "eligible_rows": int(len(y)),
        "internal_train_rows": int(train_stop),
        "internal_tune_rows": int(tune_stop - tune_start),
        "purged_rows_per_boundary": int(purge_rows),
        "internal_audit_rows": int(len(y) - audit_start),
        "audit_start": dates[audit_start],
        "best_iteration": best_iteration,
        "base_audit_mae": float(mean_absolute_error(audit_actual, audit_base)),
        "central_audit_mae": central_metrics["mae"],
        "tail_audit_mae": tail_metrics["mae"],
        "base_audit_rmse": _rmse(audit_actual, audit_base),
        "central_audit_rmse": central_metrics["rmse"],
        "tail_audit_rmse": tail_metrics["rmse"],
        "central_audit_spike_mae": central_metrics["spike_mae"],
        "tail_audit_spike_mae": tail_metrics["spike_mae"],
        "central_audit_spike_precision": central_metrics["spike_precision"],
        "central_audit_spike_recall": central_metrics["spike_recall"],
        "central_audit_spike_f2": central_metrics["spike_f_beta"],
        "tail_audit_spike_precision": tail_metrics["spike_precision"],
        "tail_audit_spike_recall": tail_metrics["spike_recall"],
        "tail_audit_spike_f2": tail_metrics["spike_f_beta"],
        "raw_probability_threshold": raw_threshold,
        "tune_threshold_precision": threshold_tune_metrics["precision"],
        "tune_threshold_recall": threshold_tune_metrics["recall"],
        "tune_threshold_f2": threshold_tune_metrics["f_beta"],
        "tail_candidate_key": route["candidate_key"],
        "tail_candidate_weight": route["candidate_weight"],
        "audit_spike_average_precision": float(
            average_precision_score(audit_labels, raw_probability[audit_slice])
        ),
        "audit_spike_roc_auc": float(
            roc_auc_score(audit_labels, raw_probability[audit_slice])
        ),
        "audit_raw_probability_brier": float(
            brier_score_loss(audit_labels, raw_probability[audit_slice])
        ),
        "audit_calibrated_probability_brier": float(
            brier_score_loss(audit_labels, audit_probability)
        ),
    }
    diagnostics["central_mae_skill_vs_base_pct"] = 100.0 * (
        diagnostics["base_audit_mae"] - diagnostics["central_audit_mae"]
    ) / diagnostics["base_audit_mae"]
    diagnostics["tail_spike_mae_skill_vs_central_pct"] = 100.0 * (
        diagnostics["central_audit_spike_mae"]
        - diagnostics["tail_audit_spike_mae"]
    ) / diagnostics["central_audit_spike_mae"]
    del selector
    return stacker, router, diagnostics

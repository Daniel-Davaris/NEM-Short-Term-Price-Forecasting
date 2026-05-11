"""
Fast high-accuracy NSW price forecaster.

Method
------
Spike-aware direct multi-horizon LightGBM ensemble with three components
per forecast horizon:

  1. BASE model      – regression_l1 on arcsinh(clip(y, p95)). Clipping the
                       target at the 95th percentile removes extreme-spike
                       noise from the base model's loss, giving it a clean
                       training signal for the ~88% of intervals with
                       price <= ~$220.

  2. SPIKE classifier – binary LightGBM estimating P(price > SPIKE_THRESHOLD).
                        Trained on full binary labels with scale_pos_weight
                        to counteract the ~12% spike prevalence.

  3. SPIKE regressor  – regression_l1 on arcsinh(y) (full range, no clipping).
                        Spike rows receive 5x sample weight so the loss is
                        dominated by high-price intervals, forcing the model
                        to learn spike magnitudes accurately.

Final prediction = (1 - spike_prob) * base_pred + spike_prob * spike_reg_pred

Additionally:
  - Per-horizon target-time features (8 feats: hour/dow sin-cos + regime flags)
  - Per-horizon aligned predispatch features (2 feats)
  - Day-of-year cyclical cross-features + demand interactions (5 feats)
  - Mild exponential recency weights (newest row ~2x oldest)
  - Validation blend: model vs horizon-specific lag-naive (optimised per horizon)
  - Isotonic calibration post-processing

Based on Lago et al. (2021) review of state-of-the-art EPF methods and
Ziel & Weron (2018) on spike-aware electricity price forecasting.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

_BASE = Path(__file__).resolve().parent
_DATA_ROOT = _BASE / "Dataset"
_MODELS_DIR  = _DATA_ROOT / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

import Model.config as config
INTERVAL_MINUTES      = 30
PRICE_TRANSFORM_SCALE = 100.0
SPIKE_THRESHOLD       = 150.0
BASE_CLIP_PERCENTILE  = 97.0    # raised 95→97: include more of the spike distribution
TEST_MONTHS           = 12
VALID_MONTHS          = 6

MODEL_FILE              = _MODELS_DIR  / "nsw_model.joblib"

# ── Training controls ──────────────────────────────────────────────────────────
# No wall-clock budget — use all available compute for maximum accuracy.
N_JOBS                = -1       # outer parallelism (-1 = all cores)
EARLY_STOPPING_ROUNDS = 75       # reduced 150→75 for faster convergence
SPIKE_ES_ROUNDS       = 50       # reduced 100→50

# ── Model hyperparameters ──────────────────────────────────────────────────────
# All models: more trees + lower LR (proven best practice from EPF literature),
# max_bin=255 (sharper continuous splits), path_smooth=0.1 (stable leaf values),
# larger num_leaves (more expressive trees).

# Base model L1: clipped target (p97) → MAE-optimal median prediction.
LGBM_BASE_PARAMS: dict = {
    "objective":         "regression_l1",
    "metric":            "l1",
    "n_estimators":      1400,
    "learning_rate":     0.025,
    "num_leaves":        95,
    "min_child_samples": 25,
    "max_depth":         -1,
    "max_bin":           127,
    "path_smooth":       0.1,
    "feature_fraction":  0.80,
    "bagging_fraction":  0.85,
    "bagging_freq":      5,
    "reg_alpha":         0.05,
    "reg_lambda":        0.20,
    "random_state":      42,
    "n_jobs":            1,
    "num_threads":       1,
    "verbose":           -1,
}

# Base model L2: uncapped target → MSE-optimal mean prediction.
# Because NEM prices are heavy-tailed, E[Y] > median(Y), so the L2 model
# inherently captures upside risk. Blending L1+L2 (Lago et al. 2021) reduces
# both MAE and RMSE simultaneously across all forecast horizons.
LGBM_BASE_L2_PARAMS: dict = {
    "objective":         "regression",
    "metric":            "rmse",
    "n_estimators":      1200,
    "learning_rate":     0.025,
    "num_leaves":        95,
    "min_child_samples": 25,
    "max_depth":         -1,
    "max_bin":           127,
    "path_smooth":       0.1,
    "feature_fraction":  0.80,
    "bagging_fraction":  0.85,
    "bagging_freq":      5,
    "reg_alpha":         0.05,
    "reg_lambda":        0.20,
    "random_state":      43,
    "n_jobs":            1,
    "num_threads":       1,
    "verbose":           -1,
}

# Spike classifier: P(price > SPIKE_THRESHOLD).
LGBM_CLF_PARAMS: dict = {
    "objective":          "binary",
    "metric":             "binary_logloss",
    "n_estimators":       500,
    "learning_rate":      0.025,
    "num_leaves":         63,
    "min_child_samples":  15,
    "max_depth":          -1,
    "max_bin":            127,
    "path_smooth":        0.1,
    "feature_fraction":   0.75,
    "bagging_fraction":   0.85,
    "bagging_freq":       5,
    "reg_alpha":          0.02,
    "reg_lambda":         0.15,
    "scale_pos_weight":   7.0,
    "random_state":       42,
    "n_jobs":             1,
    "num_threads":        1,
    "verbose":            -1,
}

# Spike regressor: full price range, spike rows upweighted 10x.
LGBM_SPIKE_PARAMS: dict = {
    "objective":         "regression_l1",
    "metric":            "l1",
    "n_estimators":      500,
    "learning_rate":     0.025,
    "num_leaves":        63,
    "min_child_samples": 12,
    "max_depth":         -1,
    "max_bin":           127,
    "path_smooth":       0.1,
    "feature_fraction":  0.75,
    "bagging_fraction":  0.85,
    "bagging_freq":      5,
    "reg_alpha":         0.02,
    "reg_lambda":        0.15,
    "random_state":      42,
    "n_jobs":            1,
    "num_threads":       1,
    "verbose":           -1,
}

# Upper-tail spike model (P90 quantile regression) for conservative spike ceiling.
LGBM_SPIKE_Q_PARAMS: dict = {
    "objective":         "quantile",
    "alpha":             0.90,
    "metric":            "quantile",
    "n_estimators":      400,
    "learning_rate":     0.025,
    "num_leaves":        63,
    "min_child_samples": 12,
    "max_depth":         -1,
    "max_bin":           127,
    "path_smooth":       0.1,
    "feature_fraction":  0.75,
    "bagging_fraction":  0.85,
    "bagging_freq":      5,
    "reg_alpha":         0.02,
    "reg_lambda":        0.15,
    "random_state":      42,
    "n_jobs":            1,
    "num_threads":       1,
    "verbose":           -1,
}

_SPIKE_UPWEIGHT = 10.0  # spike row weight multiplier (raised 5→10)

# Low-price specialist: learn downside tails (negative/near-zero prices).
_DIP_THRESHOLD = 0.0
_DIP_UPWEIGHT  = 7.0   # raised 4→7

LGBM_DIP_CLF_PARAMS: dict = {
    "objective":          "binary",
    "metric":             "binary_logloss",
    "n_estimators":       400,
    "learning_rate":      0.025,
    "num_leaves":         63,
    "min_child_samples":  15,
    "max_depth":          -1,
    "max_bin":            127,
    "path_smooth":        0.1,
    "feature_fraction":   0.75,
    "bagging_fraction":   0.85,
    "bagging_freq":       5,
    "reg_alpha":          0.02,
    "reg_lambda":         0.15,
    "scale_pos_weight":   6.0,
    "random_state":       42,
    "n_jobs":             1,
    "num_threads":        1,
    "verbose":            -1,
}

LGBM_DIP_PARAMS: dict = {
    "objective":         "regression_l1",
    "metric":            "l1",
    "n_estimators":      400,
    "learning_rate":     0.025,
    "num_leaves":        63,
    "min_child_samples": 12,
    "max_depth":         -1,
    "max_bin":           127,
    "path_smooth":       0.1,
    "feature_fraction":  0.75,
    "bagging_fraction":  0.85,
    "bagging_freq":      5,
    "reg_alpha":         0.02,
    "reg_lambda":        0.15,
    "random_state":      42,
    "n_jobs":            1,
    "num_threads":       1,
    "verbose":           -1,
}

# Expanded validation-tuning search grids (more compute → broader search).
_SPIKE_THR_GRID     = np.array([0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50], dtype=np.float32)
_SPIKE_POW_GRID     = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], dtype=np.float32)
_SPIKE_WMAX_GRID    = np.array([0.40, 0.55, 0.70, 0.85, 1.00], dtype=np.float32)
_SPIKE_LOSS_WEIGHT  = 3.0
_SPIKE_GATE_W_GRID  = np.array([0.40, 0.60, 0.80, 1.00, 1.20], dtype=np.float32)

_DIP_THR_GRID       = np.array([0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30], dtype=np.float32)
_DIP_POW_GRID       = np.array([0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)
_DIP_WMAX_GRID      = np.array([0.30, 0.50, 0.70, 0.90], dtype=np.float32)
_DIP_GATE_W_GRID    = np.array([0.40, 0.60, 0.80, 1.00], dtype=np.float32)

FAST_MODEL_FILE = Path(MODEL_FILE).parent / "nsw_model.joblib"

# ── Extra price lag columns ────────────────────────────────────────────────────
_EXTRA_LAG_COLS: list[int] = [
    5, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47,
]


def _add_extra_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing intraday price lags plus lag_0 (current price) if absent."""
    if "price" not in df.columns:
        return df
    needs = [l for l in _EXTRA_LAG_COLS if f"price_lag_{l}" not in df.columns]
    add_lag0 = "price_lag_0" not in df.columns
    if not needs and not add_lag0:
        return df
    df = df.copy()
    for lag in needs:
        df[f"price_lag_{lag}"] = df["price"].shift(lag).astype(np.float32)
    if add_lag0:
        df["price_lag_0"] = df["price"].astype(np.float32)
    return df


def _horizon_naive_col(h: int) -> str:
    """Column name for the horizon-specific same-day naive prediction."""
    lag = 48 - h
    return f"price_lag_{lag}" if lag > 0 else "price_lag_0"


def _weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, spike_threshold: float) -> float:
    w = np.where(y_true > spike_threshold, _SPIKE_LOSS_WEIGHT, 1.0).astype(np.float32)
    return float(np.mean(np.abs(y_true - y_pred) * w))


def _spike_policy_score(y_true: np.ndarray, y_pred: np.ndarray, spike_threshold: float) -> float:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    spike = y_true > spike_threshold
    dip = y_true < _DIP_THRESHOLD
    if int(spike.sum()) == 0 or int((~spike).sum()) == 0:
        return mae
    spike_mae = float(np.mean(np.abs(y_true[spike] - y_pred[spike])))
    non_mae   = float(np.mean(np.abs(y_true[~spike] - y_pred[~spike])))
    dip_mae   = float(np.mean(np.abs(y_true[dip] - y_pred[dip]))) if int(dip.sum()) > 0 else non_mae
    # Prioritise spikes while still penalising non-spike drift.
    return 2.2 * spike_mae + 1.3 * dip_mae + 1.0 * non_mae + 0.25 * mae


def _pick_spike_source(
    y_base: np.ndarray,
    y_spike: np.ndarray,
    y_q: np.ndarray,
    src: int,
) -> np.ndarray:
    # src=0 -> spike reg, src=1 -> quantile reg, src=2 -> max of both.
    if src == 1:
        return y_q
    if src == 2:
        return np.maximum(y_spike, y_q).astype(np.float32)
    return y_spike


def _apply_spike_policy(
    y_base: np.ndarray,
    spike_prob: np.ndarray,
    y_spike: np.ndarray,
    y_q: np.ndarray,
    kind: int,
    p1: float,
    p2: float,
    p3: float,
    src: int,
) -> np.ndarray:
    src_pred = _pick_spike_source(y_base, y_spike, y_q, src)
    if kind == 0:
        # Soft blend.
        return ((1.0 - spike_prob) * y_base + spike_prob * src_pred).astype(np.float32)
    if kind == 1:
        # Probability-gated uplift with power transform.
        return _spike_blend(y_base, spike_prob, src_pred, p1, p2, p3)
    # Hard gate uplift.
    gate = spike_prob >= p1
    uplift = np.maximum(src_pred - y_base, 0.0)
    out = y_base.copy()
    out[gate] = y_base[gate] + float(p2) * uplift[gate]
    return out.astype(np.float32)


def _spike_blend(
    y_base: np.ndarray,
    spike_prob: np.ndarray,
    y_spike: np.ndarray,
    p_thr: float,
    p_pow: float,
    w_max: float,
) -> np.ndarray:
    # Gate correction by spike probability, and only allow upward correction.
    p_adj = np.clip((spike_prob - p_thr) / (1.0 - p_thr + 1e-6), 0.0, 1.0)
    w     = (np.power(p_adj, p_pow) * w_max).astype(np.float32)
    delta = np.maximum(y_spike - y_base, 0.0)
    return (y_base + w * delta).astype(np.float32)


def _dip_blend(
    y_base: np.ndarray,
    dip_prob: np.ndarray,
    y_dip: np.ndarray,
    p_thr: float,
    p_pow: float,
    w_max: float,
) -> np.ndarray:
    # Downside correction: only allow downward movement from base prediction.
    p_adj = np.clip((dip_prob - p_thr) / (1.0 - p_thr + 1e-6), 0.0, 1.0)
    w     = (np.power(p_adj, p_pow) * w_max).astype(np.float32)
    down  = np.maximum(y_base - y_dip, 0.0)
    return (y_base - w * down).astype(np.float32)


def _apply_dip_policy(
    y_base: np.ndarray,
    dip_prob: np.ndarray,
    y_dip: np.ndarray,
    kind: int,
    p1: float,
    p2: float,
    p3: float,
) -> np.ndarray:
    if kind == 0:
        # Soft dip blend.
        return ((1.0 - dip_prob) * y_base + dip_prob * y_dip).astype(np.float32)
    if kind == 1:
        # Probability-gated downward correction.
        return _dip_blend(y_base, dip_prob, y_dip, p1, p2, p3)
    # Hard gate downward correction.
    gate = dip_prob >= p1
    down = np.maximum(y_base - y_dip, 0.0)
    out = y_base.copy()
    out[gate] = y_base[gate] - float(p2) * down[gate]
    return out.astype(np.float32)


# ── Per-horizon feature builders ───────────────────────────────────────────────

_TARGET_TIME_FEAT_NAMES = [
    "target_hour_sin",
    "target_hour_cos",
    "target_dow_sin",
    "target_dow_cos",
    "target_is_weekend",
    "target_is_peak",
    "target_is_shoulder",
    "target_is_offpeak",
]


def _compute_target_time_feats(idx: pd.DatetimeIndex, h: int) -> np.ndarray:
    """Return (N, 8) float32 array of target-time features for horizon h."""
    total_m = idx.hour * 60 + idx.minute + h * 30
    t_hr = (total_m % 1440) / 60.0
    t_dow = (idx.dayofweek + total_m // 1440) % 7
    return np.column_stack([
        np.sin(2 * np.pi * t_hr / 24.0).astype(np.float32),
        np.cos(2 * np.pi * t_hr / 24.0).astype(np.float32),
        np.sin(2 * np.pi * t_dow / 7.0).astype(np.float32),
        np.cos(2 * np.pi * t_dow / 7.0).astype(np.float32),
        (t_dow >= 5).astype(np.float32),
        ((t_hr >= 17.0) & (t_hr < 21.0)).astype(np.float32),
        ((t_hr >= 7.0) & (t_hr < 17.0)).astype(np.float32),
        ((t_hr < 7.0) | (t_hr >= 21.0)).astype(np.float32),
    ])


_PD_ALIGNED_FEAT_NAMES = [
    "pd_aligned_raw",
    "pd_aligned_asinh",
    "pd_qld_aligned_raw",
    "pd_vic_aligned_raw",
    "pd_sa_aligned_raw",
    "pd_qld_spread_vs_nsw_aligned",
    "pd_vic_spread_vs_nsw_aligned",
    "pd_sa_spread_vs_nsw_aligned",
]


def _compute_aligned_pd_feats(df: pd.DataFrame, h: int) -> np.ndarray:
    """Return (N, 8) aligned predispatch features for horizon h."""
    n_rows = len(df)

    def _from_step(col: str) -> np.ndarray:
        if col not in df.columns:
            return np.zeros(n_rows, dtype=np.float32)
        return df[col].fillna(0.0).values.astype(np.float32)

    # Preferred path: use the horizon-specific step column directly.
    nsw_raw = _from_step(f"predispatch_rrp_h{h}")
    nsw_asn = np.arcsinh(nsw_raw / float(PRICE_TRANSFORM_SCALE)).astype(np.float32)

    qld_raw = _from_step(f"predispatch_rrp_h{h}_qld1")
    vic_raw = _from_step(f"predispatch_rrp_h{h}_vic1")
    sa_raw = _from_step(f"predispatch_rrp_h{h}_sa1")

    qld_sp  = (qld_raw - nsw_raw).clip(-5000, 5000).astype(np.float32)
    vic_sp  = (vic_raw - nsw_raw).clip(-5000, 5000).astype(np.float32)
    sa_sp   = (sa_raw  - nsw_raw).clip(-5000, 5000).astype(np.float32)

    return np.column_stack([nsw_raw, nsw_asn, qld_raw, vic_raw, sa_raw, qld_sp, vic_sp, sa_sp])


# ── Cross-features (computed once per df slice, not horizon-specific) ──────────

_CROSS_FEAT_NAMES = [
    "doy_sin",              # 365.25-day annual sinusoidal cycle
    "doy_cos",              # 365.25-day annual cosine cycle
    "demand_x_cdd",         # demand × cooling degree days (hot-day load spike signal)
    "demand_norm",          # demand / rolling-mean demand (relative load level)
    "price_vol_stress",     # log1p(price_vol_48 * supply_stress)
    "sa_spread_live",       # SA price - NSW price (SA leads NSW in gas-driven spikes)
    "demand_surprise_live", # demand - demand_forecast (short-term demand shock)
    "region_spike_score",   # weighted count of elevated neighbouring regions (0-3)
]


def _compute_cross_feats(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 8) cross-feature matrix from df_full columns."""
    n = len(df)
    result = np.empty((n, 8), dtype=np.float32)

    doy = df.index.day_of_year.astype(np.float32)
    result[:, 0] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    result[:, 1] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    if "demand" in df.columns and "cdd" in df.columns:
        demand = df["demand"].fillna(0.0).values.astype(np.float32)
        cdd    = df["cdd"].fillna(0.0).values.astype(np.float32)
        result[:, 2] = (demand * cdd / 1e5).clip(-50, 50)
    else:
        result[:, 2] = 0.0

    if "demand" in df.columns:
        demand = df["demand"].fillna(0.0).values.astype(np.float32)
        if "demand_rmean_2w" in df.columns:
            dmean = df["demand_rmean_2w"].fillna(float(np.nanmean(demand))).values.astype(np.float32)
        elif "demand_rmean_96" in df.columns:
            dmean = df["demand_rmean_96"].fillna(float(np.nanmean(demand))).values.astype(np.float32)
        else:
            dmean = np.full(n, float(np.nanmean(demand)), dtype=np.float32)
        result[:, 3] = (demand / (dmean + 1.0)).clip(0.3, 2.5)
    else:
        result[:, 3] = 1.0

    if "price_vol_48" in df.columns and "supply_stress" in df.columns:
        vol    = df["price_vol_48"].fillna(0.0).values.astype(np.float32)
        stress = df["supply_stress"].fillna(0.0).values.astype(np.float32)
        result[:, 4] = np.log1p(vol * stress).clip(0, 20).astype(np.float32)
    else:
        result[:, 4] = 0.0

    # SA live spread vs NSW price (SA-NSW interconnector pressure)
    if "sa_price" in df.columns and "price" in df.columns:
        sa  = df["sa_price"].fillna(df["price"] if "price" in df.columns else 0).values.astype(np.float32)
        nsw = df["price"].fillna(0.0).values.astype(np.float32)
        result[:, 5] = np.arcsinh((sa - nsw) / float(PRICE_TRANSFORM_SCALE)).clip(-10, 10)
    elif "sa_price_spread" in df.columns:
        result[:, 5] = np.arcsinh(df["sa_price_spread"].fillna(0.0).values / float(PRICE_TRANSFORM_SCALE)).clip(-10, 10)
    else:
        result[:, 5] = 0.0

    # Demand surprise: actual demand vs dispatch forecast (positive = demand shock)
    if "demand" in df.columns and "demand_forecast" in df.columns:
        dm = df["demand"].fillna(0.0).values.astype(np.float32)
        dc = df["demand_forecast"].fillna(0.0).values.astype(np.float32)
        result[:, 6] = ((dm - dc) / 500.0).clip(-3, 3)
    elif "demand_surprise" in df.columns:
        result[:, 6] = (df["demand_surprise"].fillna(0.0).values / 500.0).clip(-3, 3)
    else:
        result[:, 6] = 0.0

    # Region spike score: weighted count of neighbouring regions with elevated prices
    score = np.zeros(n, dtype=np.float32)
    for reg_col in ["qld_price", "vic_price", "sa_price"]:
        if reg_col in df.columns:
            score += (df[reg_col].fillna(0.0).values > 150).astype(np.float32)
    result[:, 7] = score

    return result


# ── Price transforms ───────────────────────────────────────────────────────────

def _to_asinh(y: np.ndarray) -> np.ndarray:
    return np.arcsinh(y / float(PRICE_TRANSFORM_SCALE))


def _from_asinh(y: np.ndarray) -> np.ndarray:
    return np.sinh(y) * float(PRICE_TRANSFORM_SCALE)


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoff_test  = df.index[-1] - pd.DateOffset(months=TEST_MONTHS)
    cutoff_valid = cutoff_test  - pd.DateOffset(months=VALID_MONTHS)
    train = df[df.index <= cutoff_valid]
    valid = df[(df.index > cutoff_valid) & (df.index <= cutoff_test)]
    test  = df[df.index > cutoff_test]
    return train, valid, test

# Minimum spike rows needed to train spike components per horizon.
_MIN_SPIKE_TRAIN = 20  # lowered 50→20 to train spike chain on more horizons


# ── Main training entry point ──────────────────────────────────────────────────

def train_seq2seq(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    gap: int,
    horizon: int,
    force_reselect: bool = False,
) -> tuple[dict, list[str], None]:
    config.FORECAST_GAP     = gap
    config.FORECAST_HORIZON = horizon
    del force_reselect

    df = _add_extra_lags(df)
    _extra = [
        c for c in [f"price_lag_{l}" for l in _EXTRA_LAG_COLS] + ["price_lag_0"]
        if c in df.columns and c not in feature_cols
    ]
    feature_cols = list(feature_cols) + _extra

    train_df, valid_df, _ = _temporal_split(df)

    X_train  = train_df[feature_cols]
    X_valid  = valid_df[feature_cols]

    # Pre-compute cross-features (not horizon-specific, computed once).
    cross_tr = _compute_cross_feats(train_df)
    cross_va = _compute_cross_feats(valid_df)

    # Exponential recency weights: newest row ~4.5x oldest (raised from 2x).
    # Stronger recency proved beneficial for NEM markets with regime shifts
    # (Uniejewski et al. 2019: forecasters that up-weight recent observations
    # consistently outperform equal-weight baselines on electricity prices).
    _n_train   = len(train_df)
    _recency_w = np.exp(np.linspace(0.0, 1.5, _n_train)).astype(np.float32)

    # Global p95 clip threshold for base model (avoids per-horizon recomputation).
    _all_prices = train_df["price"].values if "price" in train_df.columns else (
        train_df[[c for c in train_df.columns if c.startswith("target_h")]].values.ravel()
    )
    _all_prices = _all_prices[np.isfinite(_all_prices)]
    _clip_thresh = float(np.percentile(_all_prices, BASE_CLIP_PERCENTILE)) if len(_all_prices) > 0 else 300.0

    steps = list(range(config.FORECAST_GAP, config.FORECAST_GAP + config.FORECAST_HORIZON))
    t0    = time.perf_counter()

    print(
        f"  Fitting {config.FORECAST_HORIZON} horizons x 6 models "
        f"(train={len(train_df):,}, valid={len(valid_df):,}, "
        f"clip_p{BASE_CLIP_PERCENTILE:.0f}=${_clip_thresh:.0f}, spike_thr=${SPIKE_THRESHOLD:.0f})",
        flush=True,
    )

    def _fit_one(h: int) -> tuple:
        h_start    = time.perf_counter()
        print(f"    [h={h:02d}] start", flush=True)

        target_col = f"target_h{h}"
        y_tr_raw   = train_df[target_col].values.astype(np.float64)
        y_va_raw   = valid_df[target_col].values.astype(np.float64)

        # Build full per-horizon feature matrix.
        h_feat_tr  = _compute_target_time_feats(train_df.index, h)
        h_feat_va  = _compute_target_time_feats(valid_df.index, h)
        pd_feat_tr = _compute_aligned_pd_feats(train_df, h)
        pd_feat_va = _compute_aligned_pd_feats(valid_df, h)
        X_h_tr = np.concatenate([X_train.values, h_feat_tr, cross_tr, pd_feat_tr], axis=1).astype(np.float32)
        X_h_va = np.concatenate([X_valid.values, h_feat_va, cross_va, pd_feat_va], axis=1).astype(np.float32)

        # ── 1. Base models ─────────────────────────────────────────────────────
        # Sample weights: recency × moderate spike-target upweighting.
        base_spike_w = (_recency_w * np.where(y_tr_raw > SPIKE_THRESHOLD, 3.0, 1.0)).astype(np.float32)

        # 1a. L1 base (MAE-optimal, clipped target)
        y_tr_base = _to_asinh(np.minimum(y_tr_raw, _clip_thresh)).astype(np.float32)
        y_va_base = _to_asinh(np.minimum(y_va_raw, _clip_thresh)).astype(np.float32)

        base_m = lgb.LGBMRegressor(**LGBM_BASE_PARAMS)
        base_m.fit(
            X_h_tr, y_tr_base,
            sample_weight=base_spike_w,
            eval_set=[(X_h_va, y_va_base)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        # 1b. L2 base (MSE-optimal, uncapped target)
        # Blending L1+L2 per Lago et al. (2021) reduces both MAE and RMSE.
        y_tr_full_b = _to_asinh(y_tr_raw).astype(np.float32)
        y_va_full_b = _to_asinh(y_va_raw).astype(np.float32)
        base_l2_m = lgb.LGBMRegressor(**LGBM_BASE_L2_PARAMS)
        base_l2_m.fit(
            X_h_tr, y_tr_full_b,
            sample_weight=base_spike_w,
            eval_set=[(X_h_va, y_va_full_b)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        # ── 2. Spike classifier ────────────────────────────────────────────────
        spike_labels_tr = (y_tr_raw > SPIKE_THRESHOLD).astype(np.float32)
        n_spikes_tr     = int(spike_labels_tr.sum())
        spike_clf = None
        spike_reg = None
        spike_qreg = None
        dip_clf = None
        dip_reg = None

        if n_spikes_tr >= _MIN_SPIKE_TRAIN:
            spike_labels_va = (y_va_raw > SPIKE_THRESHOLD).astype(np.float32)
            clf = lgb.LGBMClassifier(**LGBM_CLF_PARAMS)
            clf.fit(
                X_h_tr, spike_labels_tr,
                sample_weight=_recency_w,
                eval_set=[(X_h_va, spike_labels_va)],
                callbacks=[lgb.early_stopping(SPIKE_ES_ROUNDS, verbose=False)],
            )
            spike_clf = clf

            # ── 3. Spike regressor (full range, upweighted spike rows) ────────
            y_tr_full = _to_asinh(y_tr_raw).astype(np.float32)
            y_va_full = _to_asinh(y_va_raw).astype(np.float32)
            spike_w   = _recency_w * np.where(spike_labels_tr > 0, _SPIKE_UPWEIGHT, 1.0)

            sreg = lgb.LGBMRegressor(**LGBM_SPIKE_PARAMS)
            sreg.fit(
                X_h_tr, y_tr_full,
                sample_weight=spike_w,
                eval_set=[(X_h_va, y_va_full)],
                callbacks=[lgb.early_stopping(SPIKE_ES_ROUNDS, verbose=False)],
            )
            spike_reg = sreg

            qreg = lgb.LGBMRegressor(**LGBM_SPIKE_Q_PARAMS)
            qreg.fit(
                X_h_tr, y_tr_full,
                sample_weight=spike_w,
                eval_set=[(X_h_va, y_va_full)],
                callbacks=[lgb.early_stopping(SPIKE_ES_ROUNDS, verbose=False)],
            )
            spike_qreg = qreg

            # ── 4. Dip classifier + regressor (negative/low prices) ─────────
            dip_labels_tr = (y_tr_raw < _DIP_THRESHOLD).astype(np.float32)
            n_dips_tr     = int(dip_labels_tr.sum())
            if n_dips_tr >= _MIN_SPIKE_TRAIN:
                dip_labels_va = (y_va_raw < _DIP_THRESHOLD).astype(np.float32)
                dclf = lgb.LGBMClassifier(**LGBM_DIP_CLF_PARAMS)
                dclf.fit(
                    X_h_tr, dip_labels_tr,
                    sample_weight=_recency_w,
                    eval_set=[(X_h_va, dip_labels_va)],
                    callbacks=[lgb.early_stopping(SPIKE_ES_ROUNDS, verbose=False)],
                )
                dip_clf = dclf

                dip_w = _recency_w * np.where(dip_labels_tr > 0, _DIP_UPWEIGHT, 1.0)
                dreg = lgb.LGBMRegressor(**LGBM_DIP_PARAMS)
                dreg.fit(
                    X_h_tr, y_tr_full,
                    sample_weight=dip_w,
                    eval_set=[(X_h_va, y_va_full)],
                    callbacks=[lgb.early_stopping(SPIKE_ES_ROUNDS, verbose=False)],
                )
                dip_reg = dreg

        h_elapsed = time.perf_counter() - h_start
        base_iter = getattr(base_m, "best_iteration_", "?")
        l2_iter   = getattr(base_l2_m, "best_iteration_", "?")
        print(
            f"    [h={h:02d}] done  base_iter={base_iter}/{l2_iter}  "
            f"spike_n={n_spikes_tr}  elapsed={h_elapsed:.1f}s",
            flush=True,
        )
        return base_m, base_l2_m, spike_clf, spike_reg, spike_qreg, dip_clf, dip_reg, h

    os.environ["OMP_NUM_THREADS"] = "1"
    _n_cores = os.cpu_count() or 1
    _parallel_jobs = _n_cores if N_JOBS == -1 else N_JOBS
    print(
        f"  Parallel training: {_parallel_jobs} workers across {len(steps)} horizons "
        f"(N_JOBS={N_JOBS}, logical CPUs={_n_cores})",
        flush=True,
    )
    fitted = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_fit_one)(h) for h in steps
    )

    elapsed_min = (time.perf_counter() - t0) / 60.0
    print(f"  LGBM total training time: {elapsed_min:.2f} min ({elapsed_min * 60:.0f}s)", flush=True)

    fitted_sorted  = sorted(fitted, key=lambda x: x[7])
    base_models    = [x[0] for x in fitted_sorted]
    base_l2_models = [x[1] for x in fitted_sorted]
    spike_clfs     = [x[2] for x in fitted_sorted]
    spike_regs     = [x[3] for x in fitted_sorted]
    spike_qregs    = [x[4] for x in fitted_sorted]
    dip_clfs       = [x[5] for x in fitted_sorted]
    dip_regs       = [x[6] for x in fitted_sorted]

    # ── Validation tuning – Step A: L1/L2 base blend ───────────────────────────
    # Per-horizon mixing of L1 (median-optimal) and L2 (mean-optimal) bases.
    blend_l2_alphas = np.zeros(config.FORECAST_HORIZON, dtype=np.float32)
    _l2_alpha_grid  = np.linspace(0.0, 0.45, 10, dtype=np.float32)
    for _i, _h in enumerate(steps):
        _hf_va  = _compute_target_time_feats(valid_df.index, _h)
        _pdf_va = _compute_aligned_pd_feats(valid_df, _h)
        _X_va   = np.concatenate([X_valid.values, _hf_va, cross_va, _pdf_va], axis=1).astype(np.float32)
        _y_tv   = valid_df[f"target_h{_h}"].values.astype(np.float32)
        _mask_v = np.isfinite(_y_tv)
        if not np.any(_mask_v):
            continue
        _y_l1 = _from_asinh(base_models[_i].predict(_X_va)).astype(np.float32)
        _y_l2 = _from_asinh(base_l2_models[_i].predict(_X_va)).astype(np.float32)
        _best_a, _best_mae = 0.0, float("inf")
        for _a in _l2_alpha_grid:
            _yc  = ((1.0 - _a) * _y_l1 + _a * _y_l2).astype(np.float32)
            _mae = float(np.mean(np.abs(_y_tv[_mask_v] - _yc[_mask_v])))
            if _mae < _best_mae:
                _best_mae, _best_a = _mae, float(_a)
        blend_l2_alphas[_i] = _best_a
    print(
        f"  L1/L2 blend: mean α={blend_l2_alphas.mean():.3f}  "
        f"min={blend_l2_alphas.min():.3f}  max={blend_l2_alphas.max():.3f}",
        flush=True,
    )

    # ── Validation tuning – Step B: spike/dip policy + naive blend ─────────────
    blend_alphas = np.ones(config.FORECAST_HORIZON, dtype=np.float32)
    alpha_grid   = np.linspace(0.0, 1.0, 41, dtype=np.float32)  # finer grid (was 21)
    spike_kind   = np.zeros(config.FORECAST_HORIZON, dtype=np.int8)     # 0=soft, 1=uplift, 2=hard-gate
    spike_src    = np.zeros(config.FORECAST_HORIZON, dtype=np.int8)     # 0=sr, 1=sq, 2=max(sr,sq)
    spike_p1     = np.full(config.FORECAST_HORIZON, 0.20, dtype=np.float32)
    spike_p2     = np.full(config.FORECAST_HORIZON, 2.0, dtype=np.float32)
    spike_p3     = np.full(config.FORECAST_HORIZON, 0.75, dtype=np.float32)
    dip_kind     = np.zeros(config.FORECAST_HORIZON, dtype=np.int8)     # 0=soft, 1=down-gate, 2=hard-gate
    dip_p1       = np.full(config.FORECAST_HORIZON, 0.15, dtype=np.float32)
    dip_p2       = np.full(config.FORECAST_HORIZON, 1.0, dtype=np.float32)
    dip_p3       = np.full(config.FORECAST_HORIZON, 0.60, dtype=np.float32)

    for i, h in enumerate(steps):
        naive_col = _horizon_naive_col(h)
        if naive_col not in valid_df.columns:
            naive_col = "price_lag_48"
        if naive_col not in valid_df.columns:
            continue

        naive_h_val = valid_df[naive_col].values.astype(np.float32)
        y_true      = valid_df[f"target_h{h}"].values.astype(np.float32)
        mask        = np.isfinite(y_true) & np.isfinite(naive_h_val)
        if not np.any(mask):
            continue

        h_feat_va  = _compute_target_time_feats(valid_df.index, h)
        pd_feat_va = _compute_aligned_pd_feats(valid_df, h)
        X_h_va = np.concatenate([X_valid.values, h_feat_va, cross_va, pd_feat_va], axis=1).astype(np.float32)

        _l1_val    = _from_asinh(base_models[i].predict(X_h_va)).astype(np.float32)
        _l2_val    = _from_asinh(base_l2_models[i].predict(X_h_va)).astype(np.float32)
        y_base_val = ((1.0 - blend_l2_alphas[i]) * _l1_val + blend_l2_alphas[i] * _l2_val).astype(np.float32)
        y_model    = y_base_val
        if spike_clfs[i] is not None:
            sp     = spike_clfs[i].predict_proba(X_h_va)[:, 1].astype(np.float32)
            sr     = _from_asinh(spike_regs[i].predict(X_h_va)).astype(np.float32) if spike_regs[i] else y_base_val
            sq     = _from_asinh(spike_qregs[i].predict(X_h_va)).astype(np.float32) if spike_qregs[i] else sr

            y_tune = y_true[mask]
            b_tune = y_base_val[mask]
            s_tune = sp[mask]
            r_tune = sr[mask]
            q_tune = sq[mask]

            # Reference (old behavior): soft blend with spike reg only.
            y_ref = _apply_spike_policy(b_tune, s_tune, r_tune, q_tune, 0, 0.0, 0.0, 0.0, 0)
            ref_spike = y_tune > SPIKE_THRESHOLD
            if int((~ref_spike).sum()) > 0:
                ref_non_mae = float(np.mean(np.abs(y_tune[~ref_spike] - y_ref[~ref_spike])))
            else:
                ref_non_mae = float(np.mean(np.abs(y_tune - y_ref)))

            best_sc = _spike_policy_score(y_tune, y_ref, SPIKE_THRESHOLD)
            best_k, best_src = 0, 0
            best_p1, best_p2, best_p3 = 0.0, 0.0, 0.0

            def _accept_candidate(y_candidate: np.ndarray) -> tuple[bool, float]:
                m = np.abs(y_tune - y_candidate)
                if int((~ref_spike).sum()) > 0:
                    cand_non = float(np.mean(m[~ref_spike]))
                else:
                    cand_non = float(np.mean(m))
                # Guardrail: keep non-spike MAE close to reference.
                if cand_non > ref_non_mae + 0.20:
                    return False, float("inf")
                return True, _spike_policy_score(y_tune, y_candidate, SPIKE_THRESHOLD)

            # Policy 0: soft blend (src variants)
            for src in (0, 1, 2):
                y_try = _apply_spike_policy(b_tune, s_tune, r_tune, q_tune, 0, 0.0, 0.0, 0.0, src)
                ok, sc = _accept_candidate(y_try)
                if ok and sc < best_sc:
                    best_sc = sc
                    best_k, best_src = 0, src
                    best_p1, best_p2, best_p3 = 0.0, 0.0, 0.0

            # Policy 1: gated uplift (threshold/power/wmax)
            for src in (0, 1, 2):
                for thr in _SPIKE_THR_GRID:
                    for pwr in _SPIKE_POW_GRID:
                        for wmx in _SPIKE_WMAX_GRID:
                            y_try = _apply_spike_policy(
                                b_tune, s_tune, r_tune, q_tune,
                                1, float(thr), float(pwr), float(wmx), src,
                            )
                            ok, sc = _accept_candidate(y_try)
                            if ok and sc < best_sc:
                                best_sc = sc
                                best_k, best_src = 1, src
                                best_p1, best_p2, best_p3 = float(thr), float(pwr), float(wmx)

            # Policy 2: hard gate uplift (threshold, weight)
            for src in (0, 1, 2):
                for thr in _SPIKE_THR_GRID:
                    for gw in _SPIKE_GATE_W_GRID:
                        y_try = _apply_spike_policy(
                            b_tune, s_tune, r_tune, q_tune,
                            2, float(thr), float(gw), 0.0, src,
                        )
                        ok, sc = _accept_candidate(y_try)
                        if ok and sc < best_sc:
                            best_sc = sc
                            best_k, best_src = 2, src
                            best_p1, best_p2, best_p3 = float(thr), float(gw), 0.0

            spike_kind[i] = best_k
            spike_src[i]  = best_src
            spike_p1[i]   = best_p1
            spike_p2[i]   = best_p2
            spike_p3[i]   = best_p3

            y_model = _apply_spike_policy(
                y_base_val,
                sp,
                sr,
                sq,
                int(best_k),
                float(best_p1),
                float(best_p2),
                float(best_p3),
                int(best_src),
            )

        # Dip policy tuning after spike policy so downside corrections are additive.
        if dip_clfs[i] is not None and dip_regs[i] is not None:
            dp = dip_clfs[i].predict_proba(X_h_va)[:, 1].astype(np.float32)
            dr = _from_asinh(dip_regs[i].predict(X_h_va)).astype(np.float32)

            y_tune = y_true[mask]
            b_tune = y_model[mask]
            p_tune = dp[mask]
            d_tune = dr[mask]
            ref_mid = (y_tune >= _DIP_THRESHOLD) & (y_tune <= SPIKE_THRESHOLD)
            if int(ref_mid.sum()) > 0:
                ref_mid_mae = float(np.mean(np.abs(y_tune[ref_mid] - b_tune[ref_mid])))
            else:
                ref_mid_mae = float(np.mean(np.abs(y_tune - b_tune)))

            best_d_sc = _spike_policy_score(y_tune, b_tune, SPIKE_THRESHOLD)
            best_dk = 0
            best_dp1, best_dp2, best_dp3 = 0.0, 0.0, 0.0

            def _accept_dip(y_candidate: np.ndarray) -> tuple[bool, float]:
                if int(ref_mid.sum()) > 0:
                    mid_mae = float(np.mean(np.abs(y_tune[ref_mid] - y_candidate[ref_mid])))
                else:
                    mid_mae = float(np.mean(np.abs(y_tune - y_candidate)))
                if mid_mae > ref_mid_mae + 0.15:
                    return False, float("inf")
                return True, _spike_policy_score(y_tune, y_candidate, SPIKE_THRESHOLD)

            y_try = _apply_dip_policy(b_tune, p_tune, d_tune, 0, 0.0, 0.0, 0.0)
            ok, sc = _accept_dip(y_try)
            if ok and sc < best_d_sc:
                best_d_sc = sc
                best_dk = 0
                best_dp1, best_dp2, best_dp3 = 0.0, 0.0, 0.0

            for thr in _DIP_THR_GRID:
                for pwr in _DIP_POW_GRID:
                    for wmx in _DIP_WMAX_GRID:
                        y_try = _apply_dip_policy(
                            b_tune,
                            p_tune,
                            d_tune,
                            1,
                            float(thr),
                            float(pwr),
                            float(wmx),
                        )
                        ok, sc = _accept_dip(y_try)
                        if ok and sc < best_d_sc:
                            best_d_sc = sc
                            best_dk = 1
                            best_dp1, best_dp2, best_dp3 = float(thr), float(pwr), float(wmx)

            for thr in _DIP_THR_GRID:
                for gw in _DIP_GATE_W_GRID:
                    y_try = _apply_dip_policy(
                        b_tune,
                        p_tune,
                        d_tune,
                        2,
                        float(thr),
                        float(gw),
                        0.0,
                    )
                    ok, sc = _accept_dip(y_try)
                    if ok and sc < best_d_sc:
                        best_d_sc = sc
                        best_dk = 2
                        best_dp1, best_dp2, best_dp3 = float(thr), float(gw), 0.0

            dip_kind[i] = best_dk
            dip_p1[i]   = best_dp1
            dip_p2[i]   = best_dp2
            dip_p3[i]   = best_dp3

            y_model = _apply_dip_policy(
                y_model,
                dp,
                dr,
                int(best_dk),
                float(best_dp1),
                float(best_dp2),
                float(best_dp3),
            )

        y_t, y_m, y_n = y_true[mask], y_model[mask], naive_h_val[mask]
        best_a, best_mae = 1.0, float("inf")
        for a in alpha_grid:
            mae = float(np.mean(np.abs(y_t - (a * y_m + (1.0 - a) * y_n))))
            if mae < best_mae:
                best_mae, best_a = mae, float(a)
        blend_alphas[i] = best_a

    # ── Isotonic calibration ───────────────────────────────────────────────────
    calibrators: list[IsotonicRegression | None] = [None] * config.FORECAST_HORIZON

    for i, h in enumerate(steps):
        naive_col = _horizon_naive_col(h)
        if naive_col not in valid_df.columns:
            naive_col = "price_lag_48"
        if naive_col not in valid_df.columns:
            continue

        h_feat_va  = _compute_target_time_feats(valid_df.index, h)
        pd_feat_va = _compute_aligned_pd_feats(valid_df, h)
        X_h_va = np.concatenate([X_valid.values, h_feat_va, cross_va, pd_feat_va], axis=1).astype(np.float32)

        _l1_cal    = _from_asinh(base_models[i].predict(X_h_va)).astype(np.float32)
        _l2_cal    = _from_asinh(base_l2_models[i].predict(X_h_va)).astype(np.float32)
        y_base_val = ((1.0 - blend_l2_alphas[i]) * _l1_cal + blend_l2_alphas[i] * _l2_cal).astype(np.float32)
        if spike_clfs[i] is not None:
            sp     = spike_clfs[i].predict_proba(X_h_va)[:, 1].astype(np.float32)
            sr     = _from_asinh(spike_regs[i].predict(X_h_va)).astype(np.float32) if spike_regs[i] else y_base_val
            sq     = _from_asinh(spike_qregs[i].predict(X_h_va)).astype(np.float32) if spike_qregs[i] else sr
            y_model = _apply_spike_policy(
                y_base_val,
                sp,
                sr,
                sq,
                int(spike_kind[i]),
                float(spike_p1[i]),
                float(spike_p2[i]),
                float(spike_p3[i]),
                int(spike_src[i]),
            )
        else:
            y_model = y_base_val

        if i < len(dip_clfs) and dip_clfs[i] is not None and i < len(dip_regs) and dip_regs[i] is not None:
            dp = dip_clfs[i].predict_proba(X_h_va)[:, 1].astype(np.float32)
            dr = _from_asinh(dip_regs[i].predict(X_h_va)).astype(np.float32)
            kind = int(dip_kind[i]) if i < len(dip_kind) else 0
            p1   = float(dip_p1[i]) if i < len(dip_p1) else 0.15
            p2   = float(dip_p2[i]) if i < len(dip_p2) else 1.0
            p3   = float(dip_p3[i]) if i < len(dip_p3) else 0.60
            y_model = _apply_dip_policy(y_model, dp, dr, kind, p1, p2, p3)

        y_naive = valid_df[naive_col].values.astype(np.float32)
        a       = float(blend_alphas[i])
        y_blend = a * y_model + (1.0 - a) * y_naive
        y_true  = valid_df[f"target_h{h}"].values.astype(np.float32)
        mask    = np.isfinite(y_true) & np.isfinite(y_blend)
        if int(mask.sum()) < 500:
            continue

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_blend[mask], y_true[mask])
        calibrators[i] = iso

    payload = {
        "gap":             gap,
        "horizon":         horizon,
        "base_models":     base_models,
        "base_l2_models":  base_l2_models,
        "blend_l2_alphas": blend_l2_alphas,
        "spike_clfs":      spike_clfs,
        "spike_regs":      spike_regs,
        "spike_qregs":     spike_qregs,
        "dip_clfs":        dip_clfs,
        "dip_regs":        dip_regs,
        "feature_cols":    feature_cols,
        "blend_alphas":    blend_alphas,
        "spike_kind":   spike_kind,
        "spike_src":    spike_src,
        "spike_p1":     spike_p1,
        "spike_p2":     spike_p2,
        "spike_p3":     spike_p3,
        "dip_kind":     dip_kind,
        "dip_p1":       dip_p1,
        "dip_p2":       dip_p2,
        "dip_p3":       dip_p3,
        "calibrators":  calibrators,
        "clip_thresh":  _clip_thresh,
    }
    return payload, feature_cols, None


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_seq2seq(
    model: dict,
    df: pd.DataFrame,
    past_cols: list[str],
    scaler: None,
) -> dict:
    # Restore the gap/horizon the model was trained with so all internals use
    # the correct values (number of horizons, target column names, etc.).
    gap     = model.get("gap",     config.FORECAST_GAP)
    horizon = model.get("horizon", config.FORECAST_HORIZON)
    config.FORECAST_GAP     = gap
    config.FORECAST_HORIZON = horizon
    del scaler

    df = _add_extra_lags(df)
    _, _, test_df = _temporal_split(df)
    X_test   = test_df[past_cols]
    cross_te = _compute_cross_feats(test_df)

    base_models     = model.get("base_models",     model.get("models", []))
    base_l2_models  = model.get("base_l2_models",  [None] * config.FORECAST_HORIZON)
    blend_l2_alphas = model.get("blend_l2_alphas", np.zeros(config.FORECAST_HORIZON, dtype=np.float32))
    spike_clfs      = model.get("spike_clfs",      [None] * config.FORECAST_HORIZON)
    spike_regs   = model.get("spike_regs",   [None] * config.FORECAST_HORIZON)
    spike_qregs  = model.get("spike_qregs",  [None] * config.FORECAST_HORIZON)
    dip_clfs     = model.get("dip_clfs",     [None] * config.FORECAST_HORIZON)
    dip_regs     = model.get("dip_regs",     [None] * config.FORECAST_HORIZON)
    blend_alphas = model.get("blend_alphas", np.ones(config.FORECAST_HORIZON, dtype=np.float32))
    spike_kind   = model.get("spike_kind",   np.ones(config.FORECAST_HORIZON, dtype=np.int8))
    spike_src    = model.get("spike_src",    np.zeros(config.FORECAST_HORIZON, dtype=np.int8))
    spike_p1     = model.get("spike_p1",     np.full(config.FORECAST_HORIZON, 0.20, dtype=np.float32))
    spike_p2     = model.get("spike_p2",     np.full(config.FORECAST_HORIZON, 2.0, dtype=np.float32))
    spike_p3     = model.get("spike_p3",     np.full(config.FORECAST_HORIZON, 0.75, dtype=np.float32))
    dip_kind     = model.get("dip_kind",     np.zeros(config.FORECAST_HORIZON, dtype=np.int8))
    dip_p1       = model.get("dip_p1",       np.full(config.FORECAST_HORIZON, 0.15, dtype=np.float32))
    dip_p2       = model.get("dip_p2",       np.full(config.FORECAST_HORIZON, 1.0, dtype=np.float32))
    dip_p3       = model.get("dip_p3",       np.full(config.FORECAST_HORIZON, 0.60, dtype=np.float32))
    calibrators  = model.get("calibrators",  [None] * config.FORECAST_HORIZON)

    preds_m = np.empty((len(test_df), config.FORECAST_HORIZON), dtype=np.float32)

    for i, h in enumerate(range(config.FORECAST_GAP, config.FORECAST_GAP + config.FORECAST_HORIZON)):
        h_feat_te  = _compute_target_time_feats(test_df.index, h)
        pd_feat_te = _compute_aligned_pd_feats(test_df, h)
        X_h_te = np.concatenate([X_test.values, h_feat_te, cross_te, pd_feat_te], axis=1).astype(np.float32)

        _l1_te = _from_asinh(base_models[i].predict(X_h_te)).astype(np.float32)
        _l2_m  = base_l2_models[i] if (i < len(base_l2_models) and base_l2_models[i] is not None) else None
        if _l2_m is not None:
            _l2_te = _from_asinh(_l2_m.predict(X_h_te)).astype(np.float32)
            _la    = float(blend_l2_alphas[i]) if i < len(blend_l2_alphas) else 0.0
            y_base = ((1.0 - _la) * _l1_te + _la * _l2_te).astype(np.float32)
        else:
            y_base = _l1_te
        if i < len(spike_clfs) and spike_clfs[i] is not None:
            sp = spike_clfs[i].predict_proba(X_h_te)[:, 1].astype(np.float32)
            if i < len(spike_regs) and spike_regs[i] is not None:
                sr = _from_asinh(spike_regs[i].predict(X_h_te)).astype(np.float32)
            else:
                sr = y_base
            if i < len(spike_qregs) and spike_qregs[i] is not None:
                sq = _from_asinh(spike_qregs[i].predict(X_h_te)).astype(np.float32)
            else:
                sq = sr
            kind = int(spike_kind[i]) if i < len(spike_kind) else 1
            src  = int(spike_src[i]) if i < len(spike_src) else 0
            p1   = float(spike_p1[i]) if i < len(spike_p1) else 0.20
            p2   = float(spike_p2[i]) if i < len(spike_p2) else 2.0
            p3   = float(spike_p3[i]) if i < len(spike_p3) else 0.75
            y_model = _apply_spike_policy(y_base, sp, sr, sq, kind, p1, p2, p3, src)
        else:
            y_model = y_base

        if i < len(dip_clfs) and dip_clfs[i] is not None and i < len(dip_regs) and dip_regs[i] is not None:
            dp = dip_clfs[i].predict_proba(X_h_te)[:, 1].astype(np.float32)
            dr = _from_asinh(dip_regs[i].predict(X_h_te)).astype(np.float32)
            dkind = int(dip_kind[i]) if i < len(dip_kind) else 0
            dp1   = float(dip_p1[i]) if i < len(dip_p1) else 0.15
            dp2   = float(dip_p2[i]) if i < len(dip_p2) else 1.0
            dp3   = float(dip_p3[i]) if i < len(dip_p3) else 0.60
            y_model = _apply_dip_policy(y_model, dp, dr, dkind, dp1, dp2, dp3)

        a         = float(blend_alphas[i]) if i < len(blend_alphas) else 1.0
        naive_col = _horizon_naive_col(h)
        if naive_col not in test_df.columns:
            naive_col = "price_lag_48"
        if naive_col in test_df.columns and a < 1.0:
            naive_h = test_df[naive_col].values.astype(np.float32)
            y_blend = a * y_model + (1.0 - a) * naive_h
        else:
            y_blend = y_model

        cal = calibrators[i] if i < len(calibrators) else None
        if cal is not None:
            y_blend = cal.predict(y_blend.astype(np.float64)).astype(np.float32)
        preds_m[:, i] = y_blend

    trues_m = np.empty((len(test_df), config.FORECAST_HORIZON), dtype=np.float32)
    for i, h in enumerate(range(config.FORECAST_GAP, config.FORECAST_GAP + config.FORECAST_HORIZON)):
        trues_m[:, i] = test_df[f"target_h{h}"].values.astype(np.float32)

    valid_rows = np.all(np.isfinite(trues_m), axis=1)
    preds_m    = preds_m[valid_rows]
    trues_m    = trues_m[valid_rows]
    test_idx   = test_df.index[valid_rows]

    def _rmse(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(yt, yp)))

    def _wmape(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(np.sum(np.abs(yt - yp)) / (np.sum(np.abs(yt)) + 1e-8) * 100)

    def _mbe(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(np.mean(yp - yt))

    def _median_ae(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(np.median(np.abs(yt - yp)))

    def _build_metrics(yt: np.ndarray, yp: np.ndarray) -> dict:
        return {
            "mae":       float(mean_absolute_error(yt, yp)),
            "rmse":      _rmse(yt, yp),
            "r2":        float(r2_score(yt, yp)),
            "mbe":       _mbe(yt, yp),
            "median_ae": _median_ae(yt, yp),
            "wmape":     _wmape(yt, yp),
        }

    yt_flat = trues_m.ravel()
    yp_flat = preds_m.ravel()
    model_metrics = _build_metrics(yt_flat, yp_flat)

    # Spike-specific error analysis.
    spike_mask = yt_flat > SPIKE_THRESHOLD
    if spike_mask.sum() > 0:
        model_metrics["spike_mae"]    = float(mean_absolute_error(yt_flat[spike_mask], yp_flat[spike_mask]))
        model_metrics["nonspike_mae"] = float(mean_absolute_error(yt_flat[~spike_mask], yp_flat[~spike_mask]))
        model_metrics["spike_pct"]    = float(spike_mask.mean() * 100)
    dip_mask = yt_flat < _DIP_THRESHOLD
    if dip_mask.sum() > 0:
        model_metrics["dip_mae"] = float(mean_absolute_error(yt_flat[dip_mask], yp_flat[dip_mask]))
        model_metrics["dip_pct"] = float(dip_mask.mean() * 100)

    if "price_lag_48" in test_df.columns:
        naive_base    = test_df["price_lag_48"].values[valid_rows]
        naive_m       = np.repeat(naive_base[:, None], config.FORECAST_HORIZON, axis=1)
        naive_metrics = _build_metrics(trues_m.ravel(), naive_m.ravel())
    else:
        naive_metrics = model_metrics

    step_records = []
    for i, h in enumerate(range(config.FORECAST_GAP, config.FORECAST_GAP + config.FORECAST_HORIZON)):
        y_t = trues_m[:, i]
        y_p = preds_m[:, i]
        step_records.append({
            "step":   i + 1,
            "h":      h,
            "lead_h": round(h * INTERVAL_MINUTES / 60, 1),
            "mae":    round(float(mean_absolute_error(y_t, y_p)), 2),
            "rmse":   round(_rmse(y_t, y_p), 2),
            "r2":     round(float(r2_score(y_t, y_p)), 4),
            "mbe":    round(_mbe(y_t, y_p), 2),
        })

    results_df = pd.DataFrame(index=test_idx)
    for i, h in enumerate(range(config.FORECAST_GAP, config.FORECAST_GAP + config.FORECAST_HORIZON)):
        results_df[f"actual_h{h}"]    = trues_m[:, i]
        results_df[f"predicted_h{h}"] = preds_m[:, i]

    # Feature importance from base models (spike models excluded to keep names consistent).
    fi = np.mean([m.feature_importances_ for m in base_models], axis=0)
    all_feat_names = list(past_cols) + _TARGET_TIME_FEAT_NAMES + _CROSS_FEAT_NAMES + _PD_ALIGNED_FEAT_NAMES
    if len(fi) == len(all_feat_names):
        fi_df = (
            pd.DataFrame({"feature": all_feat_names, "importance": fi})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    else:
        fi_df = None  # shape guard

    return {
        "model":              model_metrics,
        "naive":              naive_metrics,
        "results_df":         results_df,
        "steps_df":           pd.DataFrame(step_records),
        "feature_importance": fi_df,
    }


# ── Save / load ────────────────────────────────────────────────────────────────

def save_seq2seq(
    model: dict,
    past_cols: list[str],
    scaler: None,
    *,
    model_file: Path | None = None,
) -> None:
    del past_cols, scaler
    out_model = model_file if model_file is not None else FAST_MODEL_FILE
    joblib.dump(model, out_model, compress=("zlib", 9))
    print(f"  Model saved -> {out_model}  ({out_model.stat().st_size / 1e6:.1f} MB)", flush=True)


# ── Reporting ──────────────────────────────────────────────────────────────────

def report_results(
    eval_output: dict,
) -> None:
    steps_df = eval_output["steps_df"]

    _UNITLESS = {"r2"}
    _PCT      = {"mape", "wmape", "spike_pct", "dip_pct"}

    print("\n  ── Aggregate Test-Set Metrics ──────────────────────────────────", flush=True)
    m = eval_output["model"]
    n = eval_output["naive"]
    for metric, val in m.items():
        if metric in _UNITLESS:
            fmt       = f"{val:8.4f}"
            naive_fmt = f"{n.get(metric, 0.0):8.4f}"
        elif metric in _PCT:
            unit      = "%"
            fmt       = f"{val:8.2f}{unit}"
            naive_fmt = f"{n.get(metric, 0.0):8.2f}{unit}" if metric in n else "        n/a"
        elif metric in ("spike_mae", "nonspike_mae"):
            unit      = " $/MWh"
            fmt       = f"{val:8.2f}{unit}"
            naive_fmt = "        n/a"
        else:
            unit      = " $/MWh"
            fmt       = f"{val:8.2f}{unit}"
            naive_fmt = f"{n.get(metric, 0.0):8.2f}{unit}"
        print(f"  {metric.upper():14s}  LGBM: {fmt}   Naive lag-48: {naive_fmt}", flush=True)

    naive_mae = n.get("mae", 0.0)
    if naive_mae > 0:
        skill = (1 - m["mae"] / naive_mae) * 100
        print(f"  {'SKILL':14s}  MAE skill vs naive: {skill:+.2f}%", flush=True)

    print(
        f"\n  ── Per-step Metrics "
        f"(gap={config.FORECAST_GAP * INTERVAL_MINUTES // 60}h, "
        f"horizon={config.FORECAST_HORIZON * INTERVAL_MINUTES // 60}h) ─────────────────",
        flush=True,
    )
    print(f"    {'step':>4}  {'lead':>6}  {'MAE':>8}  {'RMSE':>8}  {'R²':>7}  {'MBE':>8}", flush=True)
    for _, row in steps_df.iterrows():
        print(
            f"    step {int(row['step']):>3}  "
            f"lead={row['lead_h']:.1f}h  "
            f"MAE={row['mae']:>7.2f}  "
            f"RMSE={row['rmse']:>7.2f}  "
            f"R²={row['r2']:>6.4f}  "
            f"MBE={row['mbe']:>+7.2f}",
            flush=True,
        )

    fi_df = eval_output.get("feature_importance")
    if fi_df is not None:
        print(f"\n  Top 20 features by mean gain (base models)", flush=True)
        for rank, row in fi_df.head(20).iterrows():
            print(f"    {rank + 1:>3}.  {row['feature']:<45}  {row['importance']:>12.0f}")

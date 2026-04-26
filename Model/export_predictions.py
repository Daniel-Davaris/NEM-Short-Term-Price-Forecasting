"""
Export timestamp-aligned 2023 price predictions to one Excel file per state.

Each output workbook contains two columns only:
  - Timestamp
  - Predictions

The prediction at each timestamp is the model's 30-minute-ahead forecast,
aligned to the target interval so there is exactly one prediction per half-hour
across calendar year 2023.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from Dataset import load_data
from Model.feature_engineering import build_features, select_region_columns
from Model.model import (
    FORECAST_GAP,
    _add_extra_lags,
    _apply_dip_policy,
    _apply_spike_policy,
    _compute_aligned_pd_feats,
    _compute_cross_feats,
    _compute_target_time_feats,
    _from_asinh,
    _horizon_naive_col,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"
MODELS_DIR = DATA_DIR / "models"
OUTPUTS_DIR = DATA_DIR / "outputs" / "predictions"

REGIONS = ["NSW1", "QLD1", "VIC1", "SA1"]
REGION_TAGS = {
    "NSW1": "nsw",
    "QLD1": "qld",
    "VIC1": "vic",
    "SA1": "sa",
}


def _load_region_frame(raw_df: pd.DataFrame, region: str) -> pd.DataFrame:
    region_df = select_region_columns(raw_df, region)
    df_full, _, _ = build_features(region_df)
    return _add_extra_lags(df_full)


def _predict_horizon_one(model: dict, df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    base_models = model.get("base_models", model.get("models", []))
    spike_clfs = model.get("spike_clfs", [None])
    spike_regs = model.get("spike_regs", [None])
    spike_qregs = model.get("spike_qregs", [None])
    dip_clfs = model.get("dip_clfs", [None])
    dip_regs = model.get("dip_regs", [None])
    blend_alphas = model.get("blend_alphas", np.ones(1, dtype=np.float32))
    spike_kind = model.get("spike_kind", np.ones(1, dtype=np.int8))
    spike_src = model.get("spike_src", np.zeros(1, dtype=np.int8))
    spike_p1 = model.get("spike_p1", np.full(1, 0.20, dtype=np.float32))
    spike_p2 = model.get("spike_p2", np.full(1, 2.0, dtype=np.float32))
    spike_p3 = model.get("spike_p3", np.full(1, 0.75, dtype=np.float32))
    dip_kind = model.get("dip_kind", np.zeros(1, dtype=np.int8))
    dip_p1 = model.get("dip_p1", np.full(1, 0.15, dtype=np.float32))
    dip_p2 = model.get("dip_p2", np.full(1, 1.0, dtype=np.float32))
    dip_p3 = model.get("dip_p3", np.full(1, 0.60, dtype=np.float32))
    calibrators = model.get("calibrators", [None])

    x_base = df[feature_cols]
    h = FORECAST_GAP
    target_time_feats = _compute_target_time_feats(df.index, h)
    cross_feats = _compute_cross_feats(df)
    aligned_pd_feats = _compute_aligned_pd_feats(df, h)
    x_h = np.concatenate(
        [x_base.values, target_time_feats, cross_feats, aligned_pd_feats],
        axis=1,
    ).astype(np.float32)

    y_base = _from_asinh(base_models[0].predict(x_h)).astype(np.float32)

    if spike_clfs and spike_clfs[0] is not None:
        spike_prob = spike_clfs[0].predict_proba(x_h)[:, 1].astype(np.float32)
        if spike_regs and spike_regs[0] is not None:
            y_spike = _from_asinh(spike_regs[0].predict(x_h)).astype(np.float32)
        else:
            y_spike = y_base
        if spike_qregs and spike_qregs[0] is not None:
            y_q = _from_asinh(spike_qregs[0].predict(x_h)).astype(np.float32)
        else:
            y_q = y_spike
        y_model = _apply_spike_policy(
            y_base,
            spike_prob,
            y_spike,
            y_q,
            int(spike_kind[0]),
            float(spike_p1[0]),
            float(spike_p2[0]),
            float(spike_p3[0]),
            int(spike_src[0]),
        )
    else:
        y_model = y_base

    if dip_clfs and dip_regs and dip_clfs[0] is not None and dip_regs[0] is not None:
        dip_prob = dip_clfs[0].predict_proba(x_h)[:, 1].astype(np.float32)
        y_dip = _from_asinh(dip_regs[0].predict(x_h)).astype(np.float32)
        y_model = _apply_dip_policy(
            y_model,
            dip_prob,
            y_dip,
            int(dip_kind[0]),
            float(dip_p1[0]),
            float(dip_p2[0]),
            float(dip_p3[0]),
        )

    alpha = float(blend_alphas[0]) if len(blend_alphas) else 1.0
    naive_col = _horizon_naive_col(h)
    if naive_col not in df.columns:
        naive_col = "price_lag_48"
    if naive_col in df.columns and alpha < 1.0:
        naive_values = df[naive_col].values.astype(np.float32)
        y_model = alpha * y_model + (1.0 - alpha) * naive_values

    calibrator = calibrators[0] if calibrators else None
    if calibrator is not None:
        y_model = calibrator.predict(y_model.astype(np.float64)).astype(np.float32)

    prediction_index = df.index + pd.Timedelta(minutes=30 * h)
    return pd.Series(y_model, index=prediction_index, name="Predictions")


def _build_export_frame(predictions: pd.Series) -> pd.DataFrame:
    start = pd.Timestamp("2023-01-01 00:00:00")
    end = pd.Timestamp("2023-12-31 23:30:00")
    full_index = pd.date_range(start=start, end=end, freq="30min")
    aligned = predictions[~predictions.index.duplicated(keep="first")].reindex(full_index)
    return pd.DataFrame({
        "Timestamp": full_index,
        "Predictions": aligned.values,
    })


def export_region_predictions(raw_df: pd.DataFrame, region: str) -> Path:
    region_tag = REGION_TAGS[region]
    model_path = MODELS_DIR / f"{region_tag}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    print(f"Exporting 2023 predictions for {region}...", flush=True)
    model = joblib.load(model_path)
    feature_cols = model.get("feature_cols")
    if not feature_cols:
        raise ValueError(f"Model {model_path.name} does not contain feature_cols")

    df = _load_region_frame(raw_df, region)
    missing_feature_cols = [col for col in feature_cols if col not in df.columns]
    if missing_feature_cols:
        raise ValueError(
            f"Missing {len(missing_feature_cols)} model feature columns for {region}: "
            f"{missing_feature_cols[:10]}"
        )

    predictions = _predict_horizon_one(model, df, feature_cols)
    export_df = _build_export_frame(predictions)
    if export_df["Predictions"].isna().any():
        missing_count = int(export_df["Predictions"].isna().sum())
        raise ValueError(f"{region} export has {missing_count} missing 2023 prediction rows")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"{region_tag}_predictions_2023.xlsx"
    export_df.to_excel(output_path, index=False)
    print(f"Saved {len(export_df):,} rows to {output_path}", flush=True)
    return output_path


def main() -> None:
    raw_df = load_data()
    exported_files = []
    for region in REGIONS:
        exported_files.append(export_region_predictions(raw_df, region))

    print("\nCreated Excel exports:", flush=True)
    for output_path in exported_files:
        print(f"  - {output_path}", flush=True)


if __name__ == "__main__":
    main()
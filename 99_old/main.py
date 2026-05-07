"""
0_main.py  –  End-to-end pipeline for electricity price forecasts.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from Model.config import cfg_tag, FORECAST_GAP_HORIZON_COMBOS
from Dataset import load_data
from Model.feature_engineering import build_features, select_features, save_datasets, select_region_columns
from Model.model import evaluate_seq2seq, report_results, save_seq2seq, train_seq2seq


def _print_system_info() -> None:
    """Print available CPU cores and RAM before the pipeline starts."""
    cpu_count = os.cpu_count() or 1
    try:
        with open("/proc/meminfo") as _f:
            lines = _f.readlines()
        total_kb = int(next(l for l in lines if l.startswith("MemTotal")).split()[1])
        avail_kb = int(next(l for l in lines if l.startswith("MemAvailable")).split()[1])
        mem_str = (
            f"RAM: {avail_kb / 1024**2:.1f} GB available / {total_kb / 1024**2:.1f} GB total"
        )
    except Exception:
        mem_str = "RAM: unknown"
    print(f"System  |  CPUs: {cpu_count} logical cores  |  {mem_str}", flush=True)

_BASE      = Path(__file__).resolve().parent
_DATA_ROOT = _BASE / "Dataset"
MODELS_DIR = _DATA_ROOT / "models"
OUTPUTS_DIR = _DATA_ROOT / "outputs"


def _load(region: str, data_start: str, data_end: str):
    raw_df = load_data()
    raw_df = raw_df.loc[
        (raw_df.index >= pd.Timestamp(data_start)) &
        (raw_df.index <  pd.Timestamp(data_end))
    ]
    return select_region_columns(raw_df, region)


def _build(region_df, gap: int, horizon: int):
    print("Building features...", flush=True)
    df_full, df, feature_cols = build_features(region_df, gap=gap, horizon=horizon)
    print(f"  {len(feature_cols)} features, {len(df):,} rows", flush=True)
    return df_full, feature_cols


def _select(df_full, feature_cols, file_tag: str, gap: int, horizon: int, reselect_features: bool):
    print("Selecting features...", flush=True)
    feature_cols = select_features(
        df_full, feature_cols, file_tag, OUTPUTS_DIR,
        gap=gap, horizon=horizon, force_rerun=reselect_features,
    )
    return feature_cols


def _save(df_full, feature_cols, file_tag: str, gap: int, horizon: int) -> None:
    print("Saving datasets...", flush=True)
    save_datasets(df_full, feature_cols, file_tag, gap=gap, horizon=horizon)


def _train(df_full, feature_cols, file_tag: str, gap: int, horizon: int, reselect_features: bool):
    print("Training...", flush=True)
    model, past_cols, scaler = train_seq2seq(
        df_full, feature_cols, gap=gap, horizon=horizon, force_reselect=reselect_features,
    )
    save_seq2seq(model, past_cols, scaler, model_file=MODELS_DIR / f"{file_tag}_model.joblib")
    return model, past_cols, scaler


def _evaluate(model, df_full, past_cols, scaler, file_tag: str) -> None:
    print("Evaluating...", flush=True)
    eval_output = evaluate_seq2seq(model, df_full, past_cols, scaler)
    report_results(eval_output)
    _save_accuracy_report(eval_output, file_tag)
    _save_predictions_csv(eval_output, file_tag)


def _save_accuracy_report(eval_output: dict, file_tag: str) -> None:
    """Write an Excel accuracy report with three sheets: Aggregate, PerStep, FeatureImportance."""
    out_path = OUTPUTS_DIR / "accuracy_reports" / f"{file_tag}_accuracy.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_m = eval_output["model"]
    naive_m = eval_output["naive"]

    # Build aggregate metrics sheet
    rows = []
    for metric, val in model_m.items():
        rows.append({
            "metric": metric,
            "model":  round(val, 4),
            "naive":  round(naive_m.get(metric, float("nan")), 4),
        })
    naive_mae = naive_m.get("mae", 0.0)
    if naive_mae > 0:
        rows.append({
            "metric": "skill_vs_naive_pct",
            "model":  round((1 - model_m["mae"] / naive_mae) * 100, 4),
            "naive":  0.0,
        })
    aggregate_df = pd.DataFrame(rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        aggregate_df.to_excel(writer, sheet_name="Aggregate", index=False)
        eval_output["steps_df"].to_excel(writer, sheet_name="PerStep", index=False)
        fi_df = eval_output.get("feature_importance")
        if fi_df is not None:
            fi_df.to_excel(writer, sheet_name="FeatureImportance", index=False)

    print(f"  Accuracy report -> {out_path}", flush=True)


def _save_predictions_csv(eval_output: dict, file_tag: str) -> None:
    """Write test-set predictions as a flat 2-column CSV: Timestamp, Predictions.

    Each observation row is mapped to its target timestamp by shifting forward
    by gap intervals (the first horizon step), giving one unique prediction per
    target half-hour interval regardless of the model's gap or horizon.
    """
    out_path = OUTPUTS_DIR / "predictions" / f"{file_tag}_predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = eval_output["results_df"]
    gap = eval_output.get("gap", results_df.columns[0].split("h")[1])
    # First predicted column is predicted_h{gap}; strip to find the gap integer.
    first_pred_col = next(c for c in results_df.columns if c.startswith("predicted_h"))
    gap_intervals  = int(first_pred_col.split("predicted_h")[1])

    target_index = results_df.index + pd.Timedelta(minutes=30 * gap_intervals)
    flat_df = pd.DataFrame({
        "Timestamp":   target_index,
        "Predictions": results_df[first_pred_col].values,
    })

    flat_df.to_parquet(out_path, index=False)
    print(f"  Predictions parquet -> {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)", flush=True)


def run_region(
    region: str,
    gap: int,
    horizon: int,
    data_start: str,
    data_end: str,
    reselect_features: bool = True,
) -> None:
    region_tag = region.lower().replace("1", "")
    tag        = cfg_tag(gap, horizon, data_start, data_end)
    file_tag   = f"{region_tag}_{tag}"   # e.g. "nsw_g1h16_2018_2024"

    accuracy_path    = OUTPUTS_DIR / "accuracy_reports" / f"{file_tag}_accuracy.xlsx"
    predictions_path = OUTPUTS_DIR / "predictions"      / f"{file_tag}_predictions.parquet"
    model_path       = MODELS_DIR / f"{file_tag}_model.joblib"

    # ── Skip if fully complete ──────────────────────────────────────────────
    if accuracy_path.exists() and predictions_path.exists() and predictions_path.stat().st_size > 1024:
        print(f"\n[SKIP] {file_tag} — outputs already exist", flush=True)
        return

    t_region_start = time.perf_counter()
    print(f"\n{'='*60}", flush=True)
    print(f"  Region {region} | {tag}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Resume: accuracy done but predictions missing — reload and re-evaluate ──
    if accuracy_path.exists() and model_path.exists():
        print("  Resuming: accuracy report exists, regenerating predictions...", flush=True)
        import joblib as _joblib
        model     = _joblib.load(model_path)
        past_cols = model.get("feature_cols", [])
        region_df = _load(region, data_start, data_end)
        df_full, _ = _build(region_df, gap, horizon)
        _evaluate(model, df_full, past_cols, None, file_tag)
        elapsed = time.perf_counter() - t_region_start
        print(f"  Region {region} | {tag} resumed in {elapsed / 60:.2f} min", flush=True)
        return

    region_df                 = _load(region, data_start, data_end)
    df_full, feature_cols     = _build(region_df, gap, horizon)
    feature_cols              = _select(df_full, feature_cols, file_tag, gap, horizon, reselect_features)
    # _save is intentionally skipped: 24 unique parquets (~330 MB each) exceed disk capacity.
    # Run save_datasets() manually on a single model if the parquet files are needed.
    model, past_cols, scaler  = _train(df_full, feature_cols, file_tag, gap, horizon, reselect_features)
    _evaluate(model, df_full, past_cols, scaler, file_tag)

    # Delete the model file once evaluation is complete — predictions and accuracy
    # reports are already saved, and re-running will skip via the skip guard above.
    # This prevents disk exhaustion when training many models sequentially.
    if model_path.exists():
        model_path.unlink()
        print(f"  Model file deleted (disk space): {model_path.name}", flush=True)

    elapsed = time.perf_counter() - t_region_start
    print(f"  Region {region} | {tag} finished in {elapsed / 60:.2f} min ({elapsed:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Pipeline configuration – 4 states × 3 timeframes × 2 gap/horizon = 24 models
# ---------------------------------------------------------------------------

STATES = ["NSW1", "QLD1", "VIC1", "SA1"]

# Timeframes: (data_start, data_end) where data_end is exclusive.
# The last 12 months of each window is used as the test set.
TIMEFRAMES = [
    ("2018/01/01", "2024/01/01"),   # train ~2018-2022, test 2023
    ("2018/01/01", "2025/01/01"),   # train ~2018-2023, test 2024
    ("2018/01/01", "2026/01/01"),   # train ~2018-2024, test 2025
]

# Gap/horizon combos — imported from config (single source of truth shared
# with the ingest scripts).
GAP_HORIZON_COMBOS = FORECAST_GAP_HORIZON_COMBOS


_print_system_info()
_pipeline_start = time.perf_counter()

_total_runs = len(STATES) * len(TIMEFRAMES) * len(GAP_HORIZON_COMBOS)
print(f"\nRunning {_total_runs} models ({len(STATES)} states × {len(TIMEFRAMES)} timeframes × {len(GAP_HORIZON_COMBOS)} gap/horizon combos)", flush=True)

_run_num = 0
for _state in STATES:
    for _data_start, _data_end in TIMEFRAMES:
        for _gap, _horizon in GAP_HORIZON_COMBOS:
            _run_num += 1
            print(f"\n[{_run_num}/{_total_runs}] {_state} | gap={_gap} horizon={_horizon} | {_data_start[:4]}–{_data_end[:4]}", flush=True)
            run_region(
                region=_state,
                gap=_gap,
                horizon=_horizon,
                data_start=_data_start,
                data_end=_data_end,
                reselect_features=True,
            )

_total_elapsed = time.perf_counter() - _pipeline_start
print(f"\nTotal pipeline runtime: {_total_elapsed / 60:.2f} min ({_total_elapsed:.1f}s)", flush=True)


"""
Evaluate all pre-trained models (NSW, QLD, VIC, SA) and output comprehensive metrics report.

This script loads each region's trained model from the models/ directory,
evaluates it on the test set using evaluate_seq2seq(), and generates a
consolidated metrics report with:
  - Aggregate test-set metrics (MAE, RMSE, R², spike/dip analysis)
  - Per-step metrics across all 48 forecast horizons
  - Feature importance rankings
  - Comparative performance analysis

Output files:
  - outputs/all_states_metrics_summary.txt  — human-readable report
  - outputs/all_states_metrics_data.csv    — machine-readable metrics by state
  - outputs/all_states_steps_by_state.csv — per-step breakdown for each state
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from Dataset import load_data
from Model.feature_engineering import build_features, select_features, select_region_columns
from Model.model import evaluate_seq2seq

_BASE = Path(__file__).resolve().parent
_DATA_ROOT = _BASE / "Dataset"
MODELS_DIR = _DATA_ROOT / "models"
OUTPUTS_DIR = _DATA_ROOT / "outputs"
DATASETS_DIR = _DATA_ROOT / "all_engineered"

REGIONS = ["NSW1", "QLD1", "VIC1", "SA1"]
REGION_TAGS = {
    "NSW1": "nsw",
    "QLD1": "qld",
    "VIC1": "vic",
    "SA1":  "sa",
}


def _load_region_data(region: str):
    """Load and prepare data for a region."""
    print(f"  Loading data for {region}...", flush=True)
    raw_df = load_data()
    region_df = select_region_columns(raw_df, region)
    return region_df


def _build_features_for_region(region_df):
    """Build and engineer features for a region."""
    df_full, df, feature_cols = build_features(region_df)
    return df_full, feature_cols


def _select_features_for_region(df_full, feature_cols, region_tag: str):
    """Select optimal features for a region."""
    feature_cols = select_features(df_full, feature_cols, region_tag, OUTPUTS_DIR, force_rerun=False)
    return feature_cols


def evaluate_region(region: str) -> dict:
    """
    Evaluate a single region's pre-trained model.
    
    Returns
    -------
    dict
        Contains 'state', 'eval_output', and 'model' for this region.
    """
    region_tag = REGION_TAGS[region]
    model_file = MODELS_DIR / f"{region_tag}_model.joblib"
    
    if not model_file.exists():
        print(f"  ⚠ Model file not found: {model_file}", flush=True)
        return None
    
    print(f"\n  [INFO] Evaluating {region}", flush=True)
    
    try:
        # Load model
        print(f"  [*] Loading trained model...", flush=True)
        model = joblib.load(model_file)
        
        # Load and prepare data
        region_df = _load_region_data(region)
        df_full, feature_cols = _build_features_for_region(region_df)
        feature_cols = _select_features_for_region(df_full, feature_cols, region_tag)
        
        # Evaluate
        print(f"  [*] Evaluating on test set...", flush=True)
        # Use the feature_cols stored in the model (which were used during training)
        stored_feature_cols = model.get("feature_cols", feature_cols)
        scaler = model.get("scaler", None)
        eval_output = evaluate_seq2seq(model, df_full, stored_feature_cols, scaler)
        
        print(f"  [+] MAE: {eval_output['model']['mae']:7.2f} $/MWh", flush=True)
        if 'spike_pct' in eval_output['model']:
            print(f"  [+] Spikes: {eval_output['model']['spike_pct']:5.2f}%", flush=True)
        print(f"  [OK] {region} complete", flush=True)
        
        return {
            "state": region,
            "region_tag": region_tag,
            "eval_output": eval_output,
            "model": model,
        }
    
    except Exception as e:
        print(f"  [ERROR] Error evaluating {region}: {e}", flush=True)
        return None


def generate_metrics_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Extract aggregate metrics from all states into a single DataFrame.
    
    Each row = one state, columns = all metrics.
    """
    records = []
    for res in results:
        if res is None:
            continue
        state = res["state"]
        m = res["eval_output"]["model"]
        record = {"state": state}
        record.update(m)
        records.append(record)
    
    return pd.DataFrame(records)


def generate_steps_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Extract per-step metrics from all states.
    
    Each row = one step per state.
    """
    all_steps = []
    for res in results:
        if res is None:
            continue
        state = res["state"]
        steps_df = res["eval_output"]["steps_df"]
        steps_df = steps_df.copy()
        steps_df.insert(0, "state", state)
        all_steps.append(steps_df)
    
    if all_steps:
        return pd.concat(all_steps, ignore_index=True)
    else:
        return pd.DataFrame()


def write_summary_report(results: list[dict], output_file: Path) -> None:
    """
    Write a human-readable text report to file.
    
    Includes:
      - Overall summary table
      - Per-state aggregate metrics
      - Per-state per-step metrics
      - Feature importance for each state
    """
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ELECTRICITY PRICE FORECASTING — ALL STATES EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("AGGREGATE METRICS SUMMARY\n")
        f.write("-" * 80 + "\n")
        metrics_df = generate_metrics_dataframe(results)
        if not metrics_df.empty:
            # Display key metrics
            key_metrics = ["state", "mae", "rmse", "r2", "mbe", "wmape"]
            if "spike_pct" in metrics_df.columns:
                key_metrics.extend(["spike_mae", "spike_pct", "dip_mae"])
            display_cols = [col for col in key_metrics if col in metrics_df.columns]
            f.write(metrics_df[display_cols].to_string(index=False))
            f.write("\n\n")
        
        # Per-state details
        for res in results:
            if res is None:
                continue
            state = res["state"]
            region_tag = res["region_tag"]
            eval_out = res["eval_output"]
            
            f.write("=" * 80 + "\n")
            f.write(f"STATE: {state} (Tag: {region_tag})\n")
            f.write("=" * 80 + "\n\n")
            
            # Aggregate metrics
            m = eval_out["model"]
            f.write("AGGREGATE TEST-SET METRICS\n")
            f.write("-" * 80 + "\n")
            for key, val in m.items():
                if key in ("spike_pct", "dip_pct", "wmape"):
                    f.write(f"  {key:20s}: {val:8.2f}%\n")
                elif key in ("r2",):
                    f.write(f"  {key:20s}: {val:8.4f}\n")
                else:
                    f.write(f"  {key:20s}: {val:8.2f} $/MWh\n")
            
            # Naive comparison
            f.write("\nNAIVE LAG-48 BASELINE\n")
            f.write("-" * 80 + "\n")
            n = eval_out["naive"]
            for key in ["mae", "rmse", "r2", "mbe"]:
                if key in n:
                    if key == "r2":
                        f.write(f"  {key:20s}: {n[key]:8.4f}\n")
                    else:
                        f.write(f"  {key:20s}: {n[key]:8.2f} $/MWh\n")
            
            # Skill score
            if n.get("mae", 0) > 0:
                skill = (1 - m["mae"] / n["mae"]) * 100
                f.write(f"  {'SKILL SCORE':20s}: {skill:+.2f}%\n")
            
            # Per-step metrics
            f.write("\nPER-STEP METRICS (All 48 Forecast Horizons)\n")
            f.write("-" * 80 + "\n")
            steps_df = eval_out["steps_df"]
            f.write(f"{'Step':>6} {'Lead':>6} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MBE':>8}\n")
            f.write("-" * 80 + "\n")
            for _, row in steps_df.iterrows():
                f.write(
                    f"{int(row['step']):>6} "
                    f"{row['lead_h']:>6.1f}h "
                    f"{row['mae']:>8.2f} "
                    f"{row['rmse']:>8.2f} "
                    f"{row['r2']:>8.4f} "
                    f"{row['mbe']:>8.2f}\n"
                )
            
            # Feature importance
            fi_df = eval_out.get("feature_importance")
            if fi_df is not None and not fi_df.empty:
                f.write("\nTOP 20 MOST IMPORTANT FEATURES\n")
                f.write("-" * 80 + "\n")
                for idx, (_, row) in enumerate(fi_df.head(20).iterrows(), start=1):
                    f.write(f"  {idx:2d}. {row['feature']:40s} {row['importance']:10.6f}\n")
            
            f.write("\n")
    
    print(f"\n✓ Summary report saved to: {output_file}", flush=True)


def main():
    """Main evaluation pipeline."""
    print("\n" + "=" * 80, flush=True)
    print("EVALUATING ALL PRE-TRAINED MODELS", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    # Ensure output directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Evaluate all regions
    results = []
    for region in REGIONS:
        res = evaluate_region(region)
        if res is not None:
            results.append(res)
    
    if not results:
        print("\n❌ No successful evaluations. Exiting.", flush=True)
        return
    
    # Generate outputs
    print("\n" + "-" * 80, flush=True)
    print("GENERATING OUTPUTS", flush=True)
    print("-" * 80 + "\n", flush=True)
    
    # 1. Metrics summary (parquet)
    metrics_df = generate_metrics_dataframe(results)
    metrics_csv = OUTPUTS_DIR / "all_states_metrics_data.parquet"
    metrics_df.to_parquet(metrics_csv, index=False)
    print(f"✓ Metrics data saved to:   {metrics_csv}", flush=True)
    
    # 2. Per-step metrics (parquet)
    steps_df = generate_steps_dataframe(results)
    steps_csv = OUTPUTS_DIR / "all_states_steps_by_state.parquet"
    steps_df.to_parquet(steps_csv, index=False)
    print(f"✓ Step metrics saved to:   {steps_csv}", flush=True)
    
    # 3. Summary report (TXT)
    summary_txt = OUTPUTS_DIR / "all_states_metrics_summary.txt"
    write_summary_report(results, summary_txt)
    
    print("\n" + "=" * 80, flush=True)
    print("✓ EVALUATION COMPLETE", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()

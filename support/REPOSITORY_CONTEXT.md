# Forecasting Repository — Complete Architecture & Context

**Date Created**: May 20, 2026  
**Purpose**: Comprehensive reference for the NEM spike-aware electricity price forecasting system  
**Scope**: Architecture, data flow, model design, features, dependencies, and known issues

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Data Pipeline & Flow](#data-pipeline--flow)
4. [Feature Engineering](#feature-engineering)
5. [Feature Selection Process](#feature-selection-process)
6. [Model Architecture](#model-architecture)
7. [Configuration & Constants](#configuration--constants)
8. [Dependencies & Environment](#dependencies--environment)
9. [Key Files & Functions](#key-files--functions)
10. [Known Issues & Gaps](#known-issues--gaps)
11. [Improvements & Recommendations](#improvements--recommendations)
12. [Quick Reference](#quick-reference)

---

## Project Overview

**System**: Spike-Aware Ensemble Electricity Price Forecaster for Australian NEM  
**Target**: Regional Reference Price (RRP) in NSW, QLD, VIC, and SA  
**Resolution**: 5-minute market intervals (aggregated to 30-min intervals for model input)  
**Forecast Horizon**: 48 hours ahead (96 × 30-min steps)  
**Training Window**: 2019–2026 historical data  
**Data Sources**: 
- AEMO (nemosis library): historical market data, predispatch forecasts
- Weather APIs: temperature for demand interactions  
- Manual corrections: holidays, outages

**Prediction Frequency**: Rolling updates, daily retraining assumed

---

## Repository Structure

```
Forecasting/
├── 0_Config/
│   └── 0_variables.ipynb          # Global config, state definitions, paths
├── 1_Dataset/
│   ├── 0_Analysis.ipynb            # EDA, data profiling
│   ├── 1_Data_sources.ipynb        # Data ingestion, validation
│   ├── Pre_processing/
│   │   ├── temporary_cache/        # Raw CSV files (AEMO dumps)
│   │   │   └── raw/
│   │   └── weather_data/           # Temperature series
│   └── Processed_data/             # Cleaned parquets
├── 2_Features_build/               # Feature engineering (200–400+ features)
│   ├── 0_all.ipynb                 # Master feature builder
│   ├── 1_dispatch_price.ipynb      # Lag features from historical prices
│   ├── 2_dispatch_region_sum.ipynb # Regional demand aggregates
│   ├── 3_generation_fuel.ipynb     # Thermal/renewable generation
│   ├── 4_STTM_DWGM.ipynb           # Financial market signals
│   ├── 5_weather.ipynb             # CDD/HDD demand interactions
│   ├── 6_1_predispatch_price.ipynb # Multi-horizon predispatch forecasts (critical!)
│   ├── 6_2_predispatch_region_sum.ipynb
│   ├── 6_3_predispatch_interconnector.ipynb
│   ├── 7_pdpasa_region_solution.ipynb # PASA reliability targets
│   ├── 8_1_bid_stack.ipynb         # Supply curve analysis
│   ├── 8_2_bid_prices.ipynb        # Bid strategy signals
│   ├── 9_target_features.ipynb     # Time-of-day, calendar features
│   └── Feature_data/               # Output: parquets by feature group
├── 3_Features_select/              # Feature ranking & selection
│   ├── 0_correlation_matrix.ipynb  # Remove multicollinear pairs
│   ├── 1_aggregate_targets.ipynb   # Target construction per state/horizon
│   ├── 2_feature_ranking.ipynb     # Mutual information ranking
│   ├── 3_remove_duplicate_features.ipynb
│   ├── 4_feature_selection.ipynb   # CV-based best-K selection
│   ├── 5_feature_output.ipynb      # Final feature matrices
│   └── Selected_features/          # Output: X_train, X_validate, X_test
├── 4_Model/
│   ├── 0_vars.ipynb                # Model hyperparameters, training config
│   ├── 1_feature_creation.ipynb    # Load & validate feature matrices
│   ├── 2_train_valid_test_split.ipynb # Temporal split (pre/post-2023)
│   ├── 3_models.ipynb              # Main training: 6 components × 96 horizons
│   ├── Data/                       # Input: X_train, y_train, etc. (parquets)
│   └── parameters/
│       ├── __init__.py
│       ├── lgbm_params.py          # LightGBM hyperparameters (19 configs)
│       └── search_grids.py         # Hyperparameter tuning grids
├── Later/
│   ├── 3_evaluate.ipynb            # Backtesting, metrics, visualizations
│   ├── 4_save.ipynb                # Export predictions to S3/local
│   └── m3.ipynb                    # Sandbox/experiments
├── support/
│   ├── AGENTS.md                   # Copilot agent definitions
│   ├── Analysis.md                 # Known issues & improvements (20 items)
│   ├── commands.md                 # Common CLI/terminal commands
│   ├── data_checker.ipynb          # Data validation utilities
│   ├── parquet_sizes.py            # Inspect artifact sizes
│   ├── paths.py                    # Path resolution: local vs S3
│   ├── requirements-main.txt       # Python 3.13 dependencies
│   ├── requirements-subprocess.txt # Python 3.11 subprocess (nemseer)
│   ├── local_ssh/
│   │   ├── ssh_config
│   │   └── Credentials/            # AWS keys (SSH and API access)
│   └── 99_old/                     # Legacy scripts (deprecated)
└── README.md                       # Quick start guide
```

---

## Data Pipeline & Flow

### Stage 1: Raw Data Ingestion (1_Dataset)

**Input**: AEMO market data (nemosis API), weather APIs, manual calendars  
**Process**:
- Download 5-min market data via nemosis.download_raw_data()
- Validate for NaNs, duplicates, extreme outliers
- Resample/aggregate to 30-min intervals if needed
- Fetch predispatch forecasts (published forecasts of RRP for next 48 hours)
- Match weather data (temperature, solar irradiance) by timestamp

**Output** (Processed_data/):
- dispatch_price_<STATE>.parquet          # Historical RRP (target)
- dispatch_region_sum_<STATE>.parquet     # Regional demand sum
- predispatch_price_<STATE>.parquet       # Multi-horizon forecasts (h1..h96)
- generation_by_fuel_<STATE>.parquet      # Thermal/renewable breakdown
- weather_<LOCATION>.parquet              # Temp, CDD, HDD

### Stage 2: Feature Engineering (2_Features_build)

**Input**: Processed parquets from Stage 1  
**Process**: 8–10 parallel feature builders create lagged, aggregated, interaction features

**Key Feature Groups** (200–400+ total):

| Group | Count | Examples |
|-------|-------|----------|
| **Time** | 10–15 | hour_sin, dow, is_holiday, peak_flag |
| **Price Lags** | 50–80 | rrp_lag_1, rrp_lag_48, rrp_mean_daily |
| **System** | 40–60 | reserve_margin, demand_surprise, coal_util |
| **Predispatch** | 96 | h1_forecast, h24_forecast, ..., h96_forecast ⭐ |
| **Weather** | 15–20 | temp, cdd_demand_interaction, hdd_interaction |
| **Bid/Supply** | 20–30 | bid_curve_slope, equilibrium_quantity |
| **Interconnectors** | 10–15 | nsw_to_qld_flow, sa_to_vic_flow |
| **Interactions** | 30–50 | supply_stress × predispatch, coal_outage × demand |

**Critical Note**: Predispatch RRP forecasts (h1..h96) are the **single most important exogenous signal**—they represent professional forecasts from AEMO and capture systemic supply/demand imbalances.

### Stage 3: Feature Selection (3_Features_select)

**Process** (4-stage filter):

1. **Correlation Filter** – Remove multicollinear pairs (ρ > 0.95)
2. **Target Aggregation** – Construct per-state, per-horizon targets
3. **Feature Ranking** – Rank by Mutual Information with target; select top-K by CV
4. **Deduplication** – Remove near-duplicates; final feature matrices

**Known Issues**:
- **P1**: Predispatch RRP may not be included (low marginal MI, high conditional importance)
- **P2**: Sparse spike predictors (wind forecasts, outages) missed by MI ranking
- **P3**: CV uses MAE loss; spike-aware model downstream needs spike loss
- **P4**: CV window (2019–2023) outdated; misses post-2022 NEM regime

### Stage 4: Model Training (4_Model)

**Input**: X_train, X_validate, X_test, y_train, y_validate, y_test  
**Process**: Train 6 components × 96 horizons = 672 models in parallel

---

## Feature Engineering

### Complete Feature Groups

#### Time-of-Day & Calendar (9_target_features.ipynb)
- hour_sin, hour_cos, dow_sin, dow_cos (periodic encoding)
- is_peak, is_shoulder, is_holiday, is_weekend
- month of year

#### Price Lag Features (1_dispatch_price.ipynb)
- Intraday: rrp_lag_1, lag_2, lag_4, lag_12
- Daily/Weekly: rrp_lag_48, rrp_lag_336
- Statistics: rrp_mean_1h, rrp_std_1h, rrp_ema_12

#### System Stress Indicators (2_dispatch_region_sum.ipynb, 3_generation_fuel.ipynb)
- demand_sum, supply_available, reserve_margin
- demand_surprise = actual - forecast_24h_ago
- thermal_utilization, renewable_ratio
- coal_outage_flag, gas_util_high

#### Predispatch Forecasts (6_1_predispatch_price.ipynb) ⭐ CRITICAL
- h1_forecast, h2_forecast, ..., h96_forecast (AEMO's published forecasts)
- predispatch_mean, predispatch_std, predispatch_trend
- Represent systemic supply/demand balance information

#### Weather (5_weather.ipynb)
- temp_celsius, cdd, hdd
- demand_cdd_interaction, demand_hdd_interaction

#### Bid & Supply Curve (8_1_bid_stack.ipynb, 8_2_bid_prices.ipynb)
- bid_curve_slope, equilibrium_quantity, supply_at_150
- n_bidders_active

#### Interaction Features (0_all.ipynb)
- supply_stress_signal × predispatch_mean
- coal_outage_flag × demand_sum
- renewable_ratio × wind_forecast_high

---

## Feature Selection Process

### 4-Stage Pipeline

1. **Correlation Filter** – Remove ρ > 0.95 pairs → 250–350 features
2. **MI Ranking** – Rank by mutual_information(feature, target)
3. **CV Best-K** – Grid search K ∈ {50, 100, 150, ..., 300} on 2019–2023 data
4. **Dedup & Output** – Final feature matrices (X_train, X_validate, X_test)

### Known Issues

| Issue | Severity | Root Cause | Fix |
|-------|----------|-----------|-----|
| Predispatch may drop | High | MI-based ranking ignores conditional importance | Rerank by LightGBM gain post-CV |
| Sparse spike predictors | High | MI on continuous target; misses 1% spikes | Run parallel MI for spike events |
| Calendar dropout | Medium | Correlation + MI competition | Add MUST_KEEP_FEATURES env |
| Wrong CV loss | Medium | MAE selected for; spike model wants quantile | Change to RMSE/quantile loss |
| Outdated CV window | Medium | 2019–2023; misses post-2022 regime | Extend to 2026 or rolling split |

---

## Model Architecture

### 6-Component Ensemble Per Horizon

For each forecast step h ∈ {1, 2, ..., 96}:

#### Component 1: Base L1 (Median Model)
- Objective: regression with L1 (MAE) loss
- Target: y_h_base = asinh(min(y_h, p97_threshold))
- Weight: recency × spike_upweight
- Use case: Robust median price forecast

#### Component 2: Base L2 (Mean Model)
- Objective: regression with L2 (MSE) loss
- Target: y_h_full = asinh(y_h) (uncapped)
- Weight: recency × spike_upweight
- Use case: Capture mean, influenced by spikes

#### Components 3–5: Spike Chain (if ≥20 spikes)
- **Spike Classifier**: Detects high-price events (y > $150)
- **Spike Regressor**: Specialized predictions for extreme prices
- **Spike Quantile**: q95 confidence intervals

#### Component 6: Dip Chain (if ≥20 dips, y < $0)
- **Dip Classifier** + **Dip Regressor**: Handle negative prices

### Ensemble Prediction Policy

**Hard Gating** (Preferred):
```
if spike_prob > threshold:
    y_pred = spike_regressor_pred + uncertainty
elif dip_prob > threshold:
    y_pred = dip_regressor_pred
else:
    y_pred = 0.5 * base_l1 + 0.5 * base_l2
```

**Post-hoc**: Isotonic calibration if ≥500 validation samples

---

## Configuration & Constants

### Key Constants (0_Config, 4_Model)

| Constant | Value | Purpose |
|----------|-------|---------|
| SPIKE_THRESHOLD | $150/MWh | Spike event definition |
| _DIP_THRESHOLD | $0/MWh | Dip (negative price) event |
| BASE_CLIP_PERCENTILE | 97 | Clip base model target at p97 |
| PRICE_TRANSFORM_SCALE | 100 | arcsinh divisor |
| _SPIKE_UPWEIGHT | 10.0 | Spike regressor weight |
| _DIP_UPWEIGHT | 7.0 | Dip regressor weight |
| _MIN_SPIKE_TRAIN | 20 | Min samples to train spike chain |
| EARLY_STOPPING_ROUNDS | 75 | Base model early stopping |
| SPIKE_ES_ROUNDS | 50 | Spike/dip model early stopping |

### LightGBM Hyperparameters (parameters/lgbm_params.py)

- **LGBM_BASE_PARAMS**: max_depth=7, learning_rate=0.05, n_estimators=500
- **LGBM_BASE_L2_PARAMS**: Same but metric='mse'
- **LGBM_CLF_PARAMS**: max_depth=6, is_unbalanced=True (spike classifier)
- **LGBM_SPIKE_PARAMS**: max_depth=8, learning_rate=0.03, n_estimators=700
- **Similar for DIP variants**

---

## Dependencies & Environment

### Python Versions

| Use | Version | Reason |
|-----|---------|--------|
| Main pipeline | 3.13 | Latest stable, better performance |
| Subprocess (nemseer) | 3.11 | nemseer library only supports 3.11 |

### Main Dependencies (requirements-main.txt)

```
pandas>=2.0, numpy>=1.24, pyarrow>=13.0
lightgbm>=4.0, scikit-learn>=1.3
nemosis>=0.9.3, holidays>=0.35
joblib>=1.3, tqdm>=4.65, matplotlib>=3.7, seaborn>=0.12
s3fs>=2023.9, boto3>=1.28 (optional)
```

---

## Key Files & Functions

### Core Utilities

- **support/paths.py**: Path resolution (local vs S3)
- **support/data_checker.ipynb**: Data validation
- **support/parquet_sizes.py**: Artifact size inspection

### Notebook Execution Order

1. 0_Config/0_variables.ipynb
2. 1_Dataset/1_Data_sources.ipynb
3. 2_Features_build/0_all.ipynb
4. 3_Features_select/0_correlation_matrix.ipynb → 4_feature_selection.ipynb
5. 4_Model/0_vars.ipynb → 1_feature_creation.ipynb → 2_train_valid_test_split.ipynb
6. 4_Model/3_models.ipynb
7. Later/3_evaluate.ipynb

### Key Data Structures

**Feature Matrix** (X_train, X_validate, X_test):
- Shape: (n_samples, n_features) ~ (50k, 180)
- Type: float32
- Index: DatetimeIndex (UTC, 30-min intervals)
- Storage: Parquet (zlib-9, ~180 MB)

**Target Matrix** (y_train, y_validate, y_test):
- Shape: (n_samples, 96)
- Columns: ['h1', 'h2', ..., 'h96']
- Values: float (RRP in $/MWh, raw)
- Storage: Parquet (~8 MB)

**Trained Models**:
- 7 model types × 96 horizons = 672 models
- Format: Joblib (sparse trees)
- Size: ~50 MB each, ~35 GB total

---

## Known Issues & Gaps

### High Priority (P1–P2)

| ID | Issue | Root Cause | Impact | Fix | Effort |
|----|-------|-----------|--------|-----|--------|
| **P1** | Predispatch RRP may drop | MI-based ranking ignores conditional importance | 10%+ accuracy loss | Rerank by gain; force-keep | 2h |
| **P2** | Sparse spike predictors | MI on continuous target | Missed spike drivers | Parallel MI for spike events | 1h |
| **P2c** | Calendar features dropout | Correlation + MI competition | Seasonal broken | MUST_KEEP_FEATURES | 30m |
| **P3** | Wrong CV loss (MAE) | Misaligned objectives | Suboptimal features | Use quantile loss | 1h |
| **P4** | Outdated CV window | 2019–2023; misses post-2022 | Features not current | Extend to 2026 | 2h |

### Medium Priority (P5–P8)

| ID | Issue | Severity | Fix | Effort |
|----|-------|----------|-----|--------|
| **P5** | No spike prediction policy | High | Implement gating logic | 3h |
| **P6** | Multistate duplication | Medium | Multitask architecture | 8h |
| **P7** | No retraining schedule | Medium | Rolling refit | 4h |
| **P8** | S3 incomplete | Medium | Test + document | 2h |

### Technical Debt

- No unit tests
- Hardcoded paths in notebooks
- Data version control absent (use DVC)
- Notebooks not modularized
- No docstrings in feature engineering

---

## Improvements & Recommendations

### Quick Wins (1–2 hours)

1. Force-keep predispatch + core features via MUST_KEEP_FEATURES
2. Add spike-aware feature selection (parallel MI ranking)
3. Fix CV loss to RMSE/quantile
4. Add data validation assertions

### Medium Effort (4–8 hours)

5. Implement spike prediction policy (gating logic)
6. Add rolling retraining schedule
7. Hyperparameter tuning (Optuna)
8. Uncertainty quantification (quantile predictions)

### High Effort (1–2 weeks)

9. Multitask multihorizon architecture
10. Data version control (DVC)

---

## Quick Reference

### Essential Files

| Path | Purpose | Edit Frequency |
|------|---------|-----------------|
| 0_Config/0_variables.ipynb | Global state | Monthly |
| 3_Features_select/4_feature_selection.ipynb | Feature selection | Quarterly |
| 4_Model/0_vars.ipynb | Model hyperparams | Monthly |
| 4_Model/3_models.ipynb | Training loop | Rarely |
| parameters/lgbm_params.py | LightGBM configs | Quarterly |

### Common Commands

```bash
# Full pipeline
jupyter nbconvert --execute 0_Config/0_variables.ipynb
jupyter nbconvert --execute 1_Dataset/1_Data_sources.ipynb
# ... see support/commands.md for full sequence

# Data quality check
jupyter notebook support/data_checker.ipynb

# Inspect sizes
python support/parquet_sizes.py
```

### Typical Performance

| Metric | Value | Notes |
|--------|-------|-------|
| RMSE (all prices) | $25–35/MWh | Standard forecasting |
| RMSE (spikes) | $60–100/MWh | Harder to predict |
| Spike detection AUC | 0.75–0.85 | Classifier accuracy |
| Training time | 1–2 hours | Parallelized |
| Inference time | <1 sec | 96 horizons × 4 states |

### Debugging Checklist

- ✅ X_train.index == y_train.index (alignment)
- ✅ All features float32 (dtype)
- ✅ Count NaNs: X_train.isnull().sum()
- ✅ Features scaled: mean ~0, std ~1
- ✅ No data leakage (future features in training)
- ✅ Non-overlapping train/val/test windows

---

**Last Updated**: 2026-05-20  
**Author**: Copilot Analysis  
**Scope**: Complete system architecture, 5-stage pipeline, 672 models


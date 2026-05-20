# Forecasting Repository — Comprehensive Analysis

**Project**: NEM Regional Electricity Price Forecasting  
**Domain**: 5-minute interval RRP forecasts for NSW, QLD, VIC, SA  
**Time Window**: 2018–2026  
**Repository**: `/Users/danieldavaris/Forecasting`

---

## 1. PROJECT STRUCTURE & DIRECTORIES

### Overview
```
Forecasting/
├── 0_Config/              → Pipeline configuration & variable definitions
├── 1_Dataset/             → Data ingestion, loading, and raw analysis
├── 2_Features_build/      → Feature engineering (200–400+ features)
├── 3_Features_select/     → Feature ranking, dedup, and selection
├── 4_Model/               → Model training, validation, evaluation
├── Later/                 → Evaluation & model-save notebooks (not in main flow)
├── support/               → Shared utilities, old reference code
├── README.md              → Project overview
└── AGENTS.md              → Agent rules and conventions
```

### Directory Details

#### **0_Config/**
- **Purpose**: Environment setup and global configuration
- **Key file**: `0_variables.ipynb`
  - Sets environment variables: `TARGET`, `FEATURE_DATASET`, `HORIZON_HOURS`, `OUTPUT_RESOLUTION`, date ranges
  - Used as a `:run` dependency across all downstream notebooks

#### **1_Dataset/**
- **Purpose**: Raw data ingestion and exploratory analysis
- **Subdirectories**:
  - `Processed_data/`: Output parquet files (processed/cached raw data)
  - `Pre_processing/temporary_cache/`: Intermediate downloads (AEMO/weather raw data)
  
- **Key notebooks**:
  - `0_Analysis.ipynb` – Interactive data exploration (time series plots via Plotly)
  - `1_Data_sources.ipynb` – Data ingestion pipeline:
    - Fetches AEMO NEM data via `nemosis` library
    - Fetches weather (Sydney, Melbourne, Brisbane, Adelaide)
    - Merges into consolidated 30-min parquet files
    - Outputs parquet files to `Processed_data/`:
      - `1_dispatch_price.parquet` – Regional RRPs (NSW, QLD, VIC, SA)
      - `2_dispatch_region_sum.parquet` – System totals (avail_gen, interchange, demand forecast)
      - `3_generation_fuel.parquet` – Generator/fuel mix by region
      - `4_STTM_DWGM.parquet` – Gas market prices
      - `5_weather.parquet` – Temperature, wind speed
      - `6_1_predispatch_price.parquet` – Multi-horizon predispatch RRPs (h1–h96)
      - `6_2_predispatch_region_sum.parquet` – Predispatch system forecasts

#### **2_Features_build/**
- **Purpose**: Feature engineering from raw data
- **Output**: Single wide feature matrix `Feature_data/1_dispatch_price.parquet` (~200–400+ float32 columns)
- **Per-notebook outputs**:
  - `0_all.ipynb` – Adds interaction features (cross-products of price, supply, demand)
  - `1_dispatch_price.ipynb` – Time features, price lags (1, 2, 4, 12, 48, 96, 336 day patterns)
  - `2_dispatch_region_sum.ipynb` – Reserve margin, thermal utilization, demand surprise
  - `3_generation_fuel.ipynb` – Coal outage flags, fuel mix ratios
  - `4_STTM_DWGM.ipynb` – Gas forward market pressure indicators
  - `5_weather.ipynb` – CDD/HDD demand interactions
  - `6_1_predispatch_price.ipynb` – Predispatch RRPs aligned to horizons (h1–h96 features)
  - `6_2_predispatch_region_sum.ipynb` – Predispatch system forecasts aligned per horizon
  - `6_3_predispatch_interconnector.ipynb` – (Optional) Interconnector flow predictions

#### **3_Features_select/**
- **Purpose**: Feature ranking, deduplication, and horizon-specific selection
- **Output**: 
  - `Selected_features/NSW_selected_features_1_dispatch_price.parquet` – Boolean mask (n_features × 96 horizons)
  - `Selected_features/NSW_targets_agg_1_dispatch_price.parquet` – Target matrix (target_h1 .. target_h96)
  
- **Pipeline (4-stage filter)**:
  - **Stage 1: Relevance** (`2_feature_ranking.ipynb`)
    - Mutual information per (feature, horizon) on 200k subsample (2019–2022)
    - Output: `gain_matrix`, `horizon_best_k`
  
  - **Stage 2: Redundancy** (`4_remove_duplicate_features.ipynb`)
    - Greedy dedup in MI-rank order
    - Drop features with |Pearson| > 0.95 OR |Spearman| > 0.95 correlation to a kept feature
  
  - **Stage 3: Conditional Importance** (`5_feature_selection.ipynb`)
    - LightGBM TimeSeriesSplit CV on full-range target
    - Grid search k ∈ 10 linearly-spaced values (1 → n_survived_features)
    - Selects best_k per horizon minimizing MAE
  
  - **Stage 4: Output** (`6_feature_output.ipynb`)
    - Per-horizon: select top-best_k features by MI within dedup survivors
    - Output: Boolean matrix (1 if feature selected for horizon, 0 else)
  
- **Known limitations** (documented in Analysis.md):
  - Stage 4 ranks by raw MI; doesn't use LightGBM gain (can miss conditional-importance features)
  - Greedy dedup can drop wrong sibling (e.g., QLD price when NSW is leader)
  - No spike-aware ranking (spikes dominated by ~1% of intervals)
  - Calendar/predispatch features not forced; at MI's mercy

#### **4_Model/**
- **Purpose**: Model training, tuning, evaluation, and serialization
- **Structure**:
  - `0_vars.ipynb` – Model constants, LightGBM hyperparameters (commented-out reference)
  - `1_feature_creation.ipynb` – Cross-feature builders (auxiliary/cross-feature computation)
  - `2_train_valid_test_split.ipynb` – Data prep + temporal split
  - `3_models.ipynb` – Main training loop (6-component spike-aware ensemble per horizon)
  - `Data/` – X_train, X_validate, X_test, y_train, y_validate, y_test parquet files
  - `parameters/` – Python module with LightGBM hyperparameters + search grids

- **Model Components** (per horizon):
  1. **Base L1** – regression_l1 on arcsinh(clip(y, p97))
     - MAE-optimal (median prediction)
     - Clipping removes spike noise from loss
  2. **Base L2** – regression (MSE) on arcsinh(y) full range
     - MSE-optimal (mean prediction)
     - Captures upside risk in heavy-tailed NEM prices
  3. **Spike Classifier** – binary logistic, P(price > $150)
     - scale_pos_weight=7 for class imbalance
  4. **Spike Regressor** – regression_l1 on full range, 10x spike-row weight
  5. **Spike Quantile (P90)** – quantile regression for conservative ceiling
  6. **Dip Classifier + Regressor** – mirrors spike design for y < $0

- **Training Details**:
  - **Temporal split**: Train on all except last 18 months; valid = last 6 months before test
  - **Recency weighting**: exp(linspace(0, 1.5, n_train)) so newest rows ~4.5× oldest
  - **Per-horizon features**:
    - Base features (from 2_Features_build)
    - 8 target-time features (hour/dow sin-cos + regime flags)
    - 4 cross-features (day-of-year cycle, SA-NSW spread, region spike count)
    - 8 predispatch-aligned features
  - **Hyperparameters**: n_estimators={1200–1400}, learning_rate=0.025, num_leaves={63–95}
  - **Early stopping**: 75 rounds (base), 50 rounds (spike/dip)
  - **Validation-tuning per horizon**:
    - L1/L2 blend weight α ∈ [0, 0.45] (10 values)
    - Spike policy (3 kinds × grid of spike_thr, spike_pow, spike_wmax, spike_gate_w)
    - Dip policy (3 kinds × similar grid)
    - Naive-blend α ∈ [0, 1] (41 values)
  - **Isotonic calibration**: Per-horizon on validation set (if ≥500 samples)

#### **Later/**
- `3_evaluate.ipynb` – Test-set evaluation (per-step metrics, breakdown by spike/dip/normal)
- `4_save.ipynb` – Model serialization via joblib + reporting
- `m3.ipynb` – Miscellaneous exploration

#### **support/**
- `paths.py` – Central path resolver (local vs S3 I/O)
- `parquet_sizes.py` – Utility to list parquet file sizes
- `requirements-main.txt` – Python 3.13 dependencies (nemosis, lightgbm, scikit-learn, pandas, pyarrow, holidays)
- `requirements-subprocess.txt` – Python 3.11 subprocess env (nemseer)
- `AGENTS.md` – Agent rules: never delete code, always merge, preserve structure
- `Analysis.md` – In-depth feature-selection design doc; known gaps and fixes
- `commands.md` – AWS S3 sync commands, process kill commands
- `99_old/` – Archived code (reference only):
  - `config.py` – Old forecast gap/horizon combos
  - `main.py` – Old end-to-end pipeline entry point
  - `export_predictions.py` – Old prediction export
  - `feature_engineering.py` – Old feature builder (reference patterns)
  - `evaluate_all_states.py` – Old evaluation harness

---

## 2. NOTEBOOK FILES — COMPLETE INVENTORY

### 0_Config/
| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `0_variables.ipynb` | Environment setup | Sets TARGET, FEATURE_DATASET, HORIZON_HOURS, OUTPUT_RESOLUTION, date ranges |

### 1_Dataset/
| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `0_Analysis.ipynb` | Exploratory analysis | Plotly time series, data coverage plots |
| `1_Data_sources.ipynb` | Data ingestion | 6 parquet files: prices, system, fuel, gas, weather, predispatch |

### 2_Features_build/
| Notebook | Purpose | Output Columns |
|----------|---------|-----------------|
| `0_all.ipynb` | Interaction features | supply_stress_x_price_l1, pd_rrp_x_supply_stress2, coal_outage_x_high_demand, etc. |
| `1_dispatch_price.ipynb` | Time + price lags | hour_sin/cos, dow_sin/cos, is_peak, is_holiday, nsw_price_asinh_lag_{1,2,4,12,48,96,336}, etc. |
| `2_dispatch_region_sum.ipynb` | System features | reserve_margin, demand_fcst_error, avail_gen_lag_{1,2,4,48,96,336}, thermal_util, etc. |
| `3_generation_fuel.ipynb` | Fuel mix | Coal generation, fuel ratios, outage flags |
| `4_STTM_DWGM.ipynb` | Gas market | Forward market spreads, pressure indicators |
| `5_weather.ipynb` | Demand interactions | CDD, HDD, wind interactions with demand |
| `6_1_predispatch_price.ipynb` | Predispatch RRPs | predispatch_rrp_h1 .. predispatch_rrp_h96 (multi-horizon) |
| `6_2_predispatch_region_sum.ipynb` | Predispatch system | Predispatch avail_gen, demand, reserve aligned to each horizon |
| `6_3_predispatch_interconnector.ipynb` | (Optional) Interconnector forecasts | Interconnector flow predictions |

**Output**: Single parquet `2_Features_build/Feature_data/1_dispatch_price.parquet` with 200–400+ float32 columns + DatetimeIndex

### 3_Features_select/
| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `0_variables.ipynb` | Env setup | Inherits from 0_Config; sets feature selection ranges |
| `1_aggregate_targets.ipynb` | Target building | Creates {TARGET}_targets_agg_1_dispatch_price.parquet (shifted h1–h96) |
| `2_feature_ranking.ipynb` | MI ranking | horizon_best_k, gain_matrix |
| `3_correlation_matrix.ipynb` | Correlation analysis | Pearson/Spearman correlation heatmap |
| `4_remove_duplicate_features.ipynb` | Dedup by correlation | Greedy dedup, flags correlates |
| `5_feature_selection.ipynb` | CV feature selection | LightGBM TimeSeriesSplit, best_k per horizon by MAE |
| `6_feature_output.ipynb` | Final selection | NSW_selected_features_1_dispatch_price.parquet (bool mask n_features × 96) |

### 4_Model/
| Notebook | Purpose | Key Computations |
|----------|---------|------------------|
| `0_vars.ipynb` | Constants + hyperparams | SPIKE_THRESHOLD=$150, BASE_CLIP_PERCENTILE=97%, LightGBM params |
| `1_feature_creation.ipynb` | Cross-feature builders | compute_cross_feats() → doy_sin/cos, sa_spread, region_spike_score |
| `2_train_valid_test_split.ipynb` | Data prep | Loads features/targets, splits temporal, outputs X_train/X_validate/X_test, y_* |
| `3_models.ipynb` | **Main training** | train_seq2seq() → 6 models × 96 horizons, validation-tuned parameters |

### Later/
| Notebook | Purpose |
|----------|---------|
| `3_evaluate.ipynb` | Per-step metrics, spike/dip breakdown, skill score, feature importance |
| `4_save.ipynb` | Serialize models via joblib, generate Excel accuracy reports |
| `m3.ipynb` | Exploration/scratch |

---

## 3. PYTHON SCRIPTS — DETAILED ANALYSIS

### **support/paths.py**
**Purpose**: Central path resolver for local & S3 environments

**Key Functions**:
- `resolve(repo_root_relative: str) -> str`
  - **Input**: Relative path from repo root (e.g., "1_Dataset/Processed_data/1_dispatch_price.parquet")
  - **Output**: Absolute local path (default) or S3 URI (if `USE_S3=1`)
  - **Logic**: Checks `USE_S3` environment variable; returns `s3://forecasting-nem-dd/<key>` if true

**Constants**:
- `S3_BUCKET = "forecasting-nem-dd"`
- `_REPO_ROOT = Path(__file__).parent` (auto-resolve to support/ directory)

**Usage**:
```python
import sys; sys.path.insert(0, "..")
from paths import resolve
df = pd.read_parquet(resolve("1_Dataset/Processed_data/1_dispatch_price.parquet"))
```

### **4_Model/parameters/lgbm_params.py**
**Purpose**: LightGBM hyperparameter presets for 6 model components

**Defined Dicts** (all exported via `__init__.py`):

1. **LGBM_BASE_PARAMS**
   - `objective: "regression_l1"`, `metric: "l1"`
   - `n_estimators: 1400`, `learning_rate: 0.025`
   - `num_leaves: 95`, `max_depth: -1`
   - `feature_fraction: 0.80`, `bagging_fraction: 0.85`, `bagging_freq: 5`
   - `reg_alpha: 0.05`, `reg_lambda: 0.20`
   - **Purpose**: MAE-optimal base model on clipped target

2. **LGBM_BASE_L2_PARAMS**
   - `objective: "regression"` (MSE), `metric: "rmse"`
   - `n_estimators: 1200` (vs 1400 for L1)
   - **Purpose**: Mean-optimal base model (captures upside risk)

3. **LGBM_CLF_PARAMS**
   - `objective: "binary"`, `metric: "binary_logloss"`
   - `n_estimators: 500`, `num_leaves: 63`
   - `scale_pos_weight: 7.0`
   - **Purpose**: Spike binary classifier P(price > $150)

4. **LGBM_SPIKE_PARAMS**
   - `objective: "regression_l1"`, `n_estimators: 500`
   - **Purpose**: Spike regressor on full range, upweighted spike rows

5. **LGBM_SPIKE_Q_PARAMS**
   - `objective: "quantile"`, `alpha: 0.90`
   - **Purpose**: P90 quantile regressor for conservative spike ceiling

6. **LGBM_DIP_CLF_PARAMS**
   - `objective: "binary"`, `scale_pos_weight: 6.0`
   - **Purpose**: Dip (negative price) classifier

7. **LGBM_DIP_PARAMS**
   - `objective: "regression_l1"`, `n_estimators: 400`
   - **Purpose**: Dip regressor on full range

**All params set**: `n_jobs: 1`, `num_threads: 1` (parallelism via outer joblib loop over horizons)

### **4_Model/parameters/search_grids.py**
**Status**: Currently empty/commented-out. Contains legacy grids for validation-tuning:
- `_SPIKE_THR_GRID, _SPIKE_POW_GRID, _SPIKE_WMAX_GRID, _SPIKE_GATE_W_GRID`
- `_DIP_THR_GRID, _DIP_POW_GRID, _DIP_WMAX_GRID, _DIP_GATE_W_GRID`
- Intended for spike/dip policy hyperparameter sweeps (in 4_Model/3_models.ipynb)

### **4_Model/parameters/__init__.py**
**Purpose**: Export all LightGBM params and search grids as a module

**Exports**:
```python
from .lgbm_params import (LGBM_BASE_PARAMS, LGBM_BASE_L2_PARAMS, LGBM_CLF_PARAMS, ...)
from .search_grids import (_SPIKE_THR_GRID, ...)
```

### **support/99_old/config.py**
**Purpose**: (Archived) Legacy forecast configuration

**Key Constants**:
- `DATA_START_DATE = "2018/01/01"`, `DATA_END_DATE = "2026/01/01"`
- `FORECAST_GAP_HORIZON_COMBOS = [(1, 16), (12, 24)]`
  - (gap=1, horizon=16): 30 min ahead, 8-hour window
  - (gap=12, horizon=24): 6 hours ahead, 12-hour window

**Functions**:
- `cfg_tag(gap, horizon, data_start, data_end) -> str`
  - Returns compact config string for filenames (e.g., "g1h16_2018_2026")
- `get_forecast_step_intervals() -> list`
  - Returns sorted unique steps across all gap/horizon combos

**Status**: Reference only; superseded by 0_Config/0_variables.ipynb

### **support/99_old/main.py**
**Purpose**: (Archived) End-to-end pipeline orchestrator

**Functions**:
- `_load(region, data_start, data_end) -> df` – Load + filter data
- `_build(region_df, gap, horizon) -> (df_full, feature_cols)` – Feature engineering
- `_select(df_full, feature_cols, ...) -> feature_cols` – Feature selection
- `_train(df_full, feature_cols, ...) -> (model, past_cols, scaler)` – Training
- `_evaluate(model, df_full, ...) -> None` – Evaluation
- `run_region(region, gap, horizon, data_start, data_end, ...) -> None` – Entry point

**Output artifacts**:
- Excel accuracy reports with aggregate, per-step, and feature-importance sheets
- Parquet predictions CSV with (Timestamp, Predictions) columns

### **support/99_old/export_predictions.py**
**Purpose**: (Archived) Export 2023 predictions per state to Excel

**Functions**:
- `_load_region_frame(raw_df, region) -> df` – Load + feature + lag processing
- `_predict_horizon_one(model, df, feature_cols) -> Series` – Single-horizon forecast

**Output**: One Excel file per state with (Timestamp, Predictions)

### **support/99_old/feature_engineering.py**
**Purpose**: (Archived) Reference patterns for feature building

**Key Functions**:
- `select_region_columns(raw_df, region) -> df` – Extract region-specific columns
- Feature groups:
  - Time/calendar: hour, day-of-week, month, sin/cos encoding, holiday
  - Lags: [1, 2, 3, 4, 6, 8, 12, 24, 48, 96, 336, 672 steps]
  - Rolling stats: mean, std, min, max over [4, 8, 24, 48, 96, 336, 672, 2016 steps]
  - Price regimes: recent spikes, negative prices, volatility
  - Degree-days: CDD/HDD with base_temp=18°C

**Constants**:
- `ANNUAL_LAG = 17_532` (1 year in 30-min intervals)
- `NSW_COAL_MAX_MW = 8_500` (Eraring + Bayswater + Mt Piper + Vales Point)
- `_BASE_TEMP_C = 18.0` (Australian standard)

### **support/99_old/evaluate_all_states.py**
**Purpose**: (Archived) Multi-region evaluation harness

### **support/parquet_sizes.py**
**Purpose**: Utility to list all parquet files by size

**Output**: Table of (Size MB, File path), total GB, count

---

## 4. CONFIGURATION & ENVIRONMENT VARIABLES

### **0_Config/0_variables.ipynb** (Primary Configuration)

**Environment Variables Set**:
- `TARGET` = "NSW" (default; can be QLD, VIC, SA)
- `FEATURE_DATASET` = "1_dispatch_price.parquet"
- `FEATURE_DATASET_START` = "2019/01/01" (feature build window)
- `FEATURE_DATASET_END` = "2026/01/01"
- `FEATURE_SELECTION_SUBSAMPLE_START` = "2019/01/01" (CV training window)
- `FEATURE_SELECTION_SUBSAMPLE_END` = "2023/01/01"
- `HORIZON_HOURS` = 48 (default)
- `OUTPUT_RESOLUTION` = 30 (minutes)
- Default forecast: 48 hours × 30-min intervals = 96 horizons (h1–h96)

### **4_Model/0_vars.ipynb** (Model Constants)

**Core Constants**:
- `PRICE_TRANSFORM_SCALE = 100.0` (arcsinh scale factor)
- `SPIKE_THRESHOLD = 150.0` ($/MWh; defines spike regime)
- `_DIP_THRESHOLD = 0.0` ($/MWh; defines dip regime)
- `BASE_CLIP_PERCENTILE = 97.0` (clip base model target at p97 to remove spike noise)
- `_SPIKE_UPWEIGHT = 10.0` (sample weight multiplier for spike rows)
- `_DIP_UPWEIGHT = 7.0` (sample weight multiplier for dip rows)
- `_MIN_SPIKE_TRAIN = 20` (minimum spike samples to train spike chain)
- `EARLY_STOPPING_ROUNDS = 75` (base models)
- `SPIKE_ES_ROUNDS = 50` (spike/dip components)

### **Path Resolution** (support/paths.py)

- **Local mode** (default): `resolve(path) → /Users/danieldavaris/Forecasting/<path>`
- **S3 mode** (if `USE_S3=1`): `resolve(path) → s3://forecasting-nem-dd/<path>`
- **S3 Bucket**: `forecasting-nem-dd` (region: ap-southeast-2)

### **Environment Setup** (support/AGENTS.md)

- **Main env** (Python 3.13): `C:\Users\danie\.venvs\forecasting-main`
  - Dependencies: nemosis, pandas, numpy, pyarrow, scikit-learn, lightgbm, holidays, matplotlib, seaborn, joblib, ipykernel, nbformat
  - Used for all notebooks except subprocess calls

- **Subprocess env** (Python 3.11): `C:\Users\danie\.venvs\forecasting-subprocess`
  - Dependencies: nemseer, pandas
  - Invoked as subprocess from main env (nemseer requires 3.11)

---

## 5. DATA PIPELINE & FLOW

### **Stage 1: Data Ingestion (1_Dataset/)**

**Input**: AEMO NEM public data + external weather feeds  
**Process** (1_Data_sources.ipynb):
1. Fetch via `nemosis` (AEMO API) for 2018–2026
2. Fetch weather (Sydney, Melbourne, Brisbane, Adelaide)
3. Merge 30-min intervals
4. Output 6 parquet files to `Processed_data/`:

| File | Columns | Granularity |
|------|---------|------------|
| `1_dispatch_price.parquet` | nsw_price, qld_price, vic_price, sa_price | 30-min |
| `2_dispatch_region_sum.parquet` | avail_gen, interchange, demand_fcst | 30-min (system-wide) |
| `3_generation_fuel.parquet` | coal_mw, gas_mw, hydro_mw, solar_mw, wind_mw (per region) | 30-min |
| `4_STTM_DWGM.parquet` | gas_forward_price, spread | 30-min |
| `5_weather.parquet` | temp_sydney, wind_sydney, temp_melbourne, etc. | 30-min or hourly |
| `6_1_predispatch_price.parquet` | predispatch_rrp_h1_nsw, ..., predispatch_rrp_h96_nsw, etc. | 30-min (multi-horizon) |

**Output**: Wide parquet files cached locally or on S3

---

### **Stage 2: Feature Engineering (2_Features_build/)**

**Input**: `1_Dataset/Processed_data/*.parquet`  
**Process**: 9 notebooks compute feature groups

| Feature Group | Notebooks | Method | Output Columns |
|---------------|-----------|--------|-----------------|
| Time/Calendar | 1_dispatch_price.ipynb | Sin/cos periodicity | hour_sin, hour_cos, dow_sin, dow_cos, is_peak, is_shoulder, is_off_peak, is_holiday, is_offday, month_sin, month_cos |
| Price Lags | 1_dispatch_price.ipynb | Historical lookback | nsw_price_asinh_lag_{1,2,4,12,48,96,336,335,337}, nsw_price_asinh_rmean_{48,336}, + other regions |
| System Reserves | 2_dispatch_region_sum.ipynb | Computed from avail_gen, demand_fcst | reserve_margin, reserve_margin_pct, demand_fcst_error, avail_gen_lag_{1,2,4,48,96,336}, interchange_lag_{1,2,4,48,96,336} |
| Thermal Utilization | 2_dispatch_region_sum.ipynb | Coal/gas dispatch | thermal_util, thermal_surplus_mw, dispatch_gen_lag_{1,2,4,48,96,336}, rolling stats (rmean_48, rmin_48, etc.) |
| Fuel Mix | 3_generation_fuel.ipynb | Fuel type breakdown | coal_pct, gas_pct, hydro_pct, renewable_pct, coal_outage_flag |
| Gas Market | 4_STTM_DWGM.ipynb | Forward spreads | gas_forward_spread, forward_pressure |
| Weather | 5_weather.ipynb | Demand interactions | CDD, HDD, temp_max, wind_avg (per city) |
| Predispatch RRPs | 6_1_predispatch_price.ipynb | Multi-horizon | predispatch_rrp_h1, predispatch_rrp_h2, ..., predispatch_rrp_h96 (crucial exogenous signal) |
| Interactions | 0_all.ipynb | Cross-products | supply_stress_x_price_l1, pd_rrp_x_supply_stress2, coal_outage_x_high_demand, etc. |

**Output**: Single parquet `2_Features_build/Feature_data/1_dispatch_price.parquet`
- Rows: 30-min DatetimeIndex (2019–2026)
- Columns: 200–400+ float32 features

**Key Insight**: Predispatch RRPs (h1–h96) are the single most important exogenous signal; included alongside 150+ lags/interactions

---

### **Stage 3: Feature Selection (3_Features_select/)**

**Input**: `2_Features_build/Feature_data/1_dispatch_price.parquet`  
**Process**: 4-stage filter

#### **Stage 1: Relevance (2_feature_ranking.ipynb)**
- Mutual information per (feature, horizon) on 200k subsample (2019–2022)
- Output: `gain_matrix` (n_features × n_horizons), per-horizon top-K rankings
- **Limitation**: MI is marginal; doesn't capture spike information; low for calendar features

#### **Stage 2: Redundancy (4_remove_duplicate_features.ipynb)**
- Greedy dedup in MI-rank order
- Remove feature if |Pearson| > 0.95 OR |Spearman| > 0.95 to a kept feature
- Reduces ~400 → ~200–300 unique features
- **Limitation**: Greedy picks wrong sibling sometimes (e.g., NSW price over QLD for spike regime)

#### **Stage 3: Conditional Importance (5_feature_selection.ipynb)**
- LightGBM TimeSeriesSplit CV on continuous target
- Grid search k ∈ 10 linearly-spaced values
- Selects best_k per horizon minimizing MAE
- **Limitation**: CV loss is MAE (wrong for spike-aware system); per-horizon picks may oscillate; no multi-seed averaging

#### **Stage 4: Output (6_feature_output.ipynb)**
- Per-horizon: select top-best_k features by MI within survivors
- Output: `NSW_selected_features_1_dispatch_price.parquet` (boolean mask n_features × 96)
- **Limitation**: Ranks by MI, not by LightGBM gain (misses conditional-importance features like predispatch RRP)

**Output Files**:
- `NSW_targets_agg_1_dispatch_price.parquet` – Target matrix (target_h1 .. target_h96)
- `NSW_selected_features_1_dispatch_price.parquet` – Boolean selection mask (n_features × 96)
- `horizon_best_k_1_dispatch_price.parquet` – Best-k value per horizon

---

### **Stage 4: Model Training (4_Model/)**

**Input**:
- `2_Features_build/Feature_data/1_dispatch_price.parquet` (features)
- `3_Features_select/Selected_features/NSW_targets_agg_1_dispatch_price.parquet` (targets h1–h96)
- `3_Features_select/Selected_features/NSW_selected_features_1_dispatch_price.parquet` (selection mask)

**Process** (5 notebooks):

#### **0_vars.ipynb** – Constants & Hyperparameters
- Defines 7 constants (SPIKE_THRESHOLD, BASE_CLIP_PERCENTILE, upweights, ES rounds)
- Imports 7 LightGBM parameter dicts from `parameters/lgbm_params.py`

#### **1_feature_creation.ipynb** – Cross-Feature Builders
- `compute_cross_feats(df) -> DataFrame`
  - **doy_sin, doy_cos**: 365.25-day annual cycle
  - **sa_spread_live**: arcsinh((SA - NSW) / 100) – interconnector pressure
  - **region_spike_score**: count of QLD/VIC/SA with price > $150 (0–3)
- Auxiliary price columns for naive baseline

#### **2_train_valid_test_split.ipynb** – Data Preparation
- Reads feature + target parquets
- **Temporal split**:
  - Train: all except last 18 months
  - Validate: 6 months before test
  - Test: most recent 12 months
- Outputs: X_train, X_validate, X_test, y_train, y_validate, y_test to `Data/`

#### **3_models.ipynb** – Main Training Loop
**Function**: `train_seq2seq() -> (base_models, base_l2_models, spike_clfs, spike_regs, spike_qregs, dip_clfs, dip_regs)`

**Per-horizon fitting** (parallelized over 96 horizons via joblib):

1. **Load data**:
   - X_train: base features (1500–2000 rows)
   - y_train[target_col]: target for horizon h
   - X_validate, y_validate: validation set

2. **Compute recency weights**:
   ```
   _recency_w = exp(linspace(0, 1.5, n_train))
   ```
   Newest rows ~4.5× oldest (based on Uniejewski et al. 2019)

3. **Build per-horizon feature matrix**:
   ```
   tr_time = _compute_target_time_feats(_tr_idx_dt, horizon)  # 8 feats
   X_h_tr = concatenate([X_tr_base, tr_time], axis=1)
   ```

4. **Compute p97 clip threshold**:
   ```
   _clip_thresh = percentile(y_tr[target_col], 97)
   ```

5. **Train 6 component models**:

   **a. Base L1 Model**
   ```
   y_tr_base_t = arcsinh(min(y_tr, _clip_thresh)) / 100
   base_m = LGBMRegressor(objective='regression_l1', n_estimators=1400, ...)
   base_m.fit(X_h_tr, y_tr_base_t, sample_weight=recency × spike_up3x)
   ```
   - MAE-optimal median prediction
   - Clipping removes spike noise

   **b. Base L2 Model**
   ```
   y_tr_full = arcsinh(y_tr) / 100  # uncapped
   base_l2_m = LGBMRegressor(objective='regression', n_estimators=1200, ...)
   base_l2_m.fit(X_h_tr, y_tr_full, sample_weight=recency × spike_up3x)
   ```
   - MSE-optimal mean prediction
   - Blending L1+L2 reduces both MAE and RMSE

   **c. Spike Classifier** (if n_spikes ≥ 20)
   ```
   spike_labels = (y_tr > 150).astype(float)
   clf = LGBMClassifier(objective='binary', scale_pos_weight=7, ...)
   clf.fit(X_h_tr, spike_labels, sample_weight=recency)
   ```

   **d. Spike Regressor** (if n_spikes ≥ 20)
   ```
   spike_w = recency × where(spike_labels > 0, 10, 1)
   sreg = LGBMRegressor(objective='regression_l1', n_estimators=500, ...)
   sreg.fit(X_h_tr, y_tr_full, sample_weight=spike_w)
   ```

   **e. Spike Quantile (P90)** (if n_spikes ≥ 20)
   ```
   qreg = LGBMRegressor(objective='quantile', alpha=0.90, ...)
   qreg.fit(X_h_tr, y_tr_full, sample_weight=spike_w)
   ```

   **f. Dip Classifier + Regressor** (if n_dips ≥ 20)
   - Similar design as spike, but for y < $0
   - scale_pos_weight=6, upweight=7

6. **Return**:
   - (base_m, base_l2_m, spike_clf, spike_reg, spike_qreg, dip_clf, dip_reg) for each horizon

**Output**: 96 tuples of 7 models each (672 total LightGBM estimators)

---

### **Stage 5: Validation Tuning & Evaluation (Later/)**

**Input**: Trained models, validation set, test set  
**Process** (3_evaluate.ipynb, 4_save.ipynb):

1. **Per-horizon prediction pipeline**:
   - Blend L1 + L2 base predictions (α tuned on validation)
   - Apply spike policy (soft blend / gated uplift / hard gate)
   - Apply dip policy (similar)
   - Blend model with lag-naive baseline (α tuned on validation)
   - Isotonic calibration (if ≥500 samples)

2. **Metrics**:
   - Per-step: MAE, RMSE, R², MBE
   - Aggregate: MAE, RMSE, R², MAPE, WMAPE, MBE
   - Spike breakdown: spike_MAE, nonspike_MAE, spike_%, dip_MAE, dip_%
   - Skill: (1 - model_MAE / naive_MAE) × 100%

3. **Output artifacts**:
   - Model serialization: joblib.dump(model_dict, nsw_model.joblib) with zlib-9 compression
   - Excel accuracy report with sheets: Aggregate, PerStep, FeatureImportance

---

## 6. FEATURE ENGINEERING DETAILS

### **Feature Groups**

#### **Time/Calendar Features**
- **Periodicity**: sin/cos encoding for hour-of-day, day-of-week, month
  - Formula: `sin(2π × value / period)`, `cos(2π × value / period)`
  - Avoids artificial ordering bias
- **Regime flags**:
  - `is_peak`: 7am–10pm weekday
  - `is_shoulder`: 6–7am, 10pm–midnight weekday
  - `is_off_peak`: midnight–6am, all day weekend
  - `is_holiday`: Australian NSW public holidays
  - `is_offday`: weekend or holiday

**Purpose**: Capture daily/weekly/annual electricity demand seasonality

#### **Price Lags**
- **Intervals**: [1, 2, 4, 12, 48, 96, 336, 335, 337]
- **Explanation**: Autoregressive lags at multiple timescales
  - 1–4: Recent intraday momentum
  - 12: 6-hour seasonal pattern
  - 48: 24-hour (daily) pattern
  - 96: 48-hour (two-day) pattern
  - 336: 7-day (weekly) pattern
  - 335, 337: Weekly ±1 offset (handles daylight-saving transitions)
- **Transform**: arcsinh(price / 100)
- **Variants**: Regional (NSW, QLD, VIC, SA) + rolling statistics (mean, min, max over 48/336 windows)

**Purpose**: NEM prices are highly autocorrelated; lags capture momentum

#### **System Features**
- **Reserve margin**: (avail_gen - demand_fcst) / demand_fcst
  - Low margin → higher prices
- **Demand surprise**: demand_actual - demand_fcst
- **Thermal utilization**: dispatch_gen / avail_gen
- **Interconnector flows**: interchange_{nsw,qld,vic,sa}

**Purpose**: System stress is a key price driver

#### **Fuel Mix**
- **Generation by type**: coal_mw, gas_mw, hydro_mw, solar_mw, wind_mw
- **Ratios**: coal_pct, gas_pct, renewable_pct
- **Outage flag**: coal_mw < historical_max (e.g., Eraring offline)

**Purpose**: Fuel constraints and coal scarcity drive price spikes

#### **Weather**
- **Temperatures**: T_min, T_max, T_avg (Sydney, Melbourne, Brisbane, Adelaide)
- **Degree-days**:
  - CDD (Cooling Degree Days): max(0, T_avg - 18)
  - HDD (Heating Degree Days): max(0, 18 - T_avg)
- **Interactions**: CDD × demand_fcst, HDD × demand_fcst

**Purpose**: Extreme temperatures drive HVAC demand → high prices

#### **Predispatch RRPs** (Most Important)
- **Source**: AEMO predispatch forecasts
- **Columns**: predispatch_rrp_h1, predispatch_rrp_h2, ..., predispatch_rrp_h96
- **Meaning**: Market's own forecast of RRP for horizons h1–h96
- **Transform**: arcsinh(predispatch_rrp / 100)

**Purpose**: Predispatch is the single best exogenous signal; crucial for spike forecasting

#### **Interaction Features** (0_all.ipynb)
- **Cross-products**:
  - supply_stress × price_lag_1: When reserve margin is low and price just spiked
  - predispatch_rrp_h{target} × supply_stress: Predispatch signal × system stress
  - coal_outage_flag × high_demand: Coal unavailable during demand peak
  - region_spike_count × price_lag: Neighbor spikes → contagion signal
- **SA-NSW spread**: arcsinh((sa_price - nsw_price) / 100)
  - Interconnector congestion indicator

**Purpose**: Nonlinear interactions improve spike forecasting

---

## 7. FEATURE SELECTION PROCESS

### **Current 4-Stage Pipeline**

**Stage 1: Relevance (Mutual Information)**
- Ranks features by MI(feature; target) per horizon
- **Pros**: Captures nonlinear dependence, ignores linear associations
- **Cons**: 
  - Smooths spikes (rare events)
  - Low on calendar features (cyclical, deterministic)
  - Requires binning (information loss)

**Stage 2: Redundancy (Correlation Dedup)**
- Greedy: keep highest-MI, remove correlates (|r| > 0.95)
- **Pros**: Reduces multicollinearity, shrinks feature set
- **Cons**: Greedy order can drop important sibling (e.g., QLD price if NSW ranks higher but QLD has unique spike info)

**Stage 3: Conditional Importance (LightGBM CV)**
- Per-horizon: sweep k ∈ [1, 10 linear points], pick best_k by MAE
- **Pros**: Accounts for feature interactions; model-based
- **Cons**: 
  - CV loss is MAE (penalizes spikes down); wrong objective for spike-aware system
  - Linear k-grid has poor resolution in elbow region
  - Single subsample + 3 folds → best_k noisy (flips between adjacent values)
  - Per-horizon picks don't smooth; inconsistent feature sets across horizons

**Stage 4: Final Selection (MI-rank)**
- Per-horizon: select top-best_k by MI among survivors
- **Pros**: Simple
- **Cons**: Ignores LightGBM gain (misses conditional-importance features like predispatch RRP)

### **Known Gaps & Recommended Fixes** (from Analysis.md)

**Priority 1 (highest leverage, smallest change)**:
- **Re-rank stage 4 by LightGBM gain, not MI**
  - After determining best_k in stage 3, refit one LightGBM at best_k on full subsample
  - Take feature_importances_, select top-best_k by gain
  - Cost: 96 extra fits (~1 sec each)
  - Benefit: Catches predispatch RRP and other conditional-importance features

**Priority 2 (essential for spike-aware system)**:
- **Add spike-target ranking pass + union**
  - Run parallel ranking against y_spike = (price > 300).astype(int) using mutual_info_classif
  - Take union of (MI-top-K continuous) ∪ (MI-top-K spike) before stage 2
  - Cost: one extra ranking pass (~10 sec)
  - Benefit: Captures wind forecast, thermal outage triggers that have low MI but high spike correlation

**Priority 3 (covers MI's blind spot)**:
- **Force-keep calendar + predispatch features**
  - Add MUST_KEEP_FEATURES env list in 0_Config/0_variables.ipynb
  - Bypass dedup for them; force-include in stage 4 regardless of best_k
  - Cost: negligible
  - Benefit: Guarantees hour/dow/holiday/predispatch are always available

**Priority 4 (aligns selection with downstream objective)**:
- **Switch CV loss from MAE to RMSE or quantile loss (q90/q95)**
  - Change metric in stage 3 from "l1" to "rmse" or quantile
  - Cost: negligible (same CV loop)
  - Benefit: Selects features that minimize spike error, not median error

**Priority 5 (robustness)**:
- **Log-space k-grid + multi-seed CV**
  - Replace linear [1, 10] with log-spaced: np.unique(np.round(np.logspace(0, log10(n), 10)).astype(int))
  - Run stage 3 with 2–3 different RNG seeds; average MAE before picking best_k
  - Cost: 2–3× more CV fits (~30–50 sec)
  - Benefit: Better resolution in elbow region; noisy best_k averaging

**Priority 6 (data relevance)**:
- **Extend CV window past 2022 or split pre/post-regime**
  - Current CV subsample: 2019–2023 (misses post-2022 gas crisis, coal retirements, battery participation)
  - Option A: Extend to 2026
  - Option B: Run separate pipelines for pre-2023 and post-2023, union features
  - Cost: negligible (just parameter change)
  - Benefit: Features selected for current NEM regime (2024+)

---

## 8. MODEL ARCHITECTURE — SPIKE-AWARE ENSEMBLE

### **Overview**

Per forecast horizon (h ∈ 1..96), the model is a 6-component ensemble designed to handle extreme price events:

```
┌─────────────────────────────────────────────────────────────┐
│ Input: Feature vector X_h (2000+ floats)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼─────┐ ┌──▼─────┐ ┌──▼─────────┐
    │ Base L1  │ │ Base L2 │ │ Spike      │ (if n_spikes ≥ 20)
    │ (MAE)    │ │ (MSE)   │ │ Chain      │
    │ p97-clip │ │ uncap   │ │ 3 models   │
    └────┬─────┘ └──┬─────┘ └──┬─────────┘
         │          │          │
         └──────────┼──────────┘
                    │
            ┌───────▼────────┐
            │ Spike Policy   │ (soft blend / gated uplift / hard gate)
            │ Tune on valid  │
            └───────┬────────┘
                    │
         ┌──────────┼──────────┐
         │          │          │
    ┌────▼─────┐ ┌──▼─────┐ ┌──▼─────────┐
    │ Dip      │ │ Naive  │ │ Isotonic   │
    │ Chain    │ │ Blend  │ │ Calibrate  │
    │ (if n_dips≥20) │        │ (if n≥500)  │
    └────┬─────┘ └──┬─────┘ └──┬─────────┘
         │          │          │
         └──────────┼──────────┘
                    │
                ┌───▼────┐
                │ Output │ ŷ_h
                └────────┘
```

### **Component Details**

#### **Base Models (L1 + L2)**

**Base L1**:
- **Objective**: regression_l1 (MAE loss)
- **Target**: arcsinh(min(y, p97)) / 100
  - Clipping at 97th percentile removes extreme spike noise
  - Allows base model to focus on main distribution (median-optimal)
- **Sample weight**: recency × where(y > $150, 3.0, 1.0)
  - Mild 3× upweight on spikes (doesn't dominate loss)
- **Hyperparameters**: 1400 estimators, lr=0.025, num_leaves=95

**Base L2**:
- **Objective**: regression (MSE loss)
- **Target**: arcsinh(y) / 100 (uncapped, full range)
  - Heavy-tailed NEM prices → E[Y] >> median(Y)
  - L2 captures upside risk inherently
- **Sample weight**: same as L1
- **Hyperparameters**: 1200 estimators, lr=0.025, num_leaves=95

**Blending**:
- On validation set, tune α ∈ [0, 0.45] (10 values)
- Final base prediction: (1 − α) × L1 + α × L2
- α ≈ 0.3–0.4 typical (empirically, Lago et al. 2021 demonstrated ~20–40% L2 reduces both MAE and RMSE)

**Rationale**: L1 is robust; L2 is mean-optimal. Blending gets best of both worlds.

#### **Spike Chain** (if n_spikes ≥ 20)

**Spike Classifier**:
- **Objective**: binary logistic
- **Target**: (y > $150).astype(int)
- **Sample weight**: recency
- **scale_pos_weight**: 7.0 (handles class imbalance; ~12% positive class)
- **Hyperparameters**: 500 estimators, lr=0.025, num_leaves=63
- **Output**: P(spike) ∈ [0, 1] per sample

**Spike Regressor**:
- **Objective**: regression_l1
- **Target**: arcsinh(y) / 100 (full range)
- **Sample weight**: recency × where(spike_label > 0, 10.0, 1.0)
  - 10× upweight on spike rows (spike specialist)
- **Hyperparameters**: 500 estimators, lr=0.025, num_leaves=63
- **Output**: ŷ_spike ∈ ℝ (can be >p97)

**Spike Quantile (P90)**:
- **Objective**: quantile (α=0.90)
- **Target**: arcsinh(y) / 100
- **Sample weight**: same as spike regressor
- **Purpose**: Conservative ceiling estimate; prevents under-prediction in spike tail
- **Output**: ŷ_q90 (90th percentile estimate)

**Spike Policy** (3 kinds, tuned per horizon):
1. **Soft blend**: ŷ_h = (1 − P_spike) × ŷ_base + P_spike × ŷ_spike
   - Smooth interpolation based on spike probability
   
2. **Gated uplift**: ŷ_h = ŷ_base + gate(P_spike) × uplift_factor × (ŷ_spike − ŷ_base)
   - Gate: piecewise linear thresholding on P_spike
   - Only apply spike regressor when confidence high
   
3. **Hard gate**: if P_spike > threshold, use ŷ_spike; else use ŷ_base
   - Binary switch based on spike probability

**Validation tuning**:
- Per policy kind, grid-search: spike_threshold ∈ [0.03, 0.50], spike_power ∈ [0.5, 4.0], etc.
- Select kind + parameters that minimize: 2.2 × spike_MAE + 1.3 × dip_MAE + 1.0 × normal_MAE + 0.25 × total_MAE
- (Weights reflect business priorities: spikes >> dips >> normal prices)

#### **Dip Chain** (if n_dips ≥ 20)

Mirrors spike design but for y < $0 (negative prices):
- **Dip Classifier**: binary, scale_pos_weight=6
- **Dip Regressor**: regression_l1, 7× upweight
- **Dip Policy**: 3 kinds (soft blend / downside gate / hard gate)
- **Validation tuning**: Similar grid search, but for downside tail

**Rationale**: Negative prices (oversupply events) are increasingly common in NEM; need specialized model

#### **Naive Baseline Blending**

After spike/dip policies, blend model prediction with lag-naive baseline:
- **Naive baseline**: lag-336 (same day of week)
- **Grid search**: α ∈ [0, 1] (41 values)
- **Final**: ŷ_final = (1 − α) × ŷ_policy + α × ŷ_naive_lag_336

**Rationale**: Strong weekly seasonality in NEM; lag-naive is hard to beat at long horizons. Blending improves stability.

#### **Isotonic Calibration**

On validation set (if ≥500 samples):
- Fit IsotonicRegression(y_validate, ŷ_policy_validate)
- On test set, apply: ŷ_calibrated = isotonic_fn(ŷ_policy_test)

**Rationale**: Model predictions often biased; monotonic transformation improves calibration

### **Per-Horizon Feature Matrix**

For horizon h, inputs to all 6 models:
```
X_h = [
  base_features (1500–2000 cols),           # from 2_Features_build
  target_time_feats (8 cols):               # hour/dow sin-cos + regime flags
    hour_sin_h, hour_cos_h, dow_sin_h, dow_cos_h,
    is_peak_h, is_shoulder_h, is_off_peak_h, is_holiday_h,
  cross_features (4 cols):                  # doy_sin/cos, sa_spread, region_spike_score
    doy_sin, doy_cos, sa_spread_live, region_spike_score,
  predispatch_aligned (8 cols):             # predispatch RRPs for h-1, h, h+1, h+2, h-2, h-4, h-8, h-12
    predispatch_rrp_h_{h-2}, ..., predispatch_rrp_h_{h+2},
  (total: ~1530–2020 cols per horizon)
]
```

**Key insight**: Horizon-specific features allow model to adapt to different forecast lengths (early vs late horizons need different signals)

---

## 9. DEPENDENCIES & ENVIRONMENT

### **Main Environment (Python 3.13)**

File: `support/requirements-main.txt`

| Package | Version | Purpose |
|---------|---------|---------|
| nemosis | ≥3.8.0 | AEMO NEM data API |
| pandas | ≥2.0.0 | DataFrames |
| numpy | ≥1.24.0 | Numerical computing |
| pyarrow | ≥14.0.0 | Fast parquet I/O |
| s3fs | ≥2023.6.0 | S3 transparent I/O |
| scikit-learn | ≥1.3.0 | IsotonicRegression, metrics |
| lightgbm | ≥4.3.0 | **Core model library** |
| holidays | ≥0.46 | Australian public holidays |
| matplotlib | ≥3.7.0 | Plotting |
| seaborn | ≥0.13.0 | Statistical plots |
| openpyxl | ≥3.1.0 | Excel I/O |
| tqdm | ≥4.65.0 | Progress bars |
| joblib | ≥1.3.0 | Model serialization, parallelism |
| ipykernel | ≥6.0.0 | Jupyter kernel |
| nbformat | ≥5.10.0 | Notebook format |

**Installation**:
```bash
python3.13 -m venv ~/venv313
source ~/venv313/bin/activate
pip install -r requirements-main.txt
```

### **Subprocess Environment (Python 3.11)**

File: `support/requirements-subprocess.txt`

| Package | Version | Purpose |
|---------|---------|---------|
| nemseer | ≥1.0.0 | AEMO PASA data API (requires 3.11) |
| pandas | ≥2.0.0 | DataFrames |

**Installation**:
```bash
python3.11 -m venv ~/venv311
source ~/venv311/bin/activate
pip install -r requirements-subprocess.txt
```

**Why two envs?**
- nemseer (PASA data) requires Python 3.11
- Main pipeline uses Python 3.13
- Notebooks call nemseer as a subprocess, capturing results

---

## 10. KEY FUNCTIONS & CLASSES

### **Feature Builders** (4_Model/1_feature_creation.ipynb)

**`compute_cross_feats(df: pd.DataFrame) -> pd.DataFrame`**
- **Input**: DataFrame with 30-min index + price columns
- **Output**: 4-column DataFrame with cross-features
- **Columns**:
  - `doy_sin`: sin(2π × day_of_year / 365.25)
  - `doy_cos`: cos(2π × day_of_year / 365.25)
  - `sa_spread_live`: arcsinh((sa_price − nsw_price) / 100).clip(−10, 10)
  - `region_spike_score`: count of QLD/VIC/SA with price > $150
- **Scale factor**: PRICE_TRANSFORM_SCALE = 100

**`_compute_target_time_feats(idx_dt: DatetimeIndex, horizon: int) -> np.ndarray`**
- **Input**: DatetimeIndex, forecast horizon (1–96)
- **Output**: (n_samples, 8) array of float32
- **Features**:
  - hour_sin, hour_cos, dow_sin, dow_cos (periodicity shifted by horizon)
  - is_peak, is_shoulder, is_off_peak, is_holiday
- **Purpose**: Horizon-specific calendar features (capture daily seasonality at forecast time)

**`_compute_aligned_pd_feats(df: pd.DataFrame, horizon: int) -> np.ndarray`**
- **Input**: Feature DataFrame, horizon
- **Output**: (n_samples, 8) array of predispatch RRPs aligned to h
- **Purpose**: Predispatch forecasts for h-2, h-1, h, h+1, h+2, h+8, h+12, etc.

**Price transforms**:
- `_to_asinh(y: np.ndarray) -> np.ndarray`: arcsinh(y / 100)
- `_from_asinh(y: np.ndarray) -> np.ndarray`: sinh(y) * 100
- **Rationale**: arcsinh handles both positive and negative prices; saturates at extremes

### **Model Training** (4_Model/3_models.ipynb)

**`train_seq2seq() -> (base_models, base_l2_models, spike_clfs, spike_regs, spike_qregs, dip_clfs, dip_regs)`**
- **Input**: Loaded X_train, X_validate, y_train, y_validate (from Data/)
- **Output**: Lists of 7 models × 96 horizons
- **Parallelization**: joblib.Parallel(n_jobs=-1) over horizons
- **Per-horizon**:
  1. Compute target-time features + recency weights
  2. Fit base L1 (clipped target)
  3. Fit base L2 (uncapped target)
  4. If n_spikes ≥ 20: fit spike classifier, regressor, quantile
  5. If n_dips ≥ 20: fit dip classifier, regressor
  6. Return tuple of models
- **Early stopping**: 75 rounds (base), 50 rounds (spike/dip)

### **Utility Functions** (support/paths.py)

**`resolve(repo_root_relative: str) -> str`**
- **Input**: Relative path from repo root
- **Output**: Absolute local path OR S3 URI
- **Logic**:
  ```python
  if USE_S3:  # env var
    return f"s3://forecasting-nem-dd/{repo_root_relative}"
  else:
    return str(Path(__file__).parent / repo_root_relative)
  ```
- **Usage**: Transparent switching between local & S3 I/O via single environment variable

### **Feature Selection** (3_Features_select/)

**`mutual_info_regression(X, y, discrete_features, random_state)`** (scikit-learn)
- Per-horizon: compute MI(feature; target) on 200k subsample
- Output: 1D array of MI scores per feature

**`LGBMRegressor(...).fit(X, y, sample_weight, eval_set, callbacks)`** (LightGBM)
- Per-horizon in stage 3: grid-search k ∈ [1, 10 linear], pick best_k by MAE
- Feature importance: `model.feature_importances_` (gain ranking)

### **Data I/O**

**`pd.read_parquet(path) -> DataFrame`** (pandas + pyarrow)
- Read from local or S3 via `resolve(path)`
- Fast columnar format

**`joblib.dump(model, path, compress='zlib')`** (joblib)
- Serialize 672 LightGBM models + metadata to single file (~500 MB)
- zlib-9 compression minimizes disk footprint

---

## 11. HARDCODED VALUES & CONSTANTS

### **Price & Regime Thresholds**
- **SPIKE_THRESHOLD = $150/MWh** (defines spike event)
  - Australian policy: ≥ $300 is unscheduled outage; $150 is conservative spike threshold
- **_DIP_THRESHOLD = $0/MWh** (defines dip event)
  - Negative prices occur during oversupply (e.g., high solar)
- **PRICE_TRANSFORM_SCALE = 100.0** (arcsinh divisor)
  - Scales prices to [−10, +10] range for numerical stability
- **BASE_CLIP_PERCENTILE = 97.0** (p97 threshold for base model target)
  - Raised 95 → 97 to include more spike distribution
  - Removes ~3% of extremes, reduces spike noise in base loss

### **Sample Weighting**
- **_SPIKE_UPWEIGHT = 10.0** (spike regressor weight multiplier)
  - Raised 5 → 10; gives spikes 10× influence on spike regressor loss
- **_DIP_UPWEIGHT = 7.0** (dip regressor weight multiplier)
  - Raised 4 → 7; gives dips 7× influence
- **Recency weight formula**: exp(linspace(0, 1.5, n_train))
  - Newest rows ~exp(1.5) ≈ 4.5× oldest rows
  - Based on Uniejewski et al. 2019 EPF study

### **Early Stopping**
- **EARLY_STOPPING_ROUNDS = 75** (base models)
  - Reduced 150 → 75 for faster convergence
- **SPIKE_ES_ROUNDS = 50** (spike/dip models)
  - More aggressive stopping on smaller sample sizes

### **Min Sample Thresholds**
- **_MIN_SPIKE_TRAIN = 20** (minimum spike rows to train spike chain)
  - If fewer, spike/dip models skipped for that horizon
- **Isotonic calibration**: ≥500 samples on validation set
  - Else isotonic step skipped

### **Forecast Parameters**
- **FORECAST_GAP = 1** (default: 30 min ahead)
  - First prediction horizon is 30 min in future
- **FORECAST_HORIZON = 96** (default: 48 hours in 30-min intervals)
  - 96 × 30 min = 2,880 min = 48 hours
- **OUTPUT_RESOLUTION = 30** (minutes)
  - All data aligned to 30-min grid

### **Data Windows**
- **DATA_START_DATE = 2018/01/01**, **DATA_END_DATE = 2026/01/01**
  - Full historical data range (archived in config.py)
- **FEATURE_DATASET_START = 2019/01/01**, **FEATURE_DATASET_END = 2026/01/01**
  - Feature build window (exclude 2018 for warmup)
- **FEATURE_SELECTION_CV_SUBSAMPLE_START = 2019/01/01**, **...END = 2023/01/01**
  - Feature ranking/selection on 200k subsample (pre-2022 regime)
  - **Known limitation**: Misses post-2022 gas crisis; candidate for extension

### **Temporal Split**
- **Train**: all except last 18 months
- **Validate**: 6 months before test
- **Test**: most recent 12 months
- **Formula**: if today is 2026/01, test = [2024/01, 2026/01), validate = [2023/07, 2024/01)

### **Model Hyperparameters** (see parameters/lgbm_params.py for full list)
- Base L1: num_leaves=95, n_estimators=1400, reg_alpha=0.05, reg_lambda=0.20
- Spike classifier: scale_pos_weight=7.0, num_leaves=63, n_estimators=500
- All models: num_threads=1, n_jobs=1 (parallelism via outer joblib loop)

---

## 12. ISSUES, LIMITATIONS & TECHNICAL DEBT

### **Feature Selection Pipeline**

**Issue 1: Stage 4 ranks by MI, not LightGBM gain**
- **Problem**: Predispatch RRPs may have low marginal MI but high conditional importance (classic feature interaction case)
- **Impact**: May not appear in final feature set despite CV proving their value
- **Recommended fix** (Priority 1): Re-rank stage 4 by LightGBM gain from stage 3 CV
- **Status**: Documented in Analysis.md

**Issue 2: Greedy dedup can drop the wrong sibling**
- **Problem**: Among {nsw_price_rmean_2016, qld_price_rmean_2016, vic_price_rmean_2016}, greedy keeps highest MI; drops others. But for spike regime, QLD may have unique information
- **Recommended fix** (Priority 2b): Refit LightGBM with both vs sibling on subsample; reinstate if MAE improves >1%
- **Status**: Documented in Analysis.md

**Issue 3: No spike-aware ranking**
- **Problem**: MI ranking uses continuous target (price). ~1% of intervals are >$300 (extreme spikes). MI smooths spikes; wind_forecast_error may have MI≈0.05 but be the only spike trigger
- **Recommended fix** (Priority 2): Add parallel ranking against y_spike=(price>300), take union of top-K
- **Status**: Not implemented; high priority

**Issue 4: Calendar/predispatch features not forced**
- **Problem**: hour, day-of-week, holiday, predispatch RRP are causally essential but may have low MI numerically
- **Recommended fix** (Priority 2c): Add MUST_KEEP_FEATURES env list; bypass dedup + force inclusion
- **Status**: Not implemented

**Issue 5: CV loss is MAE (wrong for spike-aware system)**
- **Problem**: MAE rewards median predictions; penalizes spike accuracy. Downstream model is spike-aware ensemble; should select features for spike, not median
- **Recommended fix** (Priority 3): Change CV metric from l1 to rmse or quantile loss (q90/q95)
- **Status**: Not implemented

**Issue 6: Single subsample + 3 folds → noisy best_k**
- **Problem**: K-grid winner often flips between adjacent values by <0.5% MAE (well within noise floor)
- **Recommended fix** (Priority 3b): Repeat CV with 2–3 RNG seeds; average MAE before picking best_k
- **Status**: Not implemented

**Issue 7: CV window 2019–2022 misses post-2022 regime change**
- **Problem**: NEM prices changed post-2022 (gas crisis, coal retirements, battery participation)
- **Recommended fix** (Priority 4): Extend FEATURE_SELECTION_CV_SUBSAMPLE_END to 2026, or split pre/post-2022
- **Status**: Noted; not implemented

**Issue 8: Linear k-grid has poor resolution**
- **Problem**: With N_K_VALUES=10 linear, on 800 features k ∈ {1, 89, 178, ...}. MAE curve elbows below 50; all resolution in first gap
- **Recommended fix** (Priority 3c): Use log-spaced grid: np.unique(np.round(np.logspace(0, log10(n), 10)))
- **Status**: Not implemented

**Issue 9: Per-horizon selection, no joint smoothing**
- **Problem**: Best-k oscillates across horizons (e.g., k=80 at h12, k=20 at h13) → inconsistent feature sets
- **Recommended fix** (Priority 4b): Smooth best_k across horizons or use union within 4-hour bands
- **Status**: Not implemented

---

### **Model Training & Validation**

**Issue 10: Validation-tuning search space under-explored**
- **Problem**: 3 spike policy kinds × grid of 4–5 hyperparams each × 3 dip policies × naive blend α = 100s+ grid evaluations per horizon
- **Current**: Grids commented out in search_grids.py; likely not run
- **Recommended fix**: Implement grid search or Optuna sweep per horizon on validation set
- **Status**: Not implemented; potential bottleneck

**Issue 11: Isotonic calibration skipped if <500 validation samples**
- **Problem**: For early horizons or sparse regions, may not have enough data
- **Recommended fix**: Use smaller sample thresholds or alternative calibration (e.g., Platt scaling)
- **Status**: Acceptable; low-priority

---

### **Data & Infrastructure**

**Issue 12: Predispatch RRPs may have NaN gaps**
- **Problem**: AEMO predispatch forecasts not always available (e.g., system outages, API failures)
- **Current handling**: Implicit fillna(0) or forward-fill; unclear
- **Recommended fix**: Explicit imputation strategy + missing-value flag
- **Status**: Not clearly documented

**Issue 13: S3 sync script is manual**
- **Problem**: commands.md lists `aws s3 sync` commands; not automated
- **Recommended fix**: Script-based sync with CI/CD integration
- **Status**: Manual; acceptable for R&D

---

### **Code Organization**

**Issue 14: Two venv requirement**
- **Problem**: Nemseer (PASA data) requires Python 3.11; main pipeline requires 3.13
- **Current handling**: Documented in AGENTS.md; subprocess calls
- **Recommended fix**: Containerize (Docker) to avoid version conflict
- **Status**: Works but manual; low priority

**Issue 15: Notebooks + .py modules duplication**
- **Problem**: Feature builders defined in both 1_feature_creation.ipynb and 99_old/feature_engineering.py
- **Recommended fix**: Consolidate into single `feature_builders.py` module
- **Status**: 99_old code is reference; acceptable

**Issue 16: Configuration scattered**
- **Problem**: Constants in 4_Model/0_vars.ipynb, parameters in parameters/, env vars in 0_Config/0_variables.ipynb
- **Recommended fix**: Single config.py or dataclass
- **Status**: Manageable; low priority

---

### **Performance & Scalability**

**Issue 17: Per-horizon training parallelized; inner parallelism disabled**
- **Problem**: joblib.Parallel(n_jobs=-1) over 96 horizons; each LightGBM fit has num_threads=1
- **Current**: All cores used but serialized within-model
- **Recommended fix**: Tune thread pool size vs horizon parallelism for optimal throughput
- **Status**: Acceptable; likely not a bottleneck

**Issue 18: Large feature matrix stored in memory**
- **Problem**: Feature_data/1_dispatch_price.parquet with 200–400+ columns × 50k+ rows ≈ 500 MB
- **Current handling**: Loaded via pandas; fits in RAM
- **Recommended fix**: Consider parquet chunking for very large deployments
- **Status**: Not a concern for current scale

---

### **Business Logic**

**Issue 19: Spike/dip weights hardcoded**
- **Problem**: _SPIKE_UPWEIGHT=10, _DIP_UPWEIGHT=7 are fixed; no sensitivity analysis
- **Recommended fix**: Grid search over weights; tune on holdout business metric (e.g., spike MAE)
- **Status**: Could improve; low priority

**Issue 20: Spike policy score formula is ad-hoc**
- **Problem**: 2.2 × spike_MAE + 1.3 × dip_MAE + 1.0 × normal_MAE + 0.25 × total_MAE weights are hardcoded
- **Recommended fix**: Calibrate weights to stakeholder priorities (business cost of forecast error)
- **Status**: Acceptable; could be parameterized

---

## SUMMARY TABLE

| Category | Item | Status | Priority |
|----------|------|--------|----------|
| **Feature Selection** | Re-rank by LightGBM gain | Not implemented | P1 |
| | Spike-aware ranking pass | Not implemented | P2 |
| | Force-keep calendar/predispatch | Not implemented | P2c |
| | Change CV loss to RMSE/quantile | Not implemented | P3 |
| | Multi-seed CV averaging | Not implemented | P3b |
| | Log-spaced k-grid | Not implemented | P3c |
| | Extend CV window past 2022 | Not implemented | P4 |
| | Smooth best_k across horizons | Not implemented | P4b |
| **Model Training** | Implement validation-tuning grid search | Not implemented | Medium |
| | Improve isotonic calibration for <500 samples | Not implemented | Low |
| **Data** | Explicit predispatch NaN handling | Not documented | Medium |
| | Automate S3 sync | Not automated | Low |
| **Infrastructure** | Containerize (Docker) | Not done | Low |
| **Code** | Consolidate feature builders | Not done | Low |
| | Unify configuration | Not done | Low |
| **Performance** | Optimize thread pool tuning | Not done | Low |

---

## CONCLUSION

The **Forecasting repository is a sophisticated spike-aware ensemble electricity price forecaster** with:

1. **Strong foundation**: 4-stage feature selection pipeline, 6-component per-horizon model, validation-tuned policies
2. **Clear gaps**: Feature selection under-ranks conditional-importance features; no spike-aware ranking; CV loss misaligned with objective
3. **Known technical debt**: Documented in Analysis.md; prioritized fixes available
4. **Production-ready infrastructure**: S3 integration, dual Python environments, serialized models, Excel reporting

**Highest-impact improvements** (in priority order):
1. Re-rank stage 4 by LightGBM gain (catches predispatch RRP)
2. Add spike-aware ranking pass + union
3. Force-keep calendar/predispatch features
4. Switch CV loss to RMSE/quantile
5. Extend CV window past 2022 for regime relevance

import pandas as pd
from pathlib import Path
CWD = Path(__file__).resolve().parent 


# Shared
PIPELINE_START_DATE = pd.to_datetime("2019/01/01")
PIPELINE_END_DATE = pd.to_datetime("2026/01/01")
FEATURE_SELECTION_START_DATE = pd.to_datetime("2019/01/01")
FEATURE_SELECTION_END_DATE = pd.to_datetime("2024/01/01")
FEATURE_SELECTION_SUBSAMPLE_AMOUNT = 400000

# Targets
TARGET_DATASET_NAME = "1_dispatch_price.parquet"
TARGET_DATASET_PATH = CWD/"1_Dataset"/"Processed_data"/TARGET_DATASET_NAME
ALL_TARGET_COLS = "nsw_price, qld_price, sa_price, vic_price"
# Single switch that retargets the whole pipeline to a different NEM region.
# Change to "qld", "vic" or "sa" and every feature notebook reuses the same code.
TARGET_REGION = "nsw"
SELECTED_TARGET_COLUMN_PREFIX = TARGET_REGION.upper()
SELECTED_TARGET_COLUMN_POSTFIX = "price"
SELECTED_TARGET_COLUMN_NAME = f"{TARGET_REGION}_{SELECTED_TARGET_COLUMN_POSTFIX}"
AGG_TARGET_DATASET_PATH = CWD/"3_Build_targets"/"Target_data"/"Targets.parquet"
HORIZON_LENGTH_IN_HOURS = 48
HORIZON_GRANULARITY_IN_MINUTES = 30
HORIZON_COUNT = int(HORIZON_LENGTH_IN_HOURS * (60 // HORIZON_GRANULARITY_IN_MINUTES))
FEATURE_GRANULARITY_IN_MINUTES = 5
HORIZON_LENGTH_IN_HOURS
# Features
# Combined matrix written by 2_Features_build/0_all.ipynb (all per-source
# Feature_data/*.parquet concatenated on the dispatch-price index).
FEATURE_DATASET_NAME = "0_all_features.parquet"
FEATURES_DATASET_PATH = CWD/"2_Features_build"/"Feature_data"/FEATURE_DATASET_NAME
FEATURES_DATASET_FOR_SELECTION_PATH = CWD/"4_Features_select"/"Selected_features"/"FEATURES_DATASET_FOR_SELECTION.parquet"

# Feature ranking
FEATURES_RANKED_ORDERED = CWD/"4_Features_select"/"Selected_features"/"FEATURES_RANKED_ORDERED.parquet"

# Get unique features
FEATURES_UNIQUE_RANKED_LIMIT = 100000000
FEATURES_RANKED_ORDERED_UNIQUE_PATH = CWD/"4_Features_select"/"Selected_features"/"FEATURES_RANKED_ORDERED_UNIQUE.parquet"
FEATURES_UNIQUE_DATA_PATH = CWD/"4_Features_select"/"Selected_features"/"FEATURES_UNIQUE_DATA.parquet"

# Get optimal number of features
FEATURES_OPTIMAL_AMOUNT_PATH =  CWD/"4_Features_select"/"Selected_features"/"FEATURES_OPTIMAL_AMOUNT.parquet"


# Training
TRAIN_START = pd.to_datetime("2019/01/01")   
VALID_START = pd.to_datetime("2024/07/01") # 6 months before TEST_START
TEST_START = pd.to_datetime("2025/01/01") # 12 months before FEATURE_DATASET_END
TRAINED_MODELS_PATH =  CWD/"5_Model"/"Data"/"3_trained_models"

# Post training
PRICE_TRANSFORM_SCALE = 100.0
SPIKE_THRESHOLD = 150.0
DIP_THRESHOLD = 0.0
FULL_RANGE_ALPHA_PATH = CWD/"5_Model/Data/4_combine_models/full_range_alphas.csv"
FULL_RANGE_BLEND_PATH = CWD/"5_Model/Data/4_combine_models/full_range_blend.parquet"
SPIKE_MODELS_BLEND_PARAMS_PATH = CWD/"5_Model/Data/4_combine_models/spike_models_blend_params.csv"
DIP_MODELS_BLEND_PARAMS_PATH = CWD/"5_Model/Data/4_combine_models/dip_models_blend_params.csv"
ALL_MODELS_BEST_BLEND_PATH = CWD/"5_Model/Data/4_combine_models/all_models_best_blend.joblib"
FINAL_PARAMS_PATH = CWD/"5_Model/Data/4_combine_models/final_params.joblib"

# Evaluation
ACTUAL_VS_PREDICTED_TEST_SET  = CWD/"5_Model/Data/5_model_results/actual_vs_predicted_test_set.parquet"



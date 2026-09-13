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
SELECTED_FEATURES_DIR = CWD/"4_Features_select"/"Selected_features"
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
HOLISTIC_MODEL_DIR = CWD/"5_Model/Data/4_combine_models"
VALIDATION_COMPONENT_PREDICTIONS_PATH = HOLISTIC_MODEL_DIR/"1_validation_component_predictions.parquet"
VALIDATION_META_FEATURES_PATH = HOLISTIC_MODEL_DIR/"2_validation_meta_features.parquet"
RESIDUAL_STACKERS_DIR = HOLISTIC_MODEL_DIR/"3_residual_stackers"
RESIDUAL_STACKER_DIAGNOSTICS_PATH = HOLISTIC_MODEL_DIR/"3_residual_stacker_diagnostics.csv"
VALIDATION_CENTRAL_PREDICTIONS_PATH = HOLISTIC_MODEL_DIR/"3_validation_predictions.parquet"
FINAL_PARAMS_PATH = HOLISTIC_MODEL_DIR/"final_params.joblib"

# Evaluation
MODEL_RESULTS_DIR = CWD/"5_Model/Data/5_model_results"
AEMO_PREDISPATCH_PATH = CWD/"1_Dataset/Processed_data/6_1_predispatch_price.parquet"
ACTUAL_VS_PREDICTED_TEST_SET = MODEL_RESULTS_DIR/"actual_vs_predicted_test_set.parquet"

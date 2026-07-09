
import pandas as pd
from pathlib import Path
CWD = Path(__file__).resolve().parent 


# Shared
PIPELINE_START_DATE = pd.to_datetime("2019/01/01")
PIPELINE_END_DATE = pd.to_datetime("2026/01/01")
FEATURE_SELECTION_START_DATE = pd.to_datetime("2019/01/01")
FEATURE_SELECTION_END_DATE = pd.to_datetime("2024/01/01")
FEATURE_SELECTION_SUBSAMPLE_AMOUNT = 200

# Targets
TARGET_DATASET_NAME = "1_dispatch_price.parquet"
TARGET_DATASET_PATH = CWD/"1_Dataset"/"Processed_data"/TARGET_DATASET_NAME
ALL_TARGET_COLS = "nsw_price, qld_price, sa_price, vic_price"
SELECTED_TARGET_COLUMN_PREFIX = "NSW"
SELECTED_TARGET_COLUMN_POSTFIX = "price"
SELECTED_TARGET_COLUMN_NAME = "nsw_price"
AGG_TARGET_DATASET_PATH = CWD/"3_Build_targets"/"Target_data"/"Targets.parquet"
HORIZON_LENGTH_IN_HOURS = 48
HORIZON_GRANULARITY_IN_MINUTES = 30
HORIZON_COUNT = HORIZON_LENGTH_IN_HOURS * (60 // HORIZON_GRANULARITY_IN_MINUTES)
FEATURE_GRANULARITY_IN_MINUTES = 5

# Features
FEATURE_DATASET_NAME = "1_dispatch_price.parquet"
FEATURES_DATASET_PATH = CWD/"2_Features_build"/"Feature_data"/FEATURE_DATASET_NAME
FEATURES_DATASET_FOR_SELECTION_PATH = CWD/"4_Features_select"/"Selected_features"/"FEATURES_DATASET_FOR_SELECTION.parquet"

# Feature ranking
FEATURES_RANKED_ORDERED = CWD/"4_Features_select"/"Selected_features"/"FEATURES_RANKED_ORDERED.parquet"

# Get unique features
FEATURES_UNIQUE_RANKED_LIMIT = 100
FEATURES_RANKED_ORDERED_UNIQUE_PATH = CWD/"4_Features_select"/"Selected_features"/"FEATURES_RANKED_ORDERED_UNIQUE.parquet"
FEATURES_UNIQUE_DATA_PATH = CWD/"4_Features_select"/"Selected_features"/"FEATURES_UNIQUE_DATA.parquet"

# Get optimal number of features
FEATURES_OPTIMAL_AMOUNT_PATH =  CWD/"4_Features_select"/"Selected_features"/"FEATURES_OPTIMAL_AMOUNT.parquet"

# Split into (train / valid / test)
TRAIN_START = pd.to_datetime("2019/01/01")   
VALID_START = pd.to_datetime("2024/07/01") # 6 months before TEST_START
TEST_START = pd.to_datetime("2025/01/01") # 12 months before FEATURE_DATASET_END

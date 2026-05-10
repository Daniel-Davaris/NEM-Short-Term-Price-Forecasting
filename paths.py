"""
paths.py — central path resolver for local and S3 environments.

Usage in notebooks:
    import sys; sys.path.insert(0, "..")   # or "." from repo root
    from paths import resolve

    df = pd.read_parquet(resolve("1_Dataset/Processed_data/1_dispatch_price.parquet"))
    df.to_parquet(resolve("2_Features build/Feature_data/1_dispatch_price.parquet"))

Set USE_S3=1 in the environment to route all I/O through S3 (EC2 with IAM role).
Leave unset (or USE_S3=0) to use local absolute paths (default on local machine).
"""

import os
from pathlib import Path

S3_BUCKET = "forecasting-nem-dd"
USE_S3 = os.environ.get("USE_S3", "0") == "1"

_REPO_ROOT = Path(__file__).parent

# Local directory names with spaces → S3 key names with underscores
_S3_KEY_MAP = [
    ("2_Features build/",   "2_Features_build/"),
    ("3_Targets build/",    "3_Targets_build/"),
    ("4_Features select/",  "4_Features_select/"),
]


def resolve(repo_root_relative: str) -> str:
    """Return an absolute local path or S3 URI for a repo-root-relative path.

    On local (USE_S3=0): returns the absolute local path.
    On EC2   (USE_S3=1): returns s3://forecasting-nem-dd/<key>.
    """
    if USE_S3:
        key = repo_root_relative
        for local, remote in _S3_KEY_MAP:
            key = key.replace(local, remote)
        return f"s3://{S3_BUCKET}/{key}"
    return str(_REPO_ROOT / repo_root_relative)

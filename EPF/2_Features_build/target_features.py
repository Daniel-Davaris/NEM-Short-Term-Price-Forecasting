"""Horizon-aware TARGET-time calendar features.

These are deliberately NOT ordinary observed features. Every other feature source
describes conditions known AT the forecast origin; a target-time feature instead
describes the calendar context of the time each per-horizon model is *forecasting*
(origin + h * HORIZON_GRANULARITY_IN_MINUTES).

Two properties make them special and dictate how they must be wired in:

1. Leakage-free: the calendar is known arbitrarily far in advance, so using the
   target time's hour / day-of-week introduces no future information.

2. They must be injected PER HORIZON at model-build time, NOT passed through the
   shared 0_all -> ranking -> de-duplication -> selection pipeline. For horizon h
   only the h-th block is the correct descriptor, yet the 96 blocks are near
   identical phase-shifted copies of the same periodic signal (e.g. adjacent
   sin-of-hour blocks correlate ~0.99). The global |corr| > 0.95 de-duplication
   in 4_Features_select/2_remove_duplicate_features would therefore collapse all
   96 blocks down to a handful of survivors tied to a single, wrong horizon,
   destroying the per-horizon alignment. Injecting block h directly into
   horizon h's model keeps the mapping exact and guaranteed.

Location: EPF/2_Features_build/target_features.py. Consumers add that folder to
sys.path and `import target_features`.

Consumers: 5_Model/2_train_models (training), 3_build_holistic_model (blend/spike
tuning) and 4_evaluate_model (test prediction) all append these columns to each
horizon's feature block, always in TARGET_TIME_FEATURE_NAMES order so the column
order matches between fit and predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# This module lives in EPF/2_Features_build/ (a digit-prefixed folder that can't
# be a Python sub-package), so consumers import it by adding that folder to
# sys.path: `import target_features`. It in turn imports EPF.variables via the
# repo root, which every consumer already adds to sys.path first.
from EPF import variables

# Fixed column order. Appended to the RIGHT of the per-horizon selected features
# at both fit and predict time, so training (numpy, positional) and evaluation
# (DataFrame) stay consistent.
TARGET_TIME_FEATURE_NAMES = [
    "tgt_sin_hour",
    "tgt_cos_hour",
    "tgt_sin_dow",
    "tgt_cos_dow",
    "tgt_is_weekend",
    "tgt_is_evening",
    "tgt_is_daytime",
    "tgt_is_overnight",
]


def compute_target_time_feats(idx: pd.DatetimeIndex, h: int) -> np.ndarray:
    """Return an (N, 8) float32 array of target-time features for horizon h.

    h is a 1-based horizon bucket; each bucket is
    variables.HORIZON_GRANULARITY_IN_MINUTES minutes wide, so the described
    target time is ``idx`` shifted forward by ``h * granularity`` minutes.
    Columns follow TARGET_TIME_FEATURE_NAMES:
      sin/cos hour, sin/cos day-of-week, is_weekend, is_evening, is_daytime,
      is_overnight.
    """
    interval = variables.HORIZON_GRANULARITY_IN_MINUTES
    total_m = idx.hour * 60 + idx.minute + h * interval
    t_hr = (total_m % 1440) / 60.0
    t_dow = (idx.dayofweek + total_m // 1440) % 7
    return np.column_stack([
        np.sin(2 * np.pi * t_hr / 24.0),
        np.cos(2 * np.pi * t_hr / 24.0),
        np.sin(2 * np.pi * t_dow / 7.0),
        np.cos(2 * np.pi * t_dow / 7.0),
        (t_dow >= 5),
        ((t_hr >= 17.0) & (t_hr < 21.0)),
        ((t_hr >= 7.0) & (t_hr < 17.0)),
        ((t_hr < 7.0) | (t_hr >= 21.0)),
    ]).astype(np.float32)


def append_target_time_feats(X: pd.DataFrame, h: int) -> pd.DataFrame:
    """Append horizon-h target-time features to a per-horizon feature DataFrame.

    The calendar context is derived from ``X.index`` (the forecast-origin
    timestamps). The 8 columns are appended on the right in
    TARGET_TIME_FEATURE_NAMES order. Use this on the DataFrame prediction paths
    (holistic tuning / evaluation); for the numpy training path use
    ``compute_target_time_feats`` and ``np.hstack`` to keep the same order.
    """
    extra = pd.DataFrame(
        compute_target_time_feats(X.index, h),
        index=X.index,
        columns=TARGET_TIME_FEATURE_NAMES,
    )
    return pd.concat([X.astype(np.float32), extra], axis=1)

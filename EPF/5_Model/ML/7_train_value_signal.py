"""Train the decision-focused value signal after the eight component models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
from EPF import variables

sys.path.insert(0, str(MODEL_DIR))
from value_signal import COMPONENT_NAMES, fit_value_signal


def component_path() -> Path:
    if variables.VALIDATION_COMPONENT_PREDICTIONS_PATH.exists():
        return variables.VALIDATION_COMPONENT_PREDICTIONS_PATH
    matches = sorted(variables.HOLISTIC_MODEL_DIR.glob("1_validation_component_predictions*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected {variables.VALIDATION_COMPONENT_PREDICTIONS_PATH} or one matching validation archive."
        )
    return matches[0]


validation = pd.read_parquet(component_path())
origins = validation.index[
    (validation.index.hour == 0)
    & (validation.index.minute == 0)
    & (validation.index + pd.Timedelta(hours=variables.HORIZON_LENGTH_IN_HOURS) <= variables.TEST_START)
]
if origins.empty:
    raise ValueError("No complete midnight validation paths are available for value-signal training.")

components = {
    name: validation.loc[origins, [f"{name}_h{h}" for h in range(1, variables.HORIZON_COUNT + 1)]].to_numpy(
        dtype=np.float32
    )
    for name in COMPONENT_NAMES
}
actual = pd.read_parquet(
    variables.AGG_TARGET_DATASET_PATH,
    columns=[f"target_h{h}" for h in range(1, variables.HORIZON_COUNT + 1)],
).loc[origins].to_numpy(dtype=np.float32)

model = fit_value_signal(components, actual, origins)
variables.HOLISTIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save(variables.VALUE_SIGNAL_MODEL_PATH)
print(f"Saved value signal trained on {len(origins)} midnight origins to {variables.VALUE_SIGNAL_MODEL_PATH}")

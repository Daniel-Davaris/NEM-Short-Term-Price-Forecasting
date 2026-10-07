"""Decision-focused post-processing for the eight component price models.

The residual stacker remains the statistically oriented central forecast.  The
value signal learns the high-price and low-price ordering from the same eight
component outputs, combines those rankings, maps them back to a conservative
price-shaped curve, and blends it into the central forecast. The optimizer can
therefore consume it without any special signal handling.

Only forecast information available at the origin is used.  Training is
restricted to midnight origins because that is the operational cadence used by
the rolling optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd


COMPONENT_NAMES = (
    "base_l1",
    "base_l2",
    "spike_probability",
    "spike_mae",
    "spike_q90",
    "dip_probability",
    "dip_mae",
    "dip_q10",
)
PRICE_COMPONENTS = ("base_l1", "base_l2", "spike_mae", "spike_q90", "dip_mae", "dip_q10")


def _component_cube(components: Mapping[str, np.ndarray]) -> np.ndarray:
    missing = set(COMPONENT_NAMES) - set(components)
    if missing:
        raise KeyError(f"Missing value-signal components: {sorted(missing)}")
    cube = np.stack([np.asarray(components[name], dtype=np.float32) for name in COMPONENT_NAMES], axis=2)
    if cube.ndim != 3:
        raise ValueError("Each component must be an origin-by-horizon matrix.")
    if not np.isfinite(cube).all():
        raise ValueError("Value-signal components contain non-finite values.")
    return cube


def _row_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic within-row percentile ranks in [0, 1]."""
    count = values.shape[1]
    if count < 2:
        return np.zeros_like(values, dtype=np.float32)
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.argsort(order, axis=1, kind="stable")
    return (ranks / (count - 1)).astype(np.float32)


def build_features(
    components: Mapping[str, np.ndarray],
    forecast_origins: pd.DatetimeIndex | None = None,
) -> np.ndarray:
    """Build path-aware features from all eight trained-model outputs."""
    cube = _component_cube(components)
    prices = cube[:, :, [COMPONENT_NAMES.index(name) for name in PRICE_COMPONENTS]]

    median = np.median(prices, axis=1, keepdims=True)
    iqr = np.maximum(
        np.quantile(prices, 0.75, axis=1, keepdims=True)
        - np.quantile(prices, 0.25, axis=1, keepdims=True),
        10.0,
    )
    robust_deviation = np.clip((prices - median) / iqr, -10.0, 10.0)
    price_ranks = np.stack([_row_ranks(prices[:, :, index]) for index in range(prices.shape[2])], axis=2)
    transformed_prices = np.arcsinh(np.clip(prices, -1_000.0, 5_000.0) / 100.0)
    probabilities = np.clip(cube[:, :, [2, 5]], 1e-5, 1 - 1e-5)

    horizon_count = cube.shape[1]
    horizon_minutes = np.arange(horizon_count, dtype=np.float32) * 30.0
    if forecast_origins is None:
        minute_of_day = np.broadcast_to(horizon_minutes % (24 * 60), (len(cube), horizon_count))
    else:
        origins = pd.DatetimeIndex(forecast_origins)
        if len(origins) != len(cube):
            raise ValueError("forecast_origins must have one value per component row.")
        origin_minutes = (origins.hour * 60 + origins.minute).to_numpy(dtype=np.float32)[:, None]
        minute_of_day = (origin_minutes + horizon_minutes[None, :]) % (24 * 60)
    angle = minute_of_day * (2 * np.pi / (24 * 60))
    day_ahead = np.broadcast_to((np.arange(horizon_count) >= 48).astype(np.float32), angle.shape)
    time_features = np.stack([np.sin(angle), np.cos(angle), day_ahead], axis=2)

    return np.concatenate(
        [transformed_prices, probabilities, robust_deviation, price_ranks, time_features],
        axis=2,
    ).astype(np.float32)


@dataclass
class ValueSignalModel:
    high_ranker: object
    low_ranker: object
    centered_price_quantiles: np.ndarray
    blend_weight: float = 0.10
    deviation_scale: float = 0.60
    horizon_count: int = 96
    metadata: dict = field(default_factory=dict)

    def predict(
        self,
        components: Mapping[str, np.ndarray],
        central_forecast: np.ndarray,
        forecast_origins: pd.DatetimeIndex | None = None,
    ) -> np.ndarray:
        central = np.asarray(central_forecast, dtype=np.float32)
        if central.ndim != 2 or central.shape[1] != self.horizon_count:
            raise ValueError(f"central_forecast must have shape (n, {self.horizon_count}).")

        features = build_features(components, forecast_origins)
        if features.shape[:2] != central.shape:
            raise ValueError("Component and central forecast shapes do not match.")
        flat_features = features.reshape(-1, features.shape[2])
        high_scores = self.high_ranker.predict(flat_features).reshape(central.shape)
        low_scores = self.low_ranker.predict(flat_features).reshape(central.shape)
        decision_scores = _row_ranks(high_scores) - _row_ranks(low_scores)
        score_ranks = _row_ranks(decision_scores)
        quantile_index = np.rint(score_ranks * (len(self.centered_price_quantiles) - 1)).astype(int)
        centered_curve = np.take(self.centered_price_quantiles, quantile_index)
        base_level = np.median(np.asarray(components["base_l1"]), axis=1, keepdims=True)
        rank_signal = base_level + self.deviation_scale * centered_curve
        signal = central + self.blend_weight * (rank_signal - central)
        return signal.astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "ValueSignalModel":
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a ValueSignalModel.")
        return model


def fit_value_signal(
    components: Mapping[str, np.ndarray],
    actual_prices: np.ndarray,
    forecast_origins: pd.DatetimeIndex,
) -> ValueSignalModel:
    """Fit the compact rank layer on operational midnight forecast paths."""
    from lightgbm import LGBMRanker

    actual = np.asarray(actual_prices, dtype=np.float32)
    features = build_features(components, forecast_origins)
    if actual.shape != features.shape[:2]:
        raise ValueError("actual_prices must match the component origin-by-horizon shape.")

    relevance = np.floor(_row_ranks(actual) * 31.999).astype(np.int8)
    ranker_parameters = dict(
        objective="lambdarank",
        metric="ndcg",
        lambdarank_truncation_level=18,
        label_gain=list(range(32)),
        num_leaves=15,
        min_child_samples=40,
        learning_rate=0.03,
        n_estimators=400,
        reg_lambda=4.0,
        reg_alpha=0.2,
        verbosity=-1,
        n_jobs=-1,
        random_state=42,
    )
    flat_features = features.reshape(-1, features.shape[2])
    groups = [actual.shape[1]] * len(actual)
    high_ranker = LGBMRanker(**ranker_parameters).fit(flat_features, relevance.ravel(), group=groups)
    low_ranker = LGBMRanker(**ranker_parameters).fit(flat_features, (31 - relevance).ravel(), group=groups)

    centered = actual - np.median(actual, axis=1, keepdims=True)
    centered = np.clip(centered, -500.0, 1_500.0)
    calibration = np.quantile(centered.ravel(), np.linspace(0, 1, actual.shape[1])).astype(np.float32)
    return ValueSignalModel(
        high_ranker=high_ranker,
        low_ranker=low_ranker,
        centered_price_quantiles=calibration,
        horizon_count=actual.shape[1],
        metadata={
            "training_start": str(pd.DatetimeIndex(forecast_origins).min()),
            "training_end": str(pd.DatetimeIndex(forecast_origins).max()),
            "training_origins": len(actual),
            "component_names": COMPONENT_NAMES,
            "purpose": "joint 2h/4h storage dispatch ranking",
        },
    )

"""Forecast-archive and optimizer-export helpers.

The forecasting pipeline stores one row per forecast origin and one column per
30-minute horizon.  This lossless archive is deliberately kept separate from
the optimizer handoff, which needs one price for each delivery interval.
Choosing a forecast cadence or an overlap policy therefore never requires the
models to be run again.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_forecast_archive(
    path: Path | str,
    *,
    column_prefix: str | None = None,
    horizon_count: int = 96,
) -> pd.DataFrame:
    """Read an origin-by-horizon archive and normalize its index.

    Supplying ``column_prefix`` reads only that method's horizon columns. This
    keeps the lightweight optimizer export from loading the much wider combined
    evaluation archive into memory.
    """
    path = Path(path)
    expected = (
        [f"{column_prefix}_h{h}" for h in range(1, horizon_count + 1)]
        if column_prefix is not None else None
    )
    if path.suffix.lower() == ".parquet":
        forecasts = pd.read_parquet(path, columns=expected)
    else:
        if expected is None:
            forecasts = pd.read_csv(path)
        else:
            selected = set(expected) | {"forecast_origin", "date", "Date"}
            forecasts = pd.read_csv(path, usecols=lambda column: column in selected)

    for name in ("forecast_origin", "date", "Date"):
        if name in forecasts.columns:
            forecasts = forecasts.set_index(name)
            break
    forecasts.index = pd.to_datetime(forecasts.index)
    forecasts.index.name = "forecast_origin"
    return forecasts.sort_index()


def optimizer_price_series(
    forecasts: pd.DataFrame,
    *,
    column_prefix: str,
    region: str,
    horizon_count: int = 96,
    horizon_minutes: int = 30,
    forecast_cadence: str | pd.Timedelta = "48h",
    overlap: str = "latest_origin",
    anchor: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Collapse the complete forecast archive to the optimizer's A2 schema.

    ``forecasts`` remains the source of truth.  ``forecast_cadence`` merely
    selects which saved origins would have been operational runs.  When those
    48-hour forecast windows overlap, ``overlap`` chooses the value retained
    for each delivery interval: ``latest_origin`` (shortest lead),
    ``earliest_origin`` (longest lead), or ``mean``.

    The optimizer labels intervals by their start.  In this project h1 is the
    average price over the first 30 minutes after the origin, so h1 is exported
    at ``Date == forecast_origin``; h2 is exported 30 minutes later, etc.
    """
    if forecasts.empty:
        raise ValueError("The forecast archive is empty.")

    work = forecasts.copy()
    work.index = pd.to_datetime(work.index)
    work = work.sort_index()
    expected = [f"{column_prefix}_h{h}" for h in range(1, horizon_count + 1)]
    missing = [column for column in expected if column not in work]
    if missing:
        raise KeyError(f"Forecast archive is missing {len(missing)} columns: {missing[:5]}")

    cadence = pd.Timedelta(forecast_cadence)
    if cadence <= pd.Timedelta(0):
        raise ValueError("forecast_cadence must be positive.")
    anchor_time = pd.Timestamp(anchor) if anchor is not None else work.index[0]
    elapsed = work.index - anchor_time
    selected = work.loc[(elapsed % cadence) == pd.Timedelta(0), expected]
    if selected.empty:
        raise ValueError("No saved forecast origins match the requested cadence and anchor.")

    step = pd.Timedelta(minutes=horizon_minutes)
    pieces = []
    for horizon, column in enumerate(expected, start=1):
        pieces.append(pd.DataFrame({
            "Date": selected.index + (horizon - 1) * step,
            "forecast_origin": selected.index,
            "Predicted_price": selected[column].to_numpy(),
        }))
    candidates = pd.concat(pieces, ignore_index=True).dropna(subset=["Predicted_price"])

    if overlap == "mean":
        collapsed = candidates.groupby("Date", as_index=False)["Predicted_price"].mean()
    elif overlap in {"latest_origin", "earliest_origin"}:
        candidates = candidates.sort_values(["Date", "forecast_origin"])
        keep = "last" if overlap == "latest_origin" else "first"
        collapsed = candidates.drop_duplicates("Date", keep=keep)[["Date", "Predicted_price"]]
    else:
        raise ValueError("overlap must be 'latest_origin', 'earliest_origin', or 'mean'.")

    output = collapsed.rename(columns={"Predicted_price": f"{region.lower()}_price"})
    return output.sort_values("Date").reset_index(drop=True)


def optimizer_price_vintages(
    forecasts: pd.DataFrame,
    *,
    column_prefix: str,
    region: str,
    horizon_count: int = 96,
    horizon_minutes: int = 30,
    forecast_cadence: str | pd.Timedelta = "24h",
    anchor: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return every delivery price from each operational forecast origin.

    Unlike :func:`optimizer_price_series`, this deliberately does not collapse
    overlapping forecasts.  The optimizer uses the vintage issued at the start
    of each rolling window, preventing later forecast runs from leaking into an
    earlier dispatch decision.
    """
    if forecasts.empty:
        raise ValueError("The forecast archive is empty.")

    work = forecasts.copy()
    work.index = pd.to_datetime(work.index)
    work = work.sort_index()
    expected = [f"{column_prefix}_h{h}" for h in range(1, horizon_count + 1)]
    missing = [column for column in expected if column not in work]
    if missing:
        raise KeyError(f"Forecast archive is missing {len(missing)} columns: {missing[:5]}")

    cadence = pd.Timedelta(forecast_cadence)
    if cadence <= pd.Timedelta(0):
        raise ValueError("forecast_cadence must be positive.")
    anchor_time = pd.Timestamp(anchor) if anchor is not None else work.index[0]
    selected = work.loc[((work.index - anchor_time) % cadence) == pd.Timedelta(0), expected]
    if selected.empty:
        raise ValueError("No saved forecast origins match the requested cadence and anchor.")

    step = pd.Timedelta(minutes=horizon_minutes)
    pieces = []
    for horizon, column in enumerate(expected, start=1):
        pieces.append(pd.DataFrame({
            "forecast_origin": selected.index,
            "Date": selected.index + (horizon - 1) * step,
            f"{region.lower()}_price": selected[column].to_numpy(),
        }))
    return (
        pd.concat(pieces, ignore_index=True)
        .dropna(subset=[f"{region.lower()}_price"])
        .sort_values(["forecast_origin", "Date"])
        .reset_index(drop=True)
    )


def write_optimizer_price_csv(
    forecasts: pd.DataFrame,
    output_path: Path | str,
    **selection,
) -> pd.DataFrame:
    """Write ``Date,<region>_price`` in the 2_Optimizer A2 input format."""
    output = optimizer_price_series(forecasts, **selection)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output

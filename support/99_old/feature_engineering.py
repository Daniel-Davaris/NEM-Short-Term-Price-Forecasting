"""
feature_engineering.py  –  Build the supervised-learning feature matrix from
                            the raw 30-min price/demand time series.

Feature groups
--------------
1. Time / calendar features (hour, day-of-week, month, cyclical encodings,
   weekend flag, Australian NSW public-holiday flag, peak-period flags)
2. Lag features  (price and demand at past intervals)
3. Rolling statistics (mean, std, min, max over multiple windows)
4. Price-regime indicators (recent spikes, negative prices, volatility)

"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import holidays
import numpy as np
import pandas as pd
import Model.config as config

PRICE_TRANSFORM_SCALE = 100.0
NSW_COAL_MAX_MW    = 8_500   # Eraring 2880 + Bayswater 2640 + Mt Piper 1400 + Vales Point 1320

ANNUAL_LAG         = 17_532  # ~1 year in 30-min intervals
ANNUAL_LAG_SPREAD  = 2       # ± spread around the anchor

LAG_INTERVALS = [
    1, 2, 3, 4, 6, 8, 12, 24,
    48, 49, 50, 51, 52,
    96, 97, 98,
    # 3-day and 4-day same-period anchors — captures weekly envelope effects
    # without redundancy with the weekly (336) lag.
    144, 143, 145,
    192, 191, 193,
    336, 335, 337,
    672, 671, 673,
]

ROLLING_WINDOWS = [4, 8, 24, 48, 96, 336, 672, 2016]

# Base temperature for heating/cooling degree calculations (standard for Australia)
_BASE_TEMP_C = 18.0

# NSW public holidays lookup (covers the full data range)
_NSW_HOLIDAYS = holidays.Australia(state="NSW", years=range(2018, 2025))

_REGION_META = {
    "NSW1": {"price_col": "nsw_price", "suffix": "nsw", "self_neighbour": None},
    "QLD1": {"price_col": "qld_price", "suffix": "qld", "self_neighbour": "qld_price"},
    "VIC1": {"price_col": "vic_price", "suffix": "vic", "self_neighbour": "vic_price"},
    "SA1": {"price_col": "sa_price", "suffix": "sa", "self_neighbour": "sa_price"},
}


def select_region_columns(raw_df: pd.DataFrame, region: str) -> pd.DataFrame:
    meta = _REGION_META.get(region)
    if meta is None:
        raise ValueError(f"Unsupported region: {region}. Supported: {sorted(_REGION_META)}")

    df = raw_df.copy()
    price_col = meta["price_col"]
    if price_col not in df.columns:
        raise ValueError(f"Required price column '{price_col}' missing for {region}")

    df["price"] = df[price_col].astype(np.float32)

    suffix = meta["suffix"]
    for base in ("demand", "avail_gen", "interchange", "demand_forecast", "dispatch_gen"):
        src = f"{base}_{suffix}"
        if src in df.columns:
            df[base] = df[src]

    coal_src = f"coal_mw_{suffix}"
    if coal_src in df.columns:
        df["coal_mw"] = df[coal_src]

    for sig in ("pasa_avail_gen", "pasa_demand50", "pasa_reserve_cond"):
        src = f"{sig}_{suffix}"
        if src in df.columns:
            df[sig] = df[src]

    self_neighbour = meta["self_neighbour"]
    if self_neighbour and self_neighbour in df.columns:
        df[self_neighbour] = np.nan

    region_lower = region.lower()
    step_src_cols = []
    for h in range(config.FORECAST_GAP, config.FORECAST_GAP + config.FORECAST_HORIZON):
        src = f"predispatch_rrp_h{h}_{region_lower}"
        if src in df.columns:
            df[f"predispatch_rrp_h{h}"] = df[src].astype(np.float32)
            step_src_cols.append(src)

    for h in range(config.FORECAST_GAP, config.FORECAST_GAP + config.FORECAST_HORIZON):
        for sig in ("pasa_avail_gen", "pasa_demand50", "pasa_reserve_cond"):
            src = f"{sig}_h{h}_{suffix}"
            if src in df.columns:
                df[f"{sig}_h{h}"] = df[src].astype(np.float32)
                if src not in step_src_cols:
                    step_src_cols.append(src)

    return df.drop(columns=[c for c in step_src_cols if c in df.columns], errors="ignore")


# ---------------------------------------------------------------------------
# Individual feature builders
# ---------------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df = df.copy()

    # Raw calendar components
    df["hour"]        = idx.hour.astype(np.int8)
    df["dayofweek"]   = idx.dayofweek.astype(np.int8)
    df["month"]       = idx.month.astype(np.int8)
    df["dayofyear"]   = idx.day_of_year.astype(np.int16)

    # Cyclical (sin/cos) encodings so the model sees periodicity
    df["hour_sin"]    = np.sin(2 * np.pi * idx.hour / 24).astype(np.float32)
    df["hour_cos"]    = np.cos(2 * np.pi * idx.hour / 24).astype(np.float32)
    df["dow_sin"]     = np.sin(2 * np.pi * idx.dayofweek / 7).astype(np.float32)
    df["dow_cos"]     = np.cos(2 * np.pi * idx.dayofweek / 7).astype(np.float32)
    df["month_sin"]   = np.sin(2 * np.pi * (idx.month - 1) / 12).astype(np.float32)
    df["month_cos"]   = np.cos(2 * np.pi * (idx.month - 1) / 12).astype(np.float32)

    # Binary flags
    df["is_weekend"]  = (idx.dayofweek >= 5).astype(np.float32)
    df["is_holiday"]  = np.array(
        [d.date() in _NSW_HOLIDAYS for d in idx], dtype=np.float32
    )
    # Peak (17–20 h) and off-peak (< 7 h or ≥ 21 h) periods
    df["is_peak"]     = ((idx.hour >= 17) & (idx.hour <= 20)).astype(np.float32)
    df["is_shoulder"] = ((idx.hour >= 7)  & (idx.hour < 17)).astype(np.float32)
    df["is_off_peak"] = ((idx.hour < 7)   | (idx.hour >= 21)).astype(np.float32)

    # Combined weekend/holiday flag — demand behaviour is quite different
    df["is_offday"]   = np.maximum(df["is_weekend"], df["is_holiday"]).astype(np.float32)

    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in sorted(set(LAG_INTERVALS)):
        df[f"price_lag_{lag}"]  = df["price"].shift(lag).astype(np.float32)
    # Intermediate price lags at 1-hour resolution bridging the gaps in LAG_INTERVALS.
    # With config.FORECAST_GAP=24 (12 h), prediction steps run from h=24 to h=71.
    # price_lag_{h} captures same-duration autocorrelation for each step.
    # LAG_INTERVALS already covers anchors at 24, 48-52, 96, ...; this loop
    # fills the even-interval gaps across the full 12h–36h look-back range.
    for lag in range(26, 72, 2):         # 26, 28, ..., 70 (step = 1 h = 2 intervals)
        if lag not in LAG_INTERVALS:
            df[f"price_lag_{lag}"] = df["price"].shift(lag).astype(np.float32)
    # Demand lags at key calendar anchors
    for lag in [48, 96, 336, 672]:
        df[f"demand_lag_{lag}"] = df["demand"].shift(lag).astype(np.float32)
    return df


def _add_long_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seasonal and 2-week-ahead context features.

    NSW electricity prices have a strong annual seasonal pattern:
    - Summer (Dec-Feb): high demand from AC, solar peaks mid-day
    - Winter (Jun-Aug): evening heating peaks, low solar
    Prices in Jan 2024 are partly predicted by what happened in Jan 2023.

    Only computed when the DataFrame is long enough to have valid annual lags
    (requires at least ~13 months of data before the first row used for training).
    Rows before ANNUAL_LAG will remain NaN and be dropped by dropna().
    """
    df   = df.copy()
    p    = df["price"]
    d    = df["demand"]
    ap   = np.arcsinh(p / PRICE_TRANSFORM_SCALE)
    BASE = ANNUAL_LAG

    # --- Annual price lags (same period last year ± spread) ---
    for offset in range(-ANNUAL_LAG_SPREAD, ANNUAL_LAG_SPREAD + 1):
        lag = BASE + offset
        df[f"price_lag_annual_{'+' if offset >= 0 else ''}{offset}"] = (
            p.shift(lag).astype(np.float32)
        )
        df[f"price_asinh_lag_annual_{'+' if offset >= 0 else ''}{offset}"] = (
            ap.shift(lag).astype(np.float32)
        )

    # --- Annual demand lag ---
    df["demand_lag_annual"] = d.shift(BASE).astype(np.float32)

    # --- Rolling statistics over the same week last year (annual context) ---
    # 1-week window centred on the annual anchor: [BASE-48, BASE+48]
    # Approximated as a rolling window on the shifted series.
    p_annual = p.shift(BASE)
    df["price_annual_rmean_96"]  = p_annual.rolling(96, min_periods=24).mean().astype(np.float32)
    df["price_annual_rmax_96"]   = p_annual.rolling(96, min_periods=24).max().astype(np.float32)
    df["price_annual_rstd_96"]   = p_annual.rolling(96, min_periods=24).std().astype(np.float32)

    # Spike count over the same week last year
    df["price_annual_spike_96"]  = (p_annual >= 300).rolling(96, min_periods=24).sum().astype(np.float32)

    # Year-on-year price change (current vs same period last year)
    df["price_yoy_change"]  = (p - p.shift(BASE)).astype(np.float32)
    df["price_yoy_ratio"]   = (p / (p.shift(BASE).abs() + 1)).clip(0, 20).astype(np.float32)

    # --- 2-week rolling statistics (not in main ROLLING_WINDOWS to keep loop tight) ---
    df["price_rmean_2w"]  = p.rolling(672, min_periods=336).mean().astype(np.float32)
    df["price_rmax_2w"]   = p.rolling(672, min_periods=336).max().astype(np.float32)
    df["price_rmin_2w"]   = p.rolling(672, min_periods=336).min().astype(np.float32)
    df["price_rstd_2w"]   = p.rolling(672, min_periods=336).std().astype(np.float32)
    df["demand_rmean_2w"] = d.rolling(672, min_periods=336).mean().astype(np.float32)

    # 6-week rolling mean (seasonal trend)
    df["price_rmean_6w"]  = p.rolling(2016, min_periods=1008).mean().astype(np.float32)
    df["demand_rmean_6w"] = d.rolling(2016, min_periods=1008).mean().astype(np.float32)

    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling statistics computed on prices/demand available at each timestamp."""
    df = df.copy()
    p = df["price"]
    d = df["demand"]
    for w in ROLLING_WINDOWS:
        min_p = max(1, w // 2)
        rolled = p.rolling(w, min_periods=min_p)
        df[f"price_rmean_{w}"] = rolled.mean().astype(np.float32)
        df[f"price_rstd_{w}"]  = rolled.std().astype(np.float32)
        df[f"price_rmax_{w}"]  = rolled.max().astype(np.float32)
        df[f"price_rmin_{w}"]  = rolled.min().astype(np.float32)
    for w in [4, 24, 48, 96]:
        df[f"demand_rmean_{w}"] = d.rolling(w, min_periods=1).mean().astype(np.float32)
    return df


def _add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capture price-spike and volatility regime signals.
    All windows look backward only, so there is no future leakage.
    """
    df = df.copy()
    p  = df["price"]

    # Was there a high-price spike in the last 24 h?
    df["spike_flag_48"]    = (p.rolling(48).max()  > 300).astype(np.float32)
    # Was there a negative price in the last 24 h?
    df["neg_flag_48"]      = (p.rolling(48).min()  < 0).astype(np.float32)
    # Recent volatility
    df["price_vol_48"]     = p.rolling(48).std().astype(np.float32)
    df["price_vol_336"]    = p.rolling(336).std().astype(np.float32)
    # Quantile context (recent price level relative to weekly range)
    df["price_q90_336"]    = p.rolling(336).quantile(0.90).astype(np.float32)
    df["price_q10_336"]    = p.rolling(336).quantile(0.10).astype(np.float32)
    df["price_pct_rank_48"] = (
        p.rolling(48).rank(pct=True)
    ).astype(np.float32)

    return df


def _add_time_since_spike_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Track how long ago the last price spike occurred, at multiple thresholds.

    The NEM clusters spikes: a unit that tripped once is likely still offline,
    and the market is already in a stressed state.  How many hours ago the last
    spike occurred is therefore a strong predictor of both the probability AND
    the magnitude of the next spike.

    Three thresholds capture different stress levels:
      $150 — elevated price (supply getting tight)
      $300 — spike threshold used by the ensemble classifier
      $1000 — extreme/emergency pricing

    For each threshold we compute:
      hours_since_spike_<N> : hours since the last interval with price >= N (capped)
      spike_in_last_Xh_<N>  : binary flag — was there a spike in the last X hours?

    All windows are strictly backward-looking — no future leakage.
    """
    df   = df.copy()
    p    = df["price"]
    # 30-min intervals between rows
    INTERVAL_H = 0.5
    # Maximum hours to report (cap at 2 weeks to avoid unbounded values early in series)
    MAX_HOURS  = 336.0  # 2 weeks

    for threshold in [150, 300, 1000]:
        spike_flag = (p >= threshold).astype(np.float32)

        # Build a Series of the interval index of the most recent spike,
        # then convert the gap to hours.
        # Strategy: cumsum of spike_flag gives a group ID that increments at
        # each spike.  shift(1) carries the last spike group forward.
        # Within each group, row position gives intervals-since-last-spike.
        cumsum   = spike_flag.cumsum()
        # intervals since last spike = row_number within current no-spike run
        # Use a cumulative position counter minus the position of last spike
        positions       = pd.RangeIndex(len(df))
        last_spike_pos  = (
            pd.Series(np.where(spike_flag.values, positions, np.nan), index=df.index)
            .ffill()
            .fillna(-MAX_HOURS / INTERVAL_H)  # no prior spike seen → treat as max
        )
        intervals_since = (pd.Series(positions, index=df.index) - last_spike_pos).clip(upper=MAX_HOURS / INTERVAL_H)
        hours_since     = (intervals_since * INTERVAL_H).astype(np.float32)
        col             = f"hours_since_spike_{threshold}"
        df[col]         = hours_since

        # Log-scale version reduces the numeric range for the tree learner
        df[f"log1p_hours_since_spike_{threshold}"] = np.log1p(hours_since).astype(np.float32)

        # Binary flags: was there a spike at each lookback horizon?
        for lookback_h in [1, 6, 12, 24, 48, 168]:   # 1h, 6h, 12h, 24h, 48h, 1wk
            intervals = int(lookback_h / INTERVAL_H)
            df[f"spike_{threshold}_last_{lookback_h}h"] = (
                spike_flag.rolling(intervals, min_periods=1).max().astype(np.float32)
            )

    # Combined: hours since any of the three thresholds (summarises overall stress state)
    df["hours_since_any_spike"] = df[["hours_since_spike_150",
                                      "hours_since_spike_300",
                                      "hours_since_spike_1000"]].min(axis=1).astype(np.float32)

    return df


def _add_system_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived from AEMO DISPATCHREGIONSUM system-level columns:
    available generation, net interchange, and demand forecast.
    These are strong predictors of supply tightness and price spikes.
    """
    df = df.copy()
    ag = df["avail_gen"]
    ic = df["interchange"]
    dc = df["demand_forecast"]
    dm = df["demand"]

    # Reserve margin: how much headroom between available gen and demand
    df["reserve_margin"]     = ((ag - dm) / (dm + 1)).clip(-2, 10).astype(np.float32)
    df["reserve_margin_pct"] = (ag / (dm + 1)).clip(0, 5).astype(np.float32)

    # Demand forecast vs actual demand (indicates demand surprise)
    df["demand_fcst_error"]  = (dc - dm).astype(np.float32)

    # Lags for available generation
    for lag in [1, 2, 4, 48, 96, 336]:
        df[f"avail_gen_lag_{lag}"] = ag.shift(lag).astype(np.float32)

    # Lags for net interchange (positive = importing into NSW)
    for lag in [1, 2, 4, 48, 96, 336]:
        df[f"interchange_lag_{lag}"] = ic.shift(lag).astype(np.float32)
    df["interchange_rmean_48"]  = ic.rolling(48).mean().astype(np.float32)
    df["interchange_rmean_336"] = ic.rolling(336).mean().astype(np.float32)

    # Demand forecast lags
    for lag in [1, 2, 48]:
        df[f"demand_fcst_lag_{lag}"] = dc.shift(lag).astype(np.float32)

    # Reserve margin rolling stats (tightening/loosening supply)
    rm = df["reserve_margin"]
    df["reserve_rmean_48"]  = rm.rolling(48).mean().astype(np.float32)
    df["reserve_rmin_48"]   = rm.rolling(48).min().astype(np.float32)

    # Dispatched generation features (from DISPATCHABLEGENERATION column)
    if "dispatch_gen" in df.columns:
        dg = df["dispatch_gen"]
        # Thermal utilisation: what fraction of available capacity is dispatched
        df["thermal_util"]       = (dg / (ag + 1)).clip(0, 1.5).astype(np.float32)
        # Spare thermal headroom (MW) — low values precede price spikes
        df["thermal_surplus_mw"] = (ag - dg).clip(-2000, 8000).astype(np.float32)
        for lag in [1, 2, 4, 48, 96, 336]:
            df[f"dispatch_gen_lag_{lag}"] = dg.shift(lag).astype(np.float32)
        df["dispatch_gen_chg_4h"]   = dg.diff(8).astype(np.float32)    # 4h change
        df["dispatch_gen_chg_24h"]  = dg.diff(48).astype(np.float32)   # 24h change
        df["dispatch_gen_rmin_48"]  = dg.rolling(48).min().astype(np.float32)
        df["dispatch_gen_rmean_48"] = dg.rolling(48).mean().astype(np.float32)
        df["thermal_surplus_rmin_48"] = df["thermal_surplus_mw"].rolling(48).min().astype(np.float32)

    return df


def _add_generation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived from DISPATCH_UNIT_SCADA coal generation (coal_mw).
    A sudden drop in coal MW signals a generator trip — the primary cause
    of extreme NSW price spikes.
    No-op (returns df unchanged) if coal_mw is absent or all-NaN.
    """
    df = df.copy()
    if "coal_mw" not in df.columns or df["coal_mw"].isna().all():
        return df

    coal = df["coal_mw"]

    # Absolute level lags
    for lag in [1, 2, 4, 48, 96, 336]:
        df[f"coal_lag_{lag}"] = coal.shift(lag).astype(np.float32)

    # Rolling statistics
    df["coal_rmean_4"]   = coal.rolling(4).mean().astype(np.float32)
    df["coal_rmean_48"]  = coal.rolling(48).mean().astype(np.float32)
    df["coal_rmin_48"]   = coal.rolling(48).min().astype(np.float32)
    df["coal_rmin_96"]   = coal.rolling(96).min().astype(np.float32)

    # Rate-of-change — rapid drops signal generator trips
    df["coal_chg_1h"]  = coal.diff(2).astype(np.float32)    # 2×30-min = 1h
    df["coal_chg_6h"]  = coal.diff(12).astype(np.float32)   # 6h
    df["coal_chg_24h"] = coal.diff(48).astype(np.float32)   # 24h

    # Outage flag: coal dropped by >600 MW (approx 1 unit) within last 2h
    df["coal_outage_flag"] = (
        coal.rolling(4).min() < coal.shift(4) - 600
    ).astype(np.float32)

    # Capacity utilisation relative to approximate installed max
    df["coal_util"]          = (coal / NSW_COAL_MAX_MW).clip(0, 1.05).astype(np.float32)
    df["coal_low_flag"]      = (coal < NSW_COAL_MAX_MW * 0.55).astype(np.float32)
    df["coal_low_count_48"]  = df["coal_low_flag"].rolling(48).sum().astype(np.float32)
    df["coal_low_count_336"] = df["coal_low_flag"].rolling(336).sum().astype(np.float32)

    return df


def _add_arcsinh_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    arcsinh-transformed price lags.
    arcsinh handles negatives and compresses extreme spikes,
    giving the model a better-scaled view of price history.
    """
    df   = df.copy()
    scale = PRICE_TRANSFORM_SCALE
    ap   = np.arcsinh(df["price"] / scale).rename("_ap")
    for lag in [1, 2, 4, 12, 48, 96, 336, 335, 337]:
        df[f"price_asinh_lag_{lag}"] = ap.shift(lag).astype(np.float32)
    df["price_asinh_rmean_48"]  = ap.rolling(48).mean().astype(np.float32)
    df["price_asinh_rmean_336"] = ap.rolling(336).mean().astype(np.float32)
    return df


def _add_spike_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supply-stress and spike-history features.
    All look-back only — zero leakage.
    """
    df = df.copy()
    p  = df["price"]
    ag = df["avail_gen"]
    dm = df["demand"]
    dc = df["demand_forecast"]

    # How loaded the system is (>1 means demand > available gen)
    supply_stress = dm / (ag + 1)
    df["supply_stress"]          = supply_stress.clip(0, 2).astype(np.float32)
    df["supply_stress_max_48"]   = supply_stress.rolling(48).max().astype(np.float32)
    df["supply_stress_max_96"]   = supply_stress.rolling(96).max().astype(np.float32)

    # Count of intervals in last 24h with very tight supply (>92% utilisation)
    df["tight_count_48"]  = (supply_stress > 0.92).rolling(48).sum().astype(np.float32)
    df["tight_count_336"] = (supply_stress > 0.92).rolling(336).sum().astype(np.float32)

    # Detects generator outages: large drops in available generation
    df["avail_gen_chg_24h"] = ag.pct_change(48).clip(-1, 1).astype(np.float32)
    df["avail_gen_rmin_48"] = ag.rolling(48).min().astype(np.float32)
    df["avail_gen_rmin_96"] = ag.rolling(96).min().astype(np.float32)

    # Price momentum — how fast prices are moving right now
    df["price_mom_4"]  = p.diff(4).astype(np.float32)    # 2h
    df["price_mom_12"] = p.diff(12).astype(np.float32)   # 6h
    df["price_mom_48"] = p.diff(48).astype(np.float32)   # 24h

    # Price acceleration (second derivative) — is the current spike accelerating?
    df["price_accel_4"]  = p.diff(4).diff(4).clip(-2000, 2000).astype(np.float32)
    df["price_accel_12"] = p.diff(12).diff(12).clip(-2000, 2000).astype(np.float32)

    # Spike and negative price counts in recent history
    df["spike_count_48"]  = (p >= 300).rolling(48).sum().astype(np.float32)
    df["spike_count_336"] = (p >= 300).rolling(336).sum().astype(np.float32)
    df["neg_count_48"]    = (p < 0).rolling(48).sum().astype(np.float32)

    # Spike intensity: cumulative spike energy in last 24h (not just count)
    SPIKE_THR = 150.0
    spike_excess = (p - SPIKE_THR).clip(lower=0)
    df["spike_intensity_48"]  = spike_excess.rolling(48).sum().astype(np.float32)
    df["spike_intensity_336"] = spike_excess.rolling(336).sum().astype(np.float32)

    # Price percentile rank within recent window (how extreme is current price?)
    df["price_pctrank_48"]  = p.rolling(48).rank(pct=True).astype(np.float32)
    df["price_pctrank_336"] = p.rolling(336).rank(pct=True).astype(np.float32)

    # Short-term demand surprise: actual demand vs dispatch forecast
    df["demand_surprise"]     = (dm - dc).astype(np.float32)
    df["demand_surprise_abs"] = (dm - dc).abs().astype(np.float32)
    df["demand_surprise_rm48"] = (dm - dc).rolling(48).mean().astype(np.float32)

    # Supply headroom as percentage: (avail - demand) / demand
    df["supply_headroom_pct"] = ((ag - dm) / (dm + 1)).clip(-0.5, 1.0).astype(np.float32)

    # Forward tightness proxy: demand forecast vs available generation
    fwd_tight = (dc - ag).clip(-5000, 5000)
    df["fwd_tight_proxy"]      = fwd_tight.astype(np.float32)
    df["fwd_tight_proxy_lag48"] = fwd_tight.shift(48).astype(np.float32)

    return df


def _add_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag and rolling features for neighbouring NEM region prices (QLD, VIC, SA).
    Price spreads between regions indicate interconnector pressure — a
    key leading indicator of NSW spikes. SA1 is the most volatile NEM
    region and is a leading indicator of gas-driven spikes that propagate
    through the VIC-NSW interconnector.
    """
    df = df.copy()
    new_cols: dict = {}
    for col in ["qld_price", "vic_price", "sa_price"]:
        if col not in df.columns:
            continue
        p = df[col]
        # Level lags
        for lag in [1, 2, 4, 48, 96, 336]:
            new_cols[f"{col}_lag_{lag}"] = p.shift(lag).astype(np.float32)
        # Rolling means
        for w in [4, 24, 48]:
            new_cols[f"{col}_rmean_{w}"] = p.rolling(w).mean().astype(np.float32)
        # Recent spike count in neighbour region
        new_cols[f"{col}_spike_48"] = (p >= 300).rolling(48).sum().astype(np.float32)
        # arcsinh-transformed lags (handles negatives + compresses spikes)
        ap = np.arcsinh(p / PRICE_TRANSFORM_SCALE)
        for lag in [1, 48, 336]:
            new_cols[f"{col}_asinh_lag_{lag}"] = ap.shift(lag).astype(np.float32)
        # Spread vs NSW price (interconnector pressure direction)
        spread = (p - df["price"]).astype(np.float32)
        new_cols[f"{col}_spread"]      = spread
        new_cols[f"{col}_spread_lag1"] = spread.shift(1).astype(np.float32)
        # Rolling max — SA extreme spikes are a strong warning signal
        new_cols[f"{col}_rmax_48"]  = p.rolling(48).max().astype(np.float32)
        new_cols[f"{col}_rmax_336"] = p.rolling(336).max().astype(np.float32)

    # Multi-region spike co-occurrence: all three neighbours elevated simultaneously
    regions_present = [c for c in ["qld_price", "vic_price", "sa_price"] if c in df.columns]
    if len(regions_present) >= 2:
        flags = pd.concat([(df[c] >= 150).astype(np.float32) for c in regions_present], axis=1)
        new_cols["multi_region_spike"] = flags.min(axis=1)  # 1 only if ALL elevated
        new_cols["region_spike_count"] = flags.sum(axis=1).astype(np.float32)  # 0-3

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _add_coal_bid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived from BIDDAYOFFER_D declared maximum availability for
    NSW coal generators — the closest available substitute for PDPASA
    scheduled-outage data.

    MAXAVAIL is submitted by generators before 12:30 the day prior.
    A low value signals a unit is booked off for scheduled maintenance.
    No-op if coal_declared_avail is absent or all-NaN.
    """
    df = df.copy()
    if "coal_declared_avail" not in df.columns or df["coal_declared_avail"].isna().all():
        return df

    cda = df["coal_declared_avail"]

    # Direct declared availability signal
    df["coal_declared_avail"] = cda.astype(np.float32)

    # Ratio: declared vs actual dispatched coal (deviation flags maintenance)
    if "coal_mw" in df.columns:
        df["coal_bid_vs_actual"]     = (cda - df["coal_mw"]).clip(-5000, 5000).astype(np.float32)
        df["coal_bid_util"]          = (df["coal_mw"] / (cda + 1)).clip(0, 1.5).astype(np.float32)

    # Rolling trend: is declared availability declining? (more units coming offline)
    df["coal_bid_rmean_48"]      = cda.rolling(48).mean().astype(np.float32)
    df["coal_bid_rmin_48"]       = cda.rolling(48).min().astype(np.float32)
    df["coal_bid_trend_48"]      = (cda - cda.shift(48)).astype(np.float32)   # 24h change
    df["coal_bid_trend_96"]      = (cda - cda.shift(96)).astype(np.float32)   # 48h change

    # Low declared availability flag (major maintenance period)
    df["coal_bid_low"]           = (cda < NSW_COAL_MAX_MW * 0.65).astype(np.float32)
    df["coal_bid_low_count_48"]  = df["coal_bid_low"].rolling(48).sum().astype(np.float32)

    # Lags so the model can see the recent bid history
    for lag in [1, 2, 48, 96, 336]:
        df[f"coal_bid_lag_{lag}"] = cda.shift(lag).astype(np.float32)

    return df


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temperature-based features from Open-Meteo historical data.

    Temperature is a top-3 driver of NSW electricity demand:
      - Summer heatwaves -> AC load surge -> demand/price spikes
      - Winter cold snaps -> heating load -> evening peak amplification
      - Solar output is correlated with warm, clear days

    Two forward-looking signals are included:
      - temp_t24: actual temperature 24h ahead (proxy for the real BOM
        24h-ahead forecast; at inference time replace with actual forecast)
      - hdd_t24 / cdd_t24: degree features at forecast horizon

    No-op if temp_sydney is absent or all-NaN.
    """
    df = df.copy()
    if "temp_sydney" not in df.columns or df["temp_sydney"].isna().all():
        return df

    temp = df["temp_sydney"]
    BASE = _BASE_TEMP_C

    # ---- Current temperature context (known at forecast time) ----
    df["temp_sydney"] = temp.astype(np.float32)
    df["hdd"]  = (BASE - temp).clip(lower=0).astype(np.float32)
    df["cdd"]  = (temp - BASE).clip(lower=0).astype(np.float32)
    # Quadratic terms capture non-linear demand response at extremes
    df["cdd_sq"] = (df["cdd"] ** 2).astype(np.float32)
    df["hdd_sq"] = (df["hdd"] ** 2).astype(np.float32)

    # Daily pattern
    df["temp_rmax_48"] = temp.rolling(48, min_periods=1).max().astype(np.float32)
    df["temp_rmin_48"] = temp.rolling(48, min_periods=1).min().astype(np.float32)
    df["temp_range_48"] = (df["temp_rmax_48"] - df["temp_rmin_48"]).astype(np.float32)

    # Regime flags
    df["heatwave_flag"] = (temp.rolling(96, min_periods=1).max() > 35).astype(np.float32)
    df["cold_flag"]     = (temp.rolling(48, min_periods=1).min() < 5).astype(np.float32)

    # Temperature change
    df["temp_chg_24h"] = temp.diff(48).astype(np.float32)

    # Historical lags for context
    for lag in [1, 2, 48, 96, 336]:
        df[f"temp_lag_{lag}"] = temp.shift(lag).astype(np.float32)

    # ---- 24h-ahead temperature (the key forward signal for next-day demand) ----
    # In training: actual temperature at T+24h (near-perfect proxy for BOM forecast)
    # In inference: replace with actual BOM 24h-ahead temperature forecast
    temp_t24 = temp.shift(-config.FORECAST_HORIZON)
    df["temp_t24"]    = temp_t24.astype(np.float32)
    df["hdd_t24"]     = (BASE - temp_t24).clip(lower=0).astype(np.float32)
    df["cdd_t24"]     = (temp_t24 - BASE).clip(lower=0).astype(np.float32)
    df["cdd_t24_sq"]  = (df["cdd_t24"] ** 2).astype(np.float32)
    df["hdd_t24_sq"]  = (df["hdd_t24"] ** 2).astype(np.float32)
    df["heatwave_t24"] = (temp_t24 > 35).astype(np.float32)
    df["cold_t24"]     = (temp_t24 < 5).astype(np.float32)
    df["temp_chg_t24_vs_now"] = (temp_t24 - temp).astype(np.float32)

    # ---- Newcastle temperature (Hunter Valley industrial load) ----
    if "temp_newcastle" in df.columns and not df["temp_newcastle"].isna().all():
        tn = df["temp_newcastle"]
        df["temp_newcastle"]   = tn.astype(np.float32)
        df["temp_nc_t24"]      = tn.shift(-config.FORECAST_HORIZON).astype(np.float32)
        df["temp_nc_cdd"]      = (tn - BASE).clip(lower=0).astype(np.float32)
        df["temp_nc_hdd"]      = (BASE - tn).clip(lower=0).astype(np.float32)
        df["temp_nc_cdd_t24"]  = ((tn.shift(-config.FORECAST_HORIZON) - BASE).clip(lower=0)).astype(np.float32)

    # ---- Feels-like temperature (thermal comfort proxy — better than dry temp) ----
    if "feelslike_sydney" in df.columns and not df["feelslike_sydney"].isna().all():
        fl = df["feelslike_sydney"]
        df["feelslike_cdd"]      = (fl - BASE).clip(lower=0).astype(np.float32)
        df["feelslike_hdd"]      = (BASE - fl).clip(lower=0).astype(np.float32)
        df["feelslike_cdd_sq"]   = (df["feelslike_cdd"] ** 2).astype(np.float32)
        fl_t24 = fl.shift(-config.FORECAST_HORIZON)
        df["feelslike_t24"]      = fl_t24.astype(np.float32)
        df["feelslike_cdd_t24"]  = (fl_t24 - BASE).clip(lower=0).astype(np.float32)
        df["feelslike_hdd_t24"]  = (BASE - fl_t24).clip(lower=0).astype(np.float32)
        df["feelslike_heatwave"] = (fl.rolling(96, min_periods=1).max() > 38).astype(np.float32)

    # ---- Humidity and dew point (amplify cooling/heating demand at extremes) ----
    if "humidity_sydney" in df.columns and not df["humidity_sydney"].isna().all():
        hu = df["humidity_sydney"]
        df["humidity_t24"]      = hu.shift(-config.FORECAST_HORIZON).astype(np.float32)
        df["humidity_rmean_48"] = hu.rolling(48, min_periods=1).mean().astype(np.float32)
        df["humid_high_flag"]   = (hu > 80).astype(np.float32)
        # Heat index proxy: cooling demand × humidity (amplification at high RH)
        if "feelslike_cdd" in df.columns:
            df["humidity_x_cdd"] = (df["feelslike_cdd"] * hu / 100.0).astype(np.float32)

    if "dew_sydney" in df.columns and not df["dew_sydney"].isna().all():
        dw = df["dew_sydney"]
        df["dew_t24"]         = dw.shift(-config.FORECAST_HORIZON).astype(np.float32)
        # High dew point (>18°C) → humid sultry conditions → more AC demand
        df["dew_high_flag"]   = (dw > 18).astype(np.float32)
        df["dew_t24_high"]    = (dw.shift(-config.FORECAST_HORIZON) > 18).astype(np.float32)

    # ---- Cross-region temperatures (neighboring state demand context) ----
    # QLD demand → QLD price → QNI interconnector pressure on NSW
    if "temp_brisbane" in df.columns and not df["temp_brisbane"].isna().all():
        tb = df["temp_brisbane"]
        df["cdd_brisbane"]      = (tb - BASE).clip(lower=0).astype(np.float32)
        df["hdd_brisbane"]      = (BASE - tb).clip(lower=0).astype(np.float32)
        df["temp_t24_brisbane"] = tb.shift(-config.FORECAST_HORIZON).astype(np.float32)
        df["cdd_t24_brisbane"]  = ((tb.shift(-config.FORECAST_HORIZON) - BASE).clip(lower=0)).astype(np.float32)

    # VIC demand → VIC price → VNI/Heywood interconnector pressure on NSW
    if "temp_melbourne" in df.columns and not df["temp_melbourne"].isna().all():
        tm = df["temp_melbourne"]
        df["cdd_melbourne"]      = (tm - BASE).clip(lower=0).astype(np.float32)
        df["hdd_melbourne"]      = (BASE - tm).clip(lower=0).astype(np.float32)
        df["temp_t24_melbourne"] = tm.shift(-config.FORECAST_HORIZON).astype(np.float32)
        df["cdd_t24_melbourne"]  = ((tm.shift(-config.FORECAST_HORIZON) - BASE).clip(lower=0)).astype(np.float32)

    # SA demand — SA-NSW path is indirect (via VIC) but useful in extreme events
    if "temp_adelaide" in df.columns and not df["temp_adelaide"].isna().all():
        ta = df["temp_adelaide"]
        df["cdd_adelaide"]      = (ta - BASE).clip(lower=0).astype(np.float32)
        df["hdd_adelaide"]      = (BASE - ta).clip(lower=0).astype(np.float32)
        df["temp_t24_adelaide"] = ta.shift(-config.FORECAST_HORIZON).astype(np.float32)

    return df


def _add_pasa_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived from PDPASA_REGIONSOLUTION 24h-ahead availability forecast.

    pasa_avail_gen = AEMO forecast of total available generation 24h from now.
    This captures scheduled outages — a generator booked off for maintenance
    tomorrow — which is the primary driver of predictable price spikes.

    No-op if pasa_avail_gen is absent or all-NaN.
    """
    df = df.copy()
    if "pasa_avail_gen" not in df.columns or df["pasa_avail_gen"].isna().all():
        return df

    pa   = df["pasa_avail_gen"]
    dem50 = df["pasa_demand50"] if "pasa_demand50" in df.columns else pd.Series(np.nan, index=df.index)

    # Direct 24h-ahead signals
    df["pasa_avail_gen"]  = pa.astype(np.float32)
    df["pasa_demand50"]   = dem50.astype(np.float32)

    # Forward reserve margin: spare capacity AEMO expects 24h from now
    df["pasa_margin"]     = (pa - dem50).clip(-5000, 10000).astype(np.float32)
    df["pasa_util"]       = (dem50 / (pa + 1)).clip(0, 2).astype(np.float32)

    # Deviation from current available gen (measures scheduled outage delta)
    if "avail_gen" in df.columns:
        df["pasa_vs_now"]     = (pa - df["avail_gen"]).clip(-8000, 8000).astype(np.float32)
        df["pasa_vs_now_pct"] = ((pa - df["avail_gen"]) / (df["avail_gen"] + 1)).clip(-1, 1).astype(np.float32)

    # Demand surprise vs PASA50: how much does actual demand deviate from 24h-ahead
    # PASA demand forecast? Positive = demand exceeded the forecast = supply tighter.
    if "demand" in df.columns and not dem50.isna().all():
        d_vs_pasa = (df["demand"] - dem50).clip(-3000, 3000)
        df["demand_vs_pasa50"]      = d_vs_pasa.astype(np.float32)
        df["demand_vs_pasa50_abs"]  = d_vs_pasa.abs().astype(np.float32)
        df["demand_vs_pasa50_rm48"] = d_vs_pasa.rolling(48, min_periods=1).mean().astype(np.float32)
        # PASA utilisation: how loaded will system be 24h from now vs PASA forecast?
        df["pasa_demand_ratio"] = (df["demand"] / (dem50 + 1)).clip(0.5, 2.0).astype(np.float32)

    # Rolling trend: is PASA forecast availability declining over next day?
    # min_periods=1 ensures partial windows produce a value rather than NaN,
    # which is critical when pasa_avail_gen has data gaps (ffill is limited).
    df["pasa_avail_rmean_4"]  = pa.rolling(4,  min_periods=1).mean().astype(np.float32)
    df["pasa_avail_rmean_48"] = pa.rolling(48, min_periods=1).mean().astype(np.float32)
    df["pasa_avail_rmin_48"]  = pa.rolling(48, min_periods=1).min().astype(np.float32)
    df["pasa_avail_trend_48"] = (pa - pa.shift(48)).astype(np.float32)  # 24h trend in PASA forecast

    # Low-availability flag (scheduled outage bringing generation below safe floor)
    df["pasa_avail_low"]          = (pa < 8000).astype(np.float32)
    df["pasa_avail_low_count_48"] = df["pasa_avail_low"].rolling(48, min_periods=1).sum().astype(np.float32)

    # Reserve condition code from PDPASA (0=normal, >0=reserve concern)
    if "pasa_reserve_cond" in df.columns:
        df["pasa_reserve_cond"] = df["pasa_reserve_cond"].fillna(0).astype(np.float32)
        df["pasa_low_reserve"]  = (df["pasa_reserve_cond"] > 0).astype(np.float32)

    # Fill any remaining NaN in pasa-derived features with 0 so data gaps
    # don't silently drop rows in the downstream dropna step.
    _pasa_derived = [
        "pasa_margin", "pasa_util", "pasa_vs_now", "pasa_vs_now_pct",
        "pasa_avail_rmean_4", "pasa_avail_rmean_48", "pasa_avail_rmin_48",
        "pasa_avail_trend_48", "pasa_avail_low", "pasa_avail_low_count_48",
        "pasa_low_reserve",
    ]
    for col in _pasa_derived:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _add_gas_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gas price proxy features derived from Henry Hub daily spot price.

    Gas is the marginal fuel in NSW during shoulder and peak periods.
    Movements in the gas price shift the short-run marginal cost of
    open-cycle and combined-cycle gas turbines, directly lifting or
    suppressing the price stack.  Key signals:

      gas_price_hh      — current daily price (USD/mmbtu, broadcast to 30-min)
      gas_srmc_proxy    — approximate SRMC of OCGT (gas_price * 10 USD/MWh)
      gas_lags          — 1d / 7d / 30d lags for short/medium-term trend     gas_roc_7d/30d    — rate-of-change (week-on-week, month-on-month)
      gas_rmean_7d/30d  — rolling averages (trend regime)
      gas_regime_high   — binary flag for top-quartile gas price periods
      gas_spread_vs_30d — deviation from 30-day mean (spike relative to trend)

    No-op if gas_price_hh is absent or all-NaN.
    """
    df = df.copy()
    OCGT_HEAT_RATE = 10.0       # approximate GJ/MWh (or mmbtu/MWh) for OCGT
    HIGH_GAS_QUANTILE = 0.75    # "high gas price" regime threshold

    if "gas_price_hh" not in df.columns or df["gas_price_hh"].isna().all():
        return df

    g = df["gas_price_hh"]
    df["gas_price_hh"] = g.astype(np.float32)

    # Short-run marginal cost proxy: gas_price × heat-rate gives USD/MWh
    # (units differ from AUD/MWh, but the model learns the scaling)
    df["gas_srmc_proxy"] = (g * OCGT_HEAT_RATE).astype(np.float32)

    # Lags (daily → 48 intervals, weekly → 336, monthly → 1440)
    for lag, name in [(48, "1d"), (96, "2d"), (336, "7d"), (1440, "30d")]:
        df[f"gas_lag_{name}"] = g.shift(lag).astype(np.float32)

    # Rolling averages — regime filter
    g_rmean_7  = g.rolling(336,  min_periods=1).mean()
    g_rmean_30 = g.rolling(1440, min_periods=1).mean()
    df["gas_rmean_7d"]  = g_rmean_7.astype(np.float32)
    df["gas_rmean_30d"] = g_rmean_30.astype(np.float32)

    # Rate of change: current vs 7-day-ago and 30-day-ago rolling mean
    df["gas_roc_7d"]  = ((g - g.shift(336))  / (g.shift(336)  + 0.01)).clip(-2, 2).astype(np.float32)
    df["gas_roc_30d"] = ((g - g.shift(1440)) / (g.shift(1440) + 0.01)).clip(-2, 2).astype(np.float32)

    # Deviation from 30-day trend (spike relative to recent baseline)
    df["gas_spread_vs_30d"] = (g - g_rmean_30).astype(np.float32)

    # Gas price percentile rank within recent windows (is gas expensive vs norms?)
    df["gas_pctrank_336"]  = g.rolling(336,  min_periods=24).rank(pct=True).astype(np.float32)
    df["gas_pctrank_1440"] = g.rolling(1440, min_periods=336).rank(pct=True).astype(np.float32)

    # High-gas-price regime flag (set when price is in top quartile of last 6 months)
    g_quantile = g.rolling(8640, min_periods=336).quantile(HIGH_GAS_QUANTILE)
    df["gas_regime_high"] = (g > g_quantile).astype(np.float32)

    return df


def _add_predispatch_step_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived from the multi-step predispatch and PDPASA columns
    produced by Dataset/7_predispatch_source.py / Dataset/8_pdpasa_source.py.

    Column naming convention (set by prepare_region_data):
      predispatch_rrp_h{k}[T]  = AEMO pre-dispatch RRP forecast for T+k
                                  from the run issued at T
      pasa_avail_gen_h{k}[T]   = PDPASA available-generation forecast for T+k
      pasa_demand50_h{k}[T]    = PDPASA 50th-percentile demand forecast for T+k

    For each prediction step h in [config.FORECAST_GAP,
    config.FORECAST_GAP + config.FORECAST_HORIZON), this function adds:
      pd_step_h{h}          — raw RRP forecast ($/MWh) for that step
      pd_step_h{h}_asinh    — arcsinh(forecast / 100)
      pd_step_h{h}_spike    — binary: forecast > $300
      pasa_avail_h{h}       — PDPASA available generation for that step (if present)
      pasa_margin_h{h}      — PDPASA spare capacity (avail - demand50) for that step

    No-op if no predispatch_rrp_h{k} columns are present.
    """
    df = df.copy()
    gap     = config.FORECAST_GAP
    horizon = config.FORECAST_HORIZON
    SPIKE_THR = 300.0

    for h in range(gap, gap + horizon):
        rrp_col = f"predispatch_rrp_h{h}"
        if rrp_col not in df.columns or df[rrp_col].isna().all():
            continue

        p = df[rrp_col].astype("float64")
        df[f"pd_step_h{h}"]       = p.astype(np.float32)
        df[f"pd_step_h{h}_asinh"] = np.arcsinh(p / 100.0).astype(np.float32)
        df[f"pd_step_h{h}_spike"] = (p > SPIKE_THR).astype(np.float32)

    # PDPASA step columns for PASA available generation and margin
    for h in range(gap, gap + horizon):
        avail_col = f"pasa_avail_gen_h{h}"
        dem50_col = f"pasa_demand50_h{h}"
        if avail_col in df.columns and not df[avail_col].isna().all():
            pa = df[avail_col].astype("float64")
            df[f"pasa_avail_h{h}"] = pa.astype(np.float32)
            if dem50_col in df.columns and not df[dem50_col].isna().all():
                d50 = df[dem50_col].astype("float64")
                df[f"pasa_margin_h{h}"] = (pa - d50).clip(-5000, 10000).astype(np.float32)

    return df


def _add_predispatch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy hook kept for compatibility; step-horizon features are added separately."""
    return df


def _add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Solar radiation features from local weather CSV files.

    Solar radiation is the dominant supply-side signal the codebase was
    previously missing:
      - High midday solar (>500 W/m²) suppresses net demand and spot price
      - The duck curve effect: price rises sharply in the evening as solar
        falls off and demand peaks from heating/cooking
      - A high-solar-tomorrow forecast reduces the incentive to generate
        thermally tonight, creating anticipatory price moves

    The 24h-ahead solar signal (shift(-config.FORECAST_HORIZON)) acts as a proxy
    for AEMO's own solar generation forecast at the target interval.
    At inference time, replace with actual BOM clear-sky / NWP forecast.

    No-op if solar_rad_sydney is absent or all-NaN.
    """
    df = df.copy()

    if "solar_rad_sydney" not in df.columns or df["solar_rad_sydney"].isna().all():
        return df

    sol = df["solar_rad_sydney"]

    # Compressed value — log1p since solar_rad >= 0, removes outlier skew
    df["solar_rad_log1p"]     = np.log1p(sol).astype(np.float32)

    # 24h-ahead solar (key forward signal: what will solar generation look like
    # at the actual forecast horizon?)
    sol_t24 = sol.shift(-config.FORECAST_HORIZON)
    df["solar_rad_t24"]       = sol_t24.astype(np.float32)
    df["solar_rad_t24_log1p"] = np.log1p(sol_t24).astype(np.float32)

    # Historical lags
    df["solar_rad_lag_48"]    = sol.shift(48).astype(np.float32)    # yesterday
    df["solar_rad_lag_96"]    = sol.shift(96).astype(np.float32)    # 2 days ago
    df["solar_rad_lag_336"]   = sol.shift(336).astype(np.float32)   # 1 week ago

    # Rolling stats (daytime solar context)
    df["solar_rad_rmean_48"]  = sol.rolling(48, min_periods=1).mean().astype(np.float32)
    df["solar_rad_rmax_48"]   = sol.rolling(48, min_periods=1).max().astype(np.float32)
    df["solar_rad_rmean_336"] = sol.rolling(336, min_periods=1).mean().astype(np.float32)

    # Daily cumulated solar proxy (sum of last 24 intervals ≈ 12h window)
    df["solar_cumday_24h"]    = sol.rolling(24, min_periods=1).sum().astype(np.float32)

    # Regime flags
    df["solar_high_flag"]     = (sol > 500).astype(np.float32)
    df["solar_t24_high_flag"] = (sol_t24 > 500).astype(np.float32)

    # Ramp: tomorrow's solar minus today's (rising solar day → price relief ahead)
    df["solar_ramp_t24"]      = (sol_t24 - sol).astype(np.float32)

    # Interaction: high forecast solar AND tight supply → conflict signal
    if "supply_stress" in df.columns:
        df["solar_t24_x_stress"] = (
            np.log1p(sol_t24) * df["supply_stress"].astype("float64")
        ).clip(-20, 20).astype(np.float32)

    # --- QLD (Brisbane) solar — affects QLD-NSW interconnector dispatch ---
    if "solar_rad_brisbane" in df.columns and not df["solar_rad_brisbane"].isna().all():
        sol_b = df["solar_rad_brisbane"]
        df["solar_rad_brisbane_t24"]      = sol_b.shift(-config.FORECAST_HORIZON).astype(np.float32)
        df["solar_rad_brisbane_rmean_48"] = sol_b.rolling(48, min_periods=1).mean().astype(np.float32)

    # --- VIC (Melbourne) solar — affects VIC-NSW interconnector dispatch ---
    if "solar_rad_melbourne" in df.columns and not df["solar_rad_melbourne"].isna().all():
        sol_m = df["solar_rad_melbourne"]
        df["solar_rad_melbourne_t24"] = sol_m.shift(-config.FORECAST_HORIZON).astype(np.float32)

    return df


def _add_wind_cloud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wind speed and cloud cover features from local weather CSV files.

    Wind speed is a proxy for wind generation capacity:
      - NSW wind generation (mainly Capital and Sapphire wind farms) is modest
        but meaningful at the margin; Queensland wind is larger
      - High wind forecast → more renewable dispatch → lower thermal price

    Cloud cover affects solar generation and is the primary modulator of the
    solar radiation signal.

    No-op if windspeed_sydney is absent or all-NaN.
    """
    df = df.copy()

    # --- NSW wind speed ---
    if "windspeed_sydney" in df.columns and not df["windspeed_sydney"].isna().all():
        ws = df["windspeed_sydney"]

        df["windspeed_t24"]        = ws.shift(-config.FORECAST_HORIZON).astype(np.float32)
        df["windspeed_lag_48"]     = ws.shift(48).astype(np.float32)
        df["windspeed_rmean_48"]   = ws.rolling(48, min_periods=1).mean().astype(np.float32)
        df["windspeed_rmean_336"]  = ws.rolling(336, min_periods=1).mean().astype(np.float32)
        df["windspeed_high_flag"]  = (ws > 30).astype(np.float32)  # strong wind gen
        df["windspeed_t24_high"]   = (ws.shift(-config.FORECAST_HORIZON) > 30).astype(np.float32)

    # --- QLD wind speed ---
    if "windspeed_brisbane" in df.columns and not df["windspeed_brisbane"].isna().all():
        ws_b = df["windspeed_brisbane"]
        df["windspeed_brisbane_t24"]      = ws_b.shift(-config.FORECAST_HORIZON).astype(np.float32)
        df["windspeed_brisbane_rmean_48"] = ws_b.rolling(48, min_periods=1).mean().astype(np.float32)

    # --- NSW cloud cover ---
    if "cloudcover_sydney" in df.columns and not df["cloudcover_sydney"].isna().all():
        cc = df["cloudcover_sydney"]
        df["cloudcover_t24"]       = cc.shift(-config.FORECAST_HORIZON).astype(np.float32)
        df["cloudcover_lag_48"]    = cc.shift(48).astype(np.float32)
        df["cloudcover_rmean_48"]  = cc.rolling(48, min_periods=1).mean().astype(np.float32)
        df["clear_sky_flag"]       = (cc < 20).astype(np.float32)
        df["clear_sky_t24_flag"]   = (cc.shift(-config.FORECAST_HORIZON) < 20).astype(np.float32)
        df["overcast_t24_flag"]    = (cc.shift(-config.FORECAST_HORIZON) > 80).astype(np.float32)

        # Cloud-solar interaction: tomorrow cloud cover modulates solar forecast
        if "solar_rad_t24" in df.columns:
            cc_t24 = cc.shift(-config.FORECAST_HORIZON) / 100.0
            df["solar_x_cloud_t24"] = (
                df["solar_rad_t24"] * (1.0 - cc_t24.clip(0, 1))
            ).astype(np.float32)

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-motivated interaction terms for electricity price forecasting.

    Based on insights from Lago et al. (2021) "Forecasting day-ahead electricity
    prices: A review of state-of-the-art algorithms" and Ziel & Weron (2018)
    "Day-ahead electricity price forecasting with high-dimensional structures".
    Cross-feature products expose non-linear market dynamics that additive
    decision-tree splits cannot represent in a single feature independently.
    """
    df = df.copy()

    # Supply-stress × lagged price: when capacity headroom is tight AND recent
    # prices were high, forward prices spike non-linearly.
    if "supply_stress" in df.columns and "price_lag_1" in df.columns:
        df["supply_stress_x_price_l1"] = (
            df["supply_stress"].fillna(0.0) * df["price_lag_1"].fillna(0.0)
        ).astype(np.float32)

    # Predispatch RRP × supply-stress²: tighter than supply_stress alone because
    # it captures the dispatch-stack non-linearity at high loading.
    if "predispatch_rrp_1" in df.columns and "supply_stress" in df.columns:
        df["pd_rrp_x_supply_stress2"] = (
            df["predispatch_rrp_1"].fillna(0.0)
            * df["supply_stress"].fillna(0.0) ** 2
        ).astype(np.float32)

    # Coal outage × high-demand flag: demand-side amplification when thermal
    # capacity is absent raises prices faster than either alone.
    if "coal_outage_mw" in df.columns and "demand_lag_1" in df.columns:
        demand_75 = df["demand_lag_1"].quantile(0.75)
        df["coal_outage_x_high_demand"] = (
            df["coal_outage_mw"].fillna(0.0)
            * (df["demand_lag_1"] > demand_75).astype(np.float32)
        ).astype(np.float32)

    # Hot-day peak: temperature during afternoon peak is the primary driver of
    # air-conditioning load in NEM regions; interaction reveals the critical
    # temperature threshold behaviour.
    temp_col = next(
        (c for c in ("temp_sydney", "temp_brisbane", "temp_melbourne", "temp_adelaide")
         if c in df.columns),
        None,
    )
    if temp_col is not None and "is_peak" in df.columns:
        df["hot_day_peak"] = (
            np.maximum(df[temp_col].fillna(df[temp_col].median()) - 25.0, 0.0)
            * df["is_peak"]
        ).astype(np.float32)

    # Cooling-degree-days × demand lag: represents load-weighted temperature.
    if "cooling_deg_day" in df.columns and "demand_lag_1" in df.columns:
        df["cdd_x_demand"] = (
            df["cooling_deg_day"].fillna(0.0) * df["demand_lag_1"].fillna(0.0)
        ).astype(np.float32)

    # Spike-history × predispatch forecast: recent spike count amplified by
    # a high predispatch RRP is a strong forward-spike signal.
    if "spike_count_24h" in df.columns and "predispatch_rrp_1" in df.columns:
        df["spike_history_x_pd_forecast"] = (
            df["spike_count_24h"].fillna(0.0)
            * np.log1p(np.maximum(df["predispatch_rrp_1"].fillna(0.0), 0.0))
        ).astype(np.float32)

    # Forward-squeeze: predispatch RRP minus lagged spot; positive squeeze
    # (forward > spot) signals an anticipated tightening.
    if "predispatch_rrp_1" in df.columns and "price_lag_1" in df.columns:
        df["forward_squeeze"] = (
            df["predispatch_rrp_1"].fillna(0.0) - df["price_lag_1"].fillna(0.0)
        ).astype(np.float32)

    return df


def build_features(
    df: pd.DataFrame,
    *,
    gap: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add all feature columns to `df` and create the supervised target columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw 30-min series with columns: price, demand, avail_gen, interchange,
        demand_forecast, qld_price, vic_price.
    gap : int
        Lead-time in 30-min intervals before the first forecast step.
    horizon : int
        Number of consecutive 30-min steps to forecast.

    Returns
    -------
    df_full : pd.DataFrame
        Superset dataset containing raw source columns, all engineered columns,
        and all supervised target columns before row filtering.
    df_selected : pd.DataFrame
        Train-ready dataset containing only model input columns and target
        columns, with rows dropped where required features/targets are NaN.
    feature_cols : list[str]
        Ordered list of predictor column names (excludes price, demand, target)
    """
    config.FORECAST_GAP     = gap
    config.FORECAST_HORIZON = horizon
    df = _add_time_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_regime_features(df)
    df = _add_time_since_spike_features(df)
    df = _add_system_features(df)
    df = _add_arcsinh_price_lags(df)
    df = _add_spike_predictors(df)
    df = _add_region_features(df)
    df = _add_generation_features(df)
    df = _add_coal_bid_features(df)
    df = _add_pasa_features(df)
    df = _add_weather_features(df)
    df = _add_solar_features(df)
    df = _add_wind_cloud_features(df)
    df = _add_gas_features(df)
    df = _add_predispatch_step_features(df)
    df = _add_long_range_features(df)
    df = _add_interaction_features(df)

    # Supervised targets: one per step from config.FORECAST_GAP to config.FORECAST_GAP + config.FORECAST_HORIZON
    for h in range(config.FORECAST_GAP, config.FORECAST_GAP + config.FORECAST_HORIZON):
        df[f"target_h{h}"] = df["price"].shift(-h).astype(np.float32)

    # Exclude raw source columns from feature list
    _non_feature = {"price", "demand", "avail_gen", "interchange",
                    "demand_forecast", "qld_price", "vic_price", "sa_price",
                    "dispatch_gen", "coal_mw", "coal_declared_avail",
                    "pasa_avail_gen", "pasa_demand50", "pasa_reserve_cond",
                    "temp_sydney", "temp_newcastle",
                    "feelslike_sydney", "humidity_sydney", "dew_sydney",
                    "solar_rad_sydney", "windspeed_sydney", "cloudcover_sydney",
                    "temp_brisbane", "solar_rad_brisbane", "windspeed_brisbane",
                    "temp_melbourne", "solar_rad_melbourne",
                    "temp_adelaide",
                    "gas_price_hh",
                    # NSW source columns (remapped to canonical names by main.py)
                    "nsw_price", "demand_nsw", "avail_gen_nsw", "interchange_nsw",
                    "demand_forecast_nsw", "dispatch_gen_nsw", "coal_mw_nsw",
                    "pasa_avail_gen_nsw", "pasa_demand50_nsw", "pasa_reserve_cond_nsw"}
    # Exclude raw multi-step predispatch/PASA source columns (derived features
    # pd_step_h{h}* and pasa_avail_h{h}* are the actual features).
    _non_feature |= {c for c in df.columns
                     if c.startswith("predispatch_rrp_h") or
                     c.startswith(("pasa_avail_gen_h", "pasa_demand50_h", "pasa_reserve_cond_h"))}
    _non_feature |= {f"target_h{h}" for h in range(config.FORECAST_GAP, config.FORECAST_GAP + config.FORECAST_HORIZON)}
    feature_cols = [c for c in df.columns if c not in _non_feature]

    # Drop rows where essential features or target are NaN.
    # Annual / YoY lag features are excluded from this requirement — they are
    # NaN for the first ~1 year of data (ANNUAL_LAG = 17,532 intervals) and
    # LightGBM handles NaN natively via NA splits.  Excluding them keeps ~14%
    # more training rows (≈17 K rows) that would otherwise be wasted.
    # pd_rrp_* / pd_step_* features: AEMO predispatch data has ~22% NaN gaps;
    # LightGBM handles NaN via its native NA split direction.
    _long_lookback = {c for c in feature_cols
                      if "annual" in c or "yoy" in c
                      or c.startswith(("pd_rrp", "pd_qld", "pd_vic", "pd_sa", "pd_reg",
                                       "pd_step", "pasa_avail_h", "pasa_margin_h"))}
    _last_target   = f"target_h{config.FORECAST_GAP + config.FORECAST_HORIZON - 1}"
    required_cols  = [c for c in feature_cols if c not in _long_lookback] + [_last_target]
    df_selected = df.dropna(subset=required_cols).copy()
    target_cols = [f"target_h{h}" for h in range(config.FORECAST_GAP, config.FORECAST_GAP + config.FORECAST_HORIZON)]
    df_selected = df_selected[feature_cols + target_cols].copy()

    return df, df_selected, feature_cols


def select_features(
    df_full: pd.DataFrame,
    feature_cols: list[str],
    region_tag: str,
    cache_dir: Path,
    *,
    gap: int,
    horizon: int,
    force_rerun: bool = False,
    n_features: int = 200,
    corr_threshold: float = 0.90,
    test_months: int = 12,
    valid_months: int = 6,
) -> list[str]:
    """Select the most informative features for a region via mutual information.

    Results are cached to <cache_dir>/selected_features_<region_tag>.json so
    the expensive MI computation only runs once per region.
    """
    import json
    from sklearn.feature_selection import mutual_info_regression

    config.FORECAST_GAP     = gap
    config.FORECAST_HORIZON = horizon

    cache_file = Path(cache_dir) / f"selected_features_{region_tag}.json"

    if not force_rerun and cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        cols = [c for c in cached if c in df_full.columns]
        if cols:
            print(f"  Loaded {len(cols)} cached features from {cache_file.name}", flush=True)
            return cols

    cutoff_test  = df_full.index[-1] - pd.DateOffset(months=test_months)
    cutoff_valid = cutoff_test - pd.DateOffset(months=valid_months)
    train_df = df_full[df_full.index <= cutoff_valid]

    if train_df.empty:
        print("  Warning: empty train split, falling back to all features", flush=True)
        return feature_cols

    null_frac  = train_df[feature_cols].isna().mean()
    candidates = [c for c in feature_cols if null_frac[c] <= 0.05]
    if len(candidates) < 30:
        candidates = list(feature_cols)

    X = train_df[candidates].fillna(0.0).values.astype(np.float32)

    # Subsample for MI — up to 100 k rows for a more reliable ranking on large datasets.
    rng = np.random.default_rng(42)
    n_mi = min(100_000, len(X))
    mi_idx = rng.choice(len(X), size=n_mi, replace=False)
    mi_idx.sort()
    X_mi = X[mi_idx]

    t1    = f"target_h{config.FORECAST_GAP}"
    tmid  = f"target_h{config.FORECAST_GAP + config.FORECAST_HORIZON // 2}"
    tlast = f"target_h{config.FORECAST_GAP + config.FORECAST_HORIZON - 1}"

    scores = np.zeros(len(candidates), dtype=np.float64)
    for tcol in [t1, tmid, tlast]:
        if tcol not in train_df.columns:
            continue
        y_full = train_df[tcol].fillna(train_df[tcol].median()).values.astype(np.float32)
        y_mi   = y_full[mi_idx]
        scores += mutual_info_regression(X_mi, y_mi, discrete_features=False, n_neighbors=3, random_state=42, n_jobs=-1)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # Greedy correlation dedup: drop features too correlated with already-selected ones.
    # Vectorised: maintain a (n_selected, n_rows) matrix and use matmul for batch correlation.
    X_norm = X - X.mean(axis=0, keepdims=True)
    X_norm /= (X_norm.std(axis=0, keepdims=True) + 1e-8)
    col_to_idx = {c: i for i, c in enumerate(candidates)}
    selected: list[str] = []
    sel_mat: np.ndarray | None = None  # shape (n_selected, n_rows)
    k = min(n_features, len(candidates))

    for col, _score in ranked:
        if len(selected) >= k:
            break
        vec = X_norm[:, col_to_idx[col]]          # (n_rows,)
        if sel_mat is not None:
            corr = np.abs(sel_mat @ vec) / len(vec)  # (n_selected,) — batch dot product
            if corr.max() > corr_threshold:
                continue
            sel_mat = np.vstack([sel_mat, vec[np.newaxis, :]])
        else:
            sel_mat = vec[np.newaxis, :]
        selected.append(col)

    if not selected:
        selected = feature_cols[:k]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"  Selected {len(selected)} features (from {len(candidates)}) -> {cache_file.name}", flush=True)
    return selected


_DATASETS_DIR = Path(__file__).resolve().parent.parent / "Dataset"


def save_datasets(
    df_full: pd.DataFrame,
    feature_cols: list[str],
    region_tag: str,
    *,
    gap: int,
    horizon: int,
) -> None:
    """Persist the full engineered dataset and the selected-feature subset as Parquet.

    Parquet with snappy compression is typically 10-20x smaller than CSV for
    this dataset, preventing "no space left on device" errors on small disks.
    """
    config.FORECAST_GAP     = gap
    config.FORECAST_HORIZON = horizon
    target_cols = [f"target_h{h}" for h in range(gap, gap + horizon)]
    keep_cols   = [c for c in feature_cols + target_cols if c in df_full.columns]

    out_full = _DATASETS_DIR / "all_engineered" / f"{region_tag}.parquet"
    out_sel  = _DATASETS_DIR / "selected"       / f"{region_tag}.parquet"

    # Cast int8 columns to int16 to avoid Parquet's lack of int8 support.
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        int8_cols = [c for c in df.columns if df[c].dtype == np.int8]
        if int8_cols:
            df = df.copy()
            df[int8_cols] = df[int8_cols].astype(np.int16)
        return df

    _prep(df_full).to_parquet(out_full, compression="snappy", index=True)
    _prep(
        df_full.dropna(subset=[f"target_h{config.FORECAST_GAP + config.FORECAST_HORIZON - 1}"])[keep_cols]
    ).to_parquet(out_sel, compression="snappy", index=True)

    sizes = f"{out_full.stat().st_size / 1e6:.1f} MB / {out_sel.stat().st_size / 1e6:.1f} MB"
    print(f"  Saved datasets (all / selected): {sizes}", flush=True)

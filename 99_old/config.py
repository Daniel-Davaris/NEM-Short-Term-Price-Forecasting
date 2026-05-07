"""
config.py  –  Shared pipeline constants and per-region forecast configuration.

FORECAST_GAP and FORECAST_HORIZON are mutable module-level attributes set
at the entry point of each pipeline function (build_features, train_seq2seq,
etc.) via explicit gap/horizon keyword arguments from run_region().
They carry no meaningful default — they must always be set before use.
"""

# ---------------------------------------------------------------------------
# Global data window — must span all per-region data_start/data_end values.
# data_loader.py uses these to define the outer fetch/cache boundary.
# ---------------------------------------------------------------------------
DATA_START_DATE = "2018/01/01"
DATA_END_DATE   = "2026/01/01"

# FORECAST_GAP and FORECAST_HORIZON are set dynamically by run_region() in
# main.py before every pipeline call — never define them here.


def cfg_tag(gap: int, horizon: int, data_start: str, data_end: str) -> str:
    """Return a compact string encoding the forecast config for use in filenames.

    Examples
    --------
    >>> cfg_tag(24, 48, "2018/01/01", "2024/01/01")
    'g24h48_2018_2024'
    """
    return f"g{gap}h{horizon}_{str(data_start)[:4]}_{str(data_end)[:4]}"


# ---------------------------------------------------------------------------
# Gap/horizon combos — single source of truth used by both main.py and the
# ingest scripts to determine which forecast horizons to fetch and store.
# ---------------------------------------------------------------------------
FORECAST_GAP_HORIZON_COMBOS: list = [
    (1,  16),   # gap=1  (30 min ahead), horizon=16 (8 h window)
    (12, 24),   # gap=12 (6 h ahead),    horizon=24 (12 h window)
]


def get_forecast_step_intervals() -> list:
    """Return the sorted unique set of all prediction-step interval counts
    (in 30-min units) across all FORECAST_GAP_HORIZON_COMBOS.

    Example: combos [(1,16),(12,24)] → steps 1–16 ∪ 12–35 = [1, 2, …, 35]
    These are the horizons the ingest scripts (Dataset/7_predispatch_source.py,
    Dataset/8_pdpasa_source.py) will extract and store as separate columns.
    """
    steps: set = set()
    for gap, horizon in FORECAST_GAP_HORIZON_COMBOS:
        for k in range(gap, gap + horizon):
            steps.add(k)
    return sorted(steps)

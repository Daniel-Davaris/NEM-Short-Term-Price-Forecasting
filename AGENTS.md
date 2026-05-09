# Forecasting — Agent Instructions

NEM Regional Reference Price (RRP) forecaster: 5-minute electricity price predictions for NSW, QLD, VIC, SA using a spike-aware LightGBM ensemble. See [README.md](README.md) for project intent.

## Editing rules (highest priority)

- **Never delete existing code** unless the user explicitly names a function/block to remove. Treat all existing code — including commented-out lines, alternative implementations, and exploratory cells — as intentional.
- **Always merge, never replace.** When adding new behavior, extend the existing code (new branch, new cell, additional column, optional parameter). Do not rewrite a function/cell from scratch when an additive change works.
- **Preserve structure on edits**: keep cell boundaries, existing imports, variable names, and ordering. Add new imports/constants alongside existing ones rather than reorganizing.
- If a change seems to require deletion or a rewrite, stop and ask first.

## Repository layout (data flows top-to-bottom)

Numbered folders are pipeline stages; outputs of stage N feed stage N+1 via parquet files.

| Stage | Folder | Output dir |
|-------|--------|------------|
| 1 | [1_Dataset/](1_Dataset/) — fetch AEMO data (nemosis) | `1_Dataset/Processed_data/` |
| 2 | [2_Features build/](2_Features%20build/) — 12 feature notebooks | `2_Features build/Feature_data/` |
| 3 | [3_Targets build/](3_Targets%20build/) — define price targets per horizon | `3_Targets build/Target_data/` |
| 4 | [4_Features select/](4_Features%20select/) — feature selection | `4_Features select/Selected_features/` |
| 5 | [5_Model/model.py](5_Model/model.py) — multi-horizon LightGBM (base + spike classifier + spike regressor), isotonic calibration, exponential recency weighting | — |

`99_old/` is archived code — do not edit, but consult for reference patterns.

## Environments — two venvs, do not mix

- **`.venv-main`** (Python 3.13) — default for everything: nemosis, scikit-learn, lightgbm, pandas, pyarrow, holidays. Deps: [requirements-main.txt](requirements-main.txt).
- **`.venv-subprocess`** (Python 3.11) — used **only** for nemseer (which requires 3.11). Invoked as a subprocess from main env. Deps: [requirements-subprocess.txt](requirements-subprocess.txt).

Never add a package to a venv other than its dedicated requirements file.

## Conventions

- **Notebooks are the primary code surface.** Only `5_Model/model.py` is a `.py` file. Add new logic as new cells in the relevant numbered notebook; do not extract notebook code into modules unless asked.
- **All data I/O is parquet** (pyarrow engine). Do not introduce CSV reads/writes for derived data.
- **First cell of each notebook** sets CPU parallelization env vars (`NUMEXPR_*`, `OPENBLAS_*`, `MKL_*`, `OMP_*`, pyarrow threads). Preserve this cell verbatim.
- **Cell markers**: `# COMMAND ----------` (configured in `.vscode/settings.json`).
- **Feature patterns to follow when adding features**: arcsinh transforms for heavy-tailed values; lag set `{1, 2, 4, 12, 48, 96, 336}` intervals; rolling means; sin/cos cyclical time encoding; holiday flags; peak/off-peak indicators.
- **Model constants** in `model.py`: `SPIKE_THRESHOLD=150`, `BASE_CLIP_PERCENTILE=97`, `INTERVAL_MINUTES=30`. Don't change without being asked.

## Before you commit changes

- Ran the affected notebook end-to-end (or stated you didn't and why).
- Output parquet still loads in the next stage's first cell.
- No existing cell removed, no existing function deleted.

- Leave spacing / whitespace as it, do not attempt to inline '=' over mutliple lines or create more whitespace

- always make changes directly to the python files, notebooks etc, never provide the code in the chat unless explicidly asked to do so
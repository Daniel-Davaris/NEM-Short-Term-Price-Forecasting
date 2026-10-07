# Price-model workflow

`Data/` is shared by both model families and remains at this level.

## Run order

1. `ML/1_derive_new_targets.ipynb`
2. `ML/2_train_models.ipynb`
3. `ML/3_generate_validation_predictions.ipynb`
4. `ML/4_build_meta_features.ipynb`
5. `ML/5_train_residual_stackers.ipynb`
6. `ML/6_save_final_model.ipynb`
7. `python ML/7_train_value_signal.py`
8. `Heuristic/1_generate_naive_predictions.ipynb`
9. `Heuristic/2_create_operating_protocol.ipynb`
10. `4_1_generate_test_predictions.ipynb`
11. `4_2_evaluate_test_predictions.ipynb`
12. `4_3_export_optimizer_prices.ipynb`

`ML/7_train_value_signal.py` leaves the central residual-stacker forecast
unchanged. It trains a compact decision-focused layer from all eight component
outputs at the midnight operating cadence. Paired high-price and low-price
rankers learn the within-window ordering, convert it to a conservative
price-shaped curve, and blend 10% of it into the central forecast. The
resulting archive prefix is
`value_signal`; its optimizer exports are
`A2_future_price_ST_predicted_value_signal.csv` and
`A2_future_price_ST_predicted_value_signal_vintages.csv`.

The ML and heuristic generators save every complete 96-horizon forecast at
the configured backtest-origin cadence (currently 30 minutes). The optimizer
export uses the same cadence, so every rolling solve has its own vintage.

The naive heuristic is 48-hour block persistence: the 96 observed half-hour
prices ending at a forecast origin are copied forward as the next 96 prices.
It is a varying curve, and every value is known at the forecast origin.

## Optimizer handoff

`2_Optimizer` routes realistic short-term prices to the raw A2 source named
`A2_future_price_ST_predicted.csv`. Its loader expects a CSV with:

- `Date`: unique optimizer interval-start timestamps;
- one numeric regional price column, such as `nsw_price`.

`4_3_export_optimizer_prices.ipynb` creates exactly that schema for the ML,
value-signal, naive, and operating-protocol methods. It also writes a companion
`*_vintages.csv` containing `forecast_origin`, delivery `Date`, and price so
each rolling optimization uses only the forecast available at its start. It writes the
selected method to the optimizer's canonical filename,
`Data/5_model_results/A2_future_price_ST_predicted.csv`. The cell can use any
selected run cadence and resolve overlapping forecast windows using the latest
origin, earliest origin, or their mean. Changing those choices reruns only the
export notebook, not the prediction pipeline.

To make A2 available to the optimizer, copy both canonical files to
`../2_Optimizer/optimizer/1_Dataset/1_Raw_data/`:

- `A2_future_price_ST_predicted.csv`
- `A2_future_price_ST_predicted_vintages.csv`

The current vintage-aware optimizer run uses the files at their native
30-minute granularity and renames the selected regional column to `Scenario 1`
internally.

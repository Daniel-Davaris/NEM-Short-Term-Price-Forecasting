Revised analysis — given the full pipeline
After reading 0_variables.ipynb, 1_aggregate_targets.ipynb, 2_feature_ranking.ipynb, 4_remove_duplicate_features.ipynb, 5_feature_selection.ipynb, and 6_feature_output.ipynb, the pipeline is already a 3-stage filter:

Stage	Notebook	Method
1. Relevance	2_feature_ranking	MI per (feature, horizon), 200k subsample 2019–2022
2. Redundancy	4_remove_duplicate_features	Greedy dedup in MI-rank order, drop if any kept feature has |Pearson|>0.95 OR |Spearman|>0.95
3. Conditional	5_feature_selection	LightGBM TimeSeriesSplit CV, sweep k ∈ 10-point grid, pick best_k per horizon by MAE
4. Output	6_feature_output	Per-horizon Boolean: top-best_k by MI (within unique survivors)
So most of my previous "general" recommendations are already covered. The real remaining gaps are below — these are specific to what is actually implemented.

Real gaps in the current pipeline
1. Stage 4 still ranks by raw MI within the survivors
6_feature_output picks top_k_features = mi_matrix[h].nlargest(k). The LightGBM CV in stage 3 only chooses k — it never tells stage 4 which k features. So a feature that has lower marginal MI but high LightGBM gain (the classic conditional-importance case: predispatch RRP for the target half-hour) never gets selected, even though stage 3's CV implicitly demonstrated its value.
Fix: in stage 5/6, after determining best_k, refit one LightGBM at best_k on the full subsample, take gain importance, and select the top-best_k features by gain, not by MI. Cheap (one extra fit per horizon, ~96 fits).

2. Greedy MI-order dedup can drop the wrong sibling
Among nsw_price_rmean_2016, qld_price_rmean_2016, vic_price_rmin_2016, the MI-leader survives and the others are flagged as duplicates at |Spearman|>0.95. But for the spike regime, qld might add unique information that vic can't replace. Greedy keeps highest-marginal-MI; mRMR-style would keep highest-conditional-MI.
Fix (cheap): after the greedy pass, for each removed feature with MI within 5% of its replacement, run a fast LightGBM with both vs only the survivor on a 50k subsample; reinstate the dropped one if MAE drops by >1%. Or simpler: tighten the dedup threshold to 0.97 for cross-region variants of the same statistic.

3. No spike-aware ranking
The whole pipeline ranks against continuous price. NEM economics are dominated by ~1% of intervals (>$300/MWh in NSW). MI heavily smooths spikes; a feature like wind_forecast_error_5min may have MI ≈ 0.05 but be the only signal that flags a $14k spike.
Fix: add a parallel ranking pass against y_spike = (price > 300).astype(int) using mutual_info_classif. Take the union of (MI-top-K continuous) ∪ (MI-top-K spike) before stage 2 dedup. Costs one extra ranking pass.

4. No "must-keep" list — calendar / predispatch features are at MI's mercy
Hour-of-day, day-of-week, holiday, half-hour-of-week, predispatch RRP at the target horizon — these may have low MI numerically but are causally essential. Currently they have to compete with rolling stats on equal footing.
Fix: add MUST_KEEP_FEATURES env list in 0_variables.ipynb; bypass dedup for them and force-include in stage 6 regardless of best_k.

5. CV loss is MAE — wrong loss for a spike-aware system
5_feature_selection uses objective=regression_l1 and metric=mae. MAE rewards predicting the median; it actively down-weights spike accuracy. The downstream "spike-aware ensemble" is then handed a feature set that was selected to ignore spikes.
Fix: either (a) change CV metric to RMSE or pinball loss at q90/q95, or (b) run two parallel CVs (MAE for base regressor, quantile loss for spike head) and union the selected features per horizon.

6. Single 200k subsample → no estimator-noise floor on best_k
Stage 3 fits with bagging_fraction=0.5, n_estimators=100 fixed, num_leaves=15, on one subsample, 3 folds. The k-grid winner often flips between adjacent grid points by <0.5% MAE — well within noise.
Fix: repeat stage 3 with 2–3 different RNG seeds for the subsample, average MAE across seeds before picking best_k. Cheap insurance against picking a noisy k.

7. CV window 2019–2022 misses the post-2022 regime change
Subsample range is fixed at 2019/01/01 → 2023/01/01 (FEATURE_SELECTION_CV_SUBSAMPLE_*). NEM prices changed character significantly post-2022 (gas crisis, coal retirements, more battery participation). Features selected against pre-2023 data may be sub-optimal for 2024+ inference.
Fix: extend FEATURE_SELECTION_CV_SUBSAMPLE_END to 2026, or run two pipelines (pre-2022, post-2022) and use the union, or use the latest 12 months as a hold-out to validate selection stability.

8. K-grid is linear in [1, n_features]
With N_K_VALUES=10 linearly spaced, on say 800 features that gives k ∈ {1, 89, 178, 267, …}. The MAE-vs-k curve typically elbows below k=50, so all useful resolution is in the gap between the first two grid points.
Fix: use a log-spaced grid: np.unique(np.round(np.logspace(0, log10(n), 10)).astype(int)) → e.g. {1, 3, 8, 19, 47, 113, 270, 645, …}. Better resolution where it matters.

9. Stage 3 evaluates each horizon independently; no joint selection
Producing a coherent 48-hour trajectory benefits from features that help across horizons. Per-horizon best-k picks may oscillate (e.g., k=80 at h12, k=20 at h13) producing inconsistent feature sets.
Fix: after per-horizon best_k, smooth across adjacent horizons (rolling median of best_k over a 5-horizon window), and/or take the union of top-best_k features within each ~4-hour band so the model uses a stable feature set per band.

Net answer to your original question
MI is fine as stage 1 of your existing 4-stage pipeline. The pipeline already addresses the generic MI weaknesses (redundancy via stage 2; conditional model-based via stage 3 picking k). The actually-impactful upgrades for NSW 48h forecasting are, in priority order:

Re-rank stage 4 by LightGBM gain, not MI. (biggest win, smallest change)
Add a spike-target ranking pass + union. (essential for spike-aware ensemble)

Force-keep calendar + predispatch features. (covers MI's blind spot on cyclical features)

Switch CV loss to RMSE or quantile loss. (aligns selection with downstream model objective)

Log-space the k-grid + multi-seed CV. (cheap, kills noisy best_k picks)

Extend CV window past 2022 or split pre/post-regime. (relevance for current market)

Items 1–3 are the highest leverage; together they address the "MI is marginal, ignores low-cardinality features, and is biased on heavy tails" weaknesses without throwing away the MI-based scaffolding you already have.
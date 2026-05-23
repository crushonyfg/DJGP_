# UCI small-train grouped benchmark, train=200, residual-complete

This folder combines the original five-seed train=200 benchmark with the ridge-residual LMJGP variants.

Sources:
- `../smalltrain_multiseed_grouped_5seeds/`: original `xgb_bootstrap`, `dkl_svgp`, `lmjgp_transductive`, and `lmjgp_supervised` run.
- `../smalltrain_multiseed_grouped_5seeds_residual_only/`: `lmjgp_transductive_residual_ridge` and `lmjgp_supervised_residual_ridge` run.
- `../smalltrain_multiseed_grouped_5seeds_wine_transductive_seed0_filter/`: clean rerun of Wine seed 0 `lmjgp_transductive`.

The Wine seed 0 `lmjgp_transductive` row in the original run had a non-reproducible variance explosion in CRPS/interval width while RMSE stayed normal. The row is replaced here with the clean rerun (`RMSE=0.7522`, `CRPS=0.4195`, `Cov90=0.875`) so aggregate uncertainty metrics are interpretable.

Settings match the original run: train=200, test=200, 5 seeds, Parkinsons subject-grouped split, tuned per-dataset standardization/neighborhood presets, LMJGP steps=300, MC=3, max training anchors=128.

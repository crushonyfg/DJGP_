# Inductive Projection Comparison Artifacts

This folder contains pilot outputs from
`experiments/uci/compare_inductive_projection.py`.

Very-small-train XGBoost comparison:

- `smalltrain_200_1seed_200test_xgb.csv`
- `smalltrain_200_1seed_200test_xgb.json`

Command:

```powershell
python experiments/uci/compare_inductive_projection.py --num_exp 1 `
  --methods "lmjgp_transductive,cem_em_transductive,lmjgp_supervised,cem_em_supervised,xgboost" `
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" `
  --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 `
  --neighbor_pool train --n 35 --m2 40 --lmjgp_steps 200 `
  --n_outer 5 --n_m_steps 80 --MC_num 3 `
  --lmjgp_supervised_train_mc 2 --lmjgp_supervised_grad_check 0 `
  --xgb_boots 30 --scaling_preset uci_empirical `
  --out_dir experiments/uci/inductive_projection_compare `
  --out_json experiments/uci/inductive_projection_compare/smalltrain_200_1seed_200test_xgb.json `
  --out_csv experiments/uci/inductive_projection_compare/smalltrain_200_1seed_200test_xgb.csv
```

Headline: XGBoost has the best RMSE on all three datasets; supervised LMJGP
has the best CRPS on Wine and Appliances.

Small-train XGBoost comparison:

- `smalltrain_500_1seed_200test_xgb.csv`
- `smalltrain_500_1seed_200test_xgb.json`

Command:

```powershell
python experiments/uci/compare_inductive_projection.py --num_exp 1 `
  --methods "lmjgp_transductive,cem_em_transductive,lmjgp_supervised,cem_em_supervised,xgboost" `
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" `
  --max_total_train 500 --max_test_anchors 200 --max_train_anchors 128 `
  --neighbor_pool train --n 35 --m2 40 --lmjgp_steps 200 `
  --n_outer 5 --n_m_steps 80 --MC_num 3 `
  --lmjgp_supervised_train_mc 2 --lmjgp_supervised_grad_check 0 `
  --xgb_boots 30 --scaling_preset uci_empirical `
  --out_dir experiments/uci/inductive_projection_compare `
  --out_json experiments/uci/inductive_projection_compare/smalltrain_500_1seed_200test_xgb.json `
  --out_csv experiments/uci/inductive_projection_compare/smalltrain_500_1seed_200test_xgb.csv
```

Headline: XGBoost has the best RMSE on all three datasets in this one-seed
500-train pilot. LMJGP transductive has the best Wine CRPS, and supervised
LMJGP has the best Appliances CRPS.

The current medium run is:

- `uci4_methods_empstd_1seed_64anchors.csv`
- `uci4_methods_empstd_1seed_64anchors.json`

Command:

```powershell
python experiments/uci/compare_inductive_projection.py --num_exp 1 `
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" `
  --max_test_anchors 64 --max_train_anchors 64 --n 35 --m2 40 `
  --lmjgp_steps 200 --n_outer 5 --n_m_steps 80 --MC_num 3 `
  --lmjgp_supervised_train_mc 2 --scaling_preset uci_empirical `
  --out_dir experiments/uci/inductive_projection_compare `
  --out_json experiments/uci/inductive_projection_compare/uci4_methods_empstd_1seed_64anchors.json `
  --out_csv experiments/uci/inductive_projection_compare/uci4_methods_empstd_1seed_64anchors.csv
```

Headline: supervised CEM-EM is best on Wine and Parkinsons for both RMSE and
CRPS in this one-seed 64-anchor run; LMJGP transductive is best on Appliances.

The earlier strict-LMJGP pilot is:

- `pilot_3uci_strict_lmjgp_preset.csv`
- `pilot_3uci_strict_lmjgp_preset.json`

Command:

```powershell
python experiments/uci/compare_inductive_projection.py --num_exp 1 `
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" `
  --max_test_anchors 12 --max_train_anchors 24 --n 12 --m2 20 `
  --lmjgp_steps 20 --n_outer 2 --n_m_steps 10 --MC_num 2 `
  --lmjgp_supervised_train_mc 2 --scaling_preset uci_empirical `
  --out_dir experiments/uci/inductive_projection_compare `
  --out_json experiments/uci/inductive_projection_compare/pilot_3uci_strict_lmjgp_preset.json `
  --out_csv experiments/uci/inductive_projection_compare/pilot_3uci_strict_lmjgp_preset.csv
```

The older `pilot_3uci*` files are pre-strict-LMJGP or pre-varfloor diagnostic
runs kept so the numerical history is traceable. The project-wide summary is
in `docs/experiments_log.md`.

Gradient/leakage audit smoke:

- `smoke_lmjgp_supervised_gradcheck.csv`
- `smoke_lmjgp_supervised_gradcheck.json`

This run verifies that the predictive-only supervised LMJGP loss has nonzero
q(V) gradients and decreases after one optimizer step, while diagnostics record
that anchor labels are not used as region observations.

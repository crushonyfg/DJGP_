# UCI Small-Train Summary

This note summarizes the current UCI small-train experiments around
transductive projection learning, supervised-anchor projection learning,
XGBoost, and the first LMJGP remedy ablations. The main setting is one seed
with 200 labeled training rows and 200 test rows unless stated otherwise.

These are pilot diagnostics, not final paper conclusions. Early diagnostic
sections are single-seed; the later train=200/500/1000 tables are five-seed
summaries. LMJGP runs are stochastic through q(V) initialization and Monte
Carlo prediction.

Current consolidated CSV source:
`experiments/uci/smalltrain_multiseed_bestpreset_summary/aggregate_bestpresets.csv`.
The matching per-seed rows are in
`experiments/uci/smalltrain_multiseed_bestpreset_summary/metrics_bestpresets.csv`.

Paper-comparable CSV with the two selected repaired projection variants added:
`experiments/uci/repaired_projection_variants_paper_test200_2methods/aggregate_with_baselines.csv`.
The repaired-only aggregate is
`experiments/uci/repaired_projection_variants_paper_test200_2methods/aggregate_paper_named.csv`.

## Standardization Preset

The empirical preset used in the small-train runs is:

| dataset | standardize X | standardize y | default n in latest small-train runs | note |
|---|---:|---:|---:|---|
| Wine Quality | yes | no | 25 | `std_x` was empirically best among tried settings. |
| Parkinsons Telemonitoring | no | no | 25 | Raw was the earlier multi-method preset from random-row pilots; grouped LMJGP now favors `std_x`. |
| Appliances Energy Prediction | yes | yes | 35 | Metrics are converted back to the original y scale. |

The same preset is referred to as `--scaling_preset uci_empirical` in the UCI
runners.

For the paper-comparable repaired-variant run, the exact recorded flags in
`aggregate_paper_named.csv` are:

| dataset | split | standardize X | standardize y | n | methods |
|---|---|---:|---:|---:|---|
| Wine Quality | random row | yes | no | 25 | CEM transductive VI, LMJGP low-rank CEM residual |
| Parkinsons Telemonitoring | subject grouped | yes | no | 25 | CEM transductive VI, LMJGP low-rank CEM residual |
| Appliances Energy Prediction | random row | yes | yes | 35 | CEM transductive VI, LMJGP low-rank CEM residual |

So, yes: the selected `cem_em_transductive_vi_fullcov_ensemble` and
`lmjgp_transductive_low_rank_cem_residuals` rows were run with Wine=`std_x`
and Appliances=`std_x,std_y`. Parkinsons used the later grouped `std_x`
policy used for the LMJGP-family best-preset rows.

## Paper-Comparable Train=200/500 Snapshot

This is the currently safest table source for the paper-style UCI comparison.
It uses the same test size and split policy across the included rows:

- `max_test_anchors=200`, `num_exp=5`.
- Wine/Appliances: random-row split.
- Parkinsons: subject-grouped split.
- XGBoost/DKL/base-LMJGP rows come from the best-preset summary.
- Repaired projection rows come from
  `experiments/uci/repaired_projection_variants_paper_test200_2methods/`.

Combined artifact:
`experiments/uci/repaired_projection_variants_paper_test200_2methods/aggregate_with_baselines.csv`.

Format below is `RMSE / CRPS / Cov90 / Width90`.

### Wine Quality

| method | train=200 | train=500 |
|---|---:|---:|
| XGB-bootstrap | 0.751 / 0.441 / 0.680 / 1.375 | 0.720 / 0.420 / 0.652 / 1.265 |
| DKL-SVGP | 0.966 / 0.674 / 0.229 / 0.481 | 0.852 / 0.598 / 0.224 / 0.486 |
| LMJGP transductive | 0.779 / 0.436 / 0.857 / 2.214 | 0.740 / 0.416 / 0.844 / 2.072 |
| LMJGP transductive + ridge residual | 0.770 / 0.429 / 0.823 / 2.010 | 0.746 / 0.418 / 0.809 / 1.931 |
| LMJGP supervised | 0.775 / 0.437 / 0.853 / 2.190 | 0.756 / 0.423 / 0.827 / 2.056 |
| LMJGP supervised + ridge residual | 0.764 / 0.429 / 0.818 / 2.021 | 0.737 / 0.416 / 0.802 / 1.931 |
| CEM-EM transductive VI full-cov ensemble | 0.850 / 0.514 / 0.652 / 1.559 | 0.839 / 0.491 / 0.684 / 1.478 |
| LMJGP transductive low-rank CEM residuals | 0.779 / 0.440 / 0.852 / 2.194 | 0.750 / 0.423 / 0.826 / 2.046 |

### Parkinsons Telemonitoring

| method | train=200 | train=500 |
|---|---:|---:|
| XGB-bootstrap | 9.479 / 6.741 / 0.320 / 9.932 | 9.533 / 7.037 / 0.246 / 7.866 |
| DKL-SVGP | 7.814 / 4.923 / 0.657 / 15.393 | 7.860 / 4.934 / 0.650 / 15.267 |
| LMJGP transductive | 8.997 / 5.502 / 0.733 / 22.735 | 8.937 / 5.513 / 0.703 / 20.990 |
| LMJGP transductive + ridge residual | 9.455 / 5.854 / 0.679 / 21.152 | 9.437 / 5.907 / 0.642 / 19.944 |
| LMJGP supervised | 8.912 / 5.447 / 0.716 / 22.262 | 9.130 / 5.713 / 0.656 / 20.445 |
| LMJGP supervised + ridge residual | 9.428 / 5.751 / 0.699 / 21.239 | 9.332 / 5.793 / 0.650 / 20.398 |
| CEM-EM transductive VI full-cov ensemble | 10.022 / 6.472 / 0.570 / 18.093 | 9.921 / 6.588 / 0.491 / 15.702 |
| LMJGP transductive low-rank CEM residuals | 9.288 / 5.746 / 0.678 / 21.374 | 9.143 / 5.723 / 0.652 / 19.821 |

### Appliances Energy Prediction

| method | train=200 | train=500 |
|---|---:|---:|
| XGB-bootstrap | 104.002 / 44.994 / 0.881 / 169.697 | 98.477 / 43.366 / 0.880 / 176.073 |
| DKL-SVGP | 119.605 / 57.589 / 0.479 / 46.176 | 113.843 / 52.370 / 0.581 / 58.881 |
| LMJGP transductive | 108.710 / 45.034 / 0.825 / 158.773 | 105.935 / 42.933 / 0.825 / 151.823 |
| LMJGP transductive + ridge residual | 103.892 / 48.385 / 0.783 / 158.168 | 103.683 / 44.900 / 0.814 / 159.221 |
| LMJGP supervised | 106.938 / 44.979 / 0.832 / 162.801 | 106.074 / 42.076 / 0.847 / 156.642 |
| LMJGP supervised + ridge residual | 102.981 / 47.665 / 0.783 / 162.750 | 104.515 / 45.191 / 0.809 / 160.909 |
| CEM-EM transductive VI full-cov ensemble | 111.753 / 46.877 / 0.681 / 112.814 | 108.911 / 45.685 / 0.685 / 120.184 |
| LMJGP transductive low-rank CEM residuals | 111.669 / 46.352 / 0.751 / 126.719 | 107.386 / 43.513 / 0.764 / 138.959 |

Main reading:

- Wine: XGBoost is the strongest RMSE baseline; LMJGP-family methods are much
  better calibrated and usually better on CRPS. The selected repaired variants
  do not beat the existing LMJGP rows.
- Parkinsons grouped: DKL-SVGP is the strongest RMSE/CRPS method. The repaired
  CEM/low-rank variants do not close the grouped-split gap.
- Appliances: XGBoost and ridge-residual LMJGP are best on RMSE depending on
  train size; supervised LMJGP is strongest on CRPS. The selected repaired
  variants are reasonable ablations but not headline winners.
- The cleanest paper use for `cem_em_transductive_vi_fullcov_ensemble` is as a
  posterior-ensemble ablation. The low-rank CEM residual variant is more
  competitive, but should also be positioned as an ablation rather than the
  main UCI result.

## Original 200-Train Comparison

Source artifact:
`experiments/uci/inductive_projection_compare/smalltrain_200_1seed_200test_xgb.csv`.

Settings:

- `max_total_train=200`, `max_test_anchors=200`, `max_train_anchors=128`.
- `n=35` for the original all-method runner.
- `lmjgp_steps=200`, `MC_num=3`.
- Methods: transductive LMJGP, transductive CEM-EM, supervised LMJGP,
  supervised CEM-EM, and XGBoost bootstrap.
- Metrics are on the original response scale.

| dataset | method | RMSE | CRPS | Cov90 | Width90 |
|---|---|---:|---:|---:|---:|
| Wine Quality | lmjgp_transductive | 0.772 | 0.427 | 0.865 | 2.143 |
| Wine Quality | cem_em_transductive | 0.819 | 0.520 | 0.585 | 1.293 |
| Wine Quality | lmjgp_supervised | 0.756 | 0.427 | 0.870 | 2.156 |
| Wine Quality | cem_em_supervised | 0.802 | 0.456 | 0.810 | 1.992 |
| Wine Quality | xgboost | 0.728 | 0.433 | 0.665 | 1.304 |
| Parkinsons Telemonitoring | lmjgp_transductive | 8.346 | 4.797 | 0.780 | 20.868 |
| Parkinsons Telemonitoring | cem_em_transductive | 8.139 | 4.633 | 0.755 | 18.041 |
| Parkinsons Telemonitoring | lmjgp_supervised | 8.209 | 4.698 | 0.800 | 22.133 |
| Parkinsons Telemonitoring | cem_em_supervised | 8.458 | 4.959 | 0.725 | 18.294 |
| Parkinsons Telemonitoring | xgboost | 5.558 | 3.393 | 0.630 | 10.521 |
| Appliances Energy Prediction | lmjgp_transductive | 107.425 | 40.995 | 0.855 | 124.445 |
| Appliances Energy Prediction | cem_em_transductive | 114.921 | 52.604 | 0.365 | 50.853 |
| Appliances Energy Prediction | lmjgp_supervised | 105.653 | 39.833 | 0.865 | 124.470 |
| Appliances Energy Prediction | cem_em_supervised | 108.678 | 42.464 | 0.825 | 122.985 |
| Appliances Energy Prediction | xgboost | 103.260 | 41.900 | 0.865 | 123.576 |

Main reading:

- XGBoost is best on RMSE for all three datasets in this one-seed run.
- Supervised LMJGP has the best CRPS on Wine and Appliances.
- CEM-EM supervised is not yet competitive enough to replace LMJGP; it is
  useful as an inductive empirical-Bayes projection baseline.

## LMJGP Step and Neighborhood Sweep

Source artifact:
`experiments/uci/smalltrain_pca_viz/metrics.csv`.

Settings:

- XGBoost plus supervised LMJGP only.
- `neighbors in {25,35}`, `steps in {200,300,500}`.
- `max_total_train=200`, `max_test_anchors=200`, `max_train_anchors=128`.
- Empirical standardization preset.

| dataset | XGBoost RMSE | XGBoost CRPS | best LMJGP config | best LMJGP RMSE | best LMJGP CRPS |
|---|---:|---:|---|---:|---:|
| Wine Quality | 0.728 | 0.433 | `n=25, steps=300` | 0.751 | 0.420 |
| Parkinsons Telemonitoring | 5.558 | 3.393 | `n=25, steps=300` | 7.773 | 4.399 |
| Appliances Energy Prediction | 103.260 | 41.900 | `n=35, steps=300` | 105.065 | 40.622 |

Main reading:

- In this seed, 300 steps was better than 200 or 500 for all three datasets.
- `n=25` is better for Wine and Parkinsons; Appliances prefers `n=35`.
- PC1 plots are useful for qualitative inspection but hide pointwise error,
  especially high-response tails in Appliances.

## Remedy Ablations

Source artifact:
`experiments/uci/smalltrain_lmjgp_remedies/metrics.csv`.

Settings:

- `max_total_train=200`, `max_test_anchors=200`, `max_train_anchors=128`.
- `neighbors=auto`: Wine/Parkinsons use `n=25`, Appliances uses `n=35`.
- `lmjgp_steps=300`, `MC_num=3`.
- Empirical standardization preset.
- The remedy variants are experiment-only and do not overwrite the existing
  `lmjgp_transductive` or `lmjgp_supervised` implementations.

Methods:

- `lmjgp_supervised_base`: strict supervised q(V) plus JumpGP prediction.
- `lmjgp_supervised_hybrid_pls`: raw-X and PLS-score union neighborhoods.
- `lmjgp_supervised_residual_ridge`: ridge global mean plus LMJGP residuals.
- `lmjgp_supervised_hybrid_residual_ridge`: ridge-score hybrid neighborhoods
  plus residual prediction.
- `lmjgp_supervised_localgp_*`: plain local-GP kernel diagnostics. These are
  not JumpGP/CEM predictors and should not be treated as final method variants.

| dataset | XGBoost RMSE | base LMJGP RMSE | best remedy by RMSE | best remedy RMSE | best remedy CRPS | note |
|---|---:|---:|---|---:|---:|---|
| Wine Quality | 0.728 | 0.754 | hybrid PLS neighborhood | 0.750 | 0.423 | Small improvement over base; XGBoost remains best RMSE. |
| Parkinsons Telemonitoring | 5.558 | 8.427 | hybrid PLS neighborhood | 7.989 | 4.600 | Improves base but remains far behind XGBoost. |
| Appliances Energy Prediction | 103.260 | 109.017 | ridge residual LMJGP | 102.740 | 42.840 | Best RMSE in this run, but CRPS is worse than XGBoost and base LMJGP. |

Kernel diagnostic:

| dataset | local-GP RBF RMSE | local-GP Matern-3/2 RMSE | local-GP Matern-5/2 RMSE |
|---|---:|---:|---:|
| Wine Quality | 0.769 | 0.758 | 0.762 |
| Parkinsons Telemonitoring | 8.131 | 8.041 | 8.066 |
| Appliances Energy Prediction | 104.615 | 104.360 | 104.436 |

Main reading:

- Matern gives only small changes in the local-GP diagnostic. It is a useful
  kernel ablation, not the main fix.
- Appliances is the clearest case where a global mean plus residual LMJGP can
  improve point prediction. It reduces `y>q90` RMSE from 323.3 for the remedy
  base LMJGP to 298.6, but uncertainty quality still needs work.
- Parkinsons looks more like systematic structure miss than a local kernel
  smoothness issue.

## Parkinsons Subject-Grouped Split

Source artifact:
`experiments/uci/parkinsons_grouped_split_200/metrics.csv`.

Motivation:

The historical Parkinsons UCI random-row split puts repeated measurements from
the same subject in both train and test. In the 200-train/200-test random-row
subsample, 196 of 200 test rows have a subject that also appears in training.
The grouped split is a stricter diagnostic: split by true `subject#` first, then
subsample rows.

Settings:

- 33 train subjects, 9 test subjects, zero subject overlap.
- `max_total_train=200`, `max_test_anchors=200`, `max_train_anchors=128`.
- Raw Parkinsons features, `n=25`, `lmjgp_steps=300`, `MC_num=3`.

| method | RMSE | CRPS | Cov90 | Width90 |
|---|---:|---:|---:|---:|
| xgboost | 7.828 | 5.263 | 0.430 | 10.527 |
| lmjgp_supervised | 9.063 | 5.707 | 0.675 | 21.714 |
| lmjgp_residual_ridge | 8.710 | 5.445 | 0.680 | 21.200 |
| cem_em_supervised | 9.543 | 6.339 | 0.570 | 17.958 |

Main reading:

- Grouped Parkinsons is much harder than random-row Parkinsons.
- Ridge residuals improve LMJGP but still do not beat XGBoost.
- XGBoost has the best RMSE/CRPS but undercovers strongly in this one-seed
  grouped split. LMJGP variants are wider and better covered.
- Parkinsons should be reported with a split caveat, and ideally with both
  random-row and subject-grouped sensitivity results.

## Five-Seed Train=200 Run

Source artifact:
`experiments/uci/smalltrain_multiseed_grouped_5seeds_with_residual/aggregate.csv`.

Settings:

- 5 seeds.
- `max_total_train=200`, `max_test_anchors=200`, `max_train_anchors=128`.
- Methods: `xgb_bootstrap`, `dkl_svgp`, transductive/supervised LMJGP, and
  ridge-residual variants of both LMJGP methods.
- DKL uses `epochs=80`; XGBoost uses 30 bootstrap models.
- Wine uses `std_x,n=25`; Parkinsons uses subject-grouped raw features with
  `n=25`; Appliances uses `std_x,std_y,n=35`.
- The original Wine seed 0 transductive LMJGP CRPS/width explosion was not
  reproduced by a clean rerun, so this combined artifact uses the clean rerun.

| dataset | method | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Wine Quality | xgb_bootstrap | 0.751 | 0.037 | 0.441 | 0.022 | 0.680 | 1.375 |
| Wine Quality | dkl_svgp | 0.966 | 0.021 | 0.674 | 0.015 | 0.229 | 0.481 |
| Wine Quality | lmjgp_transductive | 0.779 | 0.019 | 0.436 | 0.011 | 0.857 | 2.214 |
| Wine Quality | lmjgp_transductive_residual_ridge | 0.770 | 0.026 | 0.429 | 0.011 | 0.823 | 2.010 |
| Wine Quality | lmjgp_supervised | 0.775 | 0.027 | 0.437 | 0.013 | 0.853 | 2.190 |
| Wine Quality | lmjgp_supervised_residual_ridge | 0.764 | 0.024 | 0.429 | 0.013 | 0.818 | 2.021 |
| Parkinsons Telemonitoring | xgb_bootstrap | 9.479 | 0.870 | 6.741 | 0.748 | 0.320 | 9.932 |
| Parkinsons Telemonitoring | dkl_svgp | 7.814 | 1.167 | 4.923 | 0.990 | 0.657 | 15.393 |
| Parkinsons Telemonitoring | lmjgp_transductive | 9.517 | 0.786 | 5.992 | 0.655 | 0.673 | 20.687 |
| Parkinsons Telemonitoring | lmjgp_transductive_residual_ridge | 9.352 | 0.783 | 5.851 | 0.667 | 0.661 | 20.354 |
| Parkinsons Telemonitoring | lmjgp_supervised | 9.502 | 0.558 | 5.996 | 0.513 | 0.652 | 20.774 |
| Parkinsons Telemonitoring | lmjgp_supervised_residual_ridge | 9.511 | 0.818 | 5.953 | 0.694 | 0.665 | 20.287 |
| Appliances Energy Prediction | xgb_bootstrap | 104.002 | 5.667 | 44.994 | 3.000 | 0.881 | 169.697 |
| Appliances Energy Prediction | dkl_svgp | 119.605 | 8.100 | 57.589 | 6.544 | 0.479 | 46.176 |
| Appliances Energy Prediction | lmjgp_transductive | 108.710 | 8.033 | 45.034 | 3.892 | 0.825 | 158.773 |
| Appliances Energy Prediction | lmjgp_transductive_residual_ridge | 103.892 | 7.259 | 48.385 | 4.977 | 0.783 | 158.168 |
| Appliances Energy Prediction | lmjgp_supervised | 106.938 | 9.650 | 44.979 | 4.876 | 0.832 | 162.801 |
| Appliances Energy Prediction | lmjgp_supervised_residual_ridge | 102.981 | 5.667 | 47.665 | 4.582 | 0.783 | 162.750 |

Main reading:

- Wine: XGBoost remains best RMSE; ridge-residual supervised LMJGP has the best
  CRPS and is close on RMSE.
- Parkinsons grouped: DKL-SVGP remains best on RMSE and CRPS. LMJGP residuals
  help transductive RMSE/CRPS slightly but do not close the gap.
- Appliances: ridge-residual supervised LMJGP is best RMSE at train=200, but
  non-residual supervised LMJGP is better on CRPS.

## Five-Seed Train=500 Run

Source artifact:
`experiments/uci/smalltrain_multiseed_grouped_train500_5seeds_with_residual/aggregate.csv`.

Settings match the train=200 run except `max_total_train=500`. No row triggered
numeric omission; all aggregate rows use raw 200-test-point metrics.

| dataset | method | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Wine Quality | xgb_bootstrap | 0.720 | 0.018 | 0.420 | 0.008 | 0.652 | 1.265 |
| Wine Quality | dkl_svgp | 0.852 | 0.064 | 0.598 | 0.049 | 0.224 | 0.486 |
| Wine Quality | lmjgp_transductive | 0.740 | 0.031 | 0.416 | 0.020 | 0.844 | 2.072 |
| Wine Quality | lmjgp_transductive_residual_ridge | 0.746 | 0.027 | 0.418 | 0.015 | 0.809 | 1.931 |
| Wine Quality | lmjgp_supervised | 0.756 | 0.029 | 0.423 | 0.021 | 0.827 | 2.056 |
| Wine Quality | lmjgp_supervised_residual_ridge | 0.737 | 0.026 | 0.416 | 0.017 | 0.802 | 1.931 |
| Parkinsons Telemonitoring | xgb_bootstrap | 9.533 | 0.727 | 7.037 | 0.630 | 0.246 | 7.866 |
| Parkinsons Telemonitoring | dkl_svgp | 7.860 | 1.030 | 4.934 | 0.886 | 0.650 | 15.267 |
| Parkinsons Telemonitoring | lmjgp_transductive | 9.901 | 0.544 | 6.329 | 0.460 | 0.608 | 19.837 |
| Parkinsons Telemonitoring | lmjgp_transductive_residual_ridge | 9.827 | 0.594 | 6.319 | 0.488 | 0.589 | 19.098 |
| Parkinsons Telemonitoring | lmjgp_supervised | 9.886 | 0.595 | 6.336 | 0.491 | 0.605 | 19.600 |
| Parkinsons Telemonitoring | lmjgp_supervised_residual_ridge | 9.785 | 0.727 | 6.309 | 0.585 | 0.587 | 18.950 |
| Appliances Energy Prediction | xgb_bootstrap | 98.477 | 9.591 | 43.366 | 4.692 | 0.880 | 176.073 |
| Appliances Energy Prediction | dkl_svgp | 113.843 | 8.097 | 52.370 | 4.938 | 0.581 | 58.881 |
| Appliances Energy Prediction | lmjgp_transductive | 105.935 | 9.026 | 42.933 | 4.049 | 0.825 | 151.823 |
| Appliances Energy Prediction | lmjgp_transductive_residual_ridge | 103.683 | 7.152 | 44.900 | 2.504 | 0.814 | 159.221 |
| Appliances Energy Prediction | lmjgp_supervised | 106.074 | 7.857 | 42.076 | 4.194 | 0.847 | 156.642 |
| Appliances Energy Prediction | lmjgp_supervised_residual_ridge | 104.515 | 8.349 | 45.191 | 3.144 | 0.809 | 160.909 |

Main reading:

- Wine: XGBoost remains best RMSE; transductive LMJGP and supervised
  residual LMJGP slightly beat XGBoost on CRPS.
- Parkinsons grouped: DKL-SVGP is again best RMSE/CRPS. XGBoost undercovers
  severely.
- Appliances: XGBoost is best RMSE; non-residual supervised LMJGP is best
  CRPS. Residuals improve LMJGP RMSE but hurt CRPS.

## Five-Seed Train=1000 Run

Source artifacts:

- Wine/Appliances:
  `experiments/uci/smalltrain_multiseed_train1000_wine_appliances_5seeds/`.
- Parkinsons XGBoost/DKL:
  `experiments/uci/parkinsons_grouped_raw_xgb_dkl_train1000_5seeds/`.
- Parkinsons LMJGP with `std_x`:
  `experiments/uci/parkinsons_grouped_stdx_lmjgp_train1000_5seeds/`.
- Consolidated CSV:
  `experiments/uci/smalltrain_multiseed_bestpreset_summary/aggregate_bestpresets.csv`.

Settings match the train=500 run except `max_total_train=1000`. For Parkinsons,
LMJGP-family methods use grouped `std_x`; raw grouped LMJGP was not rerun.
No train=1000 row triggered numeric omission.

| dataset | method | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Wine Quality | xgb_bootstrap | 0.689 | 0.015 | 0.397 | 0.016 | 0.697 | 1.331 |
| Wine Quality | dkl_svgp | 0.759 | 0.039 | 0.516 | 0.028 | 0.281 | 0.502 |
| Wine Quality | lmjgp_transductive | 0.731 | 0.012 | 0.411 | 0.006 | 0.850 | 2.043 |
| Wine Quality | lmjgp_transductive_residual_ridge | 0.721 | 0.020 | 0.403 | 0.014 | 0.828 | 1.948 |
| Wine Quality | lmjgp_supervised | 0.727 | 0.016 | 0.406 | 0.009 | 0.841 | 1.997 |
| Wine Quality | lmjgp_supervised_residual_ridge | 0.724 | 0.026 | 0.403 | 0.013 | 0.808 | 1.901 |
| Parkinsons Telemonitoring | xgb_bootstrap | 9.794 | 0.570 | 7.426 | 0.235 | 0.187 | 6.479 |
| Parkinsons Telemonitoring | dkl_svgp | 7.860 | 1.036 | 4.926 | 0.876 | 0.650 | 15.159 |
| Parkinsons Telemonitoring | lmjgp_transductive | 9.227 | 0.785 | 5.836 | 0.716 | 0.628 | 19.856 |
| Parkinsons Telemonitoring | lmjgp_transductive_residual_ridge | 9.652 | 1.060 | 6.171 | 0.838 | 0.583 | 18.558 |
| Parkinsons Telemonitoring | lmjgp_supervised | 9.009 | 0.671 | 5.653 | 0.642 | 0.645 | 19.851 |
| Parkinsons Telemonitoring | lmjgp_supervised_residual_ridge | 9.730 | 1.022 | 6.198 | 0.852 | 0.580 | 18.570 |
| Appliances Energy Prediction | xgb_bootstrap | 96.244 | 8.165 | 41.266 | 4.436 | 0.878 | 167.394 |
| Appliances Energy Prediction | dkl_svgp | 110.378 | 8.429 | 51.086 | 4.327 | 0.589 | 63.233 |
| Appliances Energy Prediction | lmjgp_transductive | 105.126 | 9.136 | 41.728 | 5.517 | 0.824 | 145.124 |
| Appliances Energy Prediction | lmjgp_transductive_residual_ridge | 98.326 | 6.952 | 41.055 | 3.730 | 0.802 | 143.010 |
| Appliances Energy Prediction | lmjgp_supervised | 106.537 | 6.211 | 42.117 | 3.507 | 0.814 | 135.310 |
| Appliances Energy Prediction | lmjgp_supervised_residual_ridge | 100.679 | 8.352 | 42.083 | 3.891 | 0.810 | 142.782 |

Main reading:

- Wine: by train=1000, XGBoost is best on both RMSE and CRPS. LMJGP residuals
  remain close on CRPS but no longer lead.
- Parkinsons grouped: DKL-SVGP remains best. `std_x` supervised LMJGP is the
  best LMJGP row and still beats XGBoost bootstrap on RMSE/CRPS.
- Appliances: XGBoost is best RMSE; transductive residual LMJGP is slightly
  best CRPS and nearly matches XGBoost RMSE.

## Parkinsons Grouped Scaling Check

Source artifacts:

- Probe: `experiments/uci/parkinsons_grouped_scaling_probe_200_stdx/` and
  `experiments/uci/parkinsons_grouped_scaling_probe_200_stdxy/`.
- XGB/DKL `std_x` probe:
  `experiments/uci/parkinsons_grouped_stdx_xgb_dkl_probe_summary/`.
- DKL implementation sanity check:
  `experiments/uci/dkl_sanity_parkinsons_grouped_seed0/`.
- Five-seed follow-up:
  `experiments/uci/parkinsons_grouped_stdx_lmjgp_train200_5seeds/` and
  `experiments/uci/parkinsons_grouped_stdx_lmjgp_train500_5seeds/`.

Motivation:

The no-standardization Parkinsons preset came from earlier random-row pilots.
Under subject-grouped splitting, a seed0 probe showed that `std_x` is better
than both raw and `std_x,std_y` for the strongest non-residual LMJGP variants.

Seed0 train=200 probe:

| scaling | best method | RMSE | CRPS | Cov90 | Width90 |
|---|---|---:|---:|---:|---:|
| raw | lmjgp_transductive_residual_ridge | 8.389 | 5.152 | 0.700 | 21.781 |
| std_x | lmjgp_supervised | 8.350 | 5.122 | 0.715 | 22.997 |
| std_x,std_y | lmjgp_transductive | 8.438 | 5.212 | 0.720 | 22.819 |

Five-seed LMJGP-only follow-up with Parkinsons `std_x`:

| train | method | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |
|---:|---|---:|---:|---:|---:|---:|---:|
| 200 | lmjgp_transductive | 8.997 | 0.619 | 5.502 | 0.459 | 0.733 | 22.735 |
| 200 | lmjgp_transductive_residual_ridge | 9.455 | 0.497 | 5.854 | 0.499 | 0.679 | 21.152 |
| 200 | lmjgp_supervised | 8.912 | 0.757 | 5.447 | 0.598 | 0.716 | 22.262 |
| 200 | lmjgp_supervised_residual_ridge | 9.428 | 0.548 | 5.751 | 0.477 | 0.699 | 21.239 |
| 500 | lmjgp_transductive | 8.937 | 1.010 | 5.513 | 0.813 | 0.703 | 20.990 |
| 500 | lmjgp_transductive_residual_ridge | 9.437 | 0.746 | 5.907 | 0.738 | 0.642 | 19.944 |
| 500 | lmjgp_supervised | 9.130 | 0.999 | 5.713 | 0.877 | 0.656 | 20.445 |
| 500 | lmjgp_supervised_residual_ridge | 9.332 | 0.902 | 5.793 | 0.799 | 0.650 | 20.398 |

Main reading:

- For grouped Parkinsons, use `std_x` for LMJGP. The earlier raw preset is no
  longer the best LMJGP setting once the split is subject-grouped.
- Non-residual LMJGP benefits most from `std_x`. Ridge residuals are not useful
  for Parkinsons after `X` standardization.
- A seed0 probe for XGBoost/DKL over train=200/500/1000 showed XGBoost is
  unchanged by `std_x`, while DKL-SVGP becomes much worse with the current
  hyperparameters. Therefore the consolidated policy remains raw for
  XGBoost/DKL and `std_x` only for LMJGP-family rows.
- DKL sanity diagnostics confirm this is not a leakage or reporting-scale bug:
  X scaling is train-only, DKL y-standardization is train-only and mapped back
  to original scale, but `std_x` collapses the learned latent feature scale and
  predictive sigma under the current DKL hyperparameters.
- DKL-SVGP still remains best overall on the grouped split, but `std_x` LMJGP
  beats XGBoost bootstrap on RMSE and CRPS in the train=200/500 comparisons.

## Projection-JGP Appendix Baselines

Source artifact:
`experiments/uci/smalltrain_jgp_projection_train500_5seeds/aggregate.csv`.

Settings:

- Train=500, test=200, 5 seeds.
- Methods: `jgp_pca`, `jgp_sir`.
- PCA and SIR are fit on training rows only.
- Wine/Appliances use random-row splits; Parkinsons uses the subject-grouped
  split. Appliances uses train-only target standardization for fitting and
  original-scale reporting.

| dataset | method | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Wine Quality | jgp_pca | 0.839 | 0.028 | 0.478 | 0.016 | 0.768 | 1.855 |
| Wine Quality | jgp_sir | 0.803 | 0.039 | 0.469 | 0.028 | 0.752 | 1.794 |
| Parkinsons Telemonitoring | jgp_pca | 10.202 | 0.905 | 6.914 | 1.016 | 0.452 | 14.261 |
| Parkinsons Telemonitoring | jgp_sir | 10.158 | 0.317 | 6.933 | 0.466 | 0.432 | 14.177 |
| Appliances Energy Prediction | jgp_pca | 113.204 | 6.567 | 49.100 | 3.935 | 0.615 | 103.354 |
| Appliances Energy Prediction | jgp_sir | 120.179 | 11.293 | 51.661 | 7.403 | 0.631 | 98.180 |

Main reading:

- Classical deterministic projection-JGP baselines do not change the small-train
  conclusions. At train=500, they are weaker than the best XGBoost/DKL/LMJGP
  rows on all three datasets.
- This is enough for an appendix sanity table: ordinary PCA/SIR projection plus
  JGP is not driving the LMJGP behavior.

## Conformalized Quantile XGBoost

Source artifact:
`experiments/uci/xgb_conformal_quantile_smalltrain_5seeds/aggregate.csv`.
The direct comparison against naive bootstrap XGBoost is in
`experiments/uci/xgb_conformal_quantile_smalltrain_5seeds/xgb_cqr_vs_bootstrap.csv`.

Settings:

- Train=200/500/1000, test=200, 5 seeds.
- Split conformal CQR with 80% proper training and 20% calibration inside each
  small training set.
- Lower/upper XGBoost quantile models are fit on proper training rows; a
  separate squared-error XGBoost model fit on all training rows provides RMSE.
- CQR intervals are direct conformal intervals, not Gaussian predictive
  posteriors, so CRPS is not reported.

| train | dataset | RMSE | Cov90 | Width90 | Cov95 | Width95 |
|---:|---|---:|---:|---:|---:|---:|
| 200 | Wine Quality | 0.770 | 0.960 | 3.793 | 0.992 | 5.095 |
| 200 | Parkinsons Telemonitoring | 9.813 | 0.831 | 23.703 | 0.919 | 30.277 |
| 200 | Appliances Energy Prediction | 107.852 | 0.900 | 269.672 | 0.940 | 480.719 |
| 500 | Wine Quality | 0.743 | 0.910 | 2.736 | 0.963 | 3.831 |
| 500 | Parkinsons Telemonitoring | 9.657 | 0.707 | 19.015 | 0.864 | 25.302 |
| 500 | Appliances Energy Prediction | 99.415 | 0.946 | 343.492 | 0.978 | 599.270 |
| 1000 | Wine Quality | 0.703 | 0.915 | 2.591 | 0.956 | 3.614 |
| 1000 | Parkinsons Telemonitoring | 9.955 | 0.649 | 17.409 | 0.786 | 21.861 |
| 1000 | Appliances Energy Prediction | 98.860 | 0.905 | 205.004 | 0.962 | 389.265 |

Main reading:

- CQR largely fixes Wine and Appliances 90% coverage, but it does so by widening
  intervals substantially. For example, Wine train=500 XGB bootstrap had
  `Cov90=0.652, Width90=1.265`; CQR has `Cov90=0.910, Width90=2.736`.
- On Parkinsons grouped, CQR improves coverage a lot over naive XGB bootstrap
  but still undercovers at train=500/1000. This is expected because calibration
  rows come from training subjects while test rows are unseen subjects, so the
  split-conformal exchangeability assumption is stressed.
- This is a strong appendix robustness check: XGB can be calibrated, but its
  calibrated intervals become much wider and still do not dominate LMJGP/DKL
  under subject-grouped shift.

## Current Practical Conclusions

| dataset | current best RMSE signal | current best CRPS/UQ signal | recommended next action |
|---|---|---|---|
| Wine Quality | XGBoost is best RMSE at train=200/500/1000. | LMJGP variants lead CRPS at train=200/500, but XGBoost leads by train=1000. | Keep `std_x,n=25,steps=300`; residual helps LMJGP RMSE/CRPS modestly but does not beat XGBoost at 1000. |
| Parkinsons Telemonitoring | DKL-SVGP is best on the grouped split; among LMJGP variants, non-residual `std_x` is best. | DKL-SVGP is best CRPS; `std_x` LMJGP beats XGBoost CRPS and covers better. | For grouped LMJGP use `std_x,n=25,steps=300`; do not use the earlier raw preset for final LMJGP reporting. |
| Appliances Energy Prediction | Residual LMJGP is best RMSE at train=200, while XGBoost is best at train=500/1000. | LMJGP is best or essentially tied on CRPS across 200/500/1000, especially residual transductive at 1000. | Residuals help mean prediction; calibrate intervals and continue high-y tail diagnostics. |

Overall:

- The supervised-anchor direction is cleaner statistically than test-anchor
  transductive fitting and remains worth developing.
- The most promising UCI remedy is not Matern alone. It is global mean plus
  residual local prediction, especially for Appliances.
- Response-aware hybrid neighborhoods need more tuning. The PLS version is not
  enough for Appliances.
- The train=200/500/1000 multi-seed pattern is now stable enough for internal
  direction-setting. For paper-level claims, the cleaner story is dataset
  specific: UCI does not show a universal RMSE win, but supervised/LMJGP
  variants can be competitive on CRPS/calibration and Appliances residual
  modeling.

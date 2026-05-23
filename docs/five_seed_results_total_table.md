# Five-Seed Total Results Table

Source index: `docs/five_seed_results_summary.md`. Consolidates requested method groups: `xgboost`, `dkl`, `dmgp`, `lmjgp`, and `self_cem`, now including UCI, PDEBench, and paper-style synthetic L2/LH. For method families with multiple presets, the row is the best-RMSE preset in the available 5-seed aggregate and `raw` records the selected method. Missing rows mean the summary-backed sources do not currently contain that method for that dataset/train/subset.

| benchmark | dataset | train | subset | method | raw | n | RMSE mean+/-std | CRPS mean+/-std | Cov90 | W90 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pdebench | PDEBench Burgers ShockJump | 200 | all | xgboost | xgb_bootstrap | 5 | 0.8309+/-0.0306 | 0.3813+/-0.0294 | 0.9175 | 1.9604 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | all | dkl | dkl_svgp | 5 | 0.9225+/-0.0465 | 0.4706+/-0.0426 | 0.7000 | 1.4414 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | all | dmgp | deep_mahalanobis_gp | 5 | 0.8297+/-0.0512 | 0.3916+/-0.0409 | 0.9055 | 2.0422 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | all | lmjgp | lmjgp_predict_vi | 5 | 0.8866+/-0.0293 | 0.3517+/-0.0104 | 0.8235 | 1.3452 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | all | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| pdebench | PDEBench Burgers ShockJump | 200 | far_from_threshold | xgboost | xgb_bootstrap | 5 | 0.7002+/-0.0344 | 0.3081+/-0.0340 | 0.9497 | 1.8663 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | far_from_threshold | dkl | dkl_svgp | 5 | 0.7097+/-0.0542 | 0.3481+/-0.0669 | 0.7551 | 1.4198 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | far_from_threshold | dmgp | deep_mahalanobis_gp | 5 | 0.7014+/-0.0225 | 0.3198+/-0.0342 | 0.9592 | 1.9694 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | far_from_threshold | lmjgp | lmjgp_predict_vi | 5 | 0.6899+/-0.0249 | 0.2210+/-0.0100 | 0.8823 | 0.9011 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | far_from_threshold | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| pdebench | PDEBench Burgers ShockJump | 200 | near_threshold | xgboost | xgb_bootstrap | 5 | 1.1143+/-0.0615 | 0.5843+/-0.0414 | 0.8283 | 2.2212 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | near_threshold | dkl | dkl_svgp | 5 | 1.3412+/-0.1259 | 0.8102+/-0.0620 | 0.5472 | 1.5013 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | near_threshold | dmgp | deep_mahalanobis_gp | 5 | 1.1041+/-0.1502 | 0.5906+/-0.1039 | 0.7566 | 2.2440 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | near_threshold | lmjgp | lmjgp_predict_vi | 5 | 1.2821+/-0.0604 | 0.7142+/-0.0430 | 0.6604 | 2.5770 | ok |
| pdebench | PDEBench Burgers ShockJump | 200 | near_threshold | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| pdebench | PDEBench Burgers ShockJump | 500 | all | xgboost | xgb_bootstrap | 5 | 0.7242+/-0.0074 | 0.3206+/-0.0051 | 0.9270 | 1.7830 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | all | dkl | dkl_svgp | 5 | 0.9364+/-0.0324 | 0.5094+/-0.0566 | 0.6055 | 1.2275 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | all | dmgp | deep_mahalanobis_gp | 5 | 0.7308+/-0.0129 | 0.3202+/-0.0107 | 0.9400 | 1.8768 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | all | lmjgp | lmjgp_predict_vi | 5 | 0.7895+/-0.0216 | 0.2934+/-0.0108 | 0.8390 | 1.2003 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | all | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| pdebench | PDEBench Burgers ShockJump | 500 | far_from_threshold | xgboost | xgb_bootstrap | 5 | 0.6444+/-0.0146 | 0.2752+/-0.0033 | 0.9558 | 1.7318 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | far_from_threshold | dkl | dkl_svgp | 5 | 0.7421+/-0.0568 | 0.3982+/-0.0740 | 0.6415 | 1.2311 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | far_from_threshold | dmgp | deep_mahalanobis_gp | 5 | 0.6971+/-0.0231 | 0.2946+/-0.0101 | 0.9497 | 1.8047 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | far_from_threshold | lmjgp | lmjgp_predict_vi | 5 | 0.6607+/-0.0390 | 0.2038+/-0.0185 | 0.8721 | 0.8368 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | far_from_threshold | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| pdebench | PDEBench Burgers ShockJump | 500 | near_threshold | xgboost | xgb_bootstrap | 5 | 0.9085+/-0.0401 | 0.4465+/-0.0255 | 0.8472 | 1.9252 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | near_threshold | dkl | dkl_svgp | 5 | 1.3319+/-0.0466 | 0.8179+/-0.0140 | 0.5057 | 1.2177 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | near_threshold | dmgp | deep_mahalanobis_gp | 5 | 0.8154+/-0.0422 | 0.3914+/-0.0268 | 0.9132 | 2.0767 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | near_threshold | lmjgp | lmjgp_predict_vi | 5 | 1.0665+/-0.0378 | 0.5418+/-0.0264 | 0.7472 | 2.2084 | ok |
| pdebench | PDEBench Burgers ShockJump | 500 | near_threshold | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| pdebench | PDEBench Burgers ShockJump | 1000 | all | xgboost | xgb_bootstrap | 5 | 0.7018+/-0.0137 | 0.3090+/-0.0029 | 0.9225 | 1.7579 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | all | dkl | dkl_svgp | 5 | 0.8925+/-0.0570 | 0.4588+/-0.0317 | 0.6665 | 1.3176 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | all | dmgp | deep_mahalanobis_gp | 5 | 0.7000+/-0.0099 | 0.3019+/-0.0070 | 0.9370 | 1.7050 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | all | lmjgp | lmjgp_predict_vi | 5 | 0.7551+/-0.0169 | 0.2623+/-0.0091 | 0.8415 | 1.0287 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | all | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| pdebench | PDEBench Burgers ShockJump | 1000 | far_from_threshold | xgboost | xgb_bootstrap | 5 | 0.6336+/-0.0080 | 0.2702+/-0.0040 | 0.9483 | 1.7250 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | far_from_threshold | dkl | dkl_svgp | 5 | 0.6869+/-0.0444 | 0.3407+/-0.0527 | 0.7211 | 1.3082 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | far_from_threshold | dmgp | deep_mahalanobis_gp | 5 | 0.6707+/-0.0143 | 0.2778+/-0.0091 | 0.9490 | 1.6374 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | far_from_threshold | lmjgp | lmjgp_predict_vi | 5 | 0.6734+/-0.0219 | 0.2037+/-0.0093 | 0.8646 | 0.7583 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | far_from_threshold | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| pdebench | PDEBench Burgers ShockJump | 1000 | near_threshold | xgboost | xgb_bootstrap | 5 | 0.8629+/-0.0312 | 0.4168+/-0.0177 | 0.8509 | 1.8493 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | near_threshold | dkl | dkl_svgp | 5 | 1.2924+/-0.1843 | 0.7865+/-0.0877 | 0.5151 | 1.3438 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | near_threshold | dmgp | deep_mahalanobis_gp | 5 | 0.7754+/-0.0111 | 0.3686+/-0.0085 | 0.9038 | 1.8927 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | near_threshold | lmjgp | lmjgp_predict_vi | 5 | 0.9452+/-0.0232 | 0.4248+/-0.0162 | 0.7774 | 1.7786 | ok |
| pdebench | PDEBench Burgers ShockJump | 1000 | near_threshold | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 200 | all | xgboost | xgb_bootstrap | 5 | 2.6846+/-0.2000 | 1.5337+/-0.1038 | 0.6470 | 4.7954 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 200 | all | dkl | dkl_svgp | 5 | 2.7057+/-0.3076 | 1.7969+/-0.1642 | 0.2890 | 2.0121 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 200 | all | dmgp | deep_mahalanobis_gp | 5 | 2.4926+/-0.3178 | 1.3585+/-0.1524 | 0.8070 | 6.0085 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 200 | all | lmjgp | lmjgp_pls_kl | 5 | 2.7398+/-0.2808 | 1.4742+/-0.1301 | 0.8060 | 6.9089 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 200 | all | self_cem | cem_em_transductive_vi_fullcov_ensemble | 5 | 2.8622+/-0.3239 | 1.5295+/-0.1485 | 0.7610 | 6.0431 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 500 | all | xgboost | xgb_bootstrap | 5 | 2.3832+/-0.1195 | 1.3408+/-0.0763 | 0.7230 | 4.9180 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 500 | all | dkl | dkl_svgp | 5 | 2.3390+/-0.0949 | 1.4748+/-0.0556 | 0.3910 | 2.0975 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 500 | all | dmgp | deep_mahalanobis_gp | 5 | 2.2262+/-0.1525 | 1.2115+/-0.0590 | 0.8220 | 5.4320 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 500 | all | lmjgp | lmjgp_pls_kl | 5 | 2.3984+/-0.2054 | 1.2804+/-0.0881 | 0.8450 | 6.4892 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 500 | all | self_cem | cem_em_transductive_vi_fullcov_ensemble | 5 | 2.4663+/-0.1334 | 1.3101+/-0.0607 | 0.8190 | 6.1557 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 1000 | all | xgboost | xgb_bootstrap | 5 | 2.4356+/-0.1898 | 1.4051+/-0.1129 | 0.6890 | 4.8471 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 1000 | all | dkl | dkl_svgp | 5 | 2.3749+/-0.1055 | 1.5586+/-0.0807 | 0.3480 | 1.9943 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 1000 | all | dmgp | deep_mahalanobis_gp | 5 | 2.3286+/-0.1247 | 1.3037+/-0.0722 | 0.7460 | 5.2110 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 1000 | all | lmjgp | lmjgp_baseline | 5 | 2.4041+/-0.1434 | 1.3147+/-0.0773 | 0.8050 | 6.1275 | ok |
| synthetic_l2lh | Synthetic L2/LH - L2_phantom rff q2 | 1000 | all | self_cem | qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1 | 5 | 2.5115+/-0.1825 | 1.3821+/-0.1013 | 0.7550 | 5.5383 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 200 | all | xgboost | xgb_bootstrap | 5 | 621.63+/-195.82 | 303.56+/-164.04 | 0.7760 | 566.63 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 200 | all | dkl | dkl_svgp | 5 | 553.11+/-153.77 | 277.29+/-138.98 | 0.5940 | 130.67 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 200 | all | dmgp | deep_mahalanobis_gp | 5 | 563.73+/-151.94 | 248.09+/-121.27 | 0.7780 | 419.55 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 200 | all | lmjgp | lmjgp_baseline | 5 | 600.98+/-178.72 | 271.65+/-138.16 | 0.6830 | 557.85 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 200 | all | self_cem | cem_em_transductive_vi_fullcov_ensemble | 4 | 550.55+/-163.54 | 230.25+/-119.74 | 0.5463 | 248.20 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 500 | all | xgboost | xgb_bootstrap | 5 | 618.71+/-115.81 | 300.69+/-116.93 | 0.7630 | 518.71 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 500 | all | dkl | dkl_svgp | 5 | 538.04+/-95.664 | 263.33+/-93.259 | 0.6060 | 132.67 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 500 | all | dmgp | deep_mahalanobis_gp | 5 | 567.61+/-84.687 | 240.73+/-73.445 | 0.8060 | 389.17 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 500 | all | lmjgp | lmjgp_pls_kl | 5 | 588.10+/-95.139 | 260.05+/-78.851 | 0.7440 | 737.93 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 500 | all | self_cem | cem_em_transductive_vi_fullcov_ensemble | 5 | 601.74+/-96.347 | 254.43+/-72.303 | 0.5260 | 309.35 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 1000 | all | xgboost | xgb_bootstrap | 5 | 603.20+/-106.84 | 296.83+/-102.98 | 0.7550 | 503.46 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 1000 | all | dkl | dkl_svgp | 5 | 528.88+/-84.471 | 268.49+/-80.486 | 0.5960 | 138.15 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 1000 | all | dmgp | deep_mahalanobis_gp | 5 | 513.64+/-86.545 | 216.63+/-65.291 | 0.8030 | 408.06 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 1000 | all | lmjgp | lmjgp_baseline | 5 | 595.85+/-105.11 | 258.69+/-79.871 | 0.6790 | 539.78 | ok |
| synthetic_l2lh | Synthetic L2/LH - LH rff q5 | 1000 | all | self_cem | qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1 | 5 | 590.71+/-123.98 | 250.42+/-79.350 | 0.7890 | 285.26 | ok |
| uci | Appliances Energy Prediction | 200 | all | xgboost | xgb_bootstrap | 5 | 104.00+/-5.6673 | 44.994+/-3.0001 | 0.8810 | 169.70 | ok |
| uci | Appliances Energy Prediction | 200 | all | dkl | dkl_svgp | 5 | 119.60+/-8.1002 | 57.589+/-6.5436 | 0.4790 | 46.176 | ok |
| uci | Appliances Energy Prediction | 200 | all | dmgp | deep_mahalanobis_gp | 5 | 107.31+/-9.1149 | 45.379+/-5.5959 | 0.8730 | 174.19 | ok |
| uci | Appliances Energy Prediction | 200 | all | lmjgp | lmjgp_supervised_residual_ridge | 5 | 102.98+/-5.6665 | 47.665+/-4.5818 | 0.7830 | 162.75 | ok |
| uci | Appliances Energy Prediction | 200 | all | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| uci | Appliances Energy Prediction | 500 | all | xgboost | xgb_bootstrap | 5 | 98.477+/-9.5908 | 43.366+/-4.6923 | 0.8800 | 176.07 | ok |
| uci | Appliances Energy Prediction | 500 | all | dkl | dkl_svgp | 5 | 113.84+/-8.0967 | 52.370+/-4.9380 | 0.5810 | 58.881 | ok |
| uci | Appliances Energy Prediction | 500 | all | dmgp | deep_mahalanobis_gp | 5 | 104.97+/-7.8986 | 44.701+/-3.1947 | 0.8930 | 198.29 | ok |
| uci | Appliances Energy Prediction | 500 | all | lmjgp | lmjgp_transductive_residual_ridge | 5 | 103.68+/-7.1516 | 44.900+/-2.5044 | 0.8140 | 159.22 | ok |
| uci | Appliances Energy Prediction | 500 | all | self_cem | self_cem2_nstd0.05_kcov0.1 | 5 | 109.28+/-6.6901 | 43.335+/-4.1235 | 0.7550 | 100.26 | ok |
| uci | Appliances Energy Prediction | 1000 | all | xgboost | xgb_bootstrap | 5 | 96.244+/-8.1650 | 41.266+/-4.4359 | 0.8780 | 167.39 | ok |
| uci | Appliances Energy Prediction | 1000 | all | dkl | dkl_svgp | 5 | 110.38+/-8.4286 | 51.086+/-4.3269 | 0.5890 | 63.233 | ok |
| uci | Appliances Energy Prediction | 1000 | all | dmgp | deep_mahalanobis_gp | 5 | 103.66+/-7.3882 | 43.933+/-3.5901 | 0.8860 | 189.96 | ok |
| uci | Appliances Energy Prediction | 1000 | all | lmjgp | lmjgp_transductive_residual_ridge | 5 | 98.326+/-6.9517 | 41.055+/-3.7301 | 0.8020 | 143.01 | ok |
| uci | Appliances Energy Prediction | 1000 | all | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| uci | Parkinsons Telemonitoring | 200 | all | xgboost | xgb_bootstrap | 5 | 9.4787+/-0.8698 | 6.7406+/-0.7476 | 0.3200 | 9.9324 | ok |
| uci | Parkinsons Telemonitoring | 200 | all | dkl | dkl_svgp | 5 | 7.8143+/-1.1671 | 4.9228+/-0.9898 | 0.6570 | 15.393 | ok |
| uci | Parkinsons Telemonitoring | 200 | all | dmgp | deep_mahalanobis_gp | 5 | 8.6731+/-0.5201 | 5.4898+/-0.5415 | 0.5840 | 16.695 | ok |
| uci | Parkinsons Telemonitoring | 200 | all | lmjgp | lmjgp_supervised | 5 | 8.9120+/-0.7569 | 5.4470+/-0.5978 | 0.7160 | 22.262 | ok |
| uci | Parkinsons Telemonitoring | 200 | all | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| uci | Parkinsons Telemonitoring | 500 | all | xgboost | xgb_bootstrap | 5 | 9.5327+/-0.7271 | 7.0371+/-0.6295 | 0.2460 | 7.8662 | ok |
| uci | Parkinsons Telemonitoring | 500 | all | dkl | dkl_svgp | 5 | 7.8602+/-1.0305 | 4.9345+/-0.8857 | 0.6500 | 15.267 | ok |
| uci | Parkinsons Telemonitoring | 500 | all | dmgp | deep_mahalanobis_gp | 5 | 9.0286+/-1.0397 | 5.8317+/-1.0570 | 0.5440 | 15.449 | ok |
| uci | Parkinsons Telemonitoring | 500 | all | lmjgp | lmjgp_transductive | 5 | 8.9366+/-1.0100 | 5.5135+/-0.8134 | 0.7030 | 20.990 | ok |
| uci | Parkinsons Telemonitoring | 500 | all | self_cem | self_cem2_nstd0.15_kcov0.1 | 5 | 10.235+/-0.9036 | 6.7697+/-0.9655 | 0.4750 | 14.680 | ok |
| uci | Parkinsons Telemonitoring | 1000 | all | xgboost | xgb_bootstrap | 5 | 9.7943+/-0.5703 | 7.4257+/-0.2346 | 0.1870 | 6.4790 | ok |
| uci | Parkinsons Telemonitoring | 1000 | all | dkl | dkl_svgp | 5 | 7.8600+/-1.0360 | 4.9263+/-0.8764 | 0.6500 | 15.159 | ok |
| uci | Parkinsons Telemonitoring | 1000 | all | dmgp | deep_mahalanobis_gp | 5 | 8.9489+/-0.8538 | 5.8193+/-0.8874 | 0.5180 | 14.251 | ok |
| uci | Parkinsons Telemonitoring | 1000 | all | lmjgp | lmjgp_supervised | 5 | 9.0095+/-0.6705 | 5.6530+/-0.6419 | 0.6450 | 19.851 | ok |
| uci | Parkinsons Telemonitoring | 1000 | all | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| uci | Wine Quality | 200 | all | xgboost | xgb_bootstrap | 5 | 0.7508+/-0.0373 | 0.4405+/-0.0217 | 0.6800 | 1.3749 | ok |
| uci | Wine Quality | 200 | all | dkl | dkl_svgp | 5 | 0.9656+/-0.0209 | 0.6745+/-0.0146 | 0.2290 | 0.4813 | ok |
| uci | Wine Quality | 200 | all | dmgp | deep_mahalanobis_gp | 5 | 0.7333+/-0.0212 | 0.4112+/-0.0107 | 0.8300 | 1.8568 | ok |
| uci | Wine Quality | 200 | all | lmjgp | lmjgp_supervised_residual_ridge | 5 | 0.7641+/-0.0235 | 0.4289+/-0.0131 | 0.8180 | 2.0214 | ok |
| uci | Wine Quality | 200 | all | self_cem |  |  |  |  |  |  | missing_in_summary_sources |
| uci | Wine Quality | 500 | all | xgboost | xgb_bootstrap | 5 | 0.7202+/-0.0182 | 0.4199+/-0.0077 | 0.6520 | 1.2646 | ok |
| uci | Wine Quality | 500 | all | dkl | dkl_svgp | 5 | 0.8517+/-0.0641 | 0.5980+/-0.0492 | 0.2240 | 0.4861 | ok |
| uci | Wine Quality | 500 | all | dmgp | deep_mahalanobis_gp | 5 | 0.7256+/-0.0452 | 0.4148+/-0.0331 | 0.7590 | 1.5935 | ok |
| uci | Wine Quality | 500 | all | lmjgp | lmjgp_supervised_residual_ridge | 5 | 0.7366+/-0.0264 | 0.4162+/-0.0169 | 0.8020 | 1.9307 | ok |
| uci | Wine Quality | 500 | all | self_cem | self_cem2_nstd0.15_kcov0.1 | 5 | 0.7849+/-0.0390 | 0.4489+/-0.0212 | 0.7610 | 1.6958 | ok |
| uci | Wine Quality | 1000 | all | xgboost | xgb_bootstrap | 5 | 0.6890+/-0.0154 | 0.3969+/-0.0161 | 0.6970 | 1.3315 | ok |
| uci | Wine Quality | 1000 | all | dkl | dkl_svgp | 5 | 0.7594+/-0.0393 | 0.5155+/-0.0276 | 0.2810 | 0.5020 | ok |
| uci | Wine Quality | 1000 | all | dmgp | deep_mahalanobis_gp | 5 | 0.7106+/-0.0242 | 0.4082+/-0.0168 | 0.7410 | 1.5064 | ok |
| uci | Wine Quality | 1000 | all | lmjgp | lmjgp_transductive_residual_ridge | 5 | 0.7213+/-0.0200 | 0.4027+/-0.0138 | 0.8280 | 1.9477 | ok |
| uci | Wine Quality | 1000 | all | self_cem |  |  |  |  |  |  | missing_in_summary_sources |

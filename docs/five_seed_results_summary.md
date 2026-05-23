# 5-Seed Results Summary

This file is a convenience index for every current 5-seed or `seeds0to4`
result artifact found in the repository.  Narrative interpretation still lives
in `docs/experiments_log.md`.

Update rule:

- Any new 5-seed or `seeds0to4` result must be added here.
- If the run produced persistent artifacts, it must also remain logged in
  `docs/experiments_log.md`.
- If a folder is partial, aborted, or only a helper/source rerun, mark it
  explicitly instead of mixing it with the main result tables.

## Formal Logged Results

### Synthetic

| Logged entry | Artifacts directory | Primary files to check | Status / quick note |
| --- | --- | --- | --- |
| `paper_l2_lh_xstd_train200_500_5seeds` | `experiments/synthetic/paper_l2_lh_xstd_train200_500_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed paper-style L2/LH `train=200,500` baseline table. |
| `paper_l2_lh_xstd_train1000_5seeds` | `experiments/synthetic/paper_l2_lh_xstd_train1000_5seeds/` | `aggregate.csv`, `aggregate_omit_blowup.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed paper-style L2/LH `train=1000` baseline table. |
| `paper_l2_lh_stdxy_lmjgp_train1000_5seeds` | `experiments/synthetic/paper_l2_lh_stdxy_lmjgp_train1000_5seeds/` | `aggregate.csv`, `aggregate_omit_blowup.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed `std_x,std_y` LMJGP comparison against the `xstd` baseline. |
| `lowrank_jump_xstd_baselines_train200_500_5seeds` | `experiments/synthetic/lowrank_jump_xstd_baselines_train200_500_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed 5-seed toy low-rank baseline table. |
| `lowrank_jump_qtrue5_xstd_baselines_train200_500_5seeds` | `experiments/synthetic/lowrank_jump_qtrue5_xstd_baselines_train200_500_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed `Q_true=5` toy low-rank baseline table. |
| `lowrank_jump_xstd_baselines_moderate_5seeds` | `experiments/synthetic/lowrank_jump_xstd_baselines_moderate_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed representative moderate-noise toy table. |
| `lowrank_jump_cem_vi_xstd_q2_q5_train200_500_5seeds` | `experiments/synthetic/lowrank_jump_cem_vi_xstd_q2_q5_train200_500_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed CEM-VI low-rank toy comparison. |
| `deep_mahalanobis_gp_paper_train200_500_5seeds` | `experiments/synthetic/deep_mahalanobis_gp_paper_train200_500_5seeds/` | `dmgp_aggregate.csv`, `dmgp_metrics.csv`, `dmgp_summary.json`, `README.md` | Completed DeepMahalanobisGP paper-style `train=200,500` table. |
| `deep_mahalanobis_gp_paper_train1000_5seeds` | `experiments/synthetic/deep_mahalanobis_gp_paper_train1000_5seeds/` | `dmgp_aggregate.csv`, `dmgp_metrics.csv`, `dmgp_summary.json`, `README.md` | Completed DeepMahalanobisGP paper-style `train=1000` table. |
| `deep_mahalanobis_gp_global_z_jgp_train1000_5seeds` | `experiments/synthetic/deep_mahalanobis_gp_global_z_jgp_train1000_5seeds/` | `global_z_jgp_aggregate.csv`, `global_z_jgp_metrics.csv`, `global_z_jgp_summary.json`, `README.md` | Completed DeepMahalanobisGP global-Z + JGP 5-seed aggregate. |
| `vem_w_softgamma_controls_train1000_5seeds` | `experiments/synthetic/vem_w_softgamma_controls_train1000_5seeds/` | `aggregate.csv`, `aggregate_pointfiltered.csv`, `aggregate_omit_blowup.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed learned/control VEM-W comparison; filtered views are important here. |
| `global_wgp_cem_selected_pca_map_5seeds_20260514` | `experiments/synthetic/global_wgp_cem_selected_pca_map_5seeds_20260514/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed selected global WGP-CEM 5-seed run. |
| `uncertain_w_cem_qr_alt_selected_5seeds_20260514` | `experiments/synthetic/uncertain_w_cem_qr_alt_selected_5seeds_20260514/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed selected uncertain-W / self-CEM / qR 5-seed run. |
| `selfcem_l2_lh_train200_500_5seeds_20260519` | `experiments/synthetic/selfcem_l2_lh_train200_500_5seeds_20260519/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed non-MC self-CEM fill-in for paper-style L2/LH `train=200,500`; use with existing train=1000 self-CEM rows. |
| `paper_l2_lh_xstd_lh_pls_kl_seed3_pointfilter_20260519` | `experiments/synthetic/paper_l2_lh_xstd_train1000_lh_pls_kl_seed3_pointfilter_20260519/` | `aggregate_repaired_seed3_pointfiltered_5seeds.csv`, `metrics_repaired_seed3_pointfiltered_5seeds.csv`, `metrics_pointfiltered.csv`, `README.md` | Repaired auxiliary 5-seed aggregate for LH train=1000 `lmjgp_pls_kl`: seeds 0,1,2,4 from the original raw rows plus seed 3 point-filtered at metric time; original raw/4-seed files are preserved. |
| `uncertain_w_cem_qr_hybrid_anchor_5seeds_20260516` | `experiments/synthetic/uncertain_w_cem_qr_hybrid_anchor_5seeds_20260516/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed 5-seed hybrid-anchor check; seed0 gain did not survive. |
| `uncertain_w_cem_qr_hybrid_ensemble_lh_5seeds_20260517` | `experiments/synthetic/uncertain_w_cem_qr_hybrid_ensemble_lh_5seeds_20260517/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed LH 5-seed ensemble check; ensemble helps the hybrid head but is still not the best robust LH line. |
| `mc_cem_qr_lh_b_pca_learnZ_constU_test200_5seeds_20260518` | `experiments/synthetic/mc_cem_qr_lh_b_pca_learnZ_constU_test200_5seeds_20260518/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed MC-CEM qR best-40-anchor follow-up on LH seeds 0-4/test=200; competitive but not better than existing LH controls. |
| `mc_cem_qr_lh_a_uniform_pca_constU_h3g5_test200_5seeds_20260518` | `experiments/synthetic/mc_cem_qr_lh_a_uniform_pca_constU_h3g5_test200_5seeds_20260518/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed requested route-A MC-CEM qR follow-up; slightly worse than the route-B full-test follow-up. |
| `weak_inner_w_paper_train1000_seed0` | `experiments/synthetic/weak_inner_w_paper_train1000_seed0/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Logged entry contains selected 5-seed variants even though the folder name ends with `seed0`. |

### UCI

| Logged entry | Artifacts directory | Primary files to check | Status / quick note |
| --- | --- | --- | --- |
| `smalltrain_multiseed_grouped_5seeds` | `experiments/uci/smalltrain_multiseed_grouped_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed tuned 200-train main comparison; later superseded for reporting by the residual-inclusive aggregate. |
| `smalltrain_multiseed_grouped_5seeds_with_residual` | `experiments/uci/smalltrain_multiseed_grouped_5seeds_with_residual/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed tuned 200-train reporting aggregate; current preferred 200-train summary. |
| `smalltrain_multiseed_grouped_train500_5seeds_with_residual` | `experiments/uci/smalltrain_multiseed_grouped_train500_5seeds_with_residual/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed tuned 500-train comparison with residual variants. |
| `smalltrain_multiseed_train1000_bestpresets` | `experiments/uci/smalltrain_multiseed_train1000_wine_appliances_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Formal 5-seed train=1000 Wine/Appliances artifact. |
| `smalltrain_multiseed_train1000_bestpresets` | `experiments/uci/parkinsons_grouped_raw_xgb_dkl_train1000_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Formal 5-seed train=1000 grouped Parkinsons XGBoost/DKL artifact. |
| `smalltrain_multiseed_train1000_bestpresets` | `experiments/uci/parkinsons_grouped_stdx_lmjgp_train1000_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Formal 5-seed train=1000 grouped Parkinsons LMJGP artifact. |
| `parkinsons_grouped_stdx_lmjgp` | `experiments/uci/parkinsons_grouped_stdx_lmjgp_train200_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Formal grouped Parkinsons LMJGP-only 5-seed train=200 run. |
| `parkinsons_grouped_stdx_lmjgp` | `experiments/uci/parkinsons_grouped_stdx_lmjgp_train500_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Formal grouped Parkinsons LMJGP-only 5-seed train=500 run. |
| `repaired_projection_variants_train200_500_5seeds` | `experiments/uci/repaired_projection_variants_train200_500_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed 5-seed UCI repaired-projection pilot. |
| `smalltrain_jgp_projection_train500_5seeds` | `experiments/uci/smalltrain_jgp_projection_train500_5seeds/` | `aggregate.csv`, `metrics.csv`, `diagnostics.csv`, `summary.json`, `README.md` | Completed appendix projection-JGP 5-seed baselines. |
| `xgb_conformal_quantile_smalltrain_5seeds` | `experiments/uci/xgb_conformal_quantile_smalltrain_5seeds/` | `aggregate.csv`, `metrics.csv`, `xgb_cqr_vs_bootstrap.csv`, `summary.json`, `README.md` | Completed conformalized XGBoost uncertainty baseline. |
| `uci_dmgp_smalltrain_5seeds_20260517` | `experiments/uci/dmgp_smalltrain_5seeds_20260517/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md`, `manifest.json` | Completed UCI DMGP train=200/500/1000 5-seed artifact. |
| `uncertain_w_cem_train500_selfcem_5seeds_20260515` | `experiments/uci/uncertain_w_cem_train500_selfcem_5seeds_20260515/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed 5-seed UCI self-CEM benchmark. |
| `uci_selfcem_train200_1000_5seeds_20260519` | `experiments/uci/uncertain_w_cem_train200_1000_selfcem_5seeds_20260519/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed non-MC self-CEM fill-in for UCI `train=200,1000`; use with existing train=500 self-CEM rows. |
| `uncertain_w_cem_train500_selfcem_5seeds_20260515` | `experiments/uci/smalltrain_xgb_dkl_train500_5seeds_20260515/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Baseline comparison directory referenced by the same formal 5-seed UCI self-CEM entry. |

### PDEBench and Cross-Benchmark

| Logged entry | Artifacts directory | Primary files to check | Status / quick note |
| --- | --- | --- | --- |
| `pdebench_burgers_balanced_delta01_q3n25_seeds0to4` | `results/pdebench_burgers_balanced_delta01_n2000_xgb_dkl_lmjgp_q3_n25_seeds0to4_summary.csv` and `results/pdebench_burgers_balanced_delta01_n2000_xgb_dkl_lmjgp_q3_n25_seeds0to4_summary.json` | summary `.csv/.json` | Formal seeds 0-4 PDEBench baseline comparison. |
| `pdebench_burgers_jgp_pca_q3n25` | `results/pdebench_burgers_balanced_delta01_n2000_jgp_pca_q3_n25_seeds0to4.json`, `results/pdebench_burgers_balanced_delta01_n2000_jgp_pca_q3_n25_seeds0to4.pkl`, `results/pdebench_burgers_balanced_delta01_n2000_jgp_pca_vs_lmjgp_q3_n25_seeds0to4_summary.csv`, `results/pdebench_burgers_balanced_delta01_n2000_jgp_pca_vs_lmjgp_q3_n25_seeds0to4_summary.json` | raw `.json/.pkl` plus comparison summary `.csv/.json` | Formal deterministic JGP-PCA vs LMJGP seeds 0-4 comparison. |
| `vem_em_uci_pdebench_5seeds` | `experiments/vem/vem_em_uci_pdebench_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed cross-benchmark VEM-EM 5-seed aggregate. |
| `pdebench_burgers_train200_500_1000_baselines_5seeds_20260517` | `experiments/pdebench/burgers_train200_500_1000_xgb_dkl_lmjgp_5seeds_20260517/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed PDEBench train-size baseline artifact for `xgb_bootstrap`, `dkl_svgp`, and `lmjgp_predict_vi`. |
| `pdebench_burgers_train200_500_1000_dmgp_5seeds_20260517` | `experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md`, `manifest.json` | Completed PDEBench train-size DMGP 5-seed artifact. |
| `pdebench_burgers_selfcem_train200_500_1000_5seeds_20260519` | `experiments/pdebench/burgers_train200_500_1000_selfcem_5seeds_20260519/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Completed non-MC self-CEM fill-in for PDEBench Burgers ShockJump `train=200,500,1000`. |

## Supporting, Partial, or Not-Yet-Logged 5-Seed Artifacts

These paths contain 5-seed material but should not be treated as the main
result table unless they are promoted into `docs/experiments_log.md`.

| Artifact path | Primary files to check | Classification | Note |
| --- | --- | --- | --- |
| `experiments/synthetic/uncertain_w_cem_qr_controls_5seeds_20260514/` | `aggregate.csv`, `metrics.csv` | Partial / ignore | The log explicitly says this user-aborted folder has only one original-CEM row and should be ignored. |
| `experiments/synthetic/projected_dataset_transductive_selfcem_static_5seeds_20260517/` | `global_z_supervised_jgp_metrics.csv` | Partial / ignore | The log explicitly says the attempted 5-seed expansion was stopped after interruption; do not use as a finished result. |
| `experiments/synthetic/paper_l2_lh_xstd_train1000_lh_pls_kl_pointfiltered_5seeds_20260519/` | `metrics.csv` | Failed / ignore | Mistaken full-5-seed PLS-KL point-filter attempt; all rows failed with `NameError` before the import fix. Do not use. |
| `experiments/uci/smalltrain_multiseed_grouped_5seeds_residual_only/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Supporting source run | Source run used by `smalltrain_multiseed_grouped_5seeds_with_residual`. |
| `experiments/uci/smalltrain_multiseed_grouped_5seeds_wine_transductive_seed0_filter/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Supporting source run | Clean rerun source used by `smalltrain_multiseed_grouped_5seeds_with_residual`. |
| `experiments/uci/smalltrain_multiseed_grouped_train500_5seeds/` | `aggregate.csv`, `metrics.csv`, `summary.json`, `README.md` | Unlogged / partial aggregate | Directory exists with 5-seed intent, but the README currently shows only 2 seeds in the aggregate table; do not treat as the final 500-train summary. |
| `experiments/uci/repaired_projection_variants_selected_test200_5seeds/` | `run.stdout.log`, `run.stderr.log` | Log-only artifact | Top-level aggregate files are not present. Treat as rerun logs unless promoted into the main experiment log. |

## Existing Cross-Run Helper Aggregates

| Helper path | What it is for |
| --- | --- |
| `experiments/uci/smalltrain_multiseed_bestpreset_summary/` | Consolidated CSV source for the UCI `train=200,500,1000` 5-seed small-train tables. |
| `docs/uci_smalltrain_summary.md` | Narrative synthesis of the UCI 5-seed small-train comparisons and presets. |

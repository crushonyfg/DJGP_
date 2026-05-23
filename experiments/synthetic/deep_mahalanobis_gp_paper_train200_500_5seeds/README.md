# DeepMahalanobisGP Paper Synthetic Train 200/500

This folder contains the external DeepMahalanobisGP baseline on the paper-style
L2/LH synthetic datasets with `train=200` and `train=500`, `test=200`, and five
seeds.

Run commands from the repository root:

```powershell
conda activate jumpGP
python experiments/synthetic/export_paper_synthetic_for_dmgp.py ^
  --settings l2_q2_train200,l2_q2_train500,lh_q5_train200,lh_q5_train500 ^
  --seeds 0,1,2,3,4 ^
  --out_dir experiments/synthetic/deep_mahalanobis_gp_paper_train200_500_5seeds

conda activate dmgp_tf1
python experiments/synthetic/run_deep_mahalanobis_gp_paper.py ^
  --data_dir experiments/synthetic/deep_mahalanobis_gp_paper_train200_500_5seeds ^
  --out_dir experiments/synthetic/deep_mahalanobis_gp_paper_train200_500_5seeds ^
  --settings l2_q2_train200,l2_q2_train500,lh_q5_train200,lh_q5_train500 ^
  --seeds 0,1,2,3,4 --steps1 300 --steps2 700 --w_samples 0
```

Files:

- `dmgp_metrics.csv`: per-setting, per-seed metrics.
- `dmgp_aggregate.csv`: five-seed aggregate metrics.
- `dmgp_summary.json`: exact runner arguments and per-run metadata.
- `export_summary.json`: paper synthetic generator provenance.
- `*_W_train_mean.npy`, `*_W_test_mean.npy`: train/test posterior mean
  projection arrays from DeepMahalanobisGP, saved for downstream diagnostics.
- `dmgp_train200_500_5seeds.log`: raw TensorFlow/GPflow log.

The data are paper-style L2/LH RFF-expanded synthetic datasets.  X is
standardized with train-only statistics.  DeepMahalanobisGP trains on
train-only standardized Y and reports RMSE/CRPS/interval width on the original
Y scale.

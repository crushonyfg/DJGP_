# LH train=1000 PLS-KL seed-3 point-filter repair

See `docs/experiments_log.md`, entry
`paper_l2_lh_xstd_lh_pls_kl_seed3_pointfilter_20260519`.

This folder reruns only the known exploding `lh_q5_train1000` /
`lmjgp_pls_kl` seed 3 row and records built-in point-filtered Gaussian UQ
metrics.  The original 4-seed and raw aggregates are not overwritten.

Primary files:

- `metrics.csv`: resumed runner rows.
- `metrics_pointfiltered.csv`: point-filtered metric rows emitted by the
  metric path.
- `metrics_repaired_seed3_pointfiltered_5seeds.csv`: old seeds 0,1,2,4 plus
  the repaired seed 3 point-filtered row.
- `aggregate_repaired_seed3_pointfiltered_5seeds.csv`: repaired 5-seed
  aggregate.


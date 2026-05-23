# PDEBench Burgers self-CEM train-size 5-seed run

See `docs/experiments_log.md`, entry
`pdebench_burgers_selfcem_train200_500_1000_5seeds_20260519`.

This folder fills the missing non-MC self-CEM results for the balanced
PDEBench Burgers ShockJump surrogate at train sizes 200, 500, and 1000,
seeds 0-4.

Primary files:

- `metrics.csv`: per-seed rows with all/near/far subset metrics.
- `aggregate.csv`: 5-seed aggregate rows by train size and subset.
- `summary.json`: runner metadata.


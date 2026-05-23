# `cem_em_park_grid/` — CEM-aligned EM hyperparameter sweep on Parkinsons

See the dated entry in [`docs/experiments_log.md`](../../../docs/experiments_log.md)
for purpose, headline numbers, and status.

For the algorithm description, fixes, and recommended defaults that came out
of this sweep, read
[`docs/cem_em_projection.md`](../../../docs/cem_em_projection.md).

## Files

- `run_grid.ps1` — driver that produced the 9-cell grid.
- `summarize.py` — aggregate the JSON outputs into a markdown table.
- `summary.md` — current aggregated table (run `summarize.py` again to refresh).
- `a{α}_l{λ}.{csv,json,log}` — one triple per grid cell.
- `best_with_lmjgp.{csv,json,log}` — winning cell re-run with LMJGP-VI baseline.
- `smoke.{csv,json,log}` — initial smoke test (pre-fix; useful as the negative
  control for the `W_0_init` row-normalise bug, see §4.1 of the topic doc).
- `smoke2.{csv,json,log}` — same smoke after the `W_0_init` fix; the
  `RMSE 7.6 → 3.6` jump comes from this single change.

## Reproducing

From repo root:

```powershell
conda activate jumpGP
cd <repo-root>
powershell -ExecutionPolicy Bypass -File experiments/uci/cem_em_park_grid/run_grid.ps1
python experiments/uci/cem_em_park_grid/summarize.py | Tee-Object -FilePath experiments/uci/cem_em_park_grid/summary.md
```

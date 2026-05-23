# Legacy Code

Frozen historical code from earlier iterations of the DJGP project. Not part of the active codebase. Kept for reproducibility of older results and as reference for the paper's experimental history.

## Subdirectories

| Folder | Contents |
|---|---|
| `compat_shims/` | Thin re-export shims that used to sit at the repo root and forward to `experiments/<area>/*.py` or `djgp.*`. Kept here so old notebooks / external scripts with hardcoded `from active_djgp_acquisition import ...` style imports still find them after the reorg. **Will break if `sys.path` does not include this folder.** |
| `early_models/` | Pre-DJGP baselines: BNNJGP, NNJGP, NN_test, DeepGP_test, SIR_GP. |
| `early_djgp/` | Earlier cross-validation / validation variants of DJGP. |
| `early_jumpgp/` | Earlier CV / local variants of the JumpGP wrapper. |
| `old_vi_utils/` | Pre-`djgp.variational` versions of the variational utilities. |
| `old_experiments/` | First-iteration experiment orchestration scripts. |
| `old_plotting/` | Early plotting helpers superseded by `experiments/<area>/plot_*.py`. |
| `old_data_gen/` | Early synthetic generators superseded by `src/data_gen/`. |
| `r_code/` | R scripts and R history (OHT dataset analysis). |
| `jumpgp_code_py/` | Older vendored Jump GP copy. The current vendored canonical lives at `src/JumpGaussianProcess/`. |

## Policy

- Do not add new code here.
- If something here turns out to still be needed by active code, promote it back to `src/` rather than reaching into `legacy/` from elsewhere.

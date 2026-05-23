# Refactor Plan

Goal: turn the current flat research workspace into a maintainable single Git repository without breaking the main DJGP experiments.

This plan intentionally avoids a one-shot move of every file. The repo has many root-level imports and dynamic `import_module(...)` calls, so each phase should leave the repository runnable.

## Target Layout

```text
DJGP/
  djgp/
    __init__.py
    variational.py
    minibatch.py
    active_learning.py
    metrics.py
    sir.py
  jumpgp/
    __init__.py
    ... vendored Jump GP implementation ...
  data/
    synthetic.py
    highdim.py
    autoencoder.py
  experiments/
    synthetic/
    uci/
    erosion/
    oht/
    active_learning/
  scripts/
    smoke/
  docs/
  tests/
```

## Phase 0: Preserve Current State

- Keep the previous hygiene changes: `.gitignore`, docs, `AGENTS.md`, and removal of caches from Git.
- Do not delete legacy scripts yet.
- Record current working commands for the likely main workflows:
  - `python DJGP_test.py --folder_name <dataset-folder> ...`
  - `python new_dataset.py`
  - `python erosion_exp_alone.py ...`
  - `python run_synth_al4jgp.py`

## Phase 1: Add Smoke Fixtures and Tests Before Moving Code

Create a tiny deterministic fixture so smoke tests are fast and do not require full paper-scale runs.

Planned files:

- `tests/smoke_utils.py` dynamically creates a tiny `dataset.pkl` in a temp directory.
- `tests/test_imports.py`
- `tests/test_smoke_djgp.py`
- `tests/test_smoke_jumpgp.py`
- `scripts/smoke/run_smoke.ps1`

Smoke checks:

- Import core modules:
  - `DJGP_test`
  - `JumpGP_test`
  - `check_VI_utils_gpu_acc_UZ_qumodified_cor`
  - `active_djgp_acquisition`
  - `JumpGaussianProcess.JumpGP_LD`
- Run `JumpGP_test.py` on the tiny fixture with very small `M`.
- Run `DJGP_test.py` on the tiny fixture with tiny settings, for example `--Q 2 --m1 2 --m2 3 --n 5 --num_steps 1 --MC_num 1`.
- Treat full `new_dataset.py` and `erosion_exp_alone.py` as integration targets, not default smoke tests, because they are slow and may require network/data dependencies.

## Phase 2: Create Packages Without Moving Old Entrypoints

Add package modules that wrap or re-export current implementation:

- `djgp/variational.py` imports from `check_VI_utils_gpu_acc_UZ_qumodified_cor.py`.
- `djgp/minibatch.py` imports from `new_minibatch_try.py`.
- `djgp/active_learning.py` imports from `active_djgp_acquisition.py`.
- `djgp/sir.py` contains the reusable SIR implementation currently duplicated in `JumpGP_test.py` and `new_dataset.py`.
- `jumpgp/` either wraps `JumpGaussianProcess/` or becomes the canonical moved location.

Keep old root-level scripts working during this phase.

## Phase 3: Move Canonical Implementation Files

Status: started. These files have been moved with root-level compatibility shims:

- `check_VI_utils_gpu_acc_UZ_qumodified_cor.py` -> `djgp/variational.py`
- `new_minibatch_try.py` -> `djgp/minibatch.py`
- `active_djgp_acquisition.py` -> `djgp/active_learning.py`
- `calculate_bias_and_variance.py` -> `djgp/acquisition_metrics.py`

Move implementation modules after smoke tests pass:

| Current file | Target |
| --- | --- |
| `check_VI_utils_gpu_acc_UZ_qumodified_cor.py` | `djgp/variational.py` |
| `new_minibatch_try.py` | `djgp/minibatch.py` |
| `active_djgp_acquisition.py` | `djgp/active_learning.py` |
| `calculate_bias_and_variance.py` | `djgp/acquisition_metrics.py` |
| `utils1.py` | split into `djgp/legacy_mcmc.py` and `djgp/jumpgp_bridge.py` |
| `JumpGaussianProcess/` | `jumpgp/` |

For each moved file:

- Update imports.
- Leave a short compatibility shim at the old path for one transition commit.
- Run smoke tests.

## Phase 4: Move Experiment Entrypoints

Status: started. These files have been moved with root-level compatibility entrypoints:

- `new_dataset.py` -> `experiments/uci/new_dataset.py`
- `erosion_exp_alone.py` -> `experiments/erosion/erosion_exp_alone.py`
- `erosion_exp.py` -> `experiments/erosion/erosion_exp.py`
- `L2_erosion_exp_alone.py` -> `experiments/erosion/L2_erosion_exp_alone.py`
- `erosion_new_dataset.py` -> `experiments/erosion/erosion_new_dataset.py`
- `minibatch_LH.py` -> `experiments/synthetic/minibatch_LH.py`
- `validation_new_dataset.py` -> `experiments/synthetic/validation_new_dataset.py`
- `experiment_new.py` -> `experiments/synthetic/experiment_new.py`
- `compare_K.py` -> `experiments/synthetic/compare_K.py`
- `jgp_alone_highdata.py` -> `experiments/synthetic/jgp_alone_highdata.py`
- `L2_jgp_alone_highdata.py` -> `experiments/synthetic/L2_jgp_alone_highdata.py`
- `plot_uci_pca.py` -> `experiments/uci/plot_uci_pca.py`
- `plot_uci_pca_nonstationary.py` -> `experiments/uci/plot_uci_pca_nonstationary.py`
- `OHTDataset_analysis.py` -> `experiments/oht/OHTDataset_analysis.py`
- `OHT_dataset_test.py` -> `experiments/oht/OHT_dataset_test.py`
- `run_synth_al4jgp.py` -> `experiments/active_learning/run_synth_al4jgp.py`
- `run_synth_and_plot_acq.py` -> `experiments/active_learning/run_synth_and_plot_acq.py`
- `run_synth_and_plot_acq_v2.py` -> `experiments/active_learning/run_synth_and_plot_acq_v2.py`

`experiments/uci/new_dataset.py` now exposes a real `main()` and no longer runs the full UCI benchmark on import.
`experiments/erosion/erosion_new_dataset.py` and `experiments/synthetic/validation_new_dataset.py` now also expose real `main()` functions and no longer run full experiments on import.

Move experiment scripts by topic:

| Current files | Target folder |
| --- | --- |
| `new_dataset.py`, `plot_uci_pca.py`, `plot_uci_pca_nonstationary.py`, `UCIdataset_autoencoder.py` | `experiments/uci/` |
| `erosion_exp.py`, `erosion_exp_alone.py`, `erosion_new_dataset.py`, `L2_erosion_exp_alone.py` | `experiments/erosion/` |
| `experiment.py`, `experiment_new.py`, `compare_K.py`, `minibatch_LH.py`, `jgp_alone_highdata.py`, `validation_new_dataset.py` | `experiments/synthetic/` |
| `run_synth_al4jgp.py`, `run_synth_and_plot_acq.py`, `run_synth_and_plot_acq_v2.py` | `experiments/active_learning/` |
| `OHTDataset_analysis.py`, `OHT_dataset_test.py`, `main_OHT4D_TGP_R1.R` | `experiments/oht/` |

Add CLI wrappers if needed so old commands continue to print a helpful message or forward to the new module.

## Phase 5: Move Data Generation and Baselines

Move reusable data and baseline code:

| Current files | Target |
| --- | --- |
| `data_generate.py`, `new_highdata_gen.py`, `new_highdata_gen_utils.py` | `data/` |
| `AE.py`, `LHDataset_AE.py` | `data/autoencoder.py` or `models/autoencoder.py` |
| `DeepGP_test.py`, `NNJGP.py`, `BNNJGP.py`, `NN_test.py`, `SIR_GP.py` | `baselines/` |

## Phase 6: Remove Legacy Duplicates

Only after smoke and integration checks pass:

- Decide whether `JumpGP_code_py/` is still needed. If not, remove it.
- Remove compatibility shims.
- Remove empty or abandoned prototypes such as `visualize.py` if confirmed unused.
- Keep paper artifacts and essential datasets either tracked deliberately or documented as external artifacts.

## Verification Policy

After each phase:

```powershell
conda run -n jumpGP python -m py_compile <touched-files>
conda run -n jumpGP python -m unittest tests.test_imports
conda run -n jumpGP python -m unittest tests.test_smoke_jumpgp
conda run -n jumpGP python -m unittest tests.test_smoke_djgp
```

For slower checks:

```powershell
conda run -n jumpGP python DJGP_test.py --folder_name <tiny-dataset-folder> --Q 2 --m1 2 --m2 3 --n 5 --num_steps 1 --MC_num 1
conda run -n jumpGP python JumpGP_test.py --folder_name <tiny-dataset-folder> --M 5 --device cpu
```

## Risk Notes

- `new_dataset.py` may fetch UCI data over the network and should not be a default smoke test.
- `erosion_exp_alone.py` is an orchestrator and may be too slow for a quick test. Use import/argument parsing as smoke, then one tiny generated run as integration if feasible.
- `experiments/active_learning/run_synth_al4jgp.py` references an external `ActiveLearner`; import is optional, but calling `run_al4jgp()` requires `ACTIVE_JGP_DIR` or a valid `PYTHONPATH`.
- Many scripts use dynamic imports and root-relative paths; moving experiments should happen after package wrappers exist.

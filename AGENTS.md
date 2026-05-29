# DJGP Agent Guide

This repository contains the code used around `docs/papers/DJGP.pdf` and `docs/papers/appendix.pdf` for Deep Jump Gaussian Processes. The code is research-stage and many scripts are historical experiments. Start here before changing code.

**Reorganization on 2026-05-23**: The repo was restructured into an `src/` layout. Active code lives under `src/{djgp,jumpgp,JumpGaussianProcess,shared,data_gen}/`, installed via `pip install -e .`. Historical code is frozen under `legacy/`. See `docs/repository_map.md` for the full layout and an import cheat sheet.

## Canonical Reading Order

1. Read `docs/repository_map.md` for the file taxonomy and date-based importance estimate.
2. Read `docs/paper_code_map.md` for the mapping from paper notation to implementation names.
3. Read `docs/experiments_log.md` to see what experiments are live, what was
   superseded, and where each experiment's artifacts live. Always check this
   before re-running anything that may already exist.
4. Read `docs/five_seed_results_summary.md` when you need the current index of
   all 5-seed / `seeds0to4` result folders and helper artifacts.
5. For deep-dive on benchmark results and tuning, read:
   - `docs/uci_smalltrain_summary.md` — 200-train UCI comparison summary,
     empirical standardization presets, hyperparameter notes, and Parkinsons
     grouped-split notes.
6. Prefer the package modules for new work (all under `src/`, installable via `pip install -e .`):
   - `djgp.variational`
   - `djgp.minibatch`
   - `djgp.active_learning`
   - `djgp.projections` (projection learners for the DJGP model)
   - `jumpgp` (facade over `JumpGaussianProcess`)
   - `data_gen.synthetic` / `data_gen.highdata` / `data_gen.autoencoder`
   - `shared.utils1`, `shared.djgp_runner`, `shared.jumpgp_runner`
7. For the canonical CLI entry, use `shared.djgp_runner` (was `DJGP_test.py`) and `shared.jumpgp_runner` (was `JumpGP_test.py`). Their `main()` is also exposed as `python -m shared.djgp_runner …`.
8. For the vendored Jump GP dependency, use `src/JumpGaussianProcess/` via the `jumpgp` facade. Treat `legacy/jumpgp_code_py/` as legacy unless a specific old script still requires it.

## Main Code Paths

- `src/shared/djgp_runner.py` is the command-line wrapper for DJGP training/prediction on a folder containing `dataset.pkl`. Invoke via `python -m shared.djgp_runner --folder_name <folder> --Q 5 --m1 4 --m2 40 --n 35 --num_steps 300 --MC_num 5`.
- `src/shared/jumpgp_runner.py` is the JGP/SIR baseline runner. Invoke via `python -m shared.jumpgp_runner --folder_name <folder>`.
- `src/djgp/variational.py` contains the full-batch variational implementation: `qW_from_qV`, `compute_ELBO`, `train_vi`, and `predict_vi`.
- `src/djgp/minibatch.py` contains the minibatch ELBO/training variant.
- `src/djgp/active_learning.py` wraps DJGP fitting/prediction and active-learning acquisition.
- `src/jumpgp/` is the package facade for the canonical vendored `src/JumpGaussianProcess/` code.
- `experiments/uci/new_dataset.py` is the late UCI benchmark script.
- `experiments/synthetic/minibatch_LH.py` and `experiments/synthetic/validation_new_dataset.py` are larger synthetic/LH experiment harnesses.
- `experiments/synthetic/experiment_new.py`, `experiments/synthetic/compare_K.py`, and `experiments/synthetic/jgp_alone_highdata.py` are additional synthetic sweeps.
- `experiments/active_learning/run_synth_al4jgp.py`, `experiments/active_learning/run_synth_and_plot_acq.py`, and `experiments/active_learning/run_synth_and_plot_acq_v2.py` are the synthetic active-learning entrypoints.

## Git Hygiene

- The repository should be a single Git repository.
- Do not add nested `.git` directories.
- Do not add `__pycache__`, `.pyc`, `.ipynb_checkpoints`, Excel lock files, build artifacts (`build/`, `*.egg-info/`), or generated experiment outputs.
- **When committing/pushing, only upload source and documentation — i.e. `.py` and `.md` files (plus config like `pyproject.toml`/`.gitignore`). Never upload experiment result files** (`.csv`, `.json`, `.pkl`, `.npz`, `.npy`, `.log`, `.png`, and result/output directories such as `experiments/**/*_seed*/`, `*_smoke*/`, `results/`). Stage files explicitly rather than `git add -A`.
- Active code lives under `src/`. The repo root should stay flat with only `README.md`, `AGENTS.md`, `pyproject.toml`, `.gitignore`, and the top-level dirs (`src/`, `experiments/`, `scripts/`, `notebooks/`, `tests/`, `docs/`, `data/`, `results/`, `legacy/`).

## Naming Notes

- In the paper, the global inducing outputs for the projection process are denoted `R`; in the code, the corresponding variational parameters are named `V_params`.
- The paper often uses latent dimension `K`; many scripts use `Q` for the target learned latent dimension.
- `DJGP` and `LMJGP` are used inconsistently in scripts. In the current code, `lmjgp` usually refers to the proposed DJGP method with local Mahalanobis/projection learning plus JGP.

## Safe Change Policy

- Prefer small, documented edits over moving files. After the 2026-05-23 reorg, do not re-introduce root-level shims; if you need a script visible at root, add it to `scripts/` instead.
- **Conda env: always run Python in the `jumpGP` conda env** (`conda run -n jumpGP python ...` or `conda activate jumpGP`). The package is `pip install -e .`-installed there; bare `python` or `PYTHONPATH` hacks fail with `ModuleNotFoundError: No module named 'djgp'`. There is **no** `DJGP` conda env — the env is named `jumpGP`.
- If refactoring imports, compile-check the touched files (`python -m compileall src experiments scripts tests`) and run `python -m unittest tests.test_imports` in the `jumpGP` env.
- After any change to `src/` package structure, run `pip install -e .` again so the editable install picks up new subpackages.
- Keep generated result files out of commits unless they are explicitly part of a paper artifact. Generated outputs belong under `results/<area>/`.

## Experiment-logging policy

Whenever you produce persistent experimental artifacts (raw CSV / JSON /
pickle / log / plot files that someone else, or future-you, would want to
understand later), you MUST also update the project-wide log so the
artifacts can be found and the headline result is captured in one place.

- The single source of truth is `docs/experiments_log.md`. Add a new
  dated section using the existing entry format (purpose, runner, settings,
  artifacts folder, headline result, status, follow-ups).
- Save artifacts under `experiments/<area>/<short_tag>/`. Prefer one folder
  per experiment so logs / CSV / JSON / driver scripts stay together.
- If an experiment has more than ~3 output files, also add a small
  `README.md` inside its folder pointing back to the `experiments_log.md`
  entry.
- If the result is a 5-seed run or a `seeds0to4` aggregate, you MUST also
  update `docs/five_seed_results_summary.md`.  Mark partial / aborted /
  supporting helper folders explicitly instead of mixing them with the formal
  result rows.
- For results that warrant a deeper write-up (algorithm description,
  hyperparameter rationale, failure modes), create a topic doc in `docs/`
  and link it from both `AGENTS.md` (Canonical Reading Order) and the
  matching `experiments_log.md` entry.
- When an experiment is superseded by a newer one, mark the old entry
  `status: superseded by <new entry>` rather than deleting it.

## Metric numeric-filtering policy

- Predictive-metric code for Gaussian UQ outputs MUST handle numerical
  blow-ups at the metric-computation layer, not through ad hoc post-hoc reruns
  or one-off cleanup scripts after a table looks suspicious.
- When predictions include pointwise `mu` and `sigma`, metric utilities should
  always be able to emit both:
  - raw metrics over all points, and
  - point-filtered metrics that drop only non-finite points or points whose
    predictive standard deviation exceeds the documented extreme-variance
    threshold.
- Any point-filtered result row MUST record the filtering rule, threshold,
  `n_metric_points`, `n_dropped_metric_points`, `max_sigma_before_filter`, and
  `max_sigma_after_filter` in the CSV/JSON artifacts.  Do not silently replace
  a raw aggregate with a filtered aggregate.
- For 5-seed summaries, preserve existing raw / 4-seed / omit-blowup aggregate
  files.  Add new filtered or repaired 5-seed aggregate files alongside them
  with explicit filenames and log entries, so readers can see exactly which
  result is raw, row-omitted, or point-filtered.
- If an existing runner cannot save enough pointwise prediction information to
  apply this policy, fix the shared metric/result-writing path before relying
  on that runner for formal results.

## Long-running experiment workflow

- For a large experiment or sweep, do not inspect, interpret, or report results
  one row / one variant at a time.  Let the whole planned run finish, then read
  the completed aggregate once and analyze it as a batch.
- The only exception is staged experimentation: when a complete tuning or
  pilot run finishes, analyze that finished tuning result before launching the
  next larger experiment.  Do not use partial rows from a still-running job to
  steer conclusions or start follow-up runs.
- While a long run is active, monitor only for process completion or hard
  failures such as non-empty stderr / missing process.  Avoid frequent
  result-table polling that encourages premature interpretation.

## Synthetic benchmark provenance policy

- Before running or reporting any synthetic benchmark, explicitly verify and
  record the generator family.  Do not describe a new toy generator as a paper
  synthetic dataset.
- The paper-style high-dimensional synthetic data are generated through
  `new_highdata_gen.py` / `new_highdata_gen_utils.py` (LH/high-dimensional
  expansion) and `data_generate.py` plus the L2/phantom legacy runners where
  applicable.  A direct Gaussian-X low-rank toy such as
  `compare_lmjgp_tricks_synth.generate_lowrank_jump` is only a toy sanity
  check unless a log entry and final answer explicitly say otherwise.
- Every synthetic result folder and `docs/experiments_log.md` entry must state:
  base low-dimensional generator, expansion method (`rff`, `poly`,
  `autoencoder`, none, etc.), latent dimension, observed dimension, train/test
  sizes, and whether X/Y standardization was applied.
- Always verify the generator against the paper-code map before interpreting synthetic results.

## Executable Python scripts: header run instructions (English)

- Any **runnable** `.py` file (CLI entrypoint, experiment driver, or `if __name__ == "__main__"` script) MUST keep an **English** module docstring at the **very top** (before imports) that includes:
  - **How to run** the script: `conda activate jumpGP` (the env is named `jumpGP`, not `DJGP`), `cd` to repo root, and copy-pastable `python ...` examples.
  - The **minimum** examples needed after a change: default invocation plus any new or renamed flags.
- When you **modify** such a file (CLI args, defaults, or primary workflow), **update that docstring in the same PR/edit** so it stays accurate.
- For UCI baselines specifically, also keep `experiments/uci/README.md` aligned with the docstring in `experiments/uci/compare_uci_baselines.py` (or note there that the `.py` header is canonical).
- For projection-posterior diagnostics, keep `experiments/diagnostics/run_projection_w_diagnostics.py` header docstring and `experiments/diagnostics/README.md` aligned when changing CLI or defaults.

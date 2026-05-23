# Deep Jump Gaussian Process Code

Research code for the paper `docs/papers/DJGP.pdf` and appendix `docs/papers/appendix.pdf`.

The repository was reorganized on **2026-05-23** into an `src/` layout. See `legacy/reorg_2026-05-23/` for the migration scripts and `docs/repository_map.md` for the file taxonomy.

## Setup

```powershell
conda activate jumpGP   # has torch, gpytorch, pyro, etc.
pip install -e .        # makes src/* importable as top-level packages
```

After `pip install -e .`, the following are importable from anywhere:

- `djgp` — main package (variational, minibatch, active_learning, projections, …)
- `jumpgp` — facade for the canonical vendored Jump GP
- `JumpGaussianProcess` — vendored Jump GP source
- `shared` — runner libs (`shared.djgp_runner`, `shared.jumpgp_runner`) and shared utils (`shared.utils1`, …)
- `data_gen` — synthetic / high-dim / autoencoder data generators

## Layout

```
src/                  # current code (pip install -e . target)
  djgp/               # full-batch + minibatch + active-learning + projections
  jumpgp/             # facade over vendored JumpGaussianProcess
  JumpGaussianProcess/  # vendored canonical Jump GP
  shared/             # utils.py, utils1.py, djgp_runner, jumpgp_runner, deepgp, …
  data_gen/           # synthetic, highdata, autoencoder generators
experiments/          # experiment harnesses by area (uci, synthetic, erosion, oht, …)
scripts/              # standalone driver scripts (try_new_JGP, simulate_case_d_linear, …)
notebooks/            # exploratory/, analysis/, legacy/
tests/                # smoke tests (run in jumpGP env)
docs/                 # research docs, papers/, figures/
data/                 # raw input datasets (OHTDataset.mat, housing.arff)
results/              # generated artifacts: uci/, synthetic/, lh/, oht/, diagnostics/, tables/, plots/, pdebench/
legacy/               # frozen historical code (early models, old CV/VI utils, R code, JumpGP_code_py, compat_shims, reorg_2026-05-23 scripts)
```

## Main Entry Points

- `python -m shared.djgp_runner --folder_name <folder>` — DJGP training and prediction.
- `python -m shared.jumpgp_runner --folder_name <folder>` — JGP/SIR baseline.
- `python -m experiments.uci.new_dataset` — UCI Wine/Parkinsons/Appliances benchmark.
- `python -m experiments.active_learning.run_synth_al4jgp` — synthetic active learning.
- `python -m experiments.synthetic.minibatch_LH` — LH synthetic minibatch.
- `python -m experiments.erosion.erosion_exp_alone` — erosion benchmark.

## Smoke Tests

```powershell
conda activate jumpGP
python -m unittest tests.test_imports
python -m unittest tests.test_smoke_jumpgp
python -m unittest tests.test_smoke_djgp
```

## Reading Order

1. `AGENTS.md` — agent-oriented workflow guidance and policies.
2. `docs/repository_map.md` — file taxonomy (post-reorg layout + historical notes).
3. `docs/paper_code_map.md` — paper notation ↔ code names.
4. `docs/experiments_log.md` — index of all experiments and where artifacts live.

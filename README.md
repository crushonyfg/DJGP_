# Deep Jump Gaussian Process (DJGP)

Code for the paper *"Deep Jump Gaussian Process"* (`docs/papers/DJGP.pdf`).

DJGP is a **transductive local regression model** for datasets with
discontinuous structure (jumps, regime changes).  For each test point it
finds its *k* nearest training neighbours, projects them into a learned
low-dimensional space, and fits a local Gaussian Process with a robust
inlier/outlier mixture head.  An ensemble of subspace initialisations is
used; members are selected automatically without access to test labels.

---

## Setup

Create and activate the conda environment, then install the package:

```bash
conda create -n DJGP python=3.11 -y
conda activate DJGP
pip install -r requirements.txt
pip install -e .             # makes src/* importable as top-level packages
```

---

## Reproducing benchmark results

### Synthetic datasets (L2, LH)

Two latent families — **L2** (2-D latent, jump boundary from a phantom image) and
**LH** (5-D latent, heteroscedastic jumps) — are each lifted to a `D`-dimensional
observed space (`--ambient`, default 30) by one of four **expansions** of the
latent:

| expansion  | observed features `X = f(z)`                                              |
|------------|---------------------------------------------------------------------------|
| `rff`      | random Fourier features (smooth, full-rank)                               |
| `poly`     | random degree-3 polynomial lift, mixed to `D` dims                        |
| `manifold` | Swiss-roll (constant-curvature) embedding of the latent, mixed to `D` dims |
| `noise`    | latent rotated together with `D−d` `y`-independent nuisance dimensions    |

```bash
# Quick check — seed 0, all four expansions on both datasets
python run_synthetic.py --seeds 0

# A single expansion (fastest sanity check)
python run_synthetic.py --seeds 0 --expansions rff

# Full 5-seed reproduction
python run_synthetic.py --seeds 0,1,2,3,4
```

This runs **DJGP**, **XGBoost**, and **DKL-SVGP** for every
`dataset × expansion` combination (train=1000 / test=200) and prints a
RMSE / CRPS / cov90 table.  Results are saved to `results/synthetic/metrics.csv`.
Runtime scales with the number of expansions × datasets × seeds × methods, so the
seed-0 default (8 dataset–expansion cells) takes appreciably longer than a single
cell; restrict with `--expansions`, `--datasets`, or `--methods` for a fast check.

Useful flags:
```bash
# pick expansions / datasets / ambient dimension
python run_synthetic.py --seeds 0 --datasets L2 --expansions rff,poly --ambient 30

# LH also supports a 50-D ambient space
python run_synthetic.py --seeds 0 --datasets LH --ambient 50

# also include the DeepGP baseline
python run_synthetic.py --seeds 0 --methods DJGP,XGBoost,DKL-SVGP,DeepGP
```

### UCI datasets (Wine Quality, Parkinsons, Appliances)

The benchmark requires pre-processed data splits.  Generate them once:
```bash
python experiments/uci/export_smalltrain_dmgp_splits.py
```

Then run:
```bash
# Quick check — seed 0 only (~10–20 min on GPU)
python run_uci.py --seeds 0

# Full 5-seed reproduction
python run_uci.py --seeds 0,1,2,3,4
```

This runs **DJGP**, **XGBoost**, and **DKL-SVGP** on Wine Quality, Parkinsons
Telemonitoring (subject-grouped split), and Appliances Energy Prediction.
Results are saved to `results/uci/metrics.csv`.

---

## Using DJGP on your own data

```python
import numpy as np
from djgp.api import fit_predict

# X_train: [N, D] float array   y_train: [N] float array   X_test: [T, D]
result = fit_predict(X_train, y_train, X_test, y_test=y_test)

print(f"RMSE  {result['rmse']:.3f}")
print(f"CRPS  {result['crps']:.3f}")
print(f"cov90 {result['cov90']:.3f}")

# Predictive distribution
mu    = result["mu"]     # shape [T]
sigma = result["sigma"]  # shape [T]
```

**Key parameters** (all keyword-only, with sensible defaults):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Q` | `5` | Projection dimension (try `2` for sharp-boundary data) |
| `n_neighbors` | `25` | Local neighbourhood size *k* |
| `steps` | `300` | Training iterations |
| `noise_mode` | `"data_driven"` | `"data_driven"`: heteroscedastic per-anchor noise (recommended); `"fixed"`: homoscedastic prior |
| `n_ensemble` | `4` | Number of ensemble members to mix |
| `seed` | `0` | Random seed |
| `device` | auto | `"cpu"` or `"cuda"` |
| `auto_tune` | `False` | If `True`, runs a short sweep over Q ∈ {2, 5} and noise_mode and selects the best |

---

## Package layout

```
src/
  djgp/               # DJGP model and projections
    api.py            # public fit_predict interface (this file)
    projections/      # core model implementation
    baselines/        # XGBoost and DKL-SVGP baselines
    evaluation/       # CRPS, calibration metrics
  jumpgp/             # vendored Jump GP
  shared/             # utilities, DeepGP, runners
  data_gen/           # synthetic data generators
experiments/
  synthetic/          # synthetic experiment harnesses
  uci/                # UCI experiment harnesses
run_synthetic.py      # benchmark entry point (synthetic)
run_uci.py            # benchmark entry point (UCI)
```

---

## Smoke tests

```bash
conda activate DJGP
python -m pytest tests/test_smoke_ism_lmjgp.py -v   # DJGP smoke test (~1 min)
python -m pytest tests/test_smoke_jumpgp.py -v       # JumpGP smoke test
```

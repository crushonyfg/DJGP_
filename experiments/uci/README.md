# UCI Experiments

Contains UCI benchmark and plotting scripts:

For the current 200-train UCI synthesis, including empirical standardization
presets, remedy ablations, and Parkinsons grouped-split notes, see
`docs/uci_smalltrain_summary.md`.

The consolidated 5-seed train=200/500/1000 CSVs live in
`experiments/uci/smalltrain_multiseed_bestpreset_summary/`.

## Running `compare_uci_baselines.py` (English)

From the repository root with conda env `jumpGP`:

```bash
conda activate jumpGP
cd /path/to/DJGP

# Default: LMJGP + DKL + XGB
python experiments/uci/compare_uci_baselines.py --num_exp 10

# Several datasets
python experiments/uci/compare_uci_baselines.py --num_exp 10 --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"

# Optional JumpGP (any combination)
python experiments/uci/compare_uci_baselines.py --num_exp 10 --jgp_raw --jgp_pca --jgp_sir

# Optional DeepGP
python experiments/uci/compare_uci_baselines.py --num_exp 10 --deepgp

# DKL + XGB only (skip LMJGP)
python experiments/uci/compare_uci_baselines.py --num_exp 1 --no_lmjgp

# Custom output file / hyperparameters
python experiments/uci/compare_uci_baselines.py --num_exp 10 --xgb_boots 30 --dkl_epochs 80 --out uci_results.pkl
```

The **source of truth** for run commands is the module docstring at the top of
`experiments/uci/compare_uci_baselines.py` (keep it updated when the CLI changes).

---

- `new_dataset.py`: late UCI benchmark script for Wine Quality, Parkinsons Telemonitoring, and Appliances Energy Prediction.
- `uci_data.py`: shared loaders / SIR helper (same splits as `new_dataset.py`).
- `compare_uci_baselines.py`: **默认**跑 **DKL + XGBoost bootstrap + LMJGP**；JumpGP / DeepGP 默认关闭，用开关打开。示例:
  - `python experiments/uci/compare_uci_baselines.py --num_exp 10` — 默认三方法。
  - `python experiments/uci/compare_uci_baselines.py --num_exp 10 --jgp_raw --jgp_sir` — 外加 raw / SIR 两种 JGP。
  - `python experiments/uci/compare_uci_baselines.py --num_exp 10 --deepgp` — 外加 GPyTorch DeepGP。
  - `python experiments/uci/compare_uci_baselines.py --num_exp 1 --no_lmjgp` — 仅 DKL + XGB（冒烟）。
- `compare_inductive_projection.py`: compares current test-anchor
  transductive LMJGP/CEM-EM against supervised training-anchor variants that
  induce `W_*` or `C_*` at test anchors. The source of truth for commands is
  the module docstring.
- `visualize_smalltrain_pca_predictions.py`: runs the 200-train XGBoost vs
  supervised-LMJGP PC1 visualization and LMJGP step/neighborhood sweep. The
  source of truth for commands is the module docstring.
- `try_lmjgp_remedies.py`: experiment-only 200-train remedy ablations for
  supervised LMJGP: PLS-score hybrid neighborhoods, ridge global-mean residuals,
  and plain local-GP RBF/Matern kernel diagnostics. The source of truth for
  commands is the module docstring.
- `compare_parkinsons_grouped_split.py`: Parkinsons-specific subject-grouped
  split comparison for XGBoost, supervised LMJGP, ridge-residual LMJGP, and
  supervised CEM-EM. The source of truth for commands is the module docstring.
- `compare_smalltrain_multiseed_grouped.py`: tuned 200-train multiseed runner
  for XGBoost bootstrap, DKL-SVGP, transductive/supervised LMJGP, and
  ridge-residual variants of both LMJGP modes; Parkinsons uses subject-grouped
  splits. It also supports Parkinsons-only scaling probes through
  `--parkinsons_standardize_x` and `--parkinsons_standardize_y`. The source of
  truth for commands is the module docstring.
- `dkl_sanity_check.py`: DKL-SVGP preprocessing and learned-hyperparameter
  diagnostic, used to audit train-only scaling and Parkinsons `std_x`
  sensitivity. The source of truth for commands is the module docstring.
- `compare_smalltrain_jgp_projection.py`: appendix-style PCA/SIR + JumpGP
  baselines on the small-train split presets. The source of truth for commands
  is the module docstring.
- `compare_xgb_conformal_quantile.py`: split conformalized quantile XGBoost
  interval baseline for the small-train split presets. The source of truth for
  commands is the module docstring.
- `plot_uci_pca.py`: PCA summary plot utility.
- `plot_uci_pca_nonstationary.py`: PCA nonstationarity diagnostic plot utility.

**Dependencies:** `xgboost` for the bootstrap baseline (`pip install xgboost`). Other methods use existing `gpytorch` / `torch` stack.

Root `new_dataset.py` and `plot_uci_pca*.py` files are compatibility entrypoints.

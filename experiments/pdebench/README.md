# PDEBench Experiments

PDEBench-derived scalar-QoI surrogate benchmarks.

- `compare_pdebench_burgers.py`: compares XGB bootstrap, DKL-SVGP, LMJGP `predict_vi`, and CEM-EM projection variants on a processed Burgers `dataset.pkl`.

The runner assumes data has already been converted with `scripts/data/prepare_pdebench_burgers.py`; it does not download PDEBench data.
CEM-EM defaults follow `experiments/uci/compare_3way_10seeds.py`: `Q=5` when unset, `n=35`, `n_pls=5`, `n_pca=5`, `n_outer=5`, `n_m_steps=80`, `lr=0.01`, `lambda_gate=0.1`, and `lambda_smooth=0.01`.

Example:

```powershell
conda run -n jumpGP python -m experiments.pdebench.compare_pdebench_burgers `
  --dataset-folder data\processed\pdebench_burgers_shockjump `
  --num_exp 1 `
  --max_anchors 200 `
  --out_json results\pdebench_burgers_compare.json
```

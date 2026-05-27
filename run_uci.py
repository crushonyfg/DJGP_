"""Reproduce the DJGP UCI benchmark.

Three datasets from the UCI repository:
  * Wine Quality              — random-row 80/20 split, 11 features
  * Parkinsons Telemonitoring — subject-grouped split (harder, tests distribution shift)
  * Appliances Energy Prediction — random-row split, 26 features

Methods: DJGP, XGBoost, DKL-SVGP.
Metrics: RMSE / CRPS / cov90 (mean ± std over seeds).

Pre-processed data splits are required; run once if not present:
    python experiments/uci/export_smalltrain_dmgp_splits.py

Usage
-----
Quick check (seed 0 only):
    python run_uci.py --seeds 0

Full 5-seed reproduction:
    python run_uci.py --seeds 0,1,2,3,4

Select datasets:
    python run_uci.py --datasets Wine,Parkinsons --seeds 0

Results written to --out (default: results/uci/metrics.csv).
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.baselines import run_dkl_svgp_regression, run_xgboost_bootstrap_regression  # noqa: E402
from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402
from experiments.synthetic._exp_uci_hetnoise import run_dataset_seed  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-selected benchmark configurations
# ---------------------------------------------------------------------------
_NPZ_DIR = REPO / "experiments/uci/dmgp_smalltrain_5seeds_20260517/data"

_SLUG = {
    "Wine":       "wine_quality",
    "Parkinsons": "parkinsons_telemonitoring",
    "Appliances": "appliances_energy_prediction",
}
_FULL_NAME = {
    "Wine":       "Wine Quality",
    "Parkinsons": "Parkinsons Telemonitoring",
    "Appliances": "Appliances Energy Prediction",
}
_CONFIGS = {
    "Wine":       {"Q": 5, "n_neighbors": 25, "steps": 500, "noise_mode": "data_driven"},
    "Parkinsons": {"Q": 5, "n_neighbors": 25, "steps": 200, "noise_mode": "data_driven"},
    "Appliances": {"Q": 5, "n_neighbors": 35, "steps": 200, "noise_mode": "fixed"},
}

_METRIC_KEYS = ["dataset", "seed", "method", "rmse", "crps", "cov90"]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_djgp(ds, seed, cfg, npz_dir, device):
    rows = run_dataset_seed(
        _FULL_NAME[ds], seed,
        K=6, steps=cfg["steps"], Q=cfg["Q"],
        n_neighbors=cfg["n_neighbors"], device=device,
        noise_mode=cfg["noise_mode"],
        nd_frac=1.0, nd_prior_std=0.3, nd_prior_weight=3.0,
        rho_mode="near1", npz_dir=npz_dir)
    for r in rows:
        if r["head"] == "analytic" and r["topk"] == 4 and r["combiner"] == "mixture":
            return {"dataset": ds, "seed": seed, "method": "DJGP",
                    "rmse": r["rmse"], "crps": r["crps"], "cov90": r["cov90"]}
    return None


def _run_baselines(ds, seed, npz_dir, device):
    path = Path(npz_dir) / f"uci_{_SLUG[ds]}_train1000_seed{seed}.npz"
    d = np.load(path, allow_pickle=True)
    Xtr = np.asarray(d["X_train"], np.float64)
    Xte = np.asarray(d["X_test"], np.float64)
    ym = float(np.asarray(d["y_mean"]).ravel()[0])
    ys = float(np.asarray(d["y_scale"]).ravel()[0])
    ytr = np.asarray(d["y_train_std"], np.float64).ravel() * ys + ym
    yte = np.asarray(d["y_test"], np.float64).ravel()

    rows = []
    rmse, crps, el, det = run_xgboost_bootstrap_regression(
        Xtr, ytr, Xte, yte, n_bootstrap=30, random_state=seed)
    uq = pack_gaussian_uq_list(rmse, crps, el, yte, det["mu"], det["sigma"])
    rows.append({"dataset": ds, "seed": seed, "method": "XGBoost",
                 "rmse": float(uq[0]), "crps": float(uq[1]), "cov90": float(uq[3])})

    rmse, crps, el, mu, sig = run_dkl_svgp_regression(
        Xtr, ytr, Xte, yte, device=device, epochs=80, seed=seed)
    uq = pack_gaussian_uq_list(rmse, crps, el, yte, mu, sig)
    rows.append({"dataset": ds, "seed": seed, "method": "DKL-SVGP",
                 "rmse": float(uq[0]), "crps": float(uq[1]), "cov90": float(uq[3])})
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(rows):
    agg: dict = collections.defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["method"])].append(r)
    print(f"\n{'Dataset':12s}  {'Method':12s}  {'RMSE':>16}  {'CRPS':>16}  {'cov90':>10}")
    print("-" * 72)
    for ds in ("Wine", "Parkinsons", "Appliances"):
        for mth in ("DJGP", "XGBoost", "DKL-SVGP"):
            g = agg.get((ds, mth))
            if not g:
                continue
            def ms(k):  # noqa: E306
                v = np.array([x[k] for x in g])
                return f"{v.mean():.3f}±{v.std():.3f}"
            print(f"{ds:12s}  {mth:12s}  {ms('rmse'):>16}  {ms('crps'):>16}  {ms('cov90'):>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DJGP UCI benchmark")
    p.add_argument("--datasets", default="Wine,Parkinsons,Appliances",
                   help="comma-separated from: Wine, Parkinsons, Appliances")
    p.add_argument("--seeds", default="0",
                   help="comma-separated seeds (e.g. 0,1,2,3,4 for full reproduction)")
    p.add_argument("--npz_dir", default=str(_NPZ_DIR),
                   help="directory with uci_*_train1000_seed*.npz files")
    p.add_argument("--out", default="results/uci/metrics.csv")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [s for s in a.datasets.split(",") if s]
    seeds = [int(s) for s in a.seeds.split(",") if s]
    npz_dir = Path(a.npz_dir)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not npz_dir.exists():
        print(f"WARNING: data directory not found: {npz_dir}")
        print("Run:  python experiments/uci/export_smalltrain_dmgp_splits.py")
        return

    allrows: list[dict] = []
    t0 = time.perf_counter()

    for ds in datasets:
        cfg = _CONFIGS[ds]
        for sd in seeds:
            print(f"\n[{ds}  seed={sd}]", flush=True)
            try:
                r = _run_djgp(ds, sd, cfg, npz_dir, device)
                if r:
                    allrows.append(r)
                    print(f"  DJGP      rmse={r['rmse']:.3f}  crps={r['crps']:.3f}  cov90={r['cov90']:.3f}")
            except Exception as e:
                print(f"  DJGP  ERROR: {e}", flush=True)
            try:
                for r in _run_baselines(ds, sd, npz_dir, device):
                    allrows.append(r)
                    print(f"  {r['method']:10s}rmse={r['rmse']:.3f}  crps={r['crps']:.3f}  cov90={r['cov90']:.3f}")
            except Exception as e:
                print(f"  baselines ERROR: {e}", flush=True)
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_METRIC_KEYS)
                w.writeheader()
                w.writerows(allrows)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s.  Results -> {out}")
    _print_table(allrows)


if __name__ == "__main__":
    main()

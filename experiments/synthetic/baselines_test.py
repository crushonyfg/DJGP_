"""
CLI entry for DKL and XGBoost-bootstrap baselines on synthetic folders containing dataset.pkl.

Return shape matches ``djgp.evaluation.RESULT_ROW_KEYS``::

    [rmse, mean_crps, wall_sec, coverage_90, mean_width_90, coverage_95, mean_width_95]

How to run (from repository root; conda env: jumpGP):

  conda activate jumpGP
  cd <path-to-DJGP-repo>

  # Deep Kernel Learning (SVGP)
  python experiments/synthetic/baselines_test.py --folder_name YOUR_DATA_FOLDER --method dkl

  # XGBoost bootstrap ensemble (epistemic + homoscedastic aleatoric for CRPS)
  python experiments/synthetic/baselines_test.py --folder_name YOUR_DATA_FOLDER --method xgb_boot --xgb_boots 30 --seed 0

  # Optional: more DKL training epochs
  python experiments/synthetic/baselines_test.py --folder_name YOUR_DATA_FOLDER --method dkl --dkl_epochs 80 --seed 0

Agents: when you change arguments or defaults here, update this docstring accordingly.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402

from shared.jumpgp_runner import load_dataset  # noqa: E402
from djgp.baselines import run_dkl_svgp_regression, run_xgboost_bootstrap_regression  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="DKL / XGBoost-bootstrap on synthetic dataset folder")
    p.add_argument("--folder_name", type=str, required=True)
    p.add_argument("--method", type=str, choices=("dkl", "xgb_boot"), default="dkl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--dkl_epochs", type=int, default=80)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

    dataset = load_dataset(args.folder_name)
    X_train = dataset["X_train"].detach().cpu().numpy()
    Y_train = dataset["Y_train"].detach().cpu().numpy().ravel()
    X_test = dataset["X_test"].detach().cpu().numpy()
    Y_test = dataset["Y_test"].detach().cpu().numpy().ravel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    if args.method == "dkl":
        rmse, crps, _dt, mu_np, sig_np = run_dkl_svgp_regression(
            X_train,
            Y_train,
            X_test,
            Y_test,
            device=device,
            epochs=args.dkl_epochs,
            seed=args.seed,
        )
        run_time = time.perf_counter() - t0
        out = pack_gaussian_uq_list(rmse, crps, run_time, Y_test, mu_np, sig_np)
        print(
            f"[baselines_test dkl] RMSE={rmse:.4f} CRPS={crps:.4f} "
            f"cov90={out[3]:.3f} cov95={out[5]:.3f} time={run_time:.2f}s"
        )
        return out

    rmse, crps, _elapsed_inner, details = run_xgboost_bootstrap_regression(
        X_train,
        Y_train,
        X_test,
        Y_test,
        n_bootstrap=args.xgb_boots,
        random_state=args.seed,
    )
    run_time = time.perf_counter() - t0
    out = pack_gaussian_uq_list(rmse, crps, run_time, Y_test, details["mu"], details["sigma"])
    print(
        f"[baselines_test xgb_boot] RMSE={rmse:.4f} CRPS={crps:.4f} "
        f"cov90={out[3]:.3f} cov95={out[5]:.3f} time={run_time:.2f}s"
    )
    return out


if __name__ == "__main__":
    main()

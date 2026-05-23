"""Part 2: Fit JumpGP directly on LH seed=0 latent (z_train, y_train) -> z_test.

Reports RMSE and mean CRPS on the original-scale y. M=35 (paper LH n).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("/sessions/elegant-eloquent-carson/mnt/DJGP")
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from JumpGaussianProcess.jumpgp import JumpGP, compute_metrics  # noqa: E402


def fit_and_score(x_train, y_train, x_test, y_test, M, mode="CEM"):
    t0 = time.perf_counter()
    jgp = JumpGP(
        np.asarray(x_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.float64).reshape(-1),
        np.asarray(x_test, dtype=np.float64),
        L=1,
        M=M,
        mode=mode,
        bVerbose=False,
    )
    jgp.fit()
    elapsed = time.perf_counter() - t0
    rmse, mean_crps = jgp.metrics(y_test)
    mu = np.asarray(jgp.jump_results["mu"]).ravel()
    sig2 = np.asarray(jgp.jump_results["sig2"]).ravel()
    sigma = np.sqrt(np.maximum(sig2, 1e-12))
    yt = np.asarray(y_test, dtype=np.float64).reshape(-1)
    z90 = 1.6448536269514722
    z95 = 1.959963984540054
    cov90 = float(np.mean((yt >= mu - z90 * sigma) & (yt <= mu + z90 * sigma)))
    cov95 = float(np.mean((yt >= mu - z95 * sigma) & (yt <= mu + z95 * sigma)))
    width90 = float(np.mean(2 * z90 * sigma))
    width95 = float(np.mean(2 * z95 * sigma))
    return {
        "M": int(M),
        "mode": mode,
        "rmse": float(rmse),
        "mean_crps": float(mean_crps),
        "wall_sec": float(elapsed),
        "coverage_90": cov90,
        "mean_width_90": width90,
        "coverage_95": cov95,
        "mean_width_95": width95,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["z", "X"], default="z")
    ap.add_argument("--M", type=int, default=35)
    ap.add_argument("--train_size", type=int, default=1000, choices=[200, 500, 1000])
    ap.add_argument("--mode", choices=["CEM", "VEM", "SEM"], default="CEM")
    args = ap.parse_args()

    base = PROJECT_ROOT / "experiments" / "synthetic"
    if args.train_size == 1000:
        p = base / "deep_mahalanobis_gp_paper_train1000_seed0" / "lh_q5_train1000_seed0.npz"
    else:
        p = base / "deep_mahalanobis_gp_paper_train200_500_5seeds" / f"lh_q5_train{args.train_size}_seed0.npz"
    d = np.load(p)

    if args.target == "z":
        x_tr, x_te = d["z_train"], d["z_test"]
    else:
        x_tr, x_te = d["X_train"], d["X_test"]
    y_tr, y_te = d["y_train"], d["y_test"]

    print(
        f"LH seed=0 train_size={args.train_size}: target={args.target} "
        f"x_tr={x_tr.shape} y_tr range=({y_tr.min():.2f},{y_tr.max():.2f}) M={args.M} mode={args.mode}",
        flush=True,
    )
    res = fit_and_score(x_tr, y_tr, x_te, y_te, M=args.M, mode=args.mode)
    print(json.dumps(res, indent=2))
    out = Path(f"/sessions/elegant-eloquent-carson/mnt/outputs/jgp_lh_seed0_train{args.train_size}_{args.target}_M{args.M}_{args.mode}.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

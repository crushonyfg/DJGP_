"""XGB-bootstrap + DKL-SVGP on PDEBench Burgers npz (train=1000, test<=200).

Matches the ISM runner's split exactly (test capped at 200, same near/far-threshold
subsets) so the baselines are directly comparable to
`experiments/synthetic/_exp_pdebench_hetnoise.py`. Original-scale metrics.
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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.baselines import run_dkl_svgp_regression, run_xgboost_bootstrap_regression  # noqa: E402
from experiments.synthetic._exp_u_ensemble_5seed import _crps_vec  # noqa: E402

NPZ = REPO / "experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517/data"
Z90 = 1.6448536269514722


def _subset_metrics(y, mu, sig, mask):
    y, mu, sig = y[mask], mu[mask], np.maximum(sig[mask], 1e-9)
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    crps = float(np.mean(_crps_vec(y, mu, sig)))
    cov = float(np.mean(np.abs((y - mu) / sig) <= Z90))
    return rmse, crps, cov


def run_one(seed, max_test, device, xgb_boots, dkl_epochs):
    d = np.load(NPZ / f"pdebench_burgers_train1000_seed{seed}.npz", allow_pickle=True)
    Xtr = np.asarray(d["X_train"], np.float64)
    Xte = np.asarray(d["X_test"], np.float64)[:max_test]
    ym = float(np.asarray(d["y_mean"]).reshape(-1)[0]); ys = float(np.asarray(d["y_scale"]).reshape(-1)[0])
    ytr = np.asarray(d["y_train_std"], np.float64).reshape(-1) * ys + ym
    yte = np.asarray(d["y_test"], np.float64).reshape(-1)[:max_test]
    tdist = np.asarray(d["threshold_distance_test"], np.float64).reshape(-1)[:max_test]
    delta = float(np.asarray(d["near_threshold_delta"]).reshape(-1)[0])
    near = tdist <= delta
    subsets = {"all": np.ones_like(near, bool), "near": near, "far": ~near}

    preds = {}
    _, _, _, det = run_xgboost_bootstrap_regression(Xtr, ytr, Xte, yte, n_bootstrap=xgb_boots, random_state=seed)
    preds["xgb_bootstrap"] = (np.asarray(det["mu"]).reshape(-1), np.asarray(det["sigma"]).reshape(-1))
    _, _, _, mu, sig = run_dkl_svgp_regression(Xtr, ytr, Xte, yte, device=device, epochs=dkl_epochs, seed=seed)
    preds["dkl_svgp"] = (np.asarray(mu).reshape(-1), np.asarray(sig).reshape(-1))

    rows = []
    for method, (mu, sig) in preds.items():
        for sub, mask in subsets.items():
            if mask.sum() == 0:
                continue
            rmse, crps, cov = _subset_metrics(yte, mu, sig, mask)
            rows.append(dict(seed=seed, method=method, subset=sub, n_pts=int(mask.sum()),
                             rmse=rmse, crps=crps, cov90=cov))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--dkl_epochs", type=int, default=80)
    p.add_argument("--out", default="experiments/synthetic/_exp_pdebench_baselines/metrics.csv")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    allrows = []; t0 = time.perf_counter()
    for sd in seeds:
        try:
            rows = run_one(sd, a.max_test, device, a.xgb_boots, a.dkl_epochs)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"ERROR seed{sd}: {type(exc).__name__}: {exc}", flush=True); continue
        allrows.extend(rows)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
        print(f"done seed{sd} ({time.perf_counter()-t0:.0f}s)", flush=True)

    agg = collections.defaultdict(list)
    for rr in allrows:
        agg[(rr["method"], rr["subset"])].append(rr)
    print("\nmethod          subset  rmse              crps              cov90")
    for key in sorted(agg):
        g = agg[key]
        def ms(k): v = np.array([x[k] for x in g]); return v.mean(), v.std()
        rm, rs = ms("rmse"); cm, cs = ms("crps"); vm, vs = ms("cov90")
        print(f"{key[0]:14s} {key[1]:5s}  {rm:6.3f}+/-{rs:5.3f}  {cm:6.3f}+/-{cs:5.3f}  {vm:.3f}+/-{vs:.3f}")


if __name__ == "__main__":
    main()

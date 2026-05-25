"""XGB-bootstrap + DKL-SVGP baselines on exported UCI npz splits (train=1000/test=200).

Reads the same `uci_{slug}_train{N}_seed{S}.npz` files the ISM runner uses (std_x done,
y standardised inside; baselines train on original-scale y reconstructed from
y_train_std*y_scale+y_mean and report original-scale RMSE/CRPS/cov90). Lets XGB/DKL be
compared to ISM-LMJGP on the identical subsampled split (e.g. the random-row Parkinsons).
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
from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402

SLUG = {"Wine Quality": "wine_quality",
        "Parkinsons Telemonitoring": "parkinsons_telemonitoring",
        "Appliances Energy Prediction": "appliances_energy_prediction"}


def run_one(dataset, seed, npz_dir, train_size, device, xgb_boots, dkl_epochs):
    d = np.load(Path(npz_dir) / f"uci_{SLUG[dataset]}_train{train_size}_seed{seed}.npz", allow_pickle=True)
    Xtr = np.asarray(d["X_train"], np.float64)
    Xte = np.asarray(d["X_test"], np.float64)
    ym = float(np.asarray(d["y_mean"]).reshape(-1)[0]); ys = float(np.asarray(d["y_scale"]).reshape(-1)[0])
    ytr = np.asarray(d["y_train_std"], np.float64).reshape(-1) * ys + ym   # original scale
    yte = np.asarray(d["y_test"], np.float64).reshape(-1)                   # original scale

    rows = []
    rmse, crps, el, det = run_xgboost_bootstrap_regression(Xtr, ytr, Xte, yte, n_bootstrap=xgb_boots, random_state=seed)
    uq = pack_gaussian_uq_list(rmse, crps, el, yte, det["mu"], det["sigma"])
    rows.append(dict(dataset=dataset, seed=seed, method="xgb_bootstrap",
                     rmse=float(uq[0]), crps=float(uq[1]), cov90=float(uq[3])))

    rmse, crps, dt, mu, sig = run_dkl_svgp_regression(Xtr, ytr, Xte, yte, device=device, epochs=dkl_epochs, seed=seed)
    uq = pack_gaussian_uq_list(rmse, crps, dt, yte, mu, sig)
    rows.append(dict(dataset=dataset, seed=seed, method="dkl_svgp",
                     rmse=float(uq[0]), crps=float(uq[1]), cov90=float(uq[3])))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="Parkinsons Telemonitoring")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--train_size", type=int, default=1000)
    p.add_argument("--npz_dir", default="experiments/uci/parkinsons_randomrow_train1000_5seeds/data")
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--dkl_epochs", type=int, default=80)
    p.add_argument("--out", default="experiments/uci/_exp_baselines_npz/metrics.csv")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    datasets = [x for x in a.datasets.split(",") if x]
    allrows = []; t0 = time.perf_counter()
    for ds in datasets:
        for sd in seeds:
            try:
                rows = run_one(ds, sd, a.npz_dir, a.train_size, device, a.xgb_boots, a.dkl_epochs)
            except Exception as exc:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"ERROR {ds} seed{sd}: {type(exc).__name__}: {exc}", flush=True); continue
            allrows.extend(rows)
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
            print(f"done {ds} seed{sd} ({time.perf_counter()-t0:.0f}s)", flush=True)

    agg = collections.defaultdict(list)
    for rr in allrows:
        agg[(rr["dataset"], rr["method"])].append(rr)
    print("\ndataset                         method          rmse              crps              cov90")
    for key in sorted(agg):
        g = agg[key]
        def ms(k): v = np.array([x[k] for x in g]); return v.mean(), v.std()
        rm, rs = ms("rmse"); cm, cs = ms("crps"); vm, vs = ms("cov90")
        print(f"{key[0]:31s} {key[1]:14s}  {rm:7.3f}+/-{rs:5.3f}  {cm:7.3f}+/-{cs:5.3f}  {vm:.3f}+/-{vs:.3f}")


if __name__ == "__main__":
    main()

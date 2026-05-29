"""Reproduce the DJGP synthetic benchmark.

Two datasets are used in the paper:
  * L2  — 2-D latent space, jump boundary from a 2-D phantom image, 20-D observed (RFF)
  * LH  — 5-D latent space with heteroscedastic jumps, 30-D observed (RFF)
Both use train=1000 / test=200.

Methods
-------
  DJGP        : our method (analytic predictive, top-4 ensemble)
  XGBoost     : bootstrap XGBoost ensemble (uncertainty via bootstrap spread)
  DKL-SVGP    : Deep Kernel Learning + Sparse Variational GP
  DeepGP      : Deep Gaussian Process via gpytorch (optional; skipped if gpytorch unavailable)

Metrics: RMSE / CRPS / cov90 (90% predictive interval coverage), mean ± std over seeds.

Usage
-----
Quick check (seed 0 only, ~5–10 min on GPU):
    python run_synthetic.py --seeds 0

Full 5-seed reproduction:
    python run_synthetic.py --seeds 0,1,2,3,4

Select methods:
    python run_synthetic.py --seeds 0 --methods DJGP,XGBoost

Results are written to --out (default: results/synthetic/metrics.csv).
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
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.baselines import run_dkl_svgp_regression, run_xgboost_bootstrap_regression  # noqa: E402
from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS, _generate_paper_data, _set_seed)
from experiments.synthetic._exp_u_ensemble_5seed import run_setting_seed  # noqa: E402

try:
    from shared.deepgp import DeepGP, train_model as _dgp_train
    from torch.utils.data import DataLoader, TensorDataset
    import gpytorch  # noqa: F401
    _DGP_OK = True
except Exception:
    _DGP_OK = False

# ---------------------------------------------------------------------------
# Pre-selected benchmark configurations (used for all reported results)
# ---------------------------------------------------------------------------
# Per-dataset DJGP recipe. The observed features are produced by one of four
# expansions of the latent (rff/poly/manifold/noise) at ambient dim --ambient;
# the concrete setting key is f"{fam}_d{ambient}_{expansion}" (registered by
# compare_paper_synthetic_baselines.register_grid_presets on import).
_CONFIGS = {
    "L2": {"fam": "l2", "n_neighbors": 25, "steps": 300, "noise_mode": "fixed"},
    "LH": {"fam": "lh", "n_neighbors": 35, "steps": 300, "noise_mode": "data_driven"},
}

_METRIC_KEYS = ["dataset", "expansion", "seed", "method", "rmse", "crps", "cov90"]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_djgp(ds, exp, setting, seed, cfg, device):
    rows = run_setting_seed(
        setting, seed,
        K=8, steps=cfg["steps"], n_neighbors=cfg["n_neighbors"], n_val=150,
        device=device, select_by="disagree",
        noise_mode=cfg["noise_mode"], rho_mode="near1")
    for r in rows:
        if r["head"] == "analytic" and r["topk"] == 4 and r["combiner"] == "mixture":
            return {"dataset": ds, "expansion": exp, "seed": seed, "method": "DJGP",
                    "rmse": r["rmse"], "crps": r["crps"], "cov90": r["cov90"]}
    return None


def _run_baselines(ds, exp, seed, preset, device, methods):
    _set_seed(seed)
    data = _generate_paper_data(preset, seed, device)
    sc = StandardScaler()
    Xtr = sc.fit_transform(data["X_train"])
    Xte = sc.transform(data["X_test"])
    ytr, yte = data["y_train"], data["y_test"]
    rows = []
    if "XGBoost" in methods:
        rmse, crps, el, det = run_xgboost_bootstrap_regression(
            Xtr, ytr, Xte, yte, n_bootstrap=30, random_state=seed)
        uq = pack_gaussian_uq_list(rmse, crps, el, yte, det["mu"], det["sigma"])
        rows.append({"dataset": ds, "expansion": exp, "seed": seed, "method": "XGBoost",
                     "rmse": float(uq[0]), "crps": float(uq[1]), "cov90": float(uq[3])})
    if "DKL-SVGP" in methods:
        rmse, crps, el, mu, sig = run_dkl_svgp_regression(
            Xtr, ytr, Xte, yte, device=device, epochs=80, seed=seed)
        uq = pack_gaussian_uq_list(rmse, crps, el, yte, mu, sig)
        rows.append({"dataset": ds, "expansion": exp, "seed": seed, "method": "DKL-SVGP",
                     "rmse": float(uq[0]), "crps": float(uq[1]), "cov90": float(uq[3])})
    if "DeepGP" in methods:
        if not _DGP_OK:
            print("  [DeepGP] skipped (gpytorch not available)", flush=True)
        else:
            try:
                row = _run_dgp(ds, seed, Xtr, ytr, Xte, yte, device)
                row["expansion"] = exp
                rows.append(row)
            except Exception as e:
                print(f"  [DeepGP] ERROR: {e}", flush=True)
    return rows


def _run_dgp(ds, seed, Xtr, ytr, Xte, yte, device):
    from argparse import Namespace
    from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
    ym, ys = ytr.mean(), max(ytr.std(), 1e-8)
    to32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    loader = DataLoader(TensorDataset(to32(Xtr), to32((ytr - ym) / ys)), batch_size=128, shuffle=True)
    te_loader = DataLoader(TensorDataset(to32(Xte), to32((yte - ym) / ys)), batch_size=128)
    model = DeepGP(to32(Xtr).shape, 10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, to32(Xtr).shape[0]))
    args = Namespace(num_epochs=300, patience=30, clip_grad=1.0, print_freq=9999, eval=True)
    _dgp_train(model, loader, te_loader, opt, mll, args)
    model.eval()
    with torch.no_grad():
        pm, pv, _ = model.predict(te_loader)
    mu = pm.mean(0).cpu().numpy() * ys + ym
    sigma = pv.mean(0).clamp_min(1e-12).sqrt().cpu().numpy() * ys
    from djgp.evaluation.crps import mean_gaussian_crps_torch
    rmse = float(np.sqrt(np.mean((mu - yte) ** 2)))
    crps = float(mean_gaussian_crps_torch(
        torch.as_tensor(yte, dtype=torch.float32),
        torch.as_tensor(mu, dtype=torch.float32),
        torch.as_tensor(sigma, dtype=torch.float32)).item())
    uq = pack_gaussian_uq_list(rmse, crps, 0.0, yte, mu, sigma)
    return {"dataset": ds, "seed": seed, "method": "DeepGP",
            "rmse": float(uq[0]), "crps": float(uq[1]), "cov90": float(uq[3])}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(rows, expansions):
    agg: dict = collections.defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r.get("expansion", "rff"), r["method"])].append(r)
    print(f"\n{'Dataset':8s}  {'Expansion':10s}  {'Method':12s}  {'RMSE':>16}  {'CRPS':>16}  {'cov90':>10}")
    print("-" * 82)
    for ds in ("L2", "LH"):
        for exp in expansions:
            for mth in ("DJGP", "XGBoost", "DKL-SVGP", "DeepGP"):
                g = agg.get((ds, exp, mth))
                if not g:
                    continue
                def ms(k):  # noqa: E306
                    v = np.array([x[k] for x in g])
                    return f"{v.mean():.3f}±{v.std():.3f}"
                print(f"{ds:8s}  {exp:10s}  {mth:12s}  {ms('rmse'):>16}  {ms('crps'):>16}  {ms('cov90'):>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DJGP synthetic benchmark")
    p.add_argument("--datasets", default="L2,LH",
                   help="comma-separated list from: L2, LH")
    p.add_argument("--expansions", default="rff,poly,manifold,noise",
                   help="comma-separated observed-feature expansions of the latent: "
                        "rff, poly, manifold, noise")
    p.add_argument("--ambient", type=int, default=30,
                   help="ambient observed dimension D (grid preset {fam}_d{D}_{exp}); "
                        "L2 supports 30, LH supports 30 or 50")
    p.add_argument("--methods", default="DJGP,XGBoost,DKL-SVGP",
                   help="comma-separated list from: DJGP, XGBoost, DKL-SVGP, DeepGP")
    p.add_argument("--seeds", default="0",
                   help="comma-separated seeds (e.g. 0,1,2,3,4 for full reproduction)")
    p.add_argument("--out", default="results/synthetic/metrics.csv")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [s for s in a.datasets.split(",") if s]
    expansions = [s for s in a.expansions.split(",") if s]
    methods = [s for s in a.methods.split(",") if s]
    seeds = [int(s) for s in a.seeds.split(",") if s]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    allrows: list[dict] = []
    t0 = time.perf_counter()

    for ds in datasets:
        cfg = _CONFIGS[ds]
        for exp in expansions:
            setting = f"{cfg['fam']}_d{a.ambient}_{exp}"
            if setting not in SETTING_PRESETS:
                print(f"\n[skip {ds}/{exp}] no preset '{setting}' "
                      f"(L2 supports d30; LH supports d30/d50)", flush=True)
                continue
            preset = dict(SETTING_PRESETS[setting])
            for sd in seeds:
                print(f"\n[{ds}/{exp}  d{a.ambient}  seed={sd}]", flush=True)
                if "DJGP" in methods:
                    try:
                        r = _run_djgp(ds, exp, setting, sd, cfg, device)
                        if r:
                            allrows.append(r)
                            print(f"  DJGP      rmse={r['rmse']:.3f}  crps={r['crps']:.3f}  cov90={r['cov90']:.3f}")
                    except Exception as e:
                        print(f"  DJGP  ERROR: {e}")
                bl = [m for m in methods if m != "DJGP"]
                if bl:
                    try:
                        for r in _run_baselines(ds, exp, sd, preset, device, bl):
                            allrows.append(r)
                            print(f"  {r['method']:10s}rmse={r['rmse']:.3f}  crps={r['crps']:.3f}  cov90={r['cov90']:.3f}")
                    except Exception as e:
                        print(f"  baselines ERROR: {e}")
                # incremental save
                with out.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=_METRIC_KEYS)
                    w.writeheader()
                    w.writerows(allrows)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s.  Results -> {out}")
    _print_table(allrows, expansions)


if __name__ == "__main__":
    main()

"""ISM-LMJGP with learnable / heteroscedastic observation noise on UCI.

Goal: fix the analytic-direct undercoverage flagged in
``docs/ism_lmjgp_analytic.md`` 8c-calib (the obs-noise pins at the fixed 0.25
prior, so sigma is uniformly too small). The ``data_driven`` noise mode anchors
each anchor's noise prior at a nearest-neighbour-difference estimate of the local
out-of-sample residual scale (heteroscedastic, Bayesian: log-Gaussian prior +
``obs_noise_max`` cap), letting the learnable noise grow to the residual scale.

Per (dataset, seed): train K analytic members (PLS/PCA/PCA-scaled + random
Stiefel), full-batch; predict analytic-direct + sample-W CEM on the full test
set; select members by UNSUPERVISED head-disagreement (``RMSE(mu_analytic,
mu_sampleW)/y_std``); report top-1/2/4 rmse/crps/cov90 (mixture + robust
combiners) for both heads. Defaults: Q=5, n=25, Wine steps=500, Parkinsons
steps=200 (the good settings from the doc).

Wine/Appliances random-row, Parkinsons SUBJECT-GROUPED; std_x and y standardised
inside the local head (original-scale metrics). Offline npz exports.
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from djgp.projections.structured_metric_lmjgp import (  # noqa: E402
    AnalyticStructuredMetricLMJGPConfig, build_fixed_local_inducing_from_w0,
    predict_analytic_structured_metric_lmjgp, train_analytic_structured_metric_lmjgp)
from experiments.synthetic._exp_u_ensemble_5seed import (  # noqa: E402
    _metrics_from_members, _metrics_robust, _sample_w_predict_raw, _zero_slopes)
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import (  # noqa: E402
    _neighbor_stack, _projection_W0, _run_meanw_cem_init, _set_seed)

NPZ_DIR = REPO / "experiments/uci/dmgp_smalltrain_5seeds_20260517/data"
SLUG = {"Wine Quality": "wine_quality",
        "Parkinsons Telemonitoring": "parkinsons_telemonitoring",
        "Appliances Energy Prediction": "appliances_energy_prediction"}
NMAP = {"Wine Quality": 25, "Parkinsons Telemonitoring": 25, "Appliances Energy Prediction": 35}
STEPMAP = {"Wine Quality": 500, "Parkinsons Telemonitoring": 200, "Appliances Energy Prediction": 200}


def _rho_kwargs(rho_mode: str) -> dict:
    if rho_mode == "near1":
        # rho ~ 1: a strong prior + high floor forces a near-all-inlier local head
        # (effectively disables the robust outlier mixture).
        return dict(rho_init=0.97, rho_prior_strength=10.0, min_rho_mean=0.9, min_rho_penalty_weight=50.0)
    if rho_mode == "free":
        # free rho: weak prior, low floor; lets the gate find a low/heterogeneous rho.
        return dict(rho_init=0.8, rho_prior_strength=0.1, min_rho_mean=0.05, min_rho_penalty_weight=0.0)
    return dict(rho_init=0.8, rho_prior_strength=2.0, min_rho_mean=0.55, min_rho_penalty_weight=50.0)


def _make_config(steps, Q, T, seed, noise_mode, nd_frac, nd_prior_std, nd_prior_weight, rho_mode):
    kw = dict(steps=steps, lr=0.01, batch_anchors=T, trace_target=float(Q),
              trace_normalize=True, eval_interval=max(1, steps), restore_state="final", seed=seed)
    kw.update(_rho_kwargs(rho_mode))
    if noise_mode == "data_driven":
        kw.update(obs_noise_prior_mode="data_driven", obs_noise_data_driven_frac=nd_frac,
                  obs_noise_prior_std=nd_prior_std, obs_noise_prior_weight=nd_prior_weight,
                  obs_noise_max=1.5, init_noise_at_prior=True)
    return AnalyticStructuredMetricLMJGPConfig(**kw)


def run_dataset_seed(dataset, seed, K, steps, Q, n_neighbors, device, noise_mode,
                     nd_frac, nd_prior_std, nd_prior_weight, rho_mode, npz_dir=NPZ_DIR, train_size=1000):
    _set_seed(seed)
    npz = np.load(Path(npz_dir) / f"uci_{SLUG[dataset]}_train{train_size}_seed{seed}.npz", allow_pickle=True)
    Xtr_s = np.asarray(npz["X_train"], np.float64)
    Xte_s = np.asarray(npz["X_test"], np.float64)
    ytr_s = np.asarray(npz["y_train_std"], np.float64).reshape(-1)
    yte = np.asarray(npz["y_test"], np.float64).reshape(-1)
    ym = float(np.asarray(npz["y_mean"]).reshape(-1)[0])
    ys = float(np.asarray(npz["y_scale"]).reshape(-1)[0])

    Xtr_t = torch.as_tensor(Xtr_s, device=device, dtype=torch.float64)
    Xte_t = torch.as_tensor(Xte_s, device=device, dtype=torch.float64)
    ytr_s_t = torch.as_tensor(ytr_s, device=device, dtype=torch.float64)
    T = Xte_t.shape[0]
    Xn, yn, _ = _neighbor_stack(Xte_t, Xtr_t, ytr_s_t, n_neighbors=n_neighbors)
    if yn.dim() == 3:
        yn = yn.squeeze(-1)

    inits = [_projection_W0(Xtr_s, ytr_s, Q, m, seed=seed + 513) for m in ["pls", "pca", "pca_scaled"][:K]]
    r = 0
    while len(inits) < K:
        rr = np.random.RandomState(seed + 1000 + r)
        W = rr.randn(Q, Xtr_s.shape[1]); W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
        inits.append(W); r += 1

    cem_state, _ = _run_meanw_cem_init(Xte_t, Xn, yn, torch.as_tensor(inits[0], device=device, dtype=torch.float64))
    cem_state["gate"] = _zero_slopes(cem_state["gate"])

    an_mu, an_sd, sw_mu, sw_sd = [], [], [], []
    noise_med, prior_med = [], []
    for ki, W0n in enumerate(inits):
        W0 = torch.as_tensor(W0n, device=device, dtype=torch.float64)
        C, mu_u0 = build_fixed_local_inducing_from_w0(Xte_t, Xn, yn, W0, m_inducing=10)
        cfg = _make_config(steps, Q, T, seed + 211 + 17 * ki, noise_mode,
                           nd_frac, nd_prior_std, nd_prior_weight, rho_mode)
        res = train_analytic_structured_metric_lmjgp(Xte_t, Xn, yn, C, init_W=W0, init_mu_u=mu_u0, config=cfg)
        mu_a, var_a, _ = predict_analytic_structured_metric_lmjgp(Xte_t, res, trace_target=float(Q), trace_normalize=True)
        an_mu.append(mu_a.detach().cpu().numpy() * ys + ym)
        an_sd.append(var_a.detach().sqrt().cpu().numpy() * abs(ys))
        mu_s, var_s = _sample_w_predict_raw(res, Xte_t, Xn, yn, cem_state, float(Q), S=3, seed=seed + ki)
        sw_mu.append(mu_s.detach().cpu().numpy() * ys + ym)
        sw_sd.append(var_s.detach().sqrt().cpu().numpy() * abs(ys))
        noise_med.append(float(res.log_noise_var.exp().sqrt().median().item()) * abs(ys))
        prior_med.append(float(res.diagnostics.get("noise_prior_anchor_median", float("nan"))) * abs(ys))

    an_mu = np.stack(an_mu); an_sd = np.stack(an_sd); sw_mu = np.stack(sw_mu); sw_sd = np.stack(sw_sd)
    disagree = np.sqrt(((an_mu - sw_mu) ** 2).mean(axis=1)) / (abs(ys) + 1e-12)
    order = np.argsort(disagree)

    rows = []
    for topk in (1, 2, 4):
        sel = order[:topk]
        for head, M, S in (("analytic", an_mu, an_sd), ("sampleW", sw_mu, sw_sd)):
            for comb, fn in (("mixture", _metrics_from_members), ("robust", _metrics_robust)):
                rmse, crps, cov = fn(yte, M, S, sel)
                rows.append(dict(dataset=dataset, seed=seed, noise_mode=noise_mode, rho_mode=rho_mode,
                                 head=head, topk=topk, combiner=comb, Q=Q, steps=steps,
                                 rmse=rmse, crps=crps, cov90=cov,
                                 noise_std_med=float(np.median(noise_med)),
                                 prior_anchor_med=float(np.nanmedian(prior_med))))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="Wine Quality,Parkinsons Telemonitoring")
    p.add_argument("--seeds", default="0")
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--Q", type=int, default=5)
    p.add_argument("--noise_mode", default="data_driven", choices=("fixed", "data_driven"))
    p.add_argument("--nd_frac", type=float, default=1.0, help="scale on the NN-difference residual estimate")
    p.add_argument("--nd_prior_std", type=float, default=0.3, help="log-space prior std for data-driven noise")
    p.add_argument("--nd_prior_weight", type=float, default=3.0)
    p.add_argument("--rho_mode", default="default", choices=("default", "near1", "free"))
    p.add_argument("--steps_override", type=int, default=-1)
    p.add_argument("--npz_dir", default=str(NPZ_DIR), help="directory holding uci_{slug}_train{N}_seed{N}.npz")
    p.add_argument("--train_size", type=int, default=1000)
    p.add_argument("--out", default="experiments/synthetic/_exp_uci_hetnoise/metrics.csv")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    datasets = [x for x in a.datasets.split(",") if x]
    allrows = []; t0 = time.perf_counter()
    for ds in datasets:
        steps = a.steps_override if a.steps_override > 0 else STEPMAP[ds]
        for sd in seeds:
            try:
                rows = run_dataset_seed(ds, sd, a.K, steps, a.Q, NMAP[ds], device, a.noise_mode,
                                        a.nd_frac, a.nd_prior_std, a.nd_prior_weight, a.rho_mode,
                                        npz_dir=a.npz_dir, train_size=a.train_size)
            except Exception as exc:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"ERROR {ds} seed{sd}: {type(exc).__name__}: {exc}", flush=True); continue
            allrows.extend(rows)
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
            print(f"done {ds} seed{sd} noise={a.noise_mode} rho={a.rho_mode} ({time.perf_counter()-t0:.0f}s)", flush=True)

    agg = collections.defaultdict(list)
    for rr in allrows:
        agg[(rr["dataset"], rr["head"], rr["combiner"], rr["topk"])].append(rr)
    print(f"\n=== noise_mode={a.noise_mode} rho_mode={a.rho_mode} Q={a.Q} (mean+/-sd over seeds) ===")
    print("dataset                         head     comb     topk  rmse              crps              cov90        noise/anchor")
    for key in sorted(agg):
        g = agg[key]
        def ms(k): v = np.array([x[k] for x in g]); return v.mean(), v.std()
        rm, rs = ms("rmse"); cm, cs = ms("crps"); vm, vs = ms("cov90")
        nstd = np.mean([x["noise_std_med"] for x in g]); panc = np.nanmean([x["prior_anchor_med"] for x in g])
        print(f"{key[0]:31s} {key[1]:8s} {key[2]:8s} {key[3]:4d}  {rm:7.3f}+/-{rs:5.3f}  {cm:7.3f}+/-{cs:5.3f}  "
              f"{vm:.3f}+/-{vs:.3f}  {nstd:.3f}/{panc:.3f}")


if __name__ == "__main__":
    main()

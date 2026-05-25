"""ISM-LMJGP on the PDEBench Burgers ShockJump surrogate (train=1000, test<=200).

Reuses the analytic ISM pipeline (head-disagreement selection, top-1/2/4, mixture +
robust combiners, fixed vs data-driven heteroscedastic noise) on the offline
Burgers npz exports (D=128, std_x done, y standardised inside the local head). Also
reports the near-threshold / far-from-threshold subsets (the important shock stress
test, per docs/pdebench_burgers_surrogate.md).

Baselines (train=1000, all-test reference, from
experiments/pdebench/burgers_train200_500_1000_xgb_dkl_lmjgp_5seeds_20260517):
  XGB 0.702/0.309/0.923 · LMJGP predict_vi 0.755/0.262/0.842 · DKL 0.893/0.459/0.666.
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
from experiments.synthetic._exp_uci_hetnoise import _rho_kwargs  # noqa: E402
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import (  # noqa: E402
    _neighbor_stack, _projection_W0, _run_meanw_cem_init, _set_seed)

NPZ = REPO / "experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517/data"


def _make_config(steps, Q, T, seed, noise_mode, nd_frac, nd_prior_std, nd_prior_weight, rho_mode):
    kw = dict(steps=steps, lr=0.01, batch_anchors=T, trace_target=float(Q),
              trace_normalize=True, eval_interval=max(1, steps), restore_state="final", seed=seed)
    kw.update(_rho_kwargs(rho_mode))
    if noise_mode == "data_driven":
        kw.update(obs_noise_prior_mode="data_driven", obs_noise_data_driven_frac=nd_frac,
                  obs_noise_prior_std=nd_prior_std, obs_noise_prior_weight=nd_prior_weight,
                  obs_noise_max=1.5, init_noise_at_prior=True)
    return AnalyticStructuredMetricLMJGPConfig(**kw)


def run_seed(seed, K, steps, Q, n_neighbors, max_test, device, noise_mode,
             nd_frac, nd_prior_std, nd_prior_weight, rho_mode):
    _set_seed(seed)
    d = np.load(NPZ / f"pdebench_burgers_train1000_seed{seed}.npz", allow_pickle=True)
    Xtr_s = np.asarray(d["X_train"], np.float64)
    Xte_s = np.asarray(d["X_test"], np.float64)[:max_test]
    ytr_s = np.asarray(d["y_train_std"], np.float64).reshape(-1)
    yte = np.asarray(d["y_test"], np.float64).reshape(-1)[:max_test]
    ym = float(np.asarray(d["y_mean"]).reshape(-1)[0]); ys = float(np.asarray(d["y_scale"]).reshape(-1)[0])
    tdist = np.asarray(d["threshold_distance_test"], np.float64).reshape(-1)[:max_test]
    delta = float(np.asarray(d["near_threshold_delta"]).reshape(-1)[0])
    near = tdist <= delta

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

    an_mu, an_sd, sw_mu, sw_sd, noise_med = [], [], [], [], []
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

    an_mu = np.stack(an_mu); an_sd = np.stack(an_sd); sw_mu = np.stack(sw_mu); sw_sd = np.stack(sw_sd)
    disagree = np.sqrt(((an_mu - sw_mu) ** 2).mean(axis=1)) / (abs(ys) + 1e-12)
    order = np.argsort(disagree)

    subsets = {"all": np.ones_like(near, bool), "near": near, "far": ~near}
    rows = []
    for topk in (1, 2, 4):
        sel = order[:topk]
        for head, M, S in (("analytic", an_mu, an_sd), ("sampleW", sw_mu, sw_sd)):
            for comb, fn in (("mixture", _metrics_from_members), ("robust", _metrics_robust)):
                for sub, mask in subsets.items():
                    if mask.sum() == 0:
                        continue
                    rmse, crps, cov = fn(yte[mask], M[:, mask], S[:, mask], sel)
                    rows.append(dict(seed=seed, noise_mode=noise_mode, rho_mode=rho_mode, steps=steps, Q=Q,
                                     head=head, topk=topk, combiner=comb, subset=sub,
                                     n_pts=int(mask.sum()), rmse=rmse, crps=crps, cov90=cov,
                                     noise_std_med=float(np.median(noise_med))))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0")
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--Q", type=int, default=3)
    p.add_argument("--n_neighbors", type=int, default=25)
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--noise_mode", default="fixed", choices=("fixed", "data_driven"))
    p.add_argument("--nd_frac", type=float, default=1.0)
    p.add_argument("--nd_prior_std", type=float, default=0.3)
    p.add_argument("--nd_prior_weight", type=float, default=3.0)
    p.add_argument("--rho_mode", default="near1", choices=("default", "near1", "free"))
    p.add_argument("--out", default="experiments/synthetic/_exp_pdebench_hetnoise/metrics.csv")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    allrows = []; t0 = time.perf_counter()
    for sd in seeds:
        try:
            rows = run_seed(sd, a.K, a.steps, a.Q, a.n_neighbors, a.max_test, device, a.noise_mode,
                            a.nd_frac, a.nd_prior_std, a.nd_prior_weight, a.rho_mode)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"ERROR seed{sd}: {type(exc).__name__}: {exc}", flush=True); continue
        allrows.extend(rows)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
        print(f"done seed{sd} noise={a.noise_mode} steps={a.steps} ({time.perf_counter()-t0:.0f}s)", flush=True)

    agg = collections.defaultdict(list)
    for rr in allrows:
        agg[(rr["subset"], rr["head"], rr["combiner"], rr["topk"])].append(rr)
    print(f"\n=== Burgers noise={a.noise_mode} steps={a.steps} Q={a.Q} rho={a.rho_mode} (mean+/-sd over seeds) ===")
    print("subset head     comb     topk  rmse              crps              cov90        noiseStd")
    for key in sorted(agg):
        g = agg[key]
        def ms(k): v = np.array([x[k] for x in g]); return v.mean(), v.std()
        rm, rs = ms("rmse"); cm, cs = ms("crps"); vm, vs = ms("cov90")
        nstd = np.mean([x["noise_std_med"] for x in g])
        print(f"{key[0]:5s} {key[1]:8s} {key[2]:8s} {key[3]:4d}  {rm:6.3f}+/-{rs:5.3f}  {cm:6.3f}+/-{cs:5.3f}  "
              f"{vm:.3f}+/-{vs:.3f}  {nstd:.3f}")


if __name__ == "__main__":
    main()

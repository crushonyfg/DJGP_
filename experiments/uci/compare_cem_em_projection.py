"""
Ablation of the **CEM-aligned EM projection learner** on UCI regression data.

The procedure (see :mod:`djgp.projections.cem_em_projection`) alternates::

    C-step : run JumpGP-CEM on Z_j = X_j W_j^T, freeze (r_j, w_j, logtheta_j, ms_j)
    M-step : update W = (W_0, {C_j}) via Adam to maximise the per-anchor
             GP log marginal on inliers + λ_g · (-BCE(gate)) - λ_s · smoothness
             - λ_r · row-norm penalty.

We compare four projection variants under an identical CEM-JumpGP evaluator and
the same neighbourhoods (so the only difference is the per-anchor ``W_j``):

    pls_only      : W_j = W_pls          (deterministic supervised baseline)
    cem_em_coeff  : W_j = W_0 + C_j V^T  (recommended low-rank EM)
    cem_em_free   : W_j (free per anchor)= W_0 + C_j I_D  (V = I_D, upper bound)
    lmjgp_predict_vi (optional reference): the existing variational ``train_vi +
        predict_vi`` API on the same (X_train, X_test, neighbourhoods).

All methods report the standard 7-tuple
``[rmse, mean_crps, wall_sec, cov90, w90, cov95, w95]`` plus per-anchor
diagnostics on ``W`` (trace/eff-rank/anchor dispersion/cosine to the PLS reference).

How to run (repository root, conda env: ``jumpGP``)::

    conda activate jumpGP

    # Smoke test on Wine (1 seed, 32 anchors)
    python experiments/uci/compare_cem_em_projection.py \
        --num_exp 1 --max_anchors 32 --datasets "Wine Quality" \
        --n_outer 3 --n_m_steps 50 --include_lmjgp 0

    # Full sweep including LMJGP reference and free per-anchor upper bound
    python experiments/uci/compare_cem_em_projection.py \
        --num_exp 3 --datasets "Wine Quality,Parkinsons Telemonitoring" \
        --n_outer 5 --n_m_steps 100 --lambda_gate 0.1 --lambda_smooth 0.1 \
        --lambda_scale 0.01 --include_lmjgp 1 --include_free 1 \
        --out uci_cemem_3seeds.pkl --out_json uci_cemem_3seeds.json

Agents: keep CLI flags / variant names aligned with
``compare_lowrank_factorization.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from experiments.uci.uci_data import load_uci_split  # noqa: E402
from shared.jumpgp_runner import compute_metrics, find_neighborhoods  # noqa: E402
from djgp.diagnostics.projection_posterior import sanitize_for_json  # noqa: E402
from djgp.evaluation import mean_gaussian_crps_torch  # noqa: E402
from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    compute_pls_W0,
    per_anchor_W_diagnostics,
    run_projected_jumpgp,
    train_projection_cem_em,
)
from djgp.variational import predict_vi, predict_vi_decomposed, qW_from_qV, train_vi  # noqa: E402


# =====================================================================
# Region builder + LMJGP reference (mirrors compare_lowrank_factorization.py)
# =====================================================================
def _build_neighborhoods(X_test, X_train, Y_train, device, m1: int, Q: int, n: int):
    neighborhoods = find_neighborhoods(X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n)
    regions = []
    X_nb_list, y_nb_list = [], []
    T = X_test.shape[0]
    for i in range(T):
        X_nb = neighborhoods[i]["X_neighbors"].to(device)
        y_nb = neighborhoods[i]["y_neighbors"].to(device)
        regions.append({"X": X_nb, "y": y_nb, "C": torch.randn(m1, Q, device=device)})
        X_nb_list.append(X_nb)
        y_nb_list.append(y_nb)
    return regions, X_nb_list, y_nb_list


def _train_lmjgp_reference(
    X_train, Y_train, X_test, Y_test,
    *,
    Q: int, m1: int, m2: int, n: int, num_steps: int, lr: float, MC_num: int,
    device: torch.device,
    vi_decomposition: bool = False,
):
    T, D = X_test.shape
    regions, X_nb_list, y_nb_list = _build_neighborhoods(X_test, X_train, Y_train, device, m1, Q, n)
    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
    }
    u_params = []
    for _ in range(T):
        u_params.append(
            {
                "U_logit": torch.zeros(1, device=device, requires_grad=True),
                "mu_u": torch.randn(m1, device=device, requires_grad=True),
                "Sigma_u": torch.eye(m1, device=device, requires_grad=True),
                "sigma_noise": torch.tensor(0.5, device=device, requires_grad=True),
                "sigma_k": torch.tensor(0.5, device=device, requires_grad=True),
                "omega": torch.randn(Q + 1, device=device, requires_grad=True),
            }
        )
    X_train_mean = X_train.mean(dim=0)
    X_train_std = X_train.std(dim=0)
    Z = X_train_mean + torch.randn(m2, D, device=device) * X_train_std
    hyperparams = {
        "Z": Z,
        "X_test": X_test,
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    t0 = time.perf_counter()
    V_params, u_params, hyperparams = train_vi(
        regions=regions, V_params=V_params, u_params=u_params, hyperparams=hyperparams,
        lr=lr, num_steps=num_steps, log_interval=max(50, num_steps // 5),
    )
    train_time = time.perf_counter() - t0
    mu_W, _ = qW_from_qV(
        X_test, hyperparams["Z"],
        V_params["mu_V"], V_params["sigma_V"],
        hyperparams["lengthscales"], hyperparams["var_w"],
    )
    t1 = time.perf_counter()
    if vi_decomposition:
        mu_pred, var_pred, within_var, between_var = predict_vi_decomposed(
            regions, V_params, hyperparams, M=MC_num,
        )
    else:
        mu_pred, var_pred = predict_vi(regions, V_params, hyperparams, M=MC_num)
        within_var = between_var = None
    pred_time = time.perf_counter() - t1
    sigmas_pv = torch.sqrt(torch.clamp(var_pred, min=1e-12))
    rmse_pv, crps_pv = compute_metrics(mu_pred, sigmas_pv, Y_test)
    pv_block = {
        "mu": mu_pred.detach(),
        "sigma": sigmas_pv.detach(),
        "rmse": float(rmse_pv.detach().item()),
        "mean_crps": float(crps_pv.detach().item()),
        "wall_sec": float(pred_time),
    }
    if vi_decomposition and within_var is not None:
        pv_block["var_within"] = within_var.detach()
        pv_block["var_between"] = between_var.detach()
    return {
        "X_neighbors_list": X_nb_list,
        "y_neighbors_list": y_nb_list,
        "mu_W": mu_W.detach(),
        "predict_vi": pv_block,
        "train_time_sec": float(train_time),
    }


# =====================================================================
# CEM-EM variant trainer + evaluator
# =====================================================================
def _run_cem_em_variant(
    *,
    label: str,
    V_t: torch.Tensor,  # [D, Rv]   (use I_D for free mode)
    W_pls_t: torch.Tensor,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    X_anchors: torch.Tensor,
    y_test_np: np.ndarray,
    Q: int,
    cfg: CemEmConfig,
    device: torch.device,
    init_C_scale: float,
    coeff_scale: float,
    row_normalize_W: bool,
):
    T = X_anchors.shape[0]
    D = X_anchors.shape[1]
    model = CoeffProjection(
        T=T, Q=Q, D=D, V=V_t, W0_init=W_pls_t, learn_W0=True,
        init_C_scale=init_C_scale,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).to(device)
    # Only set the auto row-norm² target when the penalty is *actually used*, i.e.
    # row_normalize_W is False AND lambda_scale > 0. Otherwise leave at default 1.0.
    cfg_local = type(cfg)(**{**cfg.__dict__})
    if (not row_normalize_W) and cfg_local.lambda_scale > 0 and cfg_local.target_row_norm_sq <= 1.0:
        med_row_sq = float((W_pls_t * W_pls_t).sum(dim=-1).median().item())
        cfg_local.target_row_norm_sq = max(med_row_sq, 1e-6)
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_neighbors_list,
        y_neighbors_list=y_neighbors_list,
        X_anchors=X_anchors,
        device=device,
        cfg=cfg_local,
        label=label,
    )
    out = run_projected_jumpgp(
        res.W,
        X_anchors,
        X_neighbors_list,
        y_neighbors_list,
        device=device,
        progress=True,
        desc=label,
    )
    mu_np = out["mu"]
    sig_np = out["sigma"]
    rmse = float(np.sqrt(np.mean((mu_np - y_test_np) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.from_numpy(y_test_np).float(),
            torch.from_numpy(mu_np).float(),
            torch.from_numpy(sig_np).float(),
        ).item()
    )
    metrics_row = pack_gaussian_uq_list(rmse, crps, float(out["elapsed_sec"]), y_test_np, mu_np, sig_np)
    diag = per_anchor_W_diagnostics(res.W, W_ref=W_pls_t)
    diag["train_time_sec"] = float(res.train_time_sec)
    diag["num_trainable_params"] = int(res.num_trainable_params)
    diag["snapshots"] = res.snapshots
    diag["history"] = res.history
    diag["jgp_elapsed_sec"] = float(out["elapsed_sec"])
    diag["best_outer"] = int(res.best_outer)
    diag["best_metric_value"] = float(res.best_metric_value)
    diag["coeff_scale"] = float(coeff_scale)
    diag["row_normalize_W"] = bool(row_normalize_W)
    return metrics_row, diag, res.W


def _run_pls_only(
    *,
    W_pls_t: torch.Tensor,
    X_anchors: torch.Tensor,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    y_test_np: np.ndarray,
    device: torch.device,
):
    T = X_anchors.shape[0]
    W = W_pls_t.unsqueeze(0).expand(T, -1, -1).contiguous()
    out = run_projected_jumpgp(
        W, X_anchors, X_neighbors_list, y_neighbors_list,
        device=device, progress=True, desc="pls_only",
    )
    mu_np = out["mu"]
    sig_np = out["sigma"]
    rmse = float(np.sqrt(np.mean((mu_np - y_test_np) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.from_numpy(y_test_np).float(),
            torch.from_numpy(mu_np).float(),
            torch.from_numpy(sig_np).float(),
        ).item()
    )
    metrics_row = pack_gaussian_uq_list(rmse, crps, float(out["elapsed_sec"]), y_test_np, mu_np, sig_np)
    diag = per_anchor_W_diagnostics(W, W_ref=W_pls_t)
    diag["jgp_elapsed_sec"] = float(out["elapsed_sec"])
    return metrics_row, diag, W


# =====================================================================
# Per-(dataset, seed) orchestration
# =====================================================================
def _per_dataset_run(
    dataset_name: str,
    seed: int,
    *,
    Q_override: int | None,
    n_pls: int,
    n_pca: int,
    n_random: int,
    n: int,
    m1: int,
    cfg: CemEmConfig,
    init_C_scale: float,
    coeff_scale: float,
    row_normalize_W: bool,
    include_free: bool,
    include_lmjgp: bool,
    lmjgp_steps: int,
    MC_num: int,
    max_anchors: int | None,
    device: torch.device,
):
    X_train_np, X_test_np, y_train_np, y_test_np_full, K = load_uci_split(dataset_name, seed)
    Q = Q_override if Q_override is not None else K

    X_train = torch.from_numpy(X_train_np).float().to(device)
    Y_train = torch.from_numpy(y_train_np).float().squeeze(-1).to(device)
    X_test = torch.from_numpy(X_test_np).float().to(device)
    Y_test = torch.from_numpy(y_test_np_full).float().squeeze(-1).to(device)
    if max_anchors is not None:
        k = min(max_anchors, X_test.shape[0])
        X_test = X_test[:k]
        Y_test = Y_test[:k]
    y_test_np = Y_test.detach().cpu().numpy()

    V_np, V_info = build_V_basis(
        X_train_np, y_train_np.ravel(),
        n_pls=n_pls, n_pca=n_pca, n_random=n_random, seed=seed,
    )
    V_t = torch.from_numpy(V_np).float().to(device)
    W_pls_np = compute_pls_W0(X_train_np, y_train_np.ravel(), Q=Q)
    W_pls_t = torch.from_numpy(W_pls_np).float().to(device)
    I_D = torch.eye(X_test.shape[1], device=device)

    # --- LMJGP reference (its own neighbourhoods, but on the same X_test slice) ---
    metrics: dict[str, list[float]] = {}
    diagnostics: dict[str, object] = {
        "V_basis": V_info,
        "Q": int(Q),
        "D": int(X_test.shape[1]),
    }
    W_per_variant: dict[str, torch.Tensor] = {}
    lmjgp_block = None
    X_nb_list = None
    y_nb_list = None
    if include_lmjgp:
        lmjgp_block = _train_lmjgp_reference(
            X_train, Y_train, X_test, Y_test,
            Q=Q, m1=m1, m2=40, n=n, num_steps=lmjgp_steps, lr=0.01, MC_num=MC_num,
            device=device,
        )
        X_nb_list = lmjgp_block["X_neighbors_list"]
        y_nb_list = lmjgp_block["y_neighbors_list"]
        pv = lmjgp_block["predict_vi"]
        metrics["lmjgp_predict_vi"] = pack_gaussian_uq_list(
            pv["rmse"], pv["mean_crps"], pv["wall_sec"],
            y_test_np, pv["mu"].cpu().numpy(), pv["sigma"].cpu().numpy(),
        )
        diagnostics["lmjgp_mu_W"] = per_anchor_W_diagnostics(lmjgp_block["mu_W"], W_ref=W_pls_t)
        diagnostics["lmjgp_train_time_sec"] = lmjgp_block["train_time_sec"]
    else:
        _, X_nb_list, y_nb_list = _build_neighborhoods(X_test, X_train, Y_train, device, m1, Q, n)

    # --- pls_only ---
    pls_row, pls_diag, W_pls_full = _run_pls_only(
        W_pls_t=W_pls_t,
        X_anchors=X_test,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        y_test_np=y_test_np,
        device=device,
    )
    metrics["pls_only"] = pls_row
    diagnostics["pls_only"] = pls_diag
    W_per_variant["pls_only"] = W_pls_full

    # --- cem_em_coeff (recommended) ---
    coeff_row, coeff_diag, W_coeff = _run_cem_em_variant(
        label="cem_em_coeff",
        V_t=V_t,
        W_pls_t=W_pls_t,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        X_anchors=X_test,
        y_test_np=y_test_np,
        Q=Q,
        cfg=cfg,
        device=device,
        init_C_scale=init_C_scale,
        coeff_scale=coeff_scale,
        row_normalize_W=row_normalize_W,
    )
    metrics["cem_em_coeff"] = coeff_row
    diagnostics["cem_em_coeff"] = coeff_diag
    W_per_variant["cem_em_coeff"] = W_coeff

    # --- cem_em_free (V = I_D upper bound) ---
    if include_free:
        free_row, free_diag, W_free = _run_cem_em_variant(
            label="cem_em_free",
            V_t=I_D,
            W_pls_t=W_pls_t,
            X_neighbors_list=X_nb_list,
            y_neighbors_list=y_nb_list,
            X_anchors=X_test,
            y_test_np=y_test_np,
            Q=Q,
            cfg=cfg,
            device=device,
            init_C_scale=init_C_scale,
            coeff_scale=coeff_scale,
            row_normalize_W=row_normalize_W,
        )
        metrics["cem_em_free"] = free_row
        diagnostics["cem_em_free"] = free_diag
        W_per_variant["cem_em_free"] = W_free

    return {
        "label": f"uci:{dataset_name}",
        "dataset": dataset_name,
        "seed": int(seed),
        "Q": int(Q),
        "Rv": int(V_t.shape[1]),
        "D": int(X_test.shape[1]),
        "max_anchors_if_subsampled": max_anchors,
        "n_test_anchors_used": int(X_test.shape[0]),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "W_per_variant_shapes": {k: list(W.shape) for k, W in W_per_variant.items()},
    }


# =====================================================================
# CLI / printing
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="CEM-aligned EM projection ablation on UCI.")
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--datasets", type=str, default="Wine Quality")
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--n", type=int, default=20, help="Neighborhood size.")
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--n_outer", type=int, default=3)
    p.add_argument("--n_m_steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lambda_gate", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.05)
    p.add_argument("--lambda_scale", type=float, default=0.01)
    p.add_argument("--init_C_scale", type=float, default=0.0,
                   help="Std of N(0, σ²) noise added to C at init (0 = pure W_pls start).")
    p.add_argument("--coeff_scale", type=float, default=1.0,
                   help="α multiplier on C_j V^T inside W_j = rowNorm(W_0 + α C_j V^T).")
    p.add_argument("--row_normalize_W", type=int, default=1,
                   help="If 1, every row of W_j is L2-normalised (recommended; lambda_scale should be 0).")
    p.add_argument("--normalize_smoothness", type=int, default=1,
                   help="If 1, divide Laplacian smoothness by Σ K_jj' (T-invariant).")
    p.add_argument("--track_best_W", type=int, default=1,
                   help="If 1, keep the W with lowest M-step loss across outer iterations.")
    p.add_argument("--include_free", type=int, default=0, help="Also run cem_em_free (V = I_D).")
    p.add_argument("--include_lmjgp", type=int, default=0)
    p.add_argument("--lmjgp_steps", type=int, default=200)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--max_anchors", type=int, default=None)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--out_json", type=str, default="")
    return p.parse_args()


def _print_metrics(dataset: str, seed: int, methods: dict[str, list[float]]) -> None:
    print(f"\n[{dataset}] seed={seed}")
    print("  method                rmse    crps    wall    cov90   w90    cov95   w95")
    for name, row in methods.items():
        vals = [f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else "  nan " for v in row]
        formatted = "  ".join(vals)
        print(f"  {name:<22}{formatted}")


def _print_diag(dataset: str, seed: int, diagnostics: dict) -> None:
    print(f"\n[{dataset}] seed={seed} diagnostics")
    skip_keys = {"V_basis", "Q", "D", "lmjgp_train_time_sec"}
    for name, diag in diagnostics.items():
        if name in skip_keys or not isinstance(diag, dict):
            continue
        printable = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in diag.items()
            if k not in {"snapshots", "history"}
        }
        print(f"  {name:<22} {json.dumps(sanitize_for_json(printable), indent=None)}")
    if "V_basis" in diagnostics:
        v = diagnostics["V_basis"]
        print(f"  V_basis                Rv={v.get('Rv')} (pls={v.get('n_pls')}, pca={v.get('n_pca')}, random={v.get('n_random')})")


def _print_snapshots(dataset: str, seed: int, diagnostics: dict) -> None:
    relevant = {k: v for k, v in diagnostics.items() if k.startswith("cem_em_") and isinstance(v, dict) and v.get("snapshots")}
    if not relevant:
        return
    print(f"\n[{dataset}] seed={seed} CEM-EM snapshots")
    for name, diag in relevant.items():
        for snap in diag["snapshots"]:
            row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in snap.items() if k != "label"}
            print(f"  {name:<22} step={int(row.get('step', -1)):>2d}  {json.dumps(row)}")


def main():
    args = parse_args()
    dataset_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    cfg = CemEmConfig(
        n_outer=args.n_outer,
        n_m_steps=args.n_m_steps,
        lr=args.lr,
        lambda_gate=args.lambda_gate,
        lambda_smooth=args.lambda_smooth,
        lambda_scale=args.lambda_scale,
        normalize_smoothness=bool(args.normalize_smoothness),
        track_best_W=bool(args.track_best_W),
        snapshot_after_each_outer=True,
        cem_progress=False,
        verbose=True,
    )

    total_res: dict = {}
    for dataset_name in dataset_list:
        total_res[dataset_name] = {}
        for seed in range(args.num_exp):
            res = _per_dataset_run(
                dataset_name,
                seed,
                Q_override=args.Q,
                n_pls=args.n_pls,
                n_pca=args.n_pca,
                n_random=args.n_random,
                n=args.n,
                m1=args.m1,
                cfg=cfg,
                init_C_scale=args.init_C_scale,
                coeff_scale=args.coeff_scale,
                row_normalize_W=bool(args.row_normalize_W),
                include_free=bool(args.include_free),
                include_lmjgp=bool(args.include_lmjgp),
                lmjgp_steps=args.lmjgp_steps,
                MC_num=args.MC_num,
                max_anchors=args.max_anchors,
                device=device,
            )
            total_res[dataset_name][seed] = res
            _print_metrics(dataset_name, seed, res["metrics"])
            _print_diag(dataset_name, seed, res["diagnostics"])
            _print_snapshots(dataset_name, seed, res["diagnostics"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"uci_cemem_projection_{stamp}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(total_res, f)
    print(f"\nSaved pickle to {out_path}")

    if args.out_json:
        def _strip(d):
            if isinstance(d, dict):
                return {k: _strip(v) for k, v in d.items() if k not in {"W"}}
            if isinstance(d, list):
                return [_strip(x) for x in d]
            return d

        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(_strip(total_res)), f, indent=2)
        print(f"Saved JSON to {args.out_json}")

    return total_res


if __name__ == "__main__":
    main()

"""
Ablation of the **low-rank tensor factorisation** ``W_j = (C_0 + ΔC_j) V^T``
on UCI regression data.

Three variants are compared back-to-back, all sharing the same neighbourhoods,
JumpGP-CEM evaluator, and metric row format
``[rmse, mean_crps, wall_sec, cov90, w90, cov95, w95]``::

    pls_only       : W_j = W_pls (frozen, no learning)
    global_C       : W_j = C_0 V^T, C_0 ∈ R^{Q × Rv} learned globally
    local_C_smooth : W_j = (C_0 + ΔC_j) V^T, per-anchor ΔC_j with anchor-graph
                     Laplacian smoothness penalty (deterministic surrogate for a
                     GP prior over c_{q,ℓ}(·))

For reference we additionally run the existing LMJGP entry point
(``train_vi`` + ``predict_vi``) under the same Q on the same anchors. Existing
``compare_uci_baselines.py`` and ``compare_metric_variants.py`` are not modified.

How to run (repository root, conda env: ``jumpGP``)::

    conda activate jumpGP

    # Smoke test on Wine (1 seed, 32 anchors)
    python experiments/uci/compare_lowrank_factorization.py \
        --num_exp 1 --num_steps 200 --max_anchors 32 --datasets "Wine Quality" \
        --include_lmjgp 1 --lmjgp_steps 200

    # Full sweep
    python experiments/uci/compare_lowrank_factorization.py --num_exp 3 \
        --datasets "Wine Quality,Parkinsons Telemonitoring" \
        --n_pls 5 --n_pca 5 --n_random 0 --num_steps 300 --include_lmjgp 1

Agents: update this docstring when CLI flags or default budgets change.
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
    LOWRANK_MODES,
    LowRankConfig,
    LowRankProjectionModel,
    build_V_basis,
    compute_pls_W0,
    per_anchor_W_diagnostics,
    run_projected_jumpgp,
    train_lowrank_projection,
)
from djgp.variational import predict_vi, qW_from_qV, train_vi  # noqa: E402


# =====================================================================
# Region builder (reuses the same neighborhood logic as the metric variants)
# =====================================================================
def _build_regions(X_test, X_train, Y_train, device, m1: int, Q: int, n: int):
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


def _stack_neighborhoods(X_nb_list, y_nb_list, device):
    """Pad/check that all neighborhoods have the same n; stack into [T, n, D] / [T, n]."""
    T = len(X_nb_list)
    n = X_nb_list[0].shape[0]
    D = X_nb_list[0].shape[1]
    X_stack = torch.empty(T, n, D, device=device)
    y_stack = torch.empty(T, n, device=device)
    for t in range(T):
        if X_nb_list[t].shape != (n, D):
            raise ValueError(f"Neighborhood {t} has shape {tuple(X_nb_list[t].shape)} != ({n},{D}).")
        X_stack[t] = X_nb_list[t]
        y_stack[t] = y_nb_list[t].view(-1)
    return X_stack, y_stack


# =====================================================================
# Optional LMJGP reference (kept tiny so it doesn't dominate runtime)
# =====================================================================
def _train_lmjgp_reference(
    X_train,
    Y_train,
    X_test,
    Y_test,
    *,
    Q: int,
    m1: int,
    m2: int,
    n: int,
    num_steps: int,
    lr: float,
    MC_num: int,
    device: torch.device,
):
    T, D = X_test.shape
    regions, X_nb_list, y_nb_list = _build_regions(X_test, X_train, Y_train, device, m1, Q, n)

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

    t_train = time.perf_counter()
    V_params, u_params, hyperparams = train_vi(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=lr,
        num_steps=num_steps,
        log_interval=max(50, num_steps // 5),
    )
    train_time = time.perf_counter() - t_train

    mu_W, _ = qW_from_qV(
        X_test,
        hyperparams["Z"],
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
    )

    t_pred = time.perf_counter()
    mu_pred, var_pred = predict_vi(regions, V_params, hyperparams, M=MC_num)
    pred_time = time.perf_counter() - t_pred
    sigmas_pv = torch.sqrt(torch.clamp(var_pred, min=1e-12))
    rmse_pv, crps_pv = compute_metrics(mu_pred, sigmas_pv, Y_test)

    return {
        "regions": regions,
        "X_neighbors_list": X_nb_list,
        "y_neighbors_list": y_nb_list,
        "mu_W": mu_W.detach(),
        "predict_vi": {
            "mu": mu_pred.detach(),
            "sigma": sigmas_pv.detach(),
            "rmse": float(rmse_pv.detach().item()),
            "mean_crps": float(crps_pv.detach().item()),
            "wall_sec": float(pred_time),
        },
        "train_time_sec": float(train_time),
    }


# =====================================================================
# Per-variant low-rank projection trainer + evaluator
# =====================================================================
def _train_and_eval_variant(
    *,
    mode: str,
    V_t: torch.Tensor,  # [D, Rv]
    W_pls_t: torch.Tensor,  # [Q, D]
    X_test: torch.Tensor,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    Q: int,
    num_steps: int,
    lr: float,
    smoothness_lambda: float,
    snapshot_steps: list[int],
    device: torch.device,
    desc_prefix: str,
) -> dict[str, object]:
    T = X_test.shape[0]
    D = X_test.shape[1]
    Rv = V_t.shape[1]
    cfg = LowRankConfig(
        mode=mode,
        Q=Q,
        Rv=Rv,
        smoothness_lambda=float(smoothness_lambda),
    )
    model = LowRankProjectionModel(
        cfg, V_t, T=T, D=D, W_pls=W_pls_t, device=device,
    )
    X_stack, y_stack = _stack_neighborhoods(X_neighbors_list, y_neighbors_list, device)
    train_res = train_lowrank_projection(
        model,
        X_neighbors_stack=X_stack,
        y_neighbors_stack=y_stack,
        X_anchors=X_test,
        num_steps=num_steps if mode != "pls_only" else min(num_steps, 50),
        lr=lr,
        log_interval=max(50, num_steps // 5),
        snapshot_steps=snapshot_steps,
        verbose=True,
    )
    W = train_res.W  # [T, Q, D]

    out = run_projected_jumpgp(
        W,
        X_test,
        X_neighbors_list,
        y_neighbors_list,
        device=device,
        progress=True,
        desc=f"{desc_prefix}/{mode}",
    )
    mu_np = out["mu"]
    sig_np = out["sigma"]
    y_np = X_test.new_zeros(T).cpu().numpy()  # placeholder; caller passes y separately
    return {
        "mode": mode,
        "W": W.detach(),
        "train_time_sec": float(train_res.train_time_sec),
        "history": train_res.history,
        "snapshots": train_res.snapshots,
        "num_trainable_params": int(train_res.num_trainable_params),
        "log_lengthscale": float(train_res.log_lengthscale),
        "log_noise_var": float(train_res.log_noise_var),
        "smoothness_lambda": float(train_res.smoothness_lambda),
        "jgp_mu": mu_np,
        "jgp_sigma": sig_np,
        "jgp_elapsed_sec": float(out["elapsed_sec"]),
        "_y_placeholder": y_np,
    }


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
    num_steps: int,
    lr: float,
    smoothness_lambda: float,
    snapshot_steps: list[int],
    max_anchors: int | None,
    include_lmjgp: bool,
    lmjgp_steps: int,
    MC_num: int,
    device: torch.device,
    n: int = 20,
    m1: int = 2,
):
    X_train_np, X_test_np, y_train_np, y_test_np, K = load_uci_split(dataset_name, seed)
    Q = Q_override if Q_override is not None else K

    X_train = torch.from_numpy(X_train_np).float().to(device)
    Y_train = torch.from_numpy(y_train_np).float().squeeze(-1).to(device)
    X_test = torch.from_numpy(X_test_np).float().to(device)
    Y_test = torch.from_numpy(y_test_np).float().squeeze(-1).to(device)
    if max_anchors is not None:
        k = min(max_anchors, X_test.shape[0])
        X_test = X_test[:k]
        Y_test = Y_test[:k]

    # Build the fixed input-side basis V from PLS + PCA + random in raw-X scale.
    V_np, V_info = build_V_basis(
        X_train_np, y_train_np.ravel(),
        n_pls=n_pls, n_pca=n_pca, n_random=n_random, seed=seed,
    )
    V_t = torch.from_numpy(V_np).float().to(device)

    # PLS init for W_0 (rows = top-Q PLS directions in raw-X scale).
    W_pls_np = compute_pls_W0(X_train_np, y_train_np.ravel(), Q=Q)
    W_pls_t = torch.from_numpy(W_pls_np).float().to(device)

    # Optional LMJGP reference (uses its own neighborhoods & training, but on the same X_test).
    lmjgp_block = None
    lmjgp_pred = None
    X_nb_list = None
    y_nb_list = None
    if include_lmjgp:
        lmjgp_block = _train_lmjgp_reference(
            X_train, Y_train, X_test, Y_test,
            Q=Q, m1=m1, m2=40, n=n, num_steps=lmjgp_steps, lr=0.01, MC_num=MC_num,
            device=device,
        )
        lmjgp_pred = lmjgp_block["predict_vi"]
        X_nb_list = lmjgp_block["X_neighbors_list"]
        y_nb_list = lmjgp_block["y_neighbors_list"]
    else:
        # We still need neighborhoods for the low-rank variants.
        _, X_nb_list, y_nb_list = _build_regions(X_test, X_train, Y_train, device, m1, Q, n)

    y_test_np_arr = Y_test.detach().cpu().numpy()

    method_results: dict[str, list[float]] = {}
    diagnostics: dict[str, object] = {
        "V_basis": V_info,
        "Q": int(Q),
        "D": int(X_test.shape[1]),
    }
    variant_blocks: dict[str, dict] = {}

    if lmjgp_pred is not None:
        method_results["lmjgp_predict_vi"] = pack_gaussian_uq_list(
            lmjgp_pred["rmse"],
            lmjgp_pred["mean_crps"],
            lmjgp_pred["wall_sec"],
            y_test_np_arr,
            lmjgp_pred["mu"].cpu().numpy(),
            lmjgp_pred["sigma"].cpu().numpy(),
        )
        # Also report the LMJGP mu_W diagnostic so we can compare anchor variation.
        diagnostics["lmjgp_mu_W"] = per_anchor_W_diagnostics(
            lmjgp_block["mu_W"], W_ref=W_pls_t,
        )
        diagnostics["lmjgp_train_time_sec"] = lmjgp_block["train_time_sec"]

    for mode in LOWRANK_MODES:
        block = _train_and_eval_variant(
            mode=mode,
            V_t=V_t,
            W_pls_t=W_pls_t,
            X_test=X_test,
            X_neighbors_list=X_nb_list,
            y_neighbors_list=y_nb_list,
            Q=Q,
            num_steps=num_steps,
            lr=lr,
            smoothness_lambda=smoothness_lambda,
            snapshot_steps=snapshot_steps,
            device=device,
            desc_prefix=f"{dataset_name}/seed={seed}",
        )
        mu_np = block["jgp_mu"]
        sig_np = block["jgp_sigma"]
        rmse = float(np.sqrt(np.mean((mu_np - y_test_np_arr) ** 2)))
        crps = float(
            mean_gaussian_crps_torch(
                torch.from_numpy(y_test_np_arr).float(),
                torch.from_numpy(mu_np).float(),
                torch.from_numpy(sig_np).float(),
            ).item()
        )
        method_results[f"lowrank_{mode}"] = pack_gaussian_uq_list(
            rmse, crps, block["jgp_elapsed_sec"], y_test_np_arr, mu_np, sig_np,
        )
        diag = per_anchor_W_diagnostics(block["W"], W_ref=W_pls_t)
        diag["train_time_sec"] = block["train_time_sec"]
        diag["num_trainable_params"] = block["num_trainable_params"]
        diag["log_lengthscale"] = block["log_lengthscale"]
        diag["log_noise_var"] = block["log_noise_var"]
        diag["snapshots"] = block["snapshots"]
        diag["smoothness_lambda"] = block["smoothness_lambda"]
        diagnostics[f"lowrank_{mode}"] = diag
        variant_blocks[mode] = {
            "history": block["history"],
            "snapshots": block["snapshots"],
            "train_time_sec": block["train_time_sec"],
        }

    return {
        "label": f"uci:{dataset_name}",
        "dataset": dataset_name,
        "seed": int(seed),
        "Q": int(Q),
        "Rv": int(V_t.shape[1]),
        "D": int(X_test.shape[1]),
        "max_anchors_if_subsampled": max_anchors,
        "n_test_anchors_used": int(X_test.shape[0]),
        "metrics": method_results,
        "diagnostics": diagnostics,
        "variant_blocks": variant_blocks,
    }


# =====================================================================
# CLI / printing
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Low-rank tensor factorisation ablation for the LMJGP first-layer projection."
    )
    p.add_argument("--num_exp", type=int, default=1, help="Number of seeds per dataset.")
    p.add_argument(
        "--datasets",
        type=str,
        default="Wine Quality",
        help="Comma-separated UCI dataset names (see experiments/uci/uci_data.py).",
    )
    p.add_argument("--Q", type=int, default=None, help="Override latent dimension Q (default: dataset K).")
    p.add_argument("--n_pls", type=int, default=5, help="PLS directions to seed V.")
    p.add_argument("--n_pca", type=int, default=5, help="PCA directions to seed V.")
    p.add_argument("--n_random", type=int, default=0, help="Random directions to seed V (orthonormalised together with PLS/PCA).")
    p.add_argument("--num_steps", type=int, default=200, help="Adam steps for low-rank variants.")
    p.add_argument("--lr", type=float, default=0.01, help="Adam learning rate for low-rank variants.")
    p.add_argument(
        "--smoothness_lambda",
        type=float,
        default=0.05,
        help="λ for the anchor-graph Laplacian smoothness penalty in local_C_smooth.",
    )
    p.add_argument(
        "--snapshot_steps",
        type=str,
        default="0,10,50,200",
        help="Comma-separated step indices to snapshot W diagnostics (will be clipped to [0, num_steps]).",
    )
    p.add_argument(
        "--max_anchors",
        type=int,
        default=None,
        help="Use only the first K test anchors (faster; per-anchor JumpGP scales linearly).",
    )
    p.add_argument("--include_lmjgp", type=int, default=0, help="Run train_vi + predict_vi as a reference (1) or skip (0).")
    p.add_argument("--lmjgp_steps", type=int, default=200, help="train_vi steps for the LMJGP reference.")
    p.add_argument("--MC_num", type=int, default=3, help="MC samples for predict_vi.")
    p.add_argument("--out", type=str, default="", help="Pickle output path.")
    p.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional JSON path for diagnostics + metrics (NaN/Inf -> null).",
    )
    return p.parse_args()


def _print_summary_row(dataset: str, seed: int, methods: dict[str, list[float]]) -> None:
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
        if name in skip_keys:
            continue
        if not isinstance(diag, dict):
            continue
        printable = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in diag.items()
            if k not in {"snapshots"}
        }
        print(f"  {name:<22} {json.dumps(sanitize_for_json(printable), indent=None)}")
    if "V_basis" in diagnostics:
        v = diagnostics["V_basis"]
        print(f"  V_basis                Rv={v.get('Rv')} (pls={v.get('n_pls')}, pca={v.get('n_pca')}, random={v.get('n_random')})")


def _print_snapshots(dataset: str, seed: int, diagnostics: dict) -> None:
    relevant = {k: v for k, v in diagnostics.items() if k.startswith("lowrank_") and isinstance(v, dict) and v.get("snapshots")}
    if not relevant:
        return
    print(f"\n[{dataset}] seed={seed} snapshots")
    for name, diag in relevant.items():
        snaps = diag.get("snapshots", [])
        for snap in snaps:
            row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in snap.items()}
            print(f"  {name:<22} step={int(row.get('step', -1)):>4d}  {json.dumps(row)}")


def main():
    args = parse_args()
    dataset_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    snapshot_steps = sorted({int(s.strip()) for s in args.snapshot_steps.split(",") if s.strip()})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

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
                num_steps=args.num_steps,
                lr=args.lr,
                smoothness_lambda=args.smoothness_lambda,
                snapshot_steps=snapshot_steps,
                max_anchors=args.max_anchors,
                include_lmjgp=bool(args.include_lmjgp),
                lmjgp_steps=args.lmjgp_steps,
                MC_num=args.MC_num,
                device=device,
            )
            total_res[dataset_name][seed] = res
            _print_summary_row(dataset_name, seed, res["metrics"])
            _print_diag(dataset_name, seed, res["diagnostics"])
            _print_snapshots(dataset_name, seed, res["diagnostics"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"uci_lowrank_factorization_{stamp}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(total_res, f)
    print(f"\nSaved pickle to {out_path}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            # Strip W tensors before JSON serialisation.
            def _strip(d):
                if isinstance(d, dict):
                    return {k: _strip(v) for k, v in d.items() if k not in {"W", "_y_placeholder"}}
                if isinstance(d, list):
                    return [_strip(x) for x in d]
                return d

            json.dump(sanitize_for_json(_strip(total_res)), f, indent=2)
        print(f"Saved JSON to {args.out_json}")

    return total_res


if __name__ == "__main__":
    main()

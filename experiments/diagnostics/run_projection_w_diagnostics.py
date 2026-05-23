"""
Train LMJGP once, then dump projection-posterior diagnostics (appendix / ablation).

Builds on q(W) from q(V): full covariance `cov_W`, mean `mu_W`, diagonal std `sigma_W`.
Exports four groups (see `experiments/diagnostics/README.md`):

1. **EW_raw** — spectral stats of **E[W]** (kappa / effective rank are ``nan`` when E[W]≈0;
   not misleading zeros).
2. **metric_E_WtW** — moments **E[W^T W]** and **E[W W^T]** (Mahalanobis / latent-side metric).
3. **kernel_metric_A** — local RBF Gram condition numbers from expected squared Mahalanobis
   distance ``delta^T E[W^T W] delta``, plus ``K + jitter I`` and ``K + jitter I + sigma_noise^2 I``.
4. **sampled_W_row_normalized** / **kernel_sampled_row_normalized** — MC draws with the same
   row-normalization as ``predict_vi``, plus per-anchor kernel diagnostics aligned with prediction.

Optional: `--max_anchors K` uses only the first K test points (faster; full test set makes predict_vi slow).

How to run (repository root, conda env jumpGP):

  conda activate jumpGP
  cd <path-to-DJGP-repo>

  # UCI — Wine Quality, fast VI (adjust --num_steps for paper-quality)
  python experiments/diagnostics/run_projection_w_diagnostics.py --mode uci --dataset "Wine Quality" --seed 0 --num_steps 500 --max_anchors 64 --out diag_wine.json

  # Synthetic folder with dataset.pkl (same layout as DJGP_test)
  python experiments/diagnostics/run_projection_w_diagnostics.py --mode synthetic --folder_name YOUR_FOLDER --num_steps 300 --max_anchors 64 --out diag_synth.json

  # More MC draws for normalized-W spectral summaries
  python experiments/diagnostics/run_projection_w_diagnostics.py --mode uci --dataset "Wine Quality" --seed 0 --num_steps 200 --mc_samples 64 --max_anchors 32 --out diag.json

Agents: update this docstring when CLI or defaults change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from experiments.uci.uci_data import load_uci_split  # noqa: E402
from shared.jumpgp_runner import compute_metrics, find_neighborhoods, load_dataset  # noqa: E402
from djgp.diagnostics.projection_posterior import (  # noqa: E402
    attach_abs_errors,
    compute_full_projection_diagnostics,
    flatten_per_anchor_for_export,
    sanitize_for_json,
)
from djgp.variational import predict_vi, qW_from_qV, train_vi  # noqa: E402


def _build_regions(X_test, X_train, Y_train, device, m1: int, Q: int, n: int):
    neighborhoods = find_neighborhoods(X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n)
    regions = []
    T = X_test.shape[0]
    z_neighbors_list: list[torch.Tensor | None] = []
    Z_train = getattr(_build_regions, "_Z_train", None)
    for i in range(T):
        X_nb = neighborhoods[i]["X_neighbors"].to(device)
        y_nb = neighborhoods[i]["y_neighbors"].to(device)
        idx = neighborhoods[i].get("indices")
        regions.append({"X": X_nb, "y": y_nb, "C": torch.randn(m1, Q, device=device)})
        if Z_train is not None and idx is not None:
            z_neighbors_list.append(Z_train[idx].to(device))
        else:
            z_neighbors_list.append(None)
    return regions, z_neighbors_list


def run_uci(
    dataset_name: str,
    seed: int,
    device: torch.device,
    num_steps: int,
    mc_samples: int,
    Q_override: int | None,
    max_anchors: int | None,
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

    _build_regions._Z_train = None

    return _train_and_diag(
        X_train,
        Y_train,
        X_test,
        Y_test,
        Q,
        device,
        num_steps,
        mc_samples,
        label=f"uci:{dataset_name}",
        z_neighbors_list=None,
        max_anchors_note=max_anchors,
    )


def run_synthetic(folder_name: str, device: torch.device, num_steps: int, mc_samples: int, Q_override: int | None, max_anchors: int | None):
    ds = load_dataset(folder_name)
    X_train = ds["X_train"].to(device).float()
    Y_train = ds["Y_train"].to(device).float().squeeze(-1)
    X_test = ds["X_test"].to(device).float()
    Y_test = ds["Y_test"].to(device).float().squeeze(-1)
    if max_anchors is not None:
        k = min(max_anchors, X_test.shape[0])
        X_test = X_test[:k]
        Y_test = Y_test[:k]

    args_dict = ds.get("args") or {}
    Q = Q_override if Q_override is not None else int(args_dict.get("Q", args_dict.get("latent_dim", 3)))

    Z_train = ds.get("Z_train")
    if Z_train is not None:
        Z_train = Z_train.to(device).float()
    _build_regions._Z_train = Z_train

    return _train_and_diag(
        X_train,
        Y_train,
        X_test,
        Y_test,
        Q,
        device,
        num_steps,
        mc_samples,
        label=f"synthetic:{folder_name}",
        z_neighbors_list=None,
        max_anchors_note=max_anchors,
    )


def _train_and_diag(
    X_train,
    Y_train,
    X_test,
    Y_test,
    Q: int,
    device: torch.device,
    num_steps: int,
    mc_samples: int,
    label: str,
    z_neighbors_list: list | None,
    max_anchors_note: int | None = None,
):
    T, D = X_test.shape
    m1, m2, n = 2, 40, 20

    regions, zn_auto = _build_regions(X_test, X_train, Y_train, device, m1, Q, n)
    if z_neighbors_list is None:
        z_neighbors_list = zn_auto

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
        lr=0.01,
        num_steps=num_steps,
        log_interval=max(50, num_steps // 5),
    )
    train_time = time.perf_counter() - t_train

    mu_W, cov_W = qW_from_qV(
        X_test,
        hyperparams["Z"],
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
    )
    sigma_W = torch.sqrt(torch.clamp(cov_W.diagonal(dim1=2, dim2=3), min=1e-24))

    sigma_k_stack = torch.stack([u["sigma_k"] for u in u_params]).view(-1)
    sigma_noise_stack = torch.stack([u["sigma_noise"] for u in u_params]).view(-1)

    t_pred = time.perf_counter()
    mu_pred, var_pred = predict_vi(regions, V_params, hyperparams, M=3)
    pred_time = time.perf_counter() - t_pred
    sigmas = torch.sqrt(var_pred)
    rmse, mean_crps = compute_metrics(mu_pred, sigmas, Y_test)

    abs_err = (mu_pred - Y_test).abs().detach().cpu().numpy()

    projection = compute_full_projection_diagnostics(
        mu_W,
        sigma_W,
        cov_W,
        regions,
        sigma_k_stack,
        sigma_noise_stack,
        z_neighbors_list=z_neighbors_list,
        mc_M=mc_samples,
        mc_seed=0,
    )
    attach_abs_errors(projection["per_anchor"], abs_err)

    out = {
        "label": label,
        "train_time_sec": train_time,
        "predict_time_sec": pred_time,
        "rmse": float(rmse.detach().cpu().item()),
        "mean_crps": float(mean_crps.detach().cpu().item()),
        "hyper": {
            "Q": Q,
            "m1": m1,
            "m2": m2,
            "n_neighbors": n,
            "num_steps": num_steps,
            "max_anchors_if_subsampled": max_anchors_note,
            "mc_samples_projection": mc_samples,
        },
        **projection,
        "per_anchor_flat": flatten_per_anchor_for_export(projection["per_anchor"]),
    }

    preview = {k: out[k] for k in ("EW_raw", "metric_E_WtW", "kernel_metric_A") if k in out}
    print(json.dumps(sanitize_for_json(preview), indent=2))
    print(f"RMSE={out['rmse']:.6f} CRPS={out['mean_crps']:.6f} train_s={train_time:.2f}")

    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("uci", "synthetic"), required=True)
    p.add_argument("--dataset", type=str, default="Wine Quality")
    p.add_argument("--folder_name", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--mc_samples", type=int, default=32)
    p.add_argument("--max_anchors", type=int, default=None, help="Use only first K test anchors (faster)")
    p.add_argument("--Q", type=int, default=None, help="Override latent rows Q")
    p.add_argument("--out", type=str, default="", help="JSON output path")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if args.mode == "uci":
        result = run_uci(args.dataset, args.seed, device, args.num_steps, args.mc_samples, args.Q, args.max_anchors)
    else:
        if not args.folder_name:
            raise SystemExit("--folder_name required for synthetic mode")
        result = run_synthetic(args.folder_name, device, args.num_steps, args.mc_samples, args.Q, args.max_anchors)

    out_path = args.out or f"projection_w_diag_{args.mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(result), f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

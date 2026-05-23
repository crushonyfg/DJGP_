"""
Compare full transductive LMJGP with low-rank residual transductive LMJGP.

The low-rank residual model fixes a deterministic projection mean ``W0`` and
learns a GP field only over low-rank coefficients:

    W(x) = W0 + C(x) V^T.

This keeps the original LMJGP expected-kernel ELBO valid because the induced
``q(W_j)`` remains Gaussian, but reduces the random projection dimension from
``Q*D`` to ``Q*R``.

How to run (repository root, conda env: ``jumpGP``):

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke test.
    python experiments/uci/compare_lowrank_residual_lmjgp.py ^
        --datasets "Wine Quality" --max_total_train 120 --max_test_anchors 8 ^
        --n 12 --lmjgp_steps 10 --MC_num 2 ^
        --out_dir experiments/uci/lowrank_residual_lmjgp_smoke

    # One-seed pilot.
    python experiments/uci/compare_lowrank_residual_lmjgp.py ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --max_total_train 500 --max_test_anchors 64 --n -1 ^
        --lmjgp_steps 300 --MC_num 3 --scaling_preset uci_empirical ^
        --out_dir experiments/uci/lowrank_residual_lmjgp_1seed

Agents: update this docstring when CLI flags or defaults change.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.projections import build_V_basis, compute_pls_W0  # noqa: E402
from djgp.variational_lowrank import (  # noqa: E402
    predict_vi_lowrank_residual,
    train_vi_lowrank_residual,
)
from experiments.uci.compare_inductive_projection import (  # noqa: E402
    RESULT_ROW_KEYS,
    SCALING_PRESETS,
    _apply_scaling,
    _build_regions,
    _pack_metrics_maybe_original_y,
    _run_lmjgp_transductive,
    _set_seed,
    _subsample,
)
from experiments.uci.uci_data import load_uci_split  # noqa: E402


DATASETS_DEFAULT = "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"


def _dataset_n(dataset: str, n: int) -> int:
    if n > 0:
        return int(n)
    return 35 if dataset == "Appliances Energy Prediction" else 25


def _init_lowrank_state(
    *,
    X_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    Q: int,
    R: int,
    m1: int,
    m2: int,
    W0_t: torch.Tensor,
    V_t: torch.Tensor,
    residual_scale: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    T, D = X_test_t.shape
    C_params = {
        "mu_C": torch.zeros(m2, Q, R, device=device, requires_grad=True),
        "sigma_C": (0.1 + 0.05 * torch.rand(m2, Q, R, device=device)).requires_grad_(True),
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
    X_mean = X_train_t.mean(dim=0)
    X_std = X_train_t.std(dim=0).clamp_min(1e-6)
    Z = (X_mean + torch.randn(m2, D, device=device) * X_std).requires_grad_(True)
    W0 = W0_t / torch.linalg.norm(W0_t, dim=-1, keepdim=True).clamp_min(1e-6)
    hyperparams = {
        "Z": Z,
        "X_test": X_test_t,
        "lengthscales": torch.ones(Q, device=device, requires_grad=True),
        "var_c": torch.tensor(1.0, device=device, requires_grad=True),
        "W0": W0.detach(),
        "V": V_t.detach(),
        "residual_scale": float(residual_scale),
    }
    return C_params, u_params, hyperparams


def _metrics_row(
    *,
    dataset: str,
    seed: int,
    method: str,
    y_scaler,
    y_test_scaled: np.ndarray,
    mu_scaled: np.ndarray,
    sigma_scaled: np.ndarray,
    wall_sec: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    vals = _pack_metrics_maybe_original_y(
        y_scaler,
        y_test_scaled,
        np.asarray(mu_scaled, dtype=np.float64),
        np.asarray(sigma_scaled, dtype=np.float64),
        wall_sec,
    )
    row = {"dataset": dataset, "seed": seed, "method": method}
    row.update({k: float(v) for k, v in zip(RESULT_ROW_KEYS, vals)})
    row.update(extra)
    return row


def run_dataset(args: argparse.Namespace, dataset: str, seed: int, device: torch.device) -> list[dict[str, Any]]:
    _set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train_full, X_test_full, y_train_full, y_test_full, K_default = load_uci_split(dataset, seed)
    X_train_full, y_train_full, _ = _subsample(X_train_full, y_train_full, args.max_total_train, seed + 500)
    X_test_sub, y_test_sub, _ = _subsample(X_test_full, y_test_full, args.max_test_anchors, seed + 1000)
    if args.scaling_preset == "uci_empirical":
        standardize_x, standardize_y = SCALING_PRESETS.get(dataset, (bool(args.standardize_x), bool(args.standardize_y)))
    else:
        standardize_x, standardize_y = bool(args.standardize_x), bool(args.standardize_y)
    (
        X_train_s,
        X_base_s,
        _X_anchor_s,
        X_test_s,
        y_train_s,
        y_base_s,
        _y_anchor_s,
        y_test_s,
        _,
        y_scaler,
    ) = _apply_scaling(
        X_train_full,
        X_train_full,
        X_train_full[:1],
        X_test_sub,
        y_train_full,
        y_train_full,
        y_train_full[:1],
        y_test_sub,
        standardize_x=standardize_x,
        standardize_y=standardize_y,
    )
    Q = int(args.Q if args.Q is not None else K_default)
    n = _dataset_n(dataset, args.n)
    X_train_t = torch.from_numpy(X_train_s).float().to(device)
    y_train_t = torch.from_numpy(y_train_s).float().to(device)
    X_test_t = torch.from_numpy(X_test_s).float().to(device)

    rows = []
    if args.include_full_lmjgp:
        print(f"[{dataset}] full LMJGP transductive", flush=True)
        mu, sig, pred_time, diag = _run_lmjgp_transductive(
            X_test_t=X_test_t,
            y_test_np=y_test_s,
            X_pool_t=X_train_t,
            y_pool_t=y_train_t,
            X_inducing_t=X_train_t,
            Q=Q,
            m1=args.m1,
            m2=args.m2,
            n=n,
            steps=args.lmjgp_steps,
            lr=args.lr,
            MC_num=args.MC_num,
            device=device,
        )
        rows.append(
            _metrics_row(
                dataset=dataset,
                seed=seed,
                method="lmjgp_full_transductive",
                y_scaler=y_scaler,
                y_test_scaled=y_test_s,
                mu_scaled=mu,
                sigma_scaled=sig,
                wall_sec=pred_time + float(diag.get("train_time_sec", 0.0)),
                extra={
                    "standardize_x": int(standardize_x),
                    "standardize_y": int(standardize_y),
                    "Q": Q,
                    "n": n,
                    "R": np.nan,
                    "best_elbo": np.nan,
                    "train_time_sec": float(diag.get("train_time_sec", np.nan)),
                    "pred_time_sec": float(pred_time),
                },
            )
        )

    print(f"[{dataset}] low-rank residual LMJGP transductive", flush=True)
    V_np, V_info = build_V_basis(
        X_train_s,
        y_train_s,
        n_pls=args.n_pls,
        n_pca=args.n_pca,
        n_random=args.n_random,
        seed=seed,
    )
    W0_np = compute_pls_W0(X_train_s, y_train_s, Q=Q)
    V_t = torch.from_numpy(V_np).float().to(device)
    W0_t = torch.from_numpy(W0_np).float().to(device)
    R = int(V_t.shape[1])
    regions, _, _ = _build_regions(
        X_test_t,
        None,
        X_train_t,
        y_train_t,
        device=device,
        m1=args.m1,
        Q=Q,
        n=n,
    )
    C_params, u_params, hyperparams = _init_lowrank_state(
        X_train_t=X_train_t,
        X_test_t=X_test_t,
        Q=Q,
        R=R,
        m1=args.m1,
        m2=args.m2,
        W0_t=W0_t,
        V_t=V_t,
        residual_scale=args.residual_scale,
        device=device,
    )
    C_params, u_params, hyperparams, train_diag = train_vi_lowrank_residual(
        regions,
        C_params,
        u_params,
        hyperparams,
        lr=args.lr,
        num_steps=args.lmjgp_steps,
        log_interval=max(10, args.lmjgp_steps // 4),
    )
    mu_t, var_t, pred_time = predict_vi_lowrank_residual(
        regions,
        C_params,
        hyperparams,
        M=args.MC_num,
        seed=seed,
    )
    rows.append(
        _metrics_row(
            dataset=dataset,
            seed=seed,
            method="lmjgp_lowrank_residual_pls_pca",
            y_scaler=y_scaler,
            y_test_scaled=y_test_s,
            mu_scaled=mu_t.detach().cpu().numpy(),
            sigma_scaled=torch.sqrt(var_t.clamp_min(1e-12)).detach().cpu().numpy(),
            wall_sec=float(train_diag["train_time_sec"]) + pred_time,
            extra={
                "standardize_x": int(standardize_x),
                "standardize_y": int(standardize_y),
                "Q": Q,
                "n": n,
                "R": R,
                "residual_scale": float(args.residual_scale),
                "best_elbo": float(train_diag["best_elbo"]),
                "train_time_sec": float(train_diag["train_time_sec"]),
                "pred_time_sec": float(pred_time),
                "V_Rv": int(V_info["Rv"]),
            },
        )
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-rank residual LMJGP UCI comparison.")
    p.add_argument("--datasets", default=DATASETS_DEFAULT)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_total_train", type=int, default=500)
    p.add_argument("--max_test_anchors", type=int, default=64)
    p.add_argument("--scaling_preset", choices=("global", "uci_empirical"), default="uci_empirical")
    p.add_argument("--standardize_x", type=int, default=1)
    p.add_argument("--standardize_y", type=int, default=1)
    p.add_argument("--include_full_lmjgp", type=int, default=1)
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--n", type=int, default=-1)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default="experiments/uci/lowrank_residual_lmjgp_1seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.num_exp):
        for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            rows.extend(run_dataset(args, dataset, seed, device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "metrics.csv", rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Low-rank Residual LMJGP\n\n"
            "Generated by `experiments/uci/compare_lowrank_residual_lmjgp.py`.\n"
            "The low-rank model fixes `W0=PLS` and `V=[PLS,PCA]` and trains a GP over residual coefficients.\n"
        )
    print(f"Wrote low-rank residual results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

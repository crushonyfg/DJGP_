"""Fast sanity check for uncertain-W JumpGP-VEM marginalisation.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/check_uncertain_w_jgp_marginalization.py

    # Optional: write the printed table to CSV.
    python experiments/synthetic/check_uncertain_w_jgp_marginalization.py ^
        --out experiments/synthetic/uncertain_w_jgp_sanity_20260514/metrics.csv

This is a synthetic diagnostic, not a paper benchmark.  It compares the existing
mean-projection JumpGP-CEM head with batched uncertain-W JGP-VEM variants on a
small 2D discontinuous function.  The expected qualitative behavior is that
uncertain-W variants keep similar point accuracy but produce wider, better
covered intervals when projection uncertainty is injected.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation import mean_gaussian_crps_torch, pack_gaussian_uq_list  # noqa: E402
from djgp.projections.uncertain_w_jgp import (  # noqa: E402
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
    SelfCEMConfig,
    UncertainWJGPConfig,
    fit_uncertain_gate_from_labels,
    fit_uncertain_kernel_hyperparams_from_labels,
    run_uncertain_w_self_cem,
    uncertain_w_cem_predict_from_labels,
    uncertain_w_jgp_vem_predict,
)
from shared.utils1 import jumpgp_ld_wrapper  # noqa: E402


def _make_data(seed: int, n_train: int, n_test: int) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    X_train = 2.0 * torch.rand(n_train, 2, generator=g, dtype=torch.float64) - 1.0
    X_test = 2.0 * torch.rand(n_test, 2, generator=g, dtype=torch.float64) - 1.0

    def f(X: torch.Tensor) -> torch.Tensor:
        return torch.sin(3.0 * X[:, 0]) + 1.6 * (X[:, 0] > 0).to(X.dtype) + 0.25 * X[:, 1]

    y_train = f(X_train) + 0.08 * torch.randn(n_train, generator=g, dtype=torch.float64)
    y_test = f(X_test) + 0.08 * torch.randn(n_test, generator=g, dtype=torch.float64)
    return X_train, y_train, X_test, y_test


def _knn(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.cdist(X_test, X_train)
    idx = dist.topk(k=n, largest=False).indices
    return X_train[idx], y_train[idx]


def _gate_for_same_side_anchor(X_anchor: torch.Tensor, slope: float = 8.0) -> torch.Tensor:
    sign = torch.where(X_anchor[:, 0] >= 0, torch.ones_like(X_anchor[:, 0]), -torch.ones_like(X_anchor[:, 0]))
    gate = torch.zeros(X_anchor.shape[0], 2, dtype=X_anchor.dtype, device=X_anchor.device)
    gate[:, 1] = slope * sign
    gate[:, 0] = slope * sign * X_anchor[:, 0]
    return gate


def _pointwise_w_cov(X_all: torch.Tensor, *, var0: float, var1: float, lengthscale: float) -> torch.Tensor:
    T, m, _D = X_all.shape
    diff = X_all.unsqueeze(2) - X_all.unsqueeze(1)
    Kpos = torch.exp(-0.5 * diff.pow(2).sum(-1) / (lengthscale**2))
    cov = torch.zeros(T, 1, 2, m, m, dtype=X_all.dtype, device=X_all.device)
    cov[:, 0, 0] = var0 * Kpos
    cov[:, 0, 1] = var1 * Kpos
    return cov


def _metrics(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, wall: float) -> dict[str, float]:
    crps = float(mean_gaussian_crps_torch(y, mu, sigma).item())
    pack = pack_gaussian_uq_list(
        float(torch.sqrt(torch.mean((mu - y) ** 2)).item()),
        crps,
        float(wall),
        y.detach().cpu().numpy(),
        mu.detach().cpu().numpy(),
        sigma.detach().cpu().numpy(),
    )
    keys = [
        "rmse",
        "mean_crps",
        "wall_sec",
        "coverage_90",
        "mean_width_90",
        "coverage_95",
        "mean_width_95",
    ]
    return {key: float(value) for key, value in zip(keys, pack)}


def _run_original_projected_jumpgp(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    T = X_anchor.shape[0]
    W = torch.tensor([[1.0, 0.0]], dtype=X_anchor.dtype)
    mu = torch.empty(T, dtype=X_anchor.dtype)
    sigma = torch.empty(T, dtype=X_anchor.dtype)
    labels = torch.empty(T, X_neighbors.shape[1], dtype=torch.bool)
    mean_const = torch.empty(T, dtype=X_anchor.dtype)
    lengthscale = torch.empty(T, 1, dtype=X_anchor.dtype)
    signal_var = torch.empty(T, dtype=X_anchor.dtype)
    noise_var = torch.empty(T, dtype=X_anchor.dtype)
    t0 = time.perf_counter()
    for t in range(T):
        Z = X_neighbors[t] @ W.T
        zt = X_anchor[t : t + 1] @ W.T
        mu_t, sig2_t, model, _h = jumpgp_ld_wrapper(
            Z.float(),
            y_neighbors[t].view(-1, 1).float(),
            zt.float(),
            mode="CEM",
            flag=False,
            device=torch.device("cpu"),
        )
        mu[t] = mu_t.detach().cpu().reshape(-1)[0].to(dtype=X_anchor.dtype)
        sigma[t] = torch.sqrt(torch.clamp(sig2_t.detach().cpu().reshape(-1)[0], min=1e-12)).to(dtype=X_anchor.dtype)
        labels[t] = model["r"].detach().cpu().reshape(-1).bool()
        mean_const[t] = model["ms"].detach().cpu().reshape(-1)[0].to(dtype=X_anchor.dtype)
        logtheta = model["logtheta"].detach().cpu().reshape(-1).to(dtype=X_anchor.dtype)
        lengthscale[t, 0] = torch.exp(logtheta[0])
        signal_var[t] = torch.exp(2.0 * logtheta[1])
        noise_var[t] = torch.exp(2.0 * logtheta[2])
    row = {
        "method": "original_projected_jumpgp_cem",
        **_metrics(y_test, mu, sigma, time.perf_counter() - t0),
        "rho_mean": float("nan"),
        "mean_kernel_vector_correction": float("nan"),
    }
    state = {
        "labels": labels,
        "mean_const": mean_const,
        "lengthscale": lengthscale,
        "signal_var": signal_var,
        "noise_var": noise_var,
    }
    return row, state


def _row_from_cem_result(
    method: str,
    y_test: torch.Tensor,
    out,
    wall: float,
    n_total: int,
) -> dict[str, Any]:
    row = {"method": method, **_metrics(y_test, out.mu, out.sigma, wall)}
    row["rho_mean"] = float(out.n_inliers.double().mean().item() / max(1, n_total))
    row["mean_kernel_vector_correction"] = float(out.diagnostics["mean_kernel_vector_correction"])
    return row


def _run_cem_compatible_variants(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    y_test: torch.Tensor,
    cem_state: dict[str, torch.Tensor],
    lambdas: list[float],
) -> list[dict[str, Any]]:
    T = X_anchor.shape[0]
    W_mu = torch.zeros(T, 1, 2, dtype=X_anchor.dtype)
    W_mu[:, 0, 0] = 1.0
    W_var_det = torch.zeros_like(W_mu)
    W_var_unc = torch.tensor([[[0.10, 0.002]]], dtype=X_anchor.dtype).expand_as(W_mu)
    rows: list[dict[str, Any]] = []
    n_total = X_neighbors.shape[1]

    t0 = time.perf_counter()
    out = uncertain_w_cem_predict_from_labels(
        X_anchor,
        X_neighbors,
        y_neighbors,
        cem_state["labels"],
        mode="anchor",
        W_mu_anchor=W_mu,
        W_var_anchor=W_var_det,
        lengthscale=cem_state["lengthscale"],
        signal_var=cem_state["signal_var"],
        noise_var=cem_state["noise_var"],
        mean_const=cem_state["mean_const"],
        correction_weight=0.0,
    )
    rows.append(_row_from_cem_result("cem_fixed_det_compat", y_test, out, time.perf_counter() - t0, n_total))

    for lam in lambdas:
        t0 = time.perf_counter()
        out = uncertain_w_cem_predict_from_labels(
            X_anchor,
            X_neighbors,
            y_neighbors,
            cem_state["labels"],
            mode="anchor",
            W_mu_anchor=W_mu,
            W_var_anchor=W_var_unc,
            lengthscale=cem_state["lengthscale"],
            signal_var=cem_state["signal_var"],
            noise_var=cem_state["noise_var"],
            mean_const=cem_state["mean_const"],
            correction_weight=lam,
        )
        rows.append(_row_from_cem_result(f"cem_uncertain_fixedhyp_kcov{lam:g}", y_test, out, time.perf_counter() - t0, n_total))

    t0 = time.perf_counter()
    hp = fit_uncertain_kernel_hyperparams_from_labels(
        X_anchor,
        X_neighbors,
        y_neighbors,
        cem_state["labels"],
        mode="anchor",
        W_mu_anchor=W_mu,
        W_var_anchor=W_var_unc,
        init_lengthscale=cem_state["lengthscale"],
        init_signal_var=cem_state["signal_var"],
        init_noise_var=cem_state["noise_var"],
        config=FixedLabelHyperoptConfig(steps=30, lr=0.04),
    )
    hyperopt_wall = time.perf_counter() - t0
    for lam in lambdas:
        t0 = time.perf_counter()
        out = uncertain_w_cem_predict_from_labels(
            X_anchor,
            X_neighbors,
            y_neighbors,
            cem_state["labels"],
            mode="anchor",
            W_mu_anchor=W_mu,
            W_var_anchor=W_var_unc,
            lengthscale=hp["lengthscale"],
            signal_var=hp["signal_var"],
            noise_var=hp["noise_var"],
            mean_const=hp["mean_const"],
            correction_weight=lam,
        )
        rows.append(
            _row_from_cem_result(
                f"cem_uncertain_hyperopt_kcov{lam:g}",
                y_test,
                out,
                hyperopt_wall + time.perf_counter() - t0,
                n_total,
            )
        )
    gate_fit = fit_uncertain_gate_from_labels(
        X_anchor,
        X_neighbors,
        cem_state["labels"],
        mode="anchor",
        W_mu_anchor=W_mu,
        W_var_anchor=W_var_unc,
        gate_init=None,
        config=FixedLabelGateConfig(steps=60, lr=0.05, l2=1e-3, max_norm=8.0),
    )
    # Fixed labels: learned gate is recorded as a separate row family even though
    # prediction itself only depends on labels and GP hyperparameters.
    for lam in lambdas:
        t0 = time.perf_counter()
        out = uncertain_w_cem_predict_from_labels(
            X_anchor,
            X_neighbors,
            y_neighbors,
            cem_state["labels"],
            mode="anchor",
            W_mu_anchor=W_mu,
            W_var_anchor=W_var_unc,
            lengthscale=hp["lengthscale"],
            signal_var=hp["signal_var"],
            noise_var=hp["noise_var"],
            mean_const=hp["mean_const"],
            correction_weight=lam,
        )
        row = _row_from_cem_result(
            f"cem_uncertain_hyperopt_gate_kcov{lam:g}",
            y_test,
            out,
            time.perf_counter() - t0,
            n_total,
        )
        row["gate_norm"] = float(torch.linalg.norm(gate_fit["gate"], dim=1).mean().item())
        rows.append(row)

    for updates in [1, 3]:
        t0 = time.perf_counter()
        state = run_uncertain_w_self_cem(
            X_anchor,
            X_neighbors,
            y_neighbors,
            cem_state["labels"],
            mode="anchor",
            W_mu_anchor=W_mu,
            W_var_anchor=W_var_unc,
            init_lengthscale=cem_state["lengthscale"],
            init_signal_var=cem_state["signal_var"],
            init_noise_var=cem_state["noise_var"],
            config=SelfCEMConfig(
                cem_updates=updates,
                hyperopt=FixedLabelHyperoptConfig(steps=30, lr=0.04),
                gate=FixedLabelGateConfig(steps=60, lr=0.05, l2=1e-3, max_norm=8.0),
                outlier_mode="background_normal",
                min_inliers=2,
                max_inlier_frac=0.95,
            ),
        )
        cem_wall = time.perf_counter() - t0
        for lam in lambdas:
            t0 = time.perf_counter()
            out = uncertain_w_cem_predict_from_labels(
                X_anchor,
                X_neighbors,
                y_neighbors,
                state["labels"],
                mode="anchor",
                W_mu_anchor=W_mu,
                W_var_anchor=W_var_unc,
                lengthscale=state["lengthscale"],
                signal_var=state["signal_var"],
                noise_var=state["noise_var"],
                mean_const=state["mean_const"],
                correction_weight=lam,
            )
            row = _row_from_cem_result(
                f"self_cem{updates}_kcov{lam:g}",
                y_test,
                out,
                cem_wall + time.perf_counter() - t0,
                n_total,
            )
            row["history"] = str(state["history"])
            rows.append(row)
    return rows


def _run_uncertain_variants(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    y_test: torch.Tensor,
) -> list[dict[str, Any]]:
    T = X_anchor.shape[0]
    W_mu = torch.zeros(T, 1, 2, dtype=X_anchor.dtype)
    W_mu[:, 0, 0] = 1.0
    gate = _gate_for_same_side_anchor(X_anchor)
    rows: list[dict[str, Any]] = []
    base_cfg = dict(lengthscale=0.30, signal_var=1.0, noise_var=0.08**2, vem_iters=8, rho_init=0.75, gh_points=24)

    for method, var, correction in [
        ("mean_w_uncertain_jgp_vem", 0.0, True),
        ("anchor_uncertain_w_no_kcov", 0.10, False),
        ("anchor_uncertain_w_kcov", 0.10, True),
    ]:
        cfg = UncertainWJGPConfig(
            mode="anchor",
            kernel_vector_correction=correction,
            **base_cfg,
        )
        t0 = time.perf_counter()
        out = uncertain_w_jgp_vem_predict(
            X_anchor,
            X_neighbors,
            y_neighbors,
            config=cfg,
            W_mu_anchor=W_mu,
            W_var_anchor=torch.tensor([[[var, 0.02 * var]]], dtype=X_anchor.dtype).expand_as(W_mu),
            gate=gate,
        )
        row = {"method": method, **_metrics(y_test, out.mu, out.sigma, time.perf_counter() - t0)}
        row["rho_mean"] = float(out.diagnostics["rho_mean"])
        row["mean_kernel_vector_correction"] = float(out.diagnostics["mean_kernel_vector_correction"])
        rows.append(row)

    X_all = torch.cat([X_anchor.unsqueeze(1), X_neighbors], dim=1)
    W_mu_all = torch.zeros(T, X_all.shape[1], 1, 2, dtype=X_anchor.dtype)
    W_mu_all[:, :, 0, 0] = 1.0
    W_cov_all = _pointwise_w_cov(X_all, var0=0.10, var1=0.002, lengthscale=0.45)
    cfg = UncertainWJGPConfig(mode="pointwise", kernel_vector_correction=True, **base_cfg)
    t0 = time.perf_counter()
    out = uncertain_w_jgp_vem_predict(
        X_anchor,
        X_neighbors,
        y_neighbors,
        config=cfg,
        W_mu_all=W_mu_all,
        W_cov_all=W_cov_all,
        gate=gate,
    )
    row = {"method": "pointwise_uncertain_w_kcov", **_metrics(y_test, out.mu, out.sigma, time.perf_counter() - t0)}
    row["rho_mean"] = float(out.diagnostics["rho_mean"])
    row["mean_kernel_vector_correction"] = float(out.diagnostics["mean_kernel_vector_correction"])
    rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast uncertain-W JumpGP-VEM sanity check.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_train", type=int, default=450)
    p.add_argument("--n_test", type=int, default=60)
    p.add_argument("--n_neighbors", type=int, default=24)
    p.add_argument("--kcov_lambdas", type=str, default="0,0.05,0.1,0.25,0.5,1.0")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> list[dict[str, Any]]:
    args = parse_args()
    X_train, y_train, X_test, y_test = _make_data(args.seed, args.n_train, args.n_test)
    X_neighbors, y_neighbors = _knn(X_train, y_train, X_test, args.n_neighbors)
    lambdas = [float(s) for s in str(args.kcov_lambdas).split(",") if s.strip()]
    original_row, cem_state = _run_original_projected_jumpgp(X_test, X_neighbors, y_neighbors, y_test)
    rows = [original_row]
    rows.extend(_run_cem_compatible_variants(X_test, X_neighbors, y_neighbors, y_test, cem_state, lambdas))
    rows.extend(_run_uncertain_variants(X_test, X_neighbors, y_neighbors, y_test))
    print("method                         rmse    crps   cov90  width90  cov95  width95  wall   rho    kcov")
    for r in rows:
        print(
            f"{r['method']:<30} {r['rmse']:6.3f} {r['mean_crps']:6.3f} "
            f"{r['coverage_90']:6.3f} {r['mean_width_90']:7.3f} "
            f"{r['coverage_95']:6.3f} {r['mean_width_95']:7.3f} "
            f"{r['wall_sec']:6.2f} {r['rho_mean']:6.3f} {r['mean_kernel_vector_correction']:7.4f}"
        )
    if args.out:
        _write_csv(Path(args.out), rows)
    return rows


if __name__ == "__main__":
    main()

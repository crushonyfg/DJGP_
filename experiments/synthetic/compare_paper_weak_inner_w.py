"""Compare weak-inner projection-learning variants on paper L2/LH synthetic data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # One-seed pilot on the train=1000 L2/LH paper-style synthetic settings.
    python experiments/synthetic/compare_paper_weak_inner_w.py ^
        --num_exp 1 ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --methods pls_static,gp_localC,pairwise_localC,pairwise_boundary_localC,gp_pairwise_boundary_localC ^
        --n_candidate 100 --final_neighbor_modes raw,projected ^
        --steps 200 --resume ^
        --out_dir experiments/synthetic/weak_inner_w_paper_train1000_seed0

    # Faster smoke test.
    python experiments/synthetic/compare_paper_weak_inner_w.py ^
        --num_exp 1 --settings l2_q2_train1000 --max_anchors 30 ^
        --methods pls_static,pairwise_boundary_localC ^
        --n_candidate 50 --final_neighbor_modes projected --steps 10 ^
        --out_dir experiments/synthetic/weak_inner_w_smoke

This runner is a practical ablation for the "q(W) stays prior-like" diagnosis.
It does not implement the full stochastic q(W)-averaged JGP-VEM bound. Instead,
it tests cheaper training objectives that deliberately make the projection W
carry more signal before the final JumpGP-CEM refit:

- ``pls_static``: fixed row-normalized PLS projection.
- ``gp_*``: exact local RBF-GP NLL with shared/frozen nuisance scale on
  standardized y.
- ``pairwise_*``: supervised metric-learning loss from local response
  similarities.
- ``*_boundary_*``: anchor weights proportional to local response variance.
- ``*_localC``: per-anchor low-rank coefficient residual around a global W0.
- ``*_globalW``: global row-normalized W0 only, no per-anchor residual.

For final prediction each trained W is evaluated through the same vendored
JumpGP-CEM prediction head used by the existing synthetic baselines. With
``--final_neighbor_modes projected``, the final local JumpGP neighborhood is
selected in projected space from a larger raw-X candidate pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import (  # noqa: E402
    CoeffProjection,
    anchor_rbf_kernel,
    build_V_basis,
    compute_pls_W0,
    laplacian_smoothness,
    per_anchor_W_diagnostics,
    run_projected_jumpgp,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


METHOD_ORDER = (
    "pls_static",
    "gp_globalW",
    "gp_localC",
    "pairwise_globalW",
    "pairwise_localC",
    "pairwise_boundary_localC",
    "gp_pairwise_globalW",
    "gp_pairwise_localC",
    "gp_pairwise_boundary_localC",
)

FINAL_NEIGHBOR_MODES = ("raw", "projected")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting",
        "family",
        "seed",
        "method",
        "final_neighbor_mode",
        "status",
        "N_train",
        "N_test",
        "latent_dim",
        "observed_dim",
        "Q_true",
        "Q",
        "n_final",
        "n_candidate",
        "gp_neighbors",
        "pair_neighbors",
        "standardize_x",
        *RESULT_ROW_KEYS,
        "train_sec",
        "pred_sec",
        "train_loss",
        "train_gp_loss",
        "train_pair_loss",
        "train_smoothness",
        "median_frob_W",
        "anchor_dispersion_frob",
        "cosine_to_ref_median",
        "error",
    ]
    keys: list[str] = []
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    for key in preferred:
        if key in all_keys:
            keys.append(key)
    keys.extend(sorted(all_keys - set(keys)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, str) and x.strip() == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault(
                (str(row["setting"]), str(row["method"]), str(row["final_neighbor_mode"])),
                [],
            ).append(row)
    out: list[dict[str, Any]] = []
    for (setting, method, final_mode), group in sorted(groups.items()):
        agg: dict[str, Any] = {
            "setting": setting,
            "method": method,
            "final_neighbor_mode": final_mode,
            "n_seeds": len(group),
        }
        for meta in (
            "family",
            "N_train",
            "N_test",
            "latent_dim",
            "observed_dim",
            "Q_true",
            "Q",
            "n_final",
            "n_candidate",
            "gp_neighbors",
            "pair_neighbors",
            "standardize_x",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in RESULT_ROW_KEYS:
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        for key in (
            "train_sec",
            "pred_sec",
            "train_loss",
            "train_gp_loss",
            "train_pair_loss",
            "train_smoothness",
            "median_frob_W",
            "anchor_dispersion_frob",
            "cosine_to_ref_median",
        ):
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            if np.isfinite(vals).any():
                agg[f"{key}_mean"] = float(np.nanmean(vals))
                agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _metrics_pack(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> list[float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, wall_sec, y, mu, sigma)


def _build_neighbor_stacks(
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    n_neighbors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    neighborhoods = find_neighborhoods(
        X_anchor.detach().cpu(),
        X_pool.detach().cpu(),
        y_pool.detach().cpu(),
        M=int(n_neighbors),
    )
    X_stack = torch.stack([item["X_neighbors"].to(device=device).float() for item in neighborhoods], dim=0)
    y_stack = torch.stack([item["y_neighbors"].to(device=device).float().view(-1) for item in neighborhoods], dim=0)
    return X_stack, y_stack


def _lists_from_stack(
    X_stack: torch.Tensor,
    y_stack: torch.Tensor,
    *,
    n_neighbors: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    X_list = [X_stack[t, :n_neighbors].contiguous() for t in range(X_stack.shape[0])]
    y_list = [y_stack[t, :n_neighbors].contiguous() for t in range(y_stack.shape[0])]
    return X_list, y_list


def _projected_neighbor_lists(
    W: torch.Tensor,
    X_anchor: torch.Tensor,
    X_candidate: torch.Tensor,
    y_candidate: torch.Tensor,
    *,
    n_neighbors: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    with torch.no_grad():
        Z_candidate = torch.einsum("tmd,tqd->tmq", X_candidate, W)
        z_anchor = torch.einsum("td,tqd->tq", X_anchor, W)
        d2 = (Z_candidate - z_anchor.unsqueeze(1)).pow(2).sum(dim=-1)
        idx = torch.topk(d2, k=int(n_neighbors), largest=False, dim=1).indices
    X_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    for t in range(X_candidate.shape[0]):
        X_list.append(X_candidate[t, idx[t]].contiguous())
        y_list.append(y_candidate[t, idx[t]].contiguous())
    return X_list, y_list


def _row_normalize(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(eps)


def _local_gp_nll_per_anchor(
    Z: torch.Tensor,
    y: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_amp: torch.Tensor,
    log_noise_var: torch.Tensor,
    *,
    jitter: float = 1e-5,
) -> torch.Tensor:
    T, n, _ = Z.shape
    l2 = torch.exp(2.0 * log_lengthscale).clamp_min(1e-12)
    amp2 = torch.exp(2.0 * log_amp).clamp_min(1e-12)
    sig2 = torch.exp(log_noise_var).clamp_min(1e-12)
    d2 = (Z.unsqueeze(2) - Z.unsqueeze(1)).pow(2).sum(-1)
    K = amp2 * torch.exp(-0.5 * d2 / l2)
    eye = torch.eye(n, device=Z.device, dtype=Z.dtype).unsqueeze(0)
    K = K + (sig2 + jitter) * eye
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
    quad = (y * alpha).sum(dim=-1)
    logdet = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(dim=-1)
    return 0.5 * (quad + logdet + n * math.log(2.0 * math.pi))


def _pairwise_metric_loss_per_anchor(
    Z: torch.Tensor,
    y: torch.Tensor,
    *,
    tau_y: float,
    margin: float,
) -> torch.Tensor:
    n = Z.shape[1]
    d2 = (Z.unsqueeze(2) - Z.unsqueeze(1)).pow(2).sum(-1)
    dy2 = (y.unsqueeze(2) - y.unsqueeze(1)).pow(2)
    s = torch.exp(-0.5 * dy2 / max(float(tau_y) ** 2, 1e-12))
    loss_mat = s * d2 + (1.0 - s) * torch.relu(float(margin) - d2).pow(2)
    mask = (~torch.eye(n, dtype=torch.bool, device=Z.device)).unsqueeze(0)
    denom = float(n * (n - 1))
    return loss_mat.masked_select(mask).view(Z.shape[0], -1).sum(dim=1) / denom


def _response_variance_weights(y_std_stack: torch.Tensor) -> torch.Tensor:
    var = y_std_stack.var(dim=1, unbiased=False)
    weights = var / var.mean().clamp_min(1e-12)
    return weights.clamp(0.25, 4.0).detach()


def _default_pair_tau(y_std_stack: torch.Tensor) -> float:
    with torch.no_grad():
        n = y_std_stack.shape[1]
        dy = (y_std_stack.unsqueeze(2) - y_std_stack.unsqueeze(1)).abs()
        mask = (~torch.eye(n, dtype=torch.bool, device=y_std_stack.device)).unsqueeze(0)
        vals = dy.masked_select(mask)
        return float(vals.median().clamp_min(0.05).item())


def _train_projection(
    method: str,
    *,
    X_anchor: torch.Tensor,
    X_gp: torch.Tensor,
    y_gp_std: torch.Tensor,
    X_pair: torch.Tensor,
    y_pair_std: torch.Tensor,
    V_t: torch.Tensor,
    W0_t: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    T, _, D = X_pair.shape
    Q = W0_t.shape[0]
    W_ref = _row_normalize(W0_t)
    if method == "pls_static":
        W = W_ref.unsqueeze(0).expand(T, -1, -1).contiguous()
        diag = per_anchor_W_diagnostics(W, W_ref=W_ref)
        diag.update(
            {
                "train_sec": 0.0,
                "train_loss": 0.0,
                "train_gp_loss": 0.0,
                "train_pair_loss": 0.0,
                "train_smoothness": 0.0,
                "train_steps": 0,
            }
        )
        return W.detach(), diag

    use_gp = "gp" in method
    use_pair = "pairwise" in method
    boundary_weighted = "boundary" in method
    global_w = "globalW" in method
    if not use_gp and not use_pair:
        raise ValueError(f"Cannot infer training losses from method {method!r}.")

    model = CoeffProjection(
        T=T,
        Q=Q,
        D=D,
        V=V_t,
        W0_init=W0_t,
        learn_W0=True,
        init_C_scale=float(args.init_C_scale),
        coeff_scale=float(args.coeff_scale),
        row_normalize=True,
    ).to(device)
    if global_w:
        model.C.requires_grad_(False)

    log_lengthscale = torch.tensor(math.log(float(args.init_lengthscale)), device=device, requires_grad=use_gp)
    log_amp = torch.tensor(0.0, device=device, requires_grad=use_gp)
    log_noise_var = torch.tensor(math.log(float(args.init_noise_var)), device=device, requires_grad=use_gp)
    params = [p for p in model.parameters() if p.requires_grad]
    if use_gp:
        params.extend([log_lengthscale, log_amp, log_noise_var])
    optimizer = torch.optim.Adam(params, lr=float(args.lr))

    pair_tau = float(args.pair_tau) if float(args.pair_tau) > 0 else _default_pair_tau(y_pair_std)
    weights = _response_variance_weights(y_pair_std if boundary_weighted else y_gp_std)
    if not boundary_weighted:
        weights = torch.ones_like(weights)
    K_anchor = None
    if (not global_w) and float(args.lambda_smooth) > 0:
        K_anchor = anchor_rbf_kernel(X_anchor).detach()

    best_loss = float("inf")
    best_W: torch.Tensor | None = None
    last_log: dict[str, float] = {}
    t0 = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad()
        W = model.W_all()
        loss = torch.zeros((), device=device)
        gp_loss = torch.zeros((), device=device)
        pair_loss = torch.zeros((), device=device)
        smoothness = torch.zeros((), device=device)

        if use_gp:
            Z_gp = torch.einsum("tnd,tqd->tnq", X_gp, W)
            nll = _local_gp_nll_per_anchor(Z_gp, y_gp_std, log_lengthscale, log_amp, log_noise_var)
            gp_loss = (weights * nll).mean()
            loss = loss + float(args.lambda_gp) * gp_loss

        if use_pair:
            Z_pair = torch.einsum("tnd,tqd->tnq", X_pair, W)
            per_anchor = _pairwise_metric_loss_per_anchor(
                Z_pair,
                y_pair_std,
                tau_y=pair_tau,
                margin=float(args.pair_margin),
            )
            pair_loss = (weights * per_anchor).mean()
            loss = loss + float(args.lambda_pair) * pair_loss

        if K_anchor is not None:
            smoothness = laplacian_smoothness(model.C, K_anchor, normalize_by_graph_weight=True)
            loss = loss + float(args.lambda_smooth) * smoothness
        if float(args.lambda_coeff) > 0 and (not global_w):
            loss = loss + float(args.lambda_coeff) * model.C.pow(2).mean()

        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_W = model.W_all().detach().clone()
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            last_log = {
                "train_loss": loss_val,
                "train_gp_loss": float(gp_loss.detach().item()),
                "train_pair_loss": float(pair_loss.detach().item()),
                "train_smoothness": float(smoothness.detach().item()),
                "log_lengthscale": float(log_lengthscale.detach().item()),
                "log_amp": float(log_amp.detach().item()),
                "log_noise_var": float(log_noise_var.detach().item()),
                "pair_tau": pair_tau,
            }
            if bool(args.verbose):
                print(
                    f"[{method} step {step:04d}] loss={last_log['train_loss']:.4f} "
                    f"gp={last_log['train_gp_loss']:.4f} pair={last_log['train_pair_loss']:.4f} "
                    f"smooth={last_log['train_smoothness']:.4f}",
                    flush=True,
                )

    train_sec = float(time.perf_counter() - t0)
    W_final = best_W if best_W is not None else model.W_all().detach()
    diag = per_anchor_W_diagnostics(W_final, W_ref=W_ref)
    diag.update(last_log)
    diag.update(
        {
            "train_sec": train_sec,
            "train_steps": int(args.steps),
            "global_w": bool(global_w),
            "boundary_weighted": bool(boundary_weighted),
            "use_gp_loss": bool(use_gp),
            "use_pairwise_loss": bool(use_pair),
            "lambda_gp": float(args.lambda_gp),
            "lambda_pair": float(args.lambda_pair),
            "lambda_smooth": float(args.lambda_smooth),
            "lambda_coeff": float(args.lambda_coeff),
            "coeff_scale": float(args.coeff_scale),
        }
    )
    return W_final.detach(), diag


def _evaluate_projection(
    W: torch.Tensor,
    *,
    method: str,
    final_mode: str,
    X_anchor: torch.Tensor,
    X_candidate: torch.Tensor,
    y_candidate: torch.Tensor,
    y_test: np.ndarray,
    n_final: int,
    train_sec: float,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    if final_mode == "raw":
        X_final, y_final = _lists_from_stack(X_candidate, y_candidate, n_neighbors=n_final)
    elif final_mode == "projected":
        X_final, y_final = _projected_neighbor_lists(
            W,
            X_anchor,
            X_candidate,
            y_candidate,
            n_neighbors=n_final,
        )
    else:
        raise ValueError(f"Unknown final neighbor mode {final_mode!r}.")

    out = run_projected_jumpgp(
        W,
        X_anchor,
        X_final,
        y_final,
        device=device,
        progress=False,
        desc=f"{method}/{final_mode}",
    )
    pred_sec = float(out["elapsed_sec"])
    pack = _metrics_pack(y_test, np.asarray(out["mu"]), np.asarray(out["sigma"]), train_sec + pred_sec)
    return pack, {"pred_sec": pred_sec}


def _base_row(
    *,
    setting: str,
    cfg: dict[str, Any],
    seed: int,
    method: str,
    final_mode: str,
    n_final: int,
    n_candidate: int,
    gp_neighbors: int,
    pair_neighbors: int,
    n_eval: int,
) -> dict[str, Any]:
    return {
        "setting": setting,
        "family": cfg["family"],
        "seed": int(seed),
        "method": method,
        "final_neighbor_mode": final_mode,
        "N_train": int(cfg["N_train"]),
        "N_test": int(n_eval),
        "latent_dim": int(cfg["d"]),
        "observed_dim": int(cfg["H"]),
        "Q_true": int(cfg["Q_true"]),
        "Q": int(cfg["Q"]),
        "n_final": int(n_final),
        "n_candidate": int(n_candidate),
        "gp_neighbors": int(gp_neighbors),
        "pair_neighbors": int(pair_neighbors),
        "standardize_x": True,
        "noise_std": float(cfg["noise_std"]),
        "noise_var": float(cfg["noise_var"]),
        "lengthscale": float(cfg["lengthscale"]),
        "kernel_var": float(cfg["kernel_var"]),
    }


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Weak-Inner W Projection Ablation on Paper Synthetic Data

Purpose: test whether projection objectives with weaker local nuisance
parameters produce more useful W than the current full LMJGP VI objective on
paper-style L2/LH train=1000 synthetic data.

This is a practical ablation, not a full implementation of stochastic
q(W)-averaged JGP-VEM. Final predictions use the same JumpGP-CEM refit head as
the existing synthetic baselines.

```json
{json.dumps(summary, indent=2)}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method, per-final-neighborhood metrics.
- `aggregate.csv`: means/stds over seeds.
- `summary.json`: exact runner arguments.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weak-inner W ablations on paper L2/LH synthetic train=1000 data.")
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--settings", type=str, default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument(
        "--methods",
        type=str,
        default="pls_static,gp_localC,pairwise_localC,pairwise_boundary_localC,gp_pairwise_boundary_localC",
    )
    p.add_argument("--final_neighbor_modes", type=str, default="raw,projected")
    p.add_argument("--max_anchors", type=int, default=None)
    p.add_argument("--n_candidate", type=int, default=100)
    p.add_argument("--gp_neighbors", type=int, default=0, help="0 means use the setting's n_final.")
    p.add_argument("--pair_neighbors", type=int, default=100)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lambda_gp", type=float, default=1.0)
    p.add_argument("--lambda_pair", type=float, default=1.0)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_coeff", type=float, default=1e-4)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--init_C_scale", type=float, default=0.0)
    p.add_argument("--init_lengthscale", type=float, default=1.0)
    p.add_argument("--init_noise_var", type=float, default=0.1)
    p.add_argument("--pair_tau", type=float, default=0.0, help="0 means median local pairwise |dy| on standardized y.")
    p.add_argument("--pair_margin", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out_dir", type=str, default="experiments/synthetic/weak_inner_w_paper_train1000_seed0")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown_settings:
        raise ValueError(f"Unknown settings {unknown_settings}; choose from {sorted(SETTING_PRESETS)}")
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    unknown_methods = sorted(set(methods) - set(METHOD_ORDER))
    if unknown_methods:
        raise ValueError(f"Unknown methods {unknown_methods}; choose from {METHOD_ORDER}")
    final_modes = [s.strip() for s in args.final_neighbor_modes.split(",") if s.strip()]
    unknown_modes = sorted(set(final_modes) - set(FINAL_NEIGHBOR_MODES))
    if unknown_modes:
        raise ValueError(f"Unknown final neighbor modes {unknown_modes}; choose from {FINAL_NEIGHBOR_MODES}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(out_dir / "metrics_partial.csv") if args.resume else []
    if args.resume and not rows:
        rows = _read_csv(out_dir / "metrics.csv")
    completed = {
        (str(r.get("setting")), int(r.get("seed")), str(r.get("method")), str(r.get("final_neighbor_mode")))
        for r in rows
        if str(r.get("status")) == "ok" and str(r.get("seed", "")).strip() != ""
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for setting in settings:
        cfg = dict(SETTING_PRESETS[setting])
        n_final = int(cfg["n"])
        gp_neighbors = int(args.gp_neighbors) if int(args.gp_neighbors) > 0 else n_final
        pair_neighbors = int(args.pair_neighbors)
        n_candidate = max(int(args.n_candidate), n_final, gp_neighbors, pair_neighbors)
        n_candidate = min(n_candidate, int(cfg["N_train"]))
        gp_neighbors = min(gp_neighbors, n_candidate)
        pair_neighbors = min(pair_neighbors, n_candidate)
        n_final = min(n_final, n_candidate)

        for seed in range(int(args.num_exp)):
            _set_seed(seed)
            data = _generate_paper_data(cfg, seed, device)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(data["X_train"])
            X_test_s = scaler.transform(data["X_test"])
            y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
            y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)
            if args.max_anchors is not None:
                k = min(int(args.max_anchors), X_test_s.shape[0])
                X_test_s = X_test_s[:k]
                y_test = y_test[:k]

            X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
            y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
            X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)
            X_cand, y_cand = _build_neighbor_stacks(
                X_test_t,
                X_train_t,
                y_train_t,
                n_neighbors=n_candidate,
                device=device,
            )

            y_mean = float(np.mean(y_train))
            y_std = float(np.std(y_train) + 1e-8)
            y_gp_std = (y_cand[:, :gp_neighbors] - y_mean) / y_std
            y_pair_std = (y_cand[:, :pair_neighbors] - y_mean) / y_std
            X_gp = X_cand[:, :gp_neighbors].contiguous()
            X_pair = X_cand[:, :pair_neighbors].contiguous()

            V_np, V_info = build_V_basis(
                X_train_s,
                y_train,
                n_pls=int(args.n_pls),
                n_pca=int(args.n_pca),
                n_random=int(args.n_random),
                seed=seed,
            )
            W0_np = compute_pls_W0(X_train_s, y_train, Q=int(cfg["Q"]))
            V_t = torch.from_numpy(V_np.astype(np.float32)).to(device)
            W0_t = torch.from_numpy(W0_np.astype(np.float32)).to(device)

            for method in methods:
                missing_modes = [
                    mode
                    for mode in final_modes
                    if (setting, seed, method, mode) not in completed
                ]
                if not missing_modes:
                    print(f"skip completed setting={setting} seed={seed} method={method}", flush=True)
                    continue
                print(f"setting={setting} seed={seed} method={method}", flush=True)
                train_diag: dict[str, Any] = {}
                try:
                    W, train_diag = _train_projection(
                        method,
                        X_anchor=X_test_t,
                        X_gp=X_gp,
                        y_gp_std=y_gp_std,
                        X_pair=X_pair,
                        y_pair_std=y_pair_std,
                        V_t=V_t,
                        W0_t=W0_t,
                        args=args,
                        device=device,
                    )
                    for final_mode in missing_modes:
                        base = _base_row(
                            setting=setting,
                            cfg=cfg,
                            seed=seed,
                            method=method,
                            final_mode=final_mode,
                            n_final=n_final,
                            n_candidate=n_candidate,
                            gp_neighbors=gp_neighbors,
                            pair_neighbors=pair_neighbors,
                            n_eval=len(y_test),
                        )
                        print(
                            f"evaluate setting={setting} seed={seed} method={method} final={final_mode}",
                            flush=True,
                        )
                        pack, eval_diag = _evaluate_projection(
                            W,
                            method=method,
                            final_mode=final_mode,
                            X_anchor=X_test_t,
                            X_candidate=X_cand,
                            y_candidate=y_cand,
                            y_test=y_test,
                            n_final=n_final,
                            train_sec=float(train_diag.get("train_sec", 0.0)),
                            device=device,
                        )
                        row = dict(base)
                        row["status"] = "ok"
                        for key, val in zip(RESULT_ROW_KEYS, pack):
                            row[key] = val
                        row.update(train_diag)
                        row.update(eval_diag)
                        row.update({"V_Rv": int(V_t.shape[1]), "V_info": json.dumps(V_info)})
                        rows.append(row)
                        completed.add((setting, seed, method, final_mode))
                        _write_csv(out_dir / "metrics_partial.csv", rows)
                except Exception as exc:
                    for final_mode in missing_modes:
                        row = _base_row(
                            setting=setting,
                            cfg=cfg,
                            seed=seed,
                            method=method,
                            final_mode=final_mode,
                            n_final=n_final,
                            n_candidate=n_candidate,
                            gp_neighbors=gp_neighbors,
                            pair_neighbors=pair_neighbors,
                            n_eval=len(y_test),
                        )
                        row["status"] = "error"
                        row["error"] = repr(exc)
                        row.update(train_diag)
                        rows.append(row)
                    _write_csv(out_dir / "metrics_partial.csv", rows)
                    print(f"ERROR setting={setting} seed={seed} method={method}: {exc!r}", flush=True)

    _write_csv(out_dir / "metrics.csv", rows)
    agg = _aggregate(rows)
    _write_csv(out_dir / "aggregate.csv", agg)
    summary = {
        "args": vars(args),
        "device": str(device),
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "methods": methods,
        "final_neighbor_modes": final_modes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_readme(out_dir, summary)
    print(f"Saved {len(rows)} rows to {out_dir}", flush=True)
    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    main()

"""
Try small-data LMJGP remedy variants on UCI datasets.

This runner is intentionally experiment-only: it does not change or overwrite
the existing ``lmjgp_transductive`` / ``lmjgp_supervised`` implementations in
``compare_inductive_projection.py``. It reuses the strict supervised q(V)
training objective, then tests three remedies:

* response-aware hybrid neighborhoods based on a training-only PLS score;
* a training-only ridge global mean with LMJGP fit to residuals;
* a plain local-GP kernel diagnostic with RBF / Matern-3/2 / Matern-5/2.

How to run (repository root, conda env: ``jumpGP``):

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke test on a tiny Wine split.
    python experiments/uci/try_lmjgp_remedies.py ^
        --datasets "Wine Quality" --max_total_train 80 --max_test_anchors 20 ^
        --max_train_anchors 32 --neighbors auto --lmjgp_steps 2 --MC_num 2 ^
        --xgb_boots 3 --out_dir experiments/uci/smalltrain_lmjgp_remedies_smoke

    # Main one-seed 200-train remedy run.
    python experiments/uci/try_lmjgp_remedies.py ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 ^
        --neighbors auto --lmjgp_steps 300 --MC_num 3 --xgb_boots 30 ^
        --scaling_preset uci_empirical --out_dir experiments/uci/smalltrain_lmjgp_remedies
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics.projection_posterior import sanitize_for_json  # noqa: E402
from djgp.evaluation import mean_gaussian_crps_torch  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.variational import qW_from_qV  # noqa: E402
from experiments.uci.compare_inductive_projection import (  # noqa: E402
    SCALING_PRESETS,
    _apply_scaling,
    _build_regions,
    _pack_metrics_maybe_original_y,
    _predict_lmjgp_from_qv,
    _run_xgboost_bootstrap,
    _set_seed,
    _split_base_anchor,
    _subsample,
    _train_lmjgp_supervised_predictive,
)
from experiments.uci.uci_data import load_uci_split  # noqa: E402


DEFAULT_METHODS = (
    "xgboost,"
    "lmjgp_supervised_base,"
    "lmjgp_supervised_hybrid_pls,"
    "lmjgp_supervised_residual_ridge,"
    "lmjgp_supervised_hybrid_residual_ridge,"
    "lmjgp_supervised_localgp_rbf,"
    "lmjgp_supervised_localgp_matern32,"
    "lmjgp_supervised_localgp_matern52"
)

NEIGHBOR_PRESET = {
    "Wine Quality": 25,
    "Parkinsons Telemonitoring": 25,
    "Appliances Energy Prediction": 35,
}

Z90 = 1.6448536269514722


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _as_metric_dict(values: list[float]) -> dict[str, float]:
    return {key: float(values[i]) for i, key in enumerate(RESULT_ROW_KEYS)}


def _invert_y(y_scaler, y_scaled: np.ndarray, sigma_scaled: np.ndarray | None = None):
    y_scaled = np.asarray(y_scaled, dtype=np.float64).reshape(-1)
    if y_scaler is None:
        sigma = None if sigma_scaled is None else np.asarray(sigma_scaled, dtype=np.float64).reshape(-1)
        return y_scaled, sigma
    y_orig = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
    if sigma_scaled is None:
        return y_orig, None
    scale = float(np.squeeze(y_scaler.scale_))
    if not np.isfinite(scale) or scale < 1e-20:
        scale = 1.0
    sigma_orig = np.asarray(sigma_scaled, dtype=np.float64).reshape(-1) * abs(scale)
    return y_orig, sigma_orig


def _fit_pls_score(X: np.ndarray, y: np.ndarray, n_components: int) -> Callable[[np.ndarray], np.ndarray]:
    n_comp = int(max(1, min(n_components, X.shape[1], X.shape[0] - 1)))
    model = PLSRegression(n_components=n_comp, scale=False)
    model.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64).reshape(-1, 1))

    def predict(X_new: np.ndarray) -> np.ndarray:
        return model.predict(np.asarray(X_new, dtype=np.float64)).reshape(-1)

    return predict


def _fit_ridge_mean(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    model = Ridge(alpha=float(alpha), random_state=0)
    t0 = time.perf_counter()
    model.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64).reshape(-1))
    elapsed = time.perf_counter() - t0

    def predict(X_new: np.ndarray) -> np.ndarray:
        return model.predict(np.asarray(X_new, dtype=np.float64)).reshape(-1)

    diag = {
        "global_mean_model": "ridge",
        "ridge_alpha": float(alpha),
        "ridge_fit_time_sec": float(elapsed),
    }
    return predict, diag


def _build_hybrid_regions(
    X_anchors: torch.Tensor,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    score_anchors: np.ndarray,
    score_pool: np.ndarray,
    device: torch.device,
    m1: int,
    Q: int,
    n_total: int,
    n_x: int,
    n_score: int,
):
    """Union raw-X nearest neighbors with nearest neighbors in predicted response score."""
    Xa = X_anchors.detach().cpu().numpy()
    Xp = X_pool.detach().cpu().numpy()
    sp = np.asarray(score_pool, dtype=np.float64).reshape(-1)
    sa = np.asarray(score_anchors, dtype=np.float64).reshape(-1)
    n_pool = Xp.shape[0]
    n_total = int(min(max(1, n_total), n_pool))
    n_x = int(min(max(1, n_x), n_pool))
    n_score = int(min(max(1, n_score), n_pool))

    regions = []
    X_nb_list = []
    y_nb_list = []
    for i in range(Xa.shape[0]):
        d_x = np.sum((Xp - Xa[i]) ** 2, axis=1)
        raw_order = np.argsort(d_x, kind="mergesort")
        d_s = np.abs(sp - sa[i])
        score_order = np.argsort(d_s, kind="mergesort")

        chosen: list[int] = []
        seen: set[int] = set()
        for idx in raw_order[:n_x]:
            j = int(idx)
            if j not in seen:
                chosen.append(j)
                seen.add(j)
        for idx in score_order[:n_score]:
            j = int(idx)
            if j not in seen:
                chosen.append(j)
                seen.add(j)
        for idx in raw_order:
            if len(chosen) >= n_total:
                break
            j = int(idx)
            if j not in seen:
                chosen.append(j)
                seen.add(j)
        chosen = chosen[:n_total]

        idx_t = torch.as_tensor(chosen, dtype=torch.long, device=X_pool.device)
        X_nb = X_pool.index_select(0, idx_t).to(device)
        y_nb = y_pool.index_select(0, idx_t).to(device).view(-1)
        regions.append({"X": X_nb, "y": y_nb, "C": torch.randn(m1, Q, device=device)})
        X_nb_list.append(X_nb)
        y_nb_list.append(y_nb)
    return regions, X_nb_list, y_nb_list


def _sample_W_from_qv(
    *,
    V_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    X_query: torch.Tensor,
    MC_num: int,
    seed: int,
) -> torch.Tensor:
    with torch.no_grad():
        mu_W, cov_W = qW_from_qV(
            X_query,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
        sigma_W = torch.sqrt(cov_W.diagonal(dim1=2, dim2=3).clamp_min(1e-12))
        gen = torch.Generator(device=X_query.device)
        gen.manual_seed(int(seed))
        eps = torch.randn(
            (MC_num,) + tuple(mu_W.shape),
            generator=gen,
            device=X_query.device,
            dtype=mu_W.dtype,
        )
        W = mu_W.unsqueeze(0) + eps * sigma_W.unsqueeze(0)
        W = W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(1e-6)
    return W


def _kernel_matrix(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    kernel: str,
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
) -> torch.Tensor:
    diff = (A.unsqueeze(-2) - B.unsqueeze(-3)) / lengthscale.view(1, -1)
    d2 = (diff * diff).sum(dim=-1).clamp_min(0.0)
    if kernel == "rbf":
        return signal_var * torch.exp(-0.5 * d2)
    r = torch.sqrt(d2 + 1e-12)
    if kernel == "matern32":
        c = np.sqrt(3.0)
        return signal_var * (1.0 + c * r) * torch.exp(-c * r)
    if kernel == "matern52":
        c = np.sqrt(5.0)
        return signal_var * (1.0 + c * r + (5.0 / 3.0) * d2) * torch.exp(-c * r)
    raise ValueError(f"Unknown kernel {kernel!r}.")


def _predict_localgp_from_qv(
    *,
    regions,
    V_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    X_query: torch.Tensor,
    MC_num: int,
    kernel: str,
    local_gp_lengthscale: list[float] | None,
    local_gp_signal_var: float,
    local_gp_noise_var: float,
    pred_var_floor: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Plain local GP diagnostic; this is not the JumpGP/CEM predictor."""
    t0 = time.perf_counter()
    W_samples = _sample_W_from_qv(
        V_params=V_params,
        hyperparams=hyperparams,
        X_query=X_query,
        MC_num=MC_num,
        seed=seed,
    )
    M, T, Q, _ = W_samples.shape
    dtype = X_query.dtype
    device = X_query.device
    if local_gp_lengthscale is None or len(local_gp_lengthscale) != Q:
        ell = torch.ones(Q, device=device, dtype=dtype)
    else:
        ell = torch.as_tensor(local_gp_lengthscale, device=device, dtype=dtype).clamp_min(1e-6)
    signal_var = torch.as_tensor(float(local_gp_signal_var), device=device, dtype=dtype).clamp_min(1e-8)
    noise_var = torch.as_tensor(float(local_gp_noise_var), device=device, dtype=dtype).clamp_min(1e-8)

    mu_s = torch.empty((M, T), device=device, dtype=dtype)
    var_s = torch.empty((M, T), device=device, dtype=dtype)
    for m in range(M):
        for t in range(T):
            X_nb = regions[t]["X"].to(device=device, dtype=dtype)
            y_nb = regions[t]["y"].to(device=device, dtype=dtype).view(-1)
            W_t = W_samples[m, t]
            Z_nb = X_nb @ W_t.T
            z_q = (X_query[t : t + 1] @ W_t.T).view(1, Q)
            n = Z_nb.shape[0]

            K = _kernel_matrix(Z_nb, Z_nb, kernel=kernel, lengthscale=ell, signal_var=signal_var)
            K = K + (noise_var + 1e-5) * torch.eye(n, device=device, dtype=dtype)
            K = 0.5 * (K + K.T)
            L = torch.linalg.cholesky(K)
            y_mean = y_nb.mean()
            y_center = (y_nb - y_mean).view(n, 1)
            alpha = torch.cholesky_solve(y_center, L).view(n)

            k_star = _kernel_matrix(Z_nb, z_q, kernel=kernel, lengthscale=ell, signal_var=signal_var).view(n)
            mu = y_mean + torch.dot(k_star, alpha)
            v = torch.cholesky_solve(k_star.view(n, 1), L).view(n)
            var = (signal_var + noise_var - torch.dot(k_star, v)).clamp_min(float(pred_var_floor))
            mu_s[m, t] = mu
            var_s[m, t] = var

    mu = mu_s.mean(dim=0)
    total_var = (var_s + (mu_s - mu.view(1, T)).pow(2)).mean(dim=0)
    elapsed = time.perf_counter() - t0
    diag = {
        "local_predictor": "plain_gp_not_jumpgp",
        "kernel": kernel,
        "MC_num": int(MC_num),
        "local_gp_signal_var": float(signal_var.detach().cpu().item()),
        "local_gp_noise_var": float(noise_var.detach().cpu().item()),
        "local_gp_lengthscale": [float(v) for v in ell.detach().cpu().numpy().reshape(-1)],
    }
    return (
        mu.detach().cpu().numpy(),
        torch.sqrt(total_var.clamp_min(1e-12)).detach().cpu().numpy(),
        float(elapsed),
        diag,
    )


def _train_supervised_state(
    *,
    X_anchor_t: torch.Tensor,
    y_anchor_target_t: torch.Tensor,
    X_anchor_nb_list: list[torch.Tensor],
    y_anchor_nb_list: list[torch.Tensor],
    X_inducing_t: torch.Tensor,
    Q: int,
    m2: int,
    steps: int,
    lr: float,
    train_mc: int,
    kl_weight: float,
    pred_var_floor: float,
    grad_check: bool,
    device: torch.device,
):
    return _train_lmjgp_supervised_predictive(
        X_anchor_t=X_anchor_t,
        y_anchor_t=y_anchor_target_t,
        X_anchor_neighbors=X_anchor_nb_list,
        y_anchor_neighbors=y_anchor_nb_list,
        X_inducing_t=X_inducing_t,
        Q=Q,
        m2=m2,
        num_steps=steps,
        lr=lr,
        train_mc=train_mc,
        kl_weight=kl_weight,
        pred_var_floor=pred_var_floor,
        grad_check=grad_check,
        device=device,
    )


def _write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _tail_contribution_rows(pred_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for row in pred_rows:
        grouped.setdefault((row["dataset"], int(row["seed"]), row["method"]), []).append(row)
    out: list[dict] = []
    for (dataset, seed, method), rows in grouped.items():
        y = np.asarray([r["y_true"] for r in rows], dtype=np.float64)
        mu = np.asarray([r["mu"] for r in rows], dtype=np.float64)
        e2 = (y - mu) ** 2
        total = float(np.sum(e2))
        for frac in (0.05, 0.10, 0.20):
            k = max(1, int(np.ceil(frac * len(e2))))
            idx = np.argsort(e2)[-k:]
            out.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "top_abs_error_frac": frac,
                    "n_points": int(k),
                    "sse_share": float(np.sum(e2[idx]) / total) if total > 0 else float("nan"),
                    "rmse_all": float(np.sqrt(np.mean(e2))),
                    "rmse_top": float(np.sqrt(np.mean(e2[idx]))),
                    "mean_y_top": float(np.mean(y[idx])),
                }
            )
    return out


def _quantile_rows(pred_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for row in pred_rows:
        grouped.setdefault((row["dataset"], int(row["seed"]), row["method"]), []).append(row)
    out: list[dict] = []
    for (dataset, seed, method), rows in grouped.items():
        y = np.asarray([r["y_true"] for r in rows], dtype=np.float64)
        mu = np.asarray([r["mu"] for r in rows], dtype=np.float64)
        sig = np.asarray([max(r["sigma"], 1e-8) for r in rows], dtype=np.float64)
        q50, q90 = np.quantile(y, [0.5, 0.9])
        groups = [
            ("y<=q50", y <= q50),
            ("q50<y<=q90", (y > q50) & (y <= q90)),
            ("y>q90", y > q90),
        ]
        for group, mask in groups:
            if not np.any(mask):
                out.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "group": group,
                        "n": 0,
                        "y_min": float("nan"),
                        "y_max": float("nan"),
                        "rmse": float("nan"),
                        "mean_crps": float("nan"),
                        "coverage_90": float("nan"),
                        "mean_width_90": float("nan"),
                        "mean_sigma": float("nan"),
                    }
                )
                continue
            yy = y[mask]
            mm = mu[mask]
            ss = sig[mask]
            lo = mm - Z90 * ss
            hi = mm + Z90 * ss
            crps = mean_gaussian_crps_torch(
                torch.as_tensor(yy, dtype=torch.float32),
                torch.as_tensor(mm, dtype=torch.float32),
                torch.as_tensor(ss, dtype=torch.float32),
            ).item()
            out.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "group": group,
                    "n": int(mask.sum()),
                    "y_min": float(np.min(yy)),
                    "y_max": float(np.max(yy)),
                    "rmse": float(np.sqrt(np.mean((yy - mm) ** 2))),
                    "mean_crps": float(crps),
                    "coverage_90": float(np.mean((yy >= lo) & (yy <= hi))),
                    "mean_width_90": float(np.mean(hi - lo)),
                    "mean_sigma": float(np.mean(ss)),
                }
            )
    return out


def _write_readme(out_dir: Path, args: argparse.Namespace, metric_rows: list[dict]) -> None:
    lines = [
        "# Small-train LMJGP Remedies",
        "",
        "This folder was produced by `experiments/uci/try_lmjgp_remedies.py`.",
        "",
        "The run keeps existing `lmjgp_transductive` / `lmjgp_supervised` code untouched and adds experiment-only variants:",
        "",
        "- `lmjgp_supervised_base`: strict supervised q(V) + JumpGP prediction.",
        "- `lmjgp_supervised_hybrid_pls`: PLS-score/raw-X union neighborhoods + JumpGP prediction.",
        "- `lmjgp_supervised_residual_ridge`: ridge global mean plus LMJGP residual prediction.",
        "- `lmjgp_supervised_hybrid_residual_ridge`: hybrid neighborhoods on ridge score plus residual prediction.",
        "- `lmjgp_supervised_localgp_*`: plain local-GP kernel diagnostics, not JumpGP/CEM.",
        "",
        "## Command",
        "",
        "```powershell",
        "python experiments/uci/try_lmjgp_remedies.py "
        f"--datasets \"{args.datasets}\" --max_total_train {args.max_total_train} "
        f"--max_test_anchors {args.max_test_anchors} --max_train_anchors {args.max_train_anchors} "
        f"--neighbors {args.neighbors} --lmjgp_steps {args.lmjgp_steps} --MC_num {args.MC_num} "
        f"--xgb_boots {args.xgb_boots} --scaling_preset {args.scaling_preset} --out_dir {args.out_dir}",
        "```",
        "",
        "## Metrics",
        "",
        "| dataset | method | n | RMSE | CRPS | Cov90 | Width90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row.get('n', '')} | "
            f"{float(row['rmse']):.4f} | {float(row['mean_crps']):.4f} | "
            f"{float(row['coverage_90']):.3f} | {float(row['mean_width_90']):.4f} |"
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Try UCI small-train LMJGP remedy variants.")
    p.add_argument("--datasets", type=str, default="Wine Quality")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_total_train", type=int, default=200)
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--max_train_anchors", type=int, default=128)
    p.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    p.add_argument("--neighbors", type=str, default="auto",
                   help="'auto' uses Wine/Parkinsons=25 and Appliances=35; otherwise an integer.")
    p.add_argument("--hybrid_x_neighbors", type=int, default=20)
    p.add_argument("--hybrid_score_neighbors", type=int, default=15)
    p.add_argument("--pls_score_components", type=int, default=3)
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--lmjgp_supervised_train_mc", type=int, default=2)
    p.add_argument("--lmjgp_supervised_kl_weight", type=float, default=1e-3)
    p.add_argument("--lmjgp_supervised_pred_var_floor", type=float, default=0.05)
    p.add_argument("--lmjgp_supervised_grad_check", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--scaling_preset", type=str, default="uci_empirical", choices=("global", "uci_empirical"))
    p.add_argument("--standardize_x", type=int, default=1)
    p.add_argument("--standardize_y", type=int, default=1)
    p.add_argument("--neighbor_pool", type=str, default="train", choices=("base", "train"))
    p.add_argument("--out_dir", type=str, default="experiments/uci/smalltrain_lmjgp_remedies")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = set(DEFAULT_METHODS.split(","))
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}", flush=True)
    print(f"writing artifacts to {out_dir}", flush=True)

    metric_rows: list[dict] = []
    pred_rows: list[dict] = []
    diagnostics: dict = {"_meta": sanitize_for_json(vars(args)), "datasets": {}}

    for dataset_name in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        _set_seed(args.seed)
        random.seed(args.seed)
        print(f"\n{'=' * 72}\n{dataset_name} | seed={args.seed}\n{'=' * 72}", flush=True)
        X_train_full, X_test_full, y_train_full, y_test_full, K_default = load_uci_split(dataset_name, args.seed)
        X_train_full, y_train_full, train_idx = _subsample(
            X_train_full, y_train_full, args.max_total_train, args.seed + 500
        )
        X_test_sub, y_test_sub, test_idx = _subsample(
            X_test_full, y_test_full, args.max_test_anchors, args.seed + 1000
        )
        X_base, y_base, X_anchor, y_anchor, split_info = _split_base_anchor(
            X_train_full,
            y_train_full,
            max_train_anchors=args.max_train_anchors,
            seed=args.seed + 2000,
        )
        if args.scaling_preset == "uci_empirical":
            standardize_x, standardize_y = SCALING_PRESETS.get(
                dataset_name, (bool(args.standardize_x), bool(args.standardize_y))
            )
        else:
            standardize_x, standardize_y = bool(args.standardize_x), bool(args.standardize_y)
        (
            X_train_s,
            X_base_s,
            X_anchor_s,
            X_test_s,
            y_train_s,
            y_base_s,
            y_anchor_s,
            y_test_s,
            _,
            y_scaler,
        ) = _apply_scaling(
            X_train_full,
            X_base,
            X_anchor,
            X_test_sub,
            y_train_full,
            y_base,
            y_anchor,
            y_test_sub,
            standardize_x=standardize_x,
            standardize_y=standardize_y,
        )

        Q = int(args.Q if args.Q is not None else K_default)
        n = NEIGHBOR_PRESET.get(dataset_name, 35) if args.neighbors == "auto" else int(args.neighbors)
        hx = min(int(args.hybrid_x_neighbors), n)
        hs = min(int(args.hybrid_score_neighbors), n)
        if hx + hs < n:
            hx = max(hx, n - hs)

        X_train_t = torch.from_numpy(X_train_s).float().to(device)
        y_train_t = torch.from_numpy(y_train_s).float().to(device)
        X_base_t = torch.from_numpy(X_base_s).float().to(device)
        y_base_t = torch.from_numpy(y_base_s).float().to(device)
        X_anchor_t = torch.from_numpy(X_anchor_s).float().to(device)
        y_anchor_t = torch.from_numpy(y_anchor_s).float().to(device)
        X_test_t = torch.from_numpy(X_test_s).float().to(device)

        if args.neighbor_pool == "train":
            X_pool_t = X_train_t
            y_pool_t = y_train_t
            X_pool_s = X_train_s
            y_pool_s = y_train_s
        else:
            X_pool_t = X_base_t
            y_pool_t = y_base_t
            X_pool_s = X_base_s
            y_pool_s = y_base_s

        pls_score = _fit_pls_score(X_train_s, y_train_s, args.pls_score_components)
        score_train = pls_score(X_train_s)
        score_base = pls_score(X_base_s)
        score_anchor = pls_score(X_anchor_s)
        score_test = pls_score(X_test_s)
        score_pool = score_train if args.neighbor_pool == "train" else score_base

        ridge_mean, ridge_diag = _fit_ridge_mean(X_train_s, y_train_s, args.ridge_alpha)
        ridge_train = ridge_mean(X_train_s)
        ridge_base = ridge_mean(X_base_s)
        ridge_anchor = ridge_mean(X_anchor_s)
        ridge_test = ridge_mean(X_test_s)
        ridge_pool = ridge_train if args.neighbor_pool == "train" else ridge_base
        y_train_resid = y_train_s - ridge_train
        y_base_resid = y_base_s - ridge_base
        y_anchor_resid = y_anchor_s - ridge_anchor
        y_pool_resid = y_pool_s - ridge_pool
        y_base_resid_t = torch.from_numpy(y_base_resid).float().to(device)
        y_anchor_resid_t = torch.from_numpy(y_anchor_resid).float().to(device)
        y_pool_resid_t = torch.from_numpy(y_pool_resid).float().to(device)

        y_test_orig, _ = _invert_y(y_scaler, y_test_s)

        diagnostics["datasets"][dataset_name] = {
            "seed": int(args.seed),
            "Q": int(Q),
            "D": int(X_train_s.shape[1]),
            "n": int(n),
            "split": {
                **split_info,
                "n_train_total": int(X_train_s.shape[0]),
                "n_test": int(X_test_s.shape[0]),
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
            },
            "scaling": {"standardize_x": bool(standardize_x), "standardize_y": bool(standardize_y)},
            "ridge": ridge_diag,
            "methods": {},
        }

        def record(method: str, mu_s: np.ndarray, sig_s: np.ndarray, wall_sec: float, diag: dict):
            metrics = _pack_metrics_maybe_original_y(y_scaler, y_test_s, mu_s, sig_s, wall_sec)
            metric = {"dataset": dataset_name, "seed": args.seed, "method": method, "n": n}
            metric.update(_as_metric_dict(metrics))
            metric_rows.append(metric)
            mu_orig, sig_orig = _invert_y(y_scaler, mu_s, sig_s)
            for i, idx in enumerate(test_idx):
                pred_rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": int(args.seed),
                        "method": method,
                        "n": int(n),
                        "test_order": int(i),
                        "test_index": int(idx),
                        "y_true": float(y_test_orig[i]),
                        "mu": float(mu_orig[i]),
                        "sigma": float(max(sig_orig[i], 1e-12)),
                        "lower90": float(mu_orig[i] - Z90 * sig_orig[i]),
                        "upper90": float(mu_orig[i] + Z90 * sig_orig[i]),
                    }
                )
            diagnostics["datasets"][dataset_name]["methods"][method] = {
                "metrics": metric,
                "diagnostics": sanitize_for_json(diag),
            }
            print(
                f"  {method}: rmse={metric['rmse']:.4f} crps={metric['mean_crps']:.4f} "
                f"cov90={metric['coverage_90']:.3f}",
                flush=True,
            )

        if "xgboost" in methods:
            print(f"[{dataset_name}] running xgboost", flush=True)
            mu, sig, wall, diag = _run_xgboost_bootstrap(
                X_train=X_train_s,
                y_train=y_train_s,
                X_test=X_test_s,
                y_test=y_test_s,
                n_bootstrap=args.xgb_boots,
                seed=args.seed,
            )
            record("xgboost", mu, sig, wall, diag)

        need_base_state = any(
            m in methods
            for m in (
                "lmjgp_supervised_base",
                "lmjgp_supervised_localgp_rbf",
                "lmjgp_supervised_localgp_matern32",
                "lmjgp_supervised_localgp_matern52",
            )
        )
        base_state = None
        if need_base_state:
            print(f"[{dataset_name}] training base supervised q(V)", flush=True)
            _, X_anchor_nb, y_anchor_nb = _build_regions(
                X_anchor_t,
                None,
                X_base_t,
                y_base_t,
                device=device,
                m1=args.m1,
                Q=Q,
                n=n,
            )
            V_params, hyperparams, train_diag = _train_supervised_state(
                X_anchor_t=X_anchor_t,
                y_anchor_target_t=y_anchor_t,
                X_anchor_nb_list=X_anchor_nb,
                y_anchor_nb_list=y_anchor_nb,
                X_inducing_t=X_train_t,
                Q=Q,
                m2=args.m2,
                steps=args.lmjgp_steps,
                lr=args.lr,
                train_mc=args.lmjgp_supervised_train_mc,
                kl_weight=args.lmjgp_supervised_kl_weight,
                pred_var_floor=args.lmjgp_supervised_pred_var_floor,
                grad_check=bool(args.lmjgp_supervised_grad_check),
                device=device,
            )
            test_regions, _, _ = _build_regions(
                X_test_t,
                None,
                X_pool_t,
                y_pool_t,
                device=device,
                m1=args.m1,
                Q=Q,
                n=n,
            )
            base_state = (V_params, hyperparams, train_diag, test_regions)
            if "lmjgp_supervised_base" in methods:
                print(f"[{dataset_name}] predicting lmjgp_supervised_base", flush=True)
                mu, sig, wall = _predict_lmjgp_from_qv(
                    regions=test_regions,
                    V_params=V_params,
                    hyperparams=hyperparams,
                    X_query=X_test_t,
                    y_query_np=y_test_s,
                    MC_num=args.MC_num,
                )
                diag = dict(train_diag)
                diag["variant"] = "base_raw_neighbors"
                record("lmjgp_supervised_base", mu, sig, wall, diag)

            local_kernel_map = {
                "lmjgp_supervised_localgp_rbf": "rbf",
                "lmjgp_supervised_localgp_matern32": "matern32",
                "lmjgp_supervised_localgp_matern52": "matern52",
            }
            for method, kernel in local_kernel_map.items():
                if method not in methods:
                    continue
                print(f"[{dataset_name}] predicting {method}", flush=True)
                mu, sig, wall, diag = _predict_localgp_from_qv(
                    regions=test_regions,
                    V_params=V_params,
                    hyperparams=hyperparams,
                    X_query=X_test_t,
                    MC_num=args.MC_num,
                    kernel=kernel,
                    local_gp_lengthscale=train_diag.get("local_gp_lengthscale"),
                    local_gp_signal_var=float(train_diag.get("local_gp_signal_var", 1.0)),
                    local_gp_noise_var=float(train_diag.get("local_gp_noise_var", 0.05)),
                    pred_var_floor=args.lmjgp_supervised_pred_var_floor,
                    seed=args.seed + 7000,
                )
                diag["variant"] = "base_qv_plain_local_gp_kernel_diagnostic"
                record(method, mu, sig, wall, diag)

        if "lmjgp_supervised_hybrid_pls" in methods:
            print(f"[{dataset_name}] running lmjgp_supervised_hybrid_pls", flush=True)
            _, X_anchor_nb, y_anchor_nb = _build_hybrid_regions(
                X_anchor_t,
                X_base_t,
                y_base_t,
                score_anchors=score_anchor,
                score_pool=score_base,
                device=device,
                m1=args.m1,
                Q=Q,
                n_total=n,
                n_x=hx,
                n_score=hs,
            )
            V_params, hyperparams, train_diag = _train_supervised_state(
                X_anchor_t=X_anchor_t,
                y_anchor_target_t=y_anchor_t,
                X_anchor_nb_list=X_anchor_nb,
                y_anchor_nb_list=y_anchor_nb,
                X_inducing_t=X_train_t,
                Q=Q,
                m2=args.m2,
                steps=args.lmjgp_steps,
                lr=args.lr,
                train_mc=args.lmjgp_supervised_train_mc,
                kl_weight=args.lmjgp_supervised_kl_weight,
                pred_var_floor=args.lmjgp_supervised_pred_var_floor,
                grad_check=False,
                device=device,
            )
            test_regions, _, _ = _build_hybrid_regions(
                X_test_t,
                X_pool_t,
                y_pool_t,
                score_anchors=score_test,
                score_pool=score_pool,
                device=device,
                m1=args.m1,
                Q=Q,
                n_total=n,
                n_x=hx,
                n_score=hs,
            )
            mu, sig, wall = _predict_lmjgp_from_qv(
                regions=test_regions,
                V_params=V_params,
                hyperparams=hyperparams,
                X_query=X_test_t,
                y_query_np=y_test_s,
                MC_num=args.MC_num,
            )
            diag = dict(train_diag)
            diag.update(
                {
                    "variant": "hybrid_pls_neighbors",
                    "hybrid_x_neighbors": int(hx),
                    "hybrid_score_neighbors": int(hs),
                    "pls_score_components": int(args.pls_score_components),
                }
            )
            record("lmjgp_supervised_hybrid_pls", mu, sig, wall, diag)

        if "lmjgp_supervised_residual_ridge" in methods:
            print(f"[{dataset_name}] running lmjgp_supervised_residual_ridge", flush=True)
            _, X_anchor_nb, y_anchor_nb = _build_regions(
                X_anchor_t,
                None,
                X_base_t,
                y_base_resid_t,
                device=device,
                m1=args.m1,
                Q=Q,
                n=n,
            )
            V_params, hyperparams, train_diag = _train_supervised_state(
                X_anchor_t=X_anchor_t,
                y_anchor_target_t=y_anchor_resid_t,
                X_anchor_nb_list=X_anchor_nb,
                y_anchor_nb_list=y_anchor_nb,
                X_inducing_t=X_train_t,
                Q=Q,
                m2=args.m2,
                steps=args.lmjgp_steps,
                lr=args.lr,
                train_mc=args.lmjgp_supervised_train_mc,
                kl_weight=args.lmjgp_supervised_kl_weight,
                pred_var_floor=args.lmjgp_supervised_pred_var_floor,
                grad_check=False,
                device=device,
            )
            test_regions, _, _ = _build_regions(
                X_test_t,
                None,
                X_pool_t,
                y_pool_resid_t,
                device=device,
                m1=args.m1,
                Q=Q,
                n=n,
            )
            mu_r, sig, wall = _predict_lmjgp_from_qv(
                regions=test_regions,
                V_params=V_params,
                hyperparams=hyperparams,
                X_query=X_test_t,
                y_query_np=y_test_s - ridge_test,
                MC_num=args.MC_num,
            )
            diag = dict(train_diag)
            diag.update({"variant": "ridge_global_mean_plus_lmjgp_residual", **ridge_diag})
            record("lmjgp_supervised_residual_ridge", mu_r + ridge_test, sig, wall, diag)

        if "lmjgp_supervised_hybrid_residual_ridge" in methods:
            print(f"[{dataset_name}] running lmjgp_supervised_hybrid_residual_ridge", flush=True)
            _, X_anchor_nb, y_anchor_nb = _build_hybrid_regions(
                X_anchor_t,
                X_base_t,
                y_base_resid_t,
                score_anchors=ridge_anchor,
                score_pool=ridge_base,
                device=device,
                m1=args.m1,
                Q=Q,
                n_total=n,
                n_x=hx,
                n_score=hs,
            )
            V_params, hyperparams, train_diag = _train_supervised_state(
                X_anchor_t=X_anchor_t,
                y_anchor_target_t=y_anchor_resid_t,
                X_anchor_nb_list=X_anchor_nb,
                y_anchor_nb_list=y_anchor_nb,
                X_inducing_t=X_train_t,
                Q=Q,
                m2=args.m2,
                steps=args.lmjgp_steps,
                lr=args.lr,
                train_mc=args.lmjgp_supervised_train_mc,
                kl_weight=args.lmjgp_supervised_kl_weight,
                pred_var_floor=args.lmjgp_supervised_pred_var_floor,
                grad_check=False,
                device=device,
            )
            test_regions, _, _ = _build_hybrid_regions(
                X_test_t,
                X_pool_t,
                y_pool_resid_t,
                score_anchors=ridge_test,
                score_pool=ridge_pool,
                device=device,
                m1=args.m1,
                Q=Q,
                n_total=n,
                n_x=hx,
                n_score=hs,
            )
            mu_r, sig, wall = _predict_lmjgp_from_qv(
                regions=test_regions,
                V_params=V_params,
                hyperparams=hyperparams,
                X_query=X_test_t,
                y_query_np=y_test_s - ridge_test,
                MC_num=args.MC_num,
            )
            diag = dict(train_diag)
            diag.update(
                {
                    "variant": "ridge_score_hybrid_neighbors_plus_residual",
                    "hybrid_x_neighbors": int(hx),
                    "hybrid_score_neighbors": int(hs),
                    **ridge_diag,
                }
            )
            record("lmjgp_supervised_hybrid_residual_ridge", mu_r + ridge_test, sig, wall, diag)

        _write_csv(
            out_dir / "metrics.csv",
            metric_rows,
            ["dataset", "seed", "method", "n", *RESULT_ROW_KEYS],
        )
        _write_csv(
            out_dir / "predictions.csv",
            pred_rows,
            [
                "dataset",
                "seed",
                "method",
                "n",
                "test_order",
                "test_index",
                "y_true",
                "mu",
                "sigma",
                "lower90",
                "upper90",
            ],
        )
        tail_rows = _tail_contribution_rows(pred_rows)
        quant_rows = _quantile_rows(pred_rows)
        _write_csv(
            out_dir / "tail_contribution.csv",
            tail_rows,
            [
                "dataset",
                "seed",
                "method",
                "top_abs_error_frac",
                "n_points",
                "sse_share",
                "rmse_all",
                "rmse_top",
                "mean_y_top",
            ],
        )
        _write_csv(
            out_dir / "true_y_quantile_metrics.csv",
            quant_rows,
            [
                "dataset",
                "seed",
                "method",
                "group",
                "n",
                "y_min",
                "y_max",
                "rmse",
                "mean_crps",
                "coverage_90",
                "mean_width_90",
                "mean_sigma",
            ],
        )
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(diagnostics), f, indent=2)
        _write_readme(out_dir, args, metric_rows)
        print(f"partial artifacts saved to {out_dir}", flush=True)

    print(f"\nSaved artifacts -> {out_dir}", flush=True)
    return diagnostics


if __name__ == "__main__":
    main()

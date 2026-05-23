"""
Diagnose learned projection matrices for standardized UCI inputs.

This runner compares projection-side diagnostics for:

* ``lmjgp_transductive``
* ``lmjgp_supervised``
* ``cem_em_transductive``
* ``cem_em_supervised``

It is intentionally diagnostic-only: it trains the projection learners and
exports summaries of the induced test-anchor projection matrices, but it does
not run the slower final JumpGP prediction pass. For LMJGP, diagnostics are
computed from row-normalized posterior samples of ``q(W_*)`` because ``E[W]``
alone can be close to zero even when the metric moment ``E[W^T W]`` is not.

How to run (repository root, conda env: ``jumpGP``):

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke test on a tiny Wine slice.
    python experiments/uci/diagnose_projection_matrices_std.py ^
        --datasets "Wine Quality" --num_exp 1 ^
        --max_total_train 120 --max_test_anchors 8 --max_train_anchors 12 ^
        --n 12 --lmjgp_steps 10 --n_outer 1 --n_m_steps 5 --mc_diag 4 ^
        --out_dir experiments/uci/projection_matrix_std_smoke

    # Main standardized-X diagnostic on the three UCI datasets.
    python experiments/uci/diagnose_projection_matrices_std.py ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --num_exp 1 --max_total_train 500 --max_test_anchors 64 --max_train_anchors 64 ^
        --n -1 --lmjgp_steps 300 --n_outer 5 --n_m_steps 80 --mc_diag 16 ^
        --standardize_x 1 --standardize_y_mode uci_empirical ^
        --out_dir experiments/uci/projection_matrix_std_diagnostics

Agents: update this docstring when CLI flags or defaults change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics.projection_posterior import (  # noqa: E402
    compute_full_projection_diagnostics,
    flatten_per_anchor_for_export,
    metric_A_from_moment,
    sample_row_normalized_W,
    sanitize_for_json,
    sym_matrix_stats,
)
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    compute_pls_W0,
    gp_condition_coefficients,
    induced_W_from_coefficients,
    per_anchor_W_diagnostics,
    sample_induced_coefficients,
    train_projection_cem_em,
)
from djgp.variational import qW_from_qV, train_vi  # noqa: E402
from experiments.uci.compare_inductive_projection import (  # noqa: E402
    SCALING_PRESETS,
    _apply_scaling,
    _build_regions,
    _init_lmjgp_params,
    _set_seed,
    _split_base_anchor,
    _subsample,
    _train_lmjgp_supervised_predictive,
)
from experiments.uci.uci_data import load_uci_split  # noqa: E402


METHODS_DEFAULT = (
    "lmjgp_transductive,lmjgp_supervised,cem_em_transductive,cem_em_supervised"
)
DATASETS_DEFAULT = (
    "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
)


def _safe_float(x: Any) -> float:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def _nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _upper_tri_values(M: np.ndarray) -> np.ndarray:
    if M.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    delta = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 0.0))


def _feature_and_metric_diagnostics(
    W_samples: torch.Tensor,
    *,
    W_ref: torch.Tensor | None = None,
) -> dict[str, float]:
    """Projection diagnostics that are invariant to row sign flips."""
    if W_samples.dim() == 3:
        W_samples = W_samples.unsqueeze(0)
    W = W_samples.detach().float().cpu()
    M, T, Q, D = W.shape
    A = torch.einsum("mtqd,mtqe->mtde", W, W)
    A_mean_anchor = A.mean(dim=0)

    traces, eff_ranks, top_eigs = [], [], []
    for t in range(T):
        st = sym_matrix_stats(A_mean_anchor[t])
        traces.append(_safe_float(st["trace"]))
        eff_ranks.append(_safe_float(st["eff_rank"]))
        top_eigs.append(_safe_float(st["top_eig"]))

    diag_weight = torch.diagonal(A, dim1=-2, dim2=-1).mean(dim=(0, 1)).numpy()
    diag_weight = np.maximum(diag_weight, 0.0)
    total = float(diag_weight.sum())
    if total > 1e-12:
        p = diag_weight / total
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        eff_features = float(np.exp(entropy))
        top_share = float(np.max(p))
        top3_share = float(np.sort(p)[-min(3, D) :].sum())
    else:
        entropy = float("nan")
        eff_features = float("nan")
        top_share = float("nan")
        top3_share = float("nan")

    mean_A = A_mean_anchor.mean(dim=0)
    mean_A_norm = float(torch.linalg.norm(mean_A).item())
    disp = torch.linalg.norm(
        A_mean_anchor - mean_A.view(1, D, D), dim=(-2, -1)
    ).numpy()
    anchor_dispersion = float(np.nanmedian(disp))
    anchor_dispersion_rel = anchor_dispersion / max(mean_A_norm, 1e-12)

    W_flat = W.reshape(M * T, Q, D)
    frobs, w_eff_ranks = [], []
    for i in range(W_flat.shape[0]):
        s = torch.linalg.svdvals(W_flat[i])
        frobs.append(float(torch.linalg.norm(W_flat[i]).item()))
        denom = float(torch.sum(s**4).item())
        if denom > 1e-20:
            w_eff_ranks.append(float((torch.sum(s**2) ** 2 / torch.sum(s**4)).item()))
        else:
            w_eff_ranks.append(float("nan"))

    out = {
        "metric_trace_median": _nanmedian(traces),
        "metric_eff_rank_median": _nanmedian(eff_ranks),
        "metric_top_eig_median": _nanmedian(top_eigs),
        "feature_weight_entropy": entropy,
        "feature_eff_count": eff_features,
        "feature_top1_share": top_share,
        "feature_top3_share": top3_share,
        "metric_anchor_dispersion_frob": anchor_dispersion,
        "metric_anchor_dispersion_rel": anchor_dispersion_rel,
        "sampled_W_frob_median": _nanmedian(frobs),
        "sampled_W_eff_rank_median": _nanmedian(w_eff_ranks),
    }

    if W_ref is not None:
        W_ref_cpu = W_ref.detach().float().cpu()
        A_ref = W_ref_cpu.T @ W_ref_cpu
        ref_norm = float(torch.linalg.norm(A_ref).item())
        cos = []
        for t in range(T):
            denom = max(float(torch.linalg.norm(A_mean_anchor[t]).item()) * ref_norm, 1e-12)
            cos.append(float(torch.sum(A_mean_anchor[t] * A_ref).item() / denom))
        out["metric_cosine_to_pls_median"] = _nanmedian(cos)
    return out


def _response_geometry_diagnostics(
    W_samples: torch.Tensor,
    X_anchor: torch.Tensor,
    y_anchor: torch.Tensor,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    *,
    k_response: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Check whether projected distances are more response-aware than raw-X distances."""
    if W_samples.dim() == 3:
        W_samples = W_samples.unsqueeze(0)
    W_np = W_samples.detach().float().cpu().numpy()
    X_anchor_np = X_anchor.detach().float().cpu().numpy()
    y_anchor_np = y_anchor.detach().float().cpu().numpy().reshape(-1)
    M, T = W_np.shape[:2]

    rows: list[dict[str, float]] = []
    raw_pair_corrs, proj_pair_corrs = [], []
    raw_knn_abs, proj_knn_abs, all_abs = [], [], []

    for t in range(T):
        Xn = X_neighbors_list[t].detach().float().cpu().numpy()
        yn = y_neighbors_list[t].detach().float().cpu().numpy().reshape(-1)
        ya = float(y_anchor_np[t])
        n = Xn.shape[0]
        if n < 2:
            continue

        raw_pair = _upper_tri_values(_pairwise_euclidean(Xn))
        y_pair = _upper_tri_values(np.abs(yn[:, None] - yn[None, :]))
        raw_corr = _pearson_corr(raw_pair, y_pair)

        raw_anchor_dist = np.sqrt(np.maximum(np.sum((Xn - X_anchor_np[t : t + 1]) ** 2, axis=1), 0.0))
        kk = max(1, min(int(k_response), n))
        raw_idx = np.argsort(raw_anchor_dist)[:kk]
        raw_knn = float(np.mean(np.abs(yn[raw_idx] - ya)))
        all_y = float(np.mean(np.abs(yn - ya)))

        proj_corr_samples, proj_knn_samples = [], []
        for m in range(M):
            Wtm = W_np[m, t]
            Zn = Xn @ Wtm.T
            Za = X_anchor_np[t : t + 1] @ Wtm.T
            proj_pair = _upper_tri_values(_pairwise_euclidean(Zn))
            proj_corr_samples.append(_pearson_corr(proj_pair, y_pair))
            proj_anchor_dist = np.sqrt(np.maximum(np.sum((Zn - Za) ** 2, axis=1), 0.0))
            proj_idx = np.argsort(proj_anchor_dist)[:kk]
            proj_knn_samples.append(float(np.mean(np.abs(yn[proj_idx] - ya))))

        proj_corr = _nanmedian(proj_corr_samples)
        proj_knn = _nanmedian(proj_knn_samples)
        raw_pair_corrs.append(raw_corr)
        proj_pair_corrs.append(proj_corr)
        raw_knn_abs.append(raw_knn)
        proj_knn_abs.append(proj_knn)
        all_abs.append(all_y)

        rows.append(
            {
                "anchor": float(t),
                "raw_pair_y_abs_corr": raw_corr,
                "proj_pair_y_abs_corr_median": proj_corr,
                "proj_minus_raw_pair_y_abs_corr": proj_corr - raw_corr,
                "raw_anchor_knn_abs_y": raw_knn,
                "proj_anchor_knn_abs_y_median": proj_knn,
                "proj_minus_raw_anchor_knn_abs_y": proj_knn - raw_knn,
                "all_neighbor_abs_y": all_y,
            }
        )

    summary = {
        "raw_pair_y_abs_corr_median": _nanmedian(raw_pair_corrs),
        "proj_pair_y_abs_corr_median": _nanmedian(proj_pair_corrs),
        "proj_minus_raw_pair_y_abs_corr_median": _nanmedian(
            [p - r for p, r in zip(proj_pair_corrs, raw_pair_corrs)]
        ),
        "raw_anchor_knn_abs_y_median": _nanmedian(raw_knn_abs),
        "proj_anchor_knn_abs_y_median": _nanmedian(proj_knn_abs),
        "proj_minus_raw_anchor_knn_abs_y_median": _nanmedian(
            [p - r for p, r in zip(proj_knn_abs, raw_knn_abs)]
        ),
        "all_neighbor_abs_y_median": _nanmedian(all_abs),
    }
    return summary, rows


def _lmjgp_full_diag_summary(
    mu_W: torch.Tensor,
    sigma_W: torch.Tensor,
    cov_W: torch.Tensor,
    regions: list[dict],
    sigma_k: torch.Tensor,
    sigma_noise: torch.Tensor,
    *,
    mc_diag: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    diag = compute_full_projection_diagnostics(
        mu_W,
        sigma_W,
        cov_W,
        regions,
        sigma_k,
        sigma_noise,
        mc_M=mc_diag,
        mc_seed=seed,
    )
    flat = flatten_per_anchor_for_export(diag["per_anchor"])
    summary: dict[str, float] = {}
    for section, values in diag.items():
        if isinstance(values, dict):
            for key, val in values.items():
                if isinstance(val, (int, float)):
                    summary[f"{section}_{key}"] = _safe_float(val)
    return summary, flat


def _train_lmjgp_transductive_for_diag(
    *,
    X_test_t: torch.Tensor,
    X_pool_t: torch.Tensor,
    y_pool_t: torch.Tensor,
    X_inducing_t: torch.Tensor,
    Q: int,
    m1: int,
    m2: int,
    n: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> dict[str, Any]:
    regions, X_nb_list, y_nb_list = _build_regions(
        X_test_t,
        None,
        X_pool_t,
        y_pool_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    T, D = X_test_t.shape
    V_params, u_params, hyperparams = _init_lmjgp_params(
        X_inducing_pool=X_inducing_t,
        X_anchors=X_test_t,
        T=T,
        D=D,
        Q=Q,
        m1=m1,
        m2=m2,
        device=device,
    )
    t0 = time.perf_counter()
    V_params, u_params, hyperparams = train_vi(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=lr,
        num_steps=steps,
        log_interval=max(50, steps // 4),
    )
    train_time = time.perf_counter() - t0
    with torch.no_grad():
        mu_W, cov_W = qW_from_qV(
            X_test_t,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
        sigma_W = torch.sqrt(torch.clamp(cov_W.diagonal(dim1=-2, dim2=-1), min=1e-24))
        sigma_k = torch.stack(
            [torch.abs(u["sigma_k"].detach()).clamp_min(1e-6) for u in u_params]
        ).view(-1)
        sigma_noise = torch.stack(
            [torch.abs(u["sigma_noise"].detach()).clamp_min(1e-6) for u in u_params]
        ).view(-1)
    return {
        "mu_W": mu_W,
        "cov_W": cov_W,
        "sigma_W": sigma_W,
        "regions": regions,
        "X_nb_list": X_nb_list,
        "y_nb_list": y_nb_list,
        "sigma_k": sigma_k,
        "sigma_noise": sigma_noise,
        "train_time_sec": train_time,
    }


def _train_lmjgp_supervised_for_diag(
    *,
    X_anchor_t: torch.Tensor,
    y_anchor_t: torch.Tensor,
    X_test_t: torch.Tensor,
    X_base_t: torch.Tensor,
    y_base_t: torch.Tensor,
    X_pool_t: torch.Tensor,
    y_pool_t: torch.Tensor,
    X_inducing_t: torch.Tensor,
    Q: int,
    m1: int,
    m2: int,
    n: int,
    steps: int,
    lr: float,
    train_mc: int,
    kl_weight: float,
    pred_var_floor: float,
    grad_check: bool,
    device: torch.device,
) -> dict[str, Any]:
    _, X_anchor_nb_list, y_anchor_nb_list = _build_regions(
        X_anchor_t,
        None,
        X_base_t,
        y_base_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    V_params, hyperparams, train_diag = _train_lmjgp_supervised_predictive(
        X_anchor_t=X_anchor_t,
        y_anchor_t=y_anchor_t,
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
    test_regions, X_test_nb_list, y_test_nb_list = _build_regions(
        X_test_t,
        None,
        X_pool_t,
        y_pool_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    with torch.no_grad():
        mu_W, cov_W = qW_from_qV(
            X_test_t,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
        sigma_W = torch.sqrt(torch.clamp(cov_W.diagonal(dim1=-2, dim2=-1), min=1e-24))
        y_var = torch.var(y_pool_t.detach()).clamp_min(1e-4)
        sigma_k = torch.ones(X_test_t.shape[0], device=device)
        sigma_noise = torch.sqrt(torch.full_like(sigma_k, float(0.05 * y_var.item())))
    return {
        "mu_W": mu_W,
        "cov_W": cov_W,
        "sigma_W": sigma_W,
        "regions": test_regions,
        "X_nb_list": X_test_nb_list,
        "y_nb_list": y_test_nb_list,
        "sigma_k": sigma_k,
        "sigma_noise": sigma_noise,
        "train_time_sec": float(train_diag.get("train_time_sec", float("nan"))),
        "train_diag": train_diag,
    }


def _train_cem_transductive_for_diag(
    *,
    X_test_t: torch.Tensor,
    X_pool_t: torch.Tensor,
    y_pool_t: torch.Tensor,
    V_t: torch.Tensor,
    W_pls_t: torch.Tensor,
    Q: int,
    m1: int,
    n: int,
    cfg: CemEmConfig,
    init_C_scale: float,
    coeff_scale: float,
    row_normalize_W: bool,
    device: torch.device,
) -> dict[str, Any]:
    _, X_nb_list, y_nb_list = _build_regions(
        X_test_t,
        None,
        X_pool_t,
        y_pool_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    model = CoeffProjection(
        T=X_test_t.shape[0],
        Q=Q,
        D=X_test_t.shape[1],
        V=V_t,
        W0_init=W_pls_t,
        learn_W0=True,
        init_C_scale=init_C_scale,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).to(device)
    cfg_local = CemEmConfig(**{**cfg.__dict__, "lambda_anchor_pred": 0.0})
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        X_anchors=X_test_t,
        device=device,
        cfg=cfg_local,
        label="cem_em_transductive_diag",
    )
    return {
        "W_samples": res.W.detach().unsqueeze(0),
        "W_mean": res.W.detach(),
        "X_nb_list": X_nb_list,
        "y_nb_list": y_nb_list,
        "train_time_sec": float(res.train_time_sec),
        "best_outer": int(res.best_outer),
        "cem_diag": per_anchor_W_diagnostics(res.W.detach(), W_ref=W_pls_t),
    }


def _train_cem_supervised_for_diag(
    *,
    X_anchor_t: torch.Tensor,
    y_anchor_t: torch.Tensor,
    X_test_t: torch.Tensor,
    X_base_t: torch.Tensor,
    y_base_t: torch.Tensor,
    X_pool_t: torch.Tensor,
    y_pool_t: torch.Tensor,
    V_t: torch.Tensor,
    W_pls_t: torch.Tensor,
    Q: int,
    m1: int,
    n: int,
    cfg: CemEmConfig,
    init_C_scale: float,
    coeff_scale: float,
    row_normalize_W: bool,
    coeff_gp_ridge: float,
    coeff_gp_lengthscale: float | None,
    mc_diag: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    _, X_anchor_nb_list, y_anchor_nb_list = _build_regions(
        X_anchor_t,
        None,
        X_base_t,
        y_base_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    model = CoeffProjection(
        T=X_anchor_t.shape[0],
        Q=Q,
        D=X_anchor_t.shape[1],
        V=V_t,
        W0_init=W_pls_t,
        learn_W0=True,
        init_C_scale=init_C_scale,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).to(device)
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_anchor_nb_list,
        y_neighbors_list=y_anchor_nb_list,
        X_anchors=X_anchor_t,
        y_anchors=y_anchor_t,
        device=device,
        cfg=cfg,
        label="cem_em_supervised_diag",
    )
    cond = gp_condition_coefficients(
        X_anchor_t,
        res.C,
        X_test_t,
        ridge=coeff_gp_ridge,
        lengthscale=coeff_gp_lengthscale,
    )
    C_samples = sample_induced_coefficients(
        cond["mean"],
        cond["var"],
        n_samples=mc_diag,
        seed=seed,
    )
    W_samples = induced_W_from_coefficients(
        C_samples,
        res.W0,
        V_t,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).detach()
    W_mean = induced_W_from_coefficients(
        cond["mean"],
        res.W0,
        V_t,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).detach()
    _, X_test_nb_list, y_test_nb_list = _build_regions(
        X_test_t,
        None,
        X_pool_t,
        y_pool_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    return {
        "W_samples": W_samples,
        "W_mean": W_mean,
        "X_nb_list": X_test_nb_list,
        "y_nb_list": y_test_nb_list,
        "train_time_sec": float(res.train_time_sec),
        "best_outer": int(res.best_outer),
        "coeff_gp_lengthscale": float(cond["lengthscale"]),
        "coeff_gp_ridge": float(cond["ridge"]),
        "mean_coeff_gp_var": float(cond["var"].detach().mean().cpu().item()),
        "cem_diag": per_anchor_W_diagnostics(W_mean, W_ref=W_pls_t),
    }


def _dataset_n(args: argparse.Namespace, dataset: str) -> int:
    if args.n > 0:
        return int(args.n)
    if dataset == "Appliances Energy Prediction":
        return 35
    return 25


def _standardize_y_for_dataset(mode: str, dataset: str, fallback: bool) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True
    if mode == "uci_empirical":
        return bool(SCALING_PRESETS.get(dataset, (True, fallback))[1])
    raise ValueError(f"Unknown standardize_y_mode={mode!r}")


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        groups[(str(row["dataset"]), str(row["method"]))].append(row)
    out: list[dict[str, Any]] = []
    skip = {
        "seed",
        "dataset",
        "method",
        "standardize_x",
        "standardize_y",
        "n",
        "Q",
        "max_total_train",
        "max_test_anchors",
        "max_train_anchors",
    }
    for (dataset, method), rows in sorted(groups.items()):
        agg: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "n_seeds": len(rows),
        }
        numeric_keys = sorted(
            {
                k
                for row in rows
                for k, v in row.items()
                if k not in skip and isinstance(v, (int, float))
            }
        )
        for key in numeric_keys:
            vals = [_safe_float(row.get(key)) for row in rows]
            agg[f"{key}_mean"] = _nanmean(vals)
            agg[f"{key}_std"] = float(np.nanstd(np.asarray(vals, dtype=np.float64)))
        out.append(agg)
    return out


def run_one_dataset_seed(
    args: argparse.Namespace,
    *,
    dataset: str,
    seed: int,
    methods: list[str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train_np, X_test_np, y_train_np, y_test_np, K = load_uci_split(dataset, seed)
    X_train_np, y_train_np, train_idx = _subsample(
        X_train_np, y_train_np, args.max_total_train, seed + 101
    )
    X_test_np, y_test_np, test_idx = _subsample(
        X_test_np, y_test_np, args.max_test_anchors, seed + 202
    )
    X_base, y_base, X_anchor, y_anchor, split_info = _split_base_anchor(
        X_train_np,
        y_train_np,
        max_train_anchors=args.max_train_anchors,
        seed=seed + 303,
    )

    standardize_x = bool(args.standardize_x)
    standardize_y = _standardize_y_for_dataset(
        args.standardize_y_mode, dataset, fallback=bool(args.standardize_y)
    )
    (
        X_train_s,
        X_base_s,
        X_anchor_s,
        X_test_s,
        y_train_s,
        y_base_s,
        y_anchor_s,
        y_test_s,
        _x_scaler,
        _y_scaler,
    ) = _apply_scaling(
        X_train_np,
        X_base,
        X_anchor,
        X_test_np,
        y_train_np,
        y_base,
        y_anchor,
        y_test_np,
        standardize_x=standardize_x,
        standardize_y=standardize_y,
    )

    Q = int(args.Q if args.Q is not None else K)
    n = _dataset_n(args, dataset)
    m1, m2 = int(args.m1), int(args.m2)

    X_train_t = torch.from_numpy(X_train_s).float().to(device)
    y_train_t = torch.from_numpy(y_train_s).float().to(device)
    X_base_t = torch.from_numpy(X_base_s).float().to(device)
    y_base_t = torch.from_numpy(y_base_s).float().to(device)
    X_anchor_t = torch.from_numpy(X_anchor_s).float().to(device)
    y_anchor_t = torch.from_numpy(y_anchor_s).float().to(device)
    X_test_t = torch.from_numpy(X_test_s).float().to(device)
    y_test_t = torch.from_numpy(y_test_s).float().to(device)

    if args.neighbor_pool == "train":
        X_pool_t, y_pool_t = X_train_t, y_train_t
    else:
        X_pool_t, y_pool_t = X_base_t, y_base_t

    V_np, V_info = build_V_basis(
        X_train_s,
        y_train_s,
        n_pls=args.n_pls,
        n_pca=args.n_pca,
        n_random=args.n_random,
        seed=seed,
    )
    W_pls_np = compute_pls_W0(X_train_s, y_train_s, Q=Q)
    V_t = torch.from_numpy(V_np).float().to(device)
    W_pls_t = torch.from_numpy(W_pls_np).float().to(device)

    cfg = CemEmConfig(
        n_outer=args.n_outer,
        n_m_steps=args.n_m_steps,
        lr=args.cem_lr,
        lambda_gate=args.lambda_gate,
        lambda_smooth=args.lambda_smooth,
        lambda_scale=args.lambda_scale,
        lambda_anchor_pred=args.lambda_anchor_pred,
        anchor_pred_var_floor=args.anchor_pred_var_floor,
        normalize_smoothness=bool(args.normalize_smoothness),
        track_best_W=bool(args.track_best_W),
        cem_progress=False,
        verbose=bool(args.cem_verbose),
    )

    summary_rows: list[dict[str, Any]] = []
    per_anchor_rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "settings": {
            "standardize_x": standardize_x,
            "standardize_y": standardize_y,
            "Q": Q,
            "n": n,
            "m1": m1,
            "m2": m2,
            "max_total_train": args.max_total_train,
            "max_test_anchors": args.max_test_anchors,
            "max_train_anchors": args.max_train_anchors,
            "neighbor_pool": args.neighbor_pool,
            "split_info": split_info,
            "V_info": V_info,
            "train_idx_size": int(len(train_idx)),
            "test_idx_size": int(len(test_idx)),
        },
        "methods": {},
    }

    for method in methods:
        print(f"[{dataset} seed={seed}] {method}", flush=True)
        method_t0 = time.perf_counter()
        extra_summary: dict[str, Any] = {}
        lmjgp_flat: list[dict[str, Any]] = []

        if method == "lmjgp_transductive":
            out = _train_lmjgp_transductive_for_diag(
                X_test_t=X_test_t,
                X_pool_t=X_pool_t,
                y_pool_t=y_pool_t,
                X_inducing_t=X_train_t,
                Q=Q,
                m1=m1,
                m2=m2,
                n=n,
                steps=args.lmjgp_steps,
                lr=args.lmjgp_lr,
                device=device,
            )
            W_samples = sample_row_normalized_W(
                out["mu_W"], out["sigma_W"], args.mc_diag, seed=seed
            )
            extra_summary, lmjgp_flat = _lmjgp_full_diag_summary(
                out["mu_W"],
                out["sigma_W"],
                out["cov_W"],
                out["regions"],
                out["sigma_k"],
                out["sigma_noise"],
                mc_diag=args.mc_diag,
                seed=seed,
            )
            X_nb_list, y_nb_list = out["X_nb_list"], out["y_nb_list"]
            train_time = out["train_time_sec"]

        elif method == "lmjgp_supervised":
            out = _train_lmjgp_supervised_for_diag(
                X_anchor_t=X_anchor_t,
                y_anchor_t=y_anchor_t,
                X_test_t=X_test_t,
                X_base_t=X_base_t,
                y_base_t=y_base_t,
                X_pool_t=X_pool_t,
                y_pool_t=y_pool_t,
                X_inducing_t=X_train_t,
                Q=Q,
                m1=m1,
                m2=m2,
                n=n,
                steps=args.lmjgp_steps,
                lr=args.lmjgp_lr,
                train_mc=args.lmjgp_supervised_train_mc,
                kl_weight=args.lmjgp_supervised_kl_weight,
                pred_var_floor=args.lmjgp_supervised_pred_var_floor,
                grad_check=bool(args.lmjgp_supervised_grad_check),
                device=device,
            )
            W_samples = sample_row_normalized_W(
                out["mu_W"], out["sigma_W"], args.mc_diag, seed=seed
            )
            extra_summary, lmjgp_flat = _lmjgp_full_diag_summary(
                out["mu_W"],
                out["sigma_W"],
                out["cov_W"],
                out["regions"],
                out["sigma_k"],
                out["sigma_noise"],
                mc_diag=args.mc_diag,
                seed=seed,
            )
            train_diag = out.get("train_diag", {})
            for key in ("best_loss", "final_loss", "gradient_mu_norm", "gradient_sigma_norm"):
                if key in train_diag:
                    extra_summary[f"train_{key}"] = _safe_float(train_diag[key])
            X_nb_list, y_nb_list = out["X_nb_list"], out["y_nb_list"]
            train_time = out["train_time_sec"]

        elif method == "cem_em_transductive":
            out = _train_cem_transductive_for_diag(
                X_test_t=X_test_t,
                X_pool_t=X_pool_t,
                y_pool_t=y_pool_t,
                V_t=V_t,
                W_pls_t=W_pls_t,
                Q=Q,
                m1=m1,
                n=n,
                cfg=cfg,
                init_C_scale=args.init_C_scale,
                coeff_scale=args.coeff_scale,
                row_normalize_W=bool(args.row_normalize_W),
                device=device,
            )
            W_samples = out["W_samples"]
            X_nb_list, y_nb_list = out["X_nb_list"], out["y_nb_list"]
            train_time = out["train_time_sec"]
            extra_summary.update({f"cem_{k}": v for k, v in out["cem_diag"].items()})
            extra_summary["best_outer"] = out["best_outer"]

        elif method == "cem_em_supervised":
            out = _train_cem_supervised_for_diag(
                X_anchor_t=X_anchor_t,
                y_anchor_t=y_anchor_t,
                X_test_t=X_test_t,
                X_base_t=X_base_t,
                y_base_t=y_base_t,
                X_pool_t=X_pool_t,
                y_pool_t=y_pool_t,
                V_t=V_t,
                W_pls_t=W_pls_t,
                Q=Q,
                m1=m1,
                n=n,
                cfg=cfg,
                init_C_scale=args.init_C_scale,
                coeff_scale=args.coeff_scale,
                row_normalize_W=bool(args.row_normalize_W),
                coeff_gp_ridge=args.coeff_gp_ridge,
                coeff_gp_lengthscale=args.coeff_gp_lengthscale,
                mc_diag=args.mc_diag,
                seed=seed,
                device=device,
            )
            W_samples = out["W_samples"]
            X_nb_list, y_nb_list = out["X_nb_list"], out["y_nb_list"]
            train_time = out["train_time_sec"]
            extra_summary.update({f"cem_{k}": v for k, v in out["cem_diag"].items()})
            extra_summary["best_outer"] = out["best_outer"]
            extra_summary["coeff_gp_lengthscale"] = out["coeff_gp_lengthscale"]
            extra_summary["coeff_gp_ridge"] = out["coeff_gp_ridge"]
            extra_summary["mean_coeff_gp_var"] = out["mean_coeff_gp_var"]

        else:
            raise ValueError(f"Unknown method={method!r}")

        feature_summary = _feature_and_metric_diagnostics(W_samples, W_ref=W_pls_t)
        response_summary, response_rows = _response_geometry_diagnostics(
            W_samples,
            X_test_t,
            y_test_t,
            X_nb_list,
            y_nb_list,
            k_response=args.k_response,
        )

        elapsed = time.perf_counter() - method_t0
        row = {
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "standardize_x": int(standardize_x),
            "standardize_y": int(standardize_y),
            "Q": Q,
            "n": n,
            "max_total_train": int(args.max_total_train or X_train_np.shape[0]),
            "max_test_anchors": int(X_test_t.shape[0]),
            "max_train_anchors": int(X_anchor_t.shape[0]),
            "train_time_sec": _safe_float(train_time),
            "elapsed_sec": elapsed,
            **feature_summary,
            **response_summary,
            **extra_summary,
        }
        summary_rows.append(row)

        for anchor_row in response_rows:
            flat = {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                **anchor_row,
            }
            anchor_idx = int(anchor_row["anchor"])
            if anchor_idx < len(lmjgp_flat):
                flat.update({f"lmjgp_{k}": v for k, v in lmjgp_flat[anchor_idx].items()})
            per_anchor_rows.append(flat)

        nested["methods"][method] = sanitize_for_json(
            {
                "summary": row,
                "per_anchor_response": response_rows,
                "lmjgp_projection_posterior_flat": lmjgp_flat,
            }
        )
    return summary_rows, per_anchor_rows, nested


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Projection-matrix diagnostics for standardized UCI LMJGP/CEM-EM variants."
    )
    p.add_argument("--datasets", type=str, default=DATASETS_DEFAULT)
    p.add_argument("--methods", type=str, default=METHODS_DEFAULT)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--max_total_train", type=int, default=500)
    p.add_argument("--max_test_anchors", type=int, default=64)
    p.add_argument("--max_train_anchors", type=int, default=64)
    p.add_argument("--neighbor_pool", choices=("base", "train"), default="base")
    p.add_argument("--standardize_x", type=int, default=1)
    p.add_argument("--standardize_y", type=int, default=0)
    p.add_argument(
        "--standardize_y_mode",
        choices=("none", "all", "uci_empirical"),
        default="uci_empirical",
        help="uci_empirical uses existing UCI preset: y-standardize Appliances only.",
    )
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--n", type=int, default=-1, help="-1 uses Wine/Parkinsons=25, Appliances=35.")
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--lmjgp_lr", type=float, default=0.01)
    p.add_argument("--lmjgp_supervised_train_mc", type=int, default=2)
    p.add_argument("--lmjgp_supervised_kl_weight", type=float, default=1e-3)
    p.add_argument("--lmjgp_supervised_pred_var_floor", type=float, default=0.05)
    p.add_argument("--lmjgp_supervised_grad_check", type=int, default=1)
    p.add_argument("--cem_lr", type=float, default=0.01)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_m_steps", type=int, default=80)
    p.add_argument("--lambda_gate", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_scale", type=float, default=0.0)
    p.add_argument("--lambda_anchor_pred", type=float, default=1.0)
    p.add_argument("--anchor_pred_var_floor", type=float, default=0.05)
    p.add_argument("--init_C_scale", type=float, default=0.0)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--row_normalize_W", type=int, default=1)
    p.add_argument("--normalize_smoothness", type=int, default=1)
    p.add_argument("--track_best_W", type=int, default=1)
    p.add_argument("--cem_verbose", type=int, default=0)
    p.add_argument("--coeff_gp_ridge", type=float, default=1e-4)
    p.add_argument("--coeff_gp_lengthscale", type=float, default=None)
    p.add_argument("--mc_diag", type=int, default=16)
    p.add_argument("--k_response", type=int, default=10)
    p.add_argument("--device", type=str, default="auto", help="'auto' uses CUDA when available.")
    p.add_argument("--out_dir", type=str, default="experiments/uci/projection_matrix_std_diagnostics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = set(METHODS_DEFAULT.split(","))
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; allowed={sorted(allowed)}")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    per_anchor_rows: list[dict[str, Any]] = []
    nested_runs: list[dict[str, Any]] = []

    for exp in range(args.num_exp):
        seed = int(args.seed0 + exp)
        for dataset in datasets:
            rows, anchor_rows, nested = run_one_dataset_seed(
                args,
                dataset=dataset,
                seed=seed,
                methods=methods,
                device=device,
            )
            summary_rows.extend(rows)
            per_anchor_rows.extend(anchor_rows)
            nested_runs.append(nested)

    aggregate_rows = _aggregate_rows(summary_rows)
    _write_dict_csv(out_dir / "summary.csv", summary_rows)
    _write_dict_csv(out_dir / "aggregate.csv", aggregate_rows)
    _write_dict_csv(out_dir / "per_anchor.csv", per_anchor_rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            sanitize_for_json(
                {
                    "args": vars(args),
                    "summary_rows": summary_rows,
                    "aggregate_rows": aggregate_rows,
                    "runs": nested_runs,
                }
            ),
            f,
            indent=2,
        )
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Standardized UCI Projection-Matrix Diagnostics\n\n"
            "Generated by `experiments/uci/diagnose_projection_matrices_std.py`.\n\n"
            "Files:\n"
            "- `summary.csv`: one row per dataset/seed/method.\n"
            "- `aggregate.csv`: mean/std over seeds for numeric diagnostics.\n"
            "- `per_anchor.csv`: response-geometry diagnostics per test anchor.\n"
            "- `summary.json`: nested run metadata and LMJGP posterior diagnostic rows.\n\n"
            "Interpretation notes:\n"
            "- `feature_eff_count` near the input dimension means the projection spreads weight broadly.\n"
            "- `metric_anchor_dispersion_rel` near zero means little anchor-specific variation.\n"
            "- positive `proj_minus_raw_pair_y_abs_corr_median` means projected pairwise distances align\n"
            "  with response differences better than raw-X distances inside local neighborhoods.\n"
            "- negative `proj_minus_raw_anchor_knn_abs_y_median` means projected nearest neighbors are\n"
            "  response-closer to the anchor than raw-X nearest neighbors.\n"
        )
    print(f"Wrote diagnostics to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

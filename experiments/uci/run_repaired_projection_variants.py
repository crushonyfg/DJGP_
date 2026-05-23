"""Run repaired CEM-EM and LMJGP projection variants on UCI small-train splits.

How to run (repository root, conda env: ``jumpGP``):

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke test.
    python experiments/uci/run_repaired_projection_variants.py ^
        --datasets "Wine Quality" --train_sizes 120 --num_exp 1 ^
        --max_test_anchors 8 --max_train_anchors 12 --n 12 ^
        --lmjgp_steps 10 --cem_vi_steps 5 --n_outer 1 --n_m_steps 5 ^
        --MC_num 2 --methods "cem_supervised,cem_transductive,lowrank_pls_pca,lmjgp_supervised" ^
        --out_dir experiments/uci/repaired_projection_variants_smoke

    # Main UCI preset run.
    python experiments/uci/run_repaired_projection_variants.py ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --train_sizes 200,500 --num_exp 5 --max_test_anchors 64 ^
        --max_train_anchors 64 --n -1 --lmjgp_steps 300 --cem_vi_steps 120 ^
        --n_outer 5 --n_m_steps 80 --MC_num 3 --scaling_preset uci_empirical ^
        --out_dir experiments/uci/repaired_projection_variants_train200_500_5seeds

Agents: update this docstring when CLI flags or defaults change.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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

from djgp.diagnostics.projection_posterior import sanitize_for_json  # noqa: E402
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    cem_step_all,
    compute_pls_W0,
    diagonal_cem_data_hessian,
    gp_condition_coefficients,
    gp_condition_coefficients_laplace_diag,
    gp_condition_coefficients_vi_fullcov,
    induced_W_from_coefficients,
    run_projected_jumpgp,
    run_projected_jumpgp_ensemble,
    sample_induced_coefficients,
    sample_induced_coefficients_diag,
    sample_induced_coefficients_fullcov,
    train_cem_coefficient_vi,
    train_projection_cem_em,
)
from djgp.variational_lowrank import predict_vi_lowrank_residual, train_vi_lowrank_residual  # noqa: E402
from experiments.uci.compare_inductive_projection import (  # noqa: E402
    RESULT_ROW_KEYS,
    SCALING_PRESETS,
    _apply_scaling,
    _build_regions,
    _pack_metrics_maybe_original_y,
    _run_lmjgp_supervised,
    _set_seed,
    _split_base_anchor,
    _subsample,
)
from experiments.uci.compare_parkinsons_grouped_split import _load_parkinsons_grouped_split  # noqa: E402
from experiments.uci.compare_lowrank_residual_lmjgp import _init_lowrank_state  # noqa: E402
from experiments.uci.uci_data import load_uci_split  # noqa: E402


DATASETS_DEFAULT = "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
PAPER_SMALLTRAIN_PRESETS = {
    "Wine Quality": {"standardize_x": True, "standardize_y": False, "split_type": "random_row"},
    "Parkinsons Telemonitoring": {"standardize_x": True, "standardize_y": False, "split_type": "subject_grouped"},
    "Appliances Energy Prediction": {"standardize_x": True, "standardize_y": True, "split_type": "random_row"},
}


def _dataset_n(dataset: str, requested_n: int) -> int:
    if requested_n > 0:
        return int(requested_n)
    return 35 if dataset == "Appliances Energy Prediction" else 25


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    preferred = [
        "dataset",
        "train_size",
        "seed",
        "family",
        "method",
        "rmse",
        "crps",
        "cov90",
        "width90",
        "cov95",
        "width95",
        "wall_sec",
    ]
    for key in preferred:
        if any(key in r for r in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], int(row["train_size"]), row["method"]), []).append(row)
    out = []
    for (dataset, train_size, method), vals in sorted(grouped.items()):
        r = {"dataset": dataset, "train_size": train_size, "method": method, "n": len(vals)}
        for key in ("rmse", "crps", "cov90", "width90", "cov95", "width95"):
            arr = np.asarray([float(v.get(key, np.nan)) for v in vals], dtype=np.float64)
            r[f"{key}_mean"] = float(np.nanmean(arr))
            r[f"{key}_std"] = float(np.nanstd(arr))
        out.append(r)
    return out


def _safe_metrics_row(
    *,
    dataset: str,
    train_size: int,
    seed: int,
    family: str,
    method: str,
    y_scaler,
    y_test_scaled: np.ndarray,
    mu_scaled: np.ndarray,
    sigma_scaled: np.ndarray,
    wall_sec: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y = np.asarray(y_test_scaled).reshape(-1)
    mu = np.asarray(mu_scaled, dtype=np.float64).reshape(-1)
    sig = np.asarray(sigma_scaled, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sig) & (sig > 0) & (np.abs(mu) < 1e8) & (sig < 1e8)
    omitted = int(mask.size - mask.sum())
    if mask.sum() < max(3, int(0.5 * mask.size)):
        vals = [float("nan")] * len(RESULT_ROW_KEYS)
    else:
        vals = _pack_metrics_maybe_original_y(y_scaler, y[mask], mu[mask], sig[mask], wall_sec)
    row = {
        "dataset": dataset,
        "train_size": int(train_size),
        "seed": int(seed),
        "family": family,
        "method": method,
        **{k: float(v) for k, v in zip(RESULT_ROW_KEYS, vals)},
        "omitted_eval_points": omitted,
    }
    row["crps"] = row.get("mean_crps", float("nan"))
    row["cov90"] = row.get("coverage_90", float("nan"))
    row["width90"] = row.get("mean_width_90", float("nan"))
    row["cov95"] = row.get("coverage_95", float("nan"))
    row["width95"] = row.get("mean_width_95", float("nan"))
    if extra:
        row.update(extra)
    return row


def _eval_W(
    *,
    W_obj: torch.Tensor,
    ensemble: bool,
    X_test_t: torch.Tensor,
    X_nb_list,
    y_nb_list,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    if ensemble:
        out = run_projected_jumpgp_ensemble(
            W_obj,
            X_test_t,
            X_nb_list,
            y_nb_list,
            device=device,
            progress=False,
            desc="repaired_projection_ensemble",
        )
    else:
        out = run_projected_jumpgp(
            W_obj,
            X_test_t,
            X_nb_list,
            y_nb_list,
            device=device,
            progress=False,
            desc="repaired_projection_map",
        )
    return out["mu"], out["sigma"], float(out["elapsed_sec"])


def _cem_bundle(
    *,
    mode: str,
    X_fit_t: torch.Tensor,
    y_fit_t: torch.Tensor | None,
    X_test_t: torch.Tensor,
    X_nb_source_t: torch.Tensor,
    y_nb_source_t: torch.Tensor,
    X_eval_pool_t: torch.Tensor,
    y_eval_pool_t: torch.Tensor,
    V_t: torch.Tensor,
    W0_t: torch.Tensor,
    Q: int,
    m1: int,
    n: int,
    cfg: CemEmConfig,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    need_gp: bool = True,
    need_laplace: bool = True,
    need_vi: bool = True,
) -> dict[str, Any]:
    _, X_fit_nb_list, y_fit_nb_list = _build_regions(
        X_fit_t,
        None,
        X_nb_source_t,
        y_nb_source_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    model = CoeffProjection(
        T=X_fit_t.shape[0],
        Q=Q,
        D=X_fit_t.shape[1],
        V=V_t,
        W0_init=W0_t,
        learn_W0=True,
        init_C_scale=args.init_C_scale,
        coeff_scale=args.coeff_scale,
        row_normalize=bool(args.row_normalize_W),
    ).to(device)
    cfg_local = CemEmConfig(**{**cfg.__dict__})
    if y_fit_t is None:
        cfg_local.lambda_anchor_pred = 0.0
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_fit_nb_list,
        y_neighbors_list=y_fit_nb_list,
        X_anchors=X_fit_t,
        y_anchors=y_fit_t,
        device=device,
        cfg=cfg_local,
        label=f"cem_em_{mode}",
    )

    cache_model = CoeffProjection(
        T=X_fit_t.shape[0],
        Q=Q,
        D=X_fit_t.shape[1],
        V=V_t,
        W0_init=res.W0.detach(),
        learn_W0=True,
        init_C_scale=0.0,
        coeff_scale=args.coeff_scale,
        row_normalize=bool(args.row_normalize_W),
    ).to(device)
    with torch.no_grad():
        cache_model.W0.copy_(res.W0.detach())
        cache_model.C.copy_(res.C.detach())
    cache = cem_step_all(
        cache_model,
        X_neighbors_list=X_fit_nb_list,
        y_neighbors_list=y_fit_nb_list,
        X_anchors=X_fit_t,
        device=device,
        progress=False,
        desc=f"cem_{mode}_cache",
    )

    W_gp = None
    W_mean = None
    coeff_gp_lengthscale = float("nan")
    mean_coeff_gp_var = float("nan")
    if need_gp:
        cond = gp_condition_coefficients(
            X_fit_t,
            res.C.detach(),
            X_test_t,
            ridge=args.coeff_gp_ridge,
            lengthscale=args.coeff_gp_lengthscale,
        )
        C_gp = sample_induced_coefficients(cond["mean"], cond["var"], n_samples=args.MC_num, seed=seed + 17)
        W_gp = induced_W_from_coefficients(
            C_gp,
            res.W0.detach(),
            V_t,
            coeff_scale=args.coeff_scale,
            row_normalize=bool(args.row_normalize_W),
        ).detach()
        W_mean = induced_W_from_coefficients(
            cond["mean"],
            res.W0.detach(),
            V_t,
            coeff_scale=args.coeff_scale,
            row_normalize=bool(args.row_normalize_W),
        ).detach()
        coeff_gp_lengthscale = float(cond["lengthscale"])
        mean_coeff_gp_var = float(cond["var"].mean().detach().cpu().item())

    W_lap = None
    laplace_mean_var = float("nan")
    laplace_hessian_median = float("nan")
    laplace_hessian_failures = 0
    if need_laplace:
        lap = diagonal_cem_data_hessian(
            C_map=res.C.detach(),
            W0=res.W0.detach(),
            V=V_t.detach(),
            X_neighbors_list=X_fit_nb_list,
            y_neighbors_list=y_fit_nb_list,
            X_anchors=X_fit_t,
            y_anchors=y_fit_t,
            cache=cache,
            coeff_scale=args.coeff_scale,
            row_normalize=bool(args.row_normalize_W),
            lambda_gate=cfg_local.lambda_gate,
            lambda_anchor_pred=cfg_local.lambda_anchor_pred,
            anchor_pred_var_floor=cfg_local.anchor_pred_var_floor,
        )
        cond_lap = gp_condition_coefficients_laplace_diag(
            X_fit_t,
            res.C.detach(),
            X_test_t,
            lap["data_hessian_diag"],
            ridge=args.coeff_gp_ridge,
            lengthscale=args.coeff_gp_lengthscale,
        )
        C_lap = sample_induced_coefficients_diag(cond_lap["mean"], cond_lap["var"], n_samples=args.MC_num, seed=seed + 29)
        W_lap = induced_W_from_coefficients(
            C_lap,
            res.W0.detach(),
            V_t,
            coeff_scale=args.coeff_scale,
            row_normalize=bool(args.row_normalize_W),
        ).detach()
        laplace_mean_var = float(cond_lap["var"].mean().detach().cpu().item())
        laplace_hessian_median = float(lap["median_data_hessian_diag"])
        laplace_hessian_failures = int(lap["n_hessian_failures"])

    W_vi = None
    vi_train_time_sec = 0.0
    vi_query_var_mean = float("nan")
    vi_anchor_var_mean = float("nan")
    vi_kl = float("nan")
    vi_best_loss = float("nan")
    if need_vi:
        vi = train_cem_coefficient_vi(
            C_init=res.C.detach(),
            W0=res.W0.detach(),
            V=V_t.detach(),
            X_anchor=X_fit_t,
            X_neighbors_list=X_fit_nb_list,
            y_neighbors_list=y_fit_nb_list,
            cache=cache,
            y_anchor=y_fit_t,
            coeff_scale=args.coeff_scale,
            row_normalize=bool(args.row_normalize_W),
            lambda_gate=cfg_local.lambda_gate,
            lambda_anchor_pred=cfg_local.lambda_anchor_pred,
            anchor_pred_var_floor=cfg_local.anchor_pred_var_floor,
            gp_ridge=args.coeff_gp_ridge,
            gp_lengthscale=args.coeff_gp_lengthscale,
            n_steps=args.cem_vi_steps,
            lr=args.cem_vi_lr,
            n_mc=args.cem_vi_mc,
            kl_weight=args.cem_vi_kl_weight,
            init_std=args.cem_vi_init_std,
            verbose=bool(args.verbose_vi),
        )
        cond_vi = gp_condition_coefficients_vi_fullcov(X_fit_t, vi, X_test_t)
        C_vi = sample_induced_coefficients_fullcov(
            cond_vi["mean"],
            cond_vi["cov"],
            n_samples=args.MC_num,
            seed=seed + 43,
        )
        W_vi = induced_W_from_coefficients(
            C_vi,
            res.W0.detach(),
            V_t,
            coeff_scale=args.coeff_scale,
            row_normalize=bool(args.row_normalize_W),
        ).detach()
        vi_train_time_sec = float(vi["train_time_sec"])
        vi_query_var_mean = float(cond_vi["query_var_mean"])
        vi_anchor_var_mean = float(cond_vi["posterior_anchor_var_mean"])
        vi_kl = float(vi["kl"])
        vi_best_loss = float(vi["best_loss"])

    _, X_eval_nb_list, y_eval_nb_list = _build_regions(
        X_test_t,
        None,
        X_eval_pool_t,
        y_eval_pool_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    return {
        "mode": mode,
        "res": res,
        "W_map_or_mean": W_mean,
        "W_gp": W_gp,
        "W_lap": W_lap,
        "W_vi": W_vi,
        "X_eval_nb_list": X_eval_nb_list,
        "y_eval_nb_list": y_eval_nb_list,
        "train_time_sec": float(res.train_time_sec + vi_train_time_sec),
        "cem_map_train_time_sec": float(res.train_time_sec),
        "cem_vi_train_time_sec": float(vi_train_time_sec),
        "best_outer": int(res.best_outer),
        "coeff_gp_lengthscale": float(coeff_gp_lengthscale),
        "mean_coeff_gp_var": float(mean_coeff_gp_var),
        "laplace_mean_var": float(laplace_mean_var),
        "laplace_hessian_median": float(laplace_hessian_median),
        "laplace_hessian_failures": int(laplace_hessian_failures),
        "vi_query_var_mean": float(vi_query_var_mean),
        "vi_anchor_var_mean": float(vi_anchor_var_mean),
        "vi_kl": float(vi_kl),
        "vi_best_loss": float(vi_best_loss),
    }


def _cem_svd_basis(W: torch.Tensor, R: int, fallback: torch.Tensor) -> torch.Tensor:
    D = W.shape[-1]
    rows = W.detach().reshape(-1, D)
    rows = rows - rows.mean(dim=0, keepdim=True)
    try:
        _u, _s, vh = torch.linalg.svd(rows, full_matrices=False)
        V = vh[: min(R, vh.shape[0])].T
    except RuntimeError:
        V = fallback[:, : min(R, fallback.shape[1])]
    if V.shape[1] < R:
        V = torch.cat([V, fallback[:, : R - V.shape[1]]], dim=1)
    V, _ = torch.linalg.qr(V[:, :R], mode="reduced")
    return V


def _run_lowrank(
    *,
    dataset: str,
    train_size: int,
    seed: int,
    method: str,
    W0_t: torch.Tensor,
    V_t: torch.Tensor,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    y_test_s: np.ndarray,
    y_scaler,
    Q: int,
    n: int,
    split_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    regions, _, _ = _build_regions(X_test_t, None, X_train_t, y_train_t, device=device, m1=args.m1, Q=Q, n=n)
    C_params, u_params, hyperparams = _init_lowrank_state(
        X_train_t=X_train_t,
        X_test_t=X_test_t,
        Q=Q,
        R=int(V_t.shape[1]),
        m1=args.m1,
        m2=args.m2,
        W0_t=W0_t,
        V_t=V_t,
        residual_scale=args.residual_scale,
        device=device,
    )
    C_params, _u, hyperparams, diag = train_vi_lowrank_residual(
        regions,
        C_params,
        u_params,
        hyperparams,
        lr=args.lr,
        num_steps=args.lmjgp_steps,
        log_interval=max(1, args.lmjgp_steps),
    )
    mu, var, pred_time = predict_vi_lowrank_residual(regions, C_params, hyperparams, M=args.MC_num, seed=seed + 101)
    sigma = torch.sqrt(var.clamp_min(1e-12)).detach().cpu().numpy()
    return _safe_metrics_row(
        dataset=dataset,
        train_size=train_size,
        seed=seed,
        family="lmjgp_lowrank_residual",
        method=method,
        y_scaler=y_scaler,
        y_test_scaled=y_test_s,
        mu_scaled=mu.detach().cpu().numpy(),
        sigma_scaled=sigma,
        wall_sec=float(diag["train_time_sec"] + pred_time),
        extra={
            "train_time_sec": float(diag["train_time_sec"]),
            "pred_time_sec": float(pred_time),
            "best_elbo": float(diag["best_elbo"]),
            "R": int(V_t.shape[1]),
            "residual_scale": float(args.residual_scale),
            "split_type": split_type,
        },
    )


def run_one(args: argparse.Namespace, dataset: str, train_size: int, seed: int, device: torch.device) -> list[dict[str, Any]]:
    _set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if bool(args.paper_smalltrain_preset) and dataset == "Parkinsons Telemonitoring":
        split = _load_parkinsons_grouped_split(seed=seed, train_subject_frac=args.train_subject_frac)
        X_train_full, y_train_full, _ = _subsample(split["X_train"], split["y_train"], train_size, seed + 500)
        X_test_sub, y_test_sub, _ = _subsample(split["X_test"], split["y_test"], args.max_test_anchors, seed + 1000)
        K_default = split["K"]
        split_type = "subject_grouped"
    else:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw, K_default = load_uci_split(dataset, seed)
        X_train_full, y_train_full, _ = _subsample(X_train_raw, y_train_raw, train_size, seed + 500)
        X_test_sub, y_test_sub, _ = _subsample(X_test_raw, y_test_raw, args.max_test_anchors, seed + 1000)
        split_type = "random_row"
    X_base, y_base, X_anchor, y_anchor, _ = _split_base_anchor(
        X_train_full,
        y_train_full,
        max_train_anchors=args.max_train_anchors,
        seed=seed + 2000,
    )
    if bool(args.paper_smalltrain_preset):
        preset = PAPER_SMALLTRAIN_PRESETS.get(dataset)
        if preset is None:
            standardize_x, standardize_y = bool(args.standardize_x), bool(args.standardize_y)
        else:
            standardize_x, standardize_y = bool(preset["standardize_x"]), bool(preset["standardize_y"])
            split_type = str(preset["split_type"])
    elif args.scaling_preset == "uci_empirical":
        standardize_x, standardize_y = SCALING_PRESETS.get(dataset, (bool(args.standardize_x), bool(args.standardize_y)))
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
        _x_scaler,
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
    n = _dataset_n(dataset, args.n)
    X_train_t = torch.from_numpy(X_train_s).float().to(device)
    y_train_t = torch.from_numpy(y_train_s).float().to(device)
    X_base_t = torch.from_numpy(X_base_s).float().to(device)
    y_base_t = torch.from_numpy(y_base_s).float().to(device)
    X_anchor_t = torch.from_numpy(X_anchor_s).float().to(device)
    y_anchor_t = torch.from_numpy(y_anchor_s).float().to(device)
    X_test_t = torch.from_numpy(X_test_s).float().to(device)
    X_pool_t, y_pool_t = (X_base_t, y_base_t) if args.neighbor_pool == "base" else (X_train_t, y_train_t)

    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    cem_output_variants = {m.strip() for m in args.cem_output_variants.split(",") if m.strip()}
    need_cem_trans = (
        "cem_transductive" in methods
        or "cem_transductive_vi" in methods
        or "lowrank_cem_mean_pls_pca" in methods
        or "lowrank_cem_mean_cem_svd" in methods
    )
    need_cem_sup = "cem_supervised" in methods or "cem_supervised_vi" in methods
    rows: list[dict[str, Any]] = []
    print(f"[{dataset} train={train_size} seed={seed}] start", flush=True)

    V_np, _ = build_V_basis(X_train_s, y_train_s, n_pls=args.n_pls, n_pca=args.n_pca, n_random=args.n_random, seed=seed)
    W0_np = compute_pls_W0(X_train_s, y_train_s, Q=Q)
    V_t = torch.from_numpy(V_np).float().to(device)
    W0_t = torch.from_numpy(W0_np).float().to(device)
    cfg = CemEmConfig(
        n_outer=args.n_outer,
        n_m_steps=args.n_m_steps,
        lr=args.lr,
        lambda_gate=args.lambda_gate,
        lambda_smooth=args.lambda_smooth,
        lambda_scale=args.lambda_scale,
        lambda_anchor_pred=args.lambda_anchor_pred,
        anchor_pred_var_floor=args.anchor_pred_var_floor,
        normalize_smoothness=bool(args.normalize_smoothness),
        track_best_W=bool(args.track_best_W),
        snapshot_after_each_outer=False,
        cem_progress=False,
        verbose=False,
    )

    cem_trans = None
    if need_cem_trans:
        print(f"[{dataset} train={train_size} seed={seed}] CEM transductive", flush=True)
        cem_trans = _cem_bundle(
            mode="transductive",
            X_fit_t=X_test_t,
            y_fit_t=None,
            X_test_t=X_test_t,
            X_nb_source_t=X_pool_t,
            y_nb_source_t=y_pool_t,
            X_eval_pool_t=X_pool_t,
            y_eval_pool_t=y_pool_t,
            V_t=V_t,
            W0_t=W0_t,
            Q=Q,
            m1=args.m1,
            n=n,
            cfg=cfg,
            args=args,
            seed=seed + 300,
            device=device,
            need_gp=("gp" in cem_output_variants and ("cem_transductive" in methods)),
            need_laplace=("laplace" in cem_output_variants and ("cem_transductive" in methods)),
            need_vi=("vi" in cem_output_variants and ("cem_transductive" in methods or "cem_transductive_vi" in methods)),
        )
    if need_cem_sup:
        print(f"[{dataset} train={train_size} seed={seed}] CEM supervised", flush=True)
        cem_sup = _cem_bundle(
            mode="supervised",
            X_fit_t=X_anchor_t,
            y_fit_t=y_anchor_t,
            X_test_t=X_test_t,
            X_nb_source_t=X_base_t,
            y_nb_source_t=y_base_t,
            X_eval_pool_t=X_pool_t,
            y_eval_pool_t=y_pool_t,
            V_t=V_t,
            W0_t=W0_t,
            Q=Q,
            m1=args.m1,
            n=n,
            cfg=cfg,
            args=args,
            seed=seed + 400,
            device=device,
            need_gp=("gp" in cem_output_variants and ("cem_supervised" in methods)),
            need_laplace=("laplace" in cem_output_variants and ("cem_supervised" in methods)),
            need_vi=("vi" in cem_output_variants and ("cem_supervised" in methods or "cem_supervised_vi" in methods)),
        )
    else:
        cem_sup = None

    for cem in [cem_trans, cem_sup]:
        if cem is None:
            continue
        prefix = f"cem_em_{cem['mode']}"
        variants = {}
        if "gp" in cem_output_variants and cem["W_gp"] is not None:
            variants[f"{prefix}_gp_cond_ensemble"] = (cem["W_gp"], True)
        if "laplace" in cem_output_variants and cem["W_lap"] is not None:
            variants[f"{prefix}_laplace_ensemble"] = (cem["W_lap"], True)
        if "vi" in cem_output_variants and cem["W_vi"] is not None:
            variants[f"{prefix}_vi_fullcov_ensemble"] = (cem["W_vi"], True)
        if args.include_cem_map:
            variants[f"{prefix}_map_or_mean"] = (cem["W_map_or_mean"], False)
        for method, (W_obj, is_ens) in variants.items():
            mu, sig, pred_sec = _eval_W(
                W_obj=W_obj,
                ensemble=is_ens,
                X_test_t=X_test_t,
                X_nb_list=cem["X_eval_nb_list"],
                y_nb_list=cem["y_eval_nb_list"],
                device=device,
            )
            rows.append(
                _safe_metrics_row(
                    dataset=dataset,
                    train_size=train_size,
                    seed=seed,
                    family="cem_em_repaired",
                    method=method,
                    y_scaler=y_scaler,
                    y_test_scaled=y_test_s,
                    mu_scaled=mu,
                    sigma_scaled=sig,
                    wall_sec=pred_sec + float(cem["train_time_sec"]),
                    extra={
                        "standardize_x": int(standardize_x),
                        "standardize_y": int(standardize_y),
                        "split_type": split_type,
                        "Q": Q,
                        "n": n,
                        "MC_num": args.MC_num if is_ens else 1,
                        "train_time_sec": float(cem["train_time_sec"]),
                        "pred_time_sec": float(pred_sec),
                        "cem_map_train_time_sec": float(cem["cem_map_train_time_sec"]),
                        "cem_vi_train_time_sec": float(cem["cem_vi_train_time_sec"]),
                        "coeff_gp_lengthscale": float(cem["coeff_gp_lengthscale"]),
                        "mean_coeff_gp_var": float(cem["mean_coeff_gp_var"]),
                        "laplace_mean_var": float(cem["laplace_mean_var"]),
                        "laplace_hessian_median": float(cem["laplace_hessian_median"]),
                        "laplace_hessian_failures": int(cem["laplace_hessian_failures"]),
                        "vi_query_var_mean": float(cem["vi_query_var_mean"]),
                        "vi_anchor_var_mean": float(cem["vi_anchor_var_mean"]),
                        "vi_kl": float(cem["vi_kl"]),
                        "vi_best_loss": float(cem["vi_best_loss"]),
                    },
                )
            )

    if "lmjgp_supervised" in methods:
        print(f"[{dataset} train={train_size} seed={seed}] LMJGP supervised", flush=True)
        mu, sig, pred_sec, diag = _run_lmjgp_supervised(
            X_anchor_t=X_anchor_t,
            y_anchor_t=y_anchor_t,
            X_test_t=X_test_t,
            y_test_np=y_test_s,
            X_base_t=X_base_t,
            y_base_t=y_base_t,
            X_pool_t=X_pool_t,
            y_pool_t=y_pool_t,
            X_inducing_t=X_train_t,
            Q=Q,
            m1=args.m1,
            m2=args.m2,
            n=n,
            steps=args.lmjgp_steps,
            lr=args.lr,
            MC_num=args.MC_num,
            train_mc=args.lmjgp_supervised_train_mc,
            kl_weight=args.lmjgp_supervised_kl_weight,
            pred_var_floor=args.lmjgp_supervised_pred_var_floor,
            grad_check=bool(args.lmjgp_supervised_grad_check),
            device=device,
        )
        rows.append(
            _safe_metrics_row(
                dataset=dataset,
                train_size=train_size,
                seed=seed,
                family="lmjgp_supervised_reparam",
                method="lmjgp_supervised_reparam",
                y_scaler=y_scaler,
                y_test_scaled=y_test_s,
                mu_scaled=mu,
                sigma_scaled=sig,
                wall_sec=pred_sec + float(diag.get("train_time_sec", 0.0)),
                extra={
                "standardize_x": int(standardize_x),
                "standardize_y": int(standardize_y),
                "split_type": split_type,
                    "Q": Q,
                    "n": n,
                    "MC_num": args.MC_num,
                    "train_time_sec": float(diag.get("train_time_sec", np.nan)),
                    "pred_time_sec": float(pred_sec),
                    "has_qV_gradient": bool(diag.get("gradient_check", {}).get("has_qV_gradient", True)),
                },
            )
        )

    if "lowrank_pls_pca" in methods:
        print(f"[{dataset} train={train_size} seed={seed}] lowrank PLS/PCA", flush=True)
        rows.append(
            _run_lowrank(
                dataset=dataset,
                train_size=train_size,
                seed=seed,
            method="lmjgp_lowrank_pls_pca",
                W0_t=W0_t,
                V_t=V_t,
                X_train_t=X_train_t,
                y_train_t=y_train_t,
                X_test_t=X_test_t,
                y_test_s=y_test_s,
                y_scaler=y_scaler,
                Q=Q,
                n=n,
                split_type=split_type,
                args=args,
                device=device,
            )
        )
    if cem_trans is not None and "lowrank_cem_mean_pls_pca" in methods:
        print(f"[{dataset} train={train_size} seed={seed}] lowrank CEM-mean W0 + PLS/PCA V", flush=True)
        W_cem_mean = cem_trans["res"].W.detach().mean(dim=0)
        rows.append(
            _run_lowrank(
                dataset=dataset,
                train_size=train_size,
                seed=seed,
                method="lmjgp_lowrank_cem_mean_pls_pca",
                W0_t=W_cem_mean,
                V_t=V_t,
                X_train_t=X_train_t,
                y_train_t=y_train_t,
                X_test_t=X_test_t,
                y_test_s=y_test_s,
                y_scaler=y_scaler,
                Q=Q,
                n=n,
                split_type=split_type,
                args=args,
                device=device,
            )
        )
    if cem_trans is not None and "lowrank_cem_mean_cem_svd" in methods:
        print(f"[{dataset} train={train_size} seed={seed}] lowrank CEM-mean W0 + CEM-SVD V", flush=True)
        W_cem_mean = cem_trans["res"].W.detach().mean(dim=0)
        V_cem = _cem_svd_basis(cem_trans["res"].W.detach(), int(V_t.shape[1]), V_t)
        rows.append(
            _run_lowrank(
                dataset=dataset,
                train_size=train_size,
                seed=seed,
                method="lmjgp_lowrank_cem_mean_cem_svd",
                W0_t=W_cem_mean,
                V_t=V_cem,
                X_train_t=X_train_t,
                y_train_t=y_train_t,
                X_test_t=X_test_t,
                y_test_s=y_test_s,
                y_scaler=y_scaler,
                Q=Q,
                n=n,
                split_type=split_type,
                args=args,
                device=device,
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repaired UCI projection variants.")
    p.add_argument("--datasets", default=DATASETS_DEFAULT)
    p.add_argument("--train_sizes", default="200,500")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--methods", default="cem_supervised,cem_transductive,lowrank_pls_pca,lowrank_cem_mean_pls_pca,lowrank_cem_mean_cem_svd,lmjgp_supervised")
    p.add_argument("--cem_output_variants", default="gp,laplace,vi",
                   help="Comma-separated CEM ensembles to evaluate: gp, laplace, vi.")
    p.add_argument("--paper_smalltrain_preset", type=int, default=0,
                   help="If 1, match the paper small-train split/scaling policy; Parkinsons uses subject-grouped std_x.")
    p.add_argument("--train_subject_frac", type=float, default=0.8)
    p.add_argument("--max_test_anchors", type=int, default=64)
    p.add_argument("--max_train_anchors", type=int, default=64)
    p.add_argument("--neighbor_pool", choices=("base", "train"), default="train")
    p.add_argument("--scaling_preset", choices=("global", "uci_empirical"), default="uci_empirical")
    p.add_argument("--standardize_x", type=int, default=1)
    p.add_argument("--standardize_y", type=int, default=1)
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--n", type=int, default=-1)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--lmjgp_supervised_train_mc", type=int, default=2)
    p.add_argument("--lmjgp_supervised_kl_weight", type=float, default=1e-3)
    p.add_argument("--lmjgp_supervised_pred_var_floor", type=float, default=0.05)
    p.add_argument("--lmjgp_supervised_grad_check", type=int, default=0)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
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
    p.add_argument("--coeff_gp_ridge", type=float, default=1e-4)
    p.add_argument("--coeff_gp_lengthscale", type=float, default=None)
    p.add_argument("--cem_vi_steps", type=int, default=120)
    p.add_argument("--cem_vi_lr", type=float, default=0.005)
    p.add_argument("--cem_vi_mc", type=int, default=2)
    p.add_argument("--cem_vi_kl_weight", type=float, default=1.0)
    p.add_argument("--cem_vi_init_std", type=float, default=0.03)
    p.add_argument("--verbose_vi", type=int, default=0)
    p.add_argument("--include_cem_map", type=int, default=0)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default="experiments/uci/repaired_projection_variants_train200_500_5seeds")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    train_sizes = [int(s.strip()) for s in args.train_sizes.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []
    for train_size in train_sizes:
        for seed in range(args.seed_start, args.seed_start + args.num_exp):
            for dataset in datasets:
                rows.extend(run_one(args, dataset, train_size, seed, device))
                _write_rows(out_dir / "metrics_partial.csv", rows)
    agg = _aggregate(rows)
    _write_rows(out_dir / "metrics.csv", rows)
    _write_rows(out_dir / "aggregate.csv", agg)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json({"args": vars(args), "rows": rows, "aggregate": agg}), f, indent=2)
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Repaired Projection Variants\n\n"
            "Generated by `experiments/uci/run_repaired_projection_variants.py`.\n\n"
            "Rows include CEM-EM GP-conditional, diagonal-Laplace, and full-covariance VI ensembles; "
            "low-rank residual LMJGP variants; and the reparameterized supervised LMJGP.\n"
        )
    print(f"Wrote repaired projection variants to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

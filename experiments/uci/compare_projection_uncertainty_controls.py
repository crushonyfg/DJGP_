"""
One-seed UCI controls for projection uncertainty and trivial LMJGP projections.

This runner answers two diagnostic questions:

1. For supervised CEM-EM, does sampling test-side coefficients from the
   GP-conditional coefficient field improve RMSE/CRPS/coverage over the
   deterministic mean-coefficient MAP plug-in?
2. For LMJGP, does the learned projection posterior beat controls that destroy
   or replace learned direction information: prior/random unit-row projections,
   moment-matched isotropic projections, feature-permuted learned samples, and
   zero-mean learned-variance samples?

    The CEM-EM rows include the existing empirical-Bayes GP-conditional
    coefficient ensemble and a diagonal-Laplace coefficient posterior ensemble.

How to run (repository root, conda env: ``jumpGP``):

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke test on tiny Wine slices.
    python experiments/uci/compare_projection_uncertainty_controls.py ^
        --datasets "Wine Quality" --max_total_train 120 --max_test_anchors 8 ^
        --max_train_anchors 12 --n 12 --lmjgp_steps 10 --n_outer 1 ^
        --n_m_steps 5 --MC_num 2 --out_dir experiments/uci/projection_controls_smoke

    # One-seed UCI pilot with current empirical scaling presets.
    python experiments/uci/compare_projection_uncertainty_controls.py ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --max_total_train 500 --max_test_anchors 64 --max_train_anchors 64 ^
        --n -1 --lmjgp_steps 300 --n_outer 5 --n_m_steps 80 --MC_num 3 ^
        --scaling_preset uci_empirical ^
        --out_dir experiments/uci/projection_uncertainty_controls_1seed

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
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics.projection_posterior import (  # noqa: E402
    sample_row_normalized_W,
    sanitize_for_json,
)
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    compute_pls_W0,
    gp_condition_coefficients,
    gp_condition_coefficients_laplace_diag,
    induced_W_from_coefficients,
    run_projected_jumpgp,
    run_projected_jumpgp_ensemble,
    sample_induced_coefficients_diag,
    sample_induced_coefficients,
    train_projection_cem_em,
)
from djgp.projections.cem_em_projection import cem_step_all  # noqa: E402
from djgp.projections.laplace_cem import diagonal_cem_data_hessian  # noqa: E402
from experiments.uci.compare_inductive_projection import (  # noqa: E402
    SCALING_PRESETS,
    _apply_scaling,
    _build_regions,
    _pack_metrics_maybe_original_y,
    _set_seed,
    _split_base_anchor,
    _subsample,
)
from experiments.uci.diagnose_projection_matrices_std import (  # noqa: E402
    _feature_and_metric_diagnostics,
    _safe_float,
    _train_lmjgp_transductive_for_diag,
    _write_dict_csv,
)
from experiments.uci.uci_data import load_uci_split  # noqa: E402


DATASETS_DEFAULT = "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"


def _dataset_n(dataset: str, requested_n: int) -> int:
    if requested_n > 0:
        return int(requested_n)
    if dataset == "Appliances Energy Prediction":
        return 35
    return 25


def _moment_match_isotropic_samples(
    learned_samples: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Random row-normalized samples scaled to learned row norms per sample/anchor."""
    gen = torch.Generator(device=learned_samples.device)
    gen.manual_seed(int(seed))
    R = torch.randn(
        learned_samples.shape,
        dtype=learned_samples.dtype,
        device=learned_samples.device,
        generator=gen,
    )
    R = R / torch.linalg.norm(R, dim=-1, keepdim=True).clamp_min(1e-12)
    learned_row_norm = torch.linalg.norm(learned_samples, dim=-1, keepdim=True).clamp_min(1e-12)
    return R * learned_row_norm


def _prior_unit_samples(
    shape: tuple[int, int, int, int],
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    W = torch.randn(shape, device=device, generator=gen)
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(1e-12)


def _permuted_feature_samples(W_samples: torch.Tensor, *, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=W_samples.device)
    gen.manual_seed(int(seed))
    D = W_samples.shape[-1]
    perm = torch.randperm(D, generator=gen, device=W_samples.device)
    return W_samples[..., perm]


def _zero_mean_learned_sigma_samples(
    sigma_W: torch.Tensor,
    M: int,
    *,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device=sigma_W.device)
    gen.manual_seed(int(seed))
    eps = torch.randn((M,) + tuple(sigma_W.shape), device=sigma_W.device, generator=gen)
    W = eps * sigma_W.unsqueeze(0)
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(1e-12)


def _cem_supervised_bundle(
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
    MC_num: int,
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
        label="cem_em_supervised_controls",
    )
    lap_model = CoeffProjection(
        T=X_anchor_t.shape[0],
        Q=Q,
        D=X_anchor_t.shape[1],
        V=V_t,
        W0_init=res.W0.detach(),
        learn_W0=True,
        init_C_scale=0.0,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).to(device)
    with torch.no_grad():
        lap_model.W0.copy_(res.W0.detach())
        lap_model.C.copy_(res.C.detach())
    lap_cache = cem_step_all(
        lap_model,
        X_neighbors_list=X_anchor_nb_list,
        y_neighbors_list=y_anchor_nb_list,
        X_anchors=X_anchor_t,
        device=device,
        progress=False,
        desc="cem_laplace_cache",
    )
    lap_hess = diagonal_cem_data_hessian(
        C_map=res.C.detach(),
        W0=res.W0.detach(),
        V=V_t.detach(),
        X_neighbors_list=X_anchor_nb_list,
        y_neighbors_list=y_anchor_nb_list,
        X_anchors=X_anchor_t,
        y_anchors=y_anchor_t,
        cache=lap_cache,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
        lambda_gate=cfg.lambda_gate,
        lambda_anchor_pred=cfg.lambda_anchor_pred,
        anchor_pred_var_floor=cfg.anchor_pred_var_floor,
    )
    cond = gp_condition_coefficients(
        X_anchor_t,
        res.C,
        X_test_t,
        ridge=coeff_gp_ridge,
        lengthscale=coeff_gp_lengthscale,
    )
    C_mean = cond["mean"]
    W_mean = induced_W_from_coefficients(
        C_mean,
        res.W0,
        V_t,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).detach()
    C_samples = sample_induced_coefficients(
        C_mean,
        cond["var"],
        n_samples=MC_num,
        seed=seed,
    )
    W_samples = induced_W_from_coefficients(
        C_samples,
        res.W0,
        V_t,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).detach()

    # A deliberately simple variance-inflated empirical-Bayes control. This is
    # not Laplace; it probes whether more coefficient uncertainty helps UQ.
    C_samples_wide = sample_induced_coefficients(
        C_mean,
        cond["var"] * 3.0,
        n_samples=MC_num,
        seed=seed + 917,
    )
    W_samples_wide = induced_W_from_coefficients(
        C_samples_wide,
        res.W0,
        V_t,
        coeff_scale=coeff_scale,
        row_normalize=row_normalize_W,
    ).detach()
    cond_lap = gp_condition_coefficients_laplace_diag(
        X_anchor_t,
        res.C.detach(),
        X_test_t,
        lap_hess["data_hessian_diag"],
        ridge=coeff_gp_ridge,
        lengthscale=coeff_gp_lengthscale,
    )
    C_samples_lap = sample_induced_coefficients_diag(
        cond_lap["mean"],
        cond_lap["var"],
        n_samples=MC_num,
        seed=seed + 1231,
    )
    W_samples_lap = induced_W_from_coefficients(
        C_samples_lap,
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
        "W_mean": W_mean,
        "W_samples": W_samples,
        "W_samples_wide": W_samples_wide,
        "W_samples_laplace_diag": W_samples_lap,
        "X_nb_list": X_test_nb_list,
        "y_nb_list": y_test_nb_list,
        "train_time_sec": float(res.train_time_sec),
        "best_outer": int(res.best_outer),
        "coeff_gp_lengthscale": float(cond["lengthscale"]),
        "coeff_gp_ridge": float(cond["ridge"]),
        "mean_coeff_gp_var": float(cond["var"].detach().mean().cpu().item()),
        "laplace_mean_var": float(cond_lap["var"].detach().mean().cpu().item()),
        "laplace_extra_var_mean": float(cond_lap["extra_laplace_var_mean"]),
        "laplace_anchor_var_median": float(cond_lap["posterior_anchor_var_median"]),
        "laplace_hessian_failures": int(lap_hess["n_hessian_failures"]),
        "laplace_hessian_median": float(lap_hess["median_data_hessian_diag"]),
    }


def _evaluate_gaussian_row(
    *,
    dataset: str,
    seed: int,
    family: str,
    method: str,
    y_scaler,
    y_test_scaled: np.ndarray,
    mu_scaled: np.ndarray,
    sigma_scaled: np.ndarray,
    wall_sec: float,
    diag: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _pack_metrics_maybe_original_y(
        y_scaler,
        y_test_scaled,
        np.asarray(mu_scaled, dtype=np.float64),
        np.asarray(sigma_scaled, dtype=np.float64),
        float(wall_sec),
    )
    keys = ["rmse", "crps", "wall_sec", "cov90", "width90", "cov95", "width95"]
    row = {
        "dataset": dataset,
        "seed": seed,
        "family": family,
        "method": method,
        **{k: float(v) for k, v in zip(keys, metrics)},
    }
    if diag:
        row.update(diag)
    return row


def _evaluate_W_ensemble(
    *,
    W_samples: torch.Tensor,
    X_test_t: torch.Tensor,
    X_nb_list,
    y_nb_list,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    out = run_projected_jumpgp_ensemble(
        W_samples,
        X_test_t,
        X_nb_list,
        y_nb_list,
        device=device,
        progress=False,
        desc="projection_controls",
    )
    return out["mu"], out["sigma"], float(out["elapsed_sec"])


def _evaluate_W_map(
    *,
    W: torch.Tensor,
    X_test_t: torch.Tensor,
    X_nb_list,
    y_nb_list,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    out = run_projected_jumpgp(
        W,
        X_test_t,
        X_nb_list,
        y_nb_list,
        device=device,
        progress=False,
        desc="projection_controls_map",
    )
    return out["mu"], out["sigma"], float(out["elapsed_sec"])


def run_dataset_seed(args: argparse.Namespace, dataset: str, seed: int, device: torch.device) -> list[dict[str, Any]]:
    _set_seed(seed)
    X_train_full, X_test_full, y_train_full, y_test_full, K_default = load_uci_split(dataset, seed)
    X_train_full, y_train_full, _train_idx = _subsample(
        X_train_full,
        y_train_full,
        args.max_total_train,
        seed + 500,
    )
    X_test_sub, y_test_sub, _test_idx = _subsample(
        X_test_full,
        y_test_full,
        args.max_test_anchors,
        seed + 1000,
    )
    X_base, y_base, X_anchor, y_anchor, _split_info = _split_base_anchor(
        X_train_full,
        y_train_full,
        max_train_anchors=args.max_train_anchors,
        seed=seed + 2000,
    )

    if args.scaling_preset == "uci_empirical":
        standardize_x, standardize_y = SCALING_PRESETS.get(
            dataset, (bool(args.standardize_x), bool(args.standardize_y))
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

    if args.neighbor_pool == "base":
        X_pool_t, y_pool_t = X_base_t, y_base_t
    else:
        X_pool_t, y_pool_t = X_train_t, y_train_t

    rows: list[dict[str, Any]] = []

    print(f"[{dataset} seed={seed}] LMJGP controls", flush=True)
    lmjgp = _train_lmjgp_transductive_for_diag(
        X_test_t=X_test_t,
        X_pool_t=X_pool_t,
        y_pool_t=y_pool_t,
        X_inducing_t=X_train_t,
        Q=Q,
        m1=args.m1,
        m2=args.m2,
        n=n,
        steps=args.lmjgp_steps,
        lr=args.lr,
        device=device,
    )
    learned_W = sample_row_normalized_W(
        lmjgp["mu_W"], lmjgp["sigma_W"], args.MC_num, seed=seed + 10
    )
    W_controls = {
        "lmjgp_learned_qW": learned_W,
        "lmjgp_prior_unit_rows": _prior_unit_samples(
            tuple(learned_W.shape), device=device, seed=seed + 20
        ),
        "lmjgp_isotropic_moment_matched": _moment_match_isotropic_samples(
            learned_W, seed=seed + 30
        ),
        "lmjgp_permuted_features": _permuted_feature_samples(learned_W, seed=seed + 40),
        "lmjgp_zero_mean_learned_sigma": _zero_mean_learned_sigma_samples(
            lmjgp["sigma_W"], args.MC_num, seed=seed + 50
        ),
    }
    for method, W_samples in W_controls.items():
        t0 = time.perf_counter()
        mu, sig, pred_sec = _evaluate_W_ensemble(
            W_samples=W_samples,
            X_test_t=X_test_t,
            X_nb_list=lmjgp["X_nb_list"],
            y_nb_list=lmjgp["y_nb_list"],
            device=device,
        )
        diag = {
            "standardize_x": int(standardize_x),
            "standardize_y": int(standardize_y),
            "Q": Q,
            "n": n,
            "MC_num": args.MC_num,
            "train_time_sec": float(lmjgp["train_time_sec"]),
            "pred_time_sec": float(pred_sec),
            "total_eval_sec": float(time.perf_counter() - t0),
        }
        feat_diag = _feature_and_metric_diagnostics(W_samples)
        for key in ("feature_eff_count", "feature_top1_share", "metric_anchor_dispersion_rel"):
            diag[key] = _safe_float(feat_diag.get(key))
        rows.append(
            _evaluate_gaussian_row(
                dataset=dataset,
                seed=seed,
                family="lmjgp_projection_control",
                method=method,
                y_scaler=y_scaler,
                y_test_scaled=y_test_s,
                mu_scaled=mu,
                sigma_scaled=sig,
                wall_sec=pred_sec + float(lmjgp["train_time_sec"]),
                diag=diag,
            )
        )

    print(f"[{dataset} seed={seed}] CEM-EM supervised MAP/ensemble", flush=True)
    V_np, _V_info = build_V_basis(
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
    cem = _cem_supervised_bundle(
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
        m1=args.m1,
        n=n,
        cfg=cfg,
        init_C_scale=args.init_C_scale,
        coeff_scale=args.coeff_scale,
        row_normalize_W=bool(args.row_normalize_W),
        coeff_gp_ridge=args.coeff_gp_ridge,
        coeff_gp_lengthscale=args.coeff_gp_lengthscale,
        MC_num=args.MC_num,
        seed=seed + 60,
        device=device,
    )
    cem_variants = {
        "cem_supervised_map_meanC": ("map", cem["W_mean"]),
        "cem_supervised_gp_cond_ensemble": ("ensemble", cem["W_samples"]),
        "cem_supervised_gp_cond_ensemble_varx3": ("ensemble", cem["W_samples_wide"]),
        "cem_supervised_laplace_diag_ensemble": ("ensemble", cem["W_samples_laplace_diag"]),
    }
    for method, (kind, W_obj) in cem_variants.items():
        t0 = time.perf_counter()
        if kind == "map":
            mu, sig, pred_sec = _evaluate_W_map(
                W=W_obj,
                X_test_t=X_test_t,
                X_nb_list=cem["X_nb_list"],
                y_nb_list=cem["y_nb_list"],
                device=device,
            )
            W_for_diag = W_obj.unsqueeze(0)
        else:
            mu, sig, pred_sec = _evaluate_W_ensemble(
                W_samples=W_obj,
                X_test_t=X_test_t,
                X_nb_list=cem["X_nb_list"],
                y_nb_list=cem["y_nb_list"],
                device=device,
            )
            W_for_diag = W_obj
        diag = {
            "standardize_x": int(standardize_x),
            "standardize_y": int(standardize_y),
            "Q": Q,
            "n": n,
            "MC_num": args.MC_num if kind == "ensemble" else 1,
            "train_time_sec": float(cem["train_time_sec"]),
            "pred_time_sec": float(pred_sec),
            "total_eval_sec": float(time.perf_counter() - t0),
            "coeff_gp_lengthscale": float(cem["coeff_gp_lengthscale"]),
            "mean_coeff_gp_var": float(cem["mean_coeff_gp_var"]),
            "laplace_mean_var": float(cem["laplace_mean_var"]),
            "laplace_extra_var_mean": float(cem["laplace_extra_var_mean"]),
            "laplace_anchor_var_median": float(cem["laplace_anchor_var_median"]),
            "laplace_hessian_failures": int(cem["laplace_hessian_failures"]),
            "laplace_hessian_median": float(cem["laplace_hessian_median"]),
            "best_outer": int(cem["best_outer"]),
        }
        feat_diag = _feature_and_metric_diagnostics(W_for_diag, W_ref=W_pls_t)
        for key in ("feature_eff_count", "feature_top1_share", "metric_anchor_dispersion_rel"):
            diag[key] = _safe_float(feat_diag.get(key))
        rows.append(
            _evaluate_gaussian_row(
                dataset=dataset,
                seed=seed,
                family="cem_em_projection_uncertainty",
                method=method,
                y_scaler=y_scaler,
                y_test_scaled=y_test_s,
                mu_scaled=mu,
                sigma_scaled=sig,
                wall_sec=pred_sec + float(cem["train_time_sec"]),
                diag=diag,
            )
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["method"]), []).append(row)
    out = []
    metrics = ("rmse", "crps", "cov90", "width90", "cov95", "width95")
    for (dataset, method), vals in sorted(grouped.items()):
        row = {"dataset": dataset, "method": method, "n": len(vals)}
        for key in metrics:
            arr = np.asarray([_safe_float(v.get(key)) for v in vals], dtype=np.float64)
            row[f"{key}_mean"] = float(np.nanmean(arr))
            row[f"{key}_std"] = float(np.nanstd(arr))
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UCI projection uncertainty/triviality controls.")
    p.add_argument("--datasets", default=DATASETS_DEFAULT)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_total_train", type=int, default=500)
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
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default="experiments/uci/projection_uncertainty_controls_1seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.num_exp):
        for dataset in datasets:
            rows.extend(run_dataset_seed(args, dataset, seed, device))
    agg = _aggregate(rows)
    _write_dict_csv(out_dir / "metrics.csv", rows)
    _write_dict_csv(out_dir / "aggregate.csv", agg)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json({"args": vars(args), "rows": rows, "aggregate": agg}), f, indent=2)
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# UCI Projection Uncertainty Controls\n\n"
            "Generated by `experiments/uci/compare_projection_uncertainty_controls.py`.\n\n"
        "CEM-EM rows include empirical-Bayes GP-conditional coefficient samples,\n"
        "a variance-inflated stress test, and a diagonal-Laplace coefficient posterior.\n"
            "LMJGP controls reuse the same local neighborhoods and projected-JGP ensemble code.\n"
        )
    print(f"Wrote controls to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

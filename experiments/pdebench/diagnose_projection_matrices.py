r"""
Diagnose projection matrices on the processed PDEBench-Burgers surrogate.

The main PDEBench-Burgers comparison shows LMJGP has strong CRPS on the
near-threshold balanced shock-jump surrogate. This runner asks whether the
learned projection matrices themselves carry useful structure. It exports
projection diagnostics for:

* ``lmjgp_transductive``: the existing test-anchor LMJGP variational fit.
* ``lmjgp_supervised``: labeled train-anchor predictive q(V), induced to test anchors.
* ``cem_em_transductive``: CEM-EM coefficients fit directly at test anchors.
* ``cem_em_supervised``: CEM-EM coefficients fit at labeled train anchors and
  GP-conditioned to test anchors.
* ``pls_only`` and ``pca_train``: deterministic references.

How to run from the repository root:

    conda activate jumpGP
    cd C:\Users\yxu59\files\spring2026\park\DJGP

Smoke test:

    python experiments\pdebench\diagnose_projection_matrices.py ^
      --dataset-folder data\processed\pdebench_burgers_shockjump_balanced_delta01_n2000 ^
      --methods "lmjgp_transductive,cem_em_transductive,pls_only,pca_train" ^
      --max_test_anchors 8 --max_train_anchors 12 --Q 3 --n 12 ^
      --lmjgp_steps 10 --n_outer 1 --n_m_steps 5 --mc_diag 4 ^
      --out_dir experiments\pdebench\projection_matrix_diag_smoke

Main seed-0 diagnostic matching the current PDEBench-Burgers Q/n setting:

    python experiments\pdebench\diagnose_projection_matrices.py ^
      --dataset-folder data\processed\pdebench_burgers_shockjump_balanced_delta01_n2000 ^
      --num_exp 1 --max_test_anchors 128 --max_train_anchors 128 ^
      --Q 3 --n 25 --lmjgp_steps 300 --n_outer 5 --n_m_steps 80 ^
      --mc_diag 16 --out_dir experiments\pdebench\projection_matrix_diag_balanced_q3n25

Agents: update this docstring when CLI flags or defaults change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics.projection_posterior import (  # noqa: E402
    row_normalize_w_predict_vi,
    sample_row_normalized_W,
    sanitize_for_json,
)
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    build_V_basis,
    compute_pls_W0,
)
from experiments.pdebench.compare_pdebench_burgers import (  # noqa: E402
    _as_numpy,
    _dataset_arrays,
    _threshold_masks,
    load_dataset,
)
from experiments.uci.compare_inductive_projection import _split_base_anchor  # noqa: E402
from experiments.uci.diagnose_projection_matrices_std import (  # noqa: E402
    _aggregate_rows,
    _feature_and_metric_diagnostics,
    _lmjgp_full_diag_summary,
    _nanmedian,
    _response_geometry_diagnostics,
    _safe_float,
    _train_cem_supervised_for_diag,
    _train_cem_transductive_for_diag,
    _train_lmjgp_supervised_for_diag,
    _train_lmjgp_transductive_for_diag,
    _write_dict_csv,
)


METHODS_DEFAULT = (
    "lmjgp_transductive,lmjgp_supervised,cem_em_transductive,"
    "cem_em_supervised,pls_only,pca_train"
)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_test_meta_array(
    dataset: dict[str, Any],
    key: str,
    *,
    n_test_full: int,
    n_used: int,
) -> np.ndarray | None:
    meta = dataset.get("args", {}) or {}
    if key not in meta:
        return None
    values = np.asarray(meta[key], dtype=np.float64).reshape(-1)
    test_indices = meta.get("test_indices")
    if test_indices is not None:
        idx = np.asarray(test_indices, dtype=int).reshape(-1)
        if idx.size and len(values) >= int(np.max(idx)) + 1:
            values = values[idx]
    elif len(values) not in {n_test_full, n_used}:
        return None
    return values[:n_used].astype(float)


def _spatial_weight_diagnostics(W_samples: torch.Tensor, *, top_k: int = 5) -> dict[str, Any]:
    if W_samples.dim() == 3:
        W_samples = W_samples.unsqueeze(0)
    W = W_samples.detach().float().cpu()
    D = int(W.shape[-1])
    A_diag = torch.einsum("mtqd,mtqd->mtqd", W, W).mean(dim=(0, 1, 2)).numpy()
    A_diag = np.maximum(A_diag, 0.0)
    total = float(A_diag.sum())
    if total <= 1e-12:
        return {
            "spatial_weight_com": float("nan"),
            "spatial_weight_std": float("nan"),
            "spatial_mass_0p4_0p6": float("nan"),
            "spatial_top5_idx": "",
            "spatial_top5_pos": "",
            "spatial_top5_share": float("nan"),
        }
    p = A_diag / total
    grid = np.linspace(0.0, 1.0, D)
    com = float(np.sum(p * grid))
    std = float(np.sqrt(np.sum(p * (grid - com) ** 2)))
    band = (grid >= 0.4) & (grid <= 0.6)
    top_idx = np.argsort(p)[-min(top_k, D) :][::-1]
    return {
        "spatial_weight_com": com,
        "spatial_weight_std": std,
        "spatial_mass_0p4_0p6": float(np.sum(p[band])),
        "spatial_top5_idx": ";".join(str(int(i)) for i in top_idx),
        "spatial_top5_pos": ";".join(f"{grid[i]:.4f}" for i in top_idx),
        "spatial_top5_share": float(np.sum(p[top_idx])),
    }


def _add_group_summaries(
    row: dict[str, Any],
    response_rows: list[dict[str, float]],
    masks: dict[str, Any],
) -> None:
    if not response_rows:
        return
    for split_name in ("near_threshold", "far_from_threshold"):
        if split_name not in masks:
            continue
        mask = np.asarray(masks[split_name], dtype=bool).reshape(-1)
        vals_pair, vals_knn = [], []
        for i, anchor_row in enumerate(response_rows):
            if i < len(mask) and mask[i]:
                vals_pair.append(_safe_float(anchor_row.get("proj_minus_raw_pair_y_abs_corr")))
                vals_knn.append(_safe_float(anchor_row.get("proj_minus_raw_anchor_knn_abs_y")))
        row[f"{split_name}_n"] = int(np.sum(mask))
        row[f"{split_name}_proj_minus_raw_pair_y_abs_corr_median"] = _nanmedian(vals_pair)
        row[f"{split_name}_proj_minus_raw_anchor_knn_abs_y_median"] = _nanmedian(vals_knn)


def _deterministic_pls_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_t: torch.Tensor,
    *,
    Q: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    W_pls = compute_pls_W0(X_train, y_train, Q=Q)
    W_pls_t = torch.from_numpy(W_pls).float().to(device)
    W_norm = row_normalize_w_predict_vi(W_pls_t.unsqueeze(0)).squeeze(0)
    W_samples = W_norm.unsqueeze(0).unsqueeze(0).expand(1, X_test_t.shape[0], -1, -1).contiguous()
    return W_samples, W_norm


def _deterministic_pca_samples(
    X_train: np.ndarray,
    X_test_t: torch.Tensor,
    *,
    Q: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    pca = PCA(n_components=Q, random_state=seed)
    pca.fit(np.asarray(X_train, dtype=np.float64))
    W = torch.from_numpy(np.asarray(pca.components_, dtype=np.float32)).to(device)
    W = row_normalize_w_predict_vi(W.unsqueeze(0)).squeeze(0)
    return W.unsqueeze(0).unsqueeze(0).expand(1, X_test_t.shape[0], -1, -1).contiguous()


def _build_test_neighborhoods(
    X_test_t: torch.Tensor,
    X_pool_t: torch.Tensor,
    y_pool_t: torch.Tensor,
    *,
    Q: int,
    m1: int,
    n: int,
    device: torch.device,
):
    from experiments.uci.compare_inductive_projection import _build_regions

    _regions, X_nb_list, y_nb_list = _build_regions(
        X_test_t,
        None,
        X_pool_t,
        y_pool_t,
        device=device,
        m1=m1,
        Q=Q,
        n=n,
    )
    return X_nb_list, y_nb_list


def run_one_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    methods: list[str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _seed_all(seed)
    dataset = load_dataset(args.dataset_folder)
    X_train_full, y_train_full, X_test_full, y_test_full = _dataset_arrays(dataset, max_anchors=None)
    X_train, y_train, X_test, y_test = _dataset_arrays(dataset, max_anchors=args.max_test_anchors)
    masks = _threshold_masks(dataset, len(X_test_full), len(X_test))
    shock_test = _extract_test_meta_array(
        dataset, "shock_location", n_test_full=len(X_test_full), n_used=len(X_test)
    )
    threshold_distance = _extract_test_meta_array(
        dataset, "threshold_distance", n_test_full=len(X_test_full), n_used=len(X_test)
    )

    X_base, y_base, X_anchor, y_anchor, split_info = _split_base_anchor(
        X_train_full,
        y_train_full,
        max_train_anchors=args.max_train_anchors,
        seed=seed + 303,
    )

    Q = int(args.Q if args.Q is not None else min(3, X_train.shape[1]))
    m1 = int(args.m1)
    m2 = int(args.m2 if args.m2 is not None else min(40, max(8, len(X_train_full))))
    n = min(int(args.n), len(X_train_full))

    X_train_t = torch.from_numpy(X_train_full).float().to(device)
    y_train_t = torch.from_numpy(y_train_full).float().to(device)
    X_base_t = torch.from_numpy(X_base).float().to(device)
    y_base_t = torch.from_numpy(y_base).float().to(device)
    X_anchor_t = torch.from_numpy(X_anchor).float().to(device)
    y_anchor_t = torch.from_numpy(y_anchor).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    if args.neighbor_pool == "base":
        X_pool_t, y_pool_t = X_base_t, y_base_t
    else:
        X_pool_t, y_pool_t = X_train_t, y_train_t

    V_np, V_info = build_V_basis(
        X_train_full,
        y_train_full,
        n_pls=args.n_pls,
        n_pca=args.n_pca,
        n_random=args.n_random,
        seed=seed,
    )
    W_pls_np = compute_pls_W0(X_train_full, y_train_full, Q=Q)
    V_t = torch.from_numpy(V_np).float().to(device)
    W_pls_t = row_normalize_w_predict_vi(
        torch.from_numpy(W_pls_np).float().to(device).unsqueeze(0)
    ).squeeze(0)

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
        snapshot_after_each_outer=False,
        cem_progress=False,
        verbose=bool(args.cem_verbose),
    )

    summary_rows: list[dict[str, Any]] = []
    per_anchor_rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {
        "dataset_folder": str(args.dataset_folder),
        "seed": seed,
        "settings": {
            "Q": Q,
            "n": n,
            "m1": m1,
            "m2": m2,
            "n_train_full": int(len(X_train_full)),
            "n_test_used": int(len(X_test)),
            "split_counts": {
                k: int(np.asarray(v, dtype=bool).sum())
                for k, v in masks.items()
                if k in {"all", "near_threshold", "far_from_threshold"}
            },
            "split_info": split_info,
            "V_info": V_info,
            "neighbor_pool": args.neighbor_pool,
        },
        "methods": {},
    }

    for method in methods:
        print(f"[PDEBench projection diag seed={seed}] {method}", flush=True)
        t0 = time.perf_counter()
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

        elif method == "pls_only":
            W_samples, _W_ref = _deterministic_pls_samples(
                X_train_full,
                y_train_full,
                X_test_t,
                Q=Q,
                device=device,
            )
            X_nb_list, y_nb_list = _build_test_neighborhoods(
                X_test_t, X_pool_t, y_pool_t, Q=Q, m1=m1, n=n, device=device
            )
            train_time = 0.0

        elif method == "pca_train":
            W_samples = _deterministic_pca_samples(
                X_train_full,
                X_test_t,
                Q=Q,
                seed=seed,
                device=device,
            )
            X_nb_list, y_nb_list = _build_test_neighborhoods(
                X_test_t, X_pool_t, y_pool_t, Q=Q, m1=m1, n=n, device=device
            )
            train_time = 0.0

        else:
            raise ValueError(f"Unknown method={method!r}")

        feature_summary = _feature_and_metric_diagnostics(W_samples, W_ref=W_pls_t)
        spatial_summary = _spatial_weight_diagnostics(W_samples)
        response_summary, response_rows = _response_geometry_diagnostics(
            W_samples,
            X_test_t,
            y_test_t,
            X_nb_list,
            y_nb_list,
            k_response=args.k_response,
        )

        row = {
            "dataset": Path(args.dataset_folder).name,
            "dataset_folder": str(args.dataset_folder),
            "seed": seed,
            "method": method,
            "Q": Q,
            "n": n,
            "n_train": int(len(X_train_full)),
            "n_test": int(len(X_test)),
            "train_time_sec": _safe_float(train_time),
            "elapsed_sec": time.perf_counter() - t0,
            **feature_summary,
            **spatial_summary,
            **response_summary,
            **extra_summary,
        }
        _add_group_summaries(row, response_rows, masks)
        summary_rows.append(row)

        for anchor_row in response_rows:
            anchor_idx = int(anchor_row["anchor"])
            flat = {
                "dataset": Path(args.dataset_folder).name,
                "dataset_folder": str(args.dataset_folder),
                "seed": seed,
                "method": method,
                **anchor_row,
            }
            if "near_threshold" in masks and anchor_idx < len(masks["near_threshold"]):
                flat["near_threshold"] = int(bool(np.asarray(masks["near_threshold"])[anchor_idx]))
            if threshold_distance is not None and anchor_idx < len(threshold_distance):
                flat["threshold_distance"] = float(threshold_distance[anchor_idx])
            if shock_test is not None and anchor_idx < len(shock_test):
                flat["shock_location"] = float(shock_test[anchor_idx])
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
        description="Projection-matrix diagnostics for processed PDEBench-Burgers data."
    )
    p.add_argument(
        "--dataset-folder",
        default="data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000",
    )
    p.add_argument("--methods", default=METHODS_DEFAULT)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_test_anchors", type=int, default=128)
    p.add_argument("--max_train_anchors", type=int, default=128)
    p.add_argument("--neighbor_pool", choices=("train", "base"), default="train")
    p.add_argument("--Q", type=int, default=3)
    p.add_argument("--n", "--neighbors", dest="n", type=int, default=25)
    p.add_argument("--m1", type=int, default=4)
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
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--out_dir",
        default="experiments/pdebench/projection_matrix_diag_balanced_q3n25",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = set(METHODS_DEFAULT.split(","))
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; allowed={sorted(allowed)}")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    per_anchor_rows: list[dict[str, Any]] = []
    nested_runs: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.num_exp):
        rows, anchor_rows, nested = run_one_seed(
            args, seed=seed, methods=methods, device=device
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
            "# PDEBench-Burgers Projection-Matrix Diagnostics\n\n"
            "Generated by `experiments/pdebench/diagnose_projection_matrices.py`.\n\n"
            "Files:\n"
            "- `summary.csv`: one row per seed/method.\n"
            "- `aggregate.csv`: mean/std over seeds for numeric diagnostics.\n"
            "- `per_anchor.csv`: per-test-anchor response-geometry diagnostics, including near-threshold flags.\n"
            "- `summary.json`: nested run metadata and LMJGP posterior diagnostic rows.\n\n"
            "Positive `proj_minus_raw_pair_y_abs_corr_median` means projected pairwise distances align\n"
            "with response differences better than raw-X distances inside local neighborhoods. Negative\n"
            "`proj_minus_raw_anchor_knn_abs_y_median` means projected nearest neighbors are response-closer\n"
            "to the anchor than raw-X nearest neighbors.\n"
        )
    print(f"Wrote diagnostics to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

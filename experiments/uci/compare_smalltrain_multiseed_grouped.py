"""
Run the tuned 200-train UCI comparison over multiple seeds.

This is the current small-train "main" runner for the UCI diagnostic setting:

* Wine Quality: historical random-row split, empirical preset ``std_x``.
* Appliances Energy Prediction: historical random-row split, empirical preset
  ``std_x,std_y``.
* Parkinsons Telemonitoring: subject-grouped split, raw features, no subject
  overlap between train and test.

Methods:

* ``xgb_bootstrap``;
* ``dkl_svgp``;
* ``lmjgp_transductive`` without residuals;
* ``lmjgp_transductive_residual_ridge`` with a ridge global mean;
* ``lmjgp_supervised`` without residuals;
* ``lmjgp_supervised_residual_ridge`` with a ridge global mean.

How to run (repository root, conda env: ``jumpGP``):

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke test.
    python experiments/uci/compare_smalltrain_multiseed_grouped.py ^
        --num_exp 1 --datasets "Wine Quality" --max_total_train 80 ^
        --max_test_anchors 20 --max_train_anchors 32 --lmjgp_steps 5 ^
        --MC_num 2 --xgb_boots 3 --dkl_epochs 3 ^
        --out_dir experiments/uci/smalltrain_multiseed_grouped_smoke

    # Main 5-seed run.
    python experiments/uci/compare_smalltrain_multiseed_grouped.py ^
        --num_exp 5 ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 ^
        --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 ^
        --out_dir experiments/uci/smalltrain_multiseed_grouped_5seeds

    # Larger train budget; same tuned method settings.
    python experiments/uci/compare_smalltrain_multiseed_grouped.py ^
        --num_exp 5 --max_total_train 500 --max_test_anchors 200 ^
        --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 ^
        --lmjgp_steps 300 --MC_num 3 ^
        --out_dir experiments/uci/smalltrain_multiseed_grouped_train500_5seeds

    # Parkinsons grouped scaling probe; override only the Parkinsons preset.
    python experiments/uci/compare_smalltrain_multiseed_grouped.py ^
        --num_exp 1 --datasets "Parkinsons Telemonitoring" ^
        --methods "lmjgp_transductive,lmjgp_transductive_residual_ridge,lmjgp_supervised,lmjgp_supervised_residual_ridge" ^
        --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 ^
        --parkinsons_standardize_x 1 --parkinsons_standardize_y 1 ^
        --lmjgp_steps 300 --MC_num 3 ^
        --out_dir experiments/uci/parkinsons_grouped_scaling_probe_stdxy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.baselines import run_dkl_svgp_regression  # noqa: E402
from djgp.diagnostics.projection_posterior import sanitize_for_json  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from experiments.uci.compare_inductive_projection import (  # noqa: E402
    SCALING_PRESETS,
    _apply_scaling,
    _pack_metrics_maybe_original_y,
    _run_lmjgp_supervised,
    _run_lmjgp_transductive,
    _run_xgboost_bootstrap,
    _set_seed,
    _split_base_anchor,
    _subsample,
)
from experiments.uci.compare_parkinsons_grouped_split import _load_parkinsons_grouped_split  # noqa: E402
from experiments.uci.uci_data import load_uci_split  # noqa: E402


DEFAULT_METHODS = (
    "xgb_bootstrap,dkl_svgp,"
    "lmjgp_transductive,lmjgp_transductive_residual_ridge,"
    "lmjgp_supervised,lmjgp_supervised_residual_ridge"
)

TUNED_PRESETS = {
    "Wine Quality": {"n": 25, "standardize_x": True, "standardize_y": False, "split": "random_row"},
    "Parkinsons Telemonitoring": {"n": 25, "standardize_x": False, "standardize_y": False, "split": "subject_grouped"},
    "Appliances Energy Prediction": {"n": 35, "standardize_x": True, "standardize_y": True, "split": "random_row"},
}


def _metric_dict(values: list[float]) -> dict[str, float]:
    return {key: float(values[i]) for i, key in enumerate(RESULT_ROW_KEYS)}


def _to_original_y(
    y_scaler,
    y_scaled: np.ndarray,
    mu_scaled: np.ndarray,
    sigma_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_scaled = np.asarray(y_scaled, dtype=np.float64).reshape(-1)
    mu_scaled = np.asarray(mu_scaled, dtype=np.float64).reshape(-1)
    sigma_scaled = np.asarray(sigma_scaled, dtype=np.float64).reshape(-1)
    if y_scaler is None:
        return y_scaled, mu_scaled, sigma_scaled
    scale = float(np.squeeze(y_scaler.scale_))
    if not np.isfinite(scale) or scale < 1e-20:
        scale = 1.0
    y_orig = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
    mu_orig = y_scaler.inverse_transform(mu_scaled.reshape(-1, 1)).reshape(-1)
    sig_orig = sigma_scaled * abs(scale)
    return y_orig, mu_orig, sig_orig


def _pack_metrics_with_numeric_filter(
    *,
    y_scaler,
    y_eval_scaled: np.ndarray,
    y_train_original: np.ndarray,
    mu_scaled: np.ndarray,
    sigma_scaled: np.ndarray,
    wall_sec: float,
    method: str,
    sigma_omit_factor: float,
    max_omit_frac: float,
) -> tuple[list[float], list[float], dict]:
    """Return selected metrics, raw metrics, and numeric-filter diagnostics."""
    raw_metrics = _pack_metrics_maybe_original_y(y_scaler, y_eval_scaled, mu_scaled, sigma_scaled, wall_sec)
    y_orig, mu_orig, sig_orig = _to_original_y(y_scaler, y_eval_scaled, mu_scaled, sigma_scaled)
    y_train_original = np.asarray(y_train_original, dtype=np.float64).reshape(-1)
    y_std = float(np.nanstd(y_train_original))
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = float(max(np.nanstd(y_orig), 1.0))
    sigma_threshold = max(float(sigma_omit_factor) * y_std, 1e-8)

    finite = np.isfinite(y_orig) & np.isfinite(mu_orig) & np.isfinite(sig_orig)
    nonnegative = sig_orig >= 0.0
    not_exploded = sig_orig <= sigma_threshold
    good = finite & nonnegative & not_exploded
    n_total = int(y_orig.size)
    n_omit = int(n_total - int(np.sum(good)))
    omit_frac = float(n_omit / max(n_total, 1))

    diag = {
        "metric_policy": "raw",
        "n_eval_total": n_total,
        "n_eval_used": n_total,
        "n_numeric_omitted": n_omit,
        "numeric_omit_frac": omit_frac,
        "sigma_omit_factor": float(sigma_omit_factor),
        "sigma_omit_threshold_original_y": float(sigma_threshold),
        "max_sigma_original_y": float(np.nanmax(sig_orig)) if sig_orig.size else float("nan"),
        "numeric_filter_applied": False,
        "numeric_warning": "",
    }

    if n_omit == 0:
        return raw_metrics, raw_metrics, diag

    if omit_frac <= float(max_omit_frac) and int(np.sum(good)) >= 2:
        cleaned_metrics = _pack_metrics_maybe_original_y(
            y_scaler,
            np.asarray(y_eval_scaled).reshape(-1)[good],
            np.asarray(mu_scaled).reshape(-1)[good],
            np.asarray(sigma_scaled).reshape(-1)[good],
            wall_sec,
        )
        diag.update(
            {
                "metric_policy": "cleaned_omit_numeric_outliers",
                "n_eval_used": int(np.sum(good)),
                "numeric_filter_applied": True,
            }
        )
        return cleaned_metrics, raw_metrics, diag

    diag.update(
        {
            "metric_policy": "raw_numeric_warning_too_many_outliers",
            "numeric_warning": (
                f"{method} had {n_omit}/{n_total} numeric outlier predictions; "
                "kept raw metrics because omit fraction exceeded threshold."
            ),
        }
    )
    return raw_metrics, raw_metrics, diag


def _write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _load_dataset_for_seed(
    dataset_name: str,
    seed: int,
    *,
    max_total_train: int,
    max_test_anchors: int,
    train_subject_frac: float,
) -> dict:
    preset = TUNED_PRESETS[dataset_name]
    if preset["split"] == "subject_grouped":
        split = _load_parkinsons_grouped_split(seed=seed, train_subject_frac=train_subject_frac)
        X_train_full, y_train_full, train_idx = _subsample(
            split["X_train"],
            split["y_train"],
            max_total_train,
            seed + 500,
        )
        X_test_sub, y_test_sub, test_idx = _subsample(
            split["X_test"],
            split["y_test"],
            max_test_anchors,
            seed + 1000,
        )
        subj_train = split["subject_train"][train_idx]
        subj_test = split["subject_test"][test_idx]
        subject_overlap = np.intersect1d(np.unique(subj_train), np.unique(subj_test))
        meta = {
            **split["split_meta"],
            "split_type": "subject_grouped",
            "n_train_pool_rows": int(split["X_train"].shape[0]),
            "n_test_pool_rows": int(split["X_test"].shape[0]),
            "subsample_train_subject_count": int(len(np.unique(subj_train))),
            "subsample_test_subject_count": int(len(np.unique(subj_test))),
            "subsample_subject_overlap_count": int(len(subject_overlap)),
            "train_indices_within_group_pool": train_idx.tolist(),
            "test_indices_within_group_pool": test_idx.tolist(),
        }
        K = split["K"]
    else:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw, K = load_uci_split(dataset_name, seed)
        X_train_full, y_train_full, train_idx = _subsample(
            X_train_raw,
            y_train_raw,
            max_total_train,
            seed + 500,
        )
        X_test_sub, y_test_sub, test_idx = _subsample(
            X_test_raw,
            y_test_raw,
            max_test_anchors,
            seed + 1000,
        )
        meta = {
            "split_type": "random_row",
            "n_train_pool_rows": int(X_train_raw.shape[0]),
            "n_test_pool_rows": int(X_test_raw.shape[0]),
            "train_indices_within_pool": train_idx.tolist(),
            "test_indices_within_pool": test_idx.tolist(),
        }
    return {
        "X_train": X_train_full,
        "y_train": y_train_full,
        "X_test": X_test_sub,
        "y_test": y_test_sub,
        "K": int(K),
        "meta": meta,
    }


def _run_dkl(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    rmse, crps, elapsed, mu, sigma = run_dkl_svgp_regression(
        X_train,
        y_train,
        X_test,
        y_test,
        device=device,
        epochs=epochs,
        seed=seed,
    )
    return mu, sigma, float(elapsed), {"scaled_rmse": float(rmse), "scaled_crps": float(crps), "epochs": int(epochs)}


def _fit_ridge_residual_context(
    *,
    X_train_s: np.ndarray,
    y_train_s: np.ndarray,
    X_base_s: np.ndarray,
    y_base_s: np.ndarray,
    X_anchor_s: np.ndarray,
    y_anchor_s: np.ndarray,
    X_test_s: np.ndarray,
    X_pool_t: torch.Tensor,
    X_train_t: torch.Tensor,
    X_base_t: torch.Tensor,
    ridge_alpha: float,
    device: torch.device,
) -> dict:
    """Fit a training-only ridge mean in scaled-y space and return residual tensors."""
    t0 = time.perf_counter()
    ridge = Ridge(alpha=float(ridge_alpha), random_state=0)
    ridge.fit(np.asarray(X_train_s, dtype=np.float64), np.asarray(y_train_s, dtype=np.float64).reshape(-1))
    elapsed = time.perf_counter() - t0

    train_mean = ridge.predict(X_train_s).reshape(-1)
    base_mean = ridge.predict(X_base_s).reshape(-1)
    anchor_mean = ridge.predict(X_anchor_s).reshape(-1)
    test_mean = ridge.predict(X_test_s).reshape(-1)

    if X_pool_t.shape[0] == X_train_t.shape[0]:
        y_pool_resid = y_train_s - train_mean
        pool_name = "train"
    elif X_pool_t.shape[0] == X_base_t.shape[0]:
        y_pool_resid = y_base_s - base_mean
        pool_name = "base"
    else:
        raise ValueError("X_pool_t must align with either train or base pool for residual methods.")

    return {
        "ridge_alpha": float(ridge_alpha),
        "ridge_fit_time_sec": float(elapsed),
        "ridge_pool": pool_name,
        "y_train_resid_t": torch.from_numpy(y_train_s - train_mean).float().to(device),
        "y_base_resid_t": torch.from_numpy(y_base_s - base_mean).float().to(device),
        "y_anchor_resid_t": torch.from_numpy(y_anchor_s - anchor_mean).float().to(device),
        "y_pool_resid_t": torch.from_numpy(y_pool_resid).float().to(device),
        "test_mean_scaled": test_mean,
    }


def _aggregate(metric_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in metric_rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    out: list[dict] = []
    for (dataset, method), rows in sorted(groups.items()):
        agg = {"dataset": dataset, "method": method, "n_seeds": len(rows)}
        for key in RESULT_ROW_KEYS:
            vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _write_readme(out_dir: Path, args: argparse.Namespace, agg_rows: list[dict]) -> None:
    lines = [
        "# Small-Train Multiseed Grouped UCI",
        "",
        "This folder was produced by `experiments/uci/compare_smalltrain_multiseed_grouped.py`.",
        "",
        "Parkinsons uses subject-grouped splits; Wine and Appliances use the historical random-row split.",
        "Aggregate metrics use `metrics.csv`, which may omit isolated numeric outlier predictions according to the runner policy; raw unfiltered metrics are in `metrics_raw.csv`.",
        "",
        "## Tuned Presets",
        "",
        "| dataset | split | std_x | std_y | n | steps |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, preset in TUNED_PRESETS.items():
        lines.append(
            f"| {dataset} | {preset['split']} | {preset['standardize_x']} | "
            f"{preset['standardize_y']} | {preset['n']} | {args.lmjgp_steps} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| dataset | method | seeds | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in agg_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['n_seeds']} | "
            f"{row['rmse_mean']:.4f} | {row['rmse_std']:.4f} | "
            f"{row['mean_crps_mean']:.4f} | {row['mean_crps_std']:.4f} | "
            f"{row['coverage_90_mean']:.3f} | {row['mean_width_90_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Command",
            "",
            "```powershell",
            "python experiments/uci/compare_smalltrain_multiseed_grouped.py "
            f"--num_exp {args.num_exp} --datasets \"{args.datasets}\" "
            f"--max_total_train {args.max_total_train} --max_test_anchors {args.max_test_anchors} "
            f"--max_train_anchors {args.max_train_anchors} --xgb_boots {args.xgb_boots} "
            f"--dkl_epochs {args.dkl_epochs} --lmjgp_steps {args.lmjgp_steps} "
            f"--MC_num {args.MC_num} "
            f"--parkinsons_standardize_x {args.parkinsons_standardize_x} "
            f"--parkinsons_standardize_y {args.parkinsons_standardize_y} "
            f"--out_dir {args.out_dir}",
            "```",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tuned 200-train UCI multiseed comparison with grouped Parkinsons.")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument(
        "--datasets",
        type=str,
        default="Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction",
    )
    p.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    p.add_argument("--max_total_train", type=int, default=200)
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--max_train_anchors", type=int, default=128)
    p.add_argument("--train_subject_frac", type=float, default=0.8)
    p.add_argument("--neighbor_pool", type=str, default="train", choices=("base", "train"))
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--lmjgp_supervised_train_mc", type=int, default=2)
    p.add_argument("--lmjgp_supervised_kl_weight", type=float, default=1e-3)
    p.add_argument("--lmjgp_supervised_pred_var_floor", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--dkl_epochs", type=int, default=80)
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument(
        "--parkinsons_standardize_x",
        type=int,
        choices=(-1, 0, 1),
        default=-1,
        help="Override Parkinsons standardize_x: -1 keeps the tuned preset, 0 disables, 1 enables.",
    )
    p.add_argument(
        "--parkinsons_standardize_y",
        type=int,
        choices=(-1, 0, 1),
        default=-1,
        help="Override Parkinsons standardize_y: -1 keeps the tuned preset, 0 disables, 1 enables.",
    )
    p.add_argument(
        "--numeric_sigma_omit_factor",
        type=float,
        default=50.0,
        help="Omit predictions with sigma above this multiple of train-y std if omit fraction is small.",
    )
    p.add_argument(
        "--max_numeric_omit_frac",
        type=float,
        default=0.02,
        help="Maximum fraction of numeric outlier predictions to omit from selected metrics.",
    )
    p.add_argument("--out_dir", type=str, default="experiments/uci/smalltrain_multiseed_grouped_5seeds")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    if args.parkinsons_standardize_x >= 0:
        TUNED_PRESETS["Parkinsons Telemonitoring"]["standardize_x"] = bool(args.parkinsons_standardize_x)
    if args.parkinsons_standardize_y >= 0:
        TUNED_PRESETS["Parkinsons Telemonitoring"]["standardize_y"] = bool(args.parkinsons_standardize_y)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = set(DEFAULT_METHODS.split(","))
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")
    dataset_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    for dataset in dataset_list:
        if dataset not in TUNED_PRESETS:
            raise ValueError(f"Unknown or untuned dataset {dataset!r}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}", flush=True)
    print(f"writing artifacts to {out_dir}", flush=True)

    summary: dict = {
        "_meta": sanitize_for_json(vars(args)),
        "presets": sanitize_for_json(TUNED_PRESETS),
        "datasets": {},
    }
    metric_rows: list[dict] = []
    raw_metric_rows: list[dict] = []

    for dataset_name in dataset_list:
        preset = TUNED_PRESETS[dataset_name]
        summary["datasets"][dataset_name] = {}
        for seed in range(args.num_exp):
            _set_seed(seed)
            print(f"\n{'=' * 72}\n{dataset_name} | seed={seed}\n{'=' * 72}", flush=True)
            data = _load_dataset_for_seed(
                dataset_name,
                seed,
                max_total_train=args.max_total_train,
                max_test_anchors=args.max_test_anchors,
                train_subject_frac=args.train_subject_frac,
            )
            X_base, y_base, X_anchor, y_anchor, split_info = _split_base_anchor(
                data["X_train"],
                data["y_train"],
                max_train_anchors=args.max_train_anchors,
                seed=seed + 2000,
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
                _,
                y_scaler,
            ) = _apply_scaling(
                data["X_train"],
                X_base,
                X_anchor,
                data["X_test"],
                data["y_train"],
                y_base,
                y_anchor,
                data["y_test"],
                standardize_x=bool(preset["standardize_x"]),
                standardize_y=bool(preset["standardize_y"]),
            )

            Q = int(data["K"])
            n = int(preset["n"])
            X_train_t = torch.from_numpy(X_train_s).float().to(device)
            y_train_t = torch.from_numpy(y_train_s).float().to(device)
            X_base_t = torch.from_numpy(X_base_s).float().to(device)
            y_base_t = torch.from_numpy(y_base_s).float().to(device)
            X_anchor_t = torch.from_numpy(X_anchor_s).float().to(device)
            y_anchor_t = torch.from_numpy(y_anchor_s).float().to(device)
            X_test_t = torch.from_numpy(X_test_s).float().to(device)
            if args.neighbor_pool == "train":
                X_pool_t, y_pool_t = X_train_t, y_train_t
            else:
                X_pool_t, y_pool_t = X_base_t, y_base_t
            ridge_ctx = None
            if any("residual_ridge" in method for method in methods):
                ridge_ctx = _fit_ridge_residual_context(
                    X_train_s=X_train_s,
                    y_train_s=y_train_s,
                    X_base_s=X_base_s,
                    y_base_s=y_base_s,
                    X_anchor_s=X_anchor_s,
                    y_anchor_s=y_anchor_s,
                    X_test_s=X_test_s,
                    X_pool_t=X_pool_t,
                    X_train_t=X_train_t,
                    X_base_t=X_base_t,
                    ridge_alpha=args.ridge_alpha,
                    device=device,
                )

            seed_block = {
                "seed": int(seed),
                "Q": Q,
                "D": int(X_train_s.shape[1]),
                "n": n,
                "split": {
                    **data["meta"],
                    **split_info,
                    "n_train_rows": int(X_train_s.shape[0]),
                    "n_test_rows": int(X_test_s.shape[0]),
                },
                "scaling": {
                    "standardize_x": bool(preset["standardize_x"]),
                    "standardize_y": bool(preset["standardize_y"]),
                },
                "metrics": {},
                "diagnostics": {},
            }

            method_fns = {
                "xgb_bootstrap": lambda: _run_xgboost_bootstrap(
                    X_train=X_train_s,
                    y_train=y_train_s,
                    X_test=X_test_s,
                    y_test=y_test_s,
                    n_bootstrap=args.xgb_boots,
                    seed=seed,
                ),
                "dkl_svgp": lambda: _run_dkl(
                    X_train=X_train_s,
                    y_train=y_train_s,
                    X_test=X_test_s,
                    y_test=y_test_s,
                    epochs=args.dkl_epochs,
                    seed=seed,
                    device=device,
                ),
                "lmjgp_transductive": lambda: _run_lmjgp_transductive(
                    X_test_t=X_test_t,
                    y_test_np=y_test_s,
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
                    device=device,
                ),
                "lmjgp_transductive_residual_ridge": lambda: (
                    lambda out: (
                        out[0] + ridge_ctx["test_mean_scaled"],
                        out[1],
                        out[2],
                        {
                            **out[3],
                            "variant": "ridge_global_mean_plus_lmjgp_transductive_residual",
                            "ridge_alpha": ridge_ctx["ridge_alpha"],
                            "ridge_fit_time_sec": ridge_ctx["ridge_fit_time_sec"],
                            "ridge_pool": ridge_ctx["ridge_pool"],
                        },
                    )
                )(
                    _run_lmjgp_transductive(
                        X_test_t=X_test_t,
                        y_test_np=y_test_s - ridge_ctx["test_mean_scaled"],
                        X_pool_t=X_pool_t,
                        y_pool_t=ridge_ctx["y_pool_resid_t"],
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
                ),
                "lmjgp_supervised": lambda: _run_lmjgp_supervised(
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
                    grad_check=False,
                    device=device,
                ),
                "lmjgp_supervised_residual_ridge": lambda: (
                    lambda out: (
                        out[0] + ridge_ctx["test_mean_scaled"],
                        out[1],
                        out[2],
                        {
                            **out[3],
                            "variant": "ridge_global_mean_plus_lmjgp_supervised_residual",
                            "ridge_alpha": ridge_ctx["ridge_alpha"],
                            "ridge_fit_time_sec": ridge_ctx["ridge_fit_time_sec"],
                            "ridge_pool": ridge_ctx["ridge_pool"],
                        },
                    )
                )(
                    _run_lmjgp_supervised(
                        X_anchor_t=X_anchor_t,
                        y_anchor_t=ridge_ctx["y_anchor_resid_t"],
                        X_test_t=X_test_t,
                        y_test_np=y_test_s - ridge_ctx["test_mean_scaled"],
                        X_base_t=X_base_t,
                        y_base_t=ridge_ctx["y_base_resid_t"],
                        X_pool_t=X_pool_t,
                        y_pool_t=ridge_ctx["y_pool_resid_t"],
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
                        grad_check=False,
                        device=device,
                    )
                ),
            }

            for method in methods:
                print(f"[{dataset_name} seed={seed}] running {method}", flush=True)
                t0 = time.perf_counter()
                try:
                    mu, sig, pred_wall, diag = method_fns[method]()
                    total_wall = time.perf_counter() - t0
                    metrics, raw_metrics, metric_diag = _pack_metrics_with_numeric_filter(
                        y_scaler=y_scaler,
                        y_eval_scaled=y_test_s,
                        y_train_original=data["y_train"],
                        mu_scaled=mu,
                        sigma_scaled=sig,
                        wall_sec=pred_wall,
                        method=method,
                        sigma_omit_factor=args.numeric_sigma_omit_factor,
                        max_omit_frac=args.max_numeric_omit_frac,
                    )
                    row = {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "method": method,
                        "split_type": preset["split"],
                        "n": n,
                        "metric_policy": metric_diag["metric_policy"],
                        "n_eval_used": metric_diag["n_eval_used"],
                        "n_numeric_omitted": metric_diag["n_numeric_omitted"],
                    }
                    row.update(_metric_dict(metrics))
                    metric_rows.append(row)
                    raw_row = {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "method": method,
                        "split_type": preset["split"],
                        "n": n,
                        "metric_policy": "raw",
                        "n_eval_used": metric_diag["n_eval_total"],
                        "n_numeric_omitted": metric_diag["n_numeric_omitted"],
                    }
                    raw_row.update(_metric_dict(raw_metrics))
                    raw_metric_rows.append(raw_row)
                    seed_block["metrics"][method] = row
                    diag = dict(diag)
                    diag["total_wall_sec"] = float(total_wall)
                    diag["metric_filter"] = metric_diag
                    seed_block["diagnostics"][method] = sanitize_for_json(diag)
                    print(
                        f"  {method}: rmse={row['rmse']:.4f} crps={row['mean_crps']:.4f} "
                        f"cov90={row['coverage_90']:.3f} policy={row['metric_policy']} "
                        f"omitted={row['n_numeric_omitted']} total={total_wall:.1f}s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    total_wall = time.perf_counter() - t0
                    row = {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "method": method,
                        "split_type": preset["split"],
                        "n": n,
                        "metric_policy": "error",
                        "n_eval_used": 0,
                        "n_numeric_omitted": 0,
                        **{key: float("nan") for key in RESULT_ROW_KEYS},
                    }
                    metric_rows.append(row)
                    raw_metric_rows.append(dict(row, metric_policy="error_raw"))
                    seed_block["metrics"][method] = row
                    seed_block["diagnostics"][method] = {
                        "error": str(exc),
                        "total_wall_sec": float(total_wall),
                    }
                    print(f"  [error] {method}: {exc}", flush=True)

            summary["datasets"][dataset_name][str(seed)] = seed_block
            agg_rows = _aggregate(metric_rows)
            _write_csv(
                out_dir / "metrics.csv",
                metric_rows,
                [
                    "dataset",
                    "seed",
                    "method",
                    "split_type",
                    "n",
                    "metric_policy",
                    "n_eval_used",
                    "n_numeric_omitted",
                    *RESULT_ROW_KEYS,
                ],
            )
            _write_csv(
                out_dir / "metrics_raw.csv",
                raw_metric_rows,
                [
                    "dataset",
                    "seed",
                    "method",
                    "split_type",
                    "n",
                    "metric_policy",
                    "n_eval_used",
                    "n_numeric_omitted",
                    *RESULT_ROW_KEYS,
                ],
            )
            _write_csv(
                out_dir / "aggregate.csv",
                agg_rows,
                [
                    "dataset",
                    "method",
                    "n_seeds",
                    *[f"{key}_{stat}" for key in RESULT_ROW_KEYS for stat in ("mean", "std")],
                ],
            )
            with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(sanitize_for_json(summary), f, indent=2)
            _write_readme(out_dir, args, agg_rows)
            print(f"partial artifacts saved to {out_dir}", flush=True)

    agg_rows = _aggregate(metric_rows)
    print(f"\nSaved artifacts -> {out_dir}", flush=True)
    return {"summary": summary, "aggregate": agg_rows}


if __name__ == "__main__":
    main()

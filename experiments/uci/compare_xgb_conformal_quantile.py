"""Run conformalized quantile XGBoost on the UCI small-train presets.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Main lightweight run: train=200/500/1000, five seeds.
    python experiments/uci/compare_xgb_conformal_quantile.py ^
        --num_exp 5 --train_sizes 200,500,1000 ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --out_dir experiments/uci/xgb_conformal_quantile_smalltrain_5seeds

    # Smoke test.
    python experiments/uci/compare_xgb_conformal_quantile.py ^
        --num_exp 1 --train_sizes 80 --datasets "Wine Quality" ^
        --max_test_anchors 20 --n_estimators 5 ^
        --out_dir experiments/uci/xgb_conformal_quantile_smoke

This runner implements split conformalized quantile regression (CQR). It uses a
proper-training split to fit lower/upper XGBoost quantile models, a calibration
split to compute conformal corrections, and reports direct conformal interval
coverage/width. The point RMSE uses a separate squared-error XGBoost model fit
on all available training rows. CQR intervals are not Gaussian predictive
posteriors, so ``mean_crps`` is intentionally reported as NaN.
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
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from experiments.uci.compare_smalltrain_multiseed_grouped import (  # noqa: E402
    TUNED_PRESETS,
    _load_dataset_for_seed,
)


def _write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _fit_scalers(dataset: str, X_fit: np.ndarray, y_fit: np.ndarray):
    preset = TUNED_PRESETS[dataset]
    x_scaler = StandardScaler() if bool(preset["standardize_x"]) else None
    y_scaler = StandardScaler() if bool(preset["standardize_y"]) else None
    if x_scaler is not None:
        x_scaler.fit(X_fit)
    if y_scaler is not None:
        y_scaler.fit(np.asarray(y_fit).reshape(-1, 1))
    return x_scaler, y_scaler


def _transform_x(x_scaler, X: np.ndarray) -> np.ndarray:
    if x_scaler is None:
        return np.asarray(X, dtype=np.float64)
    return x_scaler.transform(X)


def _transform_y(y_scaler, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_scaler is None:
        return y
    return y_scaler.transform(y.reshape(-1, 1)).reshape(-1)


def _inverse_y(y_scaler, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_scaler is None:
        return y
    return y_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)


def _xgb_params(args: argparse.Namespace, *, objective: str, seed: int, quantile_alpha: float | None = None) -> dict:
    params = {
        "objective": objective,
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_lambda": float(args.reg_lambda),
        "random_state": int(seed),
        "n_jobs": int(args.n_jobs),
        "tree_method": "hist",
    }
    if quantile_alpha is not None:
        params["quantile_alpha"] = float(quantile_alpha)
    return params


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    n = int(scores.size)
    if n == 0:
        return 0.0
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(float(q_level), 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def _interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    interval_score = width + (2.0 / alpha) * (below + above)
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(width)),
        "interval_score": float(np.mean(interval_score)),
    }


def _run_cqr(
    dataset: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    rng = np.random.default_rng(seed + 7000)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    n = X_train.shape[0]
    n_cal = max(int(np.ceil(float(args.cal_frac) * n)), 1)
    n_cal = min(n_cal, n - 5)
    if n_cal < 1:
        raise ValueError("Need at least 6 training rows for split conformal.")
    perm = rng.permutation(n)
    cal_idx = np.sort(perm[:n_cal])
    fit_idx = np.sort(perm[n_cal:])

    X_fit = X_train[fit_idx]
    y_fit = y_train[fit_idx]
    X_cal = X_train[cal_idx]
    y_cal = y_train[cal_idx]

    # CQR models use proper-training-only preprocessing.
    x_scaler, y_scaler = _fit_scalers(dataset, X_fit, y_fit)
    X_fit_s = _transform_x(x_scaler, X_fit)
    X_cal_s = _transform_x(x_scaler, X_cal)
    X_test_s = _transform_x(x_scaler, X_test)
    y_fit_s = _transform_y(y_scaler, y_fit)
    y_cal_s = _transform_y(y_scaler, y_cal)

    # Point model uses all training rows for a fair XGB point-prediction RMSE.
    x_mean_scaler, y_mean_scaler = _fit_scalers(dataset, X_train, y_train)
    X_train_mean_s = _transform_x(x_mean_scaler, X_train)
    X_test_mean_s = _transform_x(x_mean_scaler, X_test)
    y_train_mean_s = _transform_y(y_mean_scaler, y_train)

    t0 = time.perf_counter()
    mean_model = xgb.XGBRegressor(
        **_xgb_params(args, objective="reg:squarederror", seed=seed),
    )
    mean_model.fit(X_train_mean_s, y_train_mean_s)
    mu_s = mean_model.predict(X_test_mean_s)
    mu = _inverse_y(y_mean_scaler, mu_s)

    interval_outputs: dict[str, dict] = {}
    for level in (0.90, 0.95):
        alpha = 1.0 - level
        lower_alpha = alpha / 2.0
        upper_alpha = 1.0 - alpha / 2.0
        lower_model = xgb.XGBRegressor(
            **_xgb_params(args, objective="reg:quantileerror", seed=seed + int(level * 1000), quantile_alpha=lower_alpha),
        )
        upper_model = xgb.XGBRegressor(
            **_xgb_params(args, objective="reg:quantileerror", seed=seed + int(level * 1000) + 1, quantile_alpha=upper_alpha),
        )
        lower_model.fit(X_fit_s, y_fit_s)
        upper_model.fit(X_fit_s, y_fit_s)

        lo_cal = lower_model.predict(X_cal_s)
        hi_cal = upper_model.predict(X_cal_s)
        lo_cal, hi_cal = np.minimum(lo_cal, hi_cal), np.maximum(lo_cal, hi_cal)
        scores = np.maximum(lo_cal - y_cal_s, y_cal_s - hi_cal)
        qhat = _finite_sample_quantile(scores, alpha)

        lo_test_s = lower_model.predict(X_test_s)
        hi_test_s = upper_model.predict(X_test_s)
        lo_test_s, hi_test_s = np.minimum(lo_test_s, hi_test_s), np.maximum(lo_test_s, hi_test_s)
        lo = _inverse_y(y_scaler, lo_test_s - qhat)
        hi = _inverse_y(y_scaler, hi_test_s + qhat)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        metrics = _interval_metrics(y_test, lo, hi, alpha)
        tag = str(int(round(level * 100)))
        interval_outputs[tag] = {
            "coverage": metrics["coverage"],
            "mean_width": metrics["mean_width"],
            "interval_score": metrics["interval_score"],
            "qhat": float(qhat),
            "cal_coverage_before": float(np.mean((y_cal_s >= lo_cal) & (y_cal_s <= hi_cal))),
            "cal_score_mean": float(np.mean(scores)),
            "cal_score_max": float(np.max(scores)),
        }
    wall = time.perf_counter() - t0
    rmse = float(np.sqrt(np.mean((mu - y_test) ** 2)))
    return {
        "rmse": rmse,
        "mean_crps": float("nan"),
        "wall_sec": float(wall),
        "coverage_90": interval_outputs["90"]["coverage"],
        "mean_width_90": interval_outputs["90"]["mean_width"],
        "coverage_95": interval_outputs["95"]["coverage"],
        "mean_width_95": interval_outputs["95"]["mean_width"],
        "interval_score_90": interval_outputs["90"]["interval_score"],
        "interval_score_95": interval_outputs["95"]["interval_score"],
        "qhat_90": interval_outputs["90"]["qhat"],
        "qhat_95": interval_outputs["95"]["qhat"],
        "cal_coverage_before_90": interval_outputs["90"]["cal_coverage_before"],
        "cal_coverage_before_95": interval_outputs["95"]["cal_coverage_before"],
        "cal_score_mean_90": interval_outputs["90"]["cal_score_mean"],
        "cal_score_mean_95": interval_outputs["95"]["cal_score_mean"],
        "n_fit": int(fit_idx.size),
        "n_cal": int(cal_idx.size),
        "x_scaler_fit": "proper_train" if x_scaler is not None else "none",
        "y_scaler_fit": "proper_train" if y_scaler is not None else "none",
        "mean_model_train_rows": int(n),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["train_size"]), row["dataset"]), []).append(row)
    keys = [
        "rmse",
        "coverage_90",
        "mean_width_90",
        "interval_score_90",
        "coverage_95",
        "mean_width_95",
        "interval_score_95",
        "qhat_90",
        "qhat_95",
        "wall_sec",
    ]
    out: list[dict] = []
    for (train_size, dataset), group in sorted(groups.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        agg = {
            "train_size": int(train_size),
            "dataset": dataset,
            "method": "xgb_cqr",
            "n_seeds": len(group),
        }
        for key in keys:
            vals = np.asarray([float(r[key]) for r in group], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conformalized quantile XGBoost on UCI small-train presets.")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--train_sizes", type=str, default="200,500,1000")
    p.add_argument(
        "--datasets",
        type=str,
        default="Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction",
    )
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--train_subject_frac", type=float, default=0.8)
    p.add_argument("--cal_frac", type=float, default=0.2)
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.9)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--out_dir", type=str, default="experiments/uci/xgb_conformal_quantile_smalltrain_5seeds")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    train_sizes = [int(x.strip()) for x in args.train_sizes.split(",") if x.strip()]
    rows: list[dict] = []

    for train_size in train_sizes:
        for dataset in datasets:
            preset = TUNED_PRESETS[dataset]
            for seed in range(args.num_exp):
                print(f"train={train_size} dataset={dataset} seed={seed}", flush=True)
                data = _load_dataset_for_seed(
                    dataset,
                    seed,
                    max_total_train=train_size,
                    max_test_anchors=args.max_test_anchors,
                    train_subject_frac=args.train_subject_frac,
                )
                metrics = _run_cqr(
                    dataset,
                    data["X_train"],
                    data["y_train"],
                    data["X_test"],
                    data["y_test"],
                    seed=seed,
                    args=args,
                )
                row = {
                    "train_size": int(train_size),
                    "dataset": dataset,
                    "seed": int(seed),
                    "method": "xgb_cqr",
                    "split_type": preset["split"],
                    "standardize_x": bool(preset["standardize_x"]),
                    "standardize_y": bool(preset["standardize_y"]),
                    "cal_frac": float(args.cal_frac),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"  rmse={row['rmse']:.4f} cov90={row['coverage_90']:.3f} "
                    f"w90={row['mean_width_90']:.3f} cov95={row['coverage_95']:.3f}",
                    flush=True,
                )
                _write_csv(
                    out_dir / "metrics.csv",
                    rows,
                    [
                        "train_size",
                        "dataset",
                        "seed",
                        "method",
                        "split_type",
                        "standardize_x",
                        "standardize_y",
                        "cal_frac",
                        "rmse",
                        "mean_crps",
                        "wall_sec",
                        "coverage_90",
                        "mean_width_90",
                        "coverage_95",
                        "mean_width_95",
                        "interval_score_90",
                        "interval_score_95",
                        "qhat_90",
                        "qhat_95",
                        "cal_coverage_before_90",
                        "cal_coverage_before_95",
                        "cal_score_mean_90",
                        "cal_score_mean_95",
                        "n_fit",
                        "n_cal",
                        "x_scaler_fit",
                        "y_scaler_fit",
                        "mean_model_train_rows",
                    ],
                )
                _write_csv(
                    out_dir / "aggregate.csv",
                    _aggregate(rows),
                    [
                        "train_size",
                        "dataset",
                        "method",
                        "n_seeds",
                        "rmse_mean",
                        "rmse_std",
                        "coverage_90_mean",
                        "coverage_90_std",
                        "mean_width_90_mean",
                        "mean_width_90_std",
                        "interval_score_90_mean",
                        "interval_score_90_std",
                        "coverage_95_mean",
                        "coverage_95_std",
                        "mean_width_95_mean",
                        "mean_width_95_std",
                        "interval_score_95_mean",
                        "interval_score_95_std",
                        "qhat_90_mean",
                        "qhat_90_std",
                        "qhat_95_mean",
                        "qhat_95_std",
                        "wall_sec_mean",
                        "wall_sec_std",
                    ],
                )

    summary = {
        "args": vars(args),
        "preset_policy": {
            "Wine Quality": "std_x, random-row",
            "Parkinsons Telemonitoring": "raw X, subject-grouped",
            "Appliances Energy Prediction": "std_x,std_y, random-row",
        },
        "note": "CQR intervals are direct conformal intervals, not Gaussian predictive posteriors; mean_crps is NaN by design.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Conformalized Quantile XGBoost\n\n"
        "Split conformalized quantile XGBoost on the current UCI small-train presets. "
        "Use `aggregate.csv` for train-size summaries and `metrics.csv` for per-seed rows. "
        "The conformal interval metrics are direct interval metrics; CRPS is not reported.\n",
        encoding="utf-8",
    )
    print(f"Saved artifacts to {out_dir}", flush=True)
    return {"metrics": rows, "aggregate": _aggregate(rows)}


if __name__ == "__main__":
    main()

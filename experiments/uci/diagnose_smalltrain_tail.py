"""
Diagnose small-train UCI residual tails and calibration.

This script consumes the prediction file produced by
``experiments/uci/visualize_smalltrain_pca_predictions.py``. It compares
XGBoost with the best supervised-LMJGP configuration per dataset and writes
tail-contribution, true-y quantile, predicted-sigma calibration, and KNN
high-response support diagnostics.

How to run (repository root, conda env: ``jumpGP``):

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/uci/diagnose_smalltrain_tail.py ^
        --viz_dir experiments/uci/smalltrain_pca_viz ^
        --out_dir experiments/uci/smalltrain_tail_diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation import mean_gaussian_crps_torch  # noqa: E402
from experiments.uci.compare_inductive_projection import (  # noqa: E402
    SCALING_PRESETS,
    _apply_scaling,
    _subsample,
)
from experiments.uci.uci_data import load_uci_split  # noqa: E402


Z90 = 1.6448536269514722


def _rmse(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(mu)) ** 2)))


def _crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    return float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(np.maximum(sigma, 1e-8), dtype=torch.float32),
        ).item()
    )


def _coverage_width(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    lo = mu - Z90 * sigma
    hi = mu + Z90 * sigma
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def _select_methods(metrics: pd.DataFrame) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for dataset, g in metrics.groupby("dataset"):
        lm = g[g["method"] == "lmjgp_supervised"].copy()
        if lm.empty:
            continue
        best = lm.loc[lm["rmse"].astype(float).idxmin()]
        selected[dataset] = {
            "lmjgp_n": str(int(float(best["n"]))),
            "lmjgp_steps": str(int(float(best["steps"]))),
            "lmjgp_rmse": float(best["rmse"]),
            "lmjgp_crps": float(best["mean_crps"]),
        }
    return selected


def _method_predictions(preds: pd.DataFrame, dataset: str, method: str, n: str = "", steps: str = "") -> pd.DataFrame:
    g = preds[(preds["dataset"] == dataset) & (preds["method"] == method)].copy()
    if method == "lmjgp_supervised":
        g = g[(g["n"].astype(str) == n) & (g["steps"].astype(str) == steps)].copy()
    return g.sort_values("test_order")


def _tail_rows(dataset: str, method: str, g: pd.DataFrame) -> list[dict]:
    y = g["y_true"].to_numpy(float)
    mu = g["mu"].to_numpy(float)
    e2 = (y - mu) ** 2
    total = float(np.sum(e2))
    rows = []
    for frac in (0.05, 0.10, 0.20):
        k = max(1, int(np.ceil(frac * len(e2))))
        idx = np.argsort(e2)[-k:]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "top_abs_error_frac": frac,
                "n_points": int(k),
                "sse_share": float(np.sum(e2[idx]) / total) if total > 0 else float("nan"),
                "rmse_all": _rmse(y, mu),
                "rmse_top": float(np.sqrt(np.mean(e2[idx]))),
                "mean_y_top": float(np.mean(y[idx])),
                "mean_abs_error_top": float(np.mean(np.sqrt(e2[idx]))),
            }
        )
    return rows


def _quantile_rows(dataset: str, method: str, g: pd.DataFrame) -> list[dict]:
    y = g["y_true"].to_numpy(float)
    mu = g["mu"].to_numpy(float)
    sigma = g["sigma"].to_numpy(float)
    q50, q90 = np.quantile(y, [0.5, 0.9])
    groups = [
        ("y<=q50", y <= q50),
        ("q50<y<=q90", (y > q50) & (y <= q90)),
        ("y>q90", y > q90),
    ]
    rows = []
    for name, mask in groups:
        cov, width = _coverage_width(y[mask], mu[mask], sigma[mask])
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "group": name,
                "n": int(np.sum(mask)),
                "y_min": float(np.min(y[mask])),
                "y_max": float(np.max(y[mask])),
                "rmse": _rmse(y[mask], mu[mask]),
                "crps": _crps(y[mask], mu[mask], sigma[mask]),
                "coverage_90": cov,
                "width_90": width,
                "mean_abs_error": float(np.mean(np.abs(y[mask] - mu[mask]))),
                "mean_sigma": float(np.mean(sigma[mask])),
            }
        )
    return rows


def _sigma_bin_rows(dataset: str, method: str, g: pd.DataFrame, bins: int = 5) -> list[dict]:
    y = g["y_true"].to_numpy(float)
    mu = g["mu"].to_numpy(float)
    sigma = g["sigma"].to_numpy(float)
    ranks = pd.qcut(pd.Series(sigma).rank(method="first"), q=bins, labels=False)
    rows = []
    for b in range(bins):
        mask = ranks.to_numpy() == b
        cov, width = _coverage_width(y[mask], mu[mask], sigma[mask])
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "sigma_bin": int(b + 1),
                "n": int(np.sum(mask)),
                "sigma_min": float(np.min(sigma[mask])),
                "sigma_max": float(np.max(sigma[mask])),
                "mean_sigma": float(np.mean(sigma[mask])),
                "mean_abs_error": float(np.mean(np.abs(y[mask] - mu[mask]))),
                "rmse": _rmse(y[mask], mu[mask]),
                "coverage_90": cov,
                "width_90": width,
            }
        )
    return rows


def _largest_error_rows(dataset: str, method: str, g: pd.DataFrame, top_k: int = 10) -> list[dict]:
    y = g["y_true"].to_numpy(float)
    mu = g["mu"].to_numpy(float)
    sigma = g["sigma"].to_numpy(float)
    abs_err = np.abs(y - mu)
    order = np.argsort(abs_err)[::-1][:top_k]
    rows = []
    for rank, i in enumerate(order, start=1):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "rank": rank,
                "test_order": int(g.iloc[i]["test_order"]),
                "test_index": int(g.iloc[i]["test_index"]),
                "pc1": float(g.iloc[i]["pc1"]),
                "y_true": float(y[i]),
                "mu": float(mu[i]),
                "sigma": float(sigma[i]),
                "abs_error": float(abs_err[i]),
                "covered_90": bool((y[i] >= mu[i] - Z90 * sigma[i]) and (y[i] <= mu[i] + Z90 * sigma[i])),
            }
        )
    return rows


def _plot_diagnostics(dataset: str, xgb: pd.DataFrame, lm: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = dataset.lower().replace(" ", "_")
    methods = [("xgboost", xgb, "#E45756"), ("lmjgp_supervised", lm, "#2A9D8F")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=160)
    for label, g, color in methods:
        y = g["y_true"].to_numpy(float)
        mu = g["mu"].to_numpy(float)
        sigma = g["sigma"].to_numpy(float)
        axes[0].scatter(y, mu, s=16, alpha=0.55, label=label, color=color)
        axes[1].scatter(y, y - mu, s=16, alpha=0.55, label=label, color=color)
        axes[2].scatter(sigma, np.abs(y - mu), s=16, alpha=0.55, label=label, color=color)

    y_all = np.concatenate([xgb["y_true"].to_numpy(float), lm["y_true"].to_numpy(float)])
    axes[0].plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], color="#333333", linewidth=1)
    axes[0].set_xlabel("true y")
    axes[0].set_ylabel("predicted mean")
    axes[0].set_title("predicted vs true")
    axes[1].axhline(0.0, color="#333333", linewidth=1)
    axes[1].set_xlabel("true y")
    axes[1].set_ylabel("residual y - mean")
    axes[1].set_title("residual vs true y")
    axes[2].set_xlabel("predicted sigma")
    axes[2].set_ylabel("|residual|")
    axes[2].set_title("absolute error vs sigma")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(dataset)
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_residual_diagnostics.png")
    plt.close(fig)


def _neighbor_support_rows(
    *,
    dataset: str,
    seed: int,
    max_total_train: int,
    max_test_anchors: int,
    standardize_x: bool,
    standardize_y: bool,
    neighbor_ns: list[int],
    preds_by_method: dict[str, pd.DataFrame],
) -> list[dict]:
    X_train_full, X_test_full, y_train_full, y_test_full, _ = load_uci_split(dataset, seed)
    X_train_full, y_train_full, _ = _subsample(X_train_full, y_train_full, max_total_train, seed + 500)
    X_test_sub, y_test_sub, _ = _subsample(X_test_full, y_test_full, max_test_anchors, seed + 1000)
    X_base = X_train_full
    y_base = np.asarray(y_train_full).reshape(-1)
    # We only need scaled X. y scaling is irrelevant for high-y support counts.
    _, X_base_s, _, X_test_s, _, _, _, _, _, _ = _apply_scaling(
        X_train_full,
        X_base,
        X_base[:1],
        X_test_sub,
        y_train_full,
        y_base,
        y_base[:1],
        y_test_sub,
        standardize_x=standardize_x,
        standardize_y=standardize_y,
    )
    d2 = ((X_test_s[:, None, :] - X_base_s[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(d2, axis=1)
    q90_train = float(np.quantile(y_base, 0.9))
    y_test = np.asarray(y_test_sub).reshape(-1)
    high_test = y_test > np.quantile(y_test, 0.9)
    rows = []
    for n in neighbor_ns:
        nn = order[:, :n]
        high_counts = (y_base[nn] > q90_train).sum(axis=1)
        for method, pred in preds_by_method.items():
            pred = pred.sort_values("test_order")
            abs_err = np.abs(pred["y_true"].to_numpy(float) - pred["mu"].to_numpy(float))
            corr = np.corrcoef(high_counts, abs_err)[0, 1] if np.std(high_counts) > 0 else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "neighbors": int(n),
                    "q90_train_y": q90_train,
                    "n_high_test": int(np.sum(high_test)),
                    "mean_high_train_neighbors_all": float(np.mean(high_counts)),
                    "mean_high_train_neighbors_high_test": float(np.mean(high_counts[high_test])),
                    "median_high_train_neighbors_high_test": float(np.median(high_counts[high_test])),
                    "frac_high_test_zero_high_neighbors": float(np.mean(high_counts[high_test] == 0)),
                    "corr_high_neighbor_count_abs_error": float(corr) if np.isfinite(corr) else "",
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose residual tails from smalltrain PC1 predictions.")
    p.add_argument("--viz_dir", type=str, default="experiments/uci/smalltrain_pca_viz")
    p.add_argument("--out_dir", type=str, default="experiments/uci/smalltrain_tail_diagnostics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_total_train", type=int, default=200)
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--scaling_preset", type=str, default="uci_empirical", choices=("global", "uci_empirical"))
    p.add_argument("--standardize_x", type=int, default=1)
    p.add_argument("--standardize_y", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    viz_dir = Path(args.viz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(viz_dir / "predictions.csv", dtype={"n": str, "steps": str})
    metrics = pd.read_csv(viz_dir / "metrics.csv", dtype={"n": str, "steps": str})
    selected = _select_methods(metrics)

    tail_rows: list[dict] = []
    quantile_rows: list[dict] = []
    sigma_rows: list[dict] = []
    largest_rows: list[dict] = []
    support_rows: list[dict] = []
    summary = {"_meta": vars(args), "selected_lmjgp": selected, "datasets": {}}

    for dataset, cfg in selected.items():
        xgb = _method_predictions(preds, dataset, "xgboost")
        lm = _method_predictions(preds, dataset, "lmjgp_supervised", cfg["lmjgp_n"], cfg["lmjgp_steps"])
        method_frames = {"xgboost": xgb, f"lmjgp_n{cfg['lmjgp_n']}_s{cfg['lmjgp_steps']}": lm}
        for method_name, frame in method_frames.items():
            tail_rows.extend(_tail_rows(dataset, method_name, frame))
            quantile_rows.extend(_quantile_rows(dataset, method_name, frame))
            sigma_rows.extend(_sigma_bin_rows(dataset, method_name, frame))
            largest_rows.extend(_largest_error_rows(dataset, method_name, frame))

        _plot_diagnostics(dataset, xgb, lm, fig_dir)

        if args.scaling_preset == "uci_empirical":
            standardize_x, standardize_y = SCALING_PRESETS.get(
                dataset,
                (bool(args.standardize_x), bool(args.standardize_y)),
            )
        else:
            standardize_x, standardize_y = bool(args.standardize_x), bool(args.standardize_y)
        support_rows.extend(
            _neighbor_support_rows(
                dataset=dataset,
                seed=int(args.seed),
                max_total_train=int(args.max_total_train),
                max_test_anchors=int(args.max_test_anchors),
                standardize_x=standardize_x,
                standardize_y=standardize_y,
                neighbor_ns=[25, 35],
                preds_by_method=method_frames,
            )
        )
        summary["datasets"][dataset] = {
            "xgboost_rmse": _rmse(xgb["y_true"].to_numpy(float), xgb["mu"].to_numpy(float)),
            "lmjgp_best_n": cfg["lmjgp_n"],
            "lmjgp_best_steps": cfg["lmjgp_steps"],
            "lmjgp_rmse": _rmse(lm["y_true"].to_numpy(float), lm["mu"].to_numpy(float)),
        }

    _write_csv(out_dir / "tail_contribution.csv", tail_rows)
    _write_csv(out_dir / "true_y_quantile_metrics.csv", quantile_rows)
    _write_csv(out_dir / "sigma_bin_calibration.csv", sigma_rows)
    _write_csv(out_dir / "largest_errors.csv", largest_rows)
    _write_csv(out_dir / "neighbor_high_y_support.csv", support_rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved diagnostics to {out_dir}")


if __name__ == "__main__":
    main()

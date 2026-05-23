"""
Visualize small-train UCI predictions along the first principal component.

This script compares XGBoost bootstrap with supervised LMJGP on bounded UCI
train/test splits. It fits PCA on the scaled training features, plots PC1
against the response, plots test predictions with 90% predictive ranges, and
records a small LMJGP step/neighborhood sweep.

How to run (repository root, conda env: ``jumpGP``):

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke test on Wine only.
    python experiments/uci/visualize_smalltrain_pca_predictions.py ^
        --datasets "Wine Quality" --max_total_train 80 --max_test_anchors 20 ^
        --max_train_anchors 32 --neighbors "10" --steps "2" --xgb_boots 3

    # Main 200-train visualization and LMJGP step/neighborhood sweep.
    python experiments/uci/visualize_smalltrain_pca_predictions.py ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 ^
        --neighbors "25,35" --steps "200,300,500" --xgb_boots 30 ^
        --scaling_preset uci_empirical --out_dir experiments/uci/smalltrain_pca_viz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics.projection_posterior import sanitize_for_json  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from experiments.uci.compare_inductive_projection import (  # noqa: E402
    SCALING_PRESETS,
    _apply_scaling,
    _pack_metrics_maybe_original_y,
    _run_lmjgp_supervised,
    _run_xgboost_bootstrap,
    _set_seed,
    _split_base_anchor,
    _subsample,
)
from experiments.uci.uci_data import load_uci_split  # noqa: E402


Z90 = 1.6448536269514722


def _parse_int_list(text: str) -> list[int]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError(f"Expected at least one integer in {text!r}.")
    return vals


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _as_metric_dict(values: list[float]) -> dict[str, float]:
    return {key: float(values[i]) for i, key in enumerate(RESULT_ROW_KEYS)}


def _invert_y(
    y_scaler,
    y_scaled: np.ndarray,
    sigma_scaled: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
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


def _plot_response_pc1(
    *,
    dataset: str,
    pc1_train: np.ndarray,
    y_train: np.ndarray,
    pc1_test: np.ndarray,
    y_test: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.scatter(pc1_train, y_train, s=12, alpha=0.35, label="train", color="#4C78A8")
    ax.scatter(pc1_test, y_test, s=18, alpha=0.75, label="test", color="#F58518")
    ax.set_xlabel("PC1")
    ax.set_ylabel("response")
    ax.set_title(f"{dataset}: PC1 vs response")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_predictions_pc1(
    *,
    dataset: str,
    pc1_test: np.ndarray,
    y_test: np.ndarray,
    xgb: dict,
    lmjgp: dict,
    out_path: Path,
) -> None:
    order = np.argsort(pc1_test)
    x = pc1_test[order]
    y = y_test[order]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
    ax.scatter(x, y, s=18, alpha=0.75, color="#222222", label="test y")

    for pred, color, label in [
        (xgb, "#E45756", "xgboost"),
        (lmjgp, "#2A9D8F", f"lmjgp_supervised n={lmjgp['n']} steps={lmjgp['steps']}"),
    ]:
        mu = pred["mu"][order]
        sigma = pred["sigma"][order]
        ax.plot(x, mu, linewidth=1.8, color=color, label=f"{label} mean")
        ax.fill_between(
            x,
            mu - Z90 * sigma,
            mu + Z90 * sigma,
            color=color,
            alpha=0.16,
            linewidth=0,
            label=f"{label} 90% range",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("response")
    ax.set_title(f"{dataset}: test predictions along PC1")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "method", "n", "steps"] + list(RESULT_ROW_KEYS)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({key: row.get(key, "") for key in keys})


def _write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "dataset",
        "method",
        "n",
        "steps",
        "test_order",
        "test_index",
        "pc1",
        "y_true",
        "mu",
        "sigma",
        "lower90",
        "upper90",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({key: row.get(key, "") for key in keys})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small-train UCI PC1 prediction visualization.")
    p.add_argument("--datasets", type=str, default="Wine Quality")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_total_train", type=int, default=200)
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--max_train_anchors", type=int, default=128)
    p.add_argument("--neighbors", type=str, default="25,35")
    p.add_argument("--steps", type=str, default="200,300,500")
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--lmjgp_supervised_train_mc", type=int, default=2)
    p.add_argument("--lmjgp_supervised_kl_weight", type=float, default=1e-3)
    p.add_argument("--lmjgp_supervised_pred_var_floor", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--scaling_preset", type=str, default="uci_empirical", choices=("global", "uci_empirical"))
    p.add_argument("--standardize_x", type=int, default=1)
    p.add_argument("--standardize_y", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="experiments/uci/smalltrain_pca_viz")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    neighbors = _parse_int_list(args.neighbors)
    steps_list = _parse_int_list(args.steps)
    dataset_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    seed = int(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    metrics_rows: list[dict] = []
    pred_rows: list[dict] = []
    summary: dict = {
        "_meta": {
            "seed": seed,
            "datasets": dataset_list,
            "max_total_train": int(args.max_total_train),
            "max_test_anchors": int(args.max_test_anchors),
            "max_train_anchors": int(args.max_train_anchors),
            "neighbors": neighbors,
            "steps": steps_list,
            "xgb_boots": int(args.xgb_boots),
            "scaling_preset": args.scaling_preset,
            "plot_interval": "90% Gaussian range",
        }
    }

    for dataset_name in dataset_list:
        _set_seed(seed)
        print(f"\n{'=' * 72}\n{dataset_name}\n{'=' * 72}", flush=True)
        X_train_full, X_test_full, y_train_full, y_test_full, q_default = load_uci_split(dataset_name, seed)
        X_train_full, y_train_full, train_idx = _subsample(
            X_train_full,
            y_train_full,
            args.max_total_train,
            seed + 500,
        )
        X_test_sub, y_test_sub, test_idx = _subsample(
            X_test_full,
            y_test_full,
            args.max_test_anchors,
            seed + 1000,
        )
        X_base, y_base, X_anchor, y_anchor, split_info = _split_base_anchor(
            X_train_full,
            y_train_full,
            max_train_anchors=args.max_train_anchors,
            seed=seed + 2000,
        )
        if args.scaling_preset == "uci_empirical":
            standardize_x, standardize_y = SCALING_PRESETS.get(
                dataset_name,
                (bool(args.standardize_x), bool(args.standardize_y)),
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

        pca = PCA(n_components=1)
        pc1_train = pca.fit_transform(X_train_s).reshape(-1)
        pc1_test = pca.transform(X_test_s).reshape(-1)
        response_fig = figures_dir / f"{_slug(dataset_name)}_pc1_response.png"
        _plot_response_pc1(
            dataset=dataset_name,
            pc1_train=pc1_train,
            y_train=np.asarray(y_train_full, dtype=np.float64).reshape(-1),
            pc1_test=pc1_test,
            y_test=np.asarray(y_test_sub, dtype=np.float64).reshape(-1),
            out_path=response_fig,
        )

        X_train_t = torch.from_numpy(X_train_s).float().to(device)
        y_train_t = torch.from_numpy(y_train_s).float().to(device)
        X_base_t = torch.from_numpy(X_base_s).float().to(device)
        y_base_t = torch.from_numpy(y_base_s).float().to(device)
        X_anchor_t = torch.from_numpy(X_anchor_s).float().to(device)
        y_anchor_t = torch.from_numpy(y_anchor_s).float().to(device)
        X_test_t = torch.from_numpy(X_test_s).float().to(device)

        dataset_summary = {
            "Q": int(q_default),
            "split": {
                **split_info,
                "n_train_total": int(X_train_full.shape[0]),
                "n_test": int(X_test_sub.shape[0]),
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
            },
            "scaling": {"standardize_x": bool(standardize_x), "standardize_y": bool(standardize_y)},
            "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "figures": {"response_pc1": str(response_fig)},
            "metrics": [],
        }

        print(f"[{dataset_name}] running xgboost", flush=True)
        _set_seed(seed)
        mu_xgb_s, sig_xgb_s, wall_xgb, diag_xgb = _run_xgboost_bootstrap(
            X_train=X_train_s,
            y_train=y_train_s,
            X_test=X_test_s,
            y_test=y_test_s,
            n_bootstrap=args.xgb_boots,
            seed=seed,
        )
        metrics_xgb = _pack_metrics_maybe_original_y(y_scaler, y_test_s, mu_xgb_s, sig_xgb_s, wall_xgb)
        mu_xgb, sig_xgb = _invert_y(y_scaler, mu_xgb_s, sig_xgb_s)
        xgb_pred = {"mu": mu_xgb, "sigma": sig_xgb, "n": "", "steps": ""}
        row_xgb = {
            "dataset": dataset_name,
            "method": "xgboost",
            "n": "",
            "steps": "",
            **_as_metric_dict(metrics_xgb),
        }
        metrics_rows.append(row_xgb)
        dataset_summary["metrics"].append(row_xgb)
        print(f"  xgboost rmse={metrics_xgb[0]:.4f} crps={metrics_xgb[1]:.4f}", flush=True)

        y_test_orig = np.asarray(y_test_sub, dtype=np.float64).reshape(-1)
        for i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "dataset": dataset_name,
                    "method": "xgboost",
                    "n": "",
                    "steps": "",
                    "test_order": i,
                    "test_index": int(idx),
                    "pc1": float(pc1_test[i]),
                    "y_true": float(y_test_orig[i]),
                    "mu": float(mu_xgb[i]),
                    "sigma": float(sig_xgb[i]),
                    "lower90": float(mu_xgb[i] - Z90 * sig_xgb[i]),
                    "upper90": float(mu_xgb[i] + Z90 * sig_xgb[i]),
                }
            )

        lmjgp_predictions: list[dict] = []
        for n in neighbors:
            for steps in steps_list:
                print(f"[{dataset_name}] running lmjgp_supervised n={n} steps={steps}", flush=True)
                _set_seed(seed + 1234 + int(n))
                mu_lm_s, sig_lm_s, wall_lm, diag_lm = _run_lmjgp_supervised(
                    X_anchor_t=X_anchor_t,
                    y_anchor_t=y_anchor_t,
                    X_test_t=X_test_t,
                    y_test_np=y_test_s,
                    X_base_t=X_base_t,
                    y_base_t=y_base_t,
                    X_pool_t=X_train_t,
                    y_pool_t=y_train_t,
                    X_inducing_t=X_train_t,
                    Q=int(q_default),
                    m1=args.m1,
                    m2=args.m2,
                    n=int(n),
                    steps=int(steps),
                    lr=args.lr,
                    MC_num=args.MC_num,
                    train_mc=args.lmjgp_supervised_train_mc,
                    kl_weight=args.lmjgp_supervised_kl_weight,
                    pred_var_floor=args.lmjgp_supervised_pred_var_floor,
                    grad_check=False,
                    device=device,
                )
                metrics_lm = _pack_metrics_maybe_original_y(y_scaler, y_test_s, mu_lm_s, sig_lm_s, wall_lm)
                mu_lm, sig_lm = _invert_y(y_scaler, mu_lm_s, sig_lm_s)
                row_lm = {
                    "dataset": dataset_name,
                    "method": "lmjgp_supervised",
                    "n": int(n),
                    "steps": int(steps),
                    **_as_metric_dict(metrics_lm),
                }
                metrics_rows.append(row_lm)
                dataset_summary["metrics"].append(row_lm)
                lm_pred = {
                    "n": int(n),
                    "steps": int(steps),
                    "mu": mu_lm,
                    "sigma": sig_lm,
                    "metrics": row_lm,
                    "diagnostics": sanitize_for_json(diag_lm),
                }
                lmjgp_predictions.append(lm_pred)
                print(f"  lmjgp n={n} steps={steps} rmse={metrics_lm[0]:.4f} crps={metrics_lm[1]:.4f}", flush=True)

                for i, idx in enumerate(test_idx):
                    pred_rows.append(
                        {
                            "dataset": dataset_name,
                            "method": "lmjgp_supervised",
                            "n": int(n),
                            "steps": int(steps),
                            "test_order": i,
                            "test_index": int(idx),
                            "pc1": float(pc1_test[i]),
                            "y_true": float(y_test_orig[i]),
                            "mu": float(mu_lm[i]),
                            "sigma": float(sig_lm[i]),
                            "lower90": float(mu_lm[i] - Z90 * sig_lm[i]),
                            "upper90": float(mu_lm[i] + Z90 * sig_lm[i]),
                        }
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        best_lmjgp = min(lmjgp_predictions, key=lambda r: float(r["metrics"]["rmse"]))
        pred_fig = figures_dir / f"{_slug(dataset_name)}_pc1_predictions.png"
        _plot_predictions_pc1(
            dataset=dataset_name,
            pc1_test=pc1_test,
            y_test=y_test_orig,
            xgb=xgb_pred,
            lmjgp=best_lmjgp,
            out_path=pred_fig,
        )
        dataset_summary["figures"]["predictions_pc1"] = str(pred_fig)
        dataset_summary["best_lmjgp_by_rmse"] = {
            "n": int(best_lmjgp["n"]),
            "steps": int(best_lmjgp["steps"]),
            "metrics": best_lmjgp["metrics"],
        }
        dataset_summary["xgboost_diagnostics"] = sanitize_for_json(diag_xgb)
        summary[dataset_name] = dataset_summary

        _write_metrics(out_dir / "metrics.csv", metrics_rows)
        _write_predictions(out_dir / "predictions.csv", pred_rows)
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(summary), f, indent=2)

    print(f"\nSaved metrics -> {out_dir / 'metrics.csv'}", flush=True)
    print(f"Saved predictions -> {out_dir / 'predictions.csv'}", flush=True)
    print(f"Saved summary -> {out_dir / 'summary.json'}", flush=True)
    print(f"Saved figures -> {figures_dir}", flush=True)
    return summary


if __name__ == "__main__":
    main()

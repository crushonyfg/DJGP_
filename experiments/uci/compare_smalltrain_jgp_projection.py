"""Run lightweight deterministic projection + JumpGP baselines on UCI splits.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Appendix-style projection baseline table, train=500, five seeds.
    python experiments/uci/compare_smalltrain_jgp_projection.py ^
        --num_exp 5 --train_sizes 500 ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --methods "jgp_pca,jgp_sir" ^
        --out_dir experiments/uci/smalltrain_jgp_projection_train500_5seeds

    # Smoke test.
    python experiments/uci/compare_smalltrain_jgp_projection.py ^
        --num_exp 1 --train_sizes 80 --datasets "Wine Quality" ^
        --max_test_anchors 20 --methods "jgp_pca,jgp_sir" ^
        --out_dir experiments/uci/smalltrain_jgp_projection_smoke

The runner uses the same split policy as the current small-train UCI summary:
Wine and Appliances use random-row splits; Parkinsons uses subject-grouped
splits. PCA and SIR transformations are fit on training rows only.
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from experiments.uci.compare_smalltrain_multiseed_grouped import (  # noqa: E402
    TUNED_PRESETS,
    _load_dataset_for_seed,
)
from experiments.uci.uci_data import sir_reduction  # noqa: E402
from jumpgp import JumpGP  # noqa: E402


def _write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _dataset_y_scaler(dataset_name: str, y_train: np.ndarray):
    if not bool(TUNED_PRESETS[dataset_name]["standardize_y"]):
        return None
    scaler = StandardScaler()
    scaler.fit(np.asarray(y_train, dtype=np.float64).reshape(-1, 1))
    return scaler


def _scale_y_for_fit(y_train: np.ndarray, scaler) -> np.ndarray:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if scaler is None:
        return y
    return scaler.transform(y.reshape(-1, 1)).reshape(-1)


def _inverse_jgp_prediction(mu_fit: np.ndarray, sigma_fit: np.ndarray, scaler) -> tuple[np.ndarray, np.ndarray]:
    mu_fit = np.asarray(mu_fit, dtype=np.float64).reshape(-1)
    sigma_fit = np.asarray(sigma_fit, dtype=np.float64).reshape(-1)
    if scaler is None:
        return mu_fit, sigma_fit
    scale = float(np.squeeze(scaler.scale_))
    mu = scaler.inverse_transform(mu_fit.reshape(-1, 1)).reshape(-1)
    return mu, sigma_fit * abs(scale)


def _run_jgp_projection(
    X_train_proj: np.ndarray,
    y_train_fit: np.ndarray,
    X_test_proj: np.ndarray,
    y_test_orig: np.ndarray,
    *,
    y_scaler,
    n_neighbors: int,
) -> tuple[list[float], dict]:
    t0 = time.perf_counter()
    jgp = JumpGP(
        np.asarray(X_train_proj, dtype=np.float64),
        np.asarray(y_train_fit, dtype=np.float64).reshape(-1),
        np.asarray(X_test_proj, dtype=np.float64),
        L=1,
        M=int(n_neighbors),
        mode="CEM",
        bVerbose=False,
    )
    jgp.fit()
    elapsed = time.perf_counter() - t0
    mu_fit = np.asarray(jgp.jump_results["mu"], dtype=np.float64).reshape(-1)
    sig2_fit = np.asarray(jgp.jump_results["sig2"], dtype=np.float64).reshape(-1)
    n_negative_sig2 = int(np.sum(~np.isfinite(sig2_fit) | (sig2_fit < 0.0)))
    sig2_fit = np.where(np.isfinite(sig2_fit) & (sig2_fit >= 0.0), sig2_fit, 1e-12)
    sig2_fit = np.maximum(sig2_fit, 1e-12)
    sigma_fit = np.sqrt(sig2_fit)
    mu, sigma = _inverse_jgp_prediction(mu_fit, sigma_fit, y_scaler)
    y_eval = np.asarray(y_test_orig, dtype=np.float64).reshape(-1)
    mean_crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y_eval, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        ).item()
    )
    metrics = pack_gaussian_uq_list(
        float(np.sqrt(np.mean((mu - y_eval) ** 2))),
        mean_crps,
        elapsed,
        y_eval,
        mu,
        sigma,
    )
    diag = {
        "n_negative_or_nonfinite_sig2": n_negative_sig2,
        "pred_sigma_mean": float(np.nanmean(sigma)),
        "pred_sigma_max": float(np.nanmax(sigma)),
    }
    return metrics, diag


def _project(method: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, K: int):
    if method == "jgp_pca":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        pca = PCA(n_components=K)
        return pca.fit_transform(X_train_s), pca.transform(X_test_s), {
            "projection": "pca",
            "projection_fit": "train_only",
            "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        }
    if method == "jgp_sir":
        X_train_sir, sir_vecs, scaler = sir_reduction(X_train, y_train, H=10, K=K)
        X_test_sir = scaler.transform(X_test) @ sir_vecs
        return X_train_sir, X_test_sir, {
            "projection": "sir",
            "projection_fit": "train_only",
            "sir_H": 10,
        }
    if method == "jgp_raw":
        return np.asarray(X_train), np.asarray(X_test), {
            "projection": "raw",
            "projection_fit": "none",
        }
    raise ValueError(f"Unknown method {method!r}")


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["train_size"], row["dataset"], row["method"]), []).append(row)
    out: list[dict] = []
    for (train_size, dataset, method), group in sorted(groups.items()):
        agg = {"train_size": train_size, "dataset": dataset, "method": method, "n_seeds": len(group)}
        for key in RESULT_ROW_KEYS:
            vals = np.asarray([float(r[key]) for r in group], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small-train PCA/SIR + JumpGP baselines.")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--train_sizes", type=str, default="500")
    p.add_argument(
        "--datasets",
        type=str,
        default="Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction",
    )
    p.add_argument("--methods", type=str, default="jgp_pca,jgp_sir")
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--train_subject_frac", type=float, default=0.8)
    p.add_argument("--out_dir", type=str, default="experiments/uci/smalltrain_jgp_projection_train500_5seeds")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_sizes = [int(x.strip()) for x in args.train_sizes.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    allowed = {"jgp_pca", "jgp_sir", "jgp_raw"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for train_size in train_sizes:
        for dataset in datasets:
            preset = TUNED_PRESETS[dataset]
            for seed in range(args.num_exp):
                data = _load_dataset_for_seed(
                    dataset,
                    seed,
                    max_total_train=train_size,
                    max_test_anchors=args.max_test_anchors,
                    train_subject_frac=args.train_subject_frac,
                )
                y_scaler = _dataset_y_scaler(dataset, data["y_train"])
                y_train_fit = _scale_y_for_fit(data["y_train"], y_scaler)
                for method in methods:
                    print(f"train={train_size} {dataset} seed={seed} {method}", flush=True)
                    row = {
                        "train_size": int(train_size),
                        "dataset": dataset,
                        "seed": int(seed),
                        "method": method,
                        "split_type": preset["split"],
                        "standardize_y": bool(preset["standardize_y"]),
                        "n": int(preset["n"]),
                    }
                    diag = dict(row)
                    try:
                        Xtr_proj, Xte_proj, proj_diag = _project(
                            method,
                            data["X_train"],
                            y_train_fit,
                            data["X_test"],
                            int(data["K"]),
                        )
                        metrics, fit_diag = _run_jgp_projection(
                            Xtr_proj,
                            y_train_fit,
                            Xte_proj,
                            data["y_test"],
                            y_scaler=y_scaler,
                            n_neighbors=int(preset["n"]),
                        )
                        row.update({key: float(metrics[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
                        row["status"] = "ok"
                        diag.update(proj_diag)
                        diag.update(fit_diag)
                        diag["status"] = "ok"
                        print(
                            f"  rmse={row['rmse']:.4f} crps={row['mean_crps']:.4f} "
                            f"cov90={row['coverage_90']:.3f}",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        row.update({key: float("nan") for key in RESULT_ROW_KEYS})
                        row["status"] = "error"
                        diag["status"] = "error"
                        diag["error"] = str(exc)
                        print(f"  [error] {exc}", flush=True)
                    metric_rows.append(row)
                    diagnostic_rows.append(diag)

                _write_csv(
                    out_dir / "metrics.csv",
                    metric_rows,
                    [
                        "train_size",
                        "dataset",
                        "seed",
                        "method",
                        "split_type",
                        "standardize_y",
                        "n",
                        "status",
                        *RESULT_ROW_KEYS,
                    ],
                )
                _write_csv(
                    out_dir / "diagnostics.csv",
                    diagnostic_rows,
                    [
                        "train_size",
                        "dataset",
                        "seed",
                        "method",
                        "split_type",
                        "standardize_y",
                        "n",
                        "status",
                        "projection",
                        "projection_fit",
                        "explained_variance_sum",
                        "sir_H",
                        "n_negative_or_nonfinite_sig2",
                        "pred_sigma_mean",
                        "pred_sigma_max",
                        "error",
                    ],
                )
                _write_csv(
                    out_dir / "aggregate.csv",
                    _aggregate(metric_rows),
                    [
                        "train_size",
                        "dataset",
                        "method",
                        "n_seeds",
                        *[f"{key}_{stat}" for key in RESULT_ROW_KEYS for stat in ("mean", "std")],
                    ],
                )

    summary = {
        "args": vars(args),
        "preset_policy": TUNED_PRESETS,
        "metrics": "metrics.csv",
        "aggregate": "aggregate.csv",
        "diagnostics": "diagnostics.csv",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Small-Train Projection + JumpGP Baselines\n\n"
        "Lightweight appendix-style PCA/SIR + JumpGP baselines. "
        "PCA/SIR are fit on training rows only; Appliances uses train-only target "
        "standardization for fitting and reports original-scale metrics.\n",
        encoding="utf-8",
    )
    print(f"Saved artifacts to {out_dir}", flush=True)
    return {"metrics": metric_rows, "aggregate": _aggregate(metric_rows)}


if __name__ == "__main__":
    main()

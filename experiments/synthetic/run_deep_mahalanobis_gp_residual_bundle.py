"""Train DeepMahalanobisGP and export a bundle for residual self-CEM heads.

How to run from the repository root with conda env ``dmgp_tf1``:

    conda activate dmgp_tf1
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/run_deep_mahalanobis_gp_residual_bundle.py ^
        --data_dir experiments/synthetic/deep_mahalanobis_gp_paper_train1000_seed0 ^
        --out_dir experiments/synthetic/dmgp_residual_bundle_lh_seed0_20260517 ^
        --settings lh_q5_train1000 --seed 0 ^
        --steps1 300 --steps2 700 --oof_folds 3

This script fits the external GPflow-1 ``DeepVMGP`` model on the exported
paper-style synthetic split, saves direct DMGP predictions, train/test
projection moments, and optional K-fold train-side OOF mean/variance arrays
for downstream residual uncertain self-CEM experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = PROJECT_ROOT.parent / "external" / "DeepMahalanobisGP"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gpflow  # noqa: E402
import tensorflow as tf  # noqa: E402
from library.models import deep_vmgp  # noqa: E402


RESULT_KEYS = (
    "rmse",
    "mean_crps",
    "wall_sec",
    "coverage_90",
    "mean_width_90",
    "coverage_95",
    "mean_width_95",
)


def _gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    from scipy.special import erf

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64).reshape(-1), 1e-12)
    z = (y - mu) / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))


def _metric_row(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64).reshape(-1), 1e-12)
    z90 = 1.6448536269514722
    z95 = 1.959963984540054
    lo90, hi90 = mu - z90 * sigma, mu + z90 * sigma
    lo95, hi95 = mu - z95 * sigma, mu + z95 * sigma
    return {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "mean_crps": float(np.mean(_gaussian_crps(y, mu, sigma))),
        "wall_sec": float(wall_sec),
        "coverage_90": float(np.mean((y >= lo90) & (y <= hi90))),
        "mean_width_90": float(np.mean(hi90 - lo90)),
        "coverage_95": float(np.mean((y >= lo95) & (y <= hi95))),
        "mean_width_95": float(np.mean(hi95 - lo95)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "setting",
        "seed",
        "method",
        "N_train",
        "N_test",
        "Q",
        "D",
        *RESULT_KEYS,
        "train_sec",
        "predict_sec",
        "oof_folds",
        "m_u",
        "m_v",
        "steps1",
        "steps2",
        "W_mean_norm_mean",
        "W_var_mean",
        "status",
        "error",
    ]
    keys: list[str] = []
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _compute_w_arrays(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sess = model.enquire_session()
    qW_mean_tf, qW_var_tf = model.compute_qW(X, full_cov=False)
    W_mean_raw, W_var_raw = sess.run([qW_mean_tf, qW_var_tf])
    W_mean = np.transpose(W_mean_raw[..., 0], (2, 0, 1))
    W_var = np.transpose(W_var_raw[:, 0, :], (1, 0))
    return W_mean, W_var


def _build_model(X_train: np.ndarray, y_train_std: np.ndarray, *, Q: int, m_u: int, m_v: int) -> Any:
    D = int(X_train.shape[1])
    with gpflow.defer_build():
        w_kerns = [gpflow.kernels.RBF(D, ARD=True) for _ in range(int(Q))]
        model = deep_vmgp.DeepVMGP(
            X_train,
            y_train_std.reshape(-1, 1),
            (int(m_u), int(Q)),
            (int(m_v), int(D)),
            w_kerns,
            full_qcov=False,
            diag_qmu=False,
            minibatch_size=None,
        )
    model.compile()
    return model


def _fit_model(
    X_train: np.ndarray,
    y_train_std: np.ndarray,
    *,
    Q: int,
    m_u: int,
    m_v: int,
    steps1: int,
    steps2: int,
    lr: float,
    initial_noise_var: float,
    seed: int,
) -> tuple[Any, float]:
    gpflow.reset_default_graph_and_session()
    np.random.seed(int(seed))
    tf.set_random_seed(int(seed))
    model = _build_model(X_train, y_train_std, Q=Q, m_u=m_u, m_v=m_v)
    variance_parameter = model.likelihood.variance
    variance_parameter.assign(float(initial_noise_var))
    t0 = time.perf_counter()
    variance_parameter.trainable = False
    if int(steps1) > 0:
        gpflow.train.AdamOptimizer(float(lr)).minimize(model, maxiter=int(steps1))
    variance_parameter.trainable = True
    if int(steps2) > 0:
        gpflow.train.AdamOptimizer(float(lr)).minimize(model, maxiter=int(steps2))
    return model, float(time.perf_counter() - t0)


def _predict_std(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_s, var_s = model.predict_y(X)
    mean = np.asarray(mean_s, dtype=np.float64).reshape(-1)
    var = np.maximum(np.asarray(var_s, dtype=np.float64).reshape(-1), 1e-12)
    return mean, var


def _save_bundle_copy(data_path: Path, out_dir: Path) -> None:
    dst = out_dir / data_path.name
    if data_path.resolve() != dst.resolve():
        shutil.copy2(data_path, dst)


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# DeepMahalanobisGP Residual Bundle

This folder stores the direct DeepMahalanobisGP fit plus the extra arrays
needed by the fixed-W residual uncertain self-CEM head.

Key files:

- `*_train_fit_mean_std.npy`, `*_train_fit_var_std.npy`: full-fit DMGP train
  predictive mean/variance on standardized y.
- `*_train_oof_mean_std.npy`, `*_train_oof_var_std.npy`: optional K-fold OOF
  train predictive mean/variance on standardized y.
- `*_test_mean_std.npy`, `*_test_var_std.npy`: full-fit DMGP test predictive
  mean/variance on standardized y.
- `*_pred_mean.npy`, `*_pred_sigma.npy`: full-fit DMGP test prediction on the
  original y scale.
- `*_W_train_mean.npy`, `*_W_train_var.npy`, `*_W_test_mean.npy`,
  `*_W_test_var.npy`: DMGP projection moments used by downstream heads.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _run_oof_predictions(
    X_train: np.ndarray,
    y_train_std: np.ndarray,
    *,
    Q: int,
    m_u: int,
    m_v: int,
    steps1: int,
    steps2: int,
    lr: float,
    initial_noise_var: float,
    seed: int,
    oof_folds: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_train = int(X_train.shape[0])
    mean_oof = np.empty(n_train, dtype=np.float64)
    var_oof = np.empty(n_train, dtype=np.float64)
    total_sec = 0.0
    splitter = KFold(n_splits=int(oof_folds), shuffle=True, random_state=int(seed))
    for fold_id, (fit_idx, hold_idx) in enumerate(splitter.split(X_train)):
        fold_seed = int(seed) + 1009 * (fold_id + 1)
        model, train_sec = _fit_model(
            X_train[fit_idx],
            y_train_std[fit_idx],
            Q=Q,
            m_u=min(int(m_u), int(len(fit_idx))),
            m_v=min(int(m_v), int(len(fit_idx))),
            steps1=steps1,
            steps2=steps2,
            lr=lr,
            initial_noise_var=initial_noise_var,
            seed=fold_seed,
        )
        t0 = time.perf_counter()
        mean_fold, var_fold = _predict_std(model, X_train[hold_idx])
        total_sec += float(train_sec + (time.perf_counter() - t0))
        mean_oof[hold_idx] = mean_fold
        var_oof[hold_idx] = var_fold
    return mean_oof, var_oof, total_sec


def run_setting(args: argparse.Namespace, setting: str, seed: int) -> dict[str, Any]:
    data_path = Path(args.data_dir) / f"{setting}_seed{seed}.npz"
    arr = np.load(data_path)
    X_train = np.asarray(arr["X_train"], dtype=np.float64)
    X_test = np.asarray(arr["X_test"], dtype=np.float64)
    y_train_std = np.asarray(arr["y_train_std"], dtype=np.float64).reshape(-1)
    y_test = np.asarray(arr["y_test"], dtype=np.float64).reshape(-1)
    y_mean = float(np.asarray(arr["y_mean"]).reshape(-1)[0])
    y_scale = float(np.asarray(arr["y_scale"]).reshape(-1)[0])
    y_scale = y_scale if abs(y_scale) > 1e-12 else 1.0
    n_train, D = X_train.shape
    n_test = int(X_test.shape[0])
    Q = int(args.Q) if int(args.Q) > 0 else (2 if setting.startswith("l2_") else 5)
    m_u = min(int(args.m_u), n_train)
    m_v = min(int(args.m_v), n_train)

    model, train_sec = _fit_model(
        X_train,
        y_train_std,
        Q=Q,
        m_u=m_u,
        m_v=m_v,
        steps1=int(args.steps1),
        steps2=int(args.steps2),
        lr=float(args.lr),
        initial_noise_var=float(args.initial_noise_var),
        seed=int(seed),
    )
    t_pred = time.perf_counter()
    train_mean_std, train_var_std = _predict_std(model, X_train)
    test_mean_std, test_var_std = _predict_std(model, X_test)
    predict_sec = float(time.perf_counter() - t_pred)

    test_mean = test_mean_std * y_scale + y_mean
    test_sigma = np.sqrt(test_var_std) * abs(y_scale)
    metrics = _metric_row(y_test, test_mean, test_sigma, train_sec + predict_sec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_bundle_copy(data_path, out_dir)
    prefix = out_dir / f"{setting}_seed{seed}"
    np.save(str(prefix) + "_train_fit_mean_std.npy", train_mean_std)
    np.save(str(prefix) + "_train_fit_var_std.npy", train_var_std)
    np.save(str(prefix) + "_test_mean_std.npy", test_mean_std)
    np.save(str(prefix) + "_test_var_std.npy", test_var_std)
    np.save(str(prefix) + "_pred_mean.npy", test_mean)
    np.save(str(prefix) + "_pred_sigma.npy", test_sigma)

    W_train_mean, W_train_var = _compute_w_arrays(model, X_train)
    W_test_mean, W_test_var = _compute_w_arrays(model, X_test)
    np.save(str(prefix) + "_W_train_mean.npy", W_train_mean)
    np.save(str(prefix) + "_W_train_var.npy", W_train_var)
    np.save(str(prefix) + "_W_test_mean.npy", W_test_mean)
    np.save(str(prefix) + "_W_test_var.npy", W_test_var)
    np.save(str(prefix) + "_W_mean.npy", W_test_mean)
    np.save(str(prefix) + "_W_var.npy", W_test_var)

    oof_sec = 0.0
    if int(args.oof_folds) > 1:
        train_oof_mean_std, train_oof_var_std, oof_sec = _run_oof_predictions(
            X_train,
            y_train_std,
            Q=Q,
            m_u=m_u,
            m_v=m_v,
            steps1=int(args.steps1),
            steps2=int(args.steps2),
            lr=float(args.lr),
            initial_noise_var=float(args.initial_noise_var),
            seed=int(seed),
            oof_folds=int(args.oof_folds),
        )
        np.save(str(prefix) + "_train_oof_mean_std.npy", train_oof_mean_std)
        np.save(str(prefix) + "_train_oof_var_std.npy", train_oof_var_std)

    return {
        "setting": setting,
        "seed": int(seed),
        "method": "deep_mahalanobis_gp_bundle",
        "N_train": int(n_train),
        "N_test": int(n_test),
        "Q": int(Q),
        "D": int(D),
        **metrics,
        "train_sec": float(train_sec),
        "predict_sec": float(predict_sec),
        "oof_folds": int(args.oof_folds),
        "oof_train_sec": float(oof_sec),
        "m_u": int(m_u),
        "m_v": int(m_v),
        "steps1": int(args.steps1),
        "steps2": int(args.steps2),
        "lr": float(args.lr),
        "W_mean_norm_mean": float(np.linalg.norm(W_test_mean, axis=-1).mean()),
        "W_var_mean": float(np.mean(W_test_var)),
        "status": "ok",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DeepMahalanobisGP and export residual-head bundles.")
    p.add_argument("--data_dir", default="experiments/synthetic/deep_mahalanobis_gp_paper_train1000_seed0")
    p.add_argument("--out_dir", default="experiments/synthetic/dmgp_residual_bundle_lh_seed0_20260517")
    p.add_argument("--settings", default="lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="", help="Optional comma-separated seed list. Overrides --seed when set.")
    p.add_argument("--Q", type=int, default=0)
    p.add_argument("--m_u", type=int, default=50)
    p.add_argument("--m_v", type=int, default=25)
    p.add_argument("--steps1", type=int, default=300)
    p.add_argument("--steps2", type=int, default=700)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--initial_noise_var", type=float, default=0.01)
    p.add_argument("--oof_folds", type=int, default=0)
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = [s.strip() for s in str(args.settings).split(",") if s.strip()]
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()] if args.seeds else [int(args.seed)]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for setting in settings:
            print(f"[run] {setting} seed={seed}", flush=True)
            try:
                rows.append(run_setting(args, setting, seed))
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "setting": setting,
                        "seed": int(seed),
                        "method": "deep_mahalanobis_gp_bundle",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            _write_csv(out_dir / "dmgp_bundle_metrics_partial.csv", rows)
    _write_csv(out_dir / "dmgp_bundle_metrics.csv", rows)
    summary = {
        "args": vars(args),
        "external_repo": str(EXTERNAL_ROOT),
        "n_rows": len(rows),
    }
    (out_dir / "dmgp_bundle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_readme(out_dir, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

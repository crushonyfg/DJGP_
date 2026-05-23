"""Run the external DeepMahalanobisGP implementation on exported synthetic data.

How to run from the repository root with conda env ``dmgp_tf1``:

    conda activate dmgp_tf1
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/run_deep_mahalanobis_gp_paper.py ^
        --data_dir experiments/synthetic/deep_mahalanobis_gp_paper_train1000_seed0 ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --seed 0 --steps1 300 --steps2 700

    python experiments/synthetic/run_deep_mahalanobis_gp_paper.py ^
        --data_dir experiments/synthetic/deep_mahalanobis_gp_paper_train1000_5seeds ^
        --out_dir experiments/synthetic/deep_mahalanobis_gp_paper_train1000_5seeds ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --seeds 0,1,2,3,4 --steps1 300 --steps2 700

This script imports the unmodified external repository at
``../external/DeepMahalanobisGP`` and trains its original GPflow-1
``DeepVMGP`` model.  Data must first be exported by
``export_paper_synthetic_for_dmgp.py``.  The model is trained on standardized
Y, matching the external notebook's preprocessing; reported metrics are
converted back to the original Y scale.  In addition to test-side predictions,
the runner saves posterior mean/variance arrays for ``W(x)`` on both train and
test inputs so downstream scripts can evaluate global mapped-data baselines
such as ``z_i = W(x_i)x_i`` followed by JumpGP.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = PROJECT_ROOT.parent / "external" / "DeepMahalanobisGP"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gpflow  # noqa: E402
import tensorflow as tf  # noqa: E402
from library import helper  # noqa: E402
from library.models import deep_vmgp  # noqa: E402


def _initial_inducing_points(X: np.ndarray, m: int) -> np.ndarray:
    return helper.initial_inducing_points(np.asarray(X), int(m))


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
    keys: list[str] = []
    preferred = [
        "setting",
        "seed",
        "method",
        "N_train",
        "N_test",
        "Q",
        "D",
        "rmse",
        "mean_crps",
        "wall_sec",
        "coverage_90",
        "mean_width_90",
        "coverage_95",
        "mean_width_95",
        "train_sec",
        "predict_sec",
        "m_u",
        "m_v",
        "steps1",
        "steps2",
        "y_standardized",
        "W_mean_norm_mean",
        "W_mean_anchor_frob_std",
        "W_var_mean",
    ]
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


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    metrics = ["rmse", "mean_crps", "coverage_90", "mean_width_90", "coverage_95", "mean_width_95", "wall_sec"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["setting"]), str(row["method"])), []).append(row)
    out_rows: list[dict[str, Any]] = []
    for (setting, method), items in sorted(groups.items()):
        agg: dict[str, Any] = {
            "setting": setting,
            "method": method,
            "n_seeds": len(items),
            "seeds": ",".join(str(int(r["seed"])) for r in items),
        }
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in items], dtype=np.float64)
            agg[f"{metric}_mean"] = float(vals.mean())
            agg[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out_rows.append(agg)
    _write_csv(path, out_rows)


def _setting_q(setting: str) -> int:
    if setting.startswith("l2_"):
        return 2
    if setting.startswith("lh_"):
        return 5
    raise ValueError(f"Cannot infer Q from setting {setting!r}")


def _compute_w_arrays(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``W_mean [N,Q,D]`` and ``W_var [N,Q]`` from DeepVMGP."""
    sess = model.enquire_session()
    qW_mean_tf, qW_var_tf = model.compute_qW(X, full_cov=False)
    W_mean_raw, W_var_raw = sess.run([qW_mean_tf, qW_var_tf])
    # compute_qW mean is [Q, D, N, 1]; variance is [Q, 1, N].
    W_mean = np.transpose(W_mean_raw[..., 0], (2, 0, 1))
    W_var = np.transpose(W_var_raw[:, 0, :], (1, 0))
    return W_mean, W_var


def run_setting(args: argparse.Namespace, setting: str, seed: int) -> dict[str, Any]:
    gpflow.reset_default_graph_and_session()
    np.random.seed(int(seed))
    tf.set_random_seed(int(seed))
    data_path = Path(args.data_dir) / f"{setting}_seed{seed}.npz"
    arr = np.load(data_path)
    X_train = np.asarray(arr["X_train"], dtype=np.float64)
    X_test = np.asarray(arr["X_test"], dtype=np.float64)
    y_train = np.asarray(arr["y_train_std"], dtype=np.float64).reshape(-1, 1)
    y_test = np.asarray(arr["y_test"], dtype=np.float64).reshape(-1)
    y_mean = float(np.asarray(arr["y_mean"]).reshape(-1)[0])
    y_scale = float(np.asarray(arr["y_scale"]).reshape(-1)[0])
    n_train, D = X_train.shape
    n_test = X_test.shape[0]
    Q = int(args.Q) if int(args.Q) > 0 else _setting_q(setting)
    m_u = int(args.m_u)
    m_v = int(args.m_v)
    if m_u <= 0:
        m_u = min(50, n_train)
    if m_v <= 0:
        m_v = min(25, n_train)

    print(f"[{setting}] build DeepVMGP Q={Q} D={D} m_u={m_u} m_v={m_v}", flush=True)
    with gpflow.defer_build():
        w_kerns = [gpflow.kernels.RBF(D, ARD=True) for _ in range(Q)]
        model = deep_vmgp.DeepVMGP(
            X_train,
            y_train,
            (m_u, Q),
            (m_v, D),
            w_kerns,
            full_qcov=False,
            diag_qmu=False,
            minibatch_size=None,
        )
    model.compile()

    t0 = time.perf_counter()
    variance_parameter = model.likelihood.variance
    variance_parameter.assign(float(args.initial_noise_var))
    variance_parameter.trainable = False
    if int(args.steps1) > 0:
        gpflow.train.AdamOptimizer(float(args.lr)).minimize(model, maxiter=int(args.steps1))
    variance_parameter.trainable = True
    if int(args.steps2) > 0:
        gpflow.train.AdamOptimizer(float(args.lr)).minimize(model, maxiter=int(args.steps2))
    train_sec = time.perf_counter() - t0

    print(f"[{setting}] predict", flush=True)
    t1 = time.perf_counter()
    mean_s, var_s = model.predict_y(X_test)
    predict_sec = time.perf_counter() - t1
    mean_s = np.asarray(mean_s, dtype=np.float64).reshape(-1)
    var_s = np.maximum(np.asarray(var_s, dtype=np.float64).reshape(-1), 1e-12)
    mean = mean_s * y_scale + y_mean
    sigma = np.sqrt(var_s) * y_scale

    sess = model.enquire_session()
    W_train_mean, W_train_var = _compute_w_arrays(model, X_train)
    W_mean, W_var = _compute_w_arrays(model, X_test)
    W_samples = None
    if int(args.w_samples) > 0:
        W_samples_raw = sess.run(model.sample_W(X_test, samples=int(args.w_samples)))
        # sample_W returns [M, Q, D, T].
        W_samples = np.transpose(W_samples_raw, (0, 3, 1, 2))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{setting}_seed{seed}_W_train_mean.npy", W_train_mean)
    np.save(out_dir / f"{setting}_seed{seed}_W_train_var.npy", W_train_var)
    np.save(out_dir / f"{setting}_seed{seed}_W_test_mean.npy", W_mean)
    np.save(out_dir / f"{setting}_seed{seed}_W_test_var.npy", W_var)
    np.save(out_dir / f"{setting}_seed{seed}_W_mean.npy", W_mean)
    np.save(out_dir / f"{setting}_seed{seed}_W_var.npy", W_var)
    if W_samples is not None:
        np.save(out_dir / f"{setting}_seed{seed}_W_samples.npy", W_samples)
    np.save(out_dir / f"{setting}_seed{seed}_pred_mean.npy", mean)
    np.save(out_dir / f"{setting}_seed{seed}_pred_sigma.npy", sigma)

    metric = _metric_row(y_test, mean, sigma, train_sec + predict_sec)
    row = {
        "setting": setting,
        "seed": int(seed),
        "method": "deep_mahalanobis_gp",
        "N_train": int(n_train),
        "N_test": int(n_test),
        "Q": int(Q),
        "D": int(D),
        **metric,
        "train_sec": float(train_sec),
        "predict_sec": float(predict_sec),
        "m_u": int(m_u),
        "m_v": int(m_v),
        "steps1": int(args.steps1),
        "steps2": int(args.steps2),
        "lr": float(args.lr),
        "y_standardized": True,
        "W_mean_norm_mean": float(np.linalg.norm(W_mean, axis=-1).mean()),
        "W_mean_anchor_frob_std": float(np.linalg.norm(W_mean, axis=(1, 2)).std()),
        "W_var_mean": float(np.mean(W_var)),
    }
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run external DeepMahalanobisGP on exported paper synthetic data.")
    p.add_argument("--data_dir", default="experiments/synthetic/deep_mahalanobis_gp_paper_train1000_seed0")
    p.add_argument("--out_dir", default="experiments/synthetic/deep_mahalanobis_gp_paper_train1000_seed0")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="", help="Optional comma-separated seed list. Overrides --seed when set.")
    p.add_argument("--Q", type=int, default=0)
    p.add_argument("--m_u", type=int, default=50)
    p.add_argument("--m_v", type=int, default=25)
    p.add_argument("--steps1", type=int, default=300)
    p.add_argument("--steps2", type=int, default=700)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--initial_noise_var", type=float, default=0.01)
    p.add_argument("--w_samples", type=int, default=3)
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [int(args.seed)]
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    for seed in seeds:
        for setting in settings:
            rows.append(run_setting(args, setting, seed))
            _write_csv(out_dir / "dmgp_metrics_partial.csv", rows)
    _write_csv(out_dir / "dmgp_metrics.csv", rows)
    _write_aggregate_csv(out_dir / "dmgp_aggregate.csv", rows)
    summary = {"args": vars(args), "rows": rows, "external_repo": str(EXTERNAL_ROOT)}
    (out_dir / "dmgp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

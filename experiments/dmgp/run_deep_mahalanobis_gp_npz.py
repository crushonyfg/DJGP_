"""Run external DeepMahalanobisGP on generic exported ``.npz`` splits.

How to run from the repository root with conda env ``dmgp_tf1``:

    conda activate dmgp_tf1
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/dmgp/run_deep_mahalanobis_gp_npz.py ^
        --manifest experiments/uci/dmgp_smalltrain_5seeds_20260517/manifest.json ^
        --out_dir experiments/uci/dmgp_smalltrain_5seeds_20260517 ^
        --steps1 300 --steps2 700

    python experiments/dmgp/run_deep_mahalanobis_gp_npz.py ^
        --manifest experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517/manifest.json ^
        --out_dir experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517 ^
        --steps1 300 --steps2 700

The manifest is produced by the benchmark-specific export scripts.  Each entry
points at one ``.npz`` file containing ``X_train``, ``X_test``,
``y_train_std``, ``y_test``, ``y_mean``, and ``y_scale``.  The external model
is trained on standardized ``y_train_std`` and metrics are reported after
transforming predictions back to the original response scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = PROJECT_ROOT.parent / "external" / "DeepMahalanobisGP"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


def _metric_row(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> Dict[str, float]:
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


def _metric_rows_by_subset(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    wall_sec: float,
    arr: np.lib.npyio.NpzFile,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    masks: List[Tuple[str, np.ndarray]] = [("all", np.ones(np.asarray(y).reshape(-1).shape[0], dtype=bool))]
    if "threshold_distance_test" in arr and "near_threshold_delta" in arr:
        dist = np.asarray(arr["threshold_distance_test"], dtype=np.float64).reshape(-1)
        delta = float(np.asarray(arr["near_threshold_delta"]).reshape(-1)[0])
        if dist.shape[0] == masks[0][1].shape[0]:
            near = dist <= delta
            masks.append(("near_threshold", near))
            masks.append(("far_from_threshold", ~near))
    for subset, mask in masks:
        if int(mask.sum()) == 0:
            continue
        row = _metric_row(
            np.asarray(y).reshape(-1)[mask],
            np.asarray(mu).reshape(-1)[mask],
            np.asarray(sigma).reshape(-1)[mask],
            wall_sec,
        )
        row["subset"] = subset
        row["n_eval"] = int(mask.sum())
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    preferred = [
        "benchmark",
        "dataset",
        "train_size",
        "setting",
        "seed",
        "method",
        "subset",
        "n_eval",
        "N_train",
        "N_test",
        "Q",
        "D",
        *RESULT_KEYS,
        "train_sec",
        "predict_sec",
        "m_u",
        "m_v",
        "steps1",
        "steps2",
        "lr",
        "split_type",
        "standardize_x",
        "standardize_y_for_baseline",
        "y_standardized_for_dmgp",
        "status",
        "error",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = (
            str(row.get("benchmark", "")),
            str(row.get("dataset", "")),
            str(row.get("train_size", "")),
            str(row.get("method", "")),
            str(row.get("subset", "all")),
        )
        groups.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for (benchmark, dataset, train_size, method, subset), items in sorted(groups.items()):
        agg: Dict[str, Any] = {
            "benchmark": benchmark,
            "dataset": dataset,
            "train_size": train_size,
            "method": method,
            "subset": subset,
            "n_seeds": len(items),
            "seeds": ",".join(str(int(r["seed"])) for r in items),
            "source": "deep_mahalanobis_gp_npz",
        }
        for metric in RESULT_KEYS:
            vals = np.asarray([float(r[metric]) for r in items], dtype=np.float64)
            agg[f"{metric}_mean"] = float(np.nanmean(vals))
            agg[f"{metric}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _read_existing_metrics(path: Path) -> Tuple[List[Dict[str, Any]], set]:
    if not path.exists():
        return [], set()
    rows: List[Dict[str, Any]] = []
    completed = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
            if row.get("status", "ok") == "ok":
                completed.add((row.get("setting"), str(row.get("seed"))))
    return rows, completed


def _load_manifest(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj, {}
    rows = obj.get("rows") or obj.get("entries") or []
    meta = {k: v for k, v in obj.items() if k not in {"rows", "entries"}}
    return rows, meta


def _compute_w_arrays(model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sess = model.enquire_session()
    qW_mean_tf, qW_var_tf = model.compute_qW(X, full_cov=False)
    W_mean_raw, W_var_raw = sess.run([qW_mean_tf, qW_var_tf])
    W_mean = np.transpose(W_mean_raw[..., 0], (2, 0, 1))
    W_var = np.transpose(W_var_raw[:, 0, :], (1, 0))
    return W_mean, W_var


def run_entry(args: argparse.Namespace, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    gpflow.reset_default_graph_and_session()
    seed = int(entry["seed"])
    np.random.seed(seed)
    tf.set_random_seed(seed)

    data_path = Path(entry["path"])
    if not data_path.is_absolute():
        data_path = Path(args.manifest).resolve().parent / data_path
    arr = np.load(str(data_path))
    X_train = np.asarray(arr["X_train"], dtype=np.float64)
    X_test = np.asarray(arr["X_test"], dtype=np.float64)
    y_train = np.asarray(arr["y_train_std"], dtype=np.float64).reshape(-1, 1)
    y_test = np.asarray(arr["y_test"], dtype=np.float64).reshape(-1)
    y_mean = float(np.asarray(arr["y_mean"]).reshape(-1)[0])
    y_scale = float(np.asarray(arr["y_scale"]).reshape(-1)[0])
    y_scale = y_scale if abs(y_scale) > 1e-12 else 1.0

    n_train, D = X_train.shape
    n_test = X_test.shape[0]
    Q = int(entry.get("Q") or args.Q)
    if Q <= 0:
        raise ValueError("Each manifest entry must provide Q, or pass --Q > 0.")
    m_u = min(int(args.m_u), n_train)
    m_v = min(int(args.m_v), n_train)
    print(
        "[{setting} seed={seed}] DeepVMGP Q={Q} D={D} N={N} T={T} m_u={mu} m_v={mv}".format(
            setting=entry["setting"],
            seed=seed,
            Q=Q,
            D=D,
            N=n_train,
            T=n_test,
            mu=m_u,
            mv=m_v,
        ),
        flush=True,
    )

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

    t1 = time.perf_counter()
    mean_s, var_s = model.predict_y(X_test)
    predict_sec = time.perf_counter() - t1
    mean_s = np.asarray(mean_s, dtype=np.float64).reshape(-1)
    var_s = np.maximum(np.asarray(var_s, dtype=np.float64).reshape(-1), 1e-12)
    mean = mean_s * y_scale + y_mean
    sigma = np.sqrt(var_s) * abs(y_scale)

    out_dir = Path(args.out_dir)
    pred_prefix = out_dir / "predictions" / f"{entry['setting']}_seed{seed}"
    pred_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(pred_prefix) + "_pred_mean.npy", mean)
    np.save(str(pred_prefix) + "_pred_sigma.npy", sigma)
    if int(args.save_w) != 0:
        W_train_mean, W_train_var = _compute_w_arrays(model, X_train)
        W_test_mean, W_test_var = _compute_w_arrays(model, X_test)
        np.save(str(pred_prefix) + "_W_train_mean.npy", W_train_mean)
        np.save(str(pred_prefix) + "_W_train_var.npy", W_train_var)
        np.save(str(pred_prefix) + "_W_test_mean.npy", W_test_mean)
        np.save(str(pred_prefix) + "_W_test_var.npy", W_test_var)

    base = {
        "benchmark": entry.get("benchmark", ""),
        "dataset": entry.get("dataset", entry.get("setting", "")),
        "train_size": int(entry.get("train_size", n_train)),
        "setting": entry["setting"],
        "seed": seed,
        "method": "deep_mahalanobis_gp",
        "N_train": int(n_train),
        "N_test": int(n_test),
        "Q": int(Q),
        "D": int(D),
        "train_sec": float(train_sec),
        "predict_sec": float(predict_sec),
        "m_u": int(m_u),
        "m_v": int(m_v),
        "steps1": int(args.steps1),
        "steps2": int(args.steps2),
        "lr": float(args.lr),
        "split_type": entry.get("split_type", ""),
        "standardize_x": entry.get("standardize_x", ""),
        "standardize_y_for_baseline": entry.get("standardize_y_for_baseline", ""),
        "y_standardized_for_dmgp": True,
        "status": "ok",
    }
    rows: List[Dict[str, Any]] = []
    for metric in _metric_rows_by_subset(y_test, mean, sigma, train_sec + predict_sec, arr):
        row = dict(base)
        row.update(metric)
        rows.append(row)
    return rows


def _write_readme(out_dir: Path, args: argparse.Namespace, aggregate_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# DeepMahalanobisGP NPZ Run",
        "",
        "Produced by `experiments/dmgp/run_deep_mahalanobis_gp_npz.py`.",
        "",
        "## Command",
        "",
        "```powershell",
        "conda run -n dmgp_tf1 python experiments/dmgp/run_deep_mahalanobis_gp_npz.py "
        f"--manifest {args.manifest} --out_dir {args.out_dir} "
        f"--steps1 {args.steps1} --steps2 {args.steps2} --m_u {args.m_u} --m_v {args.m_v}",
        "```",
        "",
        "## Aggregate",
        "",
        "| benchmark | dataset | train | subset | seeds | RMSE | CRPS | Cov90 | Width90 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row.get('benchmark', '')} | {row.get('dataset', '')} | {row.get('train_size', '')} | "
            f"{row.get('subset', 'all')} | {row.get('n_seeds', '')} | "
            f"{float(row['rmse_mean']):.4f} | {float(row['mean_crps_mean']):.4f} | "
            f"{float(row['coverage_90_mean']):.3f} | {float(row['mean_width_90_mean']):.4f} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run external DeepMahalanobisGP on exported NPZ benchmark splits.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--Q", type=int, default=0, help="Fallback latent dimension when not present in manifest.")
    p.add_argument("--m_u", type=int, default=50)
    p.add_argument("--m_v", type=int, default=25)
    p.add_argument("--steps1", type=int, default=300)
    p.add_argument("--steps2", type=int, default=700)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--initial_noise_var", type=float, default=0.01)
    p.add_argument("--save_w", type=int, default=0)
    p.add_argument("--resume", type=int, default=1)
    return p.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, manifest_meta = _load_manifest(Path(args.manifest))
    metrics_path = out_dir / "metrics.csv"
    rows: List[Dict[str, Any]]
    completed = set()
    if int(args.resume) != 0:
        rows, completed = _read_existing_metrics(metrics_path)
    else:
        rows = []

    for entry in entries:
        key = (str(entry["setting"]), str(entry["seed"]))
        if key in completed:
            print(f"[skip] {entry['setting']} seed={entry['seed']} already complete", flush=True)
            continue
        try:
            new_rows = run_entry(args, entry)
            rows.extend(new_rows)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {entry.get('setting')} seed={entry.get('seed')}: {exc}", flush=True)
            rows.append(
                {
                    "benchmark": entry.get("benchmark", ""),
                    "dataset": entry.get("dataset", entry.get("setting", "")),
                    "train_size": entry.get("train_size", ""),
                    "setting": entry.get("setting", ""),
                    "seed": entry.get("seed", ""),
                    "method": "deep_mahalanobis_gp",
                    "subset": "all",
                    "status": "error",
                    "error": repr(exc),
                }
            )
        _write_csv(metrics_path, rows)
        aggregate_rows = _aggregate(rows)
        _write_csv(out_dir / "aggregate.csv", aggregate_rows)
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "manifest_meta": manifest_meta,
                    "external_repo": str(EXTERNAL_ROOT),
                    "n_entries": len(entries),
                    "n_metric_rows": len(rows),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_readme(out_dir, args, aggregate_rows)

    aggregate_rows = _aggregate(rows)
    _write_csv(out_dir / "aggregate.csv", aggregate_rows)
    _write_readme(out_dir, args, aggregate_rows)
    print(f"Saved DeepMahalanobisGP artifacts -> {out_dir}", flush=True)
    return {"rows": rows, "aggregate": aggregate_rows}


if __name__ == "__main__":
    main()

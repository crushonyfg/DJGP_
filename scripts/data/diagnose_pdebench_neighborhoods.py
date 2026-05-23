from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose PDEBench-Burgers KNN support around near-threshold test points.")
    p.add_argument("--dataset-folder", default="data/processed/pdebench_burgers_shockjump")
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--threshold-delta", type=float, default=None)
    p.add_argument("--out_json", default="results/pdebench_inspection/neighborhood_support.json")
    p.add_argument("--out_csv", default="results/pdebench_inspection/neighborhood_support.csv")
    return p.parse_args()


def _as_numpy(x: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_dataset(folder: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(folder) / "dataset.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset.pkl at {path}")
    with path.open("rb") as f:
        return path, pickle.load(f)


def _split_meta_array(meta: dict[str, Any], key: str, n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    if key not in meta:
        raise KeyError(f"dataset args missing {key!r}")
    arr = _as_numpy(meta[key]).reshape(-1)
    train_idx = np.asarray(meta.get("train_indices", []), dtype=int)
    test_idx = np.asarray(meta.get("test_indices", []), dtype=int)
    if len(arr) == n_train + n_test and len(train_idx) == n_train and len(test_idx) == n_test:
        return arr[train_idx], arr[test_idx]
    if len(arr) == n_train + n_test:
        return arr[:n_train], arr[n_train:]
    if len(arr) == n_test:
        raise ValueError(f"metadata {key!r} only has test values; train diagnostics need train values too.")
    raise ValueError(f"Cannot map metadata {key!r} of length {len(arr)} to train/test ({n_train}, {n_test}).")


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return {name: float("nan") for name in ("mean", "median", "p25", "p75", "min", "max")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _subset_row(name: str, mask: np.ndarray, per_point: dict[str, np.ndarray]) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    row: dict[str, Any] = {"subset": name, "n_test": int(mask.sum())}
    for key, values in per_point.items():
        stats = _summary(values[mask])
        for stat_name, stat_value in stats.items():
            row[f"{key}_{stat_name}"] = stat_value
    return row


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def main() -> dict[str, Any]:
    args = parse_args()
    dataset_path, dataset = _load_dataset(args.dataset_folder)
    meta = dict(dataset.get("args", {}) or {})

    X_train = _as_numpy(dataset["X_train"]).astype(np.float32)
    X_test = _as_numpy(dataset["X_test"]).astype(np.float32)
    n_train, n_test = len(X_train), len(X_test)
    k = min(args.k, n_train)
    threshold = float(meta.get("jump_threshold", meta.get("threshold", 0.5)))
    delta = float(args.threshold_delta if args.threshold_delta is not None else meta.get("near_threshold_delta", 0.05))

    shock_train, shock_test = _split_meta_array(meta, "shock_location", n_train, n_test)
    dist_train = np.abs(shock_train.astype(np.float64) - threshold)
    dist_test = np.abs(shock_test.astype(np.float64) - threshold)
    train_near = dist_train <= delta + 1e-7
    test_near = dist_test <= delta + 1e-7
    test_far = ~test_near

    raw_knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    raw_knn.fit(X_train)
    raw_dist, raw_idx = raw_knn.kneighbors(X_test, return_distance=True)

    raw_neighbor_shock = shock_train[raw_idx]
    raw_neighbor_near = train_near[raw_idx]
    raw_neighbor_left = raw_neighbor_shock < threshold
    raw_neighbor_right = raw_neighbor_shock > threshold
    shock_gap = np.abs(raw_neighbor_shock - shock_test[:, None])

    oracle_knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    oracle_knn.fit(shock_train.reshape(-1, 1))
    oracle_dist, oracle_idx = oracle_knn.kneighbors(shock_test.reshape(-1, 1), return_distance=True)
    oracle_neighbor_near = train_near[oracle_idx]
    oracle_neighbor_left = shock_train[oracle_idx] < threshold
    oracle_neighbor_right = shock_train[oracle_idx] > threshold

    per_point = {
        "raw_d1": raw_dist[:, 0],
        f"raw_d{k}": raw_dist[:, -1],
        "raw_knn_near_count": raw_neighbor_near.sum(axis=1),
        "raw_knn_left_count": raw_neighbor_left.sum(axis=1),
        "raw_knn_right_count": raw_neighbor_right.sum(axis=1),
        "raw_knn_median_shock_absdiff": np.median(shock_gap, axis=1),
        f"oracle_shock_d{k}": oracle_dist[:, -1],
        "oracle_knn_near_count": oracle_neighbor_near.sum(axis=1),
        "oracle_knn_left_count": oracle_neighbor_left.sum(axis=1),
        "oracle_knn_right_count": oracle_neighbor_right.sum(axis=1),
    }

    rows = [
        _subset_row("all", np.ones(n_test, dtype=bool), per_point),
        _subset_row("near_threshold", test_near, per_point),
        _subset_row("far_from_threshold", test_far, per_point),
    ]
    result = {
        "dataset_path": str(dataset_path),
        "k": int(k),
        "threshold": threshold,
        "threshold_delta": delta,
        "counts": {
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_train_near": int(train_near.sum()),
            "n_test_near": int(test_near.sum()),
            "n_test_far": int(test_far.sum()),
        },
        "rows": rows,
    }

    for row in rows:
        print(
            f"{row['subset']:18s} n={row['n_test']:4d} "
            f"median raw_d{k}={row[f'raw_d{k}_median']:.4g} "
            f"median #near={row['raw_knn_near_count_median']:.1f} "
            f"median #left={row['raw_knn_left_count_median']:.1f} "
            f"median #right={row['raw_knn_right_count_median']:.1f} "
            f"median shock_gap={row['raw_knn_median_shock_absdiff_median']:.4g} "
            f"oracle median #near={row['oracle_knn_near_count_median']:.1f}",
        )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(_sanitize(result), f, indent=2)
        print(f"Saved JSON: {out}")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {out}")

    return result


if __name__ == "__main__":
    main()

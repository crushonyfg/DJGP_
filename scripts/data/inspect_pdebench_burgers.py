from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATASET_FOLDER = Path("data") / "processed" / "pdebench_burgers_shockjump"
DEFAULT_OUT = Path("results") / "pdebench_inspection" / "summary.json"
DEFAULT_PLOT_DIR = Path("results") / "pdebench_inspection"
THRESHOLDS = (0.02, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a processed PDEBench-Burgers scalar-QoI dataset.pkl."
    )
    parser.add_argument("--dataset-folder", default=str(DEFAULT_DATASET_FOLDER))
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON summary path. Use '' to skip saving JSON.")
    parser.add_argument("--save-plots", action="store_true", help="Save simple histograms with matplotlib.")
    parser.add_argument("--plot-dir", default=str(DEFAULT_PLOT_DIR))
    parser.add_argument("--bins", type=int, default=20)
    return parser.parse_args()


def _as_numpy(x: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flatten(x: Any) -> np.ndarray:
    return _as_numpy(x).astype(np.float64).reshape(-1)


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def load_dataset(folder: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(folder) / "dataset.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset.pkl at {path}")
    with path.open("rb") as f:
        return path, pickle.load(f)


def describe_array(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def histogram_summary(x: np.ndarray, *, bins: int) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return {"counts": [], "bin_edges": []}
    counts, edges = np.histogram(x, bins=bins)
    return {"counts": counts.astype(int).tolist(), "bin_edges": edges.astype(float).tolist()}


def _meta_array(meta: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in meta or meta[key] is None:
        return None
    arr = _flatten(meta[key])
    return arr if len(arr) else None


def _maybe_index(arr: np.ndarray | None, indices: Any, expected_len: int) -> np.ndarray | None:
    if arr is None:
        return None
    if len(arr) == expected_len:
        return arr
    if indices is not None:
        idx = np.asarray(indices, dtype=int).reshape(-1)
        if len(idx) == expected_len and len(arr) > int(idx.max(initial=-1)):
            return arr[idx]
    return None


def split_meta_arrays(meta: dict[str, Any], key: str, n_train: int, n_test: int) -> dict[str, np.ndarray]:
    arr = _meta_array(meta, key)
    if arr is None:
        return {}
    train_idx = meta.get("train_indices")
    test_idx = meta.get("test_indices")
    out: dict[str, np.ndarray] = {}
    train_arr = _maybe_index(arr, train_idx, n_train)
    test_arr = _maybe_index(arr, test_idx, n_test)
    if len(arr) == n_train + n_test:
        out["all"] = arr
    elif len(arr) == n_test:
        out["test"] = arr
    elif len(arr) == n_train:
        out["train"] = arr
    else:
        out["raw"] = arr
    if train_arr is not None:
        out["train"] = train_arr
    if test_arr is not None:
        out["test"] = test_arr
    return out


def threshold_counts(distance: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, dict[str, float | int]]:
    distance = np.asarray(distance, dtype=np.float64).reshape(-1)
    out = {}
    for threshold in thresholds:
        count = int(np.sum(distance <= threshold + 1e-7))
        pct = float(100.0 * count / len(distance)) if len(distance) else float("nan")
        out[str(threshold)] = {"count": count, "percentage": pct}
    return out


def build_summary(dataset_path: Path, dataset: dict[str, Any], *, bins: int) -> dict[str, Any]:
    X_train = _as_numpy(dataset["X_train"])
    X_test = _as_numpy(dataset["X_test"])
    y_train = _flatten(dataset["Y_train"])
    y_test = _flatten(dataset["Y_test"])
    meta = dict(dataset.get("args", {}) or {})

    y_all = np.concatenate([y_train, y_test])
    summary: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "Y_train": list(_as_numpy(dataset["Y_train"]).shape),
            "Y_test": list(_as_numpy(dataset["Y_test"]).shape),
        },
        "qoi": meta.get("qoi", "unknown"),
        "metadata": _sanitize(meta),
        "y": {
            "train": describe_array(y_train),
            "test": describe_array(y_test),
            "all": describe_array(y_all),
            "histogram": histogram_summary(y_all, bins=bins),
        },
        "standardization": {
            "x_standardized": meta.get("x_standardized"),
            "y_standardized": meta.get("y_standardized"),
            "y_mean": meta.get("y_mean"),
            "y_std": meta.get("y_std"),
        },
        "split": {
            "split_seed": meta.get("split_seed"),
            "test_size": meta.get("test_size"),
            "n_train_indices": len(meta.get("train_indices", [])),
            "n_test_indices": len(meta.get("test_indices", [])),
        },
    }

    shock_by_split = split_meta_arrays(meta, "shock_location", len(y_train), len(y_test))
    if shock_by_split:
        summary["shock_location"] = {
            split: {
                "stats": describe_array(values),
                "histogram": histogram_summary(values, bins=bins),
            }
            for split, values in shock_by_split.items()
        }

    distance_by_split = split_meta_arrays(meta, "threshold_distance", len(y_train), len(y_test))
    if not distance_by_split and shock_by_split and "jump_threshold" in meta:
        threshold = float(meta["jump_threshold"])
        distance_by_split = {
            split: np.abs(values - threshold) for split, values in shock_by_split.items()
        }
        summary["threshold_distance_source"] = "abs(shock_location - jump_threshold)"
    elif distance_by_split:
        summary["threshold_distance_source"] = "metadata.threshold_distance"

    if distance_by_split:
        summary["near_threshold_counts"] = {
            split: threshold_counts(values, THRESHOLDS)
            for split, values in distance_by_split.items()
        }

    return summary


def print_report(summary: dict[str, Any]) -> None:
    print(f"Dataset: {summary['dataset_path']}")
    print(f"QoI: {summary['qoi']}")
    print("Shapes:")
    for key, shape in summary["shapes"].items():
        print(f"  {key}: {shape}")

    print("Y statistics:")
    for split in ("train", "test", "all"):
        stats = summary["y"][split]
        print(
            f"  {split}: n={stats['n']} mean={stats['mean']:.6g} std={stats['std']:.6g} "
            f"min={stats['min']:.6g} max={stats['max']:.6g}"
        )
    print(f"Y histogram counts: {summary['y']['histogram']['counts']}")
    print(f"Y histogram edges: {summary['y']['histogram']['bin_edges']}")

    print("Split / standardization:")
    split = summary["split"]
    standardization = summary["standardization"]
    print(f"  split_seed={split.get('split_seed')} test_size={split.get('test_size')}")
    print(
        f"  x_standardized={standardization.get('x_standardized')} "
        f"y_standardized={standardization.get('y_standardized')} "
        f"y_mean={standardization.get('y_mean')} y_std={standardization.get('y_std')}"
    )

    print("Selected metadata:")
    for key in (
        "source",
        "qoi",
        "N",
        "T",
        "D",
        "t0",
        "t_final",
        "probe_idx",
        "jump_threshold",
        "jump_size",
        "noise_std",
    ):
        if key in summary["metadata"]:
            print(f"  {key}: {summary['metadata'][key]}")

    if "shock_location" in summary:
        print("Shock location histograms:")
        for split, item in summary["shock_location"].items():
            stats = item["stats"]
            print(
                f"  {split}: n={stats['n']} mean={stats['mean']:.6g} std={stats['std']:.6g} "
                f"min={stats['min']:.6g} max={stats['max']:.6g}"
            )
            print(f"    counts={item['histogram']['counts']}")

    if "near_threshold_counts" in summary:
        print(f"Near-threshold counts ({summary.get('threshold_distance_source')}):")
        for split, counts in summary["near_threshold_counts"].items():
            print(f"  {split}:")
            for threshold, item in counts.items():
                print(f"    <= {threshold}: {item['count']} ({item['percentage']:.2f}%)")


def save_plots(summary: dict[str, Any], plot_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    y_hist = summary["y"]["histogram"]
    if y_hist["counts"]:
        edges = np.asarray(y_hist["bin_edges"], dtype=float)
        counts = np.asarray(y_hist["counts"], dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", edgecolor="black")
        ax.set_title("PDEBench-Burgers QoI histogram")
        ax.set_xlabel("standardized y")
        ax.set_ylabel("count")
        path = plot_dir / "y_histogram.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))

    if "shock_location" in summary:
        for split, item in summary["shock_location"].items():
            hist = item["histogram"]
            if not hist["counts"]:
                continue
            edges = np.asarray(hist["bin_edges"], dtype=float)
            counts = np.asarray(hist["counts"], dtype=float)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", edgecolor="black")
            ax.set_title(f"Shock location histogram ({split})")
            ax.set_xlabel("shock location")
            ax.set_ylabel("count")
            path = plot_dir / f"shock_location_histogram_{split}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved.append(str(path))

    return saved


def main() -> dict[str, Any]:
    args = parse_args()
    dataset_path, dataset = load_dataset(args.dataset_folder)
    summary = build_summary(dataset_path, dataset, bins=args.bins)
    print_report(summary)

    if args.save_plots:
        try:
            summary["plots"] = save_plots(summary, Path(args.plot_dir))
            print("Saved plots:")
            for path in summary["plots"]:
                print(f"  {path}")
        except ImportError as exc:
            raise SystemExit("Plot saving requires matplotlib. Install it or omit --save-plots.") from exc

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(_sanitize(summary), f, indent=2)
        print(f"Saved summary JSON: {out_path}")

    return summary


if __name__ == "__main__":
    main()

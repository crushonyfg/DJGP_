"""Export PDEBench-Burgers train-size splits for DeepMahalanobisGP.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/pdebench/export_burgers_dmgp_splits.py ^
        --dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000 ^
        --train_sizes 200,500,1000 --num_exp 5 ^
        --out_dir experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517

    python experiments/pdebench/export_burgers_dmgp_splits.py ^
        --dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000 ^
        --train_sizes 80 --num_exp 1 --max_anchors 40 ^
        --out_dir experiments/pdebench/dmgp_burgers_smoke

The exported files are consumed by
``experiments/dmgp/run_deep_mahalanobis_gp_npz.py`` in the ``dmgp_tf1`` env.
The training subsampling rule matches
``experiments/pdebench/compare_burgers_train_sizes.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from experiments.pdebench.compare_burgers_train_sizes import _subsample_train  # noqa: E402
from experiments.pdebench.compare_pdebench_burgers import (  # noqa: E402
    _dataset_arrays,
    _threshold_masks,
    load_dataset,
)


def _parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _export_one(
    *,
    out_dir: Path,
    dataset: Dict[str, Any],
    train_size: int,
    seed: int,
    max_anchors: int | None,
    Q: int,
) -> Dict[str, Any]:
    X_train_full, y_train_full, X_test_full, y_test_full = _dataset_arrays(dataset, max_anchors=None)
    X_test, y_test = X_test_full, y_test_full
    if max_anchors is not None:
        k = min(int(max_anchors), len(X_test_full))
        X_test, y_test = X_test_full[:k], y_test_full[:k]
    X_train, y_train, train_idx = _subsample_train(X_train_full, y_train_full, train_size, seed)
    masks = _threshold_masks(dataset, len(X_test_full), len(X_test))

    y_mean = float(np.mean(y_train))
    y_scale = float(np.std(y_train))
    if not np.isfinite(y_scale) or y_scale < 1e-8:
        y_scale = 1.0
    y_train_std = (np.asarray(y_train, dtype=np.float64).reshape(-1) - y_mean) / y_scale

    setting = f"pdebench_burgers_train{train_size}"
    rel_path = Path("data") / f"{setting}_seed{seed}.npz"
    npz_path = out_dir / rel_path
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "X_train": np.asarray(X_train, dtype=np.float64),
        "X_test": np.asarray(X_test, dtype=np.float64),
        "y_train_std": y_train_std.astype(np.float64),
        "y_test": np.asarray(y_test, dtype=np.float64).reshape(-1),
        "y_mean": np.asarray([y_mean], dtype=np.float64),
        "y_scale": np.asarray([y_scale], dtype=np.float64),
    }
    if "threshold_distance" in masks and "near_threshold_delta" in masks:
        payload["threshold_distance_test"] = np.asarray(masks["threshold_distance"], dtype=np.float64).reshape(-1)
        payload["near_threshold_delta"] = np.asarray([float(masks["near_threshold_delta"])], dtype=np.float64)
    np.savez_compressed(npz_path, **payload)

    return {
        "benchmark": "pdebench_burgers",
        "dataset": "PDEBench Burgers ShockJump",
        "setting": setting,
        "seed": int(seed),
        "train_size": int(train_size),
        "Q": int(Q),
        "D": int(X_train.shape[1]),
        "N_train": int(X_train.shape[0]),
        "N_test": int(X_test.shape[0]),
        "path": str(rel_path).replace("\\", "/"),
        "split_type": "processed_fixed_test_train_subsample",
        "standardize_x": True,
        "standardize_y_for_baseline": True,
        "y_standardized_for_dmgp": True,
        "train_indices_within_processed_train": train_idx.tolist(),
        "near_threshold_delta": float(masks.get("near_threshold_delta", float("nan"))),
        "n_near_threshold": int(np.asarray(masks.get("near_threshold", []), dtype=bool).sum())
        if "near_threshold" in masks
        else 0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PDEBench Burgers train-size splits for DeepMahalanobisGP.")
    p.add_argument("--dataset-folder", default="data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000")
    p.add_argument("--train_sizes", default="200,500,1000")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_anchors", type=int, default=None)
    p.add_argument("--Q", type=int, default=3)
    p.add_argument("--out_dir", default="experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517")
    return p.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset_folder)
    rows: List[Dict[str, Any]] = []
    for train_size in _parse_ints(args.train_sizes):
        for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
            print(f"export PDEBench train={train_size} seed={seed}", flush=True)
            rows.append(
                _export_one(
                    out_dir=out_dir,
                    dataset=dataset,
                    train_size=train_size,
                    seed=seed,
                    max_anchors=args.max_anchors,
                    Q=args.Q,
                )
            )

    manifest = {
        "benchmark": "pdebench_burgers",
        "dataset_folder": str(Path(args.dataset_folder)),
        "args": vars(args),
        "rows": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = [
        "# PDEBench Burgers DeepMahalanobisGP Exports",
        "",
        "Produced by `experiments/pdebench/export_burgers_dmgp_splits.py`.",
        "",
        f"Rows: {len(rows)}",
        "",
        "The processed PDEBench dataset already contains train-only standardized X/Y from the fixed split; DMGP additionally trains on per-subsample standardized y and reports metrics on the processed response scale.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote manifest -> {out_dir / 'manifest.json'}", flush=True)
    return manifest


if __name__ == "__main__":
    main()

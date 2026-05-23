"""Run PDEBench-Burgers baselines for train-size sweeps.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/pdebench/compare_burgers_train_sizes.py ^
        --dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000 ^
        --train_sizes 200,500,1000 --num_exp 5 --Q 3 --neighbors 25 ^
        --lmjgp_steps 300 --out_dir experiments/pdebench/burgers_train200_500_1000_xgb_dkl_lmjgp_5seeds_20260517

    python experiments/pdebench/compare_burgers_train_sizes.py ^
        --dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000 ^
        --train_sizes 80 --num_exp 1 --max_anchors 40 --lmjgp_steps 5 ^
        --xgb_boots 3 --dkl_epochs 3 --out_dir experiments/pdebench/burgers_train_sizes_smoke

This runner reuses the processed split from ``compare_pdebench_burgers.py`` and
subsamples the standardized training pool with a deterministic per-seed
permutation.  Test points are left unchanged unless ``--max_anchors`` is set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.baselines import run_dkl_svgp_regression, run_xgboost_bootstrap_regression  # noqa: E402
from experiments.pdebench.compare_pdebench_burgers import (  # noqa: E402
    RESULT_SPLITS,
    _dataset_arrays,
    _pack_split_metrics,
    _run_lmjgp_predict_vi,
    _seed_all,
    _threshold_masks,
    load_dataset,
)


RESULT_KEYS = (
    "rmse",
    "mean_crps",
    "wall_sec",
    "coverage_90",
    "mean_width_90",
    "coverage_95",
    "mean_width_95",
)


def _parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _subsample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(int(train_size), int(X_train.shape[0]))
    rng = np.random.default_rng(int(seed) + 500)
    idx = rng.permutation(X_train.shape[0])[:n]
    return X_train[idx], y_train[idx], idx


def _metric_rows(
    *,
    train_size: int,
    seed: int,
    method: str,
    packed: Dict[str, Any],
    n_train: int,
    n_test: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for subset in RESULT_SPLITS:
        if subset not in packed:
            continue
        values = packed[subset]
        row = {
            "benchmark": "pdebench_burgers",
            "dataset": "PDEBench Burgers ShockJump",
            "train_size": int(train_size),
            "seed": int(seed),
            "method": method,
            "subset": subset,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_eval": int(packed.get(f"n_{subset}", n_test)),
        }
        for i, key in enumerate(RESULT_KEYS):
            row[key] = float(values[i])
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    preferred = [
        "benchmark",
        "dataset",
        "train_size",
        "seed",
        "method",
        "subset",
        "n_train",
        "n_test",
        "n_eval",
        *RESULT_KEYS,
    ]
    keys: List[str] = []
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
    groups: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["train_size"]), str(row["method"]), str(row["subset"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (train_size, method, subset), items in sorted(groups.items()):
        agg: Dict[str, Any] = {
            "benchmark": "pdebench_burgers",
            "dataset": "PDEBench Burgers ShockJump",
            "train_size": int(train_size),
            "method": method,
            "subset": subset,
            "n_seeds": len(items),
            "seeds": ",".join(str(int(r["seed"])) for r in items),
            "source": "compare_burgers_train_sizes",
        }
        for key in RESULT_KEYS:
            vals = np.asarray([float(r[key]) for r in items], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _write_readme(out_dir: Path, args: argparse.Namespace, agg_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# PDEBench Burgers Train-Size Baselines",
        "",
        "Produced by `experiments/pdebench/compare_burgers_train_sizes.py`.",
        "",
        "## Aggregate",
        "",
        "| train | subset | method | seeds | RMSE | CRPS | Cov90 | Width90 |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in agg_rows:
        lines.append(
            f"| {row['train_size']} | {row['subset']} | {row['method']} | {row['n_seeds']} | "
            f"{float(row['rmse_mean']):.4f} | {float(row['mean_crps_mean']):.4f} | "
            f"{float(row['coverage_90_mean']):.3f} | {float(row['mean_width_90_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Command",
            "",
            "```powershell",
            "python experiments/pdebench/compare_burgers_train_sizes.py "
            f"--dataset-folder {args.dataset_folder} --train_sizes {args.train_sizes} "
            f"--num_exp {args.num_exp} --Q {args.Q} --neighbors {args.n} "
            f"--lmjgp_steps {args.lmjgp_steps} --out_dir {args.out_dir}",
            "```",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_one(args: argparse.Namespace, train_size: int, seed: int, device: torch.device) -> List[Dict[str, Any]]:
    _seed_all(seed)
    dataset = load_dataset(args.dataset_folder)
    X_train_full, y_train_full, X_test_full, y_test_full = _dataset_arrays(dataset, max_anchors=None)
    X_test, y_test = X_test_full, y_test_full
    if args.max_anchors is not None:
        k = min(int(args.max_anchors), len(X_test_full))
        X_test, y_test = X_test_full[:k], y_test_full[:k]
    masks = _threshold_masks(dataset, len(X_test_full), len(X_test))
    X_train, y_train, _ = _subsample_train(X_train_full, y_train_full, train_size, seed)

    rows: List[Dict[str, Any]] = []
    if "xgb_bootstrap" in args.methods:
        rmse, crps, elapsed, details = run_xgboost_bootstrap_regression(
            X_train,
            y_train,
            X_test,
            y_test,
            n_bootstrap=int(args.xgb_boots),
            random_state=seed,
            xgb_params={"n_estimators": 80, "max_depth": 4},
        )
        packed = _pack_split_metrics(y_test, details["mu"], details["sigma"], elapsed, masks)
        rows.extend(
            _metric_rows(
                train_size=train_size,
                seed=seed,
                method="xgb_bootstrap",
                packed=packed,
                n_train=len(X_train),
                n_test=len(X_test),
            )
        )

    if "dkl_svgp" in args.methods:
        rmse, crps, elapsed, mu, sigma = run_dkl_svgp_regression(
            X_train,
            y_train,
            X_test,
            y_test,
            device=device,
            epochs=int(args.dkl_epochs),
            num_inducing=min(64, len(X_train)),
            seed=seed,
        )
        packed = _pack_split_metrics(y_test, mu, sigma, elapsed, masks)
        rows.extend(
            _metric_rows(
                train_size=train_size,
                seed=seed,
                method="dkl_svgp",
                packed=packed,
                n_train=len(X_train),
                n_test=len(X_test),
            )
        )

    if "lmjgp_predict_vi" in args.methods:
        lmjgp = _run_lmjgp_predict_vi(X_train, y_train, X_test, y_test, args=args, device=device)
        packed = _pack_split_metrics(y_test, lmjgp["mu"], lmjgp["sigma"], lmjgp["elapsed"], masks)
        rows.extend(
            _metric_rows(
                train_size=train_size,
                seed=seed,
                method="lmjgp_predict_vi",
                packed=packed,
                n_train=len(X_train),
                n_test=len(X_test),
            )
        )

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDEBench Burgers train-size sweep for XGB, DKL, and LMJGP.")
    p.add_argument("--dataset-folder", default="data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000")
    p.add_argument("--train_sizes", default="200,500,1000")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_anchors", type=int, default=None)
    p.add_argument("--methods", default="xgb_bootstrap,dkl_svgp,lmjgp_predict_vi")
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--Q", type=int, default=3)
    p.add_argument("--n", "--neighbors", dest="n", type=int, default=25)
    p.add_argument("--n_pls", type=int, default=3)
    p.add_argument("--xgb_boots", type=int, default=10)
    p.add_argument("--dkl_epochs", type=int, default=30)
    p.add_argument("--out_dir", default="experiments/pdebench/burgers_train200_500_1000_xgb_dkl_lmjgp_5seeds_20260517")
    return p.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    args.methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = sorted(set(args.methods) - {"xgb_bootstrap", "dkl_svgp", "lmjgp_predict_vi"})
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"args": vars(args), "rows": []}

    for train_size in _parse_ints(args.train_sizes):
        for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
            print(f"PDEBench train={train_size} seed={seed}", flush=True)
            rows = run_one(args, train_size, seed, device)
            all_rows.extend(rows)
            summary["rows"].extend(rows)
            agg_rows = _aggregate(all_rows)
            _write_csv(out_dir / "metrics.csv", all_rows)
            _write_csv(out_dir / "aggregate.csv", agg_rows)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            _write_readme(out_dir, args, agg_rows)
            print(f"partial artifacts saved to {out_dir}", flush=True)

    agg_rows = _aggregate(all_rows)
    _write_csv(out_dir / "aggregate.csv", agg_rows)
    _write_readme(out_dir, args, agg_rows)
    print(f"Saved artifacts -> {out_dir}", flush=True)
    return {"rows": all_rows, "aggregate": agg_rows}


if __name__ == "__main__":
    main()

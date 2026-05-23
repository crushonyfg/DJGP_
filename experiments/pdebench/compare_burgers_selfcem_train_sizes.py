"""Run non-MC self-CEM on PDEBench-Burgers train-size sweeps.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/pdebench/compare_burgers_selfcem_train_sizes.py ^
        --dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000 ^
        --train_sizes 200,500,1000 --num_exp 5 --Q 3 --neighbors 25 ^
        --out_dir experiments/pdebench/burgers_train200_500_1000_selfcem_5seeds_20260519

    python experiments/pdebench/compare_burgers_selfcem_train_sizes.py ^
        --train_sizes 80 --num_exp 1 --max_anchors 40 --hyper_steps 2 --gate_steps 2 ^
        --out_dir experiments/pdebench/burgers_selfcem_smoke

This runner uses the same processed split and deterministic train subsampling as
``compare_burgers_train_sizes.py``.  It reports the same all / near-threshold /
far-from-threshold subsets, but the method is the non-MC uncertain-W
``self_cem2_nstd0.1_kcov0.1`` head used in the synthetic and UCI comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics import sanitize_for_json  # noqa: E402
from djgp.projections import FixedLabelGateConfig, FixedLabelHyperoptConfig, SelfCEMConfig  # noqa: E402
from djgp.projections.uncertain_w_jgp import run_uncertain_w_self_cem, uncertain_w_cem_predict_from_labels  # noqa: E402
from experiments.pdebench.compare_burgers_train_sizes import (  # noqa: E402
    _metric_rows,
    _parse_ints,
    _subsample_train,
    _write_csv,
)
from experiments.pdebench.compare_pdebench_burgers import (  # noqa: E402
    _dataset_arrays,
    _pack_split_metrics,
    _seed_all,
    _threshold_masks,
    load_dataset,
)
from experiments.synthetic import compare_uncertain_w_cem_l2_lh as uw  # noqa: E402


METHOD_NAME = "self_cem2_nstd0.1_kcov0.1"
RESULT_KEYS = (
    "rmse",
    "mean_crps",
    "wall_sec",
    "coverage_90",
    "mean_width_90",
    "coverage_95",
    "mean_width_95",
)


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[int, str, str], List[Dict[str, Any]]] = {}
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
            "source": "compare_burgers_selfcem_train_sizes",
        }
        for key in RESULT_KEYS:
            vals = np.asarray([float(r[key]) for r in items], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _write_readme(out_dir: Path, args: argparse.Namespace, agg_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# PDEBench Burgers Self-CEM Train-Size Sweep",
        "",
        "Produced by `experiments/pdebench/compare_burgers_selfcem_train_sizes.py`.",
        "",
        "Method: non-MC `self_cem2_nstd0.1_kcov0.1`.",
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
            "python experiments/pdebench/compare_burgers_selfcem_train_sizes.py "
            f"--dataset-folder {args.dataset_folder} --train_sizes {args.train_sizes} "
            f"--num_exp {args.num_exp} --Q {args.Q} --neighbors {args.n} --out_dir {args.out_dir}",
            "```",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _run_selfcem_predict(
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    spec = uw._parse_variant(METHOD_NAME)
    X_train_t = torch.as_tensor(np.asarray(X_train, dtype=np.float64), device=device, dtype=torch.float64)
    X_test_t = torch.as_tensor(np.asarray(X_test, dtype=np.float64), device=device, dtype=torch.float64)
    y_train_t = torch.as_tensor(np.asarray(y_train, dtype=np.float64).reshape(-1), device=device, dtype=torch.float64)
    Q = int(args.Q)
    n = int(args.n)

    W0_np = uw._projection_W0(np.asarray(X_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64), Q, "pls")
    W0 = torch.as_tensor(W0_np, device=device, dtype=torch.float64)
    X_neighbors, y_neighbors, _neighbor_idx = uw._neighbor_stack(
        X_test_t,
        X_train_t,
        y_train_t,
        n_neighbors=n,
    )
    w_cov_lengthscale = float(args.w_cov_lengthscale) if args.w_cov_lengthscale is not None else uw._median_distance(X_train_t)
    w_moments = uw._make_w_moments(
        mode=spec.mode,
        W0=W0,
        X_anchor=X_test_t,
        X_neighbors=X_neighbors,
        w_var=float(args.w_var),
        w_cov_lengthscale=w_cov_lengthscale,
    )
    cem_state, cem_diag = uw._run_meanw_cem_init(X_test_t, X_neighbors, y_neighbors, W0)
    init_state = uw._init_state_for_variant(
        spec,
        X_anchor=X_test_t,
        X_neighbors=X_neighbors,
        y_neighbors=y_neighbors,
        W0=W0,
        cem_state=cem_state,
        min_inliers=int(args.min_inliers),
        response_seed_frac=float(args.response_seed_frac),
        response_inlier_frac=float(args.response_inlier_frac),
    )
    local_noise_floor = max(float(args.min_noise_var), float(spec.local_noise_std_floor or 0.0) ** 2)
    t0 = time.perf_counter()
    state = run_uncertain_w_self_cem(
        X_test_t,
        X_neighbors,
        y_neighbors,
        init_state["labels"].bool(),
        mode=spec.mode,
        init_lengthscale=init_state["lengthscale"],
        init_signal_var=init_state["signal_var"],
        init_noise_var=init_state["noise_var"],
        gate_init=init_state["gate"],
        config=SelfCEMConfig(
            cem_updates=int(spec.cem_updates),
            hyperopt=FixedLabelHyperoptConfig(
                steps=int(args.hyper_steps),
                lr=float(args.hyper_lr),
                min_noise_var=float(local_noise_floor),
                max_noise_var=float(args.max_noise_var),
                min_signal_var=float(args.min_signal_var),
                max_signal_var=float(args.max_signal_var),
                min_lengthscale=float(args.min_lengthscale),
                max_lengthscale=float(args.max_lengthscale),
            ),
            gate=FixedLabelGateConfig(
                steps=int(args.gate_steps),
                lr=float(args.gate_lr),
                l2=float(args.gate_l2),
                max_norm=float(args.gate_max_norm),
                gh_points=int(args.gh_points),
            ),
            outlier_mode="background_normal",
            outlier_sigma_mult=float(args.outlier_sigma_mult),
            background_scale_mult=float(args.background_scale_mult),
            min_inliers=int(args.min_inliers),
            max_inlier_frac=float(args.max_inlier_frac),
            refit_after_update=True,
        ),
        **w_moments,
    )
    train_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = uncertain_w_cem_predict_from_labels(
        X_test_t,
        X_neighbors,
        y_neighbors,
        state["labels"],
        mode=spec.mode,
        lengthscale=state["lengthscale"],
        signal_var=state["signal_var"],
        noise_var=state["noise_var"],
        mean_const=state["mean_const"],
        correction_weight=float(spec.kcov),
        min_inliers=int(args.min_inliers),
        **w_moments,
    )
    pred_sec = time.perf_counter() - t0
    mu = pred.mu.detach().cpu().numpy().reshape(-1)
    sigma = pred.sigma.detach().cpu().numpy().reshape(-1)
    diag = {
        "train_sec": float(train_sec),
        "pred_sec": float(pred_sec),
        "init_inlier_rate": float(init_state["labels"].to(dtype=torch.float64).mean().cpu().item()),
        "final_inlier_rate": float(state["labels"].to(dtype=torch.float64).mean().cpu().item()),
        "label_change_rate": float((state["labels"].bool() != init_state["labels"].bool()).to(dtype=torch.float64).mean().cpu().item()),
        "mean_noise_var": float(state["noise_var"].detach().mean().cpu().item()),
    }
    diag.update({f"init_{k}": v for k, v in cem_diag.items()})
    return mu, sigma, train_sec + pred_sec, sanitize_for_json(diag)


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
    mu, sigma, elapsed, diag = _run_selfcem_predict(args, X_train, y_train, X_test, seed=seed, device=device)
    packed = _pack_split_metrics(y_test, mu, sigma, elapsed, masks)
    rows = _metric_rows(
        train_size=train_size,
        seed=seed,
        method=METHOD_NAME,
        packed=packed,
        n_train=len(X_train),
        n_test=len(X_test),
    )
    for row in rows:
        row.update(diag)
        row["Q"] = int(args.Q)
        row["n_neighbors"] = int(args.n)
        row["status"] = "ok"
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDEBench Burgers non-MC self-CEM train-size sweep.")
    p.add_argument("--dataset-folder", default="data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000")
    p.add_argument("--train_sizes", default="200,500,1000")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_anchors", type=int, default=None)
    p.add_argument("--Q", type=int, default=3)
    p.add_argument("--n", "--neighbors", dest="n", type=int, default=25)
    p.add_argument("--w_var", type=float, default=0.02)
    p.add_argument("--w_cov_lengthscale", type=float, default=None)
    p.add_argument("--hyper_steps", type=int, default=30)
    p.add_argument("--hyper_lr", type=float, default=0.04)
    p.add_argument("--gate_steps", type=int, default=60)
    p.add_argument("--gate_lr", type=float, default=0.05)
    p.add_argument("--gate_l2", type=float, default=1e-3)
    p.add_argument("--gate_max_norm", type=float, default=8.0)
    p.add_argument("--gh_points", type=int, default=20)
    p.add_argument("--min_inliers", type=int, default=2)
    p.add_argument("--max_inlier_frac", type=float, default=0.95)
    p.add_argument("--outlier_sigma_mult", type=float, default=2.5)
    p.add_argument("--background_scale_mult", type=float, default=3.0)
    p.add_argument("--min_noise_var", type=float, default=1e-5)
    p.add_argument("--max_noise_var", type=float, default=10.0)
    p.add_argument("--min_signal_var", type=float, default=1e-4)
    p.add_argument("--max_signal_var", type=float, default=100.0)
    p.add_argument("--min_lengthscale", type=float, default=1e-3)
    p.add_argument("--max_lengthscale", type=float, default=20.0)
    p.add_argument("--response_seed_frac", type=float, default=0.25)
    p.add_argument("--response_inlier_frac", type=float, default=0.70)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out_dir", default="experiments/pdebench/burgers_train200_500_1000_selfcem_5seeds_20260519")
    return p.parse_args()


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> Dict[str, Any]:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = _read_csv(out_dir / "metrics.csv") if args.resume else []
    done = {
        (int(r.get("train_size", -1)), int(r.get("seed", -1)))
        for r in all_rows
        if r.get("method") == METHOD_NAME and r.get("subset") == "all" and r.get("status", "ok") == "ok"
    }
    summary: Dict[str, Any] = {"args": vars(args), "rows": []}
    print(f"device={device}", flush=True)
    for train_size in _parse_ints(args.train_sizes):
        for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
            key = (int(train_size), int(seed))
            if key in done:
                print(f"skip PDEBench train={train_size} seed={seed} method={METHOD_NAME}", flush=True)
                continue
            print(f"PDEBench train={train_size} seed={seed} method={METHOD_NAME}", flush=True)
            rows = run_one(args, train_size, seed, device)
            all_rows.extend(rows)
            summary["rows"].extend(rows)
            agg_rows = _aggregate(all_rows)
            _write_csv(out_dir / "metrics.csv", all_rows)
            _write_csv(out_dir / "aggregate.csv", agg_rows)
            (out_dir / "summary.json").write_text(json.dumps(sanitize_for_json(summary), indent=2), encoding="utf-8")
            _write_readme(out_dir, args, agg_rows)
    agg_rows = _aggregate(all_rows)
    _write_csv(out_dir / "aggregate.csv", agg_rows)
    _write_readme(out_dir, args, agg_rows)
    return {"rows": all_rows, "aggregate": agg_rows}


if __name__ == "__main__":
    main()

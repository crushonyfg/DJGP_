"""Run uncertain-W CEM/qR prediction heads on UCI train=500 splits.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke: one dataset, one seed, small test slice.
    python experiments/uci/compare_uncertain_w_cem_train500.py ^
        --datasets "Wine Quality" --train_sizes 500 --num_exp 1 --max_test_anchors 20 ^
        --variants self_cem2_nstd0.1_kcov0.1 ^
        --hyper_steps 2 --gate_steps 2 --refresh_hyper_steps 2 --refresh_gate_steps 2 ^
        --out_dir experiments/uci/uncertain_w_cem_train500_smoke

    # Main candidate: 3 UCI datasets, train=500, test=200, five seeds.
    python experiments/uci/compare_uncertain_w_cem_train500.py ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --train_sizes 500 --num_exp 5 --max_test_anchors 200 ^
        --variants self_cem2_nstd0.1_kcov0.1,qr_alt_lr2_refresh10_self_cem2_nstd0.1_kcov0.1 ^
        --out_dir experiments/uci/uncertain_w_cem_train500_5seeds_20260515

The q(R) variants reuse the synthetic uncertain-W CEM implementation.  By
default q(R) is trained at prediction/test anchors, matching the current
alternating qR experiments.  Set ``--qr_train_anchor_count`` to train q(R) on
train-set anchors and induce q(W) at test anchors.
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
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics import sanitize_for_json  # noqa: E402
from experiments.synthetic import compare_uncertain_w_cem_l2_lh as uw  # noqa: E402
from experiments.uci.compare_inductive_projection import _subsample  # noqa: E402
from experiments.uci.compare_parkinsons_grouped_split import _load_parkinsons_grouped_split  # noqa: E402
from experiments.uci.uci_data import load_uci_split  # noqa: E402


DATASETS_DEFAULT = "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
METRIC_KEYS = ["rmse", "mean_crps", "wall_sec", "coverage_90", "mean_width_90", "coverage_95", "mean_width_95"]


def _default_uw_args() -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["compare_uncertain_w_cem_l2_lh.py"]
        return uw.parse_args()
    finally:
        sys.argv = old_argv


def _parse_csv(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(s.strip()) for s in str(text).split(",") if s.strip()]


def _dataset_n(dataset: str, requested_n: int) -> int:
    if requested_n > 0:
        return int(requested_n)
    return 35 if dataset == "Appliances Energy Prediction" else 25


def _fake_cfg(dataset: str, *, train_size: int, n_test_full: int, Q: int, D: int, n: int) -> dict[str, Any]:
    return {
        "family": dataset,
        "N_train": int(train_size),
        "N_test": int(n_test_full),
        "Q": int(Q),
        "H": int(D),
        "d": int(Q),
        "Q_true": int(Q),
        "n": int(n),
        "noise_std": float("nan"),
        "noise_var": float("nan"),
        "expansion": "uci",
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    preferred = [
        "dataset",
        "train_size",
        "seed",
        "method",
        "status",
        "rmse",
        "mean_crps",
        "coverage_90",
        "mean_width_90",
        "coverage_95",
        "mean_width_95",
        "wall_sec",
    ]
    for key in preferred:
        if any(key in r for r in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_or_nan(x: Any) -> float:
    try:
        if x == "":
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["dataset"]), int(row["train_size"]), str(row["method"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset, train_size, method), group in sorted(groups.items()):
        agg: dict[str, Any] = {
            "dataset": dataset,
            "train_size": train_size,
            "method": method,
            "n": len(group),
        }
        for meta in (
            "n_neighbors",
            "m_inducing",
            "qr_train_anchor_count",
            "qr_train_anchor_strategy",
            "qr_train_n_neighbors",
            "lowrank_rank",
            "lowrank_qr_w_signal_var",
            "split_type",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in METRIC_KEYS + [
            "train_sec",
            "pred_sec",
            "init_inlier_rate",
            "final_inlier_rate",
            "label_change_rate",
            "mean_noise_var",
            "qr_kl_per_anchor",
            "qr_grad_data_over_scaled_KL_R_mu_final",
            "refresh_label_change_mean",
            "qr_train_neighbor_visited_frac",
            "test_neighbor_visited_frac",
        ]:
            vals = np.asarray([_float_or_nan(r.get(key)) for r in group], dtype=np.float64)
            if np.isfinite(vals).any():
                agg[f"{key}_mean"] = float(np.nanmean(vals))
                agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _load_split(
    dataset: str,
    seed: int,
    train_size: int,
    max_test: int,
    *,
    grouped_parkinsons: bool,
    train_subject_frac: float,
):
    if grouped_parkinsons and dataset == "Parkinsons Telemonitoring":
        split = _load_parkinsons_grouped_split(seed=seed, train_subject_frac=float(train_subject_frac))
        X_train_raw, y_train_raw, _ = _subsample(split["X_train"], split["y_train"], train_size, seed + 500)
        X_test_raw, y_test_raw, _ = _subsample(split["X_test"], split["y_test"], max_test, seed + 1000)
        return X_train_raw, X_test_raw, y_train_raw, y_test_raw, int(split["K"]), "subject_grouped", int(split["X_test"].shape[0])
    X_train_full, X_test_full, y_train_full, y_test_full, K = load_uci_split(dataset, seed)
    X_train_raw, y_train_raw, _ = _subsample(X_train_full, y_train_full, train_size, seed + 500)
    X_test_raw, y_test_raw, _ = _subsample(X_test_full, y_test_full, max_test, seed + 1000)
    return X_train_raw, X_test_raw, y_train_raw, y_test_raw, int(K), "random_row", int(X_test_full.shape[0])


def _run_one(
    args: argparse.Namespace,
    *,
    dataset: str,
    train_size: int,
    seed: int,
    variant: str,
    device: torch.device,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    spec = (
        uw.VariantSpec("original_meanw_jumpgp_cem", "anchor", "original", "cem", 0, 0.0)
        if variant == "__original__"
        else uw._parse_variant(variant)
    )
    row: dict[str, Any] = {
        "dataset": dataset,
        "train_size": int(train_size),
        "seed": int(seed),
        "method": "original_meanw_jumpgp_cem" if variant == "__original__" else variant,
        "variant": "original_meanw_jumpgp_cem" if variant == "__original__" else variant,
        "status": "failed",
    }
    try:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw, K_default, split_type, n_test_full = _load_split(
            dataset,
            seed,
            train_size,
            int(args.max_test_anchors),
            grouped_parkinsons=bool(args.grouped_parkinsons),
            train_subject_frac=float(args.train_subject_frac),
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(np.asarray(X_train_raw, dtype=np.float64)).astype(np.float64)
        X_test = scaler.transform(np.asarray(X_test_raw, dtype=np.float64)).astype(np.float64)
        y_train = np.asarray(y_train_raw, dtype=np.float64).reshape(-1)
        y_test = np.asarray(y_test_raw, dtype=np.float64).reshape(-1)
        y_mean = float(y_train.mean())
        y_std = float(y_train.std() if y_train.std() > 1e-8 else 1.0)
        y_train_std = ((y_train - y_mean) / y_std).astype(np.float64)
        Q = int(args.Q if args.Q is not None else K_default)
        n = _dataset_n(dataset, int(args.n_neighbors))
        cfg = _fake_cfg(dataset, train_size=train_size, n_test_full=n_test_full, Q=Q, D=X_train.shape[1], n=n)
        uw_args = _default_uw_args()
        uw_args.projection_init = str(args.projection_init)
        uw_args.max_test = int(args.max_test_anchors)
        uw_args.n_neighbors = int(n)
        uw_args.m_inducing = int(args.m_inducing)
        uw_args.qr_train_anchor_count = int(args.qr_train_anchor_count)
        uw_args.qr_train_anchor_strategy = str(args.qr_train_anchor_strategy)
        uw_args.qr_train_n_neighbors = int(args.qr_train_n_neighbors or n)
        uw_args.lowrank_basis = str(args.lowrank_basis)
        uw_args.lowrank_qr_w_signal_var = float(args.lowrank_qr_w_signal_var)
        uw_args.qr_steps = int(args.qr_steps)
        uw_args.qr_outer_cycles = int(args.qr_outer_cycles)
        uw_args.qr_beta_kl = float(args.qr_beta_kl)
        uw_args.qr_lr = float(args.qr_lr)
        uw_args.qr_kl_warmup_steps = int(args.qr_kl_warmup_steps)
        uw_args.refresh_hyper_steps = int(args.refresh_hyper_steps)
        uw_args.refresh_gate_steps = int(args.refresh_gate_steps)
        uw_args.hyper_steps = int(args.hyper_steps)
        uw_args.gate_steps = int(args.gate_steps)
        uw_args.include_original = True

        W0_np = uw._projection_W0(X_train, y_train_std, Q, str(args.projection_init))
        W0 = torch.as_tensor(W0_np, device=device, dtype=torch.float64)
        V_basis = None
        if int(spec.lowrank_rank) > 0:
            V_np = uw._residual_basis_V(X_train, W0_np, int(spec.lowrank_rank), mode=str(args.lowrank_basis), seed=seed + 104729)
            V_basis = torch.as_tensor(V_np, device=device, dtype=torch.float64)
        X_train_t = torch.as_tensor(X_train, device=device, dtype=torch.float64)
        X_test_t = torch.as_tensor(X_test, device=device, dtype=torch.float64)
        y_train_std_t = torch.as_tensor(y_train_std, device=device, dtype=torch.float64)
        X_neighbors, y_neighbors_std, test_neighbor_idx = uw._neighbor_stack(
            X_test_t,
            X_train_t,
            y_train_std_t,
            n_neighbors=n,
        )
        w_cov_lengthscale = float(args.w_cov_lengthscale) if args.w_cov_lengthscale is not None else uw._median_distance(X_train_t)
        inducing_idx = uw._kmeans_medoids(X_train, int(args.m_inducing), seed + 7919)
        X_inducing = X_train_t[torch.as_tensor(inducing_idx, device=device, dtype=torch.long)]
        qr_w_lengthscale = float(args.qr_w_lengthscale) if args.qr_w_lengthscale is not None else uw._median_distance(X_inducing)
        w_moments = uw._make_w_moments(
            mode=spec.mode,
            W0=W0,
            X_anchor=X_test_t,
            X_neighbors=X_neighbors,
            w_var=float(args.w_var),
            w_cov_lengthscale=w_cov_lengthscale,
        )
        cem_state, cem_diag = uw._run_meanw_cem_init(X_test_t, X_neighbors, y_neighbors_std, W0)
        if variant == "__original__":
            out = uw._run_original_row(
                args=uw_args,
                cfg=cfg,
                setting=dataset,
                seed=seed,
                cem_state=cem_state,
                cem_diag=cem_diag,
                y_test=y_test,
                y_mean=y_mean,
                y_std=y_std,
            )
        elif int(args.qr_train_anchor_count) > 0 and spec.kind == "qr_alt":
            out, _extra = uw._run_train_anchor_qr_alt_variant(
                uw_args,
                spec,
                seed=seed,
                X_train_np=X_train,
                y_train_std_np=y_train_std,
                X_train_t=X_train_t,
                y_train_std_t=y_train_std_t,
                X_anchor=X_test_t,
                X_neighbors=X_neighbors,
                y_neighbors_std=y_neighbors_std,
                test_neighbor_idx=test_neighbor_idx,
                y_test=y_test,
                y_mean=y_mean,
                y_std=y_std,
                W0=W0,
                V_basis=V_basis,
                test_w_moments=w_moments,
                test_cem_state=cem_state,
                X_inducing=X_inducing,
                qr_w_lengthscale=qr_w_lengthscale,
                w_cov_lengthscale=w_cov_lengthscale,
            )
        else:
            out, _extra = uw._run_variant(
                uw_args,
                spec,
                X_anchor=X_test_t,
                X_neighbors=X_neighbors,
                y_neighbors_std=y_neighbors_std,
                y_test=y_test,
                y_mean=y_mean,
                y_std=y_std,
                W0=W0,
                V_basis=V_basis,
                w_moments=w_moments,
                cem_state=cem_state,
                X_inducing=X_inducing,
                qr_w_lengthscale=qr_w_lengthscale,
            )
        row.update(out)
        row.update({k: sanitize_for_json(v) for k, v in cem_diag.items()})
        row.update(
            {
                "dataset": dataset,
                "train_size": int(train_size),
                "seed": int(seed),
                "method": "original_meanw_jumpgp_cem" if variant == "__original__" else variant,
                "variant": "original_meanw_jumpgp_cem" if variant == "__original__" else variant,
                "status": "ok",
                "split_type": split_type,
                "K_default": int(K_default),
                "Q": int(Q),
                "D": int(X_train.shape[1]),
                "n_neighbors": int(n),
                "m_inducing": int(args.m_inducing),
                "qr_train_anchor_count": int(args.qr_train_anchor_count),
                "qr_train_n_neighbors": int(args.qr_train_n_neighbors or n),
                "qr_train_anchor_strategy": str(args.qr_train_anchor_strategy),
                "wall_sec": float(time.perf_counter() - t0),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["wall_sec"] = float(time.perf_counter() - t0)
        print(f"FAILED dataset={dataset} train={train_size} seed={seed} variant={variant}: {row['error']}", flush=True)
    return row


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# UCI Uncertain-W CEM train=500

This folder was produced by `experiments/uci/compare_uncertain_w_cem_train500.py`.
Rows evaluate uncertain-W CEM/self-CEM and optional q(R) variants on UCI
train/test splits with train-size subsampling and a fixed test-anchor slice.

Project-wide log entry should be added to `docs/experiments_log.md` after the
run is interpreted.

Files:

- `metrics.csv`: one row per dataset/train_size/seed/method.
- `aggregate.csv`: mean/std over successful rows.
- `summary.json`: exact runner arguments.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UCI uncertain-W CEM train=500 benchmark.")
    p.add_argument("--datasets", default=DATASETS_DEFAULT)
    p.add_argument("--train_sizes", default="500")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--variants", default="self_cem2_nstd0.1_kcov0.1,qr_alt_lr2_refresh10_self_cem2_nstd0.1_kcov0.1")
    p.add_argument("--include_original", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grouped_parkinsons", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train_subject_frac", type=float, default=0.8)
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--projection_init", choices=["pls", "pca", "pca_scaled"], default="pls")
    p.add_argument("--n_neighbors", type=int, default=-1)
    p.add_argument("--m_inducing", type=int, default=40)
    p.add_argument("--qr_train_anchor_count", type=int, default=0)
    p.add_argument("--qr_train_anchor_strategy", choices=["kmeans", "random", "kmeans_random", "kmeans_yvar"], default="kmeans_yvar")
    p.add_argument("--qr_train_n_neighbors", type=int, default=None)
    p.add_argument("--w_var", type=float, default=0.02)
    p.add_argument("--w_cov_lengthscale", type=float, default=None)
    p.add_argument("--qr_w_lengthscale", type=float, default=None)
    p.add_argument("--lowrank_basis", choices=["pca_residual", "pca", "random"], default="pca_residual")
    p.add_argument("--lowrank_qr_w_signal_var", type=float, default=0.1)
    p.add_argument("--qr_steps", type=int, default=40)
    p.add_argument("--qr_lr", type=float, default=0.01)
    p.add_argument("--qr_beta_kl", type=float, default=0.05)
    p.add_argument("--qr_kl_warmup_steps", type=int, default=20)
    p.add_argument("--qr_outer_cycles", type=int, default=4)
    p.add_argument("--hyper_steps", type=int, default=30)
    p.add_argument("--gate_steps", type=int, default=60)
    p.add_argument("--refresh_hyper_steps", type=int, default=10)
    p.add_argument("--refresh_gate_steps", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out_dir", default="experiments/uci/uncertain_w_cem_train500_5seeds_20260515")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    datasets = _parse_csv(args.datasets)
    train_sizes = _parse_ints(args.train_sizes)
    variants = _parse_csv(args.variants)
    for variant in variants:
        uw._parse_variant(variant)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(out_dir / "metrics.csv") if args.resume else []
    done = {
        (str(r.get("dataset")), int(r.get("train_size", -1)), int(r.get("seed", -1)), str(r.get("method")))
        for r in rows
        if r.get("status") == "ok"
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"device={device}", flush=True)
    for train_size in train_sizes:
        for seed in range(int(args.num_exp)):
            for dataset in datasets:
                methods = list(variants)
                if bool(args.include_original):
                    methods = ["__original__"] + methods
                for method in methods:
                    method_name = "original_meanw_jumpgp_cem" if method == "__original__" else method
                    key = (dataset, train_size, seed, method_name)
                    if key in done:
                        print(f"skip dataset={dataset} train={train_size} seed={seed} method={method_name}", flush=True)
                        continue
                    print(f"run dataset={dataset} train={train_size} seed={seed} method={method_name}", flush=True)
                    row = _run_one(args, dataset=dataset, train_size=train_size, seed=seed, variant=method, device=device)
                    rows.append(row)
                    _write_rows(out_dir / "metrics.csv", rows)
                    _write_rows(out_dir / "aggregate.csv", _aggregate(rows))
                    if row.get("status") == "ok":
                        done.add(key)
    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "args": vars(args),
        "n_rows": len(rows),
        "n_ok": int(sum(1 for r in rows if r.get("status") == "ok")),
        "n_failed": int(sum(1 for r in rows if r.get("status") != "ok")),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    _write_readme(out_dir, sanitize_for_json(summary))
    print(f"Saved UCI uncertain-W CEM results to {out_dir}", flush=True)
    return {"rows": rows, "aggregate": _aggregate(rows), "summary": summary}


if __name__ == "__main__":
    main()

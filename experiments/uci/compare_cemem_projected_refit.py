"""Run CEM-EM large-candidate/projected-refit ablations on UCI data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # One-seed pilot on the current UCI train=500 setting.
    python experiments\\uci\\compare_cemem_projected_refit.py ^
        --num_exp 1 --max_total_train 500 --max_test_anchors 200 ^
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" ^
        --n_candidate 100 --n_final_auto 1 --n_clusters 64 ^
        --out_dir experiments\\uci\\cemem_projected_refit_pilot_train500_seed0

This runner tests three response-aware projection variants:

- ``cemem_small_current``: learn W on the current small raw-X KNN and evaluate
  projected JumpGP on the same small neighborhood.
- ``cemem_large_raw_final``: learn W on a larger raw-X candidate neighborhood,
  then evaluate projected JumpGP on the original small raw-X neighborhood.
- ``cemem_large_projected_final``: learn W on the larger raw-X candidate
  neighborhood, then reselect the final small neighborhood by projected-space
  distance inside that candidate pool.
- ``cluster_anchor_projected_final``: learn one W per train-only KMeans cluster
  center, assign test points to the nearest cluster, and use projected-space
  final neighbors from that cluster's candidate pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    compute_pls_W0,
    per_anchor_W_diagnostics,
    run_projected_jumpgp,
    train_projection_cem_em,
)
from experiments.uci.compare_smalltrain_multiseed_grouped import (  # noqa: E402
    TUNED_PRESETS,
    _load_dataset_for_seed,
)


METRIC_KEYS = ["rmse", "mean_crps", "wall_sec", "coverage_90", "width_90", "coverage_95", "width_95"]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def _apply_train_scaling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    standardize_x: bool,
    standardize_y: bool,
):
    X_train_s = np.asarray(X_train, dtype=np.float64)
    X_test_s = np.asarray(X_test, dtype=np.float64)
    y_train_s = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_test_s = np.asarray(y_test, dtype=np.float64).reshape(-1)
    y_scaler = None
    if standardize_x:
        x_scaler = StandardScaler().fit(X_train_s)
        X_train_s = x_scaler.transform(X_train_s)
        X_test_s = x_scaler.transform(X_test_s)
    if standardize_y:
        y_scaler = StandardScaler().fit(y_train_s.reshape(-1, 1))
        y_train_s = y_scaler.transform(y_train_s.reshape(-1, 1)).reshape(-1)
        y_test_s = y_scaler.transform(y_test_s.reshape(-1, 1)).reshape(-1)
    return X_train_s, y_train_s, X_test_s, y_test_s, y_scaler


def _pack_metrics_maybe_original_y(
    y_scaler,
    y_eval_scaled: np.ndarray,
    mu_scaled: np.ndarray,
    sigma_scaled: np.ndarray,
    wall_sec: float,
) -> list[float]:
    if y_scaler is None:
        y_eval = np.asarray(y_eval_scaled, dtype=np.float64).reshape(-1)
        mu = np.asarray(mu_scaled, dtype=np.float64).reshape(-1)
        sigma = np.asarray(sigma_scaled, dtype=np.float64).reshape(-1)
    else:
        scale = float(np.squeeze(y_scaler.scale_))
        if not np.isfinite(scale) or scale < 1e-20:
            scale = 1.0
        y_eval = y_scaler.inverse_transform(np.asarray(y_eval_scaled).reshape(-1, 1)).reshape(-1)
        mu = y_scaler.inverse_transform(np.asarray(mu_scaled).reshape(-1, 1)).reshape(-1)
        sigma = np.asarray(sigma_scaled, dtype=np.float64).reshape(-1) * abs(scale)
    rmse = float(np.sqrt(np.mean((mu - y_eval) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.from_numpy(y_eval).float(),
            torch.from_numpy(mu).float(),
            torch.from_numpy(sigma).float(),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, float(wall_sec), y_eval, mu, sigma)


def _knn_lists(
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    k: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    k = min(int(k), X_pool.shape[0])
    dist = torch.cdist(X_anchor, X_pool)
    idx = torch.topk(dist, k=k, largest=False).indices
    X_lists = [X_pool[ii].contiguous() for ii in idx]
    y_lists = [y_pool[ii].contiguous() for ii in idx]
    return X_lists, y_lists, [ii.detach().cpu() for ii in idx]


def _projected_small_from_candidates(
    W: torch.Tensor,
    X_anchor: torch.Tensor,
    X_candidates: list[torch.Tensor],
    y_candidates: list[torch.Tensor],
    k: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    X_out: list[torch.Tensor] = []
    y_out: list[torch.Tensor] = []
    for i, (X_cand, y_cand) in enumerate(zip(X_candidates, y_candidates)):
        k_i = min(int(k), X_cand.shape[0])
        z_cand = X_cand @ W[i].T
        z_anchor = X_anchor[i] @ W[i].T
        dist = ((z_cand - z_anchor) ** 2).sum(dim=1)
        idx = torch.topk(dist, k=k_i, largest=False).indices
        X_out.append(X_cand[idx].contiguous())
        y_out.append(y_cand[idx].contiguous())
    return X_out, y_out


def _make_cem_config(args: argparse.Namespace) -> CemEmConfig:
    return CemEmConfig(
        n_outer=args.n_outer,
        n_m_steps=args.n_m_steps,
        lr=args.lr,
        lambda_gate=args.lambda_gate,
        lambda_smooth=args.lambda_smooth,
        lambda_scale=args.lambda_scale,
        normalize_smoothness=bool(args.normalize_smoothness),
        track_best_W=bool(args.track_best_W),
        snapshot_after_each_outer=False,
        cem_progress=False,
        verbose=False,
    )


def _train_cem_w(
    *,
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_anchor_t: torch.Tensor,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    Q: int,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    label: str,
):
    V_np, V_info = build_V_basis(
        X_train_np,
        y_train_np,
        n_pls=args.n_pls,
        n_pca=args.n_pca,
        n_random=args.n_random,
        seed=seed,
    )
    W0_np = compute_pls_W0(X_train_np, y_train_np, Q=Q)
    V_t = torch.from_numpy(V_np).float().to(device)
    W0_t = torch.from_numpy(W0_np).float().to(device)
    model = CoeffProjection(
        T=X_anchor_t.shape[0],
        Q=Q,
        D=X_anchor_t.shape[1],
        V=V_t,
        W0_init=W0_t,
        learn_W0=True,
        init_C_scale=args.init_C_scale,
        coeff_scale=args.coeff_scale,
        row_normalize=bool(args.row_normalize_W),
    ).to(device)
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_neighbors_list,
        y_neighbors_list=y_neighbors_list,
        X_anchors=X_anchor_t,
        device=device,
        cfg=_make_cem_config(args),
        label=label,
    )
    diag = {
        "V_basis": V_info,
        "W": per_anchor_W_diagnostics(res.W, W_ref=W0_t),
        "train_time_sec": float(res.train_time_sec),
        "best_outer": int(res.best_outer),
        "best_metric_value": float(res.best_metric_value),
    }
    return res.W.detach(), diag


def _evaluate_w(
    *,
    W: torch.Tensor,
    X_anchor_t: torch.Tensor,
    X_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    y_test_scaled: np.ndarray,
    y_scaler,
    device: torch.device,
    label: str,
) -> list[float]:
    out = run_projected_jumpgp(
        W,
        X_anchor_t,
        X_neighbors_list,
        y_neighbors_list,
        device=device,
        progress=False,
        desc=label,
    )
    return _pack_metrics_maybe_original_y(
        y_scaler,
        y_test_scaled,
        out["mu"],
        out["sigma"],
        float(out["elapsed_sec"]),
    )


def _run_cluster_anchor_variant(
    *,
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    y_test_scaled: np.ndarray,
    y_scaler,
    Q: int,
    n_final: int,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    n_clusters = min(int(args.n_clusters), X_train_np.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X_train_np)
    centers_t = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
    X_cand_centers, y_cand_centers, _ = _knn_lists(centers_t, X_train_t, y_train_t, args.n_candidate)
    W_centers, diag = _train_cem_w(
        X_train_np=X_train_np,
        y_train_np=y_train_np,
        X_anchor_t=centers_t,
        X_neighbors_list=X_cand_centers,
        y_neighbors_list=y_cand_centers,
        Q=Q,
        args=args,
        seed=seed,
        device=device,
        label="cluster_anchor",
    )
    assign = torch.cdist(X_test_t, centers_t).argmin(dim=1)
    W_test = W_centers[assign].contiguous()
    X_cand_test = [X_cand_centers[int(a)] for a in assign.detach().cpu().tolist()]
    y_cand_test = [y_cand_centers[int(a)] for a in assign.detach().cpu().tolist()]
    X_final, y_final = _projected_small_from_candidates(
        W_test,
        X_test_t,
        X_cand_test,
        y_cand_test,
        n_final,
    )
    metrics = _evaluate_w(
        W=W_test,
        X_anchor_t=X_test_t,
        X_neighbors_list=X_final,
        y_neighbors_list=y_final,
        y_test_scaled=y_test_scaled,
        y_scaler=y_scaler,
        device=device,
        label="cluster_anchor_projected_final",
    )
    counts = np.bincount(assign.detach().cpu().numpy(), minlength=n_clusters)
    diag.update(
        {
            "n_clusters": int(n_clusters),
            "n_clusters_used_by_test": int(np.sum(counts > 0)),
            "max_test_per_cluster": int(counts.max(initial=0)),
            "min_nonempty_test_per_cluster": int(counts[counts > 0].min()) if np.any(counts > 0) else 0,
        }
    )
    return metrics, diag


def _run_one(dataset_name: str, seed: int, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    _seed_all(seed)
    data = _load_dataset_for_seed(
        dataset_name,
        seed,
        max_total_train=args.max_total_train,
        max_test_anchors=args.max_test_anchors,
        train_subject_frac=args.train_subject_frac,
    )
    preset = dict(TUNED_PRESETS[dataset_name])
    if dataset_name == "Parkinsons Telemonitoring" and args.parkinsons_standardize_x >= 0:
        preset["standardize_x"] = bool(args.parkinsons_standardize_x)
    if dataset_name == "Parkinsons Telemonitoring" and args.parkinsons_standardize_y >= 0:
        preset["standardize_y"] = bool(args.parkinsons_standardize_y)

    X_train_np, y_train_np, X_test_np, y_test_np, y_scaler = _apply_train_scaling(
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"],
        standardize_x=bool(preset["standardize_x"]),
        standardize_y=bool(preset["standardize_y"]),
    )
    Q = int(args.Q if args.Q is not None else data["K"])
    n_final = int(args.n_final if args.n_final is not None else preset["n"])

    X_train_t = torch.from_numpy(X_train_np).float().to(device)
    y_train_t = torch.from_numpy(y_train_np).float().to(device)
    X_test_t = torch.from_numpy(X_test_np).float().to(device)
    X_small, y_small, _ = _knn_lists(X_test_t, X_train_t, y_train_t, n_final)
    X_large, y_large, _ = _knn_lists(X_test_t, X_train_t, y_train_t, args.n_candidate)

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "cemem_small_current" in methods:
        W_small, diagnostics["cemem_small_current"] = _train_cem_w(
            X_train_np=X_train_np,
            y_train_np=y_train_np,
            X_anchor_t=X_test_t,
            X_neighbors_list=X_small,
            y_neighbors_list=y_small,
            Q=Q,
            args=args,
            seed=seed,
            device=device,
            label="cemem_small_current",
        )
        metrics = _evaluate_w(
            W=W_small,
            X_anchor_t=X_test_t,
            X_neighbors_list=X_small,
            y_neighbors_list=y_small,
            y_test_scaled=y_test_np,
            y_scaler=y_scaler,
            device=device,
            label="cemem_small_current",
        )
        rows.append(_row(dataset_name, seed, "cemem_small_current", metrics, Q, n_final, args.n_candidate, preset))

    W_large = None
    if any(m in methods for m in ("cemem_large_raw_final", "cemem_large_projected_final")):
        W_large, diagnostics["cemem_large"] = _train_cem_w(
            X_train_np=X_train_np,
            y_train_np=y_train_np,
            X_anchor_t=X_test_t,
            X_neighbors_list=X_large,
            y_neighbors_list=y_large,
            Q=Q,
            args=args,
            seed=seed,
            device=device,
            label="cemem_large",
        )
    if "cemem_large_raw_final" in methods and W_large is not None:
        metrics = _evaluate_w(
            W=W_large,
            X_anchor_t=X_test_t,
            X_neighbors_list=X_small,
            y_neighbors_list=y_small,
            y_test_scaled=y_test_np,
            y_scaler=y_scaler,
            device=device,
            label="cemem_large_raw_final",
        )
        rows.append(_row(dataset_name, seed, "cemem_large_raw_final", metrics, Q, n_final, args.n_candidate, preset))
    if "cemem_large_projected_final" in methods and W_large is not None:
        X_proj, y_proj = _projected_small_from_candidates(W_large, X_test_t, X_large, y_large, n_final)
        metrics = _evaluate_w(
            W=W_large,
            X_anchor_t=X_test_t,
            X_neighbors_list=X_proj,
            y_neighbors_list=y_proj,
            y_test_scaled=y_test_np,
            y_scaler=y_scaler,
            device=device,
            label="cemem_large_projected_final",
        )
        rows.append(_row(dataset_name, seed, "cemem_large_projected_final", metrics, Q, n_final, args.n_candidate, preset))

    if "cluster_anchor_projected_final" in methods:
        metrics, diagnostics["cluster_anchor_projected_final"] = _run_cluster_anchor_variant(
            X_train_np=X_train_np,
            y_train_np=y_train_np,
            X_train_t=X_train_t,
            y_train_t=y_train_t,
            X_test_t=X_test_t,
            y_test_scaled=y_test_np,
            y_scaler=y_scaler,
            Q=Q,
            n_final=n_final,
            args=args,
            seed=seed,
            device=device,
        )
        rows.append(_row(dataset_name, seed, "cluster_anchor_projected_final", metrics, Q, n_final, args.n_candidate, preset))

    for row in rows:
        row["diagnostics"] = diagnostics.get(row["method"], {})
    return rows


def _row(
    dataset: str,
    seed: int,
    method: str,
    metrics: list[float],
    Q: int,
    n_final: int,
    n_candidate: int,
    preset: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": int(seed),
        "method": method,
        "Q": int(Q),
        "n_final": int(n_final),
        "n_candidate": int(n_candidate),
        "standardize_x": bool(preset["standardize_x"]),
        "standardize_y": bool(preset["standardize_y"]),
        "split_type": preset["split"],
    }
    for key, value in zip(METRIC_KEYS, metrics):
        row[key] = float(value)
    return row


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset, method), group in sorted(groups.items()):
        item: dict[str, Any] = {"dataset": dataset, "method": method, "n_seeds": len(group)}
        for key in METRIC_KEYS:
            vals = np.asarray([float(r[key]) for r in group], dtype=np.float64)
            item[f"{key}_mean"] = float(np.nanmean(vals))
            item[f"{key}_std"] = float(np.nanstd(vals))
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UCI CEM-EM projected-refit ablations.")
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument(
        "--datasets",
        default="Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction",
    )
    p.add_argument(
        "--methods",
        default="cemem_small_current,cemem_large_raw_final,cemem_large_projected_final,cluster_anchor_projected_final",
    )
    p.add_argument("--max_total_train", type=int, default=500)
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--train_subject_frac", type=float, default=0.8)
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--n_final", type=int, default=None)
    p.add_argument("--n_final_auto", type=int, default=1)
    p.add_argument("--n_candidate", type=int, default=100)
    p.add_argument("--n_clusters", type=int, default=64)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_m_steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lambda_gate", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_scale", type=float, default=0.0)
    p.add_argument("--init_C_scale", type=float, default=0.0)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--row_normalize_W", type=int, default=1)
    p.add_argument("--normalize_smoothness", type=int, default=1)
    p.add_argument("--track_best_W", type=int, default=1)
    p.add_argument("--parkinsons_standardize_x", type=int, choices=(-1, 0, 1), default=-1)
    p.add_argument("--parkinsons_standardize_y", type=int, choices=(-1, 0, 1), default=-1)
    p.add_argument("--out_dir", default="experiments/uci/cemem_projected_refit_pilot_train500_seed0")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rows: list[dict[str, Any]] = []
    diag: list[dict[str, Any]] = []
    print(f"device={device}", flush=True)
    print(f"writing artifacts to {out_dir}", flush=True)
    for dataset in datasets:
        for seed in range(args.num_exp):
            print(f"{dataset} seed={seed}", flush=True)
            seed_rows = _run_one(dataset, seed, args, device)
            for row in seed_rows:
                diag.append(
                    {
                        "dataset": row["dataset"],
                        "seed": row["seed"],
                        "method": row["method"],
                        "diagnostics": row.pop("diagnostics", {}),
                    }
                )
            rows.extend(seed_rows)
            _write_csv(out_dir / "metrics.csv", rows, [
                "dataset", "seed", "method", "Q", "n_final", "n_candidate",
                "standardize_x", "standardize_y", "split_type", *METRIC_KEYS,
            ])
            _write_csv(out_dir / "aggregate.csv", _aggregate(rows), [
                "dataset", "method", "n_seeds",
                *[f"{key}_{stat}" for key in METRIC_KEYS for stat in ("mean", "std")],
            ])
            (out_dir / "summary.json").write_text(
                json.dumps({"args": vars(args), "metrics": rows, "diagnostics": diag}, indent=2, default=str),
                encoding="utf-8",
            )
    (out_dir / "README.md").write_text(
        "# UCI CEM-EM projected-refit pilot\n\n"
        "Ablates large-candidate projection learning, projected-space final "
        "neighbor reselection, and a simple cluster-anchor shared-W variant.\n",
        encoding="utf-8",
    )
    return {"metrics": rows, "aggregate": _aggregate(rows)}


if __name__ == "__main__":
    main()

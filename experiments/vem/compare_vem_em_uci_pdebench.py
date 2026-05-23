"""Run conditional-W VEM-EM on UCI small-train and PDEBench Burgers.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke test on one small UCI slice.
    python experiments/vem/compare_vem_em_uci_pdebench.py ^
        --benchmarks uci ^
        --uci_train_sizes 200 ^
        --uci_datasets "Wine Quality" ^
        --num_exp 1 --max_test_anchors 20 --max_train_anchors 40 ^
        --n_candidate 40 --n_vem 25 --steps 5 --eval_mc 1 ^
        --out_dir experiments/vem/vem_em_smoke

    # Five-seed UCI train=200/500/1000 plus PDEBench-Burgers.
    python experiments/vem/compare_vem_em_uci_pdebench.py ^
        --benchmarks uci,pdebench ^
        --uci_train_sizes 200,500,1000 ^
        --num_exp 5 --n_candidate 100 --n_vem 50 --steps 80 --eval_mc 2 ^
        --pdebench_dataset_folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000 ^
        --pdebench_Q 3 --pdebench_n 25 ^
        --out_dir experiments/vem/vem_em_uci_pdebench_5seeds

This runner evaluates the current best practical conditional-W VEM variant:
soft responsibilities, boundary-weighted anchors, posterior q(W) prediction,
and projected-space final JumpGP neighborhoods.  It is meant to create rows
directly comparable with the existing LMJGP/CEM-EM/XGB/DKL small-train tables.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from djgp.diagnostics.projection_posterior import sanitize_for_json  # noqa: E402
from djgp.evaluation import mean_gaussian_crps_torch, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.projections import build_V_basis, compute_pls_W0, run_projected_jumpgp_ensemble  # noqa: E402
from djgp.projections.conditional_vem import VemWConfig, train_conditional_vem_w  # noqa: E402
from experiments.pdebench.compare_pdebench_burgers import (  # noqa: E402
    _dataset_arrays as _pde_dataset_arrays,
)
from experiments.pdebench.compare_pdebench_burgers import (  # noqa: E402
    _threshold_masks as _pde_threshold_masks,
)
from experiments.pdebench.compare_pdebench_burgers import load_dataset as _load_pdebench_dataset  # noqa: E402
from experiments.uci.compare_smalltrain_multiseed_grouped import (  # noqa: E402
    _apply_scaling,
    _load_dataset_for_seed,
    _pack_metrics_with_numeric_filter,
    _split_base_anchor,
)


UCI_DATASETS_DEFAULT = "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
UCI_PRESETS = {
    "Wine Quality": {"n": 25, "standardize_x": True, "standardize_y": False, "split": "random_row"},
    "Parkinsons Telemonitoring": {
        "n": 25,
        "standardize_x": True,
        "standardize_y": False,
        "split": "subject_grouped",
    },
    "Appliances Energy Prediction": {
        "n": 35,
        "standardize_x": True,
        "standardize_y": True,
        "split": "random_row",
    },
}
METHOD_NAME = "vem_em_learned_qw"
PDE_SPLITS = ("all", "near_threshold", "far_from_threshold")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "benchmark",
        "train_size",
        "dataset",
        "subset",
        "seed",
        "method",
        "status",
        "n_train",
        "n_test",
        "Q",
        "n_final",
        "n_candidate",
        "n_vem",
        "standardize_x",
        "standardize_y",
        "metric_policy",
        "n_eval_used",
        "n_numeric_omitted",
        *RESULT_ROW_KEYS,
        "train_sec",
        "pred_sec",
        "best_loss",
        "gamma_mean",
        "gamma_entropy",
        "kl_coeff",
        "C_std_median",
        "error",
    ]
    keys_seen: set[str] = set()
    for row in rows:
        keys_seen.update(row.keys())
    keys = [key for key in preferred if key in keys_seen]
    keys.extend(sorted(keys_seen - set(keys)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str) and not value.strip():
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            str(row.get("benchmark", "")),
            str(row.get("train_size", "")),
            str(row.get("dataset", "")),
            str(row.get("subset", "all")),
            str(row.get("method", "")),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (benchmark, train_size, dataset, subset, method), group in sorted(groups.items()):
        agg: dict[str, Any] = {
            "benchmark": benchmark,
            "train_size": train_size,
            "dataset": dataset,
            "subset": subset,
            "method": method,
            "n_seeds": len(group),
        }
        for meta in (
            "n_train",
            "n_test",
            "Q",
            "n_final",
            "n_candidate",
            "n_vem",
            "standardize_x",
            "standardize_y",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in RESULT_ROW_KEYS:
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        for key in ("train_sec", "pred_sec", "best_loss", "gamma_mean", "gamma_entropy", "kl_coeff"):
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            if np.isfinite(vals).any():
                agg[f"{key}_mean"] = float(np.nanmean(vals))
                agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _make_vem_config(args: argparse.Namespace) -> VemWConfig:
    return VemWConfig(
        posterior=True,
        n_steps=int(args.steps),
        vem_steps=1,
        mc_train=int(args.mc_train),
        mc_gamma=int(args.mc_gamma),
        lr=float(args.lr),
        coeff_scale=float(args.coeff_scale),
        init_log_std=float(args.init_log_std),
        prior_std=float(args.prior_std),
        kl_weight=float(args.kl_weight),
        kl_warmup_steps=int(args.kl_warmup_steps),
        lambda_smooth=float(args.lambda_smooth),
        lambda_coeff=float(args.lambda_coeff),
        lambda_gate_l2=float(args.lambda_gate_l2),
        gamma_init=float(args.gamma_init),
        gamma_floor=float(args.gamma_floor),
        gamma_damping=float(args.gamma_damping),
        temperature=float(args.temperature),
        temperature_final=float(args.temperature_final),
        boundary_weighted=True,
        train_kernel=False,
        train_noise=False,
        init_lengthscale=float(args.init_lengthscale),
        init_amp=float(args.init_amp),
        init_noise_var=float(args.init_noise_var),
        outlier_sigma_mult=float(args.outlier_sigma_mult),
        jitter=float(args.jitter),
        log_interval=int(args.log_interval),
        verbose=bool(args.verbose),
    )


def _build_neighbor_stacks(
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    n_neighbors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    neighborhoods = find_neighborhoods(
        X_anchor.detach().cpu(),
        X_pool.detach().cpu(),
        y_pool.detach().cpu(),
        M=int(n_neighbors),
    )
    X_stack = torch.stack([item["X_neighbors"].to(device=device).float() for item in neighborhoods], dim=0)
    y_stack = torch.stack([item["y_neighbors"].to(device=device).float().view(-1) for item in neighborhoods], dim=0)
    return X_stack, y_stack


def _projected_neighbor_lists(
    W_select: torch.Tensor,
    X_anchor: torch.Tensor,
    X_candidate: torch.Tensor,
    y_candidate: torch.Tensor,
    *,
    n_neighbors: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    with torch.no_grad():
        Z_candidate = torch.einsum("tmd,tqd->tmq", X_candidate, W_select)
        z_anchor = torch.einsum("td,tqd->tq", X_anchor, W_select)
        d2 = (Z_candidate - z_anchor.unsqueeze(1)).pow(2).sum(dim=-1)
        idx = torch.topk(d2, k=int(n_neighbors), largest=False, dim=1).indices
    X_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    for t in range(X_candidate.shape[0]):
        X_list.append(X_candidate[t, idx[t]].contiguous())
        y_list.append(y_candidate[t, idx[t]].contiguous())
    return X_list, y_list


def _row_normalize(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(eps)


def _fit_vem_predict(
    *,
    X_train_s: np.ndarray,
    y_train_s: np.ndarray,
    X_test_s: np.ndarray,
    Q: int,
    n_final: int,
    n_candidate: int,
    n_vem: int,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    X_train_t = torch.from_numpy(np.asarray(X_train_s, dtype=np.float32)).to(device)
    y_train_t = torch.from_numpy(np.asarray(y_train_s, dtype=np.float32).reshape(-1)).to(device)
    X_test_t = torch.from_numpy(np.asarray(X_test_s, dtype=np.float32)).to(device)
    X_cand, y_cand = _build_neighbor_stacks(
        X_test_t,
        X_train_t,
        y_train_t,
        n_neighbors=n_candidate,
        device=device,
    )
    y_mean = float(np.mean(y_train_s))
    y_std = float(np.std(y_train_s) + 1e-8)
    y_vem_std = ((y_cand[:, :n_vem] - y_mean) / y_std).contiguous()
    X_vem = X_cand[:, :n_vem].contiguous()

    V_np, V_info = build_V_basis(
        X_train_s,
        y_train_s,
        n_pls=int(args.n_pls),
        n_pca=int(args.n_pca),
        n_random=int(args.n_random),
        seed=seed,
    )
    W0_np = compute_pls_W0(X_train_s, y_train_s, Q=Q)
    V_t = torch.from_numpy(V_np.astype(np.float32)).to(device)
    W0_t = torch.from_numpy(W0_np.astype(np.float32)).to(device)

    result = train_conditional_vem_w(
        X_neighbors=X_vem,
        y_neighbors=y_vem_std,
        X_anchors=X_test_t,
        V=V_t,
        W0_init=W0_t,
        cfg=_make_vem_config(args),
    )
    W_samples = result.model.sample_W(int(args.eval_mc), mean_only=False).detach()
    W_select = result.W_mean.detach()
    if args.use_global_mean_projection:
        W_select = _row_normalize(W_select.mean(dim=0, keepdim=True)).expand_as(W_select).contiguous()
    X_list, y_list = _projected_neighbor_lists(
        W_select,
        X_test_t,
        X_cand,
        y_cand,
        n_neighbors=n_final,
    )
    pred = run_projected_jumpgp_ensemble(
        W_samples,
        X_test_t,
        X_list,
        y_list,
        device=device,
        progress=False,
        desc="vem_em_projection",
    )
    diag = {
        **result.diagnostics,
        "train_sec": float(result.train_time_sec),
        "pred_sec": float(pred["elapsed_sec"]),
        "total_wall_sec": float(result.train_time_sec + pred["elapsed_sec"]),
        "V_Rv": int(V_t.shape[1]),
        "V_info": V_info,
        "eval_mc": int(args.eval_mc),
        "y_vem_mean": y_mean,
        "y_vem_std": y_std,
    }
    return np.asarray(pred["mu"], dtype=np.float64), np.asarray(pred["sigma"], dtype=np.float64), diag


def _uci_row_base(
    *,
    train_size: int,
    dataset: str,
    seed: int,
    n_train: int,
    n_test: int,
    Q: int,
    n_final: int,
    n_candidate: int,
    n_vem: int,
    standardize_x: bool,
    standardize_y: bool,
) -> dict[str, Any]:
    return {
        "benchmark": "uci",
        "train_size": int(train_size),
        "dataset": dataset,
        "subset": "all",
        "seed": int(seed),
        "method": METHOD_NAME,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "Q": int(Q),
        "n_final": int(n_final),
        "n_candidate": int(n_candidate),
        "n_vem": int(n_vem),
        "standardize_x": int(standardize_x),
        "standardize_y": int(standardize_y),
    }


def _run_uci(args: argparse.Namespace, device: torch.device, out_dir: Path, rows: list[dict[str, Any]]) -> None:
    datasets = [s.strip() for s in args.uci_datasets.split(",") if s.strip()]
    train_sizes = [int(s.strip()) for s in args.uci_train_sizes.split(",") if s.strip()]
    completed = {
        (str(r.get("benchmark")), str(r.get("train_size")), str(r.get("dataset")), int(r.get("seed", -1)))
        for r in rows
        if r.get("status") == "ok" and str(r.get("benchmark")) == "uci" and str(r.get("seed", "")).strip()
    }
    for train_size in train_sizes:
        for dataset in datasets:
            if dataset not in UCI_PRESETS:
                raise ValueError(f"Unknown UCI dataset {dataset!r}")
            preset = UCI_PRESETS[dataset]
            for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
                key = ("uci", str(train_size), dataset, seed)
                if args.resume and key in completed:
                    print(f"skip completed UCI train={train_size} dataset={dataset} seed={seed}", flush=True)
                    continue
                _set_seed(seed)
                row_base: dict[str, Any] | None = None
                try:
                    print(f"[UCI] train={train_size} dataset={dataset} seed={seed}", flush=True)
                    data = _load_dataset_for_seed(
                        dataset,
                        seed,
                        max_total_train=train_size,
                        max_test_anchors=args.max_test_anchors,
                        train_subject_frac=args.train_subject_frac,
                    )
                    X_base, y_base, X_anchor, y_anchor, _split_info = _split_base_anchor(
                        data["X_train"],
                        data["y_train"],
                        max_train_anchors=args.max_train_anchors,
                        seed=seed + 2000,
                    )
                    (
                        X_train_s,
                        _X_base_s,
                        _X_anchor_s,
                        X_test_s,
                        y_train_s,
                        _y_base_s,
                        _y_anchor_s,
                        y_test_s,
                        _x_scaler,
                        y_scaler,
                    ) = _apply_scaling(
                        data["X_train"],
                        X_base,
                        X_anchor,
                        data["X_test"],
                        data["y_train"],
                        y_base,
                        y_anchor,
                        data["y_test"],
                        standardize_x=bool(preset["standardize_x"]),
                        standardize_y=bool(preset["standardize_y"]),
                    )
                    Q = int(args.uci_Q if args.uci_Q > 0 else data["K"])
                    n_final = min(int(preset["n"]), int(X_train_s.shape[0]))
                    n_candidate = min(max(int(args.n_candidate), n_final, int(args.n_vem)), int(X_train_s.shape[0]))
                    n_vem = min(int(args.n_vem), n_candidate)
                    row_base = _uci_row_base(
                        train_size=train_size,
                        dataset=dataset,
                        seed=seed,
                        n_train=X_train_s.shape[0],
                        n_test=X_test_s.shape[0],
                        Q=Q,
                        n_final=n_final,
                        n_candidate=n_candidate,
                        n_vem=n_vem,
                        standardize_x=bool(preset["standardize_x"]),
                        standardize_y=bool(preset["standardize_y"]),
                    )
                    mu, sigma, diag = _fit_vem_predict(
                        X_train_s=X_train_s,
                        y_train_s=y_train_s,
                        X_test_s=X_test_s,
                        Q=Q,
                        n_final=n_final,
                        n_candidate=n_candidate,
                        n_vem=n_vem,
                        args=args,
                        seed=seed,
                        device=device,
                    )
                    metrics, raw_metrics, metric_diag = _pack_metrics_with_numeric_filter(
                        y_scaler=y_scaler,
                        y_eval_scaled=y_test_s,
                        y_train_original=data["y_train"],
                        mu_scaled=mu,
                        sigma_scaled=sigma,
                        wall_sec=float(diag["total_wall_sec"]),
                        method=METHOD_NAME,
                        sigma_omit_factor=float(args.numeric_sigma_omit_factor),
                        max_omit_frac=float(args.max_numeric_omit_frac),
                    )
                    row = {**row_base, "status": "ok"}
                    for k, v in zip(RESULT_ROW_KEYS, metrics):
                        row[k] = float(v)
                    row.update(
                        {
                            "metric_policy": metric_diag["metric_policy"],
                            "n_eval_used": int(metric_diag["n_eval_used"]),
                            "n_numeric_omitted": int(metric_diag["n_numeric_omitted"]),
                            "train_sec": float(diag["train_sec"]),
                            "pred_sec": float(diag["pred_sec"]),
                            "best_loss": float(diag["best_loss"]),
                            "gamma_mean": float(diag["gamma_mean"]),
                            "gamma_entropy": float(diag["gamma_entropy"]),
                            "kl_coeff": float(diag["kl_coeff"]),
                            "C_std_median": float(diag["C_std_median"]),
                            "raw_rmse": float(raw_metrics[0]),
                            "raw_mean_crps": float(raw_metrics[1]),
                        }
                    )
                    rows.append(row)
                    print(
                        f"  ok rmse={row['rmse']:.4f} crps={row['mean_crps']:.4f} "
                        f"cov90={row['coverage_90']:.3f} wall={row['wall_sec']:.1f}s "
                        f"policy={row['metric_policy']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    row = row_base or {
                        "benchmark": "uci",
                        "train_size": int(train_size),
                        "dataset": dataset,
                        "subset": "all",
                        "seed": int(seed),
                        "method": METHOD_NAME,
                    }
                    row.update({"status": "error", "error": repr(exc)})
                    for k in RESULT_ROW_KEYS:
                        row[k] = float("nan")
                    rows.append(row)
                    print(f"  ERROR {exc!r}", flush=True)
                _flush_outputs(out_dir, args, rows)


def _pde_row_metrics(y, mu, sigma, elapsed, mask: np.ndarray) -> list[float]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return [float("nan")] * len(RESULT_ROW_KEYS)
    y_m = np.asarray(y, dtype=np.float64).reshape(-1)[mask]
    mu_m = np.asarray(mu, dtype=np.float64).reshape(-1)[mask]
    sig_m = np.asarray(sigma, dtype=np.float64).reshape(-1)[mask]
    finite = np.isfinite(y_m) & np.isfinite(mu_m) & np.isfinite(sig_m) & (sig_m > 0.0)
    y_m, mu_m, sig_m = y_m[finite], mu_m[finite], sig_m[finite]
    if y_m.size == 0:
        return [float("nan")] * len(RESULT_ROW_KEYS)
    rmse = float(np.sqrt(np.mean((mu_m - y_m) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y_m, dtype=torch.float32),
            torch.as_tensor(mu_m, dtype=torch.float32),
            torch.as_tensor(sig_m, dtype=torch.float32),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, elapsed, y_m, mu_m, sig_m)


def _run_pdebench(args: argparse.Namespace, device: torch.device, out_dir: Path, rows: list[dict[str, Any]]) -> None:
    completed = {
        (str(r.get("benchmark")), str(r.get("dataset")), str(r.get("subset")), int(r.get("seed", -1)))
        for r in rows
        if r.get("status") == "ok" and str(r.get("benchmark")) == "pdebench" and str(r.get("seed", "")).strip()
    }
    dataset_obj = _load_pdebench_dataset(args.pdebench_dataset_folder)
    X_train, y_train, X_test_full, y_test_full = _pde_dataset_arrays(dataset_obj, max_anchors=None)
    X_train, y_train, X_test, y_test = _pde_dataset_arrays(dataset_obj, max_anchors=args.pdebench_max_anchors)
    masks = _pde_threshold_masks(dataset_obj, len(X_test_full), len(X_test))
    Q = min(int(args.pdebench_Q), X_train.shape[1])
    n_final = min(int(args.pdebench_n), int(X_train.shape[0]))
    n_candidate = min(max(int(args.n_candidate), n_final, int(args.n_vem)), int(X_train.shape[0]))
    n_vem = min(int(args.n_vem), n_candidate)
    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
        if args.resume and all(("pdebench", "PDEBench-Burgers", split, seed) in completed for split in PDE_SPLITS if split in masks):
            print(f"skip completed PDEBench seed={seed}", flush=True)
            continue
        _set_seed(seed)
        try:
            print(f"[PDEBench] seed={seed}", flush=True)
            mu, sigma, diag = _fit_vem_predict(
                X_train_s=X_train,
                y_train_s=y_train,
                X_test_s=X_test,
                Q=Q,
                n_final=n_final,
                n_candidate=n_candidate,
                n_vem=n_vem,
                args=args,
                seed=seed,
                device=device,
            )
            for split in PDE_SPLITS:
                if split not in masks:
                    continue
                metric = _pde_row_metrics(y_test, mu, sigma, float(diag["total_wall_sec"]), masks[split])
                row = {
                    "benchmark": "pdebench",
                    "train_size": "",
                    "dataset": "PDEBench-Burgers",
                    "subset": split,
                    "seed": int(seed),
                    "method": METHOD_NAME,
                    "status": "ok",
                    "n_train": int(X_train.shape[0]),
                    "n_test": int(np.asarray(masks[split], dtype=bool).sum()),
                    "Q": int(Q),
                    "n_final": int(n_final),
                    "n_candidate": int(n_candidate),
                    "n_vem": int(n_vem),
                    "standardize_x": "",
                    "standardize_y": "",
                    "metric_policy": "raw",
                    "n_eval_used": int(np.asarray(masks[split], dtype=bool).sum()),
                    "n_numeric_omitted": 0,
                    "train_sec": float(diag["train_sec"]),
                    "pred_sec": float(diag["pred_sec"]),
                    "best_loss": float(diag["best_loss"]),
                    "gamma_mean": float(diag["gamma_mean"]),
                    "gamma_entropy": float(diag["gamma_entropy"]),
                    "kl_coeff": float(diag["kl_coeff"]),
                    "C_std_median": float(diag["C_std_median"]),
                }
                for k, v in zip(RESULT_ROW_KEYS, metric):
                    row[k] = float(v)
                rows.append(row)
            print(
                f"  ok all_rmse={rows[-3]['rmse']:.4f} all_crps={rows[-3]['mean_crps']:.4f} "
                f"wall={diag['total_wall_sec']:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            for split in PDE_SPLITS:
                if split not in masks:
                    continue
                row = {
                    "benchmark": "pdebench",
                    "train_size": "",
                    "dataset": "PDEBench-Burgers",
                    "subset": split,
                    "seed": int(seed),
                    "method": METHOD_NAME,
                    "status": "error",
                    "error": repr(exc),
                }
                for k in RESULT_ROW_KEYS:
                    row[k] = float("nan")
                rows.append(row)
            print(f"  ERROR {exc!r}", flush=True)
        _flush_outputs(out_dir, args, rows)


def _write_readme(out_dir: Path, args: argparse.Namespace, agg: list[dict[str, Any]]) -> None:
    lines = [
        "# VEM-EM UCI/PDEBench Comparison",
        "",
        "This folder was produced by `experiments/vem/compare_vem_em_uci_pdebench.py`.",
        "",
        "Method: `vem_em_learned_qw`, using conditional-W exact local JGP-VEM, posterior q(W) samples, and projected-space final JumpGP neighborhoods.",
        "",
        "## Aggregate",
        "",
        "| benchmark | train | dataset | subset | seeds | RMSE | CRPS | Cov90 | W90 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        train = row.get("train_size", "")
        lines.append(
            f"| {row['benchmark']} | {train} | {row['dataset']} | {row['subset']} | "
            f"{row['n_seeds']} | {row['rmse_mean']:.4f} | {row['mean_crps_mean']:.4f} | "
            f"{row['coverage_90_mean']:.3f} | {row['mean_width_90_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Runner Args",
            "",
            "```json",
            json.dumps(sanitize_for_json(vars(args)), indent=2),
            "```",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _flush_outputs(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    _write_csv(out_dir / "metrics_partial.csv", rows)
    agg = _aggregate(rows)
    _write_csv(out_dir / "aggregate.csv", agg)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            sanitize_for_json(
                {
                    "args": vars(args),
                    "n_rows": len(rows),
                    "n_ok": sum(1 for row in rows if row.get("status") == "ok"),
                    "aggregate": agg,
                }
            ),
            f,
            indent=2,
        )
    _write_readme(out_dir, args, agg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VEM-EM on UCI small-train and PDEBench Burgers.")
    p.add_argument("--benchmarks", type=str, default="uci,pdebench", help="Comma list: uci,pdebench")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out_dir", type=str, default="experiments/vem/vem_em_uci_pdebench_5seeds")

    p.add_argument("--uci_train_sizes", type=str, default="200,500,1000")
    p.add_argument("--uci_datasets", type=str, default=UCI_DATASETS_DEFAULT)
    p.add_argument("--uci_Q", type=int, default=-1)
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--max_train_anchors", type=int, default=128)
    p.add_argument("--train_subject_frac", type=float, default=0.8)
    p.add_argument("--numeric_sigma_omit_factor", type=float, default=50.0)
    p.add_argument("--max_numeric_omit_frac", type=float, default=0.02)

    p.add_argument("--pdebench_dataset_folder", type=str, default="data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000")
    p.add_argument("--pdebench_Q", type=int, default=3)
    p.add_argument("--pdebench_n", type=int, default=25)
    p.add_argument("--pdebench_max_anchors", type=int, default=None)

    p.add_argument("--n_candidate", type=int, default=100)
    p.add_argument("--n_vem", type=int, default=50)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--mc_train", type=int, default=2)
    p.add_argument("--mc_gamma", type=int, default=2)
    p.add_argument("--eval_mc", type=int, default=2)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--init_log_std", type=float, default=-3.0)
    p.add_argument("--prior_std", type=float, default=1.0)
    p.add_argument("--kl_weight", type=float, default=0.05)
    p.add_argument("--kl_warmup_steps", type=int, default=60)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_coeff", type=float, default=1e-4)
    p.add_argument("--lambda_gate_l2", type=float, default=1e-4)
    p.add_argument("--gamma_init", type=float, default=0.7)
    p.add_argument("--gamma_floor", type=float, default=1e-3)
    p.add_argument("--gamma_damping", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--temperature_final", type=float, default=3.0)
    p.add_argument("--init_lengthscale", type=float, default=1.0)
    p.add_argument("--init_amp", type=float, default=1.0)
    p.add_argument("--init_noise_var", type=float, default=0.1)
    p.add_argument("--outlier_sigma_mult", type=float, default=2.5)
    p.add_argument("--jitter", type=float, default=1e-5)
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--use_global_mean_projection", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(out_dir / "metrics_partial.csv") if args.resume else []
    if args.resume and not rows:
        rows = _read_csv(out_dir / "metrics.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"writing artifacts to {out_dir}", flush=True)

    benchmarks = {s.strip().lower() for s in args.benchmarks.split(",") if s.strip()}
    unknown = benchmarks - {"uci", "pdebench"}
    if unknown:
        raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")
    if "uci" in benchmarks:
        _run_uci(args, device, out_dir, rows)
    if "pdebench" in benchmarks:
        _run_pdebench(args, device, out_dir, rows)
    _write_csv(out_dir / "metrics.csv", rows)
    _flush_outputs(out_dir, args, rows)
    print(f"saved artifacts to {out_dir}", flush=True)
    return {"aggregate": _aggregate(rows), "rows": rows}


if __name__ == "__main__":
    main()

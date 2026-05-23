"""Compare direct DMGP against a fixed-W residual uncertain self-CEM head.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_dmgp_uncertain_w_cem_head.py ^
        --bundle_dir experiments/synthetic/dmgp_residual_bundle_lh_seed0_20260517 ^
        --settings lh_q5_train1000 --seed 0 --max_test 200 ^
        --variants fit,oof ^
        --out_dir experiments/synthetic/dmgp_uncertain_w_cem_head_lh_seed0_20260517

This consumes the bundle exported by
``run_deep_mahalanobis_gp_residual_bundle.py``.  The local head keeps the
DMGP ``W`` layer fixed, residualizes ``y`` with DMGP mean predictions, injects
DMGP predictive variance on the neighborhood diagonal, and optimizes only the
uncertain self-CEM head on raw-X neighborhoods.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.projections.global_residual_selfcem import (  # noqa: E402
    _initial_residual_labels,
    _meanw_default_hypers,
    _neighbor_stack,
    _predict_per_anchor_with_extras,
    _run_local_self_cem_with_extras,
)
from djgp.projections.uncertain_w_jgp import (  # noqa: E402
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "setting",
        "family",
        "seed",
        "variant",
        "status",
        "N_train",
        "N_test",
        "n_test_eval",
        "Q",
        "observed_dim",
        "latent_dim",
        "n_neighbors",
        "global_source",
        *RESULT_ROW_KEYS,
        "direct_rmse",
        "direct_mean_crps",
        "local_self_cem_sec",
        "pred_sec",
        "global_v_g_train_mean",
        "global_v_g_test_mean",
        "w_var_mean",
        "mean_n_inliers",
        "final_inlier_rate",
        "mean_lengthscale",
        "mean_signal_var",
        "mean_noise_var",
        "fallback_count",
        "mean_kernel_vector_correction",
        "error",
    ]
    keys: list[str] = []
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


def _float_or_nan(x: Any) -> float:
    try:
        if x == "":
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["setting"]), str(row["variant"])), []).append(row)
    out: list[dict[str, Any]] = []
    value_keys = list(RESULT_ROW_KEYS) + [
        "direct_rmse",
        "direct_mean_crps",
        "local_self_cem_sec",
        "pred_sec",
        "global_v_g_train_mean",
        "global_v_g_test_mean",
        "w_var_mean",
        "mean_n_inliers",
        "final_inlier_rate",
        "mean_lengthscale",
        "mean_signal_var",
        "mean_noise_var",
        "fallback_count",
        "mean_kernel_vector_correction",
    ]
    for (setting, variant), group in sorted(groups.items()):
        row: dict[str, Any] = {"setting": setting, "variant": variant, "n_runs": len(group)}
        for meta in (
            "family",
            "N_train",
            "N_test",
            "n_test_eval",
            "Q",
            "observed_dim",
            "latent_dim",
            "n_neighbors",
            "global_source",
        ):
            row[meta] = group[0].get(meta, "")
        for key in value_keys:
            vals = np.asarray([_float_or_nan(r.get(key)) for r in group], dtype=np.float64)
            if np.isfinite(vals).any():
                row[f"{key}_mean"] = float(np.nanmean(vals))
                row[f"{key}_std"] = float(np.nanstd(vals))
        out.append(row)
    return out


def _metrics_pack(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> list[float]:
    rmse = float(np.sqrt(np.mean((np.asarray(mu).reshape(-1) - np.asarray(y).reshape(-1)) ** 2)))
    return pack_gaussian_uq_list(rmse, float("nan"), wall_sec, y, mu, sigma)


def _gaussian_pack(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> list[float]:
    from djgp.evaluation.crps import mean_gaussian_crps_torch

    y_np = np.asarray(y, dtype=np.float64).reshape(-1)
    mu_np = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma_np = np.maximum(np.asarray(sigma, dtype=np.float64).reshape(-1), 1e-12)
    rmse = float(np.sqrt(np.mean((mu_np - y_np) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y_np, dtype=torch.float32),
            torch.as_tensor(mu_np, dtype=torch.float32),
            torch.as_tensor(sigma_np, dtype=torch.float32),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, wall_sec, y_np, mu_np, sigma_np)


def _broadcast_w_var(W_mean: torch.Tensor, W_var_row: torch.Tensor) -> torch.Tensor:
    return W_var_row.unsqueeze(-1).expand_as(W_mean).contiguous()


def _base_row(setting: str, cfg: dict[str, Any], seed: int, variant: str, n_eval: int, n_neighbors: int) -> dict[str, Any]:
    return {
        "setting": setting,
        "family": cfg.get("family", ""),
        "seed": int(seed),
        "variant": variant,
        "status": "ok",
        "N_train": int(cfg.get("N_train", 0)),
        "N_test": int(cfg.get("N_test", 0)),
        "n_test_eval": int(n_eval),
        "Q": int(cfg.get("Q", 0)),
        "observed_dim": int(cfg.get("H", 0)),
        "latent_dim": int(cfg.get("d", 0)),
        "n_neighbors": int(n_neighbors),
    }


def _run_one_variant(
    *,
    setting: str,
    seed: int,
    cfg: dict[str, Any],
    variant: str,
    bundle_dir: Path,
    max_test: int | None,
    n_neighbors: int,
    device: torch.device,
    args: argparse.Namespace,
    include_direct: bool,
) -> list[dict[str, Any]]:
    arr = np.load(bundle_dir / f"{setting}_seed{seed}.npz")
    X_train = np.asarray(arr["X_train"], dtype=np.float64)
    X_test = np.asarray(arr["X_test"], dtype=np.float64)
    y_train_std = np.asarray(arr["y_train_std"], dtype=np.float64).reshape(-1)
    y_test = np.asarray(arr["y_test"], dtype=np.float64).reshape(-1)
    y_mean = float(np.asarray(arr["y_mean"]).reshape(-1)[0])
    y_scale = float(np.asarray(arr["y_scale"]).reshape(-1)[0])
    if abs(y_scale) <= 1e-12:
        y_scale = 1.0

    prefix = bundle_dir / f"{setting}_seed{seed}"
    mu_g_test_std = np.load(str(prefix) + "_test_mean_std.npy").reshape(-1)
    v_g_test_std = np.load(str(prefix) + "_test_var_std.npy").reshape(-1)
    pred_mean = np.load(str(prefix) + "_pred_mean.npy").reshape(-1)
    pred_sigma = np.load(str(prefix) + "_pred_sigma.npy").reshape(-1)
    W_test_mean = np.load(str(prefix) + "_W_test_mean.npy")
    W_test_var = np.load(str(prefix) + "_W_test_var.npy")

    if max_test is not None:
        sl = slice(0, int(max_test))
        X_test = X_test[sl]
        y_test = y_test[sl]
        mu_g_test_std = mu_g_test_std[sl]
        v_g_test_std = v_g_test_std[sl]
        pred_mean = pred_mean[sl]
        pred_sigma = pred_sigma[sl]
        W_test_mean = W_test_mean[sl]
        W_test_var = W_test_var[sl]

    direct_pack = _gaussian_pack(y_test, pred_mean, pred_sigma, 0.0)
    direct_row = _base_row(setting, cfg, seed, "deep_mahalanobis_gp_direct", len(y_test), n_neighbors)
    direct_row.update({key: float(direct_pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
    direct_row["global_source"] = "full_fit"

    rows: list[dict[str, Any]] = [direct_row] if include_direct else []
    if variant == "fit":
        train_mean_path = str(prefix) + "_train_fit_mean_std.npy"
        train_var_path = str(prefix) + "_train_fit_var_std.npy"
        global_source = "full_fit"
        method_name = "dmgp_resid_uncertain_selfcem_fit"
    elif variant == "oof":
        train_mean_path = str(prefix) + "_train_oof_mean_std.npy"
        train_var_path = str(prefix) + "_train_oof_var_std.npy"
        global_source = "oof"
        method_name = "dmgp_resid_uncertain_selfcem_oof"
    else:
        raise ValueError(f"Unknown variant {variant!r}")
    if not Path(train_mean_path).exists() or not Path(train_var_path).exists():
        return rows

    mu_g_train_std = np.load(train_mean_path).reshape(-1)
    v_g_train_std = np.load(train_var_path).reshape(-1)

    X_train_t = torch.as_tensor(X_train, dtype=torch.float64, device=device)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float64, device=device)
    y_train_std_t = torch.as_tensor(y_train_std, dtype=torch.float64, device=device)
    y_train_resid_t = y_train_std_t - torch.as_tensor(mu_g_train_std, dtype=torch.float64, device=device)

    mu_g_test_t = torch.as_tensor(mu_g_test_std, dtype=torch.float64, device=device)
    v_g_test_t = torch.as_tensor(v_g_test_std, dtype=torch.float64, device=device).clamp_min(0.0)
    v_g_train_t = torch.as_tensor(v_g_train_std, dtype=torch.float64, device=device).clamp_min(0.0)
    W_mu_anchor = torch.as_tensor(W_test_mean, dtype=torch.float64, device=device)
    W_var_anchor = _broadcast_w_var(
        W_mu_anchor,
        torch.as_tensor(W_test_var, dtype=torch.float64, device=device).clamp_min(0.0),
    )

    X_neighbors, y_resid_neighbors, test_neighbor_idx = _neighbor_stack(
        X_test_t,
        X_train_t,
        y_train_resid_t,
        n_neighbors=int(n_neighbors),
    )
    v_g_neighbors = v_g_train_t[test_neighbor_idx]
    init_labels = _initial_residual_labels(
        X_test_t,
        X_neighbors,
        y_resid_neighbors,
        min_inliers=int(args.min_inliers),
    )
    W0_init = W_mu_anchor.mean(dim=0)
    defaults = _meanw_default_hypers(
        X_test_t,
        X_neighbors,
        y_resid_neighbors,
        init_labels,
        W0_init,
        min_inliers=int(args.min_inliers),
    )
    min_noise_var_per_anchor = torch.full(
        (int(X_test_t.shape[0]),),
        max(float(args.min_noise_var), float(args.local_noise_std_floor) ** 2),
        device=device,
        dtype=torch.float64,
    )
    cem_cfg = type("TmpCfg", (), {})()
    cem_cfg.cem_updates = int(args.cem_updates)
    cem_cfg.hyperopt = FixedLabelHyperoptConfig(
        steps=int(args.hyper_steps),
        lr=float(args.hyper_lr),
        jitter=1e-5,
        min_noise_var=float(args.min_noise_var),
        max_noise_var=float(args.max_noise_var),
        min_signal_var=float(args.min_signal_var),
        max_signal_var=float(args.max_signal_var),
        min_lengthscale=float(args.min_lengthscale),
        max_lengthscale=float(args.max_lengthscale),
    )
    cem_cfg.gate = FixedLabelGateConfig(
        steps=int(args.gate_steps),
        lr=float(args.gate_lr),
        l2=float(args.gate_l2),
        max_norm=float(args.gate_max_norm),
        gh_points=int(args.gh_points),
    )
    cem_cfg.min_inliers = int(args.min_inliers)
    cem_cfg.max_inlier_frac = float(args.max_inlier_frac)
    cem_cfg.outlier_sigma_mult = float(args.outlier_sigma_mult)
    cem_cfg.background_scale_mult = float(args.background_scale_mult)

    t_cem = time.perf_counter()
    state = _run_local_self_cem_with_extras(
        X_anchor=X_test_t,
        X_neighbors=X_neighbors,
        y_resid_neighbors=y_resid_neighbors,
        init_labels=init_labels,
        W_mu_anchor=W_mu_anchor,
        W_var_anchor=W_var_anchor,
        init_lengthscale=defaults["lengthscale"],
        init_signal_var=defaults["signal_var"],
        init_noise_var=defaults["noise_var"].clamp_min(min_noise_var_per_anchor),
        min_noise_var_per_anchor=min_noise_var_per_anchor,
        extra_noise_diag=v_g_neighbors,
        extra_block_cov=None,
        cfg=cem_cfg,
    )
    cem_sec = float(time.perf_counter() - t_cem)
    t_pred = time.perf_counter()
    out = _predict_per_anchor_with_extras(
        X_anchor=X_test_t,
        X_neighbors=X_neighbors,
        y_resid_neighbors=y_resid_neighbors,
        labels=state["labels"],
        W_mu_anchor=W_mu_anchor,
        W_var_anchor=W_var_anchor,
        lengthscale=state["lengthscale"],
        signal_var=state["signal_var"],
        noise_var=state["noise_var"],
        mean_const=state["mean_const"],
        extra_noise_diag=v_g_neighbors,
        extra_block_cov=None,
        correction_weight=float(args.kcov),
        jitter=1e-5,
        min_inliers=int(args.min_inliers),
    )
    pred_sec = float(time.perf_counter() - t_pred)

    mu_y_std = (mu_g_test_t + out["mu"]).detach().cpu().numpy().reshape(-1)
    var_y_std = (v_g_test_t + out["var"]).clamp_min(float(args.min_noise_var)).detach().cpu().numpy().reshape(-1)
    mu_y = mu_y_std * y_scale + y_mean
    sigma_y = np.sqrt(var_y_std) * abs(y_scale)
    pack = _gaussian_pack(y_test, mu_y, sigma_y, cem_sec + pred_sec)

    row = _base_row(setting, cfg, seed, method_name, len(y_test), n_neighbors)
    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
    row.update(
        {
            "global_source": global_source,
            "direct_rmse": float(direct_row["rmse"]),
            "direct_mean_crps": float(direct_row["mean_crps"]),
            "local_self_cem_sec": float(cem_sec),
            "pred_sec": float(pred_sec),
            "global_v_g_train_mean": float(v_g_train_t.mean().item()),
            "global_v_g_test_mean": float(v_g_test_t.mean().item()),
            "w_var_mean": float(torch.as_tensor(W_test_var, dtype=torch.float64).mean().item()),
            "mean_n_inliers": float(out["n_inliers"].to(dtype=torch.float64).mean().item()),
            "final_inlier_rate": float(state["labels"].to(dtype=torch.float64).mean().item()),
            "mean_lengthscale": float(state["lengthscale"].mean().item()),
            "mean_signal_var": float(state["signal_var"].mean().item()),
            "mean_noise_var": float(state["noise_var"].mean().item()),
            "fallback_count": int(out["fallback_count"]),
            "mean_kernel_vector_correction": float(out["mean_kernel_vector_correction"]),
        }
    )
    rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DMGP residual uncertain self-CEM head comparison.")
    p.add_argument("--bundle_dir", default="experiments/synthetic/dmgp_residual_bundle_lh_seed0_20260517")
    p.add_argument("--out_dir", default="experiments/synthetic/dmgp_uncertain_w_cem_head_lh_seed0_20260517")
    p.add_argument("--settings", default="lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="", help="Optional comma-separated seed list. Overrides --seed when set.")
    p.add_argument("--variants", default="fit,oof")
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--cem_updates", type=int, default=2)
    p.add_argument("--kcov", type=float, default=0.1)
    p.add_argument("--local_noise_std_floor", type=float, default=0.1)
    p.add_argument("--hyper_steps", type=int, default=10)
    p.add_argument("--hyper_lr", type=float, default=0.04)
    p.add_argument("--gate_steps", type=int, default=20)
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
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = [s.strip() for s in str(args.settings).split(",") if s.strip()]
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()] if args.seeds else [int(args.seed)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for setting in settings:
            direct_written = False
            cfg = dict(SETTING_PRESETS[setting])
            for variant in variants:
                try:
                    rows.extend(
                        _run_one_variant(
                            setting=setting,
                            seed=seed,
                            cfg=cfg,
                            variant=variant,
                            bundle_dir=bundle_dir,
                            max_test=args.max_test,
                            n_neighbors=int(args.n_neighbors or cfg.get("n", 35)),
                            device=device,
                            args=args,
                            include_direct=not direct_written,
                        )
                    )
                    direct_written = True
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            "setting": setting,
                            "seed": int(seed),
                            "variant": variant,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                _write_csv(out_dir / "metrics_partial.csv", rows)
                _write_csv(out_dir / "aggregate_partial.csv", _aggregate(rows))
    _write_csv(out_dir / "metrics.csv", rows)
    agg = _aggregate(rows)
    _write_csv(out_dir / "aggregate.csv", agg)
    summary = {
        "args": vars(args),
        "device": str(device),
        "settings": {s: SETTING_PRESETS[s] for s in settings},
        "n_rows": len(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = f"""# DMGP Residual Uncertain Self-CEM Head

This folder compares direct DeepMahalanobisGP prediction against a residual
uncertain self-CEM head that keeps the DMGP W layer fixed and injects DMGP
predictive variance on the local covariance diagonal.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

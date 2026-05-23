"""Run global GP + residual self-CEM variants on paper L2/LH synthetic data.

This runner implements ``docs/residual_ablation.md`` on top of the best 5-seed
LH self-CEM baseline (``self_cem2_nstd0.1_kcov0.1`` from the
``uncertain_w_cem_qr_hybrid_ensemble_lh_5seeds_20260517`` log entry).  The
local CEM stays at ``cem_updates=2`` with ``local_noise_std_floor=0.1`` and
``kcov=0.1``; only the global mean component changes between variants.

Backends for the global GP are exposed via ``--global_backend`` so the user
can compare standard sparse SVGP against deep kernel learning (DKL) -- a fast
alternative to a full GP fit on the H=30 dimensional LH features -- and an
optional 2-layer Deep GP.

How to run from the repository root with conda env ``jumpGP``::

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke check (single seed, two variants).
    python experiments/synthetic/compare_global_residual_selfcem_lh.py ^
        --settings lh_q5_train1000 ^
        --variants diagvar,blockvar ^
        --num_exp 1 --max_test 40 ^
        --global_backend sparse_svgp --global_num_inducing 64 --global_epochs 60 ^
        --out_dir experiments/synthetic/residual_ablation_lh_smoke

    # Full 5-seed LH run with all five recommended variants and sparse SVGP global GP.
    python experiments/synthetic/compare_global_residual_selfcem_lh.py ^
        --settings lh_q5_train1000 ^
        --variants diagvar,blockvar,oob_diagvar,alt,alt_ensW ^
        --num_exp 5 --max_test 200 ^
        --global_backend sparse_svgp --global_num_inducing 128 --global_epochs 200 ^
        --resume ^
        --out_dir experiments/synthetic/residual_ablation_lh_5seeds

    # DKL-global ablation for the same variants (the deep kernel learns a
    # nonlinear feature map and is typically faster than a full GP on the
    # H=30 LH features for N_train=1000).
    python experiments/synthetic/compare_global_residual_selfcem_lh.py ^
        --settings lh_q5_train1000 ^
        --variants diagvar,alt ^
        --num_exp 5 --max_test 200 ^
        --global_backend dkl --global_num_inducing 128 --global_epochs 150 ^
        --dkl_hidden 64 --dkl_latent 8 ^
        --resume ^
        --out_dir experiments/synthetic/residual_ablation_lh_dkl_5seeds
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.baselines import GLOBAL_GP_BACKENDS, GlobalGPConfig  # noqa: E402
from djgp.diagnostics import sanitize_for_json  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import (  # noqa: E402
    GlobalResidualSelfCEMConfig,
    GLOBAL_RESIDUAL_VARIANTS,
    run_global_residual_self_cem,
)
from djgp.projections.uncertain_w_jgp import (  # noqa: E402
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


DEFAULT_VARIANTS = (
    "diagvar",
    "blockvar",
    "oob_diagvar",
    "alt",
    "alt_ensW",
)


# ---------------------------------------------------------------------------
# Logging / aggregation helpers shared with compare_uncertain_w_cem_l2_lh.py.
# ---------------------------------------------------------------------------


def _metrics_pack(
    y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float
) -> list[float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64).reshape(-1), 1e-12)
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, wall_sec, y, mu, sigma)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _float_or_nan(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if v != v:
        return float("nan")
    return v


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        groups.setdefault((str(r["setting"]), str(r["variant"])), []).append(r)
    out: list[dict[str, Any]] = []
    value_keys = list(RESULT_ROW_KEYS) + [
        "rmse",
        "mean_crps",
        "coverage_90",
        "mean_width_90",
        "coverage_95",
        "mean_width_95",
        "wall_sec",
        "global_gp_train_sec",
        "global_gp_total_sec",
        "global_gp_final_loss",
        "global_v_g_train_mean",
        "global_v_g_test_mean",
        "global_resid_train_std",
        "local_self_cem_sec",
        "mean_n_inliers",
        "final_inlier_rate",
        "mean_lengthscale",
        "mean_signal_var",
        "mean_noise_var",
        "fallback_count",
        "final_mu_g_mean",
        "final_mu_h_mean",
        "final_v_g_mean",
        "final_v_h_mean",
        "final_v_y_mean",
        "ensemble_within_var_mean",
        "ensemble_between_var_mean",
    ]
    for (setting, variant), group in sorted(groups.items()):
        agg: dict[str, Any] = {
            "setting": setting,
            "variant": variant,
            "n_runs": len(group),
        }
        for meta in (
            "family",
            "global_backend",
            "global_num_inducing",
            "global_epochs",
            "N_train",
            "N_test",
            "n_test_eval",
            "Q",
            "observed_dim",
            "latent_dim",
            "n_neighbors",
            "w_var",
            "cem_updates",
            "local_noise_std_floor",
            "kcov",
            "alt_outer_cycles",
            "oob_k_folds",
            "ensemble_samples",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in value_keys:
            vals = np.asarray([_float_or_nan(r.get(key)) for r in group], dtype=np.float64)
            if np.isfinite(vals).any():
                agg[f"{key}_mean"] = float(np.nanmean(vals))
                agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


# ---------------------------------------------------------------------------
# Variant -> config construction
# ---------------------------------------------------------------------------


def _build_residual_cfg(args: argparse.Namespace, variant: str) -> GlobalResidualSelfCEMConfig:
    if variant not in GLOBAL_RESIDUAL_VARIANTS:
        raise ValueError(
            f"Unknown residual variant {variant!r}; allowed: {GLOBAL_RESIDUAL_VARIANTS}."
        )
    hyperopt = FixedLabelHyperoptConfig(
        steps=int(args.hyper_steps),
        lr=float(args.hyper_lr),
        jitter=1e-5,
        min_noise_var=float(args.hyperopt_min_noise_var),
        max_noise_var=float(args.hyperopt_max_noise_var),
        min_signal_var=float(args.hyperopt_min_signal_var),
        max_signal_var=float(args.hyperopt_max_signal_var),
        min_lengthscale=float(args.hyperopt_min_lengthscale),
        max_lengthscale=float(args.hyperopt_max_lengthscale),
    )
    gate = FixedLabelGateConfig(
        steps=int(args.gate_steps),
        lr=float(args.gate_lr),
        l2=float(args.gate_l2),
        max_norm=float(args.gate_max_norm),
        gh_points=int(args.gh_points),
    )
    return GlobalResidualSelfCEMConfig(
        variant=variant,
        n_neighbors=int(args.n_neighbors),
        w_var=float(args.w_var),
        cem_updates=int(args.cem_updates),
        kcov=float(args.kcov),
        local_noise_std_floor=float(args.local_noise_std_floor),
        hyperopt=hyperopt,
        gate=gate,
        min_inliers=int(args.min_inliers),
        max_inlier_frac=float(args.max_inlier_frac),
        outlier_sigma_mult=float(args.outlier_sigma_mult),
        background_scale_mult=float(args.background_scale_mult),
        alt_outer_cycles=int(args.alt_outer_cycles),
        oob_k_folds=int(args.oob_k_folds),
        ensemble_samples=int(args.ensemble_samples),
        block_variance_scale=float(args.block_variance_scale),
        diag_variance_inflation=float(args.diag_variance_inflation),
        use_oob_for_test=bool(args.use_oob_for_test),
        alt_train_anchor_subsample=int(args.alt_train_anchor_subsample),
        alt_train_hyper_steps=(
            None if int(args.alt_train_hyper_steps) <= 0 else int(args.alt_train_hyper_steps)
        ),
        alt_train_gate_steps=(
            None if int(args.alt_train_gate_steps) <= 0 else int(args.alt_train_gate_steps)
        ),
    )


def _build_global_cfg(args: argparse.Namespace) -> GlobalGPConfig:
    if args.global_backend not in GLOBAL_GP_BACKENDS:
        raise ValueError(
            f"Unknown --global_backend={args.global_backend!r}; allowed: {GLOBAL_GP_BACKENDS}."
        )
    return GlobalGPConfig(
        backend=str(args.global_backend),
        num_inducing=int(args.global_num_inducing),
        epochs=int(args.global_epochs),
        batch_size=int(args.global_batch_size),
        lr=float(args.global_lr),
        weight_decay=float(args.global_weight_decay),
        init_noise_var=float(args.global_init_noise_var),
        standardize_y=True,
        dkl_hidden=int(args.dkl_hidden),
        dkl_latent=int(args.dkl_latent),
        deepgp_hidden_dim=int(args.deepgp_hidden_dim),
        deepgp_num_inducing_inner=int(args.deepgp_num_inducing_inner),
        deepgp_num_samples=int(args.deepgp_num_samples),
        jitter=float(args.global_jitter),
        verbose=bool(args.global_verbose),
    )


# ---------------------------------------------------------------------------
# Per-row execution
# ---------------------------------------------------------------------------


def _base_row(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    setting: str,
    seed: int,
    variant: str,
) -> dict[str, Any]:
    return {
        "setting": setting,
        "family": str(cfg.get("family", "")),
        "variant": variant,
        "seed": int(seed),
        "status": "pending",
        "global_backend": str(args.global_backend),
        "global_num_inducing": int(args.global_num_inducing),
        "global_epochs": int(args.global_epochs),
        "N_train": int(cfg["N_train"]),
        "N_test": int(cfg["N_test"]),
        "n_test_eval": int(min(int(args.max_test), int(cfg["N_test"]))) if args.max_test else int(cfg["N_test"]),
        "Q": int(cfg["Q"]),
        "observed_dim": int(cfg.get("H", -1)),
        "latent_dim": int(cfg.get("d", -1)),
        "n_neighbors": int(args.n_neighbors),
        "w_var": float(args.w_var),
        "cem_updates": int(args.cem_updates),
        "local_noise_std_floor": float(args.local_noise_std_floor),
        "kcov": float(args.kcov),
        "alt_outer_cycles": int(args.alt_outer_cycles),
        "oob_k_folds": int(args.oob_k_folds),
        "ensemble_samples": int(args.ensemble_samples),
    }


def _run_one(
    args: argparse.Namespace,
    setting: str,
    seed: int,
    variant: str,
    device: torch.device,
) -> dict[str, Any]:
    cfg = dict(SETTING_PRESETS[setting])
    _set_seed(seed)
    row = _base_row(args, cfg, setting, seed, variant)
    try:
        data = _generate_paper_data(cfg, seed, device)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(data["X_train"]).astype(np.float64)
        X_test = scaler.transform(data["X_test"]).astype(np.float64)
        y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
        y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)
        if args.max_test is not None:
            X_test = X_test[: int(args.max_test)]
            y_test = y_test[: int(args.max_test)]

        residual_cfg = _build_residual_cfg(args, variant)
        global_cfg = _build_global_cfg(args)

        t_total = time.perf_counter()
        result = run_global_residual_self_cem(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            Q=int(cfg["Q"]),
            cfg=residual_cfg,
            global_gp_cfg=global_cfg,
            device=device,
            seed=int(seed),
        )
        wall_sec = float(time.perf_counter() - t_total)

        pack = _metrics_pack(y_test, result.mu_y, result.sigma_y, wall_sec)
        row["status"] = "ok"
        row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
        row.update({k: sanitize_for_json(v) for k, v in result.diagnostics.items()})
        row["wall_sec"] = wall_sec
        # Decomposition diagnostics (means already in diagnostics; add rmse pieces).
        row["mu_g_rmse"] = float(np.sqrt(np.mean((result.mu_g_test - y_test) ** 2)))
        row["mu_h_only_rmse"] = float(
            np.sqrt(np.mean((result.mu_h_test - (y_test - result.mu_g_test)) ** 2))
        )
        row["combined_rmse"] = float(np.sqrt(np.mean((result.mu_y - y_test) ** 2)))
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(
            f"FAILED setting={setting} seed={seed} variant={variant}: {row['error']}",
            flush=True,
        )
    return row


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Global GP + residual self-CEM ablation on paper L2/LH data."
    )
    p.add_argument("--settings", default="lh_q5_train1000")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--max_test", type=int, default=200)

    # Local self-CEM (matches the best LH 5-seed baseline by default).
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--w_var", type=float, default=0.02)
    p.add_argument("--cem_updates", type=int, default=2)
    p.add_argument("--kcov", type=float, default=0.1)
    p.add_argument("--local_noise_std_floor", type=float, default=0.1)
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
    p.add_argument("--hyperopt_min_noise_var", type=float, default=1e-5)
    p.add_argument("--hyperopt_max_noise_var", type=float, default=10.0)
    p.add_argument("--hyperopt_min_signal_var", type=float, default=1e-4)
    p.add_argument("--hyperopt_max_signal_var", type=float, default=100.0)
    p.add_argument("--hyperopt_min_lengthscale", type=float, default=1e-3)
    p.add_argument("--hyperopt_max_lengthscale", type=float, default=20.0)

    # Variant-specific.
    p.add_argument("--alt_outer_cycles", type=int, default=1)
    p.add_argument("--oob_k_folds", type=int, default=5)
    p.add_argument("--ensemble_samples", type=int, default=5)
    p.add_argument("--block_variance_scale", type=float, default=1.0)
    p.add_argument("--diag_variance_inflation", type=float, default=1.0)
    p.add_argument(
        "--use_oob_for_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When variant=oob_diagvar, average OOB fold-models at test time instead of refitting a full-data global GP.",
    )
    p.add_argument(
        "--alt_train_anchor_subsample",
        type=int,
        default=200,
        help="Variant D/E: subsample this many training points as residual-CEM anchors; 0 means use all N_train.",
    )
    p.add_argument(
        "--alt_train_hyper_steps",
        type=int,
        default=10,
        help="Variant D/E: hyperopt steps for the train-anchor residual CEM (cheaper than test).  0 means reuse --hyper_steps.",
    )
    p.add_argument(
        "--alt_train_gate_steps",
        type=int,
        default=20,
        help="Variant D/E: gate steps for the train-anchor residual CEM (cheaper than test).  0 means reuse --gate_steps.",
    )

    # Global GP backbone.
    p.add_argument(
        "--global_backend",
        choices=list(GLOBAL_GP_BACKENDS),
        default="sparse_svgp",
        help="Pick the global GP backbone: sparse SVGP, DKL-SVGP, or 2-layer DeepGP.",
    )
    p.add_argument("--global_num_inducing", type=int, default=128)
    p.add_argument("--global_epochs", type=int, default=200)
    p.add_argument("--global_batch_size", type=int, default=512)
    p.add_argument("--global_lr", type=float, default=0.01)
    p.add_argument("--global_weight_decay", type=float, default=0.0)
    p.add_argument("--global_init_noise_var", type=float, default=0.05)
    p.add_argument("--global_jitter", type=float, default=1e-4)
    p.add_argument("--global_verbose", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dkl_hidden", type=int, default=64)
    p.add_argument("--dkl_latent", type=int, default=8)
    p.add_argument("--deepgp_hidden_dim", type=int, default=4)
    p.add_argument("--deepgp_num_inducing_inner", type=int, default=64)
    p.add_argument("--deepgp_num_samples", type=int, default=8)

    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--out_dir",
        default="experiments/synthetic/residual_ablation_lh_5seeds",
    )
    return p.parse_args()


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Global GP + Residual Self-CEM LH Ablation

Implements ``docs/residual_ablation.md`` on top of the best LH 5-seed
self-CEM baseline (``self_cem2_nstd0.1_kcov0.1``).  Each variant adds a
global smooth-mean component on top of the local self-CEM:

- ``diagvar``     -- per-neighbor global posterior variances are added on the
  local covariance diagonal while the local CEM operates on residuals
  ``y_i - mu_g(x_i)``.
- ``blockvar``    -- the full global posterior block ``Sigma_g(X_a, X_a)``
  is added inside the local CEM kernel ``K_h + Sigma_g + sigma_a^2 I``.
- ``oob_diagvar`` -- K-fold OOB ``mu_g`` / ``v_g`` for training points to
  avoid in-sample residual shrinkage.
- ``alt``         -- alternates one global GP refit with heteroscedastic
  pseudo-noise ``sigma_eps^2 + v_h_i`` after the first local self-CEM round.
- ``alt_ensW``    -- ``alt`` + sampled deterministic-W self-CEM ensemble at
  prediction time (moment-matched mixture), with each sampled ``W`` re-ranking
  a raw candidate pool before the local refit.

Files:
- ``metrics.csv``: one row per setting/seed/variant.
- ``aggregate.csv``: mean/std across successful rows.
- ``summary.json``: exact runner arguments and setting metadata.

Project log entry: ``docs/experiments_log.md`` section
``2026-05-17 - residual_ablation_lh_5seeds``.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in str(args.settings).split(",") if s.strip()]
    variants = [s.strip() for s in str(args.variants).split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown_settings:
        raise ValueError(f"Unknown settings: {unknown_settings}")
    for variant in variants:
        if variant not in GLOBAL_RESIDUAL_VARIANTS:
            raise ValueError(
                f"Unknown residual variant {variant!r}; allowed: {GLOBAL_RESIDUAL_VARIANTS}."
            )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = (
        _read_csv(out_dir / "metrics.csv") if args.resume else []
    )
    done = {
        (r.get("setting"), int(r.get("seed", -1)), r.get("variant"))
        for r in rows
        if r.get("status") == "ok"
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
        for setting in settings:
            for variant in variants:
                key = (setting, seed, variant)
                if key in done:
                    print(
                        f"skip completed setting={setting} seed={seed} variant={variant}",
                        flush=True,
                    )
                    continue
                t_one = time.perf_counter()
                print(
                    f"run setting={setting} seed={seed} variant={variant}",
                    flush=True,
                )
                rows.append(_run_one(args, setting, seed, variant, device))
                _write_csv(out_dir / "metrics.csv", rows)
                _write_csv(out_dir / "aggregate.csv", _aggregate(rows))
                wall = float(time.perf_counter() - t_one)
                last_row = rows[-1]
                rmse_s = last_row.get("rmse", "?")
                print(
                    f"  status={last_row.get('status')} rmse={rmse_s} wall_sec={wall:.1f}",
                    flush=True,
                )
    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "args": vars(args),
        "settings": {name: SETTING_PRESETS[name] for name in settings},
        "variants": variants,
        "n_rows": len(rows),
        "n_ok": int(sum(1 for r in rows if r.get("status") == "ok")),
        "n_failed": int(sum(1 for r in rows if r.get("status") != "ok")),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    _write_readme(out_dir, sanitize_for_json(summary))
    return summary


if __name__ == "__main__":
    main()

"""LMJGP q(R)/q(V) initialisation ablation on paper-style L2/LH synthetic data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # One-seed smoke / screen.
    python experiments/synthetic/compare_lmjgp_r_init_paper.py ^
        --num_exp 1 ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants pca_unit_kl,pca_dmgp_kl,sir_kl ^
        --lmjgp_steps 100 ^
        --MC_num 2 ^
        --out_dir experiments/synthetic/lmjgp_r_init_pca_smoke_20260514

    # Five-seed train=1000 comparison against the existing LMJGP table.
    python experiments/synthetic/compare_lmjgp_r_init_paper.py ^
        --num_exp 5 ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants pca_unit_kl,pca_dmgp_kl ^
        --lmjgp_steps 300 ^
        --MC_num 3 ^
        --resume ^
        --out_dir experiments/synthetic/lmjgp_r_init_pca_train1000_5seeds_20260514

This runner is intentionally additive. It does not change the public LMJGP
interfaces or overwrite the existing paper baseline folders. The paper-style
synthetic provenance follows ``compare_paper_synthetic_baselines.py``:
L2/phantom and LH generators with RFF expansion and train-only X standardisation.

Variant names:

- ``pls_kl``: same PLS initialisation, KL warm-up, row-norm and auxiliary metric
  settings used by ``compare_paper_synthetic_baselines.py``.
- ``pca_unit_kl``: replace the PLS direction matrix by PCA components, then
  calibrate the induced ``E[W]`` row norms to one, matching the existing PLS path.
- ``pca_dmgp_kl``: DeepMahalanobisGP-style PCA initialisation
  ``components * 10 / ptp(X @ components.T)^2`` and calibration to those scaled
  target row norms.
- ``pca_dmgp_raw_kl``: same scaled PCA matrix, but no post-interpolation
  calibration; this is closest to the external VMGP assignment.
- ``sir_kl``: SIR directions with the same KL/row-norm/auxiliary settings.

Agents: update this docstring when CLI defaults or variant names change.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.variational_tricks import (  # noqa: E402
    KLWarmupSchedule,
    calibrate_mu_V_to_target_mu_W,
    train_vi_with_tricks,
)
from experiments.synthetic.compare_lmjgp_tricks_synth import (  # noqa: E402
    _build_regions,
    _evaluate_predict_vi,
    _make_initial_params,
    _post_train_metric_block,
    _seed_all,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
)


VARIANT_NAMES = (
    "pls_kl",
    "pca_unit_kl",
    "pca_dmgp_kl",
    "pca_dmgp_raw_kl",
    "sir_kl",
)


@dataclass(frozen=True)
class VariantSpec:
    init_mode: str
    calibration: str
    kl_warmup: bool = True
    row_norm_weight: float = 0.1
    row_norm_mode: str = "second_moment"
    aux_metric_weight: float = 0.1
    aux_metric_mode: str = "A"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _variant_spec(name: str) -> VariantSpec:
    if name == "pls_kl":
        return VariantSpec(init_mode="pls", calibration="existing")
    if name == "pca_unit_kl":
        return VariantSpec(init_mode="pca", calibration="unit")
    if name == "pca_dmgp_kl":
        return VariantSpec(init_mode="pca_dmgp", calibration="target")
    if name == "pca_dmgp_raw_kl":
        return VariantSpec(init_mode="pca_dmgp", calibration="none")
    if name == "sir_kl":
        return VariantSpec(init_mode="sir", calibration="existing")
    raise ValueError(f"Unknown variant {name!r}; choose from {VARIANT_NAMES}.")


def _pca_matrix(X: np.ndarray, Q: int, *, dmgp_scale: bool) -> tuple[np.ndarray, dict[str, Any]]:
    Xn = np.asarray(X, dtype=np.float64)
    pca = PCA(n_components=int(Q), random_state=0)
    z = pca.fit_transform(Xn)
    W = np.asarray(pca.components_, dtype=np.float64)
    info: dict[str, Any] = {
        "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "pca_components_frob_unscaled": float(np.linalg.norm(W)),
    }
    if dmgp_scale:
        ptp = np.ptp(z, axis=0)
        scale = 10.0 / np.maximum(ptp, 1e-6) ** 2
        W = W * scale[:, None]
        info.update({
            "dmgp_ptp_min": float(np.min(ptp)),
            "dmgp_ptp_max": float(np.max(ptp)),
            "dmgp_scale_min": float(np.min(scale)),
            "dmgp_scale_max": float(np.max(scale)),
            "dmgp_scale_mean": float(np.mean(scale)),
        })
    info.update({
        "W_init_frob": float(np.linalg.norm(W)),
        "W_init_trace_WtW": float(np.sum(W * W)),
        "W_init_row_norm_min": float(np.min(np.linalg.norm(W, axis=1))),
        "W_init_row_norm_max": float(np.max(np.linalg.norm(W, axis=1))),
    })
    return W.astype(np.float32), info


def _install_pca_init(
    V_params: dict[str, torch.Tensor],
    hyperparams: dict[str, torch.Tensor],
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    Q: int,
    mode: str,
    calibration: str,
    perturb_std: float,
    seed: int,
) -> dict[str, Any]:
    W_np, info = _pca_matrix(
        X_train.detach().cpu().numpy(),
        Q,
        dmgp_scale=(mode == "pca_dmgp"),
    )
    W = torch.from_numpy(W_np).to(device=X_train.device, dtype=X_train.dtype)
    m2 = int(V_params["mu_V"].shape[0])
    mu = W.unsqueeze(0).expand(m2, Q, W.shape[1]).contiguous().clone()
    if perturb_std > 0:
        g = torch.Generator(device=X_train.device)
        g.manual_seed(int(seed))
        mu = mu + perturb_std * torch.randn(mu.shape, generator=g, device=X_train.device, dtype=mu.dtype)
    with torch.no_grad():
        V_params["mu_V"].copy_(mu)
    info["mode"] = mode
    info["calibration_mode"] = calibration
    if calibration != "none":
        target_row_norm = 1.0 if calibration == "unit" else None
        info["calibration"] = calibrate_mu_V_to_target_mu_W(
            V_params,
            hyperparams,
            X_test,
            W,
            per_row=True,
            max_alpha=5.0,
            target_row_norm=target_row_norm,
        )
    return info


def _run_one_variant(
    variant: str,
    *,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    Q: int,
    m1: int,
    m2: int,
    n: int,
    num_steps: int,
    lr: float,
    log_interval: int,
    MC_num: int,
    warmup_frac: float,
    aux_max_pairs: int,
    snapshot_steps: list[int],
    init_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    spec = _variant_spec(variant)
    T, D = X_test.shape
    _seed_all(init_seed)
    regions, _x_nb_list, _y_nb_list = _build_regions(X_test, X_train, Y_train, device, m1, Q, n)

    init_mode_for_helper = spec.init_mode if spec.init_mode in ("pls", "sir") else "random"
    V_params, u_params, hyperparams, init_info = _make_initial_params(
        X_train,
        Y_train,
        X_test,
        Q=Q,
        m1=m1,
        m2=m2,
        T=T,
        D=D,
        device=device,
        mu_V_mode=init_mode_for_helper,
        init_seed=init_seed,
    )
    if spec.init_mode.startswith("pca"):
        init_info = _install_pca_init(
            V_params,
            hyperparams,
            X_train,
            X_test,
            Q=Q,
            mode=spec.init_mode,
            calibration=spec.calibration,
            perturb_std=1e-3,
            seed=init_seed,
        )
    init_info["variant"] = variant

    warm = None
    if spec.kl_warmup:
        warm = KLWarmupSchedule(
            beta_init=0.01,
            beta_final=1.0,
            warmup_steps=max(1, int(num_steps * warmup_frac)),
            shape="linear",
        )

    t0 = time.perf_counter()
    V_params, u_params, hyperparams, history, snapshots = train_vi_with_tricks(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=lr,
        num_steps=num_steps,
        log_interval=log_interval,
        snapshot_steps=snapshot_steps,
        kl_warmup=warm,
        row_norm_weight=spec.row_norm_weight,
        row_norm_mode=spec.row_norm_mode,
        aux_metric_weight=spec.aux_metric_weight,
        aux_metric_mode=spec.aux_metric_mode,
        aux_max_pairs=aux_max_pairs,
        aux_seed=init_seed,
    )
    train_time = time.perf_counter() - t0
    _mu_W, _cov_W, _A, metric_block = _post_train_metric_block(V_params, hyperparams, X_test)
    pack, _mu_pred, _sigmas = _evaluate_predict_vi(
        regions,
        V_params,
        hyperparams,
        Y_test,
        MC_num=MC_num,
    )
    return {
        "variant": variant,
        "init_info": init_info,
        "train_time_sec": float(train_time),
        "metric_A_summary": metric_block,
        "history_tail": history[-5:] if history else [],
        "snapshots": snapshots,
        "pack": list(pack),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, str) and x.strip() == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _metric_keys(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "setting",
        "family",
        "expansion",
        "caseno",
        "seed",
        "method",
        "N_train",
        "N_test",
        "latent_dim",
        "observed_dim",
        "Q_true",
        "Q",
        "n",
        "standardize_x",
        "noise_std",
        "noise_var",
        "lengthscale",
        "kernel_var",
        *RESULT_ROW_KEYS,
        "train_time_sec",
        "init_mode",
        "init_calibration",
        "pca_explained_variance_sum",
        "W_init_frob",
        "W_init_trace_WtW",
        "median_trace_A",
        "median_eff_rank_A",
        "median_top_eig_A",
        "median_row_norm_sq_mu_W",
        "mean_row_norm_sq_mu_W",
        "snapshot0_median_frob_mu_W",
        "snapshot_final_median_frob_mu_W",
        "snapshot_final_median_E_row_norm_sq",
        "status",
        "error",
    ]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    return [k for k in preferred if k in all_keys] + sorted(all_keys - set(preferred))


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        *RESULT_ROW_KEYS,
        "train_time_sec",
        "pca_explained_variance_sum",
        "W_init_frob",
        "W_init_trace_WtW",
        "median_trace_A",
        "median_eff_rank_A",
        "median_top_eig_A",
        "median_row_norm_sq_mu_W",
        "mean_row_norm_sq_mu_W",
        "snapshot0_median_frob_mu_W",
        "snapshot_final_median_frob_mu_W",
        "snapshot_final_median_E_row_norm_sq",
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["setting"]), str(row["method"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (setting, method), group in sorted(groups.items()):
        agg: dict[str, Any] = {"setting": setting, "method": method, "n_seeds": len(group)}
        for meta in (
            "family",
            "expansion",
            "caseno",
            "N_train",
            "N_test",
            "latent_dim",
            "observed_dim",
            "Q_true",
            "Q",
            "n",
            "standardize_x",
            "noise_std",
            "noise_var",
            "lengthscale",
            "kernel_var",
            "init_mode",
            "init_calibration",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in metric_names:
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            if np.isfinite(vals).any():
                agg[f"{key}_mean"] = float(np.nanmean(vals))
                agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# LMJGP q(R)/q(V) Initialisation Ablation

Purpose: compare PCA-style initialisation of the LMJGP projection inducing
mean against the existing PLS-KL setup on the paper-style L2/LH train=1000
synthetic protocol.

This folder is additive and does not overwrite the existing paper baseline
results.

Settings:

```json
{json.dumps(summary, indent=2)}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-variant metrics and diagnostics.
- `aggregate.csv`: means/stds over completed seeds.
- `summary.json`: exact runner arguments and setting metadata.
- `metrics_partial.csv`: resumable partial output while the runner is active.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LMJGP q(R)/q(V) init ablation on paper L2/LH data.")
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--settings", type=str, default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--variants", type=str, default="pca_unit_kl,pca_dmgp_kl")
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.5)
    p.add_argument("--aux_max_pairs", type=int, default=64)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument(
        "--snapshot_steps",
        type=str,
        default="0,50",
        help="Comma-separated q(W) diagnostic steps; final step is always added.",
    )
    p.add_argument("--out_dir", type=str, default="experiments/synthetic/lmjgp_r_init_pca_train1000_5seeds_20260514")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    variants = [s.strip() for s in args.variants.split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown_settings:
        raise ValueError(f"Unknown settings {unknown_settings}; choose from {sorted(SETTING_PRESETS)}.")
    unknown_variants = sorted(set(variants) - set(VARIANT_NAMES))
    if unknown_variants:
        raise ValueError(f"Unknown variants {unknown_variants}; choose from {VARIANT_NAMES}.")

    snapshot_steps = sorted({
        int(s.strip())
        for s in args.snapshot_steps.split(",")
        if s.strip() != ""
    } | {0, int(args.lmjgp_steps)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = _read_csv(out_dir / "metrics_partial.csv") if args.resume else []
    if args.resume and not metric_rows:
        metric_rows = _read_csv(out_dir / "metrics.csv")
    completed = {
        (str(r.get("setting")), int(r.get("seed")), str(r.get("method")))
        for r in metric_rows
        if str(r.get("status")) == "ok" and str(r.get("seed", "")).strip() != ""
    }

    for setting in settings:
        cfg = dict(SETTING_PRESETS[setting])
        for seed in range(int(args.num_exp)):
            _set_seed(seed)
            data = _generate_paper_data(cfg, seed, device)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(data["X_train"])
            X_test_s = scaler.transform(data["X_test"])
            y_train = np.asarray(data["y_train"], dtype=np.float32)
            y_test = np.asarray(data["y_test"], dtype=np.float32)

            X_train_t = torch.from_numpy(np.asarray(X_train_s, dtype=np.float32)).to(device)
            Y_train_t = torch.from_numpy(y_train).to(device)
            X_test_t = torch.from_numpy(np.asarray(X_test_s, dtype=np.float32)).to(device)
            Y_test_t = torch.from_numpy(y_test).to(device)

            for variant in variants:
                if (setting, seed, variant) in completed:
                    print(f"skip completed setting={setting} seed={seed} variant={variant}", flush=True)
                    continue
                print(f"setting={setting} seed={seed} variant={variant}", flush=True)
                spec = _variant_spec(variant)
                row: dict[str, Any] = {
                    "setting": setting,
                    "family": cfg["family"],
                    "expansion": cfg["expansion"],
                    "caseno": cfg["caseno"],
                    "seed": seed,
                    "method": variant,
                    "N_train": int(cfg["N_train"]),
                    "N_test": int(cfg["N_test"]),
                    "latent_dim": int(cfg["d"]),
                    "observed_dim": int(cfg["H"]),
                    "Q_true": int(cfg["Q_true"]),
                    "Q": int(cfg["Q"]),
                    "n": int(cfg["n"]),
                    "standardize_x": True,
                    "noise_std": float(cfg["noise_std"]),
                    "noise_var": float(cfg["noise_var"]),
                    "lengthscale": float(cfg["lengthscale"]),
                    "kernel_var": float(cfg["kernel_var"]),
                    "init_mode": spec.init_mode,
                    "init_calibration": spec.calibration,
                }
                try:
                    info = _run_one_variant(
                        variant,
                        X_train=X_train_t,
                        Y_train=Y_train_t,
                        X_test=X_test_t,
                        Y_test=Y_test_t,
                        Q=int(cfg["Q"]),
                        m1=int(args.m1),
                        m2=int(args.m2),
                        n=int(cfg["n"]),
                        num_steps=int(args.lmjgp_steps),
                        lr=float(args.lr),
                        log_interval=int(args.log_interval),
                        MC_num=int(args.MC_num),
                        warmup_frac=float(args.warmup_frac),
                        aux_max_pairs=int(args.aux_max_pairs),
                        snapshot_steps=snapshot_steps,
                        init_seed=int(seed) * 1019 + 11,
                        device=device,
                    )
                    pack = info["pack"]
                    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
                    row["train_time_sec"] = float(info["train_time_sec"])
                    init_info = info["init_info"]
                    row["pca_explained_variance_sum"] = init_info.get("pca_explained_variance_sum", "")
                    row["W_init_frob"] = init_info.get("W_init_frob", "")
                    row["W_init_trace_WtW"] = init_info.get("W_init_trace_WtW", "")
                    row["dmgp_scale_mean"] = init_info.get("dmgp_scale_mean", "")
                    row["calibration_json"] = json.dumps(init_info.get("calibration", {}), sort_keys=True)
                    metric_block = info["metric_A_summary"]
                    for key, value in metric_block.items():
                        row[key] = value
                    snapshots = info.get("snapshots", [])
                    if snapshots:
                        row["snapshot0_median_frob_mu_W"] = snapshots[0].get("median_frob_mu_W", "")
                        row["snapshot_final_median_frob_mu_W"] = snapshots[-1].get("median_frob_mu_W", "")
                        row["snapshot_final_median_E_row_norm_sq"] = snapshots[-1].get("median_E_row_norm_sq", "")
                    row["history_tail_json"] = json.dumps(info.get("history_tail", []), sort_keys=True)
                    row["status"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"ERROR setting={setting} seed={seed} variant={variant}: {row['error']}", flush=True)
                metric_rows.append(row)
                _write_csv(out_dir / "metrics_partial.csv", metric_rows, _metric_keys(metric_rows))

    aggregate_rows = _aggregate(metric_rows)
    _write_csv(out_dir / "metrics.csv", metric_rows, _metric_keys(metric_rows))
    _write_csv(out_dir / "aggregate.csv", aggregate_rows, _metric_keys(aggregate_rows))
    summary = {
        "args": vars(args),
        "device": str(device),
        "settings": {name: SETTING_PRESETS[name] for name in settings},
        "variants": variants,
        "n_rows": len(metric_rows),
        "n_ok": int(sum(1 for r in metric_rows if r.get("status") == "ok")),
        "n_errors": int(sum(1 for r in metric_rows if r.get("status") == "error")),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_readme(out_dir, summary)
    print(f"Saved results to {out_dir}", flush=True)
    return {"rows": metric_rows, "aggregate": aggregate_rows, "summary": summary}


if __name__ == "__main__":
    main()

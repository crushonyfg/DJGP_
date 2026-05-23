"""Diagnose LMJGP ELBO q(V) gradients on paper-style L2/LH synthetic data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/diagnose_lmjgp_elbo_gradients_paper.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants baseline,pls_kl ^
        --num_steps 120 ^
        --out_dir experiments/synthetic/l2_lh_elbo_gradient_diagnostics

Useful faster diagnostic with fewer test anchors:

    python experiments/synthetic/diagnose_lmjgp_elbo_gradients_paper.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants baseline,pls_kl ^
        --max_anchors 100 ^
        --num_steps 80 ^
        --diag_steps 0,1,5,10,25,50,80 ^
        --out_dir experiments/synthetic/l2_lh_elbo_gradient_diagnostics_smoke
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
from djgp.diagnostics.elbo_gradients import compute_lmjgp_qv_gradient_diagnostics  # noqa: E402
from djgp.variational_tricks import KLWarmupSchedule, compute_objective_with_tricks  # noqa: E402
from experiments.synthetic.compare_lmjgp_tricks_synth import (  # noqa: E402
    _build_regions,
    _make_initial_params,
    _seed_all,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


VARIANTS = ("baseline", "kl_warmup", "pls_kl")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting",
        "family",
        "seed",
        "variant",
        "step",
        "beta_W",
        "N_train",
        "N_test_total",
        "n_anchors_used",
        "latent_dim",
        "observed_dim",
        "Q_true",
        "Q",
        "n_neighbors",
        "m1",
        "m2",
        "standardize_x",
        "standardize_y",
        "data_per_anchor",
        "KL_V_per_anchor",
        "KL_V_scaled_per_anchor",
        "KL_u_per_anchor",
        "elbo_per_anchor",
        "mu_data_grad_norm",
        "mu_KL_scaled_grad_norm",
        "mu_data_over_scaled_KL",
        "sigma_data_grad_norm",
        "sigma_KL_scaled_grad_norm",
        "sigma_data_over_scaled_KL",
        "sigma2proxy_data_grad_norm",
        "sigma2proxy_KL_scaled_grad_norm",
        "sigma2proxy_data_over_scaled_KL",
        "rho_mean",
        "rho_entropy_mean",
        "rho_hard_frac_005_095",
        "mu_boundary_share_of_split_norms",
        "sigma2proxy_boundary_share_of_split_norms",
        "kl_mu_pull_dot_scaled",
        "elapsed_sec",
    ]
    keys: list[str] = []
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    for key in preferred:
        if key in all_keys:
            keys.append(key)
    keys.extend(sorted(all_keys - set(keys)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _float_or_nan(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _latest_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_run: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["setting"]), int(row["seed"]), str(row["variant"]))
        if key not in best_by_run or int(row["step"]) > int(best_by_run[key]["step"]):
            best_by_run[key] = row

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in best_by_run.values():
        groups.setdefault((str(row["setting"]), str(row["variant"])), []).append(row)

    metrics = [
        "mu_data_over_scaled_KL",
        "sigma_data_over_scaled_KL",
        "sigma2proxy_data_over_scaled_KL",
        "rho_mean",
        "rho_entropy_mean",
        "rho_hard_frac_005_095",
        "mu_boundary_share_of_split_norms",
        "sigma2proxy_boundary_share_of_split_norms",
        "mu_W_frob_median",
        "row_second_moment_median",
        "kl_mu_pull_dot_scaled",
    ]
    out: list[dict[str, Any]] = []
    for (setting, variant), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "setting": setting,
            "variant": variant,
            "n_runs": len(group),
            "latest_step_min": min(int(g["step"]) for g in group),
            "latest_step_max": max(int(g["step"]) for g in group),
        }
        for meta in (
            "family",
            "N_train",
            "N_test_total",
            "n_anchors_used",
            "latent_dim",
            "observed_dim",
            "Q_true",
            "Q",
            "n_neighbors",
            "m1",
            "m2",
            "standardize_x",
            "standardize_y",
        ):
            row[meta] = group[0].get(meta, "")
        for metric in metrics:
            vals = np.asarray([_float_or_nan(g.get(metric)) for g in group], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals))
        out.append(row)
    return out


def _variant_schedule(variant: str, num_steps: int, warmup_frac: float) -> tuple[str, KLWarmupSchedule | None]:
    if variant == "baseline":
        return "random", None
    if variant == "kl_warmup":
        return "random", KLWarmupSchedule(
            beta_init=0.01,
            beta_final=1.0,
            warmup_steps=max(1, int(num_steps * warmup_frac)),
            shape="linear",
        )
    if variant == "pls_kl":
        return "pls", KLWarmupSchedule(
            beta_init=0.01,
            beta_final=1.0,
            warmup_steps=max(1, int(num_steps * warmup_frac)),
            shape="linear",
        )
    raise ValueError(f"Unknown variant {variant!r}")


def _trainable_params(
    V_params: dict[str, torch.Tensor],
    u_params: list[dict[str, torch.Tensor]],
    hyperparams: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    params = [V_params["mu_V"], V_params["sigma_V"]]
    for u in u_params:
        params += [
            u["U_logit"],
            u["mu_u"],
            u["Sigma_u"],
            u["sigma_noise"],
            u["omega"],
            u["sigma_k"],
        ]
    params += [hyperparams["lengthscales"], hyperparams["var_w"], hyperparams["Z"]]
    return params


def _clamp_after_step(
    V_params: dict[str, torch.Tensor],
    u_params: list[dict[str, torch.Tensor]],
    hyperparams: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        V_params["sigma_V"].clamp_(min=1e-1)
        hyperparams["var_w"].clamp_(min=1e-1)
        hyperparams["lengthscales"].clamp_(min=1e-6)
        for u in u_params:
            u["sigma_noise"].clamp_(min=1e-1)
            u["sigma_k"].clamp_(min=1e-1)


def _run_variant(
    *,
    args: argparse.Namespace,
    setting: str,
    cfg: dict[str, Any],
    seed: int,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    variant: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    Q = int(cfg["Q"])
    n_neighbors = int(cfg["n"])
    m1 = int(args.m1)
    m2 = int(args.m2)
    T, D = X_test_t.shape
    mu_V_mode, warmup = _variant_schedule(variant, int(args.num_steps), float(args.warmup_frac))

    stable_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(f"{setting}:{variant}")) % 1000
    init_seed = int(seed) * 1009 + stable_offset
    _seed_all(init_seed)
    regions, _X_nb_list, _y_nb_list = _build_regions(
        X_test_t,
        X_train_t,
        y_train_t,
        device,
        m1,
        Q,
        n_neighbors,
    )
    V_params, u_params, hyperparams, init_info = _make_initial_params(
        X_train_t,
        y_train_t,
        X_test_t,
        Q=Q,
        m1=m1,
        m2=m2,
        T=T,
        D=D,
        device=device,
        mu_V_mode=mu_V_mode,
        init_seed=init_seed,
    )

    optimizer = torch.optim.Adam(_trainable_params(V_params, u_params, hyperparams), lr=float(args.lr))
    diag_steps = sorted({s for s in args.diag_steps_list if 0 <= s <= int(args.num_steps)})
    diag_set = set(diag_steps)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    def beta_at(step: int) -> float:
        return float(warmup(step)) if warmup is not None else 1.0

    base_meta = {
        "setting": setting,
        "family": cfg["family"],
        "seed": int(seed),
        "variant": variant,
        "N_train": int(cfg["N_train"]),
        "N_test_total": int(cfg["N_test"]),
        "n_anchors_used": int(T),
        "latent_dim": int(cfg["d"]),
        "observed_dim": int(cfg["H"]),
        "Q_true": int(cfg["Q_true"]),
        "Q": int(Q),
        "n_neighbors": int(n_neighbors),
        "m1": int(m1),
        "m2": int(m2),
        "standardize_x": True,
        "standardize_y": False,
        "noise_std": float(cfg["noise_std"]),
        "noise_var": float(cfg["noise_var"]),
        "expansion": str(cfg.get("expansion", "rff")),
        "mu_V_mode": mu_V_mode,
        "warmup_steps": 0 if warmup is None else int(warmup.warmup_steps),
        "init_info_json": json.dumps(sanitize_for_json(init_info), sort_keys=True),
    }

    if 0 in diag_set:
        diag = compute_lmjgp_qv_gradient_diagnostics(
            regions,
            V_params,
            u_params,
            hyperparams,
            step=0,
            beta_W=beta_at(0),
            boundary_quantile=float(args.boundary_quantile),
        )
        rows.append({**base_meta, **diag, "elapsed_sec": time.perf_counter() - start})
        print(
            f"[{setting} seed={seed} {variant} step=0] "
            f"mu data/scaled-KL={diag['mu_data_over_scaled_KL']:.3g} "
            f"sigma2 data/scaled-KL={diag['sigma2proxy_data_over_scaled_KL']:.3g} "
            f"rho entropy={diag['rho_entropy_mean']:.3g}",
            flush=True,
        )

    for step in range(1, int(args.num_steps) + 1):
        beta_W = beta_at(step)
        optimizer.zero_grad(set_to_none=True)
        loss, _breakdown = compute_objective_with_tricks(
            regions,
            V_params,
            u_params,
            hyperparams,
            beta_W=beta_W,
            row_norm_weight=0.0,
            aux_metric_weight=0.0,
        )
        loss.backward()
        optimizer.step()
        _clamp_after_step(V_params, u_params, hyperparams)

        if step in diag_set:
            diag = compute_lmjgp_qv_gradient_diagnostics(
                regions,
                V_params,
                u_params,
                hyperparams,
                step=step,
                beta_W=beta_W,
                boundary_quantile=float(args.boundary_quantile),
            )
            rows.append({**base_meta, **diag, "elapsed_sec": time.perf_counter() - start})
            print(
                f"[{setting} seed={seed} {variant} step={step}] "
                f"mu data/scaled-KL={diag['mu_data_over_scaled_KL']:.3g} "
                f"sigma2 data/scaled-KL={diag['sigma2proxy_data_over_scaled_KL']:.3g} "
                f"rho entropy={diag['rho_entropy_mean']:.3g}",
                flush=True,
            )

    return rows


def _run_setting_seed(
    args: argparse.Namespace,
    setting: str,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    cfg = SETTING_PRESETS[setting]
    _set_seed(seed)
    data = _generate_paper_data(cfg, seed, device)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data["X_train"]).astype(np.float32)
    X_test_s = scaler.transform(data["X_test"]).astype(np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.float32)

    if args.max_anchors is not None:
        X_test_s = X_test_s[: int(args.max_anchors)]

    X_train_t = torch.from_numpy(X_train_s).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test_s).to(device)

    rows: list[dict[str, Any]] = []
    for variant in args.variants_list:
        print(
            f"running setting={setting} seed={seed} variant={variant} "
            f"anchors={X_test_t.shape[0]} steps={args.num_steps}",
            flush=True,
        )
        rows.extend(
            _run_variant(
                args=args,
                setting=setting,
                cfg=cfg,
                seed=seed,
                X_train_t=X_train_t,
                y_train_t=y_train_t,
                X_test_t=X_test_t,
                variant=variant,
                device=device,
            )
        )
    return rows


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# L2/LH LMJGP ELBO Gradient Diagnostics

This folder records training-time q(V) gradient diagnostics for the original
LMJGP ELBO on paper-style synthetic L2/LH data.

Files:

- `gradient_history.csv`: diagnostics at requested optimization steps.
- `gradient_latest.csv`: one latest-step row per setting and variant.
- `summary.json`: exact command arguments and setting metadata.

Boundary anchors are a proxy: the top local-y-variance anchors according to
`boundary_quantile`.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LMJGP ELBO q(V) gradient diagnostics on paper L2/LH data.")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--variants", default="baseline,pls_kl")
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--max_anchors", type=int, default=None)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--num_steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--warmup_frac", type=float, default=0.5)
    p.add_argument("--diag_steps", default="0,1,5,10,25,50,100,120")
    p.add_argument("--boundary_quantile", type=float, default=0.75)
    p.add_argument("--out_dir", default="experiments/synthetic/l2_lh_elbo_gradient_diagnostics")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    args.settings_list = [s.strip() for s in args.settings.split(",") if s.strip()]
    args.variants_list = [s.strip() for s in args.variants.split(",") if s.strip()]
    args.diag_steps_list = sorted({int(s.strip()) for s in args.diag_steps.split(",") if s.strip()})

    unknown_settings = sorted(set(args.settings_list) - set(SETTING_PRESETS))
    unknown_variants = sorted(set(args.variants_list) - set(VARIANTS))
    if unknown_settings:
        raise ValueError(f"Unknown settings: {unknown_settings}")
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")
    if not (0.0 < float(args.boundary_quantile) < 1.0):
        raise ValueError("--boundary_quantile must be between 0 and 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, Any]] = []
    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
        for setting in args.settings_list:
            rows.extend(_run_setting_seed(args, setting, seed, device))
            _write_csv(out_dir / "gradient_history.csv", rows)
            _write_csv(out_dir / "gradient_latest.csv", _latest_summary(rows))

    summary = {
        "args": sanitize_for_json(vars(args)),
        "device": str(device),
        "settings": {name: SETTING_PRESETS[name] for name in args.settings_list},
        "outputs": {
            "gradient_history": "gradient_history.csv",
            "gradient_latest": "gradient_latest.csv",
            "readme": "README.md",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(sanitize_for_json(summary), indent=2), encoding="utf-8")
    _write_readme(out_dir, sanitize_for_json(summary))
    print(f"Wrote diagnostics to {out_dir}", flush=True)
    return summary


if __name__ == "__main__":
    main()

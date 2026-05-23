"""Compare LMJGP-trained q(W) with an uncertain-W self-CEM prediction head.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_lmjgp_uncertain_w_cem_head_lh.py ^
        --settings lh_q5_train1000 --seed 0 --max_test 200 ^
        --lmjgp_steps 300 --MC_num 5 ^
        --out_dir experiments/synthetic/lmjgp_uncertain_w_cem_head_lh_seed0_20260517

This is a narrow head swap: train the transductive LMJGP q(W) state as in the
paper synthetic baseline, then replace sampled-W JumpGP-CEM prediction with a
moment-matched uncertain-W self-CEM head on the same raw-X neighborhoods.
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

from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.variational import predict_vi, qW_from_qV  # noqa: E402
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import (  # noqa: E402
    _metrics_pack,
    _train_lmjgp_state,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import (  # noqa: E402
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
    SelfCEMConfig,
    _default_hypers_from_labels,
    _label_diag,
    _neighbor_stack,
    _response_seed_labels,
    run_uncertain_w_self_cem,
    uncertain_w_cem_predict_from_labels,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting",
        "family",
        "seed",
        "variant",
        "status",
        "N_train",
        "N_test",
        "n_test_eval",
        "latent_dim",
        "observed_dim",
        "Q",
        "n_neighbors",
        "lmjgp_steps",
        "MC_num",
        *RESULT_ROW_KEYS,
        "lmjgp_train_sec",
        "pred_sec",
        "cem_sec",
        "mu_w_norm_mean",
        "sigma_w_mean",
        "init_inlier_rate",
        "final_inlier_rate",
        "label_change_rate",
        "mean_n_inliers",
        "mean_lengthscale",
        "mean_signal_var",
        "mean_noise_var",
        "fallback_count",
        "error",
    ]
    keys = [k for k in preferred if any(k in r for r in rows)]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _float_or_nan(x: Any) -> float:
    try:
        if x == "":
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["setting"]), str(row["variant"])), []).append(row)
    value_keys = list(RESULT_ROW_KEYS) + [
        "lmjgp_train_sec",
        "pred_sec",
        "cem_sec",
        "mu_w_norm_mean",
        "sigma_w_mean",
        "init_inlier_rate",
        "final_inlier_rate",
        "label_change_rate",
        "mean_n_inliers",
        "mean_lengthscale",
        "mean_signal_var",
        "mean_noise_var",
        "fallback_count",
    ]
    for (setting, variant), group in sorted(groups.items()):
        row: dict[str, Any] = {"setting": setting, "variant": variant, "n_runs": len(group)}
        for meta in (
            "family",
            "N_train",
            "N_test",
            "n_test_eval",
            "latent_dim",
            "observed_dim",
            "Q",
            "n_neighbors",
            "lmjgp_steps",
            "MC_num",
        ):
            row[meta] = group[0].get(meta, "")
        for key in value_keys:
            vals = np.asarray([_float_or_nan(r.get(key)) for r in group], dtype=np.float64)
            if np.isfinite(vals).any():
                row[f"{key}_mean"] = float(np.nanmean(vals))
                row[f"{key}_std"] = float(np.nanstd(vals))
        out.append(row)
    return out


def _row_normalized_moments(mu_W: torch.Tensor, cov_W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.norm(mu_W, dim=-1, keepdim=True).clamp_min(1e-6)
    mu_n = mu_W / norm
    cov_n = cov_W / norm.pow(2).unsqueeze(-1)
    return mu_n, cov_n


def _base_row(args: argparse.Namespace, cfg: dict[str, Any], setting: str, variant: str) -> dict[str, Any]:
    n_eval = int(args.max_test if args.max_test is not None else cfg["N_test"])
    return {
        "setting": setting,
        "family": cfg["family"],
        "seed": int(args.seed),
        "variant": variant,
        "status": "ok",
        "N_train": int(cfg["N_train"]),
        "N_test": int(cfg["N_test"]),
        "n_test_eval": n_eval,
        "latent_dim": int(cfg["d"]),
        "observed_dim": int(cfg["H"]),
        "Q": int(cfg["Q"]),
        "n_neighbors": int(args.n_neighbors or cfg["n"]),
        "lmjgp_steps": int(args.lmjgp_steps),
        "MC_num": int(args.MC_num),
        "standardize_x": True,
        "standardize_y_for_cem_head": True,
        "expansion": str(cfg.get("expansion", "rff")),
    }


def _run_uncertain_cem_head(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    setting: str,
    variant: str,
    X_test_t: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors_std: torch.Tensor,
    y_test: np.ndarray,
    y_mean: float,
    y_std: float,
    mu_W: torch.Tensor,
    cov_W: torch.Tensor,
    lmjgp_train_sec: float,
) -> dict[str, Any]:
    W0_init = mu_W.detach().mean(dim=0)
    W0_init = W0_init / torch.linalg.norm(W0_init, dim=-1, keepdim=True).clamp_min(1e-6)
    init_labels = _response_seed_labels(
        X_test_t,
        X_neighbors,
        y_neighbors_std,
        seed_frac=float(args.response_seed_frac),
        inlier_frac=float(args.response_inlier_frac),
        min_inliers=int(args.min_inliers),
    )
    init_hp = _default_hypers_from_labels(
        X_test_t,
        X_neighbors,
        y_neighbors_std,
        init_labels,
        W0_init,
        min_inliers=int(args.min_inliers),
    )
    gate0 = torch.zeros(
        X_test_t.shape[0],
        int(cfg["Q"]) + 1,
        device=X_test_t.device,
        dtype=X_test_t.dtype,
    )
    t_cem = time.perf_counter()
    state = run_uncertain_w_self_cem(
        X_test_t,
        X_neighbors,
        y_neighbors_std,
        init_labels,
        mode="anchor",
        W_mu_anchor=mu_W,
        W_cov_anchor=cov_W,
        init_lengthscale=init_hp["lengthscale"],
        init_signal_var=init_hp["signal_var"],
        init_noise_var=init_hp["noise_var"].clamp_min(float(args.min_noise_var)),
        gate_init=gate0,
        config=SelfCEMConfig(
            cem_updates=int(args.cem_updates),
            hyperopt=FixedLabelHyperoptConfig(
                steps=int(args.hyper_steps),
                lr=float(args.hyper_lr),
                min_noise_var=float(args.min_noise_var),
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
    )
    cem_sec = float(time.perf_counter() - t_cem)

    t_pred = time.perf_counter()
    out = uncertain_w_cem_predict_from_labels(
        X_test_t,
        X_neighbors,
        y_neighbors_std,
        state["labels"],
        mode="anchor",
        W_mu_anchor=mu_W,
        W_cov_anchor=cov_W,
        lengthscale=state["lengthscale"],
        signal_var=state["signal_var"],
        noise_var=state["noise_var"],
        mean_const=state["mean_const"],
        correction_weight=float(args.kcov),
        min_inliers=int(args.min_inliers),
    )
    pred_sec = float(time.perf_counter() - t_pred)
    mu = out.mu.detach().cpu().numpy().reshape(-1) * float(y_std) + float(y_mean)
    sigma = out.sigma.detach().cpu().numpy().reshape(-1) * abs(float(y_std))
    pack = _metrics_pack(y_test, mu, sigma, cem_sec + pred_sec)
    row = _base_row(args, cfg, setting, variant)
    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
    row.update(_label_diag(init_labels.bool(), state["labels"].bool(), X_neighbors.shape[1]))
    row.update(
        {
            "lmjgp_train_sec": float(lmjgp_train_sec),
            "cem_sec": cem_sec,
            "pred_sec": pred_sec,
            "mu_w_norm_mean": float(torch.linalg.norm(mu_W.detach(), dim=-1).mean().cpu().item()),
            "sigma_w_mean": float(torch.sqrt(torch.diagonal(cov_W.detach(), dim1=-2, dim2=-1).clamp_min(1e-12)).mean().cpu().item()),
            "mean_lengthscale": float(state["lengthscale"].detach().mean().cpu().item()),
            "mean_signal_var": float(state["signal_var"].detach().mean().cpu().item()),
            "mean_noise_var": float(state["noise_var"].detach().mean().cpu().item()),
            "fallback_count": int(out.diagnostics["fallback_count"]),
            "mean_kernel_vector_correction": float(out.diagnostics["mean_kernel_vector_correction"]),
        }
    )
    return row


def run_setting(args: argparse.Namespace, setting: str, device: torch.device) -> list[dict[str, Any]]:
    cfg = dict(SETTING_PRESETS[setting])
    _set_seed(int(args.seed))
    data = _generate_paper_data(cfg, int(args.seed), device)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data["X_train"]).astype(np.float32)
    X_test = scaler.transform(data["X_test"]).astype(np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.float32).reshape(-1)
    y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)
    if args.max_test is not None:
        X_test = X_test[: int(args.max_test)]
        y_test = y_test[: int(args.max_test)]
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() if y_train.std() > 1e-8 else 1.0)
    y_train_std = ((y_train - y_mean) / y_std).astype(np.float32)

    X_train_t = torch.from_numpy(X_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_train_t_raw = torch.from_numpy(y_train.astype(np.float32)).to(device)
    y_train_std_t = torch.from_numpy(y_train_std).to(device)
    n_neighbors = int(args.n_neighbors or cfg["n"])

    print(f"training LMJGP setting={setting} seed={args.seed}", flush=True)
    regions, V_params, _u_params, hyperparams, lmjgp_train_sec = _train_lmjgp_state(
        X_train_t,
        y_train_t_raw,
        X_test_t,
        Q=int(cfg["Q"]),
        n=n_neighbors,
        m1=int(args.m1),
        m2=int(args.m2),
        steps=int(args.lmjgp_steps),
        lr=float(args.lr),
        device=device,
    )

    rows: list[dict[str, Any]] = []
    t_pred = time.perf_counter()
    mu_lmjgp, var_lmjgp = predict_vi(regions, V_params, hyperparams, M=int(args.MC_num))
    pred_sec = float(time.perf_counter() - t_pred)
    sigma_lmjgp = torch.sqrt(var_lmjgp.clamp_min(1e-12))
    pack = _metrics_pack(
        y_test,
        mu_lmjgp.detach().cpu().numpy(),
        sigma_lmjgp.detach().cpu().numpy(),
        pred_sec,
    )
    row = _base_row(args, cfg, setting, "lmjgp_sampleW_jumpgp_refit")
    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
    row.update({"lmjgp_train_sec": float(lmjgp_train_sec), "pred_sec": pred_sec})
    rows.append(row)

    X_neighbors, y_neighbors_std, _idx = _neighbor_stack(
        X_test_t,
        X_train_t,
        y_train_std_t,
        n_neighbors=n_neighbors,
    )
    with torch.no_grad():
        mu_W, cov_W = qW_from_qV(
            X_test_t,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
        mu_W = mu_W.detach().to(device=X_test_t.device, dtype=X_test_t.dtype)
        cov_W = cov_W.detach().to(device=X_test_t.device, dtype=X_test_t.dtype)
        mu_W_n, cov_W_n = _row_normalized_moments(mu_W, cov_W)

    for variant, m, c in (
        ("lmjgp_qw_uncertain_selfcem_rawmom", mu_W, cov_W),
        ("lmjgp_qw_uncertain_selfcem_rownorm", mu_W_n, cov_W_n),
    ):
        print(f"predict setting={setting} variant={variant}", flush=True)
        rows.append(
            _run_uncertain_cem_head(
                args=args,
                cfg=cfg,
                setting=setting,
                variant=variant,
                X_test_t=X_test_t,
                X_neighbors=X_neighbors,
                y_neighbors_std=y_neighbors_std,
                y_test=y_test,
                y_mean=y_mean,
                y_std=y_std,
                mu_W=m,
                cov_W=c,
                lmjgp_train_sec=float(lmjgp_train_sec),
            )
        )
    return rows


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# LMJGP q(W) with Uncertain-W CEM Head

This run trains the transductive LMJGP q(W) state on paper-style synthetic
data, then evaluates the original sampled-W JumpGP-CEM head and two
moment-matched uncertain-W self-CEM heads using the same learned q(W).

Files:

- `metrics.csv`: one row per setting/variant.
- `aggregate.csv`: aggregate over successful rows.
- `summary.json`: exact runner arguments and setting metadata.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LMJGP-trained q(W) with uncertain-W self-CEM head.")
    p.add_argument("--settings", default="lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--n_neighbors", type=int, default=None)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--cem_updates", type=int, default=2)
    p.add_argument("--kcov", type=float, default=0.1)
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
    p.add_argument("--response_seed_frac", type=float, default=0.25)
    p.add_argument("--response_inlier_frac", type=float, default=0.70)
    p.add_argument("--out_dir", default="experiments/synthetic/lmjgp_uncertain_w_cem_head_lh_seed0_20260517")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    unknown = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for setting in settings:
        try:
            rows.extend(run_setting(args, setting, device))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "setting": setting,
                    "seed": int(args.seed),
                    "variant": "run_failed",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"FAILED setting={setting}: {rows[-1]['error']}", flush=True)
    _write_csv(out_dir / "metrics.csv", rows)
    agg = _aggregate(rows)
    _write_csv(out_dir / "aggregate.csv", agg)
    summary = {
        "args": vars(args),
        "settings": {s: SETTING_PRESETS[s] for s in settings},
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_readme(out_dir, summary)
    print(f"Saved results to {out_dir}", flush=True)
    return summary


if __name__ == "__main__":
    main()

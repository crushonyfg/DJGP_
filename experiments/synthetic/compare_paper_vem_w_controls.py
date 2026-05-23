"""Run soft-gamma conditional-W VEM controls on paper L2/LH synthetic data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Five-seed controls for the best current VEM-W configuration.
    python experiments/synthetic/compare_paper_vem_w_controls.py ^
        --num_exp 5 ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --controls learned_qw,anchor_shuffled,global_mean,prior_qw ^
        --n_candidate 100 --n_vem 50 --steps 80 --eval_mc 2 --resume ^
        --out_dir experiments/synthetic/vem_w_softgamma_controls_train1000_5seeds

    # Smoke test.
    python experiments/synthetic/compare_paper_vem_w_controls.py ^
        --num_exp 1 --settings l2_q2_train1000 --max_anchors 20 ^
        --controls learned_qw,prior_qw --n_candidate 40 --n_vem 25 ^
        --steps 5 --eval_mc 2 ^
        --out_dir experiments/synthetic/vem_w_controls_smoke

This runner trains the current best conditional-W VEM variant:

``post_vem1_boundary`` with soft responsibilities
(``temperature=5 -> 3``, ``gamma_damping=0.3``), posterior-W prediction,
and projected final JumpGP neighborhoods.  It then evaluates controls from the
same trained projection posterior:

- ``learned_qw``: samples from learned ``q(W_j)``.
- ``anchor_shuffled``: learned samples with the anchor dimension permuted.
- ``global_mean``: each sample collapsed to its global mean projection.
- ``prior_qw``: samples ``C_j`` from the coefficient prior around learned
  ``W0`` instead of from learned ``q(C_j)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import build_V_basis, compute_pls_W0  # noqa: E402
from djgp.projections.conditional_vem import VemWConfig, train_conditional_vem_w  # noqa: E402
from djgp.projections.inductive import run_projected_jumpgp_ensemble  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


CONTROL_ORDER = ("learned_qw", "anchor_shuffled", "global_mean", "prior_qw")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting",
        "family",
        "seed",
        "control",
        "status",
        "N_train",
        "N_test",
        "latent_dim",
        "observed_dim",
        "Q_true",
        "Q",
        "n_final",
        "n_candidate",
        "n_vem",
        "standardize_x",
        *RESULT_ROW_KEYS,
        "train_sec",
        "pred_sec",
        "best_loss",
        "gamma_mean",
        "gamma_entropy",
        "kl_coeff",
        "C_std_median",
        "anchor_dispersion_frob",
        "cosine_to_ref_median",
        "error",
    ]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    keys = [key for key in preferred if key in all_keys]
    keys.extend(sorted(all_keys - set(keys)))
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


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["setting"]), str(row["control"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (setting, control), group in sorted(groups.items()):
        agg: dict[str, Any] = {"setting": setting, "control": control, "n_seeds": len(group)}
        for meta in (
            "family",
            "N_train",
            "N_test",
            "latent_dim",
            "observed_dim",
            "Q_true",
            "Q",
            "n_final",
            "n_candidate",
            "n_vem",
            "standardize_x",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in RESULT_ROW_KEYS:
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        for key in (
            "train_sec",
            "pred_sec",
            "best_loss",
            "gamma_mean",
            "gamma_entropy",
            "kl_coeff",
            "C_std_median",
            "anchor_dispersion_frob",
            "cosine_to_ref_median",
        ):
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            if np.isfinite(vals).any():
                agg[f"{key}_mean"] = float(np.nanmean(vals))
                agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _metrics_pack(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    wall_sec: float,
    *,
    drop_exploded_points: bool,
    max_sigma_for_metrics: float,
) -> tuple[list[float], dict[str, Any]]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    sigma_before = sigma.copy()
    mask = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0.0)
    if drop_exploded_points:
        mask = mask & (sigma <= float(max_sigma_for_metrics))
    n_total = int(y.shape[0])
    n_keep = int(np.sum(mask))
    if n_keep == 0:
        raise ValueError("All predictive points were filtered before metric computation.")
    y = y[mask]
    mu = mu[mask]
    sigma = sigma[mask]
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        ).item()
    )
    diag = {
        "n_metric_points": n_keep,
        "n_dropped_metric_points": n_total - n_keep,
        "max_sigma_before_filter": float(np.nanmax(sigma_before)),
        "max_sigma_after_filter": float(np.nanmax(sigma)),
        "max_sigma_for_metrics": float(max_sigma_for_metrics),
    }
    return pack_gaussian_uq_list(rmse, crps, wall_sec, y, mu, sigma), diag


def _row_normalize(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(eps)


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


def _make_config(args: argparse.Namespace) -> VemWConfig:
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


def _control_samples(
    control: str,
    *,
    result,
    n_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(W_samples, W_select)`` for prediction and projected KNN."""
    model = result.model
    T = result.W_mean.shape[0]
    generator = torch.Generator(device=result.W_mean.device)
    generator.manual_seed(int(seed))
    if control == "learned_qw":
        W = model.sample_W(int(n_samples), mean_only=False).detach()
        return W, result.W_mean.detach()
    if control == "anchor_shuffled":
        W = model.sample_W(int(n_samples), mean_only=False).detach()
        perm = torch.randperm(T, generator=generator, device=W.device)
        return W[:, perm].contiguous(), result.W_mean[perm].detach().contiguous()
    if control == "global_mean":
        W = model.sample_W(int(n_samples), mean_only=False).detach()
        W_global = _row_normalize(W.mean(dim=1, keepdim=True)).expand(-1, T, -1, -1).contiguous()
        W_select = _row_normalize(result.W_mean.mean(dim=0, keepdim=True)).expand(T, -1, -1).contiguous()
        return W_global, W_select.detach()
    if control == "prior_qw":
        C_prior = float(model.coeff_scale) * 0.0  # keeps type checkers quiet
        del C_prior
        C = float(1.0) * torch.randn(
            int(n_samples),
            model.T,
            model.Q,
            model.Rv,
            device=result.W_mean.device,
            dtype=result.W_mean.dtype,
            generator=generator,
        )
        C = C * float(result.diagnostics.get("prior_std", 1.0) or 1.0)
        W_prior = model.W_from_C(C).detach()
        C0 = torch.zeros(1, model.T, model.Q, model.Rv, device=result.W_mean.device, dtype=result.W_mean.dtype)
        W_select = model.W_from_C(C0).squeeze(0).detach()
        return W_prior, W_select
    raise ValueError(control)


def _base_row(
    *,
    setting: str,
    cfg: dict[str, Any],
    seed: int,
    control: str,
    n_final: int,
    n_candidate: int,
    n_vem: int,
    n_eval: int,
) -> dict[str, Any]:
    return {
        "setting": setting,
        "family": cfg["family"],
        "seed": int(seed),
        "control": control,
        "N_train": int(cfg["N_train"]),
        "N_test": int(n_eval),
        "latent_dim": int(cfg["d"]),
        "observed_dim": int(cfg["H"]),
        "Q_true": int(cfg["Q_true"]),
        "Q": int(cfg["Q"]),
        "n_final": int(n_final),
        "n_candidate": int(n_candidate),
        "n_vem": int(n_vem),
        "standardize_x": True,
        "noise_std": float(cfg["noise_std"]),
        "noise_var": float(cfg["noise_var"]),
        "lengthscale": float(cfg["lengthscale"]),
        "kernel_var": float(cfg["kernel_var"]),
    }


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Soft-Gamma Conditional-W VEM Controls

Purpose: five-seed evaluation of the best current conditional-W VEM variant
and learned/posterior controls.

```json
{json.dumps(summary, indent=2)}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-control rows.
- `aggregate.csv`: means/stds over successful rows.
- `histories/*.json`: VEM-W training histories for each setting/seed.
- `summary.json`: exact runner arguments.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Soft-gamma VEM-W learned/prior controls on paper synthetic data.")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--settings", type=str, default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--controls", type=str, default="learned_qw,anchor_shuffled,global_mean,prior_qw")
    p.add_argument("--max_anchors", type=int, default=None)
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
    p.add_argument(
        "--drop_exploded_points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop non-finite or extreme-variance test points before computing metrics.",
    )
    p.add_argument(
        "--max_sigma_for_metrics",
        type=float,
        default=1e6,
        help="Maximum predictive sigma kept for metrics when --drop_exploded_points is enabled.",
    )
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out_dir", type=str, default="experiments/synthetic/vem_w_softgamma_controls_train1000_5seeds")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    controls = [s.strip() for s in args.controls.split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    unknown_controls = sorted(set(controls) - set(CONTROL_ORDER))
    if unknown_settings:
        raise ValueError(f"Unknown settings: {unknown_settings}")
    if unknown_controls:
        raise ValueError(f"Unknown controls: {unknown_controls}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = out_dir / "histories"
    hist_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(out_dir / "metrics_partial.csv") if args.resume else []
    if args.resume and not rows:
        rows = _read_csv(out_dir / "metrics.csv")
    completed = {
        (str(r.get("setting")), int(r.get("seed")), str(r.get("control")))
        for r in rows
        if str(r.get("status")) == "ok" and str(r.get("seed", "")).strip() != ""
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for setting in settings:
        cfg = dict(SETTING_PRESETS[setting])
        n_final = int(cfg["n"])
        n_candidate = min(max(int(args.n_candidate), n_final, int(args.n_vem)), int(cfg["N_train"]))
        n_vem = min(int(args.n_vem), n_candidate)
        n_final = min(n_final, n_candidate)
        for seed in range(int(args.seed_start), int(args.num_exp)):
            pending = [control for control in controls if (setting, seed, control) not in completed]
            if not pending:
                print(f"skip completed setting={setting} seed={seed}", flush=True)
                continue
            print(f"train setting={setting} seed={seed}", flush=True)
            _set_seed(seed)
            data = _generate_paper_data(cfg, seed, device)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(data["X_train"])
            X_test_s = scaler.transform(data["X_test"])
            y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
            y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)
            if args.max_anchors is not None:
                k = min(int(args.max_anchors), X_test_s.shape[0])
                X_test_s = X_test_s[:k]
                y_test = y_test[:k]
            X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
            y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
            X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)
            X_cand, y_cand = _build_neighbor_stacks(
                X_test_t,
                X_train_t,
                y_train_t,
                n_neighbors=n_candidate,
                device=device,
            )
            y_mean = float(np.mean(y_train))
            y_std = float(np.std(y_train) + 1e-8)
            X_vem = X_cand[:, :n_vem].contiguous()
            y_vem_std = ((y_cand[:, :n_vem] - y_mean) / y_std).contiguous()
            V_np, V_info = build_V_basis(
                X_train_s,
                y_train,
                n_pls=int(args.n_pls),
                n_pca=int(args.n_pca),
                n_random=int(args.n_random),
                seed=seed,
            )
            W0_np = compute_pls_W0(X_train_s, y_train, Q=int(cfg["Q"]))
            V_t = torch.from_numpy(V_np.astype(np.float32)).to(device)
            W0_t = torch.from_numpy(W0_np.astype(np.float32)).to(device)
            train_diag: dict[str, Any] = {}
            try:
                result = train_conditional_vem_w(
                    X_neighbors=X_vem,
                    y_neighbors=y_vem_std,
                    X_anchors=X_test_t,
                    V=V_t,
                    W0_init=W0_t,
                    cfg=_make_config(args),
                )
                result.diagnostics["prior_std"] = float(args.prior_std)
                train_diag = dict(result.diagnostics)
                train_diag.update(
                    {
                        "V_Rv": int(V_t.shape[1]),
                        "V_info": json.dumps(V_info),
                        "temperature": float(args.temperature),
                        "temperature_final": float(args.temperature_final),
                        "gamma_damping": float(args.gamma_damping),
                        "eval_mc": int(args.eval_mc),
                    }
                )
                hist_path = hist_dir / f"{setting}_seed{seed}_softgamma_post_vem1_boundary.json"
                hist_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")
                for control in pending:
                    row = _base_row(
                        setting=setting,
                        cfg=cfg,
                        seed=seed,
                        control=control,
                        n_final=n_final,
                        n_candidate=n_candidate,
                        n_vem=n_vem,
                        n_eval=len(y_test),
                    )
                    print(f"evaluate setting={setting} seed={seed} control={control}", flush=True)
                    W_samples, W_select = _control_samples(
                        control,
                        result=result,
                        n_samples=int(args.eval_mc),
                        seed=seed * 1009 + CONTROL_ORDER.index(control) * 97 + 13,
                    )
                    X_list, y_list = _projected_neighbor_lists(
                        W_select,
                        X_test_t,
                        X_cand,
                        y_cand,
                        n_neighbors=n_final,
                    )
                    out = run_projected_jumpgp_ensemble(
                        W_samples,
                        X_test_t,
                        X_list,
                        y_list,
                        device=device,
                        progress=False,
                        desc=f"{setting}/seed{seed}/{control}",
                    )
                    pred_sec = float(out["elapsed_sec"])
                    pack, metric_diag = _metrics_pack(
                        y_test,
                        np.asarray(out["mu"], dtype=np.float64),
                        np.asarray(out["sigma"], dtype=np.float64),
                        float(result.train_time_sec + pred_sec),
                        drop_exploded_points=bool(args.drop_exploded_points),
                        max_sigma_for_metrics=float(args.max_sigma_for_metrics),
                    )
                    row["status"] = "ok"
                    for key, val in zip(RESULT_ROW_KEYS, pack):
                        row[key] = val
                    row.update(train_diag)
                    row.update(metric_diag)
                    row["pred_sec"] = pred_sec
                    rows.append(row)
                    completed.add((setting, seed, control))
                    _write_csv(out_dir / "metrics_partial.csv", rows)
            except Exception as exc:
                for control in pending:
                    row = _base_row(
                        setting=setting,
                        cfg=cfg,
                        seed=seed,
                        control=control,
                        n_final=n_final,
                        n_candidate=n_candidate,
                        n_vem=n_vem,
                        n_eval=len(y_test),
                    )
                    row["status"] = "error"
                    row["error"] = repr(exc)
                    row.update(train_diag)
                    rows.append(row)
                _write_csv(out_dir / "metrics_partial.csv", rows)
                print(f"ERROR setting={setting} seed={seed}: {exc!r}", flush=True)
    _write_csv(out_dir / "metrics.csv", rows)
    agg = _aggregate(rows)
    _write_csv(out_dir / "aggregate.csv", agg)
    summary = {
        "args": vars(args),
        "device": str(device),
        "n_rows": len(rows),
        "n_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "controls": controls,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_readme(out_dir, summary)
    print(f"Saved {len(rows)} rows to {out_dir}", flush=True)
    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    main()

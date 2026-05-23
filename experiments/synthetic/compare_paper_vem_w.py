"""Compare conditional-W exact JGP-VEM projection learning on paper synthetic data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Focused seed-0 comparison on L2/LH train=1000.
    python experiments/synthetic/compare_paper_vem_w.py ^
        --num_exp 1 ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --methods map_vem1,map_vem5,post_vem1,post_vem1_boundary ^
        --n_candidate 100 --n_vem 50 --steps 120 --mc_train 2 --mc_gamma 2 ^
        --prediction_heads mean,posterior ^
        --final_neighbor_modes raw,projected ^
        --out_dir experiments/synthetic/vem_w_paper_train1000_seed0

    # Fast smoke test.
    python experiments/synthetic/compare_paper_vem_w.py ^
        --num_exp 1 --settings l2_q2_train1000 --max_anchors 20 ^
        --methods map_vem1,post_vem1_boundary --n_candidate 40 --n_vem 25 ^
        --steps 5 --prediction_heads mean,posterior --final_neighbor_modes raw ^
        --out_dir experiments/synthetic/vem_w_smoke

This runner is the more faithful implementation requested after diagnosing
prior-like LMJGP ``q(W)``.  It trains a low-rank diagonal Gaussian posterior
``q(C_j)`` that induces ``q(W_j)``.  During training, local JGP responsibilities
``gamma`` are updated by conditional-W exact VEM:

1. sample ``W_j`` from ``q(W_j)``;
2. compute the exact local GP posterior ``q(f_j | W_j, gamma_j)``;
3. update ``gamma_j`` using expected inlier likelihood plus a weak logistic
   gate in projected space;
4. update ``q(C_j)`` by maximizing a Monte Carlo estimate of the collapsed
   soft JGP bound.

The implementation supports early-stopped VEM (`vem1` vs `vem5`), MAP-W vs
posterior-W (`map_*` vs `post_*`), boundary-weighted anchors, KL annealing, and
shared/frozen nuisance hyperparameters.  Final predictions still use the same
JumpGP-CEM refit head as the rest of this repository so performance is directly
comparable to the existing synthetic baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
from djgp.projections import build_V_basis, compute_pls_W0, run_projected_jumpgp  # noqa: E402
from djgp.projections.conditional_vem import VemWConfig, train_conditional_vem_w  # noqa: E402
from djgp.projections.inductive import run_projected_jumpgp_ensemble  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


METHOD_ORDER = (
    "map_vem1",
    "map_vem5",
    "map_vem1_boundary",
    "post_vem1",
    "post_vem3",
    "post_vem1_boundary",
    "post_vem3_boundary",
    "post_vem1_trainhyp_boundary",
)
FINAL_NEIGHBOR_MODES = ("raw", "projected")
PREDICTION_HEADS = ("mean", "posterior")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting",
        "family",
        "seed",
        "method",
        "prediction_head",
        "final_neighbor_mode",
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
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault(
                (
                    str(row["setting"]),
                    str(row["method"]),
                    str(row["prediction_head"]),
                    str(row["final_neighbor_mode"]),
                ),
                [],
            ).append(row)
    out: list[dict[str, Any]] = []
    for (setting, method, head, final_mode), group in sorted(groups.items()):
        agg: dict[str, Any] = {
            "setting": setting,
            "method": method,
            "prediction_head": head,
            "final_neighbor_mode": final_mode,
            "n_seeds": len(group),
        }
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


def _metrics_pack(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> list[float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, wall_sec, y, mu, sigma)


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


def _lists_from_stack(
    X_stack: torch.Tensor,
    y_stack: torch.Tensor,
    *,
    n_neighbors: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return (
        [X_stack[t, :n_neighbors].contiguous() for t in range(X_stack.shape[0])],
        [y_stack[t, :n_neighbors].contiguous() for t in range(y_stack.shape[0])],
    )


def _projected_neighbor_lists(
    W_mean: torch.Tensor,
    X_anchor: torch.Tensor,
    X_candidate: torch.Tensor,
    y_candidate: torch.Tensor,
    *,
    n_neighbors: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    with torch.no_grad():
        Z_candidate = torch.einsum("tmd,tqd->tmq", X_candidate, W_mean)
        z_anchor = torch.einsum("td,tqd->tq", X_anchor, W_mean)
        d2 = (Z_candidate - z_anchor.unsqueeze(1)).pow(2).sum(dim=-1)
        idx = torch.topk(d2, k=int(n_neighbors), largest=False, dim=1).indices
    X_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    for t in range(X_candidate.shape[0]):
        X_list.append(X_candidate[t, idx[t]].contiguous())
        y_list.append(y_candidate[t, idx[t]].contiguous())
    return X_list, y_list


def _method_to_cfg(method: str, args: argparse.Namespace) -> VemWConfig:
    if method not in METHOD_ORDER:
        raise ValueError(f"Unknown method {method!r}")
    m = re.search(r"vem(\d+)", method)
    vem_steps = int(m.group(1)) if m else 1
    posterior = method.startswith("post_")
    boundary = "boundary" in method
    trainhyp = "trainhyp" in method
    return VemWConfig(
        posterior=posterior,
        n_steps=int(args.steps),
        vem_steps=vem_steps,
        mc_train=max(1, int(args.mc_train if posterior else 1)),
        mc_gamma=max(1, int(args.mc_gamma if posterior else 1)),
        lr=float(args.lr),
        coeff_scale=float(args.coeff_scale),
        init_log_std=float(args.init_log_std),
        prior_std=float(args.prior_std),
        kl_weight=float(args.kl_weight if posterior else 0.0),
        kl_warmup_steps=int(args.kl_warmup_steps),
        lambda_smooth=float(args.lambda_smooth),
        lambda_coeff=float(args.lambda_coeff),
        lambda_gate_l2=float(args.lambda_gate_l2),
        gamma_init=float(args.gamma_init),
        gamma_floor=float(args.gamma_floor),
        gamma_damping=float(args.gamma_damping),
        temperature=float(args.temperature),
        temperature_final=float(args.temperature_final),
        boundary_weighted=boundary,
        train_kernel=trainhyp,
        train_noise=trainhyp,
        init_lengthscale=float(args.init_lengthscale),
        init_amp=float(args.init_amp),
        init_noise_var=float(args.init_noise_var),
        outlier_sigma_mult=float(args.outlier_sigma_mult),
        jitter=float(args.jitter),
        log_interval=int(args.log_interval),
        verbose=bool(args.verbose),
    )


def _base_row(
    *,
    setting: str,
    cfg: dict[str, Any],
    seed: int,
    method: str,
    head: str,
    final_mode: str,
    n_final: int,
    n_candidate: int,
    n_vem: int,
    n_eval: int,
) -> dict[str, Any]:
    return {
        "setting": setting,
        "family": cfg["family"],
        "seed": int(seed),
        "method": method,
        "prediction_head": head,
        "final_neighbor_mode": final_mode,
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


def _evaluate(
    *,
    result,
    method: str,
    head: str,
    final_mode: str,
    X_test_t: torch.Tensor,
    X_candidate: torch.Tensor,
    y_candidate: torch.Tensor,
    y_test: np.ndarray,
    n_final: int,
    eval_mc: int,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    W_mean = result.W_mean
    if final_mode == "raw":
        X_list, y_list = _lists_from_stack(X_candidate, y_candidate, n_neighbors=n_final)
    elif final_mode == "projected":
        # Use posterior mean W for projected-neighbor selection so all W samples
        # are evaluated against the same candidate subset.
        X_list, y_list = _projected_neighbor_lists(
            W_mean,
            X_test_t,
            X_candidate,
            y_candidate,
            n_neighbors=n_final,
        )
    else:
        raise ValueError(final_mode)

    if head == "mean":
        out = run_projected_jumpgp(
            W_mean,
            X_test_t,
            X_list,
            y_list,
            device=device,
            progress=False,
            desc=f"{method}/{head}/{final_mode}",
        )
    elif head == "posterior":
        if not result.model.posterior:
            raise ValueError("posterior prediction head requested for MAP method")
        with torch.no_grad():
            W_samples = result.model.sample_W(int(eval_mc), mean_only=False).detach()
        out = run_projected_jumpgp_ensemble(
            W_samples,
            X_test_t,
            X_list,
            y_list,
            device=device,
            progress=False,
            desc=f"{method}/{head}/{final_mode}",
        )
    else:
        raise ValueError(head)
    pred_sec = float(out["elapsed_sec"])
    pack = _metrics_pack(
        y_test,
        np.asarray(out["mu"], dtype=np.float64),
        np.asarray(out["sigma"], dtype=np.float64),
        float(result.train_time_sec + pred_sec),
    )
    return pack, {"pred_sec": pred_sec}


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Conditional-W JGP-VEM on Paper Synthetic Data

Purpose: implement and test a more faithful VEM/EM-style projection learner
where a posterior over W is passed into exact local JGP-VEM updates.

```json
{json.dumps(summary, indent=2)}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method, per-prediction-head rows.
- `aggregate.csv`: means/stds over successful rows.
- `histories/*.json`: VEM-W training histories.
- `summary.json`: exact runner arguments.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional-W exact JGP-VEM synthetic comparison.")
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--settings", type=str, default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--methods", type=str, default="map_vem1,map_vem5,post_vem1,post_vem1_boundary")
    p.add_argument("--prediction_heads", type=str, default="mean,posterior")
    p.add_argument("--final_neighbor_modes", type=str, default="raw,projected")
    p.add_argument("--max_anchors", type=int, default=None)
    p.add_argument("--n_candidate", type=int, default=100)
    p.add_argument("--n_vem", type=int, default=50)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--mc_train", type=int, default=2)
    p.add_argument("--mc_gamma", type=int, default=2)
    p.add_argument("--eval_mc", type=int, default=3)
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
    p.add_argument("--gamma_damping", type=float, default=0.7)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--temperature_final", type=float, default=1.0)
    p.add_argument("--init_lengthscale", type=float, default=1.0)
    p.add_argument("--init_amp", type=float, default=1.0)
    p.add_argument("--init_noise_var", type=float, default=0.1)
    p.add_argument("--outlier_sigma_mult", type=float, default=2.5)
    p.add_argument("--jitter", type=float, default=1e-5)
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out_dir", type=str, default="experiments/synthetic/vem_w_paper_train1000_seed0")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    heads = [s.strip() for s in args.prediction_heads.split(",") if s.strip()]
    final_modes = [s.strip() for s in args.final_neighbor_modes.split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    unknown_methods = sorted(set(methods) - set(METHOD_ORDER))
    unknown_heads = sorted(set(heads) - set(PREDICTION_HEADS))
    unknown_modes = sorted(set(final_modes) - set(FINAL_NEIGHBOR_MODES))
    if unknown_settings:
        raise ValueError(f"Unknown settings: {unknown_settings}")
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    if unknown_heads:
        raise ValueError(f"Unknown prediction heads: {unknown_heads}")
    if unknown_modes:
        raise ValueError(f"Unknown final neighbor modes: {unknown_modes}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = out_dir / "histories"
    hist_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(out_dir / "metrics_partial.csv") if args.resume else []
    if args.resume and not rows:
        rows = _read_csv(out_dir / "metrics.csv")
    completed = {
        (
            str(r.get("setting")),
            int(r.get("seed")),
            str(r.get("method")),
            str(r.get("prediction_head")),
            str(r.get("final_neighbor_mode")),
        )
        for r in rows
        if str(r.get("status")) == "ok" and str(r.get("seed", "")).strip() != ""
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for setting in settings:
        cfg = dict(SETTING_PRESETS[setting])
        n_final = int(cfg["n"])
        n_candidate = max(int(args.n_candidate), n_final, int(args.n_vem))
        n_candidate = min(n_candidate, int(cfg["N_train"]))
        n_vem = min(int(args.n_vem), n_candidate)
        n_final = min(n_final, n_candidate)
        for seed in range(int(args.num_exp)):
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

            for method in methods:
                pending = [
                    (head, mode)
                    for head in heads
                    for mode in final_modes
                    if not (head == "posterior" and not method.startswith("post_"))
                    and (setting, seed, method, head, mode) not in completed
                ]
                if not pending:
                    print(f"skip completed setting={setting} seed={seed} method={method}", flush=True)
                    continue
                print(f"setting={setting} seed={seed} method={method}", flush=True)
                train_diag: dict[str, Any] = {}
                try:
                    vem_cfg = _method_to_cfg(method, args)
                    result = train_conditional_vem_w(
                        X_neighbors=X_vem,
                        y_neighbors=y_vem_std,
                        X_anchors=X_test_t,
                        V=V_t,
                        W0_init=W0_t,
                        cfg=vem_cfg,
                    )
                    train_diag = dict(result.diagnostics)
                    train_diag.update(
                        {
                            "posterior": bool(vem_cfg.posterior),
                            "vem_steps": int(vem_cfg.vem_steps),
                            "mc_train": int(vem_cfg.mc_train),
                            "mc_gamma": int(vem_cfg.mc_gamma),
                            "boundary_weighted": bool(vem_cfg.boundary_weighted),
                            "train_kernel": bool(vem_cfg.train_kernel),
                            "V_Rv": int(V_t.shape[1]),
                            "V_info": json.dumps(V_info),
                        }
                    )
                    hist_path = hist_dir / f"{setting}_seed{seed}_{method}.json"
                    hist_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")
                    for head, final_mode in pending:
                        base = _base_row(
                            setting=setting,
                            cfg=cfg,
                            seed=seed,
                            method=method,
                            head=head,
                            final_mode=final_mode,
                            n_final=n_final,
                            n_candidate=n_candidate,
                            n_vem=n_vem,
                            n_eval=len(y_test),
                        )
                        print(
                            f"evaluate setting={setting} seed={seed} method={method} "
                            f"head={head} final={final_mode}",
                            flush=True,
                        )
                        pack, eval_diag = _evaluate(
                            result=result,
                            method=method,
                            head=head,
                            final_mode=final_mode,
                            X_test_t=X_test_t,
                            X_candidate=X_cand,
                            y_candidate=y_cand,
                            y_test=y_test,
                            n_final=n_final,
                            eval_mc=int(args.eval_mc),
                            device=device,
                        )
                        row = dict(base)
                        row["status"] = "ok"
                        for key, val in zip(RESULT_ROW_KEYS, pack):
                            row[key] = val
                        row.update(train_diag)
                        row.update(eval_diag)
                        rows.append(row)
                        completed.add((setting, seed, method, head, final_mode))
                        _write_csv(out_dir / "metrics_partial.csv", rows)
                except Exception as exc:
                    for head, final_mode in pending:
                        row = _base_row(
                            setting=setting,
                            cfg=cfg,
                            seed=seed,
                            method=method,
                            head=head,
                            final_mode=final_mode,
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
                    print(f"ERROR setting={setting} seed={seed} method={method}: {exc!r}", flush=True)

    _write_csv(out_dir / "metrics.csv", rows)
    agg = _aggregate(rows)
    _write_csv(out_dir / "aggregate.csv", agg)
    summary = {
        "args": vars(args),
        "device": str(device),
        "n_rows": len(rows),
        "n_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "methods": methods,
        "prediction_heads": heads,
        "final_neighbor_modes": final_modes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_readme(out_dir, summary)
    print(f"Saved {len(rows)} rows to {out_dir}", flush=True)
    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    main()

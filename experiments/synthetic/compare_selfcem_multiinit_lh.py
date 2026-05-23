"""Screen fixed-W multi-initialization self-CEM mixtures on paper LH data.

How to run:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_selfcem_multiinit_lh.py ^
        --settings lh_q5_train1000 --seed_start 0 --num_exp 1 ^
        --max_test 40 --pool_size 120 --n_neighbors 35 ^
        --inits raw_pls,pls,pca,pairwise,rand0,rand1 ^
        --cem_updates 2 --hyper_steps 3 --gate_steps 4 ^
        --out_dir experiments/synthetic/selfcem_multiinit_lh_seed0_40_20260518

    python experiments/synthetic/compare_selfcem_multiinit_lh.py ^
        --settings lh_q5_train1000 --seed_start 0 --num_exp 1 ^
        --max_test 10 --pool_size 80 --n_neighbors 35 ^
        --inits raw_pls,pls,pca,pairwise,rand0 ^
        --cem_updates 1 --hyper_steps 1 --gate_steps 1 ^
        --out_dir experiments/synthetic/selfcem_multiinit_smoke_20260518

This runner does not train W.  It treats diverse W initializations as candidate
geometries, reranks a raw-X candidate pool under each geometry, runs one batched
deterministic-W self-CEM head over all candidates, and forms a likelihood-soft
mixture of the resulting Gaussian predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import FixedLabelGateConfig, FixedLabelHyperoptConfig, SelfCEMConfig  # noqa: E402
from djgp.projections import run_uncertain_w_self_cem, uncertain_w_cem_predict_from_labels  # noqa: E402
from djgp.projections.uncertain_w_jgp import _hard_cem_bound_per_anchor_from_w_moments  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import (  # noqa: E402
    _aggregate,
    _default_hypers_from_labels,
    _label_diag,
    _metrics_pack,
    _neighbor_stack,
    _pairwise_W0,
    _pca_W0,
    _projection_W0,
    _response_seed_labels,
    _write_csv,
)


def _random_orthonormal_W(Q: int, D: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    A = rng.standard_normal((D, D))
    q, _ = np.linalg.qr(A)
    return q[:, : int(Q)].T.astype(np.float64)


def _candidate_Ws(
    X_train: np.ndarray,
    y_train_std: np.ndarray,
    Q: int,
    names: list[str],
    *,
    seed: int,
) -> tuple[list[str], torch.Tensor]:
    Ws: list[np.ndarray] = []
    labels: list[str] = []
    D = int(X_train.shape[1])
    cache: dict[str, np.ndarray] = {}
    for raw_name in names:
        name = raw_name.strip()
        if not name:
            continue
        base = name
        if name == "raw_pls":
            base = "pls"
        if base not in cache:
            if base == "pls":
                cache[base] = _projection_W0(X_train, y_train_std, Q, "pls", seed=seed + 101)
            elif base == "pca":
                cache[base] = _pca_W0(X_train, Q, scaled=False)
            elif base == "pca_scaled":
                cache[base] = _pca_W0(X_train, Q, scaled=True)
            elif base == "pairwise":
                cache[base] = _pairwise_W0(X_train, y_train_std, Q, seed=seed + 211)
            elif base.startswith("rand"):
                suffix = base[4:]
                offset = int(suffix) if suffix.isdigit() else len(cache)
                cache[base] = _random_orthonormal_W(Q, D, seed=seed + 1009 + offset)
            else:
                raise ValueError(f"Unknown W init {name!r}.")
        Ws.append(cache[base])
        labels.append(name)
    if not Ws:
        raise ValueError("At least one W initialization is required.")
    return labels, torch.as_tensor(np.stack(Ws, axis=0), dtype=torch.float64)


def _rerank_pool(
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    W_all: torch.Tensor,
    names: list[str],
    *,
    n_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L = int(W_all.shape[0])
    T, P, D = X_pool.shape
    X_out = torch.empty(L, T, int(n_neighbors), D, device=X_anchor.device, dtype=X_anchor.dtype)
    y_out = torch.empty(L, T, int(n_neighbors), device=X_anchor.device, dtype=y_pool.dtype)
    idx_out = torch.empty(L, T, int(n_neighbors), device=X_anchor.device, dtype=torch.long)
    for ell, name in enumerate(names):
        if name == "raw_pls":
            dist = (X_pool - X_anchor.unsqueeze(1)).pow(2).sum(dim=-1)
        else:
            delta = X_pool - X_anchor.unsqueeze(1)
            z = torch.einsum("qd,tpd->tpq", W_all[ell].to(device=X_anchor.device, dtype=X_anchor.dtype), delta)
            dist = z.pow(2).sum(dim=-1)
        local = torch.topk(dist, k=min(int(n_neighbors), P), largest=False, dim=1).indices
        take = local.unsqueeze(-1).expand(-1, -1, D)
        X_out[ell] = torch.gather(X_pool, 1, take)
        y_out[ell] = torch.gather(y_pool, 1, local)
        idx_out[ell] = local
    return X_out, y_out, idx_out


def _pack_metrics(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> dict[str, float]:
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
    vals = pack_gaussian_uq_list(rmse, crps, wall_sec, y, mu, sigma)
    return {key: float(vals[i]) for i, key in enumerate(RESULT_ROW_KEYS)}


def _mixture_moments(mu: torch.Tensor, var: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = (weight * mu).sum(dim=0)
    second = (weight * (var + mu.pow(2))).sum(dim=0)
    return mean, (second - mean.pow(2)).clamp_min(1e-12)


def _run_one(args: argparse.Namespace, setting: str, seed: int, device: torch.device) -> list[dict[str, Any]]:
    cfg = dict(SETTING_PRESETS[setting])
    _set_seed(seed)
    data = _generate_paper_data(cfg, seed, device)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data["X_train"]).astype(np.float64)
    X_test = scaler.transform(data["X_test"]).astype(np.float64)
    y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
    y_test_full = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-12)
    y_train_std = ((y_train - y_mean) / y_std).astype(np.float64)
    max_test = min(int(args.max_test), X_test.shape[0])
    X_test = X_test[:max_test]
    y_test = y_test_full[:max_test]
    X_train_t = torch.as_tensor(X_train, device=device, dtype=torch.float64)
    y_train_std_t = torch.as_tensor(y_train_std, device=device, dtype=torch.float64)
    X_anchor = torch.as_tensor(X_test, device=device, dtype=torch.float64)
    Q = int(cfg["Q"])
    init_names, W_all = _candidate_Ws(
        X_train,
        y_train_std,
        Q,
        [s.strip() for s in str(args.inits).split(",") if s.strip()],
        seed=seed,
    )
    W_all = W_all.to(device=device, dtype=torch.float64)

    t0 = time.perf_counter()
    X_pool, y_pool, _raw_idx = _neighbor_stack(
        X_anchor,
        X_train_t,
        y_train_std_t,
        n_neighbors=max(int(args.pool_size), int(args.n_neighbors)),
    )
    X_by_init, y_by_init, _pool_rank_idx = _rerank_pool(
        X_anchor,
        X_pool,
        y_pool,
        W_all,
        init_names,
        n_neighbors=int(args.n_neighbors),
    )
    L, T, n, D = X_by_init.shape
    X_anchor_flat = X_anchor.unsqueeze(0).expand(L, -1, -1).reshape(L * T, D).contiguous()
    X_neighbors_flat = X_by_init.reshape(L * T, n, D).contiguous()
    y_neighbors_flat = y_by_init.reshape(L * T, n).contiguous()
    W_flat = W_all[:, None].expand(-1, T, -1, -1).reshape(L * T, Q, D).contiguous()
    zero_var = torch.zeros_like(W_flat)
    init_labels = _response_seed_labels(
        X_anchor_flat,
        X_neighbors_flat,
        y_neighbors_flat,
        seed_frac=float(args.response_seed_frac),
        inlier_frac=float(args.response_inlier_frac),
        min_inliers=int(args.min_inliers),
    )
    hp0 = _default_hypers_from_labels(
        X_anchor_flat,
        X_neighbors_flat,
        y_neighbors_flat,
        init_labels,
        W_flat[0],
        min_inliers=int(args.min_inliers),
    )
    gate0 = torch.zeros(L * T, Q + 1, device=device, dtype=torch.float64)
    state = run_uncertain_w_self_cem(
        X_anchor_flat,
        X_neighbors_flat,
        y_neighbors_flat,
        init_labels,
        mode="anchor",
        W_mu_anchor=W_flat,
        W_var_anchor=zero_var,
        init_lengthscale=hp0["lengthscale"],
        init_signal_var=hp0["signal_var"],
        init_noise_var=hp0["noise_var"],
        gate_init=gate0,
        config=SelfCEMConfig(
            cem_updates=int(args.cem_updates),
            hyperopt=FixedLabelHyperoptConfig(
                steps=int(args.hyper_steps),
                lr=float(args.hyper_lr),
                batched=True,
                min_noise_var=float(args.noise_std_floor) ** 2,
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
    train_sec = time.perf_counter() - t0

    t_pred = time.perf_counter()
    pred = uncertain_w_cem_predict_from_labels(
        X_anchor_flat,
        X_neighbors_flat,
        y_neighbors_flat,
        state["labels"],
        mode="anchor",
        W_mu_anchor=W_flat,
        W_var_anchor=zero_var,
        lengthscale=state["lengthscale"],
        signal_var=state["signal_var"],
        noise_var=state["noise_var"],
        mean_const=state["mean_const"],
        correction_weight=0.0,
        min_inliers=int(args.min_inliers),
    )
    per_bound, bound_diag = _hard_cem_bound_per_anchor_from_w_moments(
        X_anchor_flat,
        X_neighbors_flat,
        y_neighbors_flat,
        state["labels"].to(dtype=torch.float64),
        mode="anchor",
        w_moments={"W_mu_anchor": W_flat, "W_var_anchor": zero_var},
        gate=state["gate"],
        lengthscale=state["lengthscale"],
        signal_var=state["signal_var"],
        noise_var=state["noise_var"],
        mean_const=state["mean_const"],
        outlier_mode="background_normal",
        outlier_sigma_mult=float(args.outlier_sigma_mult),
        background_scale_mult=float(args.background_scale_mult),
        gh_points=int(args.gh_points),
        jitter=1e-5,
        min_inliers=int(args.min_inliers),
    )
    pred_sec = time.perf_counter() - t_pred

    mu_std = pred.mu.reshape(L, T)
    var_std = pred.var.reshape(L, T).clamp_min(1e-12)
    score = per_bound.reshape(L, T)
    weights = torch.softmax(score / max(float(args.mixture_temperature), 1e-6), dim=0)
    mix_mu_std, mix_var_std = _mixture_moments(mu_std, var_std, weights)
    oracle_idx = torch.argmax(score, dim=0)
    oracle_mu_std = mu_std.gather(0, oracle_idx.unsqueeze(0)).squeeze(0)
    oracle_var_std = var_std.gather(0, oracle_idx.unsqueeze(0)).squeeze(0)

    rows: list[dict[str, Any]] = []
    base = {
        "setting": setting,
        "family": str(cfg.get("family", "LH")),
        "seed": int(seed),
        "N_train": int(cfg["N_train"]),
        "N_test": int(cfg["N_test"]),
        "n_test_eval": int(T),
        "Q": int(Q),
        "observed_dim": int(cfg.get("observed_dim", cfg.get("H"))),
        "latent_dim": int(cfg.get("latent_dim", cfg.get("d"))),
        "n_neighbors": int(args.n_neighbors),
        "pool_size": int(args.pool_size),
        "cem_updates": int(args.cem_updates),
        "hyper_steps": int(args.hyper_steps),
        "gate_steps": int(args.gate_steps),
        "mixture_temperature": float(args.mixture_temperature),
        "standardize_x": True,
        "standardize_y_for_head": True,
        "generator_family": str(cfg.get("family", "LH")),
        "expansion": str(cfg.get("expansion", "rff")),
    }
    scale = abs(float(y_std))
    y_test_np = np.asarray(y_test, dtype=np.float64)
    for ell, name in enumerate(init_names):
        mu = mu_std[ell].detach().cpu().numpy() * float(y_std) + float(y_mean)
        sigma = var_std[ell].sqrt().detach().cpu().numpy() * scale
        row = dict(base)
        row.update(_pack_metrics(y_test_np, mu, sigma, train_sec + pred_sec))
        row.update(_label_diag(init_labels.reshape(L, T, n)[ell].bool(), state["labels"].reshape(L, T, n)[ell].bool(), n))
        row.update(
            {
                "variant": f"fixed_{name}_selfcem",
                "status": "ok",
                "kind": "fixed_init",
                "W_init": name,
                "train_sec": float(train_sec),
                "pred_sec": float(pred_sec),
                "mean_bound": float(score[ell].mean().detach().cpu().item()),
                "mean_mixture_weight": float(weights[ell].mean().detach().cpu().item()),
            }
        )
        rows.append(row)

    mix_mu = mix_mu_std.detach().cpu().numpy() * float(y_std) + float(y_mean)
    mix_sigma = mix_var_std.sqrt().detach().cpu().numpy() * scale
    mix_row = dict(base)
    mix_row.update(_pack_metrics(y_test_np, mix_mu, mix_sigma, train_sec + pred_sec))
    mix_row.update(
        {
            "variant": "likelihood_soft_mixture",
            "status": "ok",
            "kind": "mixture",
            "W_init": ",".join(init_names),
            "train_sec": float(train_sec),
            "pred_sec": float(pred_sec),
            "mixture_weight_entropy": float((-(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=0)).mean().detach().cpu().item()),
            "mixture_between_sigma_mean": float(((weights * (mu_std - mix_mu_std.unsqueeze(0)).pow(2)).sum(dim=0).sqrt() * scale).mean().detach().cpu().item()),
            "mixture_weights_json": json.dumps({name: float(weights[i].mean().detach().cpu().item()) for i, name in enumerate(init_names)}, sort_keys=True),
            "bound_diag_json": json.dumps(
                {
                    k: float(v.detach().cpu().item()) if isinstance(v, torch.Tensor) and v.numel() == 1 else sanitize_for_json(v)
                    for k, v in bound_diag.items()
                },
                sort_keys=True,
            ),
        }
    )
    rows.append(mix_row)

    uniform_weights = torch.full_like(weights, 1.0 / float(L))
    uniform_mu_std, uniform_var_std = _mixture_moments(mu_std, var_std, uniform_weights)
    uniform_mu = uniform_mu_std.detach().cpu().numpy() * float(y_std) + float(y_mean)
    uniform_sigma = uniform_var_std.sqrt().detach().cpu().numpy() * scale
    uniform_row = dict(base)
    uniform_row.update(_pack_metrics(y_test_np, uniform_mu, uniform_sigma, train_sec + pred_sec))
    uniform_row.update(
        {
            "variant": "uniform_mixture",
            "status": "ok",
            "kind": "mixture_control",
            "W_init": ",".join(init_names),
            "train_sec": float(train_sec),
            "pred_sec": float(pred_sec),
            "mixture_weight_entropy": float(math.log(float(L))),
            "mixture_between_sigma_mean": float(((uniform_weights * (mu_std - uniform_mu_std.unsqueeze(0)).pow(2)).sum(dim=0).sqrt() * scale).mean().detach().cpu().item()),
            "mixture_weights_json": json.dumps({name: float(1.0 / float(L)) for name in init_names}, sort_keys=True),
        }
    )
    rows.append(uniform_row)

    y_std_t = torch.as_tensor((y_test_np - float(y_mean)) / float(y_std), device=device, dtype=torch.float64)
    pred_logp = -0.5 * (
        math.log(2.0 * math.pi)
        + var_std.clamp_min(1e-12).log()
        + (y_std_t.unsqueeze(0) - mu_std).pow(2) / var_std.clamp_min(1e-12)
    )
    oracle_y_weights = torch.softmax(pred_logp / max(float(args.mixture_temperature), 1e-6), dim=0)
    oracle_y_mu_std, oracle_y_var_std = _mixture_moments(mu_std, var_std, oracle_y_weights)
    oracle_y_mu = oracle_y_mu_std.detach().cpu().numpy() * float(y_std) + float(y_mean)
    oracle_y_sigma = oracle_y_var_std.sqrt().detach().cpu().numpy() * scale
    oracle_y_row = dict(base)
    oracle_y_row.update(_pack_metrics(y_test_np, oracle_y_mu, oracle_y_sigma, train_sec + pred_sec))
    oracle_y_row.update(
        {
            "variant": "diagnostic_anchor_y_soft_mixture",
            "status": "ok",
            "kind": "diagnostic_oracle_y_mixture",
            "W_init": ",".join(init_names),
            "train_sec": float(train_sec),
            "pred_sec": float(pred_sec),
            "mixture_weight_entropy": float((-(oracle_y_weights.clamp_min(1e-12) * oracle_y_weights.clamp_min(1e-12).log()).sum(dim=0)).mean().detach().cpu().item()),
            "mixture_between_sigma_mean": float(((oracle_y_weights * (mu_std - oracle_y_mu_std.unsqueeze(0)).pow(2)).sum(dim=0).sqrt() * scale).mean().detach().cpu().item()),
            "mixture_weights_json": json.dumps({name: float(oracle_y_weights[i].mean().detach().cpu().item()) for i, name in enumerate(init_names)}, sort_keys=True),
            "uses_y_test_for_weights": True,
        }
    )
    rows.append(oracle_y_row)

    oracle_mu = oracle_mu_std.detach().cpu().numpy() * float(y_std) + float(y_mean)
    oracle_sigma = oracle_var_std.sqrt().detach().cpu().numpy() * scale
    oracle_row = dict(base)
    oracle_row.update(_pack_metrics(y_test_np, oracle_mu, oracle_sigma, train_sec + pred_sec))
    oracle_row.update(
        {
            "variant": "likelihood_hard_select",
            "status": "ok",
            "kind": "hard_select",
            "W_init": ",".join(init_names),
            "train_sec": float(train_sec),
            "pred_sec": float(pred_sec),
            "selected_counts_json": json.dumps({name: int((oracle_idx == i).sum().detach().cpu().item()) for i, name in enumerate(init_names)}, sort_keys=True),
        }
    )
    rows.append(oracle_row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed-W multi-init self-CEM mixture screen on paper LH data.")
    p.add_argument("--settings", default="lh_q5_train1000")
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--max_test", type=int, default=40)
    p.add_argument("--pool_size", type=int, default=120)
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--inits", default="raw_pls,pls,pca,pairwise,rand0,rand1")
    p.add_argument("--mixture_temperature", type=float, default=10.0)
    p.add_argument("--cem_updates", type=int, default=2)
    p.add_argument("--hyper_steps", type=int, default=3)
    p.add_argument("--gate_steps", type=int, default=4)
    p.add_argument("--hyper_lr", type=float, default=0.04)
    p.add_argument("--gate_lr", type=float, default=0.05)
    p.add_argument("--gate_l2", type=float, default=1e-3)
    p.add_argument("--gate_max_norm", type=float, default=8.0)
    p.add_argument("--gh_points", type=int, default=20)
    p.add_argument("--min_inliers", type=int, default=2)
    p.add_argument("--max_inlier_frac", type=float, default=0.95)
    p.add_argument("--noise_std_floor", type=float, default=0.1)
    p.add_argument("--max_noise_var", type=float, default=10.0)
    p.add_argument("--min_signal_var", type=float, default=1e-5)
    p.add_argument("--max_signal_var", type=float, default=100.0)
    p.add_argument("--min_lengthscale", type=float, default=1e-3)
    p.add_argument("--max_lengthscale", type=float, default=20.0)
    p.add_argument("--response_seed_frac", type=float, default=0.25)
    p.add_argument("--response_inlier_frac", type=float, default=0.70)
    p.add_argument("--outlier_sigma_mult", type=float, default=2.5)
    p.add_argument("--background_scale_mult", type=float, default=3.0)
    p.add_argument("--out_dir", default="experiments/synthetic/selfcem_multiinit_lh_seed0_40_20260518")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in str(args.settings).split(",") if s.strip()]
    unknown = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metrics_path = out_dir / "metrics.csv"
    if args.resume and metrics_path.exists():
        with metrics_path.open("r", newline="") as fh:
            rows = list(csv.DictReader(fh))
    done = {(r.get("setting"), int(r.get("seed", -1))) for r in rows if r.get("status") == "ok"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
        for setting in settings:
            key = (setting, seed)
            if key in done:
                print(f"skip completed setting={setting} seed={seed}", flush=True)
                continue
            print(f"run setting={setting} seed={seed}", flush=True)
            rows.extend(_run_one(args, setting, seed, device))
            _write_csv(metrics_path, rows)
            _write_csv(out_dir / "aggregate.csv", _aggregate(rows))
    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "args": vars(args),
        "settings": {name: SETTING_PRESETS[name] for name in settings},
        "n_rows": len(rows),
        "n_ok": int(sum(1 for r in rows if r.get("status") == "ok")),
    }
    (out_dir / "summary.json").write_text(json.dumps(sanitize_for_json(summary), indent=2), encoding="utf-8")
    readme = f"""# Self-CEM Multi-Initialization LH Screen

Runner: `experiments/synthetic/compare_selfcem_multiinit_lh.py`

This fixed-W screen treats W initializations as candidate geometries.  It uses
a raw-X candidate pool, reranks within that pool under each W, runs one batched
self-CEM head over all `(init, anchor)` pairs, and forms a likelihood-soft
Gaussian mixture by moment matching.

See `docs/experiments_log.md` for the dated result entry.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved results to {out_dir}", flush=True)
    return {"rows": rows, "aggregate": _aggregate(rows), "summary": summary}


if __name__ == "__main__":
    main()

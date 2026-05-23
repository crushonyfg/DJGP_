"""Run sparse-GP W(x) + local JGP-CEM ablations on paper L2/LH data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_global_wgp_cem_l2_lh.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants anchor_uniform,anchor_floor,anchor_coverage,anchor_background,pointwise_coverage,pointwise_background ^
        --anchor_count 40 --steps 60 --max_test 80 ^
        --out_dir experiments/synthetic/global_wgp_cem_l2_lh_20260514

Selected PCA/MAP/pairwise-warm 5-seed run:

    python experiments/synthetic/compare_global_wgp_cem_l2_lh.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants anchor_pca_background_map_soft,pointwise_pairwarm_map_soft ^
        --seed_start 0 --num_exp 5 --anchor_count 96 ^
        --boundary_anchor_frac 0.45 --random_anchor_frac 0.20 ^
        --steps 60 --max_test 200 --eval_mc 5 --m_inducing 50 ^
        --n_candidate 100 --pairwarm_steps 120 ^
        --out_dir experiments/synthetic/global_wgp_cem_selected_pca_map_5seeds_20260514

Faster smoke:

    python experiments/synthetic/compare_global_wgp_cem_l2_lh.py ^
        --settings l2_q2_train1000 ^
        --variants pointwise_pca_map_soft,pointwise_pairwarm_map_soft,anchor_pca_background_map_soft ^
        --anchor_count 8 --steps 3 --max_test 5 --eval_mc 1 ^
        --m_inducing 8 --n_candidate 30 --pairwarm_steps 3 ^
        --out_dir experiments/synthetic/global_wgp_cem_smoke
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
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics import sanitize_for_json  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import (  # noqa: E402
    GlobalWGPCemConfig,
    compute_pls_W0,
    predict_global_wgp_jumpgp,
    train_global_wgp_cem,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


VARIANTS = (
    "anchor_uniform",
    "anchor_floor",
    "anchor_coverage",
    "anchor_background",
    "pointwise_coverage",
    "pointwise_background",
    "pointwise_pca_map_soft",
    "pointwise_pairwarm_map_soft",
    "anchor_pca_background_map_soft",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting",
        "family",
        "seed",
        "variant",
        "mode",
        "background",
        "status",
        "N_train",
        "N_test",
        "n_train_anchors",
        "n_test_eval",
        "Q",
        "observed_dim",
        "n_train_neighbors",
        "n_candidate",
        "m_inducing",
        "init_mode",
        "map_R",
        *RESULT_ROW_KEYS,
        "train_sec",
        "pred_sec",
        "gamma_mean",
        "gamma_entropy",
        "visited_train_fraction",
        "soft_coverage_mean",
        "soft_coverage_min",
        "soft_coverage_below_min_fraction",
        "R_std_median",
        "kl_per_anchor",
        "grad_data_over_scaled_KL_R_mu_final",
        "grad_data_over_scaled_KL_R_log_std_final",
        "median_frob_W",
        "anchor_dispersion_frob",
        "cosine_to_ref_median",
        "error",
    ]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    keys = [k for k in preferred if k in all_keys]
    keys.extend(sorted(all_keys - set(keys)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    for (setting, variant), group in sorted(groups.items()):
        agg: dict[str, Any] = {
            "setting": setting,
            "variant": variant,
            "n_runs": len(group),
        }
        for meta in (
            "family",
            "mode",
            "background",
            "N_train",
            "N_test",
            "n_train_anchors",
            "n_test_eval",
            "Q",
            "observed_dim",
            "n_train_neighbors",
            "n_candidate",
            "m_inducing",
            "init_mode",
            "map_R",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in list(RESULT_ROW_KEYS) + [
            "train_sec",
            "pred_sec",
            "gamma_mean",
            "gamma_entropy",
            "visited_train_fraction",
            "soft_coverage_mean",
            "soft_coverage_min",
            "soft_coverage_below_min_fraction",
            "R_std_median",
            "kl_per_anchor",
            "grad_data_over_scaled_KL_R_mu_final",
            "grad_data_over_scaled_KL_R_log_std_final",
            "median_frob_W",
            "anchor_dispersion_frob",
            "cosine_to_ref_median",
        ]:
            vals = np.asarray([_float_or_nan(r.get(key)) for r in group], dtype=np.float64)
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


def _kmeans_medoids(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = X.shape[0]
    k = max(1, min(int(k), n))
    if k >= n:
        return np.arange(n, dtype=np.int64)
    km = KMeans(n_clusters=k, random_state=seed, n_init=5, max_iter=200)
    labels = km.fit_predict(X)
    out: list[int] = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        d2 = np.sum((X[idx] - km.cluster_centers_[c]) ** 2, axis=1)
        out.append(int(idx[int(np.argmin(d2))]))
    return np.asarray(sorted(set(out)), dtype=np.int64)


def _local_y_variance_indices(
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    *,
    k: int,
) -> tuple[np.ndarray, torch.Tensor]:
    with torch.no_grad():
        d = torch.cdist(X_t, X_t)
        nn = torch.topk(d, k=min(int(k), X_t.shape[0]), largest=False, dim=1).indices
        y_nn = y_t[nn]
        local_var = y_nn.var(dim=1, unbiased=False)
    order = torch.argsort(local_var, descending=True).detach().cpu().numpy().astype(np.int64)
    return order, local_var


def _select_train_anchors(
    X: np.ndarray,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    *,
    count: int,
    boundary_frac: float,
    random_frac: float,
    neighbor_k: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    count = min(int(count), X.shape[0])
    n_random = int(round(count * float(random_frac)))
    n_boundary = int(round(count * float(boundary_frac)))
    n_kmeans = max(1, count - n_boundary - n_random)
    medoids = list(_kmeans_medoids(X, n_kmeans, seed))
    order, local_var = _local_y_variance_indices(X_t, y_t, k=neighbor_k)
    chosen = list(medoids)
    seen = set(chosen)
    for idx in order:
        if len(chosen) >= count - n_random:
            break
        if int(idx) not in seen:
            chosen.append(int(idx))
            seen.add(int(idx))
    rng = np.random.default_rng(int(seed) + 104729)
    for idx in rng.permutation(X.shape[0]):
        if len(chosen) >= count:
            break
        if int(idx) not in seen:
            chosen.append(int(idx))
            seen.add(int(idx))
    arr = np.asarray(chosen[:count], dtype=np.int64)
    return arr, {
        "anchor_local_y_var_mean": float(local_var[arr].mean().detach().cpu().item()),
        "anchor_local_y_var_median": float(local_var[arr].median().detach().cpu().item()),
    }


def _neighbor_stack(
    X_query: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    n_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        d = torch.cdist(X_query, X_train)
        idx = torch.topk(d, k=min(int(n_neighbors), X_train.shape[0]), largest=False, dim=1).indices
    return X_train[idx].contiguous(), y_train[idx].contiguous(), idx.contiguous()


def _coverage_stats(gamma: torch.Tensor, neighbor_indices: torch.Tensor, n_train: int, c_min: float) -> dict[str, float]:
    with torch.no_grad():
        flat_idx = neighbor_indices.reshape(-1)
        flat_g = gamma.reshape(-1)
        sums = torch.zeros(n_train, device=gamma.device, dtype=gamma.dtype)
        counts = torch.zeros(n_train, device=gamma.device, dtype=gamma.dtype)
        sums.scatter_add_(0, flat_idx, flat_g)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_g))
        visited = counts > 0
        cov = sums[visited] / counts[visited].clamp_min(1.0)
        if cov.numel() == 0:
            return {
                "visited_train_fraction": 0.0,
                "soft_coverage_mean": 0.0,
                "soft_coverage_min": 0.0,
                "soft_coverage_below_min_fraction": 1.0,
            }
        return {
            "visited_train_fraction": float(visited.float().mean().item()),
            "soft_coverage_mean": float(cov.mean().item()),
            "soft_coverage_min": float(cov.min().item()),
            "soft_coverage_below_min_fraction": float((cov < float(c_min)).float().mean().item()),
        }


def _median_distance(X_t: torch.Tensor) -> float:
    with torch.no_grad():
        d = torch.cdist(X_t, X_t)
        vals = d[d > 0]
        if vals.numel() == 0:
            return 1.0
        return float(vals.median().clamp_min(1e-3).item())


def _row_normalize(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(eps)


def _pca_W0(X: np.ndarray, Q: int) -> np.ndarray:
    return PCA(n_components=int(Q), random_state=0).fit(np.asarray(X, dtype=np.float64)).components_.astype(np.float32)


def _pca_scaled_W0(X: np.ndarray, Q: int) -> np.ndarray:
    W = _pca_W0(X, Q)
    z = np.asarray(X, dtype=np.float64) @ W.T
    scale = (10.0 / np.maximum(np.ptp(z, axis=0), 1e-6) ** 2).reshape(-1, 1)
    return (W * scale).astype(np.float32)


class PointWxNet(nn.Module):
    def __init__(self, D: int, Q: int, hidden: int):
        super().__init__()
        self.Q = int(Q)
        self.D = int(D)
        self.net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, Q * D),
        )

    def W(self, X: torch.Tensor) -> torch.Tensor:
        raw = self.net(X).view(*X.shape[:-1], self.Q, self.D)
        return _row_normalize(raw)

    def z(self, X: torch.Tensor) -> torch.Tensor:
        W = self.W(X)
        return torch.einsum("...qd,...d->...q", W, X)


def _pairwise_metric_loss_per_anchor(Z: torch.Tensor, y: torch.Tensor, *, tau_y: float, margin: float) -> torch.Tensor:
    n = Z.shape[1]
    d2 = (Z.unsqueeze(2) - Z.unsqueeze(1)).pow(2).sum(-1)
    dy2 = (y.unsqueeze(2) - y.unsqueeze(1)).pow(2)
    s = torch.exp(-0.5 * dy2 / max(float(tau_y) ** 2, 1e-12))
    loss_mat = s * d2 + (1.0 - s) * torch.relu(float(margin) - d2).pow(2)
    mask = (~torch.eye(n, dtype=torch.bool, device=Z.device)).unsqueeze(0)
    return loss_mat.masked_select(mask).view(Z.shape[0], -1).sum(dim=1) / float(n * (n - 1))


def _response_variance_weights(y_std_stack: torch.Tensor) -> torch.Tensor:
    var = y_std_stack.var(dim=1, unbiased=False)
    weights = var / var.mean().clamp_min(1e-12)
    return weights.clamp(0.25, 4.0).detach()


def _default_pair_tau(y_std_stack: torch.Tensor) -> float:
    with torch.no_grad():
        n = y_std_stack.shape[1]
        dy = (y_std_stack.unsqueeze(2) - y_std_stack.unsqueeze(1)).abs()
        mask = (~torch.eye(n, dtype=torch.bool, device=y_std_stack.device)).unsqueeze(0)
        vals = dy.masked_select(mask)
        return float(vals.median().clamp_min(0.05).item())


def _train_pairwarm_net(
    *,
    X_pair: torch.Tensor,
    y_pair_std: torch.Tensor,
    X_inducing: torch.Tensor,
    Q: int,
    steps: int,
    lr: float,
    hidden: int,
    margin: float,
    weight_decay: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    D = X_pair.shape[-1]
    net = PointWxNet(D, Q, hidden).to(X_pair.device)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    weights = _response_variance_weights(y_pair_std)
    tau = _default_pair_tau(y_pair_std)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    last_loss = float("nan")
    t0 = time.perf_counter()
    for _ in range(1, int(steps) + 1):
        opt.zero_grad()
        per_anchor = _pairwise_metric_loss_per_anchor(net.z(X_pair), y_pair_std, tau_y=tau, margin=float(margin))
        loss = (weights * per_anchor).mean()
        loss.backward()
        opt.step()
        val = float(loss.detach().item())
        last_loss = val
        if val < best_loss:
            best_loss = val
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        R0 = net.W(X_inducing).detach()
    return R0, {
        "pairwarm_sec": float(time.perf_counter() - t0),
        "pairwarm_loss": float(last_loss),
        "pairwarm_best_loss": float(best_loss),
        "pairwarm_tau": float(tau),
    }


def _variant_cfg(name: str, args: argparse.Namespace, w_lengthscale: float) -> GlobalWGPCemConfig:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}")
    mode = "pointwise" if name.startswith("pointwise") else "anchor"
    background = "normal" if name.endswith("background") or "background" in name else "uniform"
    gamma_floor = 1e-4 if name == "anchor_uniform" else float(args.gamma_floor)
    coverage_weight = float(args.coverage_weight) if "coverage" in name or "background" in name else 0.0
    map_r = name.endswith("_map_soft") or "_map_" in name
    return GlobalWGPCemConfig(
        mode=mode,
        steps=int(args.steps),
        c_refresh=int(args.c_refresh),
        mc_train=int(args.mc_train),
        lr=float(args.lr),
        beta_kl=float(args.beta_kl),
        kl_warmup_steps=int(args.kl_warmup_steps),
        composite_temperature=float(args.composite_temperature),
        gamma_init=float(args.gamma_init),
        gamma_floor=float(gamma_floor),
        gamma_temperature=float(args.gamma_temperature),
        gamma_damping=float(args.gamma_damping),
        coverage_weight=coverage_weight,
        coverage_min=float(args.coverage_min),
        background=background,
        background_scale=float(args.background_scale),
        init_log_std=float(args.init_log_std),
        init_lengthscale=float(args.local_lengthscale),
        init_w_lengthscale=float(w_lengthscale),
        init_noise_var=float(args.init_noise_var),
        init_amp=float(args.init_amp),
        train_kernel=bool(args.train_kernel),
        train_noise=bool(args.train_noise),
        train_R_log_std=bool(args.train_R_log_std) and not map_r,
        sample_train_R=bool(args.sample_train_R) and not map_r,
        train_w_lengthscale=bool(args.train_w_lengthscale),
        train_w_var=bool(args.train_w_var),
        outlier_sigma_mult=float(args.outlier_sigma_mult),
        gate_l2=float(args.gate_l2),
        verbose=bool(args.verbose),
        log_interval=int(args.log_interval),
    )


def _variant_init_mode(name: str, args: argparse.Namespace) -> str:
    if "pairwarm" in name:
        return "pairwarm"
    if "pca_scaled" in name:
        return "pca_scaled"
    if "pca" in name:
        return "pca"
    return str(args.init_mode)


def _base_row(args: argparse.Namespace, cfg: dict[str, Any], setting: str, seed: int, variant: str) -> dict[str, Any]:
    mode = "pointwise" if variant.startswith("pointwise") else "anchor"
    background = "normal" if variant.endswith("background") or "background" in variant else "uniform"
    map_r = variant.endswith("_map_soft") or "_map_" in variant
    return {
        "setting": setting,
        "family": cfg["family"],
        "seed": int(seed),
        "variant": variant,
        "mode": mode,
        "background": background,
        "N_train": int(cfg["N_train"]),
        "N_test": int(cfg["N_test"]),
        "n_train_anchors": int(args.anchor_count),
        "n_test_eval": int(args.max_test if args.max_test is not None else cfg["N_test"]),
        "Q": int(cfg["Q"]),
        "observed_dim": int(cfg["H"]),
        "latent_dim": int(cfg["d"]),
        "Q_true": int(cfg["Q_true"]),
        "n_train_neighbors": int(args.train_neighbors or cfg["n"]),
        "n_candidate": int(args.n_candidate),
        "m_inducing": int(args.m_inducing),
        "init_mode": _variant_init_mode(variant, args),
        "map_R": bool(map_r),
        "standardize_x": True,
        "standardize_y_for_training": True,
        "prediction_y_scale": "original",
        "expansion": str(cfg.get("expansion", "rff")),
        "noise_std": float(cfg["noise_std"]),
        "noise_var": float(cfg["noise_var"]),
    }


def _run_one(args: argparse.Namespace, setting: str, seed: int, variant: str, device: torch.device) -> dict[str, Any]:
    cfg = dict(SETTING_PRESETS[setting])
    _set_seed(seed)
    data = _generate_paper_data(cfg, seed, device)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data["X_train"]).astype(np.float32)
    X_test_s = scaler.transform(data["X_test"]).astype(np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.float64)
    if args.max_test is not None:
        X_test_s = X_test_s[: int(args.max_test)]
        y_test = y_test[: int(args.max_test)]
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() if y_train.std() > 1e-8 else 1.0)
    y_train_std = ((y_train - y_mean) / y_std).astype(np.float32)

    X_train_t = torch.from_numpy(X_train_s).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    y_train_std_t = torch.from_numpy(y_train_std).to(device)
    X_test_t = torch.from_numpy(X_test_s).to(device)
    y_candidate_t = y_train_t
    Q = int(cfg["Q"])
    train_neighbors = int(args.train_neighbors or cfg["n"])
    n_final = int(args.n_final or cfg["n"])
    row = _base_row(args, cfg, setting, seed, variant)

    try:
        init_mode = _variant_init_mode(variant, args)
        if init_mode == "pca":
            W0_np = _pca_W0(X_train_s, Q)
        elif init_mode == "pca_scaled":
            W0_np = _pca_scaled_W0(X_train_s, Q)
        elif init_mode in {"pls", "pairwarm"}:
            W0_np = compute_pls_W0(X_train_s, y_train_std, Q=Q).astype(np.float32)
        else:
            raise ValueError(f"Unknown init_mode {init_mode!r}")
        W0_t = torch.from_numpy(W0_np).to(device)
        anchor_idx, anchor_diag = _select_train_anchors(
            X_train_s,
            X_train_t,
            y_train_std_t,
            count=int(args.anchor_count),
            boundary_frac=float(args.boundary_anchor_frac),
            random_frac=float(args.random_anchor_frac),
            neighbor_k=max(train_neighbors, int(args.local_var_neighbors)),
            seed=seed,
        )
        inducing_idx = _kmeans_medoids(X_train_s, min(int(args.m_inducing), X_train_s.shape[0]), seed + 7919)
        X_anchor = X_train_t[torch.as_tensor(anchor_idx, device=device, dtype=torch.long)]
        X_neighbors, y_neighbors_std, neighbor_indices = _neighbor_stack(
            X_anchor,
            X_train_t,
            y_train_std_t,
            n_neighbors=train_neighbors,
        )
        X_inducing = X_train_t[torch.as_tensor(inducing_idx, device=device, dtype=torch.long)]
        init_R_mu = None
        warm_diag: dict[str, float] = {}
        if init_mode == "pairwarm":
            init_R_mu, warm_diag = _train_pairwarm_net(
                X_pair=X_neighbors,
                y_pair_std=y_neighbors_std,
                X_inducing=X_inducing,
                Q=Q,
                steps=int(args.pairwarm_steps),
                lr=float(args.pairwarm_lr),
                hidden=int(args.pairwarm_hidden),
                margin=float(args.pair_margin),
                weight_decay=float(args.pairwarm_weight_decay),
            )
        X_candidate, y_candidate, _ = _neighbor_stack(
            X_test_t,
            X_train_t,
            y_candidate_t,
            n_neighbors=max(int(args.n_candidate), n_final),
        )
        w_lengthscale = float(args.w_lengthscale if args.w_lengthscale is not None else _median_distance(X_inducing))
        cem_cfg = _variant_cfg(variant, args, w_lengthscale)
        result = train_global_wgp_cem(
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            y_neighbors=y_neighbors_std,
            neighbor_indices=neighbor_indices,
            X_inducing=X_inducing,
            n_train=X_train_t.shape[0],
            Q=Q,
            init_W=W0_t,
            init_R_mu=init_R_mu,
            cfg=cem_cfg,
        )
        pred = predict_global_wgp_jumpgp(
            field=result.field,
            X_test=X_test_t,
            X_candidate=X_candidate,
            y_candidate=y_candidate,
            n_neighbors=n_final,
            mode=cem_cfg.mode,
            n_samples=int(args.eval_mc),
            device=device,
        )
        pack = _metrics_pack(y_test, pred["mu"], pred["sigma"], float(pred["elapsed_sec"]))
        row["status"] = "ok"
        row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
        diag = dict(result.diagnostics)
        diag.update(warm_diag)
        diag.update(anchor_diag)
        diag.update(_coverage_stats(result.gamma, neighbor_indices, X_train_t.shape[0], float(args.coverage_min)))
        diag["pred_sec"] = float(pred["elapsed_sec"])
        diag["n_final"] = int(n_final)
        diag["w_lengthscale_init"] = float(w_lengthscale)
        if result.history:
            last = result.history[-1]
            for key in (
                "grad_data_over_scaled_KL_R_mu",
                "grad_data_over_scaled_KL_R_log_std",
                "grad_data_R_mu",
                "grad_scaled_KL_R_mu",
                "grad_data_R_log_std",
                "grad_scaled_KL_R_log_std",
            ):
                if key in last:
                    diag[f"{key}_final"] = float(last[key])
        row.update({k: sanitize_for_json(v) for k, v in diag.items() if not isinstance(v, (list, dict))})
        row["diagnostics_json"] = json.dumps(sanitize_for_json(diag), sort_keys=True)
        history_rows = []
        for h in result.history:
            history_rows.append(
                {
                    "setting": setting,
                    "seed": int(seed),
                    "variant": variant,
                    **{k: sanitize_for_json(v) for k, v in h.items()},
                }
            )
        history_path = Path(args.out_dir) / f"history_{setting}_seed{seed}_{variant}.csv"
        _write_csv(history_path, history_rows)
        row["history_path"] = str(history_path)
        row["train_sec"] = float(result.train_sec)
        row["pred_sec"] = float(pred["elapsed_sec"])
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED setting={setting} seed={seed} variant={variant}: {row['error']}", flush=True)
    return row


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Global WGP-CEM L2/LH Ablation

This experiment trains a sparse-GP prior over projection matrices `W(x)` with
fixed training neighborhoods and a local JGP-CEM composite likelihood.

Variants:

- `anchor_uniform`: anchor-conditioned `W(c_a)` with fixed uniform outlier likelihood.
- `anchor_floor`: anchor-conditioned plus soft gamma floor.
- `anchor_coverage`: anchor-conditioned plus gamma floor and coverage penalty.
- `anchor_background`: anchor-conditioned plus gamma floor, coverage, and broad normal background `p0(y)`.
- `pointwise_coverage`: pointwise `W(x_i)` plus gamma floor and coverage.
- `pointwise_background`: pointwise plus gamma floor, coverage, and broad normal background.
- `pointwise_pca_map_soft`: pointwise `W(x_i)`, DeepVMGP-style PCA `R_mu` initialization, soft labels, and MAP-R training.
- `pointwise_pairwarm_map_soft`: pointwise `W(x_i)`, pairwise-boundary warm start for `R_mu`, soft labels, and MAP-R training.
- `anchor_pca_background_map_soft`: anchor-conditioned `W(c_a)`, PCA initialization, broad normal background, soft labels, and MAP-R training.

Files:

- `metrics.csv`: one row per setting/seed/variant.
- `aggregate.csv`: mean/std over successful rows.
- `history_<setting>_seed<seed>_<variant>.csv`: optimization diagnostics, including data/KL gradient ratios.
- `summary.json`: exact runner arguments and setting metadata.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sparse-GP W(x) + local JGP-CEM ablations on paper L2/LH data.")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--anchor_count", type=int, default=40)
    p.add_argument("--boundary_anchor_frac", type=float, default=0.5)
    p.add_argument("--random_anchor_frac", type=float, default=0.0)
    p.add_argument("--local_var_neighbors", type=int, default=35)
    p.add_argument("--train_neighbors", type=int, default=None)
    p.add_argument("--n_final", type=int, default=None)
    p.add_argument("--n_candidate", type=int, default=80)
    p.add_argument("--m_inducing", type=int, default=40)
    p.add_argument("--init_mode", choices=["pls", "pca", "pca_scaled"], default="pls")
    p.add_argument("--max_test", type=int, default=80)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--c_refresh", type=int, default=5)
    p.add_argument("--mc_train", type=int, default=1)
    p.add_argument("--eval_mc", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--beta_kl", type=float, default=0.05)
    p.add_argument("--kl_warmup_steps", type=int, default=40)
    p.add_argument("--composite_temperature", type=float, default=0.2)
    p.add_argument("--gamma_init", type=float, default=0.7)
    p.add_argument("--gamma_floor", type=float, default=0.05)
    p.add_argument("--gamma_temperature", type=float, default=3.0)
    p.add_argument("--gamma_damping", type=float, default=0.5)
    p.add_argument("--coverage_weight", type=float, default=1.0)
    p.add_argument("--coverage_min", type=float, default=0.15)
    p.add_argument("--background_scale", type=float, default=3.0)
    p.add_argument("--init_log_std", type=float, default=-3.0)
    p.add_argument("--local_lengthscale", type=float, default=1.0)
    p.add_argument("--w_lengthscale", type=float, default=None)
    p.add_argument("--init_amp", type=float, default=1.0)
    p.add_argument("--init_noise_var", type=float, default=0.1)
    p.add_argument("--train_kernel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--train_noise", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--train_R_log_std", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sample_train_R", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train_w_lengthscale", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--train_w_var", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--outlier_sigma_mult", type=float, default=2.5)
    p.add_argument("--gate_l2", type=float, default=1e-4)
    p.add_argument("--pairwarm_steps", type=int, default=120)
    p.add_argument("--pairwarm_lr", type=float, default=0.01)
    p.add_argument("--pairwarm_hidden", type=int, default=64)
    p.add_argument("--pair_margin", type=float, default=1.0)
    p.add_argument("--pairwarm_weight_decay", type=float, default=1e-4)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out_dir", default="experiments/synthetic/global_wgp_cem_l2_lh_20260514")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    variants = [s.strip() for s in args.variants.split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if unknown_settings:
        raise ValueError(f"Unknown settings: {unknown_settings}")
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(out_dir / "metrics.csv") if args.resume else []
    done = {(r.get("setting"), int(r.get("seed", -1)), r.get("variant")) for r in rows if r.get("status") == "ok"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
        for setting in settings:
            for variant in variants:
                key = (setting, seed, variant)
                if key in done:
                    print(f"skip completed setting={setting} seed={seed} variant={variant}", flush=True)
                    continue
                print(f"run setting={setting} seed={seed} variant={variant}", flush=True)
                row = _run_one(args, setting, seed, variant, device)
                rows.append(row)
                _write_csv(out_dir / "metrics.csv", rows)
                _write_csv(out_dir / "aggregate.csv", _aggregate(rows))

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
    print(f"Saved results to {out_dir}", flush=True)
    return {"rows": rows, "aggregate": _aggregate(rows), "summary": summary}


if __name__ == "__main__":
    main()

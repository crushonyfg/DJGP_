"""Run seven concrete projection-remedy variants on paper L2/LH synthetic data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_l2_lh_7remedies.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --methods p1_exact_vem,p2_soft_vem,p3_norm_gate_vem,p4_fixed_nuisance_vem,p5_whitened_kl,p6_pairwise_boundary,p7_global_wx ^
        --max_anchors 60 --steps 60 --out_dir experiments/synthetic/l2_lh_7remedies_20260514

Faster smoke:

    python experiments/synthetic/compare_l2_lh_7remedies.py ^
        --settings l2_q2_train1000 --methods p6_pairwise_boundary,p7_global_wx ^
        --max_anchors 20 --steps 5 --eval_mc 1 ^
        --out_dir experiments/synthetic/l2_lh_7remedies_smoke
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
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics import sanitize_for_json  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import build_V_basis, compute_pls_W0, per_anchor_W_diagnostics, run_projected_jumpgp  # noqa: E402
from djgp.projections.conditional_vem import VemWConfig, train_conditional_vem_w  # noqa: E402
from djgp.projections.inductive import run_projected_jumpgp_ensemble  # noqa: E402
from djgp.variational import qW_from_qV  # noqa: E402
from djgp.variational_tricks import KLWarmupSchedule, compute_objective_with_tricks  # noqa: E402
from experiments.synthetic.compare_lmjgp_qw_controls_paper import _sample_qw_controls  # noqa: E402
from experiments.synthetic.compare_lmjgp_tricks_synth import _build_regions, _make_initial_params, _seed_all  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import SETTING_PRESETS, _generate_paper_data, _set_seed  # noqa: E402
from experiments.synthetic.compare_paper_weak_inner_w import (  # noqa: E402
    _build_neighbor_stacks,
    _default_pair_tau,
    _evaluate_projection,
    _pairwise_metric_loss_per_anchor,
    _projected_neighbor_lists,
    _response_variance_weights,
    _train_projection,
)
from shared.utils1 import jumpgp_ld_wrapper  # noqa: E402


METHODS = (
    "p1_exact_vem",
    "p2_soft_vem",
    "p3_norm_gate_vem",
    "p4_fixed_nuisance_vem",
    "p5_whitened_kl",
    "p6_pairwise_boundary",
    "p7_global_wx",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting",
        "family",
        "seed",
        "method",
        "point",
        "status",
        "N_train",
        "N_test",
        "n_anchors_used",
        "latent_dim",
        "observed_dim",
        "Q_true",
        "Q",
        "n_final",
        "n_candidate",
        "standardize_x",
        "standardize_y",
        *RESULT_ROW_KEYS,
        "train_sec",
        "pred_sec",
        "gamma_mean",
        "gamma_entropy",
        "mu_data_over_scaled_KL_final",
        "sigma2proxy_data_over_scaled_KL_final",
        "median_frob_W",
        "anchor_dispersion_frob",
        "cosine_to_ref_median",
        "train_loss",
        "error",
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
            groups.setdefault((str(row["setting"]), str(row["method"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (setting, method), group in sorted(groups.items()):
        agg: dict[str, Any] = {
            "setting": setting,
            "method": method,
            "n_runs": len(group),
        }
        for meta in (
            "family",
            "point",
            "N_train",
            "N_test",
            "n_anchors_used",
            "latent_dim",
            "observed_dim",
            "Q_true",
            "Q",
            "n_final",
            "n_candidate",
            "standardize_x",
            "standardize_y",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in list(RESULT_ROW_KEYS) + [
            "train_sec",
            "pred_sec",
            "gamma_mean",
            "gamma_entropy",
            "mu_data_over_scaled_KL_final",
            "sigma2proxy_data_over_scaled_KL_final",
            "median_frob_W",
            "anchor_dispersion_frob",
            "cosine_to_ref_median",
            "train_loss",
        ]:
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


def _base_row(args: argparse.Namespace, cfg: dict[str, Any], setting: str, seed: int, method: str, T: int) -> dict[str, Any]:
    point_map = {
        "p1_exact_vem": "1_exact_local_covariance",
        "p2_soft_vem": "2_weak_decoder_soft_responsibilities",
        "p3_norm_gate_vem": "3_scale_rotation_constraints",
        "p4_fixed_nuisance_vem": "4_fixed_shared_nuisance",
        "p5_whitened_kl": "5_kl_warmup_whitened_parameterization",
        "p6_pairwise_boundary": "6_direct_boundary_pairwise_loss",
        "p7_global_wx": "7_shared_global_Wx_function",
    }
    return {
        "setting": setting,
        "family": cfg["family"],
        "seed": int(seed),
        "method": method,
        "point": point_map[method],
        "N_train": int(cfg["N_train"]),
        "N_test": int(cfg["N_test"]),
        "n_anchors_used": int(T),
        "latent_dim": int(cfg["d"]),
        "observed_dim": int(cfg["H"]),
        "Q_true": int(cfg["Q_true"]),
        "Q": int(cfg["Q"]),
        "n_final": int(cfg["n"]),
        "n_candidate": int(args.n_candidate),
        "standardize_x": True,
        "standardize_y": False,
        "noise_std": float(cfg["noise_std"]),
        "noise_var": float(cfg["noise_var"]),
        "expansion": str(cfg.get("expansion", "rff")),
    }


def _basis_tensors(X_train_s: np.ndarray, y_train: np.ndarray, Q: int, args: argparse.Namespace, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    W0 = compute_pls_W0(X_train_s, y_train, Q=Q)
    V, info = build_V_basis(
        X_train_s,
        y_train,
        n_pls=min(Q, int(args.n_pls)),
        n_pca=int(args.n_pca),
        n_random=int(args.n_random),
        seed=seed,
    )
    return (
        torch.from_numpy(W0.astype(np.float32)).to(device),
        torch.from_numpy(V.astype(np.float32)).to(device),
        info,
    )


def _vem_cfg(method: str, args: argparse.Namespace) -> VemWConfig:
    if method == "p1_exact_vem":
        return VemWConfig(
            posterior=True,
            n_steps=int(args.steps),
            vem_steps=1,
            mc_train=int(args.mc_train),
            mc_gamma=int(args.mc_gamma),
            lr=float(args.lr),
            coeff_scale=float(args.coeff_scale),
            init_log_std=float(args.init_log_std),
            kl_weight=float(args.kl_weight),
            kl_warmup_steps=int(args.kl_warmup_steps),
            lambda_smooth=float(args.lambda_smooth),
            lambda_coeff=float(args.lambda_coeff),
            lambda_gate_l2=1e-4,
            gamma_damping=0.7,
            temperature=1.5,
            temperature_final=1.0,
            boundary_weighted=False,
            train_kernel=True,
            train_noise=True,
            init_lengthscale=float(args.init_lengthscale),
            init_amp=float(args.init_amp),
            init_noise_var=float(args.init_noise_var),
            log_interval=int(args.log_interval),
            verbose=bool(args.verbose),
        )
    if method == "p2_soft_vem":
        return VemWConfig(
            posterior=True,
            n_steps=int(args.steps),
            vem_steps=1,
            mc_train=int(args.mc_train),
            mc_gamma=int(args.mc_gamma),
            lr=float(args.lr),
            coeff_scale=float(args.coeff_scale),
            init_log_std=float(args.init_log_std),
            kl_weight=float(args.kl_weight),
            kl_warmup_steps=int(args.kl_warmup_steps),
            lambda_smooth=float(args.lambda_smooth),
            lambda_coeff=float(args.lambda_coeff),
            lambda_gate_l2=1e-4,
            gamma_damping=0.3,
            temperature=5.0,
            temperature_final=3.0,
            boundary_weighted=False,
            train_kernel=False,
            train_noise=False,
            init_lengthscale=float(args.init_lengthscale),
            init_amp=float(args.init_amp),
            init_noise_var=float(args.init_noise_var),
            log_interval=int(args.log_interval),
            verbose=bool(args.verbose),
        )
    if method == "p3_norm_gate_vem":
        cfg = _vem_cfg("p2_soft_vem", args)
        cfg.lambda_gate_l2 = 1e-2
        cfg.coeff_scale = max(float(args.coeff_scale), 1.0)
        return cfg
    if method == "p4_fixed_nuisance_vem":
        cfg = _vem_cfg("p1_exact_vem", args)
        cfg.train_kernel = False
        cfg.train_noise = False
        cfg.boundary_weighted = False
        return cfg
    raise ValueError(method)


def _evaluate_vem_method(
    *,
    method: str,
    args: argparse.Namespace,
    X_test_t: torch.Tensor,
    X_vem: torch.Tensor,
    y_vem_std: torch.Tensor,
    X_candidate: torch.Tensor,
    y_candidate: torch.Tensor,
    y_test: np.ndarray,
    W0_t: torch.Tensor,
    V_t: torch.Tensor,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    cfg = _vem_cfg(method, args)
    res = train_conditional_vem_w(
        X_neighbors=X_vem,
        y_neighbors=y_vem_std,
        X_anchors=X_test_t,
        V=V_t,
        W0_init=W0_t,
        cfg=cfg,
    )
    X_final, y_final = _projected_neighbor_lists(
        res.W_mean,
        X_test_t,
        X_candidate,
        y_candidate,
        n_neighbors=int(args.n_final),
    )
    if bool(args.vem_posterior_head):
        W_samples = res.model.sample_W(int(args.eval_mc), mean_only=False).detach()
        out = run_projected_jumpgp_ensemble(
            W_samples,
            X_test_t,
            X_final,
            y_final,
            device=device,
            progress=False,
            desc=method,
        )
    else:
        out = run_projected_jumpgp(
            res.W_mean,
            X_test_t,
            X_final,
            y_final,
            device=device,
            progress=False,
            desc=method,
        )
    pack = _metrics_pack(y_test, out["mu"], out["sigma"], float(res.train_time_sec + out["elapsed_sec"]))
    diag = dict(res.diagnostics)
    diag.update(
        {
            "train_sec": float(res.train_time_sec),
            "pred_sec": float(out["elapsed_sec"]),
            "train_loss": float(res.history[-1]["loss"]) if res.history else float("nan"),
            "history_tail": res.history[-5:],
            "vem_train_kernel": bool(cfg.train_kernel),
            "vem_train_noise": bool(cfg.train_noise),
            "vem_temperature": float(cfg.temperature),
            "vem_temperature_final": float(cfg.temperature_final),
            "vem_gamma_damping": float(cfg.gamma_damping),
            "vem_lambda_gate_l2": float(cfg.lambda_gate_l2),
        }
    )
    return pack, diag


def _kernel_cholesky_by_q(hyperparams: dict[str, torch.Tensor], Q: int) -> list[torch.Tensor]:
    Z = hyperparams["Z"]
    d2 = (Z.unsqueeze(1) - Z.unsqueeze(0)).pow(2).sum(-1)
    eye = torch.eye(Z.shape[0], device=Z.device, dtype=Z.dtype)
    out: list[torch.Tensor] = []
    var_w = hyperparams["var_w"]
    for q in range(Q):
        ell = hyperparams["lengthscales"][q]
        K = var_w * torch.exp(-0.5 * d2 / ell.pow(2).clamp_min(1e-12))
        out.append(torch.linalg.cholesky(0.5 * (K + K.T) + 1e-5 * eye))
    return out


def _mu_from_whitened(eta: torch.Tensor, hyperparams: dict[str, torch.Tensor]) -> torch.Tensor:
    m2, Q, D = eta.shape
    Ls = _kernel_cholesky_by_q(hyperparams, Q)
    mu = torch.empty_like(eta)
    for q, L in enumerate(Ls):
        mu[:, q, :] = L @ eta[:, q, :]
    return mu


def _initial_eta_from_mu(mu_V: torch.Tensor, hyperparams: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        m2, Q, D = mu_V.shape
        Ls = _kernel_cholesky_by_q(hyperparams, Q)
        eta = torch.empty_like(mu_V)
        for q, L in enumerate(Ls):
            eta[:, q, :] = torch.linalg.solve_triangular(L, mu_V[:, q, :], upper=False)
    return eta


def _train_whitened_lmjgp(
    *,
    args: argparse.Namespace,
    X_train_t: torch.Tensor,
    y_train_std_t: torch.Tensor,
    X_test_t: torch.Tensor,
    Q: int,
    m1: int,
    m2: int,
    n: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor], list[dict[str, torch.Tensor]], dict[str, torch.Tensor], dict[str, Any]]:
    T, D = X_test_t.shape
    _seed_all(seed)
    regions, X_nb_list, y_nb_list = _build_regions(X_test_t, X_train_t, y_train_std_t, device, m1, Q, n)
    V_params, u_params, hyperparams, init_info = _make_initial_params(
        X_train_t,
        y_train_std_t,
        X_test_t,
        Q=Q,
        m1=m1,
        m2=m2,
        T=T,
        D=D,
        device=device,
        mu_V_mode="pls",
        init_seed=seed,
    )
    eta = _initial_eta_from_mu(V_params["mu_V"], hyperparams).detach().clone().requires_grad_(True)
    sigma_V = V_params["sigma_V"]
    warm = KLWarmupSchedule(beta_init=0.01, beta_final=1.0, warmup_steps=max(1, int(args.steps * 0.5)))
    params = [eta, sigma_V]
    for u in u_params:
        params += [u["U_logit"], u["mu_u"], u["Sigma_u"], u["sigma_noise"], u["omega"], u["sigma_k"]]
    params += [hyperparams["lengthscales"], hyperparams["var_w"], hyperparams["Z"]]
    opt = torch.optim.Adam(params, lr=float(args.lr))
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        beta = float(warm(step))
        opt.zero_grad()
        V_view = {"mu_V": _mu_from_whitened(eta, hyperparams), "sigma_V": sigma_V}
        loss, breakdown = compute_objective_with_tricks(
            regions,
            V_view,
            u_params,
            hyperparams,
            beta_W=beta,
            row_norm_weight=0.0,
            aux_metric_weight=0.0,
        )
        loss.backward()
        opt.step()
        with torch.no_grad():
            sigma_V.clamp_(min=1e-1)
            hyperparams["var_w"].clamp_(min=1e-1)
            hyperparams["lengthscales"].clamp_(min=1e-6)
            for u in u_params:
                u["sigma_noise"].clamp_(min=1e-1)
                u["sigma_k"].clamp_(min=1e-1)
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            history.append({"step": float(step), "loss": float(loss.detach().item()), **breakdown})
    train_sec = time.perf_counter() - t0
    V_final = {
        "mu_V": _mu_from_whitened(eta, hyperparams).detach().clone().requires_grad_(True),
        "sigma_V": sigma_V.detach().clone().requires_grad_(True),
    }
    diag = {
        "train_sec": float(train_sec),
        "history_tail": history[-5:],
        "train_loss": history[-1]["loss"] if history else float("nan"),
        "init_info": init_info,
        "whitened_mean_parameterization": True,
    }
    return regions, V_final, u_params, hyperparams, {**diag, "X_nb_list": X_nb_list, "y_nb_list": y_nb_list}


def _evaluate_whitened_kl(
    *,
    args: argparse.Namespace,
    X_train_t: torch.Tensor,
    y_train_std_t: torch.Tensor,
    X_test_t: torch.Tensor,
    y_test_std: np.ndarray,
    y_mean: float,
    y_std: float,
    Q: int,
    n: int,
    seed: int,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    regions, V_params, _u_params, hyperparams, diag = _train_whitened_lmjgp(
        args=args,
        X_train_t=X_train_t,
        y_train_std_t=y_train_std_t,
        X_test_t=X_test_t,
        Q=Q,
        m1=int(args.m1),
        m2=int(args.m2),
        n=n,
        seed=seed,
        device=device,
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
        controls = _sample_qw_controls(
            mu_W=mu_W,
            cov_W=cov_W,
            var_w=hyperparams["var_w"],
            M=int(args.eval_mc),
            seed=seed + 5001,
        )
    out = run_projected_jumpgp_ensemble(
        controls["learned_qw"],
        X_test_t,
        diag["X_nb_list"],
        diag["y_nb_list"],
        device=device,
        progress=False,
        desc="p5_whitened_kl",
    )
    mu = np.asarray(out["mu"], dtype=np.float64) * y_std + y_mean
    sigma = np.asarray(out["sigma"], dtype=np.float64) * y_std
    y = np.asarray(y_test_std, dtype=np.float64) * y_std + y_mean
    pack = _metrics_pack(y, mu, sigma, float(diag["train_sec"] + out["elapsed_sec"]))
    diag.update(
        {
            "pred_sec": float(out["elapsed_sec"]),
            "mu_W_frob_median": float(torch.linalg.norm(mu_W, dim=(-2, -1)).median().item()),
            "sigma_W_mean": float(torch.sqrt(torch.diagonal(cov_W, dim1=-2, dim2=-1).clamp_min(1e-12)).mean().item()),
        }
    )
    diag.pop("X_nb_list", None)
    diag.pop("y_nb_list", None)
    return pack, diag


class PointWxNet(nn.Module):
    def __init__(self, D: int, Q: int, hidden: int):
        super().__init__()
        self.D = int(D)
        self.Q = int(Q)
        self.net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, Q * D),
        )

    def W(self, X: torch.Tensor) -> torch.Tensor:
        raw = self.net(X).view(*X.shape[:-1], self.Q, self.D)
        return raw / torch.linalg.norm(raw, dim=-1, keepdim=True).clamp_min(1e-8)

    def z(self, X: torch.Tensor) -> torch.Tensor:
        W = self.W(X)
        return torch.einsum("...qd,...d->...q", W, X)


def _run_jumpgp_z(
    Z_anchor: torch.Tensor,
    Z_neighbors_list: list[torch.Tensor],
    y_neighbors_list: list[torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    mus = np.empty(len(Z_neighbors_list), dtype=np.float64)
    sigmas = np.empty(len(Z_neighbors_list), dtype=np.float64)
    t0 = time.perf_counter()
    for t, (Z_nb, y_nb) in enumerate(zip(Z_neighbors_list, y_neighbors_list)):
        mu_t, sig2_t, _, _ = jumpgp_ld_wrapper(
            Z_nb,
            y_nb.view(-1, 1),
            Z_anchor[t].view(1, -1),
            mode="CEM",
            flag=False,
            device=device,
        )
        mus[t] = float(mu_t.detach().cpu().reshape(-1)[0].item())
        sigmas[t] = float(torch.sqrt(torch.clamp(sig2_t, min=1e-12)).detach().cpu().reshape(-1)[0].item())
    return {"mu": mus, "sigma": sigmas, "elapsed_sec": float(time.perf_counter() - t0)}


def _evaluate_global_wx(
    *,
    args: argparse.Namespace,
    X_test_t: torch.Tensor,
    X_pair: torch.Tensor,
    y_pair_std: torch.Tensor,
    X_candidate: torch.Tensor,
    y_candidate: torch.Tensor,
    y_test: np.ndarray,
    Q: int,
    D: int,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    net = PointWxNet(D, Q, int(args.wx_hidden)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(args.lr), weight_decay=float(args.wx_weight_decay))
    weights = _response_variance_weights(y_pair_std)
    pair_tau = _default_pair_tau(y_pair_std)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    t0 = time.perf_counter()
    last_loss = float("nan")
    for step in range(1, int(args.steps) + 1):
        opt.zero_grad()
        Z = net.z(X_pair)
        per_anchor = _pairwise_metric_loss_per_anchor(
            Z,
            y_pair_std,
            tau_y=pair_tau,
            margin=float(args.pair_margin),
        )
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
    train_sec = time.perf_counter() - t0

    with torch.no_grad():
        Z_anchor = net.z(X_test_t)
        Z_candidate = net.z(X_candidate)
        d2 = (Z_candidate - Z_anchor.unsqueeze(1)).pow(2).sum(dim=-1)
        idx = torch.topk(d2, k=int(args.n_final), largest=False, dim=1).indices
        Z_lists = [Z_candidate[t, idx[t]].contiguous() for t in range(X_candidate.shape[0])]
        y_lists = [y_candidate[t, idx[t]].contiguous() for t in range(X_candidate.shape[0])]
        W_anchor = net.W(X_test_t).detach()
    out = _run_jumpgp_z(Z_anchor, Z_lists, y_lists, device=device)
    pack = _metrics_pack(y_test, out["mu"], out["sigma"], float(train_sec + out["elapsed_sec"]))
    diag = per_anchor_W_diagnostics(W_anchor)
    diag.update(
        {
            "train_sec": float(train_sec),
            "pred_sec": float(out["elapsed_sec"]),
            "train_loss": float(last_loss),
            "pair_tau": float(pair_tau),
            "global_wx_pointwise_z": True,
            "median_frob_W": float(torch.linalg.norm(W_anchor, dim=(-2, -1)).median().item()),
        }
    )
    return pack, diag


def _run_one(args: argparse.Namespace, setting: str, seed: int, method: str, device: torch.device) -> dict[str, Any]:
    cfg = SETTING_PRESETS[setting]
    _set_seed(seed)
    data = _generate_paper_data(cfg, seed, device)
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(data["X_train"]).astype(np.float32)
    X_test_s = scaler_x.transform(data["X_test"]).astype(np.float32)
    if args.max_anchors is not None:
        X_test_s = X_test_s[: int(args.max_anchors)]
        y_test = np.asarray(data["y_test"], dtype=np.float64)[: int(args.max_anchors)]
    else:
        y_test = np.asarray(data["y_test"], dtype=np.float64)
    y_train = np.asarray(data["y_train"], dtype=np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() if y_train.std() > 1e-8 else 1.0)
    y_train_std = ((y_train - y_mean) / y_std).astype(np.float32)
    y_test_std = ((y_test.astype(np.float32) - y_mean) / y_std).astype(np.float32)

    X_train_t = torch.from_numpy(X_train_s).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    y_train_std_t = torch.from_numpy(y_train_std).to(device)
    X_test_t = torch.from_numpy(X_test_s).to(device)
    T, D = X_test_t.shape
    Q = int(cfg["Q"])
    n_final = int(args.n_final or cfg["n"])
    args.n_final = n_final

    row = _base_row(args, cfg, setting, seed, method, T)
    W0_t, V_t, basis_info = _basis_tensors(X_train_s, y_train_std, Q, args, seed, device)

    X_candidate, y_candidate = _build_neighbor_stacks(
        X_test_t,
        X_train_t,
        y_train_t,
        n_neighbors=max(int(args.n_candidate), n_final),
        device=device,
    )
    X_pair, y_pair_std = _build_neighbor_stacks(
        X_test_t,
        X_train_t,
        y_train_std_t,
        n_neighbors=max(int(args.pair_neighbors), n_final),
        device=device,
    )
    X_vem, y_vem_std = _build_neighbor_stacks(
        X_test_t,
        X_train_t,
        y_train_std_t,
        n_neighbors=max(int(args.n_vem), n_final),
        device=device,
    )

    try:
        if method in {"p1_exact_vem", "p2_soft_vem", "p3_norm_gate_vem", "p4_fixed_nuisance_vem"}:
            pack, diag = _evaluate_vem_method(
                method=method,
                args=args,
                X_test_t=X_test_t,
                X_vem=X_vem,
                y_vem_std=y_vem_std,
                X_candidate=X_candidate,
                y_candidate=y_candidate,
                y_test=y_test,
                W0_t=W0_t,
                V_t=V_t,
                device=device,
            )
        elif method == "p5_whitened_kl":
            pack, diag = _evaluate_whitened_kl(
                args=args,
                X_train_t=X_train_t,
                y_train_std_t=y_train_std_t,
                X_test_t=X_test_t,
                y_test_std=y_test_std,
                y_mean=y_mean,
                y_std=y_std,
                Q=Q,
                n=n_final,
                seed=seed * 1009 + 515,
                device=device,
            )
        elif method == "p6_pairwise_boundary":
            weak_args = argparse.Namespace(**vars(args))
            weak_args.lambda_pair = float(args.lambda_pair)
            weak_args.lambda_gp = 0.0
            weak_args.lambda_smooth = float(args.lambda_smooth)
            weak_args.lambda_coeff = float(args.lambda_coeff)
            weak_args.init_C_scale = 0.0
            weak_args.pair_tau = 0.0
            weak_args.verbose = bool(args.verbose)
            W, diag = _train_projection(
                "pairwise_boundary_localC",
                X_anchor=X_test_t,
                X_gp=X_vem,
                y_gp_std=y_vem_std,
                X_pair=X_pair,
                y_pair_std=y_pair_std,
                V_t=V_t,
                W0_t=W0_t,
                args=weak_args,
                device=device,
            )
            pack, pred_diag = _evaluate_projection(
                W,
                method=method,
                final_mode="projected",
                X_anchor=X_test_t,
                X_candidate=X_candidate,
                y_candidate=y_candidate,
                y_test=y_test,
                n_final=n_final,
                train_sec=float(diag.get("train_sec", 0.0)),
                device=device,
            )
            diag.update(pred_diag)
        elif method == "p7_global_wx":
            pack, diag = _evaluate_global_wx(
                args=args,
                X_test_t=X_test_t,
                X_pair=X_pair,
                y_pair_std=y_pair_std,
                X_candidate=X_candidate,
                y_candidate=y_candidate,
                y_test=y_test,
                Q=Q,
                D=D,
                device=device,
            )
        else:
            raise ValueError(method)

        row["status"] = "ok"
        row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
        row.update({k: sanitize_for_json(v) for k, v in diag.items() if not isinstance(v, (list, dict))})
        row["diagnostics_json"] = json.dumps(sanitize_for_json({**diag, "basis": basis_info}), sort_keys=True)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED setting={setting} seed={seed} method={method}: {row['error']}", flush=True)
    return row


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# L2/LH Seven Projection Remedies

This run implements the seven proposed projection remedies as concrete methods
under one paper-style L2/LH protocol.

Files:

- `metrics.csv`: one row per setting/seed/method.
- `aggregate.csv`: mean/std over successful rows.
- `summary.json`: exact runner arguments and setting metadata.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seven projection-remedy variants on paper L2/LH synthetic data.")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--max_anchors", type=int, default=60)
    p.add_argument("--n_final", type=int, default=None)
    p.add_argument("--n_candidate", type=int, default=80)
    p.add_argument("--n_vem", type=int, default=50)
    p.add_argument("--pair_neighbors", type=int, default=50)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--mc_train", type=int, default=2)
    p.add_argument("--mc_gamma", type=int, default=2)
    p.add_argument("--eval_mc", type=int, default=2)
    p.add_argument("--vem_posterior_head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--init_log_std", type=float, default=-3.0)
    p.add_argument("--kl_weight", type=float, default=0.05)
    p.add_argument("--kl_warmup_steps", type=int, default=40)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_coeff", type=float, default=1e-4)
    p.add_argument("--lambda_pair", type=float, default=1.0)
    p.add_argument("--pair_margin", type=float, default=1.0)
    p.add_argument("--init_lengthscale", type=float, default=1.0)
    p.add_argument("--init_amp", type=float, default=1.0)
    p.add_argument("--init_noise_var", type=float, default=0.1)
    p.add_argument("--wx_hidden", type=int, default=64)
    p.add_argument("--wx_weight_decay", type=float, default=1e-4)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out_dir", default="experiments/synthetic/l2_lh_7remedies_20260514")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_settings:
        raise ValueError(f"Unknown settings: {unknown_settings}")
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(out_dir / "metrics.csv") if args.resume else []
    done = {(r.get("setting"), int(r.get("seed", -1)), r.get("method")) for r in rows if r.get("status") == "ok"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
        for setting in settings:
            for method in methods:
                key = (setting, seed, method)
                if key in done:
                    print(f"skip completed setting={setting} seed={seed} method={method}", flush=True)
                    continue
                print(f"run setting={setting} seed={seed} method={method}", flush=True)
                row = _run_one(args, setting, seed, method, device)
                rows.append(row)
                _write_csv(out_dir / "metrics.csv", rows)
                _write_csv(out_dir / "aggregate.csv", _aggregate(rows))

    summary = {
        "args": sanitize_for_json(vars(args)),
        "device": str(device),
        "settings": {s: SETTING_PRESETS[s] for s in settings},
        "methods": methods,
        "outputs": {"metrics": "metrics.csv", "aggregate": "aggregate.csv", "readme": "README.md"},
    }
    (out_dir / "summary.json").write_text(json.dumps(sanitize_for_json(summary), indent=2), encoding="utf-8")
    _write_readme(out_dir, sanitize_for_json(summary))
    print(f"Wrote results to {out_dir}", flush=True)
    return summary


if __name__ == "__main__":
    main()

"""Test A/B/C remedies for Original ELBO + direct-VI predict.

How to run from the repository root with conda env ``jumpGP``::

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/test_ab_remedies_direct_vi.py ^
        --setting lh_q5_train500 --seed 0 --max_test 200 ^
        --num_steps 600 --eval_every 150 --lr 0.01 ^
        --variants C0_AB,C1,C2,C3,C4,C5 ^
        --out_dir experiments/synthetic/abc_remedies_direct_vi_lh500_seed0

Gate-collapse diagnostics (each eval): ``pi_mean``, ``r_mean``,
``delta_T_mean`` (= mean of T_in - T_out), ``std_mu_f``, ``std_mu_y``,
``rmse_pi1`` (prediction with forced pi=1 on the inlier branch).

Remedy C variants (all include A+B unless ``--no_ab``):

- ``C0_AB`` / ``baseline``:  A (s-penalty) + B (sigma_V annealing) only
- ``C1``:  + omega_0 init = +2
- ``C2``:  + C1 + clamp omega_0 >= omega0_floor
- ``C3``:  + pi-gate floor prior on training ELBO
- ``C4``:  + force inlier branch only for first ``force_inlier_steps``
- ``C5``:  + clamp U_logit <= u_logit_ceil (weaker outlier branch)
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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.variational import (  # noqa: E402
    expected_Kfu,
    expected_KufKfu,
    expected_log_sigmoid_gh_batch,
    expected_sigmoid_gh,
    kl_qp,
    kl_q_u_batch,
    qW_from_qV,
)
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import (  # noqa: E402
    _predict_direct_variational_consistent,
    _stack_psd_sigma_u,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from shared.jumpgp_runner import find_neighborhoods  # noqa: E402

_LOG_2PI = math.log(2 * math.pi)
_GH_FACTOR = 1.0 / math.sqrt(math.pi)


# -------------------------------------------------------------------------
# Remedy A: s-penalty
# -------------------------------------------------------------------------


def compute_s_penalty(mu_W, cov_W, X: torch.Tensor) -> torch.Tensor:
    s = torch.einsum("tnd,tqde,tne->tnq", X, cov_W, X)
    return s.mean()


# -------------------------------------------------------------------------
# Remedy B: sigma_V schedule
# -------------------------------------------------------------------------


def sigma_v_max_schedule(
    step: int,
    total_steps: int,
    *,
    sigma_start: float = 0.2,
    sigma_end: float = 2.0,
    warmup_frac: float = 0.5,
) -> float:
    warmup_end = int(warmup_frac * total_steps)
    if step <= warmup_end:
        return sigma_start
    frac = (step - warmup_end) / max(total_steps - warmup_end, 1)
    return sigma_start + (sigma_end - sigma_start) * frac


# -------------------------------------------------------------------------
# Augmented ELBO (training) with optional Remedy C hooks
# -------------------------------------------------------------------------


def compute_elbo_augmented(
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    force_inlier_only: bool = False,
    pi_gate_floor: float = 0.0,
    lambda_pi_floor: float = 0.0,
    lambda_inlier_rate: float = 0.0,
) -> torch.Tensor:
    """Original ELBO with optional gate-stabilisation terms.

    ``force_inlier_only``: set T_out very negative so logsumexp uses inlier only.
    ``pi_gate_floor`` + ``lambda_pi_floor``: add ``lambda * mean(log pi_eff)`` with
    ``pi_eff = pi_min + (1-2*pi_min)*pi_gate``.
    ``lambda_inlier_rate``: add ``lambda * mean(log pi_gate)`` (Beta-style encouragement).
    """
    T = len(regions)
    device = regions[0]["X"].device

    X = torch.stack([r["X"] for r in regions], dim=0)
    y = torch.stack([r["y"] for r in regions], dim=0)
    C = torch.stack([r["C"] for r in regions], dim=0)

    U_logit = torch.stack([u["U_logit"] for u in u_params], dim=0).view(-1, 1)
    U = torch.sigmoid(U_logit)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(-1)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)

    eps = 1e-6
    raw = torch.stack([u["Sigma_u"] for u in u_params], dim=0)
    L = torch.tril(raw)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    pos_diag = F.softplus(diag) + eps
    L = L - torch.diag_embed(diag) + torch.diag_embed(pos_diag)
    Sigma_u = L @ L.transpose(-1, -2)

    omega = torch.stack([u["omega"] for u in u_params], dim=0)

    Z = hyperparams["Z"]
    X_test = hyperparams["X_test"]
    lengthscales = hyperparams["lengthscales"]
    var_w = hyperparams["var_w"]
    mu_V = V_params["mu_V"]
    sigma_V = V_params["sigma_V"]

    KL_V = kl_qp(Z, mu_V, sigma_V, lengthscales, var_w)

    mu_W, cov_W = qW_from_qV(X_test, Z, mu_V, sigma_V, lengthscales, var_w)
    Kfu = expected_Kfu(mu_W, cov_W, X, C, sigma_k)
    KufKfu = expected_KufKfu(mu_W, cov_W, X, C, sigma_k)

    KL_u = kl_q_u_batch(C, mu_u, Sigma_u, torch.tensor(1.0, device=device)).sum()

    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    Kuu = torch.exp(-0.5 * d2)
    Luu = torch.linalg.cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(Luu)

    V1 = sigma_k.view(-1, 1) ** 2 - torch.einsum("tij,tnji->tn", Kuu_inv, KufKfu)
    v = torch.einsum("tij,tj->ti", Kuu_inv, mu_u)
    E_fu = torch.einsum("tni,ti->tn", Kfu, v)

    Sigma_u_tilde = Sigma_u + torch.einsum("ti,tj->tij", mu_u, mu_u)
    T3 = torch.einsum("tij,tnjk,tkm,tmi->tn", Kuu_inv, KufKfu, Kuu_inv, Sigma_u_tilde)
    quad = (y.pow(2) - 2 * y * E_fu + (V1 + T3)) / (2 * sigma_noise.view(-1, 1) ** 2)

    elog_sig, elog_one_minus = expected_log_sigmoid_gh_batch(omega, mu_W, cov_W, X)
    T1 = -0.5 * _LOG_2PI - 0.5 * torch.log(sigma_noise.view(-1, 1) ** 2) + elog_sig - quad
    T2 = torch.log(U.clamp_min(1e-12)) + elog_one_minus

    if force_inlier_only:
        T2 = torch.full_like(T2, -80.0)

    region_elbo = torch.logsumexp(torch.stack([T1, T2], dim=0), dim=0).sum(-1)
    data_term = region_elbo.sum()
    elbo = data_term - KL_V - KL_u

    if pi_gate_floor > 0.0 and lambda_pi_floor > 0.0:
        pi_gate = expected_sigmoid_gh(omega, mu_W, cov_W, X)
        pmin = float(pi_gate_floor)
        pi_eff = pmin + (1.0 - 2.0 * pmin) * pi_gate
        elbo = elbo + float(lambda_pi_floor) * torch.log(pi_eff.clamp_min(1e-8)).mean()

    if lambda_inlier_rate > 0.0:
        pi_gate = expected_sigmoid_gh(omega, mu_W, cov_W, X)
        elbo = elbo + float(lambda_inlier_rate) * torch.log(pi_gate.clamp_min(1e-8)).mean()

    return elbo / T


@torch.no_grad()
def mixture_diagnostics(
    regions,
    V_params,
    u_params,
    hyperparams,
) -> dict[str, float]:
    """Per-neighbour mixture terms: pi_gate, r, T_in - T_out (matches training ELBO)."""
    device = hyperparams["Z"].device
    X = torch.stack([r["X"] for r in regions], dim=0)
    y = torch.stack([r["y"] for r in regions], dim=0)
    C = torch.stack([r["C"] for r in regions], dim=0)

    U_logit = torch.stack([u["U_logit"] for u in u_params], dim=0).view(-1, 1)
    U = torch.sigmoid(U_logit)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(-1)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)

    eps = 1e-6
    raw = torch.stack([u["Sigma_u"] for u in u_params], dim=0)
    L = torch.tril(raw)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    pos_diag = F.softplus(diag) + eps
    L = L - torch.diag_embed(diag) + torch.diag_embed(pos_diag)
    Sigma_u = L @ L.transpose(-1, -2)

    omega = torch.stack([u["omega"] for u in u_params], dim=0)
    Z = hyperparams["Z"]
    X_test = hyperparams["X_test"]
    lengthscales = hyperparams["lengthscales"]
    var_w = hyperparams["var_w"]

    mu_W, cov_W = qW_from_qV(
        X_test, Z, V_params["mu_V"], V_params["sigma_V"], lengthscales, var_w,
    )
    Kfu = expected_Kfu(mu_W, cov_W, X, C, sigma_k)
    KufKfu = expected_KufKfu(mu_W, cov_W, X, C, sigma_k)

    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    Kuu = torch.exp(-0.5 * d2)
    Kuu_inv = torch.cholesky_inverse(torch.linalg.cholesky(Kuu))

    V1 = sigma_k.view(-1, 1) ** 2 - torch.einsum("tij,tnji->tn", Kuu_inv, KufKfu)
    v = torch.einsum("tij,tj->ti", Kuu_inv, mu_u)
    E_fu = torch.einsum("tni,ti->tn", Kfu, v)
    Sigma_u_tilde = Sigma_u + torch.einsum("ti,tj->tij", mu_u, mu_u)
    T3 = torch.einsum("tij,tnjk,tkm,tmi->tn", Kuu_inv, KufKfu, Kuu_inv, Sigma_u_tilde)
    quad = (y.pow(2) - 2 * y * E_fu + (V1 + T3)) / (2 * sigma_noise.view(-1, 1) ** 2)

    elog_sig, elog_one_minus = expected_log_sigmoid_gh_batch(omega, mu_W, cov_W, X)
    T_in = -0.5 * _LOG_2PI - 0.5 * torch.log(sigma_noise.view(-1, 1) ** 2) + elog_sig - quad
    T_out = torch.log(U.clamp_min(1e-12)) + elog_one_minus

    log_r_in = T_in - torch.logsumexp(torch.stack([T_in, T_out], dim=0), dim=0)
    r = torch.exp(log_r_in)

    pi_gate = expected_sigmoid_gh(omega, mu_W, cov_W, X)
    # X_test: [T, D] -> [T, 1, D]; pi at each region's test neighbour: [T, 1] -> [T]
    pi_test = expected_sigmoid_gh(
        omega, mu_W, cov_W, X_test.unsqueeze(1),
    ).squeeze(1)

    return {
        "pi_gate_mean": float(pi_gate.mean().item()),
        "pi_gate_test_mean": float(pi_test.mean().item()),
        "r_mean": float(r.mean().item()),
        "r_min": float(r.min().item()),
        "delta_T_mean": float((T_in - T_out).mean().item()),
        "T_in_mean": float(T_in.mean().item()),
        "T_out_mean": float(T_out.mean().item()),
        "omega0_mean": float(omega[:, 0].mean().item()),
        "U_mean": float(U.mean().item()),
    }


# -------------------------------------------------------------------------
# Direct-VI predict + forced pi=1 diagnostic
# -------------------------------------------------------------------------


@torch.no_grad()
def _predict_direct_vi_forced_inlier(
    regions,
    V_params,
    u_params,
    hyperparams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as consistent direct VI but mixture uses pi=1 (inlier-only mean path)."""
    device = hyperparams["Z"].device
    T = len(regions)
    C = torch.stack([r["C"] for r in regions], dim=0)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)
    Sigma_u = _stack_psd_sigma_u(u_params)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(T).clamp_min(1e-6)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(T).clamp_min(1e-6)

    X_test = hyperparams["X_test"]
    mu_W, cov_W = qW_from_qV(
        X_test,
        hyperparams["Z"],
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
    )
    Xn = X_test.unsqueeze(1)
    m_W = expected_Kfu(mu_W, cov_W, Xn, C, sigma_k).squeeze(1)
    S_W = expected_KufKfu(mu_W, cov_W, Xn, C, sigma_k).squeeze(1)

    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    eye = torch.eye(m1, device=device, dtype=C.dtype).unsqueeze(0)
    Kuu = torch.exp(-0.5 * d2) + 1e-6 * eye
    Kinv = torch.cholesky_inverse(
        torch.linalg.cholesky(0.5 * (Kuu + Kuu.transpose(-1, -2)))
    )

    a = torch.einsum("tij,tj->ti", Kinv, mu_u)
    mu_f = (m_W * a).sum(dim=1)
    V1 = sigma_k.pow(2) - torch.einsum("tij,tij->t", Kinv, S_W)
    T3 = torch.einsum("tij,tjk,tkm,tmi->t", Kinv, S_W, Kinv, Sigma_u)
    centered_S = S_W - m_W.unsqueeze(2) * m_W.unsqueeze(1)
    V_W = torch.einsum("ti,tij,tj->t", a, centered_S, a)
    var_f = (V1 + T3 + V_W).clamp_min(1e-12)

    mu_y = mu_f
    var_y = (var_f + sigma_noise.pow(2)).clamp_min(1e-12)
    return mu_y, var_y, mu_f


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------


def train_vi_augmented(
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    lr: float,
    num_steps: int,
    eval_every: int,
    y_test: np.ndarray,
    variant_name: str,
    # A+B
    lambda_s: float = 0.0,
    anneal_sigma_v: bool = False,
    sigma_v_start: float = 0.2,
    sigma_v_end: float = 2.0,
    warmup_frac: float = 0.5,
    # C
    omega0_floor: float | None = None,
    u_logit_ceil: float | None = None,
    pi_gate_floor: float = 0.0,
    lambda_pi_floor: float = 0.0,
    lambda_inlier_rate: float = 0.0,
    force_inlier_steps: int = 0,
) -> list[dict[str, Any]]:
    params = [V_params["mu_V"], V_params["sigma_V"]]
    for u in u_params:
        params += [
            u["U_logit"], u["mu_u"], u["Sigma_u"],
            u["sigma_noise"], u["sigma_k"], u["omega"],
        ]
    params += [hyperparams["lengthscales"], hyperparams["var_w"], hyperparams["Z"]]
    optimizer = torch.optim.Adam(params, lr=lr)

    eval_records: list[dict[str, Any]] = []

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        force_inlier = int(force_inlier_steps) > 0 and step <= int(force_inlier_steps)
        elbo = compute_elbo_augmented(
            regions, V_params, u_params, hyperparams,
            force_inlier_only=force_inlier,
            pi_gate_floor=float(pi_gate_floor),
            lambda_pi_floor=float(lambda_pi_floor),
            lambda_inlier_rate=float(lambda_inlier_rate),
        )

        s_penalty_val = 0.0
        if lambda_s > 0:
            mu_W, cov_W = qW_from_qV(
                hyperparams["X_test"], hyperparams["Z"],
                V_params["mu_V"], V_params["sigma_V"],
                hyperparams["lengthscales"], hyperparams["var_w"],
            )
            X_stacked = torch.stack([r["X"] for r in regions], dim=0)
            s_mean = compute_s_penalty(mu_W, cov_W, X_stacked)
            s_penalty_val = float(lambda_s * s_mean.item())
            elbo = elbo - lambda_s * s_mean

        (-elbo).backward()
        optimizer.step()

        with torch.no_grad():
            if anneal_sigma_v:
                sv_max = sigma_v_max_schedule(
                    step, num_steps, sigma_start=sigma_v_start,
                    sigma_end=sigma_v_end, warmup_frac=warmup_frac,
                )
                V_params["sigma_V"].clamp_(min=0.01, max=sv_max)
            else:
                V_params["sigma_V"].clamp_(min=0.1)

            hyperparams["var_w"].clamp_(min=0.1)
            hyperparams["lengthscales"].clamp_(min=1e-6, max=1e2)
            for u in u_params:
                u["sigma_noise"].clamp_(min=0.1)
                u["sigma_k"].clamp_(min=0.1)
                if omega0_floor is not None:
                    u["omega"][0].clamp_(min=float(omega0_floor))
                if u_logit_ceil is not None:
                    u["U_logit"].clamp_(max=float(u_logit_ceil))

        if step % max(1, eval_every // 3) == 0 or step == 1:
            tag = " [inlier-warmup]" if force_inlier else ""
            sp = f", s_pen={s_penalty_val:.4f}" if lambda_s > 0 else ""
            print(
                f"  [{variant_name} step {step}/{num_steps}] ELBO={elbo.item():.4f}{sp}{tag}",
                flush=True,
            )

        if step % eval_every == 0 or step == num_steps:
            rec = _evaluate_direct_vi(
                regions, V_params, u_params, hyperparams, y_test,
                step=step, variant=variant_name,
            )
            eval_records.append(rec)

    return eval_records


@torch.no_grad()
def _evaluate_direct_vi(
    regions,
    V_params,
    u_params,
    hyperparams,
    y_test: np.ndarray,
    *,
    step: int,
    variant: str,
) -> dict[str, Any]:
    device = hyperparams["Z"].device
    T = len(regions)

    mu_y, var_y = _predict_direct_variational_consistent(
        regions, V_params, u_params, hyperparams,
    )
    mu_y = torch.nan_to_num(mu_y, nan=0.0)
    var_y = torch.nan_to_num(var_y, nan=1e-12).clamp_min(1e-12)
    sigma_y = torch.sqrt(var_y)

    mu_pi1, var_pi1, mu_f = _predict_direct_vi_forced_inlier(
        regions, V_params, u_params, hyperparams,
    )
    mu_pi1 = torch.nan_to_num(mu_pi1, nan=0.0)
    var_pi1 = torch.nan_to_num(var_pi1, nan=1e-12).clamp_min(1e-12)

    y_t = torch.from_numpy(y_test.astype(np.float32)).to(device)[:T]
    rmse = float(torch.sqrt(((mu_y - y_t) ** 2).mean()).item())
    crps = float(mean_gaussian_crps_torch(y_t, mu_y, sigma_y).item())
    rmse_pi1 = float(torch.sqrt(((mu_pi1 - y_t) ** 2).mean()).item())

    mix = mixture_diagnostics(regions, V_params, u_params, hyperparams)

    mu_W, cov_W = qW_from_qV(
        hyperparams["X_test"], hyperparams["Z"],
        V_params["mu_V"], V_params["sigma_V"],
        hyperparams["lengthscales"], hyperparams["var_w"],
    )
    sigma_W = torch.sqrt(cov_W.diagonal(dim1=2, dim2=3))
    X_stacked = torch.stack([r["X"] for r in regions], dim=0)
    s = torch.einsum("tnd,tqde,tne->tnq", X_stacked, cov_W, X_stacked)

    rec: dict[str, Any] = {
        "variant": variant,
        "step": int(step),
        "rmse": rmse,
        "crps": crps,
        "rmse_pi1": rmse_pi1,
        "std_mu_f": float(mu_f.std().item()),
        "std_mu_y": float(mu_y.std().item()),
        "sigma_w_mean": float(sigma_W.mean().item()),
        "s_mean": float(s.mean().item()),
        "s_max": float(s.max().item()),
        "sigma_v_mean": float(V_params["sigma_V"].mean().item()),
        "lengthscale_mean": float(hyperparams["lengthscales"].mean().item()),
    }
    rec.update(mix)

    print(
        f"    >>> [{variant} step={step}] RMSE={rmse:.4f}  CRPS={crps:.4f}  "
        f"RMSE(pi=1)={rmse_pi1:.4f}  "
        f"pi_test={rec['pi_gate_test_mean']:.4f}  r={rec['r_mean']:.4f}  "
        f"dT={rec['delta_T_mean']:.2f}  std(mu_f)={rec['std_mu_f']:.2f}  "
        f"s={rec['s_mean']:.2f}",
        flush=True,
    )
    return rec


# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------


def _build_state(
    X_train_t,
    y_train_t,
    X_test_t,
    *,
    Q: int,
    n: int,
    m1: int,
    m2: int,
    device: torch.device,
    seed: int,
    omega0_init: float | None = None,
):
    _set_seed(seed)
    neighborhoods = find_neighborhoods(
        X_test_t.cpu(), X_train_t.cpu(), y_train_t.cpu(), M=n,
    )
    T = X_test_t.shape[0]
    D = X_train_t.shape[1]

    regions = []
    for item in neighborhoods:
        regions.append({
            "X": item["X_neighbors"].to(device),
            "y": item["y_neighbors"].to(device),
            "C": torch.randn(m1, Q, device=device),
        })

    X_mean = X_train_t.mean(dim=0)
    X_std = X_train_t.std(dim=0).clamp_min(1e-6)
    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
    }
    hyperparams = {
        "Z": (X_mean + torch.randn(m2, D, device=device) * X_std).requires_grad_(True),
        "X_test": X_test_t,
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    u_params = []
    for _ in range(T):
        omega = torch.randn(Q + 1, device=device)
        if omega0_init is not None:
            omega[0] = float(omega0_init)
        u_params.append({
            "U_logit": torch.zeros(1, device=device, requires_grad=True),
            "mu_u": torch.randn(m1, device=device, requires_grad=True),
            "Sigma_u": torch.eye(m1, device=device, requires_grad=True),
            "sigma_noise": torch.tensor(0.5, device=device, requires_grad=True),
            "sigma_k": torch.tensor(0.5, device=device, requires_grad=True),
            "omega": omega.requires_grad_(True),
        })

    return regions, V_params, u_params, hyperparams


def _deep_clone_state(V_params, u_params, hyperparams):
    V2 = {
        "mu_V": V_params["mu_V"].detach().clone().requires_grad_(True),
        "sigma_V": V_params["sigma_V"].detach().clone().requires_grad_(True),
    }
    u2 = [{k: u[k].detach().clone().requires_grad_(True) for k in u} for u in u_params]
    h2 = {
        "Z": hyperparams["Z"].detach().clone().requires_grad_(True),
        "lengthscales": hyperparams["lengthscales"].detach().clone().requires_grad_(True),
        "var_w": hyperparams["var_w"].detach().clone().requires_grad_(True),
        "X_test": hyperparams["X_test"],
    }
    return V2, u2, h2


def _clone_regions(regions):
    return [{"X": r["X"], "y": r["y"], "C": r["C"].detach().clone()} for r in regions]


# -------------------------------------------------------------------------
# Variant registry
# -------------------------------------------------------------------------


def _variant_specs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Build variant configs; all C* include A+B unless ``--no_ab``."""
    use_ab = not args.no_ab
    base = {
        "lambda_s": args.lambda_s if use_ab else 0.0,
        "anneal_sigma_v": use_ab,
        "sigma_v_start": args.sigma_v_start,
        "sigma_v_end": args.sigma_v_end,
        "warmup_frac": args.warmup_frac,
        "omega0_init": None,
        "omega0_floor": None,
        "u_logit_ceil": None,
        "pi_gate_floor": 0.0,
        "lambda_pi_floor": 0.0,
        "lambda_inlier_rate": 0.0,
        "force_inlier_steps": 0,
    }

    specs: dict[str, dict[str, Any]] = {
        "C0_AB": dict(base),
        "baseline": dict(base),
        "C1": {**base, "omega0_init": args.omega0_init},
        "C2": {
            **base,
            "omega0_init": args.omega0_init,
            "omega0_floor": args.omega0_floor,
        },
        "C3": {
            **base,
            "pi_gate_floor": args.pi_gate_floor,
            "lambda_pi_floor": args.lambda_pi_floor,
        },
        "C4": {**base, "force_inlier_steps": args.force_inlier_steps},
        "C5": {**base, "u_logit_ceil": args.u_logit_ceil},
    }
    if use_ab:
        specs["A_only"] = {
            **base,
            "anneal_sigma_v": False,
        }
        specs["B_only"] = {
            **base,
            "lambda_s": 0.0,
        }
        specs["A+B"] = dict(base)
    return specs


def _plot_final_metrics(records: list[dict[str, Any]], out_path: Path) -> None:
    variants = sorted(set(r["variant"] for r in records))
    final_step = max(r["step"] for r in records)
    rmse, crps, r_mean, pi_test = [], [], [], []
    for v in variants:
        rec = next(
            (r for r in records if r["variant"] == v and r["step"] == final_step), None,
        )
        if rec is None:
            continue
        rmse.append(float(rec["rmse"]))
        crps.append(float(rec["crps"]))
        r_mean.append(float(rec.get("r_mean", float("nan"))))
        pi_test.append(float(rec.get("pi_gate_test_mean", float("nan"))))

    if not rmse:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.arange(len(variants))
    axes[0, 0].bar(x, rmse, color="tab:blue")
    axes[0, 0].set_xticks(x, variants, rotation=35, ha="right")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].set_title(f"RMSE @ step {final_step}")
    axes[0, 1].bar(x, crps, color="tab:orange")
    axes[0, 1].set_xticks(x, variants, rotation=35, ha="right")
    axes[0, 1].set_ylabel("CRPS")
    axes[0, 1].set_title("CRPS")
    axes[1, 0].bar(x, r_mean, color="tab:green")
    axes[1, 0].set_xticks(x, variants, rotation=35, ha="right")
    axes[1, 0].set_ylabel("r_mean")
    axes[1, 0].set_title("Mixture responsibility r")
    axes[1, 1].bar(x, pi_test, color="tab:red")
    axes[1, 1].set_xticks(x, variants, rotation=35, ha="right")
    axes[1, 1].set_ylabel("pi_test")
    axes[1, 1].set_title("Gate E[sigmoid] @ test")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="A/B/C remedies for direct-VI predict.")
    p.add_argument("--setting", default="lh_q5_train500")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--num_steps", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument(
        "--variants",
        default="C0_AB,C1,C2,C3,C4,C5",
        help="Comma-separated variant keys (see module docstring).",
    )
    p.add_argument("--no_ab", action="store_true", help="Disable A+B base for C variants.")
    # A
    p.add_argument("--lambda_s", type=float, default=0.1)
    p.add_argument("--sigma_v_start", type=float, default=0.2)
    p.add_argument("--sigma_v_end", type=float, default=2.0)
    p.add_argument("--warmup_frac", type=float, default=0.5)
    # C
    p.add_argument("--omega0_init", type=float, default=2.0)
    p.add_argument("--omega0_floor", type=float, default=0.0)
    p.add_argument("--pi_gate_floor", type=float, default=0.05)
    p.add_argument("--lambda_pi_floor", type=float, default=1.0)
    p.add_argument("--force_inlier_steps", type=int, default=150)
    p.add_argument("--u_logit_ceil", type=float, default=-1.0)
    p.add_argument("--out_dir", default="experiments/synthetic/ab_remedies_direct_vi_smoke")
    args = p.parse_args()

    cfg = SETTING_PRESETS[args.setting]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    _set_seed(args.seed)
    data = _generate_paper_data(cfg, args.seed, device)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data["X_train"])
    X_test_s = scaler.transform(data["X_test"])
    y_train = data["y_train"]
    y_test = np.asarray(data["y_test"], dtype=np.float64)[: args.max_test]
    X_test_s = X_test_s[: args.max_test]

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(np.asarray(y_train, np.float32)).to(device)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)

    Q = int(cfg["Q"])
    n = int(cfg["n"])

    all_specs = _variant_specs(args)
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(want) - set(all_specs))
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; available: {sorted(all_specs)}")

    regions_base, V_base, u_base, hyp_base = _build_state(
        X_train_t, y_train_t, X_test_t,
        Q=Q, n=n, m1=args.m1, m2=args.m2, device=device, seed=args.seed,
    )

    all_records: list[dict[str, Any]] = []

    for vname in want:
        spec = all_specs[vname]
        print(f"\n{'='*70}\n  VARIANT: {vname}\n  {json.dumps(spec, indent=2)}\n{'='*70}", flush=True)

        V, u, h = _deep_clone_state(V_base, u_base, hyp_base)
        if spec.get("omega0_init") is not None:
            for ui in u:
                ui["omega"] = ui["omega"].detach().clone()
                ui["omega"][0] = float(spec["omega0_init"])
                ui["omega"].requires_grad_(True)

        regions = _clone_regions(regions_base)
        recs = train_vi_augmented(
            regions, V, u, h,
            lr=args.lr,
            num_steps=args.num_steps,
            eval_every=args.eval_every,
            y_test=y_test,
            variant_name=vname,
            lambda_s=float(spec["lambda_s"]),
            anneal_sigma_v=bool(spec["anneal_sigma_v"]),
            sigma_v_start=float(spec["sigma_v_start"]),
            sigma_v_end=float(spec["sigma_v_end"]),
            warmup_frac=float(spec["warmup_frac"]),
            omega0_floor=spec.get("omega0_floor"),
            u_logit_ceil=spec.get("u_logit_ceil"),
            pi_gate_floor=float(spec.get("pi_gate_floor", 0.0)),
            lambda_pi_floor=float(spec.get("lambda_pi_floor", 0.0)),
            lambda_inlier_rate=float(spec.get("lambda_inlier_rate", 0.0)),
            force_inlier_steps=int(spec.get("force_inlier_steps", 0)),
        )
        all_records.extend(recs)

    steps_seen = sorted(set(r["step"] for r in all_records))
    print(f"\n{'='*100}\nRMSE / CRPS / r / pi_test @ each checkpoint\n{'='*100}")
    hdr = f"{'Step':>6s}"
    for vn in want:
        hdr += f" | {vn:>12s}"
    print(hdr)
    print("-" * len(hdr))
    for step in steps_seen:
        line = f"{step:>6d}"
        for vn in want:
            rec = next((r for r in all_records if r["variant"] == vn and r["step"] == step), None)
            if rec:
                line += (
                    f" | {rec['rmse']:6.1f}"
                    f"/{rec['crps']:5.0f}"
                    f" r={rec['r_mean']:.3f}"
                    f" pi={rec['pi_gate_test_mean']:.3f}"
                )
            else:
                line += " |     --"
        print(line)

    print(f"\nFINAL diagnostics (step {steps_seen[-1]}):")
    print(
        f"{'Variant':<12s} {'RMSE':>8s} {'RMSE_pi1':>9s} {'r':>8s} "
        f"{'pi_test':>8s} {'dT':>8s} {'std_muf':>8s} {'s':>8s}"
    )
    print("-" * 80)
    for vn in want:
        rec = next(
            (r for r in all_records if r["variant"] == vn and r["step"] == steps_seen[-1]), None,
        )
        if rec:
            print(
                f"{vn:<12s} {rec['rmse']:8.2f} {rec['rmse_pi1']:9.2f} "
                f"{rec['r_mean']:8.4f} {rec['pi_gate_test_mean']:8.4f} "
                f"{rec['delta_T_mean']:8.2f} {rec['std_mu_f']:8.2f} {rec['s_mean']:8.2f}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_records for k in r})
    with (out_dir / "eval_records.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_records)

    summary = {"args": vars(args), "variants": want, "n_records": len(all_records)}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_final_metrics(all_records, out_dir / "final_metrics_bars.png")

    readme = f"""# A/B/C remedies — direct VI predict

Variants run: {', '.join(want)}

Key columns in ``eval_records.csv``:
- ``r_mean``: softmax responsibility on inlier branch (neighbour-averaged)
- ``pi_gate_test_mean``: E[sigmoid(omega^T [1,Wx])] at test anchors
- ``delta_T_mean``: mean(T_in - T_out)
- ``rmse_pi1``: RMSE if mixture uses pi=1 (GP-only path)

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"\nSaved {len(all_records)} rows to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

"""Structured-metric LMJGP training for CEM-style composite likelihood.

This module keeps the local JumpGP/CEM head from ``uncertain_w_jgp`` but replaces
the free ``Q x D`` sparse-GP projection family with a structured sparse GP:

    R_l in R^Q at inducing inputs,     eta_j | R ~ GP conditioning,
    W_j = diag(exp(eta_j / 2)) U.T,    U.T @ U = I.

The training loop is stochastic over anchors and uses a detached self-CEM state
for each sampled W, then differentiates the frozen CEM likelihood through the
sampled structured metric.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import torch

from djgp.projections.uncertain_w_jgp import (
    SelfCEMConfig,
    _as_float_tensor,
    _diag_sparse_gp_kl,
    _hard_cem_bound_per_anchor_from_w_moments,
    run_uncertain_w_self_cem,
    sparse_gp_w_moments_diag,
)


EvalCallback = Callable[[int, dict[str, torch.Tensor], dict[str, float]], dict[str, float]]
AnalyticEvalCallback = Callable[
    [int, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]],
    dict[str, float],
]


@dataclass(frozen=True)
class StructuredMetricLMJGPConfig:
    """Configuration for the structured metric composite-likelihood M-step."""

    steps: int = 20
    lr: float = 0.01
    batch_anchors: int = 40
    n_samples: int = 2
    beta_kl: float = 0.02
    kl_warmup_steps: int = 10
    eta_prior_std: float = 1.0
    eta_init_log_std: float = -3.0
    train_eta_log_std: bool = True
    n_inducing: int | None = None
    w_lengthscale: float | None = None
    w_signal_var: float = 1.0
    include_conditional_residual: bool = True
    trace_target: float | None = None
    trace_normalize: bool = True
    trace_penalty_weight: float = 0.01
    outlier_mode: str = "background_normal"
    outlier_sigma_mult: float = 2.5
    background_scale_mult: float = 3.0
    gh_points: int = 20
    jitter: float = 1e-5
    min_inliers: int = 2
    grad_clip_norm: float = 10.0
    log_interval: int = 1
    eval_interval: int = 1
    restore_state: Literal["final", "best_loss", "best_direct"] = "final"
    seed: int = 0


@dataclass
class StructuredMetricLMJGPResult:
    """Structured metric training output."""

    U: torch.Tensor
    R_mu: torch.Tensor
    R_log_std: torch.Tensor
    inducing_X: torch.Tensor
    eta_mu: torch.Tensor
    eta_log_std: torch.Tensor
    w_lengthscale: float
    w_signal_var: float
    include_conditional_residual: bool
    history: list[dict[str, float]]
    train_sec: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class AnalyticStructuredMetricLMJGPConfig:
    """Configuration for analytic ISM-LMJGP with marginalised W."""

    steps: int = 40
    lr: float = 0.01
    batch_anchors: int = 40
    beta_kl_R: float = 0.02
    beta_kl_u: float = 1.0
    kl_warmup_steps: int = 10
    eta_prior_std: float = 1.0
    eta_init_log_std: float = -3.0
    train_eta_log_std: bool = True
    # Floor on the q(eta) inducing log-std. The data likelihood drives q(W) to a
    # near point-mass (eta_std collapses to ~0.02), which removes the metric-scale
    # uncertainty from the predictive. A floor keeps a minimum q(W) spread so the
    # predictive variance reflects residual metric uncertainty. Default -8 = off.
    r_log_std_min: float = -8.0
    n_inducing_R: int | None = None
    w_lengthscale: float | None = None
    w_signal_var: float = 1.0
    include_conditional_residual: bool = True
    local_signal_var: float = 1.0
    # The observation noise used to be pinned tightly at 0.1. The local GP overfits
    # the neighbour responses, so the data-optimal noise collapses to ~0.1 and the
    # inlier predictive (anchor not in the training neighbours) is far too narrow.
    # Anchor the noise at a calibration-sensible ~0.25 so test-point intervals match
    # the residual scale; the prior is the effective noise floor here.
    obs_noise_init: float = 0.25
    obs_noise_prior_std: float = 0.3
    obs_noise_prior_weight: float = 3.0
    # Observation-noise prior mode. "fixed" anchors every anchor's noise prior at
    # obs_noise_init (the original behaviour: the noise pins at ~0.25 and the
    # analytic-direct predictive uniformly undercovers on UCI, see
    # docs/ism_lmjgp_analytic.md 8c-calib). "data_driven" instead sets a
    # PER-ANCHOR (heteroscedastic) prior mean from a nearest-neighbour-difference
    # (Rice/Gasser) estimate of the local residual scale in the W0-projected
    # neighbour space, so the learnable observation noise is pulled toward the
    # genuine out-of-sample residual scale. The log-Gaussian prior keeps it
    # Bayesian and ``obs_noise_max`` caps it so it cannot blow up.
    obs_noise_prior_mode: Literal["fixed", "data_driven"] = "fixed"
    obs_noise_data_driven_frac: float = 1.0
    obs_noise_data_driven_floor: float = 0.1
    obs_noise_max: float = 1.5
    init_noise_at_prior: bool = True
    rho_init: float = 0.8
    rho_prior_strength: float = 2.0
    min_rho_mean: float = 0.55
    min_rho_penalty_weight: float = 50.0
    outlier_scale_mult: float = 3.0
    trace_target: float | None = None
    trace_normalize: bool = True
    trace_penalty_weight: float = 0.01
    grad_clip_norm: float = 10.0
    log_interval: int = 1
    eval_interval: int = 5
    restore_state: Literal["final", "best_loss", "best_direct"] = "final"
    seed: int = 0
    jitter: float = 1e-5


@dataclass
class AnalyticStructuredMetricLMJGPResult:
    """Analytic structured metric training output."""

    U: torch.Tensor
    R_mu: torch.Tensor
    R_log_std: torch.Tensor
    inducing_X: torch.Tensor
    mu_u: torch.Tensor
    u_log_std: torch.Tensor
    log_noise_var: torch.Tensor
    rho_logit: torch.Tensor
    C: torch.Tensor
    w_lengthscale: float
    w_signal_var: float
    local_signal_var: float
    include_conditional_residual: bool
    history: list[dict[str, float]]
    train_sec: float
    diagnostics: dict[str, Any]


def _repeat_first(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.unsqueeze(0).expand(int(n), *tuple(x.shape)).reshape(
        int(n) * x.shape[0], *tuple(x.shape[1:])
    ).contiguous()


def _orthonormal_columns(raw: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(raw, mode="reduced")
    sign = torch.sign(torch.diagonal(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign.view(1, -1)


def init_structured_metric_from_w0(
    W0: torch.Tensor | np.ndarray,
    *,
    n_anchors: int,
    trace_target: float | None = None,
    eta_init_log_std: float = -3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize ``U`` and per-anchor ``q(eta)`` from a fixed projection W0."""
    W0_t = _as_float_tensor(W0)
    if W0_t.dim() != 2:
        raise ValueError("W0 must have shape [Q,D].")
    Q, D = W0_t.shape
    if D < Q:
        raise ValueError("Structured metric requires observed dimension D >= Q.")
    U0 = _orthonormal_columns(W0_t.transpose(0, 1))
    target = float(Q if trace_target is None else trace_target)
    row_power = W0_t.pow(2).sum(dim=1).clamp_min(1e-8)
    row_power = row_power * (target / row_power.sum().clamp_min(1e-8))
    eta0 = row_power.log().view(1, Q).expand(int(n_anchors), Q).clone()
    eta_log_std0 = torch.full_like(eta0, float(eta_init_log_std))
    return U0, eta0, eta_log_std0


def _eta_trace(
    eta_mu: torch.Tensor,
    eta_log_std: torch.Tensor,
) -> torch.Tensor:
    return _eta_trace_from_var(eta_mu, eta_log_std.exp().pow(2))


def _eta_trace_from_var(
    eta_mu: torch.Tensor,
    eta_var: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(eta_mu + 0.5 * eta_var).sum(dim=-1)


def _default_sparse_gp_lengthscale(X: torch.Tensor) -> float:
    if X.shape[0] <= 1:
        return 1.0
    with torch.no_grad():
        d = torch.pdist(X.detach())
        d = d[torch.isfinite(d) & (d > 0)]
        if d.numel() == 0:
            return 1.0
        return float(d.median().clamp_min(1e-3).cpu().item())


def _select_inducing_X(
    X_anchor: torch.Tensor,
    n_inducing: int | None,
) -> torch.Tensor:
    T = int(X_anchor.shape[0])
    M = T if n_inducing is None else max(1, min(int(n_inducing), T))
    if M == T:
        return X_anchor.detach().clone()
    idx = torch.linspace(0, T - 1, steps=M, device=X_anchor.device).round().long().unique()
    if idx.numel() < M:
        remaining = torch.arange(T, device=X_anchor.device)
        keep = torch.ones(T, device=X_anchor.device, dtype=torch.bool)
        keep[idx] = False
        idx = torch.cat([idx, remaining[keep][: M - idx.numel()]])
    return X_anchor[idx[:M]].detach().clone()


def structured_metric_eta_moments_from_R(
    X_query: torch.Tensor | np.ndarray,
    inducing_X: torch.Tensor | np.ndarray,
    R_mu: torch.Tensor | np.ndarray,
    R_log_std: torch.Tensor | np.ndarray,
    *,
    lengthscale: float,
    signal_var: float = 1.0,
    include_conditional_residual: bool = True,
    jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return marginal ``q(eta(x))`` moments induced by structured ``q(R)``.

    ``R`` stores only the local log-diagonal metric coordinates, so it has shape
    ``[M,Q]``.  The shared subspace ``U`` is global and not part of this GP.
    """
    Xq = _as_float_tensor(X_query)
    if Xq.dim() != 2:
        raise ValueError("X_query must have shape [T,D].")
    device = Xq.device
    dtype = Xq.dtype
    U = _as_float_tensor(inducing_X, device=device).to(dtype=dtype)
    Rm = _as_float_tensor(R_mu, device=device).to(dtype=dtype)
    Rls = _as_float_tensor(R_log_std, device=device).to(dtype=dtype)
    if Rm.dim() != 2 or Rls.shape != Rm.shape:
        raise ValueError("R_mu/R_log_std must have shape [M,Q].")
    if U.shape[0] != Rm.shape[0] or U.shape[1] != Xq.shape[1]:
        raise ValueError("inducing_X dimensions are inconsistent with X_query/R.")
    eta_mu_all, eta_cov_all = sparse_gp_w_moments_diag(
        Xq.unsqueeze(1),
        U,
        Rm.unsqueeze(-1),
        torch.exp(2.0 * Rls).clamp_min(1e-12).unsqueeze(-1),
        lengthscale=float(lengthscale),
        signal_var=float(signal_var),
        include_conditional_residual=bool(include_conditional_residual),
        jitter=float(jitter),
    )
    eta_mu = eta_mu_all[:, 0, :, 0]
    eta_var = eta_cov_all[:, :, 0, 0, 0].clamp_min(1e-12)
    return eta_mu, eta_var


def structured_metric_w_moments(
    U: torch.Tensor,
    eta_mu: torch.Tensor,
    eta_log_std: torch.Tensor,
    *,
    trace_target: float,
    trace_normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``W_mu`` and full row covariance for the structured q(W)."""
    return structured_metric_w_moments_from_eta_moments(
        U,
        eta_mu,
        eta_log_std.exp().pow(2),
        trace_target=trace_target,
        trace_normalize=trace_normalize,
    )


def structured_metric_w_moments_from_eta_moments(
    U: torch.Tensor,
    eta_mu: torch.Tensor,
    eta_var: torch.Tensor,
    *,
    trace_target: float,
    trace_normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``q(W)`` moments from marginal ``q(eta)`` moments."""
    eta_eff = eta_mu
    if bool(trace_normalize):
        trace = torch.exp(eta_mu + 0.5 * eta_var).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        eta_eff = eta_mu - torch.log(trace / float(trace_target))
    scale_mean = torch.exp(0.5 * eta_eff + 0.125 * eta_var)
    scale_second = torch.exp(eta_eff + 0.5 * eta_var)
    scale_var = (scale_second - scale_mean.pow(2)).clamp_min(0.0)
    W_mu = torch.einsum("tq,dq->tqd", scale_mean, U)
    UU = torch.einsum("dq,eq->qde", U, U)
    W_cov = scale_var[:, :, None, None] * UU.unsqueeze(0)
    trace_raw = _eta_trace_from_var(eta_mu, eta_var)
    diag = {
        "trace_raw_mean": float(trace_raw.detach().mean().cpu().item()),
        "trace_raw_std": float(trace_raw.detach().std(unbiased=False).cpu().item()),
        "eta_std_median": float(eta_var.detach().sqrt().median().cpu().item()),
        "D_entropy_mean": float(
            (
                0.5 * (1.0 + math.log(2.0 * math.pi))
                + 0.5 * torch.log(eta_var.clamp_min(1e-12))
            ).sum(dim=1).mean().detach().cpu().item()
        ),
    }
    return W_mu, W_cov, diag


def structured_metric_w_moments_from_R(
    U: torch.Tensor,
    X_query: torch.Tensor | np.ndarray,
    inducing_X: torch.Tensor | np.ndarray,
    R_mu: torch.Tensor | np.ndarray,
    R_log_std: torch.Tensor | np.ndarray,
    *,
    lengthscale: float,
    signal_var: float,
    include_conditional_residual: bool,
    trace_target: float,
    trace_normalize: bool,
    jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``q(W(x))`` and ``q(eta(x))`` moments induced by ``q(R)``."""
    eta_mu, eta_var = structured_metric_eta_moments_from_R(
        X_query,
        inducing_X,
        R_mu,
        R_log_std,
        lengthscale=float(lengthscale),
        signal_var=float(signal_var),
        include_conditional_residual=bool(include_conditional_residual),
        jitter=float(jitter),
    )
    W_mu, W_cov, diag = structured_metric_w_moments_from_eta_moments(
        U,
        eta_mu,
        eta_var,
        trace_target=float(trace_target),
        trace_normalize=bool(trace_normalize),
    )
    return W_mu, W_cov, eta_mu, eta_var, diag


def sample_structured_metric_w(
    U: torch.Tensor,
    eta_mu: torch.Tensor,
    eta_log_std: torch.Tensor,
    *,
    n_samples: int,
    trace_target: float,
    trace_normalize: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample structured W with shape ``[S,T,Q,D]``."""
    eps = torch.randn(
        (int(n_samples), *tuple(eta_mu.shape)),
        device=eta_mu.device,
        dtype=eta_mu.dtype,
        generator=generator,
    )
    eta = eta_mu.unsqueeze(0) + eps * eta_log_std.exp().unsqueeze(0)
    if bool(trace_normalize):
        trace = torch.exp(eta).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        eta = eta - torch.log(trace / float(trace_target))
    scale = torch.exp(0.5 * eta)
    return torch.einsum("stq,dq->stqd", scale, U)


def sample_structured_metric_w_from_eta_moments(
    U: torch.Tensor,
    eta_mu: torch.Tensor,
    eta_var: torch.Tensor,
    *,
    n_samples: int,
    trace_target: float,
    trace_normalize: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample structured W from marginal ``q(eta)`` moments."""
    eps = torch.randn(
        (int(n_samples), *tuple(eta_mu.shape)),
        device=eta_mu.device,
        dtype=eta_mu.dtype,
        generator=generator,
    )
    eta = eta_mu.unsqueeze(0) + eps * eta_var.clamp_min(1e-12).sqrt().unsqueeze(0)
    if bool(trace_normalize):
        trace = torch.exp(eta).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        eta = eta - torch.log(trace / float(trace_target))
    scale = torch.exp(0.5 * eta)
    return torch.einsum("stq,dq->stqd", scale, U)


def _eta_kl(
    eta_mu: torch.Tensor,
    eta_log_std: torch.Tensor,
    *,
    trace_target: float,
    prior_std: float,
) -> torch.Tensor:
    Q = eta_mu.shape[1]
    prior_mu = math.log(float(trace_target) / float(Q))
    prior_var = float(prior_std) ** 2
    var = eta_log_std.exp().pow(2)
    return 0.5 * (
        (var + (eta_mu - prior_mu).pow(2)) / prior_var
        - 1.0
        + math.log(prior_var)
        - torch.log(var.clamp_min(1e-12))
    ).sum()


def build_fixed_local_inducing_from_w0(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    W0: torch.Tensor | np.ndarray,
    *,
    m_inducing: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose fixed local inducing locations near each anchor in ``W0 x`` space.

    The first inducing location is the projected anchor.  Remaining locations
    are the nearest projected raw-X neighbors.  The returned ``mu_u`` initializer
    uses the corresponding neighbor responses, with the anchor slot initialized
    from the nearest-neighbor response.
    """
    Xa = _as_float_tensor(X_anchor)
    device = Xa.device
    dtype = Xa.dtype
    Xn = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    W = _as_float_tensor(W0, device=device).to(dtype=dtype)
    if y.dim() == 3 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    T, n, D = Xn.shape
    Q = W.shape[0]
    m = max(1, int(m_inducing))
    za = torch.einsum("qd,td->tq", W, Xa)
    zn = torch.einsum("qd,tnd->tnq", W, Xn)
    dist = (zn - za.unsqueeze(1)).pow(2).sum(dim=-1)
    k = min(n, max(1, m - 1))
    idx = torch.topk(dist, k=k, largest=False).indices
    gather_idx = idx.unsqueeze(-1).expand(T, k, Q)
    C_tail = torch.gather(zn, 1, gather_idx)
    y_tail = torch.gather(y, 1, idx)
    C = torch.cat([za.unsqueeze(1), C_tail], dim=1)
    mu_u0 = torch.cat([y_tail[:, :1], y_tail], dim=1)
    if C.shape[1] < m:
        pad = m - C.shape[1]
        C = torch.cat([C, C[:, -1:].expand(T, pad, Q)], dim=1)
        mu_u0 = torch.cat([mu_u0, mu_u0[:, -1:].expand(T, pad)], dim=1)
    return C.contiguous(), mu_u0.contiguous()


def _local_kl_diag_u(
    C: torch.Tensor,
    mu_u: torch.Tensor,
    u_var: torch.Tensor,
    *,
    jitter: float,
    signal_var: float = 1.0,
) -> torch.Tensor:
    T, m, _Q = C.shape
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    K = float(signal_var) * torch.exp(-0.5 * d2) + float(jitter) * torch.eye(m, device=C.device, dtype=C.dtype).unsqueeze(0)
    L = torch.linalg.cholesky(K)
    K_inv = torch.cholesky_inverse(L)
    diag_inv = torch.diagonal(K_inv, dim1=-2, dim2=-1)
    logdet_K = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    trace = (diag_inv * u_var).sum(dim=-1)
    quad = torch.einsum("ti,tij,tj->t", mu_u, K_inv, mu_u)
    logdet_q = torch.log(u_var.clamp_min(1e-12)).sum(dim=-1)
    return 0.5 * (trace + quad - float(m) + logdet_K - logdet_q)


def _analytic_local_terms(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    C: torch.Tensor,
    W_mu: torch.Tensor,
    W_cov: torch.Tensor,
    mu_u: torch.Tensor,
    u_log_std: torch.Tensor,
    log_noise_var: torch.Tensor,
    rho_logit: torch.Tensor,
    *,
    local_signal_var: float,
    outlier_scale_mult: float,
    jitter: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if y_neighbors.dim() == 3 and y_neighbors.shape[-1] == 1:
        y_neighbors = y_neighbors.squeeze(-1)
    T, n, _D = X_neighbors.shape
    m = C.shape[1]
    sigma_f = torch.full((T,), math.sqrt(float(local_signal_var)), device=X_neighbors.device, dtype=X_neighbors.dtype)
    Kfu = _expected_Kfu_structured(W_mu, W_cov, X_neighbors, C, sigma_f)
    KufKfu = _expected_KufKfu_structured(W_mu, W_cov, X_neighbors, C, sigma_f)
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    Kuu = float(local_signal_var) * torch.exp(-0.5 * d2) + float(jitter) * torch.eye(m, device=C.device, dtype=C.dtype).unsqueeze(0)
    Luu = torch.linalg.cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(Luu)
    u_var = torch.exp(2.0 * u_log_std).clamp_min(1e-12)
    u_second = torch.diag_embed(u_var) + torch.einsum("ti,tj->tij", mu_u, mu_u)
    alpha = torch.einsum("tij,tj->ti", Kuu_inv, mu_u)
    mean_f = torch.einsum("tni,ti->tn", Kfu, alpha)
    v1 = float(local_signal_var) - torch.einsum("tij,tnji->tn", Kuu_inv, KufKfu)
    t3 = torch.einsum("tij,tnjk,tkm,tmi->tn", Kuu_inv, KufKfu, Kuu_inv, u_second)
    second_f = (v1 + t3).clamp_min(1e-12)
    noise_var = log_noise_var.exp().clamp_min(1e-8).view(T, 1)
    sqerr = y_neighbors.pow(2) - 2.0 * y_neighbors * mean_f + second_f
    log_inlier = -0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(noise_var) - 0.5 * sqerr / noise_var
    rho = torch.sigmoid(rho_logit).view(T, 1).clamp(1e-4, 1.0 - 1e-4)
    local_mean = y_neighbors.mean(dim=1, keepdim=True)
    local_var = y_neighbors.var(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
    out_var = (float(outlier_scale_mult) ** 2) * local_var
    log_outlier = -0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(out_var) - 0.5 * (y_neighbors - local_mean).pow(2) / out_var
    point_logp = torch.logsumexp(
        torch.stack([
            torch.log(rho) + log_inlier,
            torch.log1p(-rho) + log_outlier,
        ], dim=0),
        dim=0,
    )
    region_elbo = point_logp.sum(dim=1)
    kl_u = _local_kl_diag_u(C, mu_u, u_var, jitter=jitter, signal_var=local_signal_var)
    diag = {
        "region_elbo_mean": region_elbo.mean().detach(),
        "kl_u_mean": kl_u.mean().detach(),
        "rho_mean": rho.mean().detach(),
        "noise_std_mean": noise_var.sqrt().mean().detach(),
        "mean_f_abs": mean_f.abs().mean().detach(),
        "mean_second_f": second_f.mean().detach(),
    }
    return region_elbo, kl_u, diag


def _expected_Kfu_structured(mu_W: torch.Tensor, cov_W: torch.Tensor, X: torch.Tensor, C: torch.Tensor, sigma_f: torch.Tensor) -> torch.Tensor:
    T, _n, _D = X.shape
    s = torch.einsum("tnd,tqde,tne->tnq", X, cov_W, X).clamp_min(0.0)
    den = torch.sqrt(s + 1.0)
    mu_proj = torch.einsum("tqd,tnd->tnq", mu_W, X)
    diff = mu_proj.unsqueeze(2) - C.unsqueeze(1)
    exp_term = torch.exp(-0.5 * diff.pow(2) / (s.unsqueeze(2) + 1.0))
    # RBF amplitude is sigma_f**2; Psi1 = E_W[k(z,C)] carries one factor of sigma_f**2.
    return sigma_f.view(T, 1, 1).pow(2) * exp_term.prod(dim=-1) / den.prod(dim=-1, keepdim=True)


def _expected_KufKfu_structured(mu_W: torch.Tensor, cov_W: torch.Tensor, X: torch.Tensor, C: torch.Tensor, sigma_f: torch.Tensor) -> torch.Tensor:
    T, n, _D = X.shape
    s = torch.einsum("tnd,tqde,tne->tnq", X, cov_W, X).clamp_min(0.0)
    mu_proj = torch.einsum("tqd,tnd->tnq", mu_W, X)
    mid = 0.5 * (C.unsqueeze(2) + C.unsqueeze(1))
    diff = mu_proj.unsqueeze(2).unsqueeze(3) - mid.unsqueeze(1)
    # Product of two RBF kernels doubles the curvature: the marginal over a
    # Gaussian z ~ N(mu, s) of exp(-(z-mid)^2) integrates to
    # (1/sqrt(1+2s)) exp(-(mu-mid)^2/(1+2s)), so the denominator is (2s+1),
    # not (s+1) as in the single-kernel Psi1 statistic.
    denom = 2.0 * s + 1.0
    exp_term = torch.exp(-diff.pow(2) / denom.unsqueeze(2).unsqueeze(2))
    num = exp_term.prod(dim=-1)
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    prior = torch.exp(-0.25 * d2)
    den_prod = torch.sqrt(denom).prod(dim=-1)
    # Psi2 = E_W[k(z,C_i) k(z,C_j)] carries sigma_f**4 (two kernel factors).
    return sigma_f.view(T, 1, 1, 1).pow(4) * prior.unsqueeze(1) * (num / den_prod.view(T, n, 1, 1))


def train_structured_metric_lmjgp(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    init_labels: torch.Tensor | np.ndarray,
    *,
    init_W: torch.Tensor | np.ndarray,
    init_lengthscale: torch.Tensor | np.ndarray,
    init_signal_var: torch.Tensor | np.ndarray,
    init_noise_var: torch.Tensor | np.ndarray,
    gate_init: torch.Tensor | np.ndarray,
    cem_config: SelfCEMConfig,
    config: StructuredMetricLMJGPConfig,
    eval_callback: EvalCallback | None = None,
) -> StructuredMetricLMJGPResult:
    """Train structured ``q(eta), U`` with stochastic composite likelihood."""
    t0 = time.perf_counter()
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    dtype = X_anchor_t.dtype
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y_t = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    labels_t = _as_float_tensor(init_labels, device=device).to(dtype=dtype)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    if labels_t.dim() == 3 and labels_t.shape[-1] == 1:
        labels_t = labels_t.squeeze(-1)
    T, _n, D = X_neighbors_t.shape
    W0 = _as_float_tensor(init_W, device=device).to(dtype=dtype)
    Q = W0.shape[0]
    trace_target = float(Q if config.trace_target is None else config.trace_target)
    U0, eta0, eta_log_std0 = init_structured_metric_from_w0(
        W0,
        n_anchors=T,
        trace_target=trace_target,
        eta_init_log_std=float(config.eta_init_log_std),
    )
    inducing_X = _select_inducing_X(X_anchor_t, config.n_inducing).to(device=device, dtype=dtype)
    M = int(inducing_X.shape[0])
    R0 = eta0[:1].expand(M, Q).contiguous()
    U_raw = torch.nn.Parameter(U0.to(device=device, dtype=dtype).clone())
    R_mu = torch.nn.Parameter(R0.to(device=device, dtype=dtype).clone())
    R_log_std = torch.nn.Parameter(
        eta_log_std0[:1].expand(M, Q).contiguous().to(device=device, dtype=dtype).clone()
    )
    params: list[torch.nn.Parameter] = [U_raw, R_mu]
    if bool(config.train_eta_log_std):
        params.append(R_log_std)
    opt = torch.optim.Adam(params, lr=float(config.lr))
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(config.seed))
    init_lengthscale_t = _as_float_tensor(init_lengthscale, device=device).to(dtype=dtype)
    init_signal_var_t = _as_float_tensor(init_signal_var, device=device).to(dtype=dtype)
    init_noise_var_t = _as_float_tensor(init_noise_var, device=device).to(dtype=dtype)
    gate_init_t = _as_float_tensor(gate_init, device=device).to(dtype=dtype)
    w_lengthscale = (
        _default_sparse_gp_lengthscale(X_anchor_t)
        if config.w_lengthscale is None
        else float(config.w_lengthscale)
    )
    w_signal_var = float(config.w_signal_var) * float(config.eta_prior_std) ** 2
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_direct_rmse = float("inf")
    best_direct_state: dict[str, torch.Tensor] | None = None
    for step in range(1, int(config.steps) + 1):
        opt.zero_grad()
        B = min(int(config.batch_anchors), T)
        idx = torch.randperm(T, device=device, generator=gen)[:B]
        U = _orthonormal_columns(U_raw)
        eta_mu_b, eta_var_b = structured_metric_eta_moments_from_R(
            X_anchor_t[idx],
            inducing_X,
            R_mu,
            R_log_std,
            lengthscale=w_lengthscale,
            signal_var=w_signal_var,
            include_conditional_residual=bool(config.include_conditional_residual),
            jitter=float(config.jitter),
        )
        W_s = sample_structured_metric_w_from_eta_moments(
            U,
            eta_mu_b,
            eta_var_b,
            n_samples=int(config.n_samples),
            trace_target=trace_target,
            trace_normalize=bool(config.trace_normalize),
            generator=gen,
        )
        S = W_s.shape[0]
        W_flat = W_s.reshape(S * B, Q, D).contiguous()
        zero_var = torch.zeros_like(W_flat)
        X_anchor_b = _repeat_first(X_anchor_t[idx], S)
        X_neighbors_b = _repeat_first(X_neighbors_t[idx], S)
        y_b = _repeat_first(y_t[idx], S)
        labels_b = _repeat_first(labels_t[idx], S)
        state = run_uncertain_w_self_cem(
            X_anchor_b,
            X_neighbors_b,
            y_b,
            labels_b,
            mode="anchor",
            W_mu_anchor=W_flat.detach(),
            W_var_anchor=zero_var.detach(),
            init_lengthscale=_repeat_first(init_lengthscale_t[idx], S),
            init_signal_var=_repeat_first(init_signal_var_t[idx], S),
            init_noise_var=_repeat_first(init_noise_var_t[idx], S),
            gate_init=_repeat_first(gate_init_t[idx], S),
            config=cem_config,
        )
        gate = state["gate"]
        if isinstance(gate, torch.Tensor):
            gate = gate.clone()
            gate[:, 1:] = 0.0
        outlier_logp = state.get("outlier_logp") if isinstance(state, dict) else None
        if isinstance(outlier_logp, torch.Tensor) and outlier_logp.numel() == 0:
            outlier_logp = None
        rewards, reward_diag = _hard_cem_bound_per_anchor_from_w_moments(
            X_anchor_b,
            X_neighbors_b,
            y_b,
            state["labels"],
            mode="anchor",
            w_moments={"W_mu_anchor": W_flat, "W_var_anchor": zero_var},
            gate=gate,
            lengthscale=state["lengthscale"],
            signal_var=state["signal_var"],
            noise_var=state["noise_var"],
            mean_const=state["mean_const"],
            outlier_mode=str(config.outlier_mode),
            outlier_sigma_mult=float(config.outlier_sigma_mult),
            background_scale_mult=float(config.background_scale_mult),
            outlier_logp=outlier_logp,
            gh_points=int(config.gh_points),
            jitter=float(config.jitter),
            min_inliers=int(config.min_inliers),
        )
        data_objective = rewards.mean()
        kl = _diag_sparse_gp_kl(
            inducing_X,
            R_mu.unsqueeze(-1),
            R_log_std.unsqueeze(-1),
            lengthscale=w_lengthscale,
            signal_var=w_signal_var,
            jitter=float(config.jitter),
        ) / float(T)
        beta_kl = float(config.beta_kl) * min(1.0, step / max(1, int(config.kl_warmup_steps)))
        eta_mu_all, eta_var_all = structured_metric_eta_moments_from_R(
            X_anchor_t,
            inducing_X,
            R_mu,
            R_log_std,
            lengthscale=w_lengthscale,
            signal_var=w_signal_var,
            include_conditional_residual=bool(config.include_conditional_residual),
            jitter=float(config.jitter),
        )
        trace_raw = _eta_trace_from_var(eta_mu_all, eta_var_all)
        trace_penalty = (trace_raw - trace_target).pow(2).mean()
        loss = -data_objective + beta_kl * kl + float(config.trace_penalty_weight) * trace_penalty
        loss.backward()
        if float(config.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip_norm))
        opt.step()
        with torch.no_grad():
            R_log_std.clamp_(min=-8.0, max=2.0)
        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "U_raw": U_raw.detach().clone(),
                "R_mu": R_mu.detach().clone(),
                "R_log_std": R_log_std.detach().clone(),
            }
        if step == 1 or step % int(config.log_interval) == 0 or step == int(config.steps):
            with torch.no_grad():
                U_eval = _orthonormal_columns(U_raw)
                W_mu, W_cov, _eta_mu_eval, _eta_var_eval, moment_diag = structured_metric_w_moments_from_R(
                    U_eval,
                    X_anchor_t,
                    inducing_X,
                    R_mu,
                    R_log_std,
                    lengthscale=w_lengthscale,
                    signal_var=w_signal_var,
                    include_conditional_residual=bool(config.include_conditional_residual),
                    trace_target=trace_target,
                    trace_normalize=bool(config.trace_normalize),
                    jitter=float(config.jitter),
                )
            eval_diag: dict[str, float] = {}
            if eval_callback is not None and (
                step == 1 or step % max(1, int(config.eval_interval)) == 0 or step == int(config.steps)
            ):
                eval_diag = eval_callback(
                    step,
                    {"W_mu_anchor": W_mu.detach(), "W_cov_anchor": W_cov.detach()},
                    moment_diag,
                )
                if "direct_eval_rmse" in eval_diag:
                    direct_rmse = float(eval_diag["direct_eval_rmse"])
                    if direct_rmse < best_direct_rmse:
                        best_direct_rmse = direct_rmse
                        best_direct_state = {
                            "U_raw": U_raw.detach().clone(),
                            "R_mu": R_mu.detach().clone(),
                            "R_log_std": R_log_std.detach().clone(),
                        }
            row = {
                "step": float(step),
                "loss": loss_value,
                "data_objective": float(data_objective.detach().cpu().item()),
                "kl_per_anchor": float(kl.detach().cpu().item()),
                "beta_kl": float(beta_kl),
                "trace_penalty": float(trace_penalty.detach().cpu().item()),
                "mean_n_inliers": float(reward_diag["mean_n_inliers"].detach().cpu().item()),
                "fallback_count": float(reward_diag["fallback_count"].detach().cpu().item()),
            }
            row.update(moment_diag)
            row.update(eval_diag)
            history.append(row)
    if str(config.restore_state) == "best_loss" and best_state is not None:
        U_raw.data.copy_(best_state["U_raw"])
        R_mu.data.copy_(best_state["R_mu"])
        R_log_std.data.copy_(best_state["R_log_std"])
    elif str(config.restore_state) == "best_direct" and best_direct_state is not None:
        U_raw.data.copy_(best_direct_state["U_raw"])
        R_mu.data.copy_(best_direct_state["R_mu"])
        R_log_std.data.copy_(best_direct_state["R_log_std"])
    U_final = _orthonormal_columns(U_raw).detach()
    W_mu_final, _W_cov_final, eta_mu_final, eta_var_final, moment_diag = structured_metric_w_moments_from_R(
        U_final,
        X_anchor_t,
        inducing_X,
        R_mu.detach(),
        R_log_std.detach(),
        lengthscale=w_lengthscale,
        signal_var=w_signal_var,
        include_conditional_residual=bool(config.include_conditional_residual),
        trace_target=trace_target,
        trace_normalize=bool(config.trace_normalize),
        jitter=float(config.jitter),
    )
    diagnostics = {
        "best_loss": float(best_loss),
        "trace_target": float(trace_target),
        "trace_normalize": bool(config.trace_normalize),
        "restore_state": str(config.restore_state),
        "best_direct_rmse": float(best_direct_rmse),
        "final_W_row_norm_mean": float(torch.linalg.norm(W_mu_final, dim=-1).mean().cpu().item()),
        "R_n_inducing": int(M),
        "R_w_lengthscale": float(w_lengthscale),
        "R_w_signal_var": float(w_signal_var),
        "R_std_median": float(R_log_std.detach().exp().median().cpu().item()),
        "include_conditional_residual": bool(config.include_conditional_residual),
    }
    diagnostics.update(moment_diag)
    return StructuredMetricLMJGPResult(
        U=U_final,
        R_mu=R_mu.detach().clone(),
        R_log_std=R_log_std.detach().clone(),
        inducing_X=inducing_X.detach().clone(),
        eta_mu=eta_mu_final.detach().clone(),
        eta_log_std=eta_var_final.clamp_min(1e-12).sqrt().log().detach().clone(),
        w_lengthscale=float(w_lengthscale),
        w_signal_var=float(w_signal_var),
        include_conditional_residual=bool(config.include_conditional_residual),
        history=history,
        train_sec=float(time.perf_counter() - t0),
        diagnostics=diagnostics,
    )


def train_analytic_structured_metric_lmjgp(
    X_anchor: torch.Tensor | np.ndarray,
    X_neighbors: torch.Tensor | np.ndarray,
    y_neighbors: torch.Tensor | np.ndarray,
    C: torch.Tensor | np.ndarray,
    *,
    init_W: torch.Tensor | np.ndarray,
    init_mu_u: torch.Tensor | np.ndarray | None = None,
    config: AnalyticStructuredMetricLMJGPConfig,
    eval_callback: AnalyticEvalCallback | None = None,
) -> AnalyticStructuredMetricLMJGPResult:
    """Train analytic ISM-LMJGP using marginalised W and fixed local inducing locations."""
    t0 = time.perf_counter()
    X_anchor_t = _as_float_tensor(X_anchor)
    device = X_anchor_t.device
    dtype = X_anchor_t.dtype
    X_neighbors_t = _as_float_tensor(X_neighbors, device=device).to(dtype=dtype)
    y_t = _as_float_tensor(y_neighbors, device=device).to(dtype=dtype)
    if y_t.dim() == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    C_t = _as_float_tensor(C, device=device).to(dtype=dtype)
    T, _n, _D = X_neighbors_t.shape
    m = C_t.shape[1]
    W0 = _as_float_tensor(init_W, device=device).to(dtype=dtype)
    Q = W0.shape[0]
    trace_target = float(Q if config.trace_target is None else config.trace_target)
    U0, eta0, eta_log_std0 = init_structured_metric_from_w0(
        W0,
        n_anchors=T,
        trace_target=trace_target,
        eta_init_log_std=float(config.eta_init_log_std),
    )
    inducing_X = _select_inducing_X(X_anchor_t, config.n_inducing_R).to(device=device, dtype=dtype)
    M = int(inducing_X.shape[0])
    R_mu = torch.nn.Parameter(eta0[:1].expand(M, Q).contiguous().to(device=device, dtype=dtype).clone())
    R_log_std = torch.nn.Parameter(
        eta_log_std0[:1].expand(M, Q).contiguous().to(device=device, dtype=dtype).clone()
    )
    R_log_std.requires_grad_(bool(config.train_eta_log_std))
    U_raw = torch.nn.Parameter(U0.to(device=device, dtype=dtype).clone())
    if init_mu_u is None:
        mu_u0 = torch.zeros((T, m), device=device, dtype=dtype)
    else:
        mu_u0 = _as_float_tensor(init_mu_u, device=device).to(dtype=dtype)
    mu_u = torch.nn.Parameter(mu_u0.clone())
    u_log_std = torch.nn.Parameter(torch.full((T, m), -2.0, device=device, dtype=dtype))
    noise0 = max(float(config.obs_noise_init) ** 2, 1e-8)
    noise_max_logvar = math.log(max(float(config.obs_noise_max), 1e-3) ** 2)
    if str(config.obs_noise_prior_mode) == "data_driven":
        # Per-anchor nearest-neighbour-difference (Rice) estimate of the local
        # residual scale in the W0-projected neighbour space; this reflects
        # out-of-sample residuals, unlike the in-sample local-GP fit which the
        # model interpolates to ~0.
        with torch.no_grad():
            z_nb = torch.einsum("qd,tnd->tnq", W0, X_neighbors_t)
            d2_nb = (z_nb.unsqueeze(2) - z_nb.unsqueeze(1)).pow(2).sum(-1)
            big = torch.eye(z_nb.shape[1], device=device, dtype=dtype).unsqueeze(0) * 1e12
            nn_idx = (d2_nb + big).argmin(dim=2)
            y_nn = torch.gather(y_t, 1, nn_idx)
            sig_t = (0.5 * (y_t - y_nn).pow(2).mean(dim=1)).clamp_min(1e-8).sqrt()
            sig_t = (sig_t * float(config.obs_noise_data_driven_frac)).clamp(
                float(config.obs_noise_data_driven_floor), float(config.obs_noise_max)
            )
        prior_log_noise = sig_t.pow(2).log().to(device=device, dtype=dtype)
    else:
        prior_log_noise = torch.full((T,), math.log(noise0), device=device, dtype=dtype)
    init_log_noise = prior_log_noise.clone() if bool(config.init_noise_at_prior) else torch.full(
        (T,), math.log(noise0), device=device, dtype=dtype
    )
    log_noise_var = torch.nn.Parameter(init_log_noise.clone())
    rho0 = min(max(float(config.rho_init), 1e-4), 1.0 - 1e-4)
    rho_logit = torch.nn.Parameter(
        torch.full((T,), math.log(rho0 / (1.0 - rho0)), device=device, dtype=dtype)
    )
    params: list[torch.nn.Parameter] = [U_raw, R_mu, mu_u, u_log_std, log_noise_var, rho_logit]
    if R_log_std.requires_grad:
        params.append(R_log_std)
    opt = torch.optim.Adam(params, lr=float(config.lr))
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(config.seed))
    w_lengthscale = (
        _default_sparse_gp_lengthscale(X_anchor_t)
        if config.w_lengthscale is None
        else float(config.w_lengthscale)
    )
    w_signal_var = float(config.w_signal_var) * float(config.eta_prior_std) ** 2
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_direct_rmse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_direct_state: dict[str, torch.Tensor] | None = None
    prior_rho_logit = torch.tensor(math.log(rho0 / (1.0 - rho0)), device=device, dtype=dtype)
    for step in range(1, int(config.steps) + 1):
        opt.zero_grad()
        B = min(int(config.batch_anchors), T)
        idx = torch.randperm(T, device=device, generator=gen)[:B]
        U = _orthonormal_columns(U_raw)
        W_mu_b, W_cov_b, _eta_mu_b, _eta_var_b, _moment_diag_b = structured_metric_w_moments_from_R(
            U,
            X_anchor_t[idx],
            inducing_X,
            R_mu,
            R_log_std,
            lengthscale=w_lengthscale,
            signal_var=w_signal_var,
            include_conditional_residual=bool(config.include_conditional_residual),
            trace_target=trace_target,
            trace_normalize=bool(config.trace_normalize),
            jitter=float(config.jitter),
        )
        region_elbo_b, kl_u_b, local_diag = _analytic_local_terms(
            X_anchor_t[idx],
            X_neighbors_t[idx],
            y_t[idx],
            C_t[idx],
            W_mu_b,
            W_cov_b,
            mu_u[idx],
            u_log_std[idx],
            log_noise_var[idx],
            rho_logit[idx],
            local_signal_var=float(config.local_signal_var),
            outlier_scale_mult=float(config.outlier_scale_mult),
            jitter=float(config.jitter),
        )
        data_objective = region_elbo_b.mean()
        kl_u_mean = kl_u_b.mean()
        beta_R = float(config.beta_kl_R) * min(1.0, step / max(1, int(config.kl_warmup_steps)))
        kl_R = _diag_sparse_gp_kl(
            inducing_X,
            R_mu.unsqueeze(-1),
            R_log_std.unsqueeze(-1),
            lengthscale=w_lengthscale,
            signal_var=w_signal_var,
            jitter=float(config.jitter),
        ) / float(T)
        beta_u = float(config.beta_kl_u)
        eta_mu_all, eta_var_all = structured_metric_eta_moments_from_R(
            X_anchor_t,
            inducing_X,
            R_mu,
            R_log_std,
            lengthscale=w_lengthscale,
            signal_var=w_signal_var,
            include_conditional_residual=bool(config.include_conditional_residual),
            jitter=float(config.jitter),
        )
        trace_penalty = (_eta_trace_from_var(eta_mu_all, eta_var_all) - trace_target).pow(2).mean()
        noise_prior = ((log_noise_var[idx] - prior_log_noise[idx]) / max(float(config.obs_noise_prior_std), 1e-8)).pow(2).mean()
        rho = torch.sigmoid(rho_logit[idx]).clamp(1e-4, 1.0 - 1e-4)
        rho_prior = (rho_logit[idx] - prior_rho_logit).pow(2).mean()
        min_rho_penalty = torch.relu(float(config.min_rho_mean) - rho.mean()).pow(2)
        loss = (
            -data_objective
            + beta_R * kl_R
            + beta_u * kl_u_mean
            + float(config.obs_noise_prior_weight) * noise_prior
            + float(config.rho_prior_strength) * rho_prior
            + float(config.min_rho_penalty_weight) * min_rho_penalty
            + float(config.trace_penalty_weight) * trace_penalty
        )
        loss.backward()
        if float(config.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip_norm))
        opt.step()
        with torch.no_grad():
            R_log_std.clamp_(float(config.r_log_std_min), 2.0)
            u_log_std.clamp_(-8.0, 2.0)
            log_noise_var.clamp_(math.log(1e-6), noise_max_logvar)
        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {name: tensor.detach().clone() for name, tensor in {
                "U_raw": U_raw,
                "R_mu": R_mu,
                "R_log_std": R_log_std,
                "mu_u": mu_u,
                "u_log_std": u_log_std,
                "log_noise_var": log_noise_var,
                "rho_logit": rho_logit,
            }.items()}
        if step == 1 or step % int(config.log_interval) == 0 or step == int(config.steps):
            with torch.no_grad():
                U_eval = _orthonormal_columns(U_raw)
                W_mu_all, W_cov_all, _eta_mu_eval, _eta_var_eval, moment_diag = structured_metric_w_moments_from_R(
                    U_eval,
                    X_anchor_t,
                    inducing_X,
                    R_mu,
                    R_log_std,
                    lengthscale=w_lengthscale,
                    signal_var=w_signal_var,
                    include_conditional_residual=bool(config.include_conditional_residual),
                    trace_target=trace_target,
                    trace_normalize=bool(config.trace_normalize),
                    jitter=float(config.jitter),
                )
            eval_diag: dict[str, float] = {}
            if eval_callback is not None and (
                step == 1 or step % max(1, int(config.eval_interval)) == 0 or step == int(config.steps)
            ):
                eval_diag = eval_callback(
                    step,
                    {"W_mu_anchor": W_mu_all.detach(), "W_cov_anchor": W_cov_all.detach()},
                    {
                        "U": U_eval.detach(),
                        "R_mu": R_mu.detach(),
                        "R_log_std": R_log_std.detach(),
                        "inducing_X": inducing_X.detach(),
                        "mu_u": mu_u.detach(),
                        "u_log_std": u_log_std.detach(),
                        "log_noise_var": log_noise_var.detach(),
                        "rho_logit": rho_logit.detach(),
                        "C": C_t.detach(),
                        "w_lengthscale": torch.as_tensor(float(w_lengthscale), device=device, dtype=dtype),
                        "w_signal_var": torch.as_tensor(float(w_signal_var), device=device, dtype=dtype),
                        "include_conditional_residual": torch.as_tensor(
                            1.0 if bool(config.include_conditional_residual) else 0.0,
                            device=device,
                            dtype=dtype,
                        ),
                    },
                    moment_diag,
                )
                if "direct_eval_rmse" in eval_diag:
                    direct_rmse = float(eval_diag["direct_eval_rmse"])
                    if direct_rmse < best_direct_rmse:
                        best_direct_rmse = direct_rmse
                        best_direct_state = {name: tensor.detach().clone() for name, tensor in {
                            "U_raw": U_raw,
                            "R_mu": R_mu,
                            "R_log_std": R_log_std,
                            "mu_u": mu_u,
                            "u_log_std": u_log_std,
                            "log_noise_var": log_noise_var,
                            "rho_logit": rho_logit,
                        }.items()}
            row = {
                "step": float(step),
                "loss": loss_value,
                "data_objective": float(data_objective.detach().cpu().item()),
                "kl_R_per_anchor": float(kl_R.detach().cpu().item()),
                "beta_kl_R": float(beta_R),
                "kl_u_mean": float(local_diag["kl_u_mean"].cpu().item()),
                "noise_prior": float(noise_prior.detach().cpu().item()),
                "rho_prior": float(rho_prior.detach().cpu().item()),
                "min_rho_penalty": float(min_rho_penalty.detach().cpu().item()),
                "trace_penalty": float(trace_penalty.detach().cpu().item()),
                "rho_mean": float(local_diag["rho_mean"].cpu().item()),
                "noise_std_mean": float(local_diag["noise_std_mean"].cpu().item()),
            }
            row.update(moment_diag)
            row.update(eval_diag)
            history.append(row)
    restore = best_state if str(config.restore_state) == "best_loss" else None
    if str(config.restore_state) == "best_direct":
        restore = best_direct_state
    if restore is not None:
        U_raw.data.copy_(restore["U_raw"])
        R_mu.data.copy_(restore["R_mu"])
        R_log_std.data.copy_(restore["R_log_std"])
        mu_u.data.copy_(restore["mu_u"])
        u_log_std.data.copy_(restore["u_log_std"])
        log_noise_var.data.copy_(restore["log_noise_var"])
        rho_logit.data.copy_(restore["rho_logit"])
    U_final = _orthonormal_columns(U_raw).detach()
    _W_mu_final, _W_cov_final, _eta_mu_final, _eta_var_final, moment_diag = structured_metric_w_moments_from_R(
        U_final,
        X_anchor_t,
        inducing_X,
        R_mu.detach(),
        R_log_std.detach(),
        lengthscale=w_lengthscale,
        signal_var=w_signal_var,
        include_conditional_residual=bool(config.include_conditional_residual),
        trace_target=trace_target,
        trace_normalize=bool(config.trace_normalize),
        jitter=float(config.jitter),
    )
    diagnostics = {
        "best_loss": float(best_loss),
        "best_direct_rmse": float(best_direct_rmse),
        "trace_target": float(trace_target),
        "trace_normalize": bool(config.trace_normalize),
        "restore_state": str(config.restore_state),
        "R_n_inducing": int(M),
        "R_w_lengthscale": float(w_lengthscale),
        "R_w_signal_var": float(w_signal_var),
        "R_std_median": float(R_log_std.detach().exp().median().cpu().item()),
        "rho_mean": float(torch.sigmoid(rho_logit.detach()).mean().cpu().item()),
        "noise_std_mean": float(log_noise_var.detach().exp().sqrt().mean().cpu().item()),
        "noise_prior_anchor_median": float(prior_log_noise.detach().exp().sqrt().median().cpu().item()),
        "obs_noise_prior_mode": str(config.obs_noise_prior_mode),
        "local_m_inducing": int(m),
    }
    diagnostics.update(moment_diag)
    return AnalyticStructuredMetricLMJGPResult(
        U=U_final,
        R_mu=R_mu.detach().clone(),
        R_log_std=R_log_std.detach().clone(),
        inducing_X=inducing_X.detach().clone(),
        mu_u=mu_u.detach().clone(),
        u_log_std=u_log_std.detach().clone(),
        log_noise_var=log_noise_var.detach().clone(),
        rho_logit=rho_logit.detach().clone(),
        C=C_t.detach().clone(),
        w_lengthscale=float(w_lengthscale),
        w_signal_var=float(w_signal_var),
        local_signal_var=float(config.local_signal_var),
        include_conditional_residual=bool(config.include_conditional_residual),
        history=history,
        train_sec=float(time.perf_counter() - t0),
        diagnostics=diagnostics,
    )


def predict_analytic_structured_metric_lmjgp(
    X_anchor: torch.Tensor | np.ndarray,
    result: AnalyticStructuredMetricLMJGPResult,
    *,
    trace_target: float,
    trace_normalize: bool,
    mixture: bool = False,
    outlier_mean: torch.Tensor | np.ndarray | None = None,
    outlier_var: torch.Tensor | np.ndarray | None = None,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Direct VI prediction from an analytic ISM-LMJGP result.

    By default this returns the *inlier* GP predictive ``N(mu_f, var_f + noise)``
    for the anchor's own response. The ``rho`` / outlier component models
    contaminated neighbours inside the local composite likelihood during
    training; mixing a broad background into the predictive of the test point's
    response only inflates the intervals (heavy over-coverage). Set
    ``mixture=True`` (and pass ``outlier_mean``/``outlier_var`` matching the
    training background) only if a contaminated predictive is explicitly wanted.
    """
    Xa = _as_float_tensor(X_anchor, device=result.U.device).to(dtype=result.U.dtype)
    sigma2 = float(getattr(result, "local_signal_var", 1.0))
    W_mu, W_cov, _eta_mu, _eta_var, _moment_diag = structured_metric_w_moments_from_R(
        result.U,
        Xa,
        result.inducing_X,
        result.R_mu,
        result.R_log_std,
        lengthscale=float(result.w_lengthscale),
        signal_var=float(result.w_signal_var),
        include_conditional_residual=bool(result.include_conditional_residual),
        trace_target=float(trace_target),
        trace_normalize=bool(trace_normalize),
        jitter=float(jitter),
    )
    Xq = Xa.unsqueeze(1)
    sigma_f = torch.full((Xa.shape[0],), math.sqrt(sigma2), device=Xa.device, dtype=Xa.dtype)
    m_W = _expected_Kfu_structured(W_mu, W_cov, Xq, result.C, sigma_f).squeeze(1)
    S_W = _expected_KufKfu_structured(W_mu, W_cov, Xq, result.C, sigma_f).squeeze(1)
    m = result.C.shape[1]
    d2 = (result.C.unsqueeze(2) - result.C.unsqueeze(1)).pow(2).sum(-1)
    Kuu = sigma2 * torch.exp(-0.5 * d2) + float(jitter) * torch.eye(m, device=Xa.device, dtype=Xa.dtype).unsqueeze(0)
    Luu = torch.linalg.cholesky(Kuu)
    Kuu_inv = torch.cholesky_inverse(Luu)
    u_var = torch.exp(2.0 * result.u_log_std).clamp_min(1e-12)
    u_second = torch.diag_embed(u_var) + torch.einsum("ti,tj->tij", result.mu_u, result.mu_u)
    alpha = torch.einsum("tij,tj->ti", Kuu_inv, result.mu_u)
    mu_f = (m_W * alpha).sum(dim=1)
    v1 = sigma2 - torch.einsum("tij,tij->t", Kuu_inv, S_W)
    t3 = torch.einsum("tij,tjk,tkm,tmi->t", Kuu_inv, S_W, Kuu_inv, u_second)
    var_f = (v1 + t3 - mu_f.pow(2)).clamp_min(1e-10)
    noise_var = result.log_noise_var.exp().clamp_min(1e-8)
    rho = torch.sigmoid(result.rho_logit).clamp(1e-4, 1.0 - 1e-4)
    if not mixture:
        mu_y = mu_f
        var_y = (var_f + noise_var).clamp_min(1e-10)
    else:
        if outlier_mean is None:
            out_mu = torch.zeros_like(mu_f)
        else:
            out_mu = _as_float_tensor(outlier_mean, device=Xa.device).to(dtype=Xa.dtype).reshape_as(mu_f)
        if outlier_var is None:
            out_var = torch.ones_like(mu_f)
        else:
            out_var = _as_float_tensor(outlier_var, device=Xa.device).to(dtype=Xa.dtype).reshape_as(mu_f).clamp_min(1e-8)
        mu_y = rho * mu_f + (1.0 - rho) * out_mu
        second_y = rho * (var_f + noise_var + mu_f.pow(2)) + (1.0 - rho) * (out_var + out_mu.pow(2))
        var_y = (second_y - mu_y.pow(2)).clamp_min(1e-10)
    diag = {
        "rho_mean": float(rho.mean().detach().cpu().item()),
        "noise_std_mean": float(noise_var.sqrt().mean().detach().cpu().item()),
        "var_f_mean": float(var_f.mean().detach().cpu().item()),
    }
    return mu_y, var_y, diag


__all__ = [
    "AnalyticStructuredMetricLMJGPConfig",
    "AnalyticStructuredMetricLMJGPResult",
    "StructuredMetricLMJGPConfig",
    "StructuredMetricLMJGPResult",
    "build_fixed_local_inducing_from_w0",
    "init_structured_metric_from_w0",
    "predict_analytic_structured_metric_lmjgp",
    "sample_structured_metric_w",
    "sample_structured_metric_w_from_eta_moments",
    "structured_metric_eta_moments_from_R",
    "structured_metric_w_moments",
    "structured_metric_w_moments_from_R",
    "structured_metric_w_moments_from_eta_moments",
    "train_analytic_structured_metric_lmjgp",
    "train_structured_metric_lmjgp",
]

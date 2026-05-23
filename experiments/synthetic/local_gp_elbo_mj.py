"""Local GP inlier quadratic with explicit region mean ``m_j``.

Model: ``y_i = m_j + sigma_k * g(W x_i) + eps``, ``g ~ GP(0, C)``.

Uses amplitude-included ``expected_Kfu`` / ``expected_KufKfu`` (same as current code).

[
Q_{j,i} = ((y_i - m_j)^2 - 2(y_i - m_j) zeta_{j,i} + V_{1} + B_{j,i}) / (2 sigma_n^2)
]

where ``zeta = Psi1_amp @ K^{-1} mu_u`` is the mean of the *residual* ``f - m_j``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from djgp.gate_fixed_slope import expected_log_sigmoid_gate, expected_sigmoid_gate, gate_diagnostics
from djgp.structured_projection import (
    expected_Kfu_mc,
    get_mu_W_cov_W,
    kl_projection,
)
from djgp.variational import (
    expected_Kfu,
    expected_KufKfu,
    kl_q_u_batch,
    kl_qp,
    qW_from_qV,
)

_LOG_2PI = math.log(2 * math.pi)


def _psd_sigma_u(u_params: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    eps = 1e-6
    raw = torch.stack([u["Sigma_u"] for u in u_params], dim=0)
    L = torch.tril(raw)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    pos_diag = F.softplus(diag) + eps
    L = L - torch.diag_embed(diag) + torch.diag_embed(pos_diag)
    return L @ L.transpose(-1, -2)


def stack_m_j(u_params: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    """Per-region local mean ``m_j``, shape ``[T]``."""
    return torch.stack([u["m_j"] for u in u_params], dim=0).view(-1)


def gp_residual_moments(
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    proj: dict | None = None,
    fixed_w: bool = False,
    at_test: bool = False,
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
) -> dict[str, torch.Tensor]:
    """Compute ``zeta`` (E[f-m_j]), ``V1``, ``B`` and kernel pieces.

    If ``at_test``, ``X`` is ``X_test`` with shape ``[T, 1, D]``; else training neighbours.
    """
    device = regions[0]["X"].device
    if at_test:
        X = hyperparams["X_test"].unsqueeze(1)
    else:
        X = torch.stack([r["X"] for r in regions], dim=0)
    C = torch.stack([r["C"] for r in regions], dim=0)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(-1)
    Sigma_u = _psd_sigma_u(u_params)

    if proj is not None:
        mu_W, cov_W = get_mu_W_cov_W(proj, hyperparams["X_test"], hyperparams)
    else:
        mu_W, cov_W = qW_from_qV(
            hyperparams["X_test"],
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
    if fixed_w:
        cov_W = torch.zeros_like(cov_W)

    if use_mc_kfu:
        Kfu = expected_Kfu_mc(mu_W, cov_W, X, C, sigma_k, n_samples=mc_samples)
        KufKfu = expected_KufKfu(mu_W, cov_W, X, C, sigma_k)
    else:
        Kfu = expected_Kfu(mu_W, cov_W, X, C, sigma_k)
        KufKfu = expected_KufKfu(mu_W, cov_W, X, C, sigma_k)

    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    Kuu = torch.exp(-0.5 * d2)
    Kuu_inv = torch.cholesky_inverse(torch.linalg.cholesky(Kuu))

    V1 = sigma_k.view(-1, 1) ** 2 - torch.einsum("tij,tnji->tn", Kuu_inv, KufKfu)
    v = torch.einsum("tij,tj->ti", Kuu_inv, mu_u)
    zeta = torch.einsum("tni,ti->tn", Kfu, v)

    Sigma_u_tilde = Sigma_u + torch.einsum("ti,tj->tij", mu_u, mu_u)
    B = torch.einsum("tij,tnjk,tkm,tmi->tn", Kuu_inv, KufKfu, Kuu_inv, Sigma_u_tilde)

    return {
        "X": X,
        "Kfu": Kfu,
        "KufKfu": KufKfu,
        "Kuu_inv": Kuu_inv,
        "zeta": zeta,
        "V1": V1,
        "B": B,
        "mu_W": mu_W,
        "cov_W": cov_W,
    }


def inlier_quad_mj(
    y: torch.Tensor,
    m_j: torch.Tensor,
    zeta: torch.Tensor,
    V1: torch.Tensor,
    B: torch.Tensor,
    sigma_noise: torch.Tensor,
) -> torch.Tensor:
    """``Q_{j,i}`` with explicit ``m_j``; shapes ``y,zeta,V1,B``: ``[T,n]``, ``m_j``: ``[T]``."""
    y_res = y - m_j.view(-1, 1)
    return (y_res.pow(2) - 2.0 * y_res * zeta + V1 + B) / (
        2.0 * sigma_noise.view(-1, 1).pow(2)
    )


def sigma_n_learnable(spec: dict, step: int) -> bool:
    """Whether ``sigma_noise`` is optimized this step."""
    if spec.get("sigma_n_fixed") is not None:
        freeze = int(spec.get("sigma_n_freeze_steps", 0) or 0)
        return freeze > 0 and step > freeze
    return True


def apply_sigma_n_control(u_params, spec: dict, step: int) -> None:
    """Fixed / warmup / cap ``sigma_noise`` after each optimizer step."""
    sn_fix = spec.get("sigma_n_fixed")
    freeze = int(spec.get("sigma_n_freeze_steps", 0) or 0)
    sn_max = spec.get("sigma_n_max")
    for u in u_params:
        if sn_fix is not None and (freeze <= 0 or step <= freeze):
            u["sigma_noise"].fill_(float(sn_fix))
        else:
            u["sigma_noise"].clamp_(min=0.1)
            if sn_max is not None:
                u["sigma_noise"].clamp_(max=float(sn_max))


def init_sigma_n_from_spec(u_params, spec: dict) -> None:
    """Set initial ``sigma_noise`` and ``requires_grad`` for fixed / warmup variants."""
    sn_fix = spec.get("sigma_n_fixed")
    freeze = int(spec.get("sigma_n_freeze_steps", 0) or 0)
    learn = sn_fix is None or freeze > 0
    for u in u_params:
        if sn_fix is not None:
            u["sigma_noise"] = torch.tensor(float(sn_fix), device=u["sigma_noise"].device)
        if learn:
            u["sigma_noise"] = u["sigma_noise"].detach().clone().requires_grad_(True)
        else:
            u["sigma_noise"] = u["sigma_noise"].detach().clone().requires_grad_(False)


def training_beta_u_kl(spec: dict, step: int, num_steps: int) -> float:
    """Effective ``beta_u`` on ``K^{-1}``-related KL terms (tempered / warmup)."""
    if "beta_u_kl" in spec:
        return float(spec["beta_u_kl"])
    if spec.get("beta_u_warmup"):
        start = float(spec.get("beta_u_warmup_start", 0.1))
        end = float(spec.get("beta_u_warmup_end", 1.0))
        frac = (step - 1) / max(num_steps - 1, 1)
        return start + (end - start) * frac
    return 1.0


def kl_q_u_batch_tempered(
    Z: torch.Tensor,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    u_var: torch.Tensor,
    *,
    beta_u: float = 1.0,
) -> torch.Tensor:
    """``KL(q(u)||p(u))`` with ``beta_u`` on ``tr(K^{-1}Sigma)`` and ``mu^T K^{-1} mu``."""
    T, m1, Q = Z.shape
    d2 = (Z.unsqueeze(2) - Z.unsqueeze(1)).pow(2).sum(dim=-1)
    u_var_T = u_var.view(-1, 1, 1)
    K = u_var_T * torch.exp(-0.5 * d2)
    L = torch.linalg.cholesky(K)
    K_inv = torch.cholesky_inverse(L)
    logdet_K = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=1)
    trace_term = torch.einsum("tij,tji->t", K_inv, Sigma)
    Kinv_mu = torch.einsum("tij,tj->ti", K_inv, mu)
    quad_term = (mu * Kinv_mu).sum(dim=1)
    m1 = Sigma.size(-1)
    eye = torch.eye(m1, dtype=Sigma.dtype, device=Sigma.device)
    Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2)) + 1e-6 * eye
    sign, logdet_Sigma = torch.linalg.slogdet(Sigma)
    assert (sign > 0).all()
    bu = float(beta_u)
    return 0.5 * (bu * (trace_term + quad_term) - m1 + logdet_K - logdet_Sigma)


def compute_elbo_pure_inlier_mj(
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    proj: dict | None = None,
    fixed_w: bool = False,
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
    beta_u_kl: float = 1.0,
    lambda_sigma_prior: float = 0.0,
    log_sigma_n0: float = 0.0,
) -> torch.Tensor:
    T = len(regions)
    device = regions[0]["X"].device
    y = torch.stack([r["y"] for r in regions], dim=0)
    C = torch.stack([r["C"] for r in regions], dim=0)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)
    m_j = stack_m_j(u_params)
    Sigma_u = _psd_sigma_u(u_params)

    if proj is not None:
        KL_V = kl_projection(
            proj, hyperparams["Z"], hyperparams["lengthscales"], hyperparams["var_w"],
        )
    else:
        from djgp.variational import kl_qp

        KL_V = kl_qp(
            hyperparams["Z"], V_params["mu_V"], V_params["sigma_V"],
            hyperparams["lengthscales"], hyperparams["var_w"],
        )
    mom = gp_residual_moments(
        regions, V_params, u_params, hyperparams,
        proj=proj, fixed_w=fixed_w, at_test=False,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    quad = inlier_quad_mj(y, m_j, mom["zeta"], mom["V1"], mom["B"], sigma_noise)
    loglik = (
        -0.5 * _LOG_2PI
        - 0.5 * torch.log(sigma_noise.view(-1, 1) ** 2)
        - quad
    ).sum()
    if lambda_sigma_prior > 0.0:
        log_sn = torch.log(sigma_noise.clamp_min(1e-6))
        loglik = loglik - float(lambda_sigma_prior) * ((log_sn - float(log_sigma_n0)) ** 2).sum()
    KL_u = kl_q_u_batch_tempered(
        C, mu_u, Sigma_u, torch.tensor(1.0, device=device), beta_u=beta_u_kl,
    ).sum()
    return (loglik - KL_V - KL_u) / T


def mixture_scores_rho_mj(
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    proj: dict | None = None,
    gate_config: dict | None = None,
    fixed_w: bool = False,
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable ``T1``, ``T2``, posterior responsibility ``rho``; shapes ``[T,n]``."""
    device = regions[0]["X"].device
    C = torch.stack([r["C"] for r in regions], dim=0)
    Q = C.shape[-1]
    y = torch.stack([r["y"] for r in regions], dim=0)
    X = torch.stack([r["X"] for r in regions], dim=0)
    U = torch.sigmoid(torch.stack([u["U_logit"] for u in u_params], dim=0)).view(-1, 1)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    m_j = stack_m_j(u_params)
    gate_config = gate_config or {"gate_mode": "learned"}

    mom = gp_residual_moments(
        regions, V_params, u_params, hyperparams,
        proj=proj, fixed_w=fixed_w, at_test=False,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    quad = inlier_quad_mj(y, m_j, mom["zeta"], mom["V1"], mom["B"], sigma_noise)
    elog_sig, elog_one_minus = expected_log_sigmoid_gate(
        u_params, mom["mu_W"], mom["cov_W"], X, gate_config, Q,
    )
    T1 = -0.5 * _LOG_2PI - 0.5 * torch.log(sigma_noise.view(-1, 1) ** 2) + elog_sig - quad
    T2 = torch.log(U.clamp_min(1e-12)) + elog_one_minus
    log_rho = T1 - torch.logsumexp(torch.stack([T1, T2], dim=0), dim=0)
    rho = torch.exp(log_rho)
    return T1, T2, rho


def rho_anchor_penalty(rho: torch.Tensor, rho0: torch.Tensor) -> torch.Tensor:
    """Anchor ELBO ``rho`` toward fixed ``rho0`` (detached target)."""
    return (rho - rho0.detach()).pow(2).mean()


def rho_floor_penalty(
    rho: torch.Tensor,
    rho_min: float,
    *,
    per_region: bool = False,
) -> torch.Tensor:
    """``softplus(rho_min - rho_bar)^2``; mean over regions if ``per_region`` else global."""
    rho_min_t = torch.tensor(float(rho_min), device=rho.device, dtype=rho.dtype)
    if per_region:
        gap = F.softplus(rho_min_t - rho.mean(dim=-1))
        return gap.pow(2).mean()
    gap = F.softplus(rho_min_t - rho.mean())
    return gap.pow(2)


def init_u_from_spec(u_params, spec: dict) -> None:
    """Fix outlier uniform width ``u_j`` when ``fix_u`` is set."""
    u_fix = spec.get("fix_u")
    if u_fix is None:
        return
    p = float(u_fix)
    logit = math.log(p / (1.0 - p + 1e-12))
    for u in u_params:
        u["U_logit"] = torch.tensor(logit, device=u["U_logit"].device, requires_grad=False)


def compute_elbo_mixture_mj(
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    proj: dict | None = None,
    gate_config: dict | None = None,
    force_inlier_only: bool = False,
    fixed_w: bool = False,
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
    beta_u_kl: float = 1.0,
    lambda_sigma_prior: float = 0.0,
    log_sigma_n0: float = 0.0,
    rho_floor_min: float | None = None,
    lambda_rho_floor: float = 0.0,
    rho_floor_per_region: bool = False,
    rho0: torch.Tensor | None = None,
    lambda_rho_anchor: float = 0.0,
    rho_pointwise_min: float | None = None,
    lambda_rho_pointwise_floor: float = 0.0,
    rho_blend_alpha: float = 1.0,
    rho_point_blend_gamma: float | None = None,
    rho_blend_linear_data: bool = False,
) -> torch.Tensor:
    """Full mixture ELBO with ``m_j`` in the inlier ``Q`` term."""
    T = len(regions)
    device = regions[0]["X"].device
    C = torch.stack([r["C"] for r in regions], dim=0)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)
    Sigma_u = _psd_sigma_u(u_params)
    gate_config = gate_config or {"gate_mode": "learned"}

    if proj is not None:
        KL_V = kl_projection(
            proj, hyperparams["Z"], hyperparams["lengthscales"], hyperparams["var_w"],
        )
    else:
        KL_V = kl_qp(
            hyperparams["Z"], V_params["mu_V"], V_params["sigma_V"],
            hyperparams["lengthscales"], hyperparams["var_w"],
        )

    T1, T2, rho_elbo = mixture_scores_rho_mj(
        regions, V_params, u_params, hyperparams,
        proj=proj, gate_config=gate_config, fixed_w=fixed_w,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    if force_inlier_only:
        T2 = torch.full_like(T2, -80.0)

    from experiments.synthetic.mixture_rho_curriculum import compute_rho_tilde

    rho_train = compute_rho_tilde(
        rho_elbo, rho0,
        alpha=float(rho_blend_alpha),
        point_gamma=rho_point_blend_gamma,
    )

    if rho_blend_linear_data:
        data_term = (rho_train * T1 + (1.0 - rho_train) * T2).sum(dim=-1).sum()
    else:
        data_term = torch.logsumexp(torch.stack([T1, T2], dim=0), dim=0).sum(-1).sum()
    rho = rho_elbo
    elbo = data_term - KL_V

    if lambda_sigma_prior > 0.0:
        log_sn = torch.log(sigma_noise.clamp_min(1e-6))
        elbo = elbo - float(lambda_sigma_prior) * ((log_sn - float(log_sigma_n0)) ** 2).sum()

    if lambda_rho_floor > 0.0 and rho_floor_min is not None:
        elbo = elbo + float(lambda_rho_floor) * rho_floor_penalty(
            rho, float(rho_floor_min), per_region=rho_floor_per_region,
        )

    if lambda_rho_anchor > 0.0 and rho0 is not None:
        elbo = elbo - float(lambda_rho_anchor) * rho_anchor_penalty(rho, rho0)

    if lambda_rho_pointwise_floor > 0.0 and rho_pointwise_min is not None:
        rho_min_t = torch.tensor(float(rho_pointwise_min), device=rho.device, dtype=rho.dtype)
        gap = F.softplus(rho_min_t - rho)
        elbo = elbo + float(lambda_rho_pointwise_floor) * gap.pow(2).mean()

    KL_u = kl_q_u_batch_tempered(
        C, mu_u, Sigma_u, torch.tensor(1.0, device=device), beta_u=beta_u_kl,
    ).sum()
    return (elbo - KL_u) / T


@torch.no_grad()
def mixture_diagnostics_mj(
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    proj: dict | None = None,
    gate_config: dict | None = None,
    fixed_w: bool = False,
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
) -> dict[str, float]:
    """Mixture diagnostics with ``m_j`` inlier term and optional fixed gate slope."""
    device = hyperparams["Z"].device
    C = torch.stack([r["C"] for r in regions], dim=0)
    Q = C.shape[-1]
    gate_config = gate_config or {"gate_mode": "learned"}
    X = torch.stack([r["X"] for r in regions], dim=0)
    y = torch.stack([r["y"] for r in regions], dim=0)
    U = torch.sigmoid(torch.stack([u["U_logit"] for u in u_params], dim=0)).view(-1, 1)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    m_j = stack_m_j(u_params)

    mom = gp_residual_moments(
        regions, V_params, u_params, hyperparams,
        proj=proj, fixed_w=fixed_w, at_test=False,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    quad = inlier_quad_mj(y, m_j, mom["zeta"], mom["V1"], mom["B"], sigma_noise)
    elog_sig, elog_one_minus = expected_log_sigmoid_gate(
        u_params, mom["mu_W"], mom["cov_W"], X, gate_config, Q,
    )
    T_in = -0.5 * _LOG_2PI - 0.5 * torch.log(sigma_noise.view(-1, 1) ** 2) + elog_sig - quad
    T_out = torch.log(U.clamp_min(1e-12)) + elog_one_minus
    log_r_in = T_in - torch.logsumexp(torch.stack([T_in, T_out], dim=0), dim=0)
    r = torch.exp(log_r_in)

    gdiag = gate_diagnostics(
        u_params, mom["mu_W"], mom["cov_W"], X, hyperparams["X_test"], gate_config, Q,
    )
    zeta = mom["zeta"]
    y_res = y - m_j.view(-1, 1)
    return {
        **gdiag,
        "r_mean": float(r.mean().item()),
        "r_std": float(r.std().item()),
        "r_min": float(r.min().item()),
        "delta_T_mean": float((T_in - T_out).mean().item()),
        "T_in_mean": float(T_in.mean().item()),
        "T_out_mean": float(T_out.mean().item()),
        "U_mean": float(U.mean().item()),
        "std_zeta": float(zeta.std().item()),
        "ratio_zeta_yres": float(zeta.std().item() / max(y_res.std().item(), 1e-8)),
    }


@torch.no_grad()
def predict_inlier_mj(
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    proj: dict | None = None,
    fixed_w: bool = False,
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predictive mean/var of ``y`` at test anchors: ``m_j + zeta``, variance from residual GP."""
    device = hyperparams["Z"].device
    T = len(regions)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(T).clamp_min(1e-6)
    m_j = stack_m_j(u_params)

    mom = gp_residual_moments(
        regions, V_params, u_params, hyperparams,
        proj=proj, fixed_w=fixed_w, at_test=True,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    zeta = mom["zeta"].squeeze(1)
    mu_y = m_j + zeta
    var_y = (mom["V1"].squeeze(1) + mom["B"].squeeze(1) + sigma_noise.pow(2)).clamp_min(1e-12)
    return mu_y, var_y, zeta


def init_m_j_from_regions(
    regions,
    *,
    device: torch.device,
    init: str = "mean",
    learn: bool = True,
) -> list[torch.Tensor]:
    """Empirical ``m_j`` per region from neighbour ``y`` (before any centering)."""
    out: list[torch.Tensor] = []
    for r in regions:
        y_nb = r["y"]
        if init == "median":
            val = y_nb.median()
        elif init == "zero":
            val = torch.zeros((), device=device, dtype=y_nb.dtype)
        else:
            val = y_nb.mean()
        out.append(val.detach().clone().requires_grad_(learn))
    return out

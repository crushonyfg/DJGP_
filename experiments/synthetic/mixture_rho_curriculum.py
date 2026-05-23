"""Posterior responsibility anchoring / curriculum for mixture ELBO + local GP."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from djgp.gp_feature_alignment import (
    alignment_residual,
    compute_elbo_kernels,
    elbo_star_mu_u,
    ridge_coefficients,
)


def lambda_rho_anchor_schedule(spec: dict[str, Any], step: int, num_steps: int) -> float:
    """Anneal anchor strength: ``lambda_0 * (1 - t/T)_+``."""
    lam0 = float(spec.get("rho_anchor_lambda0", 0.0))
    if lam0 <= 0.0:
        return 0.0
    if not spec.get("rho_anchor_anneal", True):
        return lam0
    frac = max(0.0, 1.0 - float(step) / max(int(num_steps), 1))
    return lam0 * frac


def rho_blend_alpha(spec: dict[str, Any], step: int, num_steps: int) -> float:
    """Curriculum ``alpha_t`` for ``rho_tilde = (1-alpha)*rho0 + alpha*rho_elbo``.

    During pure warmup returns ``0`` (mixture not used). After ``rho_blend_end_step``
    stays at ``rho_blend_alpha_end``.
    """
    if use_pure_inlier_warmup(spec, step):
        return 0.0
    if spec.get("rho_point_blend_gamma") is not None:
        return 1.0
    if not spec.get("rho_blend", False):
        return 1.0
    alpha_end = float(spec.get("rho_blend_alpha_end", 1.0))
    t0 = int(spec.get("rho_blend_start_step", spec.get("rho_warmup_pure_steps", 0) + 1))
    t1 = min(int(spec.get("rho_blend_end_step", num_steps)), int(num_steps))
    if step < t0:
        return 0.0
    if step >= t1:
        return alpha_end
    return alpha_end * float(step - t0) / max(t1 - t0, 1)


def compute_rho_tilde(
    rho_elbo: torch.Tensor,
    rho0: torch.Tensor | None,
    *,
    alpha: float,
    point_gamma: float | None = None,
) -> torch.Tensor:
    """Training responsibility ``rho_tilde`` (differentiable w.r.t. ``rho_elbo``)."""
    if point_gamma is not None:
        g = float(point_gamma)
        return g + (1.0 - g) * rho_elbo
    if rho0 is None:
        return rho_elbo
    a = float(alpha)
    return (1.0 - a) * rho0.detach() + a * rho_elbo


def apply_rho_pointwise_floor(rho: torch.Tensor, rho_min: float) -> torch.Tensor:
    """``rho_min + (1 - rho_min) * rho``."""
    p = float(rho_min)
    return p + (1.0 - p) * rho


@torch.no_grad()
def init_rho0_uniform(
    regions,
    *,
    value: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Constant ``rho0``; shape ``[T, n]``."""
    y = torch.stack([r["y"] for r in regions], dim=0)
    dev = device or y.device
    return torch.full_like(y, float(value), device=dev)


@torch.no_grad()
def init_rho0_ridge(
    regions,
    proj: dict[str, Any],
    hyperparams: dict[str, Any],
    u_params,
    V_params,
    *,
    ridge_lambda: float = 1e-2,
    tau_scale: float = 1.0,
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
) -> torch.Tensor:
    """``rho0_i = sigmoid((c - e_i) / tau)`` with ``e_i = (r_i - zeta_ridge_i)^2``."""
    from experiments.synthetic.local_gp_elbo_mj import stack_m_j

    star = elbo_star_mu_u(
        regions, proj, hyperparams, u_params,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    phi = star["phi_amp"]
    r = star["r_elbo"].to(phi.dtype)
    b = ridge_coefficients(phi, r, ridge_lambda)
    z_ridge = torch.einsum("tnj,tj->tn", phi, b)
    err = (r - z_ridge).pow(2)
    c = err.median()
    tau = float(err.std().item()) * float(tau_scale) + 1e-6
    return torch.sigmoid((c - err) / tau)


@torch.no_grad()
def init_rho0_geom(
    regions,
    *,
    tau_scale: float = 1.0,
) -> torch.Tensor:
    """Higher ``rho`` for points closer to the region mean in ``X``."""
    X = torch.stack([r["X"] for r in regions], dim=0)
    center = X.mean(dim=1, keepdim=True)
    dist = (X - center).norm(dim=-1)
    d_med = dist.median()
    tau = float(dist.std().item()) * float(tau_scale) + 1e-6
    return torch.sigmoid((d_med - dist) / tau)


@torch.no_grad()
def build_rho0(
    regions,
    proj: dict[str, Any],
    hyperparams: dict[str, Any],
    u_params,
    V_params,
    spec: dict[str, Any],
) -> torch.Tensor | None:
    """Create detached ``rho0`` from ``spec['rho_init']``."""
    mode = spec.get("rho_init")
    if mode is None:
        return None
    mc_kw = dict(
        use_mc_kfu=bool(spec.get("use_mc_kfu")),
        mc_samples=int(spec.get("mc_samples", 5)),
    )
    if mode == "uniform":
        return init_rho0_uniform(
            regions, value=float(spec.get("rho_init_uniform_value", 1.0)),
        )
    if mode == "ridge":
        return init_rho0_ridge(
            regions, proj, hyperparams, u_params, V_params,
            ridge_lambda=float(spec.get("ridge_lambda", 1e-2)),
            tau_scale=float(spec.get("rho_init_tau_scale", 1.0)),
            **mc_kw,
        )
    if mode == "geom":
        return init_rho0_geom(
            regions, tau_scale=float(spec.get("rho_init_tau_scale", 1.0)),
        )
    raise ValueError(f"unknown rho_init={mode!r}")


def attach_rho0(hyperparams: dict[str, Any], rho0: torch.Tensor | None) -> None:
    if rho0 is None:
        hyperparams.pop("rho0", None)
    else:
        hyperparams["rho0"] = rho0.detach().clone()


def use_pure_inlier_warmup(spec: dict[str, Any], step: int) -> bool:
    warm = int(spec.get("rho_warmup_pure_steps", 0) or 0)
    return warm > 0 and step <= warm


def mixture_elbo_uses_mixture(spec: dict[str, Any], step: int) -> bool:
    if not spec.get("use_mixture_elbo"):
        return False
    return not use_pure_inlier_warmup(spec, step)


@torch.no_grad()
def rho0_quality_metrics(
    regions,
    proj: dict[str, Any],
    hyperparams: dict[str, Any],
    u_params,
    V_params,
    rho0: torch.Tensor,
    *,
    use_mc_kfu: bool = False,
    mc_samples: int = 5,
) -> dict[str, float]:
    """Quality of fixed anchor ``rho0``: star solve and correlations."""
    from djgp.gp_feature_alignment import (
        elbo_star_mu_u,
        elbo_star_mu_u_custom_rho,
        mixture_rho_signal_diagnostics,
    )

    star_pure = elbo_star_mu_u(
        regions, proj, hyperparams, u_params,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    star0 = elbo_star_mu_u_custom_rho(
        regions, proj, hyperparams, u_params, rho0,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    r = star_pure["r_elbo"]
    r_std = float(r.std().item()) + 1e-8
    sig = mixture_rho_signal_diagnostics(
        star_pure["phi_amp"], r, rho0, regions, proj, hyperparams, u_params,
        use_mc_kfu=use_mc_kfu, mc_samples=mc_samples,
    )
    b_ridge = ridge_coefficients(star_pure["phi_amp"], r.to(star_pure["phi_amp"].dtype), 1e-2)
    z_ridge = torch.einsum("tnj,tj->tn", star_pure["phi_amp"], b_ridge)
    rho_flat = rho0.reshape(-1).float()
    abs_r = r.reshape(-1).float().abs()
    zr_flat = z_ridge.reshape(-1).float().abs()

    def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
        xc = x - x.mean()
        yc = y - y.mean()
        den = xc.norm() * yc.norm()
        if float(den) < 1e-12:
            return float("nan")
        return float((xc * yc).sum() / den)

    B_pure = float(
        torch.einsum("tnj,tn->tj", star_pure["phi_amp"], r).norm(dim=-1).mean().item()
    )
    B_rho0 = float(
        torch.einsum("tnj,tn->tj", star_pure["phi_amp"], rho0.to(star_pure["phi_amp"].dtype) * r)
        .norm(dim=-1).mean().item()
    )
    return {
        "ratio_zeta_star_rho0_r": float(star0["zeta_star"].std().item() / r_std),
        "phiTr_rho0_over_pure": B_rho0 / max(B_pure, 1e-8),
        "corr_rho0_abs_r": _pearson(rho_flat, abs_r),
        "corr_rho0_abs_zridge": _pearson(rho_flat, zr_flat),
        **{f"rho0_{k}": v for k, v in sig.items() if k.startswith("phiTr_") or k.startswith("corr_rho")},
    }

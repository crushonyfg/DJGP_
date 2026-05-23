"""L2 projection + CEM gate diagnostics for q(R) / REINFORCE training."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from djgp.projections.uncertain_w_jgp import (
    FixedLabelGateConfig,
    fit_uncertain_gate_from_labels,
    sparse_gp_uncertain_w_moments_for_mode,
)


def _latent_anchor_delta(
    X_anchor: torch.Tensor,
    X_points: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """Anchor-conditioned projection ``z = W (x - x_a)``; shapes ``[n,Q]``."""
    delta = X_points - X_anchor.unsqueeze(0)
    return delta @ W.T


def _boundary_segment_in_box(
    gate: np.ndarray,
    z0_lim: tuple[float, float],
    z1_lim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip ``gate[0] + gate[1]*z0 + gate[2]*z1 = 0`` to the plot rectangle."""
    w0, w1, w2 = float(gate[0]), float(gate[1]), float(gate[2])
    z0a, z0b = float(z0_lim[0]), float(z0_lim[1])
    z1a, z1b = float(z1_lim[0]), float(z1_lim[1])
    if abs(w1) < 1e-10 and abs(w2) < 1e-10:
        return None

    pts: list[tuple[float, float]] = []

    def _in_box(z0: float, z1: float) -> bool:
        return z0a <= z0 <= z0b and z1a <= z1 <= z1b

    def _add(z0: float, z1: float) -> None:
        if np.isfinite(z0) and np.isfinite(z1) and _in_box(z0, z1):
            pts.append((float(z0), float(z1)))

    if abs(w2) > 1e-10:
        for z0 in (z0a, z0b):
            _add(z0, -(w0 + w1 * z0) / w2)
    if abs(w1) > 1e-10:
        for z1 in (z1a, z1b):
            _add(-(w0 + w2 * z1) / w1, z1)

    if len(pts) < 2:
        # Line misses the box (or is tangent): span along the line through the box center.
        z0c, z1c = 0.5 * (z0a + z0b), 0.5 * (z1a + z1b)
        if abs(w1) < 1e-10 and abs(w2) > 1e-10:
            z1 = np.clip(-w0 / w2, z1a, z1b)
            _add(z0a, z1)
            _add(z0b, z1)
        elif abs(w2) < 1e-10 and abs(w1) > 1e-10:
            z0 = np.clip(-w0 / w1, z0a, z0b)
            _add(z0, z1a)
            _add(z0, z1b)
        else:
            norm = float(np.hypot(w1, w2))
            t_dir = np.array([-w2, w1], dtype=np.float64) / norm
            half = 0.55 * max(z0b - z0a, z1b - z1a)
            p0 = (z0c - half * t_dir[0], z1c - half * t_dir[1])
            p1 = (z0c + half * t_dir[0], z1c + half * t_dir[1])
            _add(*p0)
            _add(*p1)
        if len(pts) < 2:
            return None

    uniq: list[tuple[float, float]] = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    if len(uniq) == 2:
        return np.array([uniq[0][0], uniq[1][0]]), np.array([uniq[0][1], uniq[1][1]])
    # >2 corners hit: keep the farthest pair.
    best_i, best_j, best_d = 0, 1, -1.0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            d = (uniq[i][0] - uniq[j][0]) ** 2 + (uniq[i][1] - uniq[j][1]) ** 2
            if d > best_d:
                best_d, best_i, best_j = d, i, j
    return (
        np.array([uniq[best_i][0], uniq[best_j][0]]),
        np.array([uniq[best_i][1], uniq[best_j][1]]),
    )


def regime_labels_from_phys(z_phys: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Two-regime labels for L2 phantom visualization.

    L2 phantom points use physical coords ``z`` with regimes separated near
  ``z_0 = 0`` (half-planes in the image).  Fall back to median split on ``y``
    when only one latent dimension is available.
    """
    z_phys = np.asarray(z_phys, dtype=np.float64)
    if z_phys.ndim == 1:
        z_phys = z_phys.reshape(1, -1)
    if z_phys.shape[1] >= 2:
        return (z_phys[:, 0] >= 0.0).astype(np.int32)
    if y is not None:
        med = float(np.median(y))
        return (np.asarray(y, dtype=np.float64).reshape(-1) >= med).astype(np.int32)
    return np.zeros(z_phys.shape[0], dtype=np.int32)


def fit_gate_for_projection(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    inlier_labels: torch.Tensor,
    W: torch.Tensor,
    *,
    gate_init: torch.Tensor | None = None,
    gate_steps: int = 6,
    gate_lr: float = 0.05,
    gate_l2: float = 1e-3,
    gh_points: int = 20,
) -> torch.Tensor:
    """Fit linear gate in the coordinate system of fixed deterministic ``W``.

    Uses fixed CEM inlier labels; only gate slopes/intercept are updated so the
    boundary ``h(z)=0`` matches the scatter under ``z = W(x - x_a)``.
    """
    W_mu = W.unsqueeze(0)
    W_var = torch.zeros_like(W_mu)
    X_a = X_anchor.reshape(1, -1)
    X_n = X_neighbors.unsqueeze(0)
    lab = inlier_labels.reshape(1, -1).to(dtype=X_n.dtype)
    g0 = None if gate_init is None else gate_init.reshape(1, -1)
    cfg = FixedLabelGateConfig(
        steps=int(gate_steps),
        lr=float(gate_lr),
        l2=float(gate_l2),
        basis="linear",
        gh_points=int(gh_points),
    )
    out = fit_uncertain_gate_from_labels(
        X_a,
        X_n,
        lab,
        mode="anchor",
        W_mu_anchor=W_mu,
        W_var_anchor=W_var,
        gate_init=g0,
        config=cfg,
    )
    return out["gate"][0]


@torch.no_grad()
def plot_l2_projection_panel(
    *,
    out_path: Path,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    W_plot: torch.Tensor,
    W0: torch.Tensor,
    gate: torch.Tensor | None,
    labels: np.ndarray | None,
    title: str,
    z_phys_anchor: np.ndarray | None = None,
    z_phys_neighbors: np.ndarray | None = None,
    sample_W_list: list[torch.Tensor] | None = None,
    draw_boundary: bool = True,
) -> None:
    """2D latent scatter + optional linear gate boundary ``h(z)=0``."""
    W_plot = W_plot.detach().cpu()
    W0 = W0.detach().cpu()
    z_nb = _latent_anchor_delta(X_anchor.cpu(), X_neighbors.cpu(), W_plot).numpy()
    Q = int(W_plot.shape[0])
    z_a = np.zeros(Q, dtype=np.float64)
    n = z_nb.shape[0]
    if labels is None:
        labels = np.zeros(n, dtype=np.int32)
    else:
        labels = np.asarray(labels).reshape(-1)[:n]

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for lab, mk, col in ((0, "o", "tab:blue"), (1, "x", "tab:orange")):
        m = labels == lab
        if not np.any(m):
            continue
        ax.scatter(z_nb[m, 0], z_nb[m, 1], s=22, marker=mk, c=col, alpha=0.75, label=f"regime {lab}")
    ax.scatter([z_a[0]], [z_a[1]], s=220, marker="*", c="red", edgecolors="k", zorder=5, label="anchor")
    pad = 0.5
    z0_lim = (float(min(z_nb[:, 0].min(), z_a[0]) - pad), float(max(z_nb[:, 0].max(), z_a[0]) + pad))
    z1_lim = (float(min(z_nb[:, 1].min(), z_a[1]) - pad), float(max(z_nb[:, 1].max(), z_a[1]) + pad))
    if draw_boundary and gate is not None:
        gate_np = np.asarray(gate.detach().cpu().numpy().reshape(-1), dtype=np.float64)
        if gate_np.size >= 3 and np.isfinite(gate_np[:3]).all():
            line = _boundary_segment_in_box(gate_np, z0_lim, z1_lim)
            if line is not None:
                ax.plot(
                    line[0], line[1],
                    color="crimson", linestyle="-", linewidth=2.2,
                    zorder=6, label=r"$h(z)=0$",
                )
    if sample_W_list:
        for k, Ws in enumerate(sample_W_list[:5]):
            zs = _latent_anchor_delta(X_anchor.cpu(), X_neighbors.cpu(), Ws.cpu()).numpy()
            ax.scatter(
                zs[:, 0], zs[:, 1], s=8, alpha=0.15, c="gray",
                label="sampled W" if k == 0 else None,
            )
    ax.set_xlim(z0_lim)
    ax.set_ylim(z1_lim)
    ax.set_xlabel(r"$z_0$")
    ax.set_ylabel(r"$z_1$")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    if z_phys_anchor is not None and z_phys_neighbors is not None:
        ax.text(
            0.02, 0.02,
            f"phys anchor=({z_phys_anchor[0]:.2f},{z_phys_anchor[1]:.2f})",
            transform=ax.transAxes, fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_l2_step_diagnostics(
    *,
    out_dir: Path,
    step: int,
    phase: str,
    anchor_idx: int,
    label: str,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    W0: torch.Tensor,
    R_mu: torch.Tensor,
    R_log_std: torch.Tensor,
    inducing_X: torch.Tensor,
    cem_state: dict[str, torch.Tensor],
    cem_anchor_idx: int = 0,
    qr_w_lengthscale: float,
    mc_w_signal_var: float,
    z_phys_anchor: np.ndarray,
    z_phys_neighbors: np.ndarray,
    regime_labels: np.ndarray,
    n_W_samples: int = 3,
    seed: int = 0,
    viz_gate_steps: int = 6,
    viz_gate_lr: float = 0.05,
    viz_gate_l2: float = 1e-3,
) -> None:
    """W0 vs E[W] panels; gate refit in each panel's projection space."""
    with torch.no_grad():
        w_mom = sparse_gp_uncertain_w_moments_for_mode(
            mode="anchor",
            X_anchor=X_anchor.unsqueeze(0),
            X_neighbors=X_neighbors.unsqueeze(0),
            inducing_X=inducing_X,
            R_mu=R_mu,
            R_log_std=R_log_std,
            lengthscale=float(qr_w_lengthscale),
            signal_var=float(mc_w_signal_var),
            include_conditional_residual=True,
        )
    W_mean = w_mom["W_mu_anchor"][0]
    W_var = w_mom["W_var_anchor"][0].clamp_min(1e-12)
    gen = torch.Generator(device=W_mean.device)
    gen.manual_seed(int(seed) + step * 1009 + anchor_idx)
    W_samples: list[torch.Tensor] = []
    if int(n_W_samples) > 0:
        eps = torch.randn((n_W_samples, *W_mean.shape), device=W_mean.device, generator=gen)
        W_samples = [W_mean + eps[i] * W_var.sqrt() for i in range(n_W_samples)]

    lab = cem_state["labels"]
    inlier = lab[cem_anchor_idx] if lab.dim() > 1 else lab
    g = cem_state["gate"]
    gate_warm = g[cem_anchor_idx] if g.dim() > 1 else g

    with torch.enable_grad():
        gate_w0 = fit_gate_for_projection(
            X_anchor, X_neighbors, inlier, W0,
            gate_init=gate_warm,
            gate_steps=viz_gate_steps,
            gate_lr=viz_gate_lr,
            gate_l2=viz_gate_l2,
        ).detach()
        gate_learned = fit_gate_for_projection(
            X_anchor, X_neighbors, inlier, W_mean,
            gate_init=gate_warm,
            gate_steps=viz_gate_steps,
            gate_lr=viz_gate_lr,
            gate_l2=viz_gate_l2,
        ).detach()

    base = out_dir / f"step{step:04d}_{phase}_j{anchor_idx}_{label}"
    plot_l2_projection_panel(
        out_path=base.with_name(base.name + "_W0.png"),
        X_anchor=X_anchor,
        X_neighbors=X_neighbors,
        y_neighbors=y_neighbors,
        W_plot=W0,
        W0=W0,
        gate=gate_w0,
        labels=regime_labels,
        title=f"{phase} step={step}  W0  j={anchor_idx} ({label})",
        z_phys_anchor=z_phys_anchor,
        z_phys_neighbors=z_phys_neighbors,
    )
    plot_l2_projection_panel(
        out_path=base.with_name(base.name + "_learned.png"),
        X_anchor=X_anchor,
        X_neighbors=X_neighbors,
        y_neighbors=y_neighbors,
        W_plot=W_mean,
        W0=W0,
        gate=gate_learned,
        labels=regime_labels,
        title=f"{phase} step={step}  E[W]  j={anchor_idx} ({label})",
        z_phys_anchor=z_phys_anchor,
        z_phys_neighbors=z_phys_neighbors,
        sample_W_list=W_samples,
    )

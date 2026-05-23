"""L2 qualitative diagnostics: projection geometry, CEM gate boundary, residual correction.

Default variant: ``D3_snfix0.3_warm120_blend0.5_slowW5x_trust`` (two-timescale slow W + trust),
with ``init_c_w0`` / ``init_inducing_from_w0x`` (W0@x latent inducing trick).

```powershell
cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP
python experiments/synthetic/plot_l2_diagnostics.py --seed 0 --num_steps 300
```

Outputs under ``experiments/synthetic/l2_diagnostics/<variant>_seed<s>/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from djgp.structured_projection import get_mu_W_cov_W  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from experiments.synthetic.local_gp_elbo_mj import (  # noqa: E402
    gp_residual_moments,
    mixture_scores_rho_mj,
    predict_inlier_mj,
    stack_m_j,
)
from experiments.synthetic.test_remedy_d_structured_vi import (  # noqa: E402
    _build_base,
    _clone_proj,
    _clone_regions,
    _collect_param_groups,
    _compute_elbo,
    _evaluate,
    _post_step_clamp,
    _clone_proj,
    _refresh_mu_u_star,
    _variant_specs,
)


def _latent_from_W(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """``X`` ``[n,D]``, ``W`` ``[Q,D]`` -> ``[n,Q]``."""
    return X @ W.T


def _latent_from_muW(X: torch.Tensor, mu_W: torch.Tensor) -> torch.Tensor:
    """``X`` ``[n,D]``, ``mu_W`` ``[n,Q,D]`` -> ``[n,Q]``."""
    return (mu_W * X.unsqueeze(-2)).sum(-1)


@torch.no_grad()
def _latent_points_one_by_one(
    X: torch.Tensor,
    proj: dict,
    hyperparams: dict,
) -> torch.Tensor:
    """Project each row of ``X`` ``[n,D]`` with ``E[W|x]``; avoids batched ``expand`` aliasing."""
    rows = []
    for i in range(X.shape[0]):
        xi = X[i : i + 1]
        mu_W, _ = get_mu_W_cov_W(proj, xi, hyperparams)
        rows.append((mu_W[0] * xi[0]).sum(dim=-1))
    return torch.stack(rows, dim=0)


@torch.no_grad()
def _region_latent(
    j: int,
    regions: list,
    proj: dict,
    hyperparams: dict,
    *,
    use_W0: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour projections ``[n,Q]`` and anchor ``[Q]`` in latent space."""
    X_nb = regions[j]["X"]
    n, _ = X_nb.shape
    W0 = proj["W0"]
    if use_W0:
        z_nb = _latent_from_W(X_nb, W0)
        x_a = hyperparams["X_test"][j : j + 1]
        z_a = _latent_from_W(x_a, W0)[0]
        return z_nb.cpu().numpy(), z_a.cpu().numpy()

    z_nb = _latent_points_one_by_one(X_nb, proj, hyperparams)
    x_a = hyperparams["X_test"][j : j + 1]
    z_a = _latent_points_one_by_one(x_a, proj, hyperparams)[0]
    return z_nb.cpu().numpy(), z_a.cpu().numpy()


@torch.no_grad()
def _region_rho(
    j: int,
    regions: list,
    proj: dict,
    u_params: list,
    hyperparams: dict,
    spec: dict,
) -> np.ndarray:
    r_j = [regions[j]]
    u_j = [u_params[j]]
    V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}
    gate_cfg = spec.get("gate", {"gate_mode": "learned"})
    _, _, rho = mixture_scores_rho_mj(
        r_j, V_dummy, u_j, hyperparams,
        proj=proj, gate_config=gate_cfg,
        use_mc_kfu=bool(spec.get("use_mc_kfu")),
        mc_samples=int(spec.get("mc_samples", 5)),
    )
    return rho[0].cpu().numpy()


def _boundary_line_2d(omega: np.ndarray, z_lim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray] | None:
    """Line ``omega0 + omega[1:].T z = 0`` in (z0,z1); returns (z0,z1) arrays."""
    w0 = float(omega[0])
    w1, w2 = float(omega[1]), float(omega[2])
    z1 = np.linspace(z_lim[0], z_lim[1], 200)
    if abs(w2) < 1e-10:
        if abs(w1) < 1e-10:
            return None
        z0 = np.full_like(z1, -w0 / w1)
        return z0, z1
    z0 = -(w0 + w2 * z1) / w1
    return z0, z1


def _per_anchor_drift(proj: dict, hyperparams: dict, j: int) -> float:
    W0 = proj["W0"]
    x = hyperparams["X_test"][j : j + 1]
    mu_W, _ = get_mu_W_cov_W(proj, x, hyperparams)
    return float(torch.linalg.norm(mu_W[0] - W0).item())


def _pick_anchors(
    anchor_stats: list[dict[str, float]],
    *,
    max_anchors: int = 6,
) -> list[tuple[str, int]]:
    """Return list of (label, region_index)."""
    T = len(anchor_stats)
    picks: list[tuple[str, int]] = []
    used: set[int] = set()

    def take(label: str, idx: int) -> None:
        if idx in used or len(picks) >= max_anchors:
            return
        picks.append((label, idx))
        used.add(idx)

    by_gain = sorted(range(T), key=lambda i: anchor_stats[i]["test_sse_gain"], reverse=True)
    by_cos = sorted(range(T), key=lambda i: anchor_stats[i]["cos_e0_zeta"], reverse=True)
    by_drift = sorted(range(T), key=lambda i: anchor_stats[i]["w_drift"], reverse=True)

    if by_gain:
        take("good_sse", by_gain[0])
    if by_cos and by_cos[0] not in used:
        take("good_cos", by_cos[0])
    if by_gain:
        take("bad_sse", by_gain[-1])
    if by_cos:
        take("bad_cos", by_cos[-1])
    if by_drift:
        take("high_drift", by_drift[-1])
        take("low_drift", by_drift[0])
    for i in range(T):
        if len(picks) >= max_anchors:
            break
        take(f"fill_{i}", i)
    return picks[:max_anchors]


def _train_with_snapshots(
    variant: str,
    spec: dict,
    regions,
    proj,
    u_params,
    hyperparams,
    *,
    lr: float,
    num_steps: int,
    eval_every: int,
    y_test_work,
    y_test_raw,
    scaler_y,
    seed: int,
) -> tuple[list[dict], dict[int, dict]]:
    """Train slow-W variant; return metrics records + state snapshots at eval steps."""
    records: list[dict] = []
    snapshots: dict[int, dict] = {}
    y_std = float(np.std(y_test_work))
    V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}
    inner_n = int(spec.get("inner_mu_steps", 5))
    lr_w = lr * float(spec.get("lr_W_scale", 0.1))
    slow_every = int(spec.get("slow_update_every", 1) or 1)
    opt_mu = opt_w = opt_slow = None
    freeze_w_prev: bool | None = None
    from experiments.synthetic.test_remedy_d_structured_vi import _freeze_W_at_step, _maybe_log_step

    for step in range(1, num_steps + 1):
        freeze_w_now = _freeze_W_at_step(spec, step)
        if freeze_w_now != freeze_w_prev:
            mu_f, w_s, slow = _collect_param_groups(proj, u_params, hyperparams, spec, step)
            opt_mu = torch.optim.Adam(mu_f, lr=lr) if mu_f else None
            opt_w = torch.optim.Adam(w_s, lr=lr_w) if w_s else None
            opt_slow = torch.optim.Adam(slow, lr=lr) if slow else None
            freeze_w_prev = freeze_w_now
        elif opt_mu is None:
            mu_f, w_s, slow = _collect_param_groups(proj, u_params, hyperparams, spec, step)
            opt_mu = torch.optim.Adam(mu_f, lr=lr) if mu_f else None
            opt_w = torch.optim.Adam(w_s, lr=lr_w) if w_s else None
            opt_slow = torch.optim.Adam(slow, lr=lr) if slow else None
            freeze_w_prev = freeze_w_now

        elbo_last = None
        sp_last = 0.0
        align_parts: dict[str, float] = {}

        for _ in range(inner_n):
            if opt_mu is None:
                break
            opt_mu.zero_grad()
            elbo, sp_last, align_parts = _compute_elbo(
                regions, proj, u_params, hyperparams, spec, step, num_steps, V_dummy,
            )
            (-elbo).backward()
            opt_mu.step()
            elbo_last = elbo

        with torch.no_grad():
            if spec.get("refresh_before_W", True):
                _refresh_mu_u_star(
                    regions, proj, u_params, hyperparams, V_dummy, spec, step, num_steps,
                )

        if opt_w is not None:
            opt_w.zero_grad()
            elbo, sp_last, align_parts = _compute_elbo(
                regions, proj, u_params, hyperparams, spec, step, num_steps, V_dummy,
            )
            (-elbo).backward()
            opt_w.step()
            elbo_last = elbo

        with torch.no_grad():
            if spec.get("refresh_after_W"):
                _refresh_mu_u_star(
                    regions, proj, u_params, hyperparams, V_dummy, spec, step, num_steps,
                )

        if opt_slow is not None and step % slow_every == 0:
            opt_slow.zero_grad()
            elbo, sp_last, align_parts = _compute_elbo(
                regions, proj, u_params, hyperparams, spec, step, num_steps, V_dummy,
            )
            (-elbo).backward()
            opt_slow.step()
            elbo_last = elbo

        with torch.no_grad():
            _post_step_clamp(proj, hyperparams, u_params, spec)
            from experiments.synthetic.local_gp_elbo_mj import apply_sigma_n_control
            apply_sigma_n_control(u_params, spec, step)

        if elbo_last is not None:
            _maybe_log_step(
                variant, step, num_steps, spec, elbo_last, sp_last, align_parts, eval_every,
                tt_tag=f"  TT(inner={inner_n})",
            )

        if step == 1 or step % eval_every == 0 or step == num_steps:
            records.append(
                _evaluate(
                    variant, regions, proj, u_params, hyperparams, V_dummy,
                    y_test_work, y_test_raw, scaler_y, step, y_std, spec, seed=seed,
                ),
            )
            snapshots[step] = {
                "proj": _clone_proj(proj),
                "u_params": copy.deepcopy(u_params),
            }
    return records, snapshots


def _plot_projection_panel(
    ax,
    z_nb: np.ndarray,
    z_a: np.ndarray,
    y_nb: np.ndarray,
    rho: np.ndarray,
    omega: np.ndarray,
    *,
    title: str,
    z_phys: np.ndarray | None = None,
) -> None:
    Q = z_nb.shape[1]
    r = np.abs(y_nb - np.median(y_nb))
    sc = ax.scatter(
        z_nb[:, 0], z_nb[:, 1] if Q > 1 else y_nb,
        c=rho,
        s=18 + 80 * (r / (r.max() + 1e-8)),
        cmap="viridis",
        alpha=0.85,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        [z_a[0]], [z_a[1] if Q > 1 else 0.0],
        marker="*",
        s=220,
        c="red",
        edgecolors="k",
        zorder=5,
        label="anchor",
    )
    z_lim = (
        float(min(z_nb[:, 0].min(), z_a[0]) - 0.5),
        float(max(z_nb[:, 0].max(), z_a[0]) + 0.5),
    )
    if Q >= 2:
        z1_lim = (
            float(min(z_nb[:, 1].min(), z_a[1]) - 0.5),
            float(max(z_nb[:, 1].max(), z_a[1]) + 0.5),
        )
        line = _boundary_line_2d(omega, z1_lim)
        if line is not None:
            z0_l, z1_l = line
            ax.plot(z0_l, z1_l, "k--", lw=1.5, label=r"$h(z)=0$")
        ax.set_xlabel(r"$z_0 = (Wx)_0$")
        ax.set_ylabel(r"$z_1 = (Wx)_1$")
    else:
        line_z0 = -float(omega[0]) / float(omega[1]) if abs(float(omega[1])) > 1e-10 else None
        if line_z0 is not None:
            ax.axvline(line_z0, color="k", ls="--", lw=1.5, label=r"$h(z)=0$")
        ax.set_xlabel(r"$z = Wx$")
        ax.set_ylabel(r"$y$ (neighbour)")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, fraction=0.046, label=r"$\rho$")
    if z_phys is not None:
        ax.plot(z_phys[:, 0], z_phys[:, 1], "x", color="gray", alpha=0.25, ms=3)
    ax.legend(loc="best", fontsize=7)


def _plot_summary_curves(out_dir: Path, records: list[dict], variant: str) -> None:
    steps = [int(r["step"]) for r in records]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.plot(steps, [r.get("W_drift_frob_mean", float("nan")) for r in records], "o-", label="||E[W]-W0||")
    ax.plot(steps, [r.get("W_metric_drift_frob_mean", float("nan")) for r in records], "s-", label="||W'W-W0'W0||")
    ax.set_xlabel("step")
    ax.set_ylabel("drift")
    ax.legend(fontsize=8)
    ax.set_title("W / metric drift")

    ax = axes[0, 1]
    ax.plot(steps, [r.get("ratio_zeta_learned_r_elbo", float("nan")) for r in records], "o-", label="z_learn/r")
    ax.plot(steps, [r.get("phiTr_rho_over_pure", float("nan")) for r in records], "s-", label="Br/Bp")
    ax2 = ax.twinx()
    ax2.plot(steps, [r.get("test_sse_gain_zeta_raw", r.get("test_sse_gain_zeta", float("nan"))) for r in records],
             "^-", color="tab:orange", label="test_SSEg")
    ax.set_xlabel("step")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="lower right", fontsize=7)
    ax.set_title("residual scale & test gain")

    ax = axes[1, 0]
    ax.plot(steps, [r.get("rmse", float("nan")) for r in records], "o-", label="direct RMSE")
    if records[0].get("rmse_cem_raw") == records[0].get("rmse_cem_raw"):
        ax.plot(steps, [r.get("rmse_cem_raw", float("nan")) for r in records], "s-", label="CEM RMSE")
    ax.set_xlabel("step")
    ax.set_ylabel("RMSE (y raw)")
    ax.legend(fontsize=8)
    ax.set_title("prediction RMSE")

    ax = axes[1, 1]
    ax.plot(steps, [r.get("crps", float("nan")) for r in records], "o-", label="direct CRPS")
    if records[0].get("crps_cem_raw") == records[0].get("crps_cem_raw"):
        ax.plot(steps, [r.get("crps_cem_raw", float("nan")) for r in records], "s-", label="CEM CRPS")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.set_title("CRPS")

    fig.suptitle(f"{variant} — training diagnostics", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_summary_curves.png", dpi=150)
    plt.close(fig)


def _plot_residual_scatter(
    out_dir: Path,
    regions,
    proj,
    u_params,
    hyperparams,
    spec,
    y_test_work: np.ndarray,
    anchor_picks: list[tuple[str, int]],
    scaler_y: StandardScaler | None,
) -> None:
    m_j = stack_m_j(u_params)
    mom_tr = gp_residual_moments(
        regions, {"mu_V": None, "sigma_V": None}, u_params, hyperparams,
        proj=proj, at_test=False,
        use_mc_kfu=bool(spec.get("use_mc_kfu")),
        mc_samples=int(spec.get("mc_samples", 5)),
    )
    mom_te = gp_residual_moments(
        regions, {"mu_V": None, "sigma_V": None}, u_params, hyperparams,
        proj=proj, at_test=True,
        use_mc_kfu=bool(spec.get("use_mc_kfu")),
        mc_samples=int(spec.get("mc_samples", 5)),
    )
    y_tr = torch.stack([r["y"] for r in regions], dim=0)
    r_tr = (y_tr - m_j.view(-1, 1)).reshape(-1).detach().cpu().numpy()
    z_tr = mom_tr["zeta"].reshape(-1).detach().cpu().numpy()

    y_te = torch.from_numpy(y_test_work.astype(np.float32)).to(m_j.device)
    e0 = (y_te - m_j).detach().cpu().numpy()
    z_te = mom_te["zeta"].squeeze(1).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, x, y, xlab in [
        (axes[0], r_tr, z_tr, r"$r_i = y_i - m_j$ (train nb)"),
        (axes[1], e0, z_te, r"$e_0 = y_* - m_j$ (test)"),
    ]:
        ax.scatter(x, y, s=8, alpha=0.35, c="steelblue")
        lim = max(np.abs(x).max(), np.abs(y).max(), 1e-6) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label=r"$\zeta = $ residual")
        for label, j in anchor_picks:
            if xlab.startswith(r"$r"):
                xx = (regions[j]["y"] - m_j[j]).detach().cpu().numpy()
                yy = mom_tr["zeta"][j].detach().cpu().numpy()
            else:
                xx = np.array([e0[j]])
                yy = np.array([z_te[j]])
            ax.scatter(xx, yy, s=120, marker="*", label=f"{label} j={j}", zorder=4)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$\zeta$ (learned correction)")
        ax.legend(fontsize=7, loc="best")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Residual vs ELBO correction (direct head)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_residual_scatter.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def _anchor_z_trajectory(
    j: int,
    snapshots: dict[int, dict],
    hyperparams: dict,
    proj_template: dict,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    steps = sorted(snapshots)
    W0 = proj_template["W0"]
    x = hyperparams["X_test"][j : j + 1]
    z0 = _latent_from_W(x, W0)[0].cpu().numpy()
    zl_list = []
    for st in steps:
        zl_list.append(_latent_points_one_by_one(x, snapshots[st]["proj"], hyperparams)[0].cpu().numpy())
    return steps, z0, np.stack(zl_list)


def main() -> None:
    p = argparse.ArgumentParser(description="L2 projection / gate / residual diagnostics.")
    p.add_argument("--setting", default="l2_q2_train500")
    p.add_argument("--variant", default="D3_snfix0.3_warm120_blend0.5_slowW5x_trust")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--num_steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=75)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--max_anchors", type=int, default=6)
    p.add_argument("--sample_W_M", type=int, default=5)
    p.add_argument("--out_dir", default="")
    args = p.parse_args()

    spec = dict(_variant_specs()[args.variant])
    spec["eval_sampleW_cem"] = True
    spec["sampleW_M"] = args.sample_W_M
    cfg = SETTING_PRESETS[args.setting]
    if int(cfg.get("Q", 2)) != 2:
        print("WARNING: Q!=2; 2D projection panels use first two latent dims only.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "experiments/synthetic/l2_diagnostics"
        / f"{args.variant}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)
    data = _generate_paper_data(cfg, args.seed, device)
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(data["X_train"])
    X_test_s = scaler_x.transform(data["X_test"])
    y_train_raw = np.asarray(data["y_train"], dtype=np.float64)
    y_test_raw = np.asarray(data["y_test"], dtype=np.float64)[: args.max_test]
    X_test_s = X_test_s[: args.max_test]
    z_test_2d = np.asarray(data["z_test"], dtype=np.float64)[: args.max_test]

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_test_work = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)
    Q, n = int(cfg["Q"]), int(cfg["n"])

    print(f"Device: {device}  setting={args.setting}  variant={args.variant}", flush=True)
    regions, proj, u_params, hyperparams, _ = _build_base(
        X_train_t, y_train_t, X_test_t,
        Q=Q, n=n, m1=args.m1, m2=args.m2, device=device, seed=args.seed,
        y_train_np=y_train,
        spec=spec,
    )
    regions = _clone_regions(regions)
    proj = _clone_proj(proj)
    if proj.get("kind") == "lowrank":
        hyperparams["mu_B_init"] = proj["mu_B"].detach().clone()

    records, snapshots = _train_with_snapshots(
        args.variant, spec, regions, proj, u_params, hyperparams,
        lr=args.lr, num_steps=args.num_steps, eval_every=args.eval_every,
        y_test_work=y_test_work, y_test_raw=y_test_raw, scaler_y=scaler_y, seed=args.seed,
    )

    with (out_dir / "metrics_timeseries.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    m_j = stack_m_j(u_params)
    _, _, zeta_te = predict_inlier_mj(
        regions, {"mu_V": None, "sigma_V": None}, u_params, hyperparams, proj=proj,
    )
    anchor_stats = []
    for j in range(len(regions)):
        e0 = float(y_test_work[j] - m_j[j].item())
        zj = float(zeta_te[j].item())
        sse0 = e0 * e0
        sse1 = (e0 - zj) ** 2
        anchor_stats.append({
            "j": j,
            "test_sse_gain": float((sse0 - sse1) / max(sse0, 1e-12)),
            "cos_e0_zeta": float(e0 * zj / (math.hypot(e0, zj) + 1e-12)),
            "w_drift": _per_anchor_drift(proj, hyperparams, j),
            "z_phys_x": float(z_test_2d[j, 0]),
            "z_phys_y": float(z_test_2d[j, 1]),
        })
    with (out_dir / "anchor_stats.json").open("w", encoding="utf-8") as f:
        json.dump(anchor_stats, f, indent=2)

    picks = _pick_anchors(anchor_stats, max_anchors=args.max_anchors)
    print("Anchor picks:", picks, flush=True)

    _plot_summary_curves(out_dir, records, args.variant)
    _plot_residual_scatter(
        out_dir, regions, proj, u_params, hyperparams, spec,
        y_test_work, picks, scaler_y,
    )

    for label, j in picks:
        z_nb0, z_a0 = _region_latent(j, regions, proj, hyperparams, use_W0=True)
        z_nbL, z_aL = _region_latent(j, regions, proj, hyperparams, use_W0=False)
        rho = _region_rho(j, regions, proj, u_params, hyperparams, spec)
        omega = u_params[j]["omega"].detach().cpu().numpy()
        y_nb = regions[j]["y"].detach().cpu().numpy()
        z_phys = z_test_2d[
            [ii for ii in range(len(regions)) if ii != j][: min(40, len(regions) - 1)]
        ]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        _plot_projection_panel(
            axes[0], z_nb0, z_a0, y_nb, rho, omega,
            title=rf"$W_0$  ({label}, j={j})",
            z_phys=None,
        )
        _plot_projection_panel(
            axes[1], z_nbL, z_aL, y_nb, rho, omega,
            title=rf"learned $E[W]$  ({label}, j={j})",
            z_phys=None,
        )
        fig.suptitle(f"L2 latent neighbourhood — {args.variant}")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_projection_{label}_j{j}.png", dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.scatter(z_test_2d[:, 0], z_test_2d[:, 1], s=6, c="lightgray", alpha=0.5)
        ax2.scatter([z_test_2d[j, 0]], [z_test_2d[j, 1]], s=180, marker="*", c="red", edgecolors="k")
        ax2.set_title(f"Physical L2 location j={j} ({label})")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        fig2.tight_layout()
        fig2.savefig(out_dir / f"fig_phys_xy_{label}_j{j}.png", dpi=150)
        plt.close(fig2)

    final_step = max(snapshots)
    for label, j in picks[:3]:
        steps, z_w0, z_learn_traj = _anchor_z_trajectory(
            j, snapshots, hyperparams, proj,
        )
        if z_learn_traj.shape[1] < 2:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(z_learn_traj[:, 0], z_learn_traj[:, 1], "o-", label=r"$E[W]x_*$ over steps")
        ax.scatter([z_w0[0]], [z_w0[1]], s=200, marker="*", c="green", label=r"$W_0 x_*$", zorder=5)
        for st, z in zip(steps, z_learn_traj):
            ax.annotate(str(st), (z[0], z[1]), fontsize=7)
        ax.set_xlabel(r"$z_0$")
        ax.set_ylabel(r"$z_1$")
        ax.set_title(f"Anchor latent path ({label}, j={j})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_anchor_path_{label}_j{j}.png", dpi=150)
        plt.close(fig)

    print(f"Saved figures and JSON to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

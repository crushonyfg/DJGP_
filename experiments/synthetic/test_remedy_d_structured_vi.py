"""Remedy D: W0-centered structured projection + m_j-corrected inlier ELBO.

Variants (all use explicit ``m_j`` in ``Q``; default ``--standardize_y``):

- ``C0_mj``: full ``q(R)`` mean-field (legacy) + ``m_j``
- ``D1``: ``W = W0 + Delta_W`` full residual, ``log(1+s)`` penalty
- ``D2``: ``W = W0 + B U^T`` low-rank residual + penalty
- ``D3``: D2 + inducing ``C`` init from ``W0 @ x`` neighbours
- ``D4``: D3 + MC ``Psi_1`` (sampled ``W``) instead of analytic ``E_q[K]``
- ``D3_cov*`` / ``D3_ridge*``: add centered alignment aux losses (see ``djgp/gp_feature_alignment.py``)
- ``D3_snfix0.3_exploreW_inner20_*``: profile-style W exploration (``explore_w_profile``): phase A/B/C
  inner ``mu_u`` steps, decaying ``lr_W``, optional ``lambda_W_trust`` and CEM acceptance.

Eval prints ``signal``, ridge ``R^2`` at ``tau=0,1``, ``zeta/r`` (learned vs ridge) for diagnostics A/B/C.

```powershell
python experiments/synthetic/test_remedy_d_structured_vi.py ^
    --setting lh_q5_train500 --seed 0 --max_test 200 --num_steps 300 ^
    --variants D3,D3_cov0.1,D3_cov1,D3_ridge0.1 ^
    --out_dir experiments/synthetic/remedy_d3_phi_align_seed0
```
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.elbo_shrinkage import elbo_star_metrics  # noqa: E402
from djgp.gp_feature_alignment import qu_diagnostics  # noqa: E402
from djgp.gp_feature_alignment import (  # noqa: E402
    alignment_losses_train,
    elbo_star_mu_u,
    elbo_star_mu_u_custom_rho,
    elbo_star_mu_u_rho,
    mixture_elbo_star_metrics,
    refresh_mu_u_toward_star,
    region_alignment_diagnostics,
)
from djgp.projections.inductive import run_projected_jumpgp_ensemble  # noqa: E402
from djgp.structured_projection import (  # noqa: E402
    build_U_numpy,
    build_W0_numpy,
    build_proj_params,
    clamp_projection,
    compute_s_penalty_proj,
    init_inducing_from_w0x,
    projection_w_drift_metrics,
    proj_trainable_tensors,
    psi1_diagnostics,
    sample_W_from_proj,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from experiments.synthetic.diagnose_inlier_gp_path_direct_vi import (  # noqa: E402
    _build_state_diagnose,
    _clone_regions,
    _deep_clone_state_diag,
    init_m_j_from_regions,
)
from experiments.synthetic.local_gp_elbo_mj import (  # noqa: E402
    apply_sigma_n_control,
    compute_elbo_mixture_mj,
    compute_elbo_pure_inlier_mj,
    gp_residual_moments,
    init_sigma_n_from_spec,
    init_u_from_spec,
    mixture_diagnostics_mj,
    predict_inlier_mj,
    sigma_n_learnable,
    stack_m_j,
    training_beta_u_kl,
)
from experiments.synthetic.mixture_rho_curriculum import (  # noqa: E402
    attach_rho0,
    build_rho0,
    compute_rho_tilde,
    lambda_rho_anchor_schedule,
    mixture_elbo_uses_mixture,
    rho0_quality_metrics,
    rho_blend_alpha,
    use_pure_inlier_warmup,
)
from experiments.synthetic.local_gp_elbo_mj import mixture_scores_rho_mj  # noqa: E402
from shared.jumpgp_runner import find_neighborhoods  # noqa: E402


def _d3_base() -> dict[str, Any]:
    return {
        "explicit_mj": True,
        "learn_mj": True,
        "proj_kind": "lowrank",
        "rank": 3,
        "lambda_s": 0.1,
        "sigma_max": 0.05,
        "w0_init": "pls",
        "basis_mode": "pca_residual",
        "init_c_w0": True,
        "lambda_cov": 0.0,
        "lambda_ridge": 0.0,
        "ridge_lambda": 1e-2,
        "cov_clip": 1.0,
        "align_proj_only": True,
        "diag_tau": True,
    }


def _rho_combo_base(**over: Any) -> dict[str, Any]:
    """Mixture ELBO + posterior ``rho`` floor + fixed outlier width ``u``."""
    base = {
        **_d3_base(),
        "use_mixture_elbo": True,
        "rho_floor_min": 0.3,
        "lambda_rho_floor": 20.0,
        "rho_floor_per_region": False,
        "fix_u": 0.05,
        "diag_shrinkage": True,
        "diag_mixture": True,
    }
    base.update(over)
    return base


def _rho_curriculum_base(**over: Any) -> dict[str, Any]:
    """Mixture + sigma_n fix, no batch rho floor (curriculum / anchor experiments)."""
    base = {
        **_d3_base(),
        "use_mixture_elbo": True,
        "sigma_n_fixed": 0.3,
        "fix_u": 0.05,
        "lambda_rho_floor": 0.0,
        "rho_floor_min": None,
        "diag_shrinkage": True,
        "diag_mixture": True,
    }
    base.update(over)
    return base


def _warm120_blend_uniform(alpha_end: float, **over: Any) -> dict[str, Any]:
    """120-step pure inlier, then ``alpha: 0 -> alpha_end`` (default release ends step 300)."""
    kw = dict(
        rho_warmup_pure_steps=120,
        rho_blend=True,
        rho_init="uniform",
        rho_init_uniform_value=1.0,
        rho_blend_alpha_end=float(alpha_end),
        rho_blend_start_step=121,
        rho_blend_end_step=300,
        rho_blend_linear_data=True,
    )
    kw.update(over)
    return _rho_curriculum_base(**kw)


def _explore_w_base(**over: Any) -> dict[str, Any]:
    """Profile-style W exploration: inner ``mu_u`` adaptation + decaying ``lr_W`` + phase C freeze."""
    base = dict(
        rho_blend_end_step=600,
        refresh_star="tilde",
        refresh_alpha=0.3,
        refresh_mu_u_every=0,
        explore_w_profile=True,
        explore_phase_a_end=80,
        explore_phase_b_end=300,
        explore_inner_mu_a=20,
        explore_inner_mu_b=8,
        explore_inner_mu_c=5,
        explore_lr_W_a=1.0,
        explore_lr_W_b=0.1,
        explore_lr_W_c=0.03,
        explore_freeze_W_phase_c=True,
        explore_phase_a_freeze_mj=True,
        explore_phase_a_freeze_gate=True,
        rho_blend_alpha_cap=0.3,
        rho_blend_alpha_cap_until_step=200,
        slow_update_every=1,
        refresh_before_W=True,
        refresh_after_W=False,
        eval_sampleW_cem=True,
        sampleW_M=8,
    )
    base.update(over)
    return _warm120_blend_uniform(0.5, **base)


def _warm120_blend_slowW(**over: Any) -> dict[str, Any]:
    """blend0.5 + refresh + block-coordinate (fast ``mu_u``, slow ``W``)."""
    base = dict(
        rho_blend_end_step=600,
        refresh_star="tilde",
        refresh_alpha=0.3,
        refresh_mu_u_every=0,
        two_timescale=True,
        inner_mu_steps=5,
        lr_W_scale=0.1,
        refresh_before_W=True,
        refresh_after_W=False,
        slow_update_every=1,
    )
    base.update(over)
    return _warm120_blend_uniform(0.5, **base)


def _variant_specs() -> dict[str, dict[str, Any]]:
    base_mj = {"explicit_mj": True, "learn_mj": True}
    return {
        "C0_mj": {**base_mj, "proj_kind": "full", "lambda_s": 0.0, "sigma_max": None},
        "D1": {
            **base_mj,
            "proj_kind": "full_centered",
            "lambda_s": 0.1,
            "sigma_max": 0.1,
            "w0_init": "pls",
        },
        "D2": {
            **base_mj,
            "proj_kind": "lowrank",
            "rank": 3,
            "lambda_s": 0.1,
            "sigma_max": 0.05,
            "w0_init": "pls",
            "basis_mode": "pca_residual",
        },
        "D3": _d3_base(),
        "D4": {
            **base_mj,
            "proj_kind": "lowrank",
            "rank": 3,
            "lambda_s": 0.1,
            "sigma_max": 0.05,
            "w0_init": "pls",
            "basis_mode": "pca_residual",
            "init_c_w0": True,
            "use_mc_kfu": True,
            "mc_samples": 5,
        },
        "D3_cov0.01": {**_d3_base(), "lambda_cov": 0.01},
        "D3_cov0.03": {**_d3_base(), "lambda_cov": 0.03},
        "D3_cov0.1": {**_d3_base(), "lambda_cov": 0.1},
        "D3_cov0.3": {**_d3_base(), "lambda_cov": 0.3},
        "D3_cov1": {**_d3_base(), "lambda_cov": 1.0},
        "D3_ridge0.1": {**_d3_base(), "lambda_ridge": 0.1},
        "D3_ridge1": {**_d3_base(), "lambda_ridge": 1.0},
        "D3_muref20": {**_d3_base(), "refresh_mu_u_every": 20, "refresh_alpha": 0.3},
        "D3_muref50": {**_d3_base(), "refresh_mu_u_every": 50, "refresh_alpha": 0.5},
        "D3_snmax0.5": {**_d3_base(), "sigma_n_max": 0.5, "diag_shrinkage": True},
        "D3_snmax0.8": {**_d3_base(), "sigma_n_max": 0.8, "diag_shrinkage": True},
        "D3_betaU0.1": {**_d3_base(), "beta_u_kl": 0.1, "diag_shrinkage": True},
        "D3_betaU0.3": {**_d3_base(), "beta_u_kl": 0.3, "diag_shrinkage": True},
        "D3_betaU_warmup": {
            **_d3_base(),
            "beta_u_warmup": True,
            "beta_u_warmup_start": 0.1,
            "beta_u_warmup_end": 1.0,
            "diag_shrinkage": True,
        },
        "D3_sigma_prior": {
            **_d3_base(),
            "lambda_sigma_prior": 1.0,
            "log_sigma_n0": 0.0,
            "diag_shrinkage": True,
        },
        "D3_snfix0.2": {**_d3_base(), "sigma_n_fixed": 0.2, "diag_shrinkage": True, "diag_mixture": True},
        "D3_snfix0.3": {**_d3_base(), "sigma_n_fixed": 0.3, "diag_shrinkage": True, "diag_mixture": True},
        "D3_snfix0.5": {**_d3_base(), "sigma_n_fixed": 0.5, "diag_shrinkage": True, "diag_mixture": True},
        "D3_snmax0.8_ref": {**_d3_base(), "sigma_n_max": 0.8, "diag_shrinkage": True, "diag_mixture": True},
        "D3_snwarm0.2_release": {
            **_d3_base(),
            "sigma_n_fixed": 0.2,
            "sigma_n_freeze_steps": 150,
            "sigma_n_max": 0.8,
            "diag_shrinkage": True,
            "diag_mixture": True,
        },
        "D3_snprior_strong": {
            **_d3_base(),
            "lambda_sigma_prior": 10.0,
            "log_sigma_n0": -1.6094379124345703,
            "diag_shrinkage": True,
            "diag_mixture": True,
        },
        "D3_snfix0.5_rhofloor0.3": _rho_combo_base(sigma_n_fixed=0.5),
        "D3_snfix0.3_rhofloor0.3": _rho_combo_base(sigma_n_fixed=0.3),
        "D3_snfix0.2_rhofloor0.3": _rho_combo_base(sigma_n_fixed=0.2),
        "D3_snmax0.8_rhofloor0.3": _rho_combo_base(sigma_n_max=0.8),
        "D3_snfix0.3_rhofloor0.5": _rho_combo_base(sigma_n_fixed=0.3, rho_floor_min=0.5),
        "D3_snfix0.3_rhofloor0.3_lam5": _rho_combo_base(
            sigma_n_fixed=0.3, lambda_rho_floor=5.0,
        ),
        "D3_snfix0.3_rhofloor0.3_refresh0.3": _rho_combo_base(
            sigma_n_fixed=0.3,
            refresh_mu_u_every=20,
            refresh_alpha=0.3,
            refresh_star="rho",
        ),
        "D3_snmax0.8_rhofloor0.3_refresh0.3": _rho_combo_base(
            sigma_n_max=0.8,
            refresh_mu_u_every=20,
            refresh_alpha=0.3,
            refresh_star="rho",
        ),
        "D3_snfix0.3_rhoinit_uniform_warm": _rho_curriculum_base(
            rho_warmup_pure_steps=100,
        ),
        "D3_snfix0.3_rhoanchor_ridge": _rho_curriculum_base(
            rho_init="ridge",
            rho_anchor_lambda0=50.0,
            rho_anchor_anneal=True,
        ),
        "D3_snfix0.3_rhoanchor_geom": _rho_curriculum_base(
            rho_init="geom",
            rho_anchor_lambda0=50.0,
            rho_anchor_anneal=True,
        ),
        "D3_snfix0.3_rho_mixfloor_pointwise": _rho_curriculum_base(
            rho_pointwise_min=0.15,
            lambda_rho_pointwise_floor=20.0,
        ),
        "D3_snfix0.3_warm80_blend_uniform_to0.5": _rho_curriculum_base(
            rho_warmup_pure_steps=80,
            rho_blend=True,
            rho_init="uniform",
            rho_init_uniform_value=1.0,
            rho_blend_alpha_end=0.5,
            rho_blend_start_step=81,
            rho_blend_end_step=120,
            rho_blend_linear_data=True,
        ),
        "D3_snfix0.3_warm80_blend_uniform_to1.0": _rho_curriculum_base(
            rho_warmup_pure_steps=80,
            rho_blend=True,
            rho_init="uniform",
            rho_init_uniform_value=1.0,
            rho_blend_alpha_end=1.0,
            rho_blend_start_step=81,
            rho_blend_end_step=120,
            rho_blend_linear_data=True,
        ),
        "D3_snfix0.3_pointblend0.3": _rho_curriculum_base(
            rho_point_blend_gamma=0.3,
            rho_blend_linear_data=True,
        ),
        "D3_snfix0.3_pure": {
            **_d3_base(),
            "sigma_n_fixed": 0.3,
            "diag_shrinkage": True,
            "diag_mixture": True,
        },
        "D3_snfix0.3_warm120_blend_uniform_to0.5_over180": _warm120_blend_uniform(0.5),
        "D3_snfix0.3_warm120_blend_uniform_to0.7_over180": _warm120_blend_uniform(0.7),
        "D3_snfix0.3_warm120_blend_uniform_to0.5_refresh0.2": _warm120_blend_uniform(
            0.5,
            refresh_mu_u_every=20,
            refresh_alpha=0.2,
            refresh_star="tilde",
        ),
        "D3_snfix0.3_warm120_blend_to0.5_refresh_curve600": _warm120_blend_uniform(
            0.5,
            rho_blend_end_step=600,
            refresh_mu_u_every=20,
            refresh_alpha=0.2,
            refresh_star="tilde",
        ),
        "D3_snfix0.3_warm120_blend_to0.5_freeze_W": _warm120_blend_uniform(
            0.5,
            freeze_W=True,
            refresh_mu_u_every=20,
            refresh_alpha=0.2,
            refresh_star="tilde",
            eval_sampleW_cem=True,
            sampleW_M=8,
        ),
        "D3_snfix0.3_warm120_blend_to0.5_freeze_mu_u": _warm120_blend_uniform(
            0.5,
            freeze_mu_u=True,
            refresh_mu_u_every=0,
        ),
        "D3_snfix0.3_warm120_blend_to0.5_freeze_W_after120": _warm120_blend_uniform(
            0.5,
            freeze_W_after_warmup=True,
            refresh_mu_u_every=20,
            refresh_alpha=0.2,
            refresh_star="tilde",
        ),
        "D3_snfix0.3_warm120_blend0.5_slowW5x": _warm120_blend_slowW(
            eval_sampleW_cem=True,
            sampleW_M=8,
        ),
        "D3_snfix0.3_warm120_blend0.5_slowW10x": _warm120_blend_slowW(inner_mu_steps=10),
        "D3_snfix0.3_warm120_blend0.5_slowW5x_trust": _warm120_blend_slowW(
            lambda_W_trust=1.0,
        ),
        "D3_snfix0.3_warm120_blend0.5_slowW5x_refresh": _warm120_blend_slowW(
            refresh_before_W=True,
            refresh_after_W=True,
        ),
        "D3_snfix0.3_warm120_blend0.5_slowW5x_freeze_slow": _warm120_blend_slowW(
            freeze_mj=True,
            freeze_gate=True,
        ),
        "D3_snfix0.3_exploreW_inner20_lrW1_decay": _explore_w_base(explore_lr_W_a=1.0),
        "D3_snfix0.3_exploreW_inner20_lrW0.5_decay": _explore_w_base(explore_lr_W_a=0.5),
        "D3_snfix0.3_exploreW_inner20_lrW0.5_trust": _explore_w_base(
            explore_lr_W_a=0.5,
            lambda_W_trust=1.0,
        ),
        "D3_snfix0.3_exploreW_inner20_lrW0.5_acceptCEM": _explore_w_base(
            explore_lr_W_a=0.5,
            explore_accept_cem=True,
            explore_accept_every=20,
            explore_accept_MC=3,
        ),
        "D3_snfix0.3_lineW_elbo_inner20": _line_w_base(),
        "D3_snfix0.3_lineW_elbo_inner20_decay": _line_w_base(
            line_tau_decay_step=150,
            line_tau_grid_early=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
            line_tau_grid_late=(0.0, 0.25, 0.5, 1.0),
        ),
        "D3_snfix0.3_lineW_holdout_inner20": _line_w_base(line_score_mode="heldout"),
        "D3_snfix0.3_lineW_cem_accept20": _line_w_base(
            line_cem_accept_every=20,
            line_cem_accept_tolerance=0.0,
        ),
        "D3_snfix0.3_lineW_elbo_trust_strong": _line_w_base(lambda_W_trust=5.0),
    }


def _print_rmse_crps_checkpoints(all_records: list[dict], variants: list[str]) -> None:
    """Summarize whether RMSE/CRPS (on original ``y`` scale) change across eval checkpoints."""
    print(f"\n{'='*80}\nRMSE / CRPS on original y scale @ checkpoints\n{'='*80}")
    hdr = f"{'step':>6s}"
    for v in variants:
        short = v.replace("D3_snfix0.3_", "")[:22]
        hdr += f"  {short:>22s}"
    print(hdr)
    steps_seen: set[int] = set()
    for r in all_records:
        steps_seen.add(int(r["step"]))
    for step in sorted(steps_seen):
        row = f"{step:6d}"
        for v in variants:
            hit = [x for x in all_records if x["variant"] == v and int(x["step"]) == step]
            if not hit:
                row += f"  {'—':>22s}"
                continue
            h = hit[-1]
            row += f"  {h.get('rmse', float('nan')):6.3f}/{h.get('crps', float('nan')):6.3f}"
        print(row)
    for v in variants:
        rows = sorted(
            [x for x in all_records if x["variant"] == v],
            key=lambda x: int(x["step"]),
        )
        if len(rows) < 2:
            continue
        rmses = [float(x["rmse"]) for x in rows]
        crpss = [float(x["crps"]) for x in rows]
        dr = max(rmses) - min(rmses)
        dc = max(crpss) - min(crpss)
        stagnant = dr < 1e-4 and dc < 1e-4
        tag = "STAGNANT" if stagnant else "moving"
        print(
            f"  [{v}] RMSE range={dr:.4f}  CRPS range={dc:.4f}  ({tag})",
            flush=True,
        )
        r_first, r_last = rows[0], rows[-1]
        s0, s1 = int(r_first["step"]), int(r_last["step"])
        print(
            f"      step {s0} -> {s1}:  RMSE {float(r_first['rmse']):.3f} -> {float(r_last['rmse']):.3f}  "
            f"CRPS {float(r_first['crps']):.3f} -> {float(r_last['crps']):.3f}",
            flush=True,
        )


def _build_base(
    X_train_t,
    y_train_t,
    X_test_t,
    *,
    Q,
    n,
    m1,
    m2,
    device,
    seed,
    y_train_np,
    spec,
):
    regions, _V, u_params, hyperparams = _build_state_diagnose(
        X_train_t, y_train_t, X_test_t,
        Q=Q, n=n, m1=m1, m2=m2, device=device, seed=seed,
        explicit_mj=True,
        mj_init="mean",
        learn_mj=spec.get("learn_mj", True),
    )
    W0_np = build_W0_numpy(
        X_train_t.cpu().numpy(), y_train_np, Q, spec.get("w0_init", "pls"), seed,
    )
    rank = int(spec.get("rank", 3))
    U_np = None
    if spec["proj_kind"] == "lowrank":
        U_np = build_U_numpy(
            X_train_t.cpu().numpy(), y_train_np, W0_np, rank,
            spec.get("basis_mode", "pca_residual"), seed + 17,
        )
        rank = U_np.shape[1]
    proj = build_proj_params(
        kind=spec["proj_kind"],
        device=device,
        m2=m2,
        Q=Q,
        D=X_train_t.shape[1],
        r=rank,
        W0_np=W0_np,
        U_np=U_np,
        sigma_init=0.05,
        sigma_max=spec.get("sigma_max"),
    )
    if spec.get("init_c_w0"):
        init_inducing_from_w0x(regions, proj["W0"], seed=seed + 3)
    init_sigma_n_from_spec(u_params, spec)
    init_u_from_spec(u_params, spec)
    V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}
    rho0 = build_rho0(regions, proj, hyperparams, u_params, V_dummy, spec)
    attach_rho0(hyperparams, rho0)
    if rho0 is not None and spec.get("log_rho0_quality", True):
        hyperparams["rho0_quality"] = rho0_quality_metrics(
            regions, proj, hyperparams, u_params, V_dummy, rho0,
            use_mc_kfu=bool(spec.get("use_mc_kfu")),
            mc_samples=int(spec.get("mc_samples", 5)),
        )
    hyperparams["proj_kind"] = spec["proj_kind"]
    if proj["kind"] == "lowrank":
        hyperparams["mu_B_init"] = proj["mu_B"].detach().clone()
    return regions, proj, u_params, hyperparams, W0_np


def _proj_eval_copy(proj: dict, *, use_W0: bool) -> dict:
    """Shallow copy; optionally zero ``mu_B`` so ``E_q[W] = W0`` at eval."""
    if not use_W0 or proj["kind"] != "lowrank":
        return proj
    out = dict(proj)
    out["mu_B"] = torch.zeros_like(proj["mu_B"])
    return out


def _freeze_W_at_step(spec: dict, step: int) -> bool:
    if spec.get("freeze_W"):
        return True
    if spec.get("freeze_W_after_warmup"):
        warm = int(spec.get("rho_warmup_pure_steps", 0) or 0)
        return step > warm
    return False


def _collect_trainable_params(
    proj,
    u_params,
    hyperparams,
    spec,
    step: int,
) -> list[torch.Tensor]:
    mu_f, w_s, slow = _collect_param_groups(proj, u_params, hyperparams, spec, step)
    return mu_f + w_s + slow


def _collect_param_groups(
    proj,
    u_params,
    hyperparams,
    spec,
    step: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """``mu_fast``, ``W_slow``, ``slow_other`` for block-coordinate training."""
    mu_fast: list[torch.Tensor] = []
    W_slow: list[torch.Tensor] = []
    slow_other: list[torch.Tensor] = [hyperparams["lengthscales"], hyperparams["var_w"], hyperparams["Z"]]

    if not spec.get("freeze_mu_u"):
        for u in u_params:
            mu_fast += [u["mu_u"], u["Sigma_u"]]

    if not _freeze_W_at_step(spec, step):
        W_slow.extend(proj_trainable_tensors(proj))

    if not spec.get("freeze_mj"):
        for u in u_params:
            if "m_j" in u and u["m_j"].requires_grad:
                slow_other.append(u["m_j"])

    for u in u_params:
        slow_other.append(u["sigma_k"])
        if u["sigma_noise"].requires_grad and not spec.get("freeze_sigma_n"):
            slow_other.append(u["sigma_noise"])
        if spec.get("use_mixture_elbo") and not spec.get("freeze_gate"):
            if u.get("omega") is not None and u["omega"].requires_grad:
                slow_other.append(u["omega"])
            if spec.get("fix_u") is None and u.get("U_logit") is not None and u["U_logit"].requires_grad:
                slow_other.append(u["U_logit"])
    return mu_fast, W_slow, slow_other


def _compute_elbo(
    regions,
    proj,
    u_params,
    hyperparams,
    spec,
    step: int,
    num_steps: int,
    V_dummy,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    beta_u = training_beta_u_kl(spec, step, num_steps)
    gate_cfg = spec.get("gate", {"gate_mode": "learned"})
    elbo_kw = dict(
        regions=regions,
        V_params=V_dummy,
        u_params=u_params,
        hyperparams=hyperparams,
        proj=proj,
        fixed_w=False,
        use_mc_kfu=bool(spec.get("use_mc_kfu")),
        mc_samples=int(spec.get("mc_samples", 5)),
        beta_u_kl=beta_u,
        lambda_sigma_prior=float(spec.get("lambda_sigma_prior", 0.0)),
        log_sigma_n0=float(spec.get("log_sigma_n0", 0.0)),
    )
    if mixture_elbo_uses_mixture(spec, step):
        lam_anchor = lambda_rho_anchor_schedule(spec, step, num_steps)
        alpha_b = rho_blend_alpha(spec, step, num_steps)
        cap = spec.get("rho_blend_alpha_cap")
        cap_until = spec.get("rho_blend_alpha_cap_until_step")
        if cap is not None and cap_until is not None and step <= int(cap_until):
            alpha_b = min(float(alpha_b), float(cap))
        elbo = compute_elbo_mixture_mj(
            gate_config=gate_cfg,
            rho_floor_min=spec.get("rho_floor_min"),
            lambda_rho_floor=float(spec.get("lambda_rho_floor", 0.0)),
            rho_floor_per_region=bool(spec.get("rho_floor_per_region", False)),
            rho0=hyperparams.get("rho0"),
            lambda_rho_anchor=lam_anchor,
            rho_pointwise_min=spec.get("rho_pointwise_min"),
            lambda_rho_pointwise_floor=float(spec.get("lambda_rho_pointwise_floor", 0.0)),
            rho_blend_alpha=alpha_b,
            rho_point_blend_gamma=spec.get("rho_point_blend_gamma"),
            rho_blend_linear_data=bool(spec.get("rho_blend_linear_data")),
            **elbo_kw,
        )
    else:
        elbo = compute_elbo_pure_inlier_mj(**elbo_kw)
    sp = 0.0
    lam_s = float(spec.get("lambda_s", 0.0))
    if lam_s > 0:
        sp_t = compute_s_penalty_proj(regions, proj, hyperparams)
        sp = float((lam_s * sp_t).item())
        elbo = elbo - lam_s * sp_t
    align_parts: dict[str, float] = {}
    lam_cov = float(spec.get("lambda_cov", 0.0))
    lam_ridge = float(spec.get("lambda_ridge", 0.0))
    if lam_cov > 0 or lam_ridge > 0:
        align_add, align_parts = alignment_losses_train(
            regions, proj, hyperparams,
            lambda_cov=lam_cov,
            lambda_ridge=lam_ridge,
            ridge_lambda=float(spec.get("ridge_lambda", 1e-2)),
            cov_clip=float(spec.get("cov_clip", 1.0)),
            detach_inducing=bool(spec.get("align_proj_only", True)),
        )
        elbo = elbo + align_add
    lam_w = float(spec.get("lambda_W_trust", 0.0))
    if lam_w > 0 and proj.get("kind") == "lowrank":
        mu_B0 = hyperparams.get("mu_B_init")
        if mu_B0 is not None:
            elbo = elbo - lam_w * (proj["mu_B"] - mu_B0).pow(2).sum()
    return elbo, sp, align_parts


@torch.no_grad()
def _refresh_mu_u_star(
    regions,
    proj,
    u_params,
    hyperparams,
    V_dummy,
    spec,
    step: int,
    num_steps: int,
) -> None:
    if spec.get("freeze_mu_u"):
        return
    mc_kw = dict(
        use_mc_kfu=bool(spec.get("use_mc_kfu")),
        mc_samples=int(spec.get("mc_samples", 5)),
    )
    refresh_mode = spec.get("refresh_star", "pure")
    gate_cfg = spec.get("gate", {"gate_mode": "learned"})
    if refresh_mode == "tilde":
        _, _, rho_el = mixture_scores_rho_mj(
            regions, V_dummy, u_params, hyperparams,
            proj=proj, gate_config=gate_cfg, **mc_kw,
        )
        alpha_r = rho_blend_alpha(spec, step, num_steps)
        rho_tilde = compute_rho_tilde(
            rho_el, hyperparams.get("rho0"),
            alpha=alpha_r,
            point_gamma=spec.get("rho_point_blend_gamma"),
        )
        star = elbo_star_mu_u_custom_rho(
            regions, proj, hyperparams, u_params, rho_tilde, **mc_kw,
        )
    elif spec.get("use_mixture_elbo") and refresh_mode == "rho":
        star = elbo_star_mu_u_rho(
            regions, proj, hyperparams, u_params, V_dummy,
            gate_config=gate_cfg, **mc_kw,
        )
    else:
        star = elbo_star_mu_u(regions, proj, hyperparams, u_params, **mc_kw)
    refresh_mu_u_toward_star(
        u_params, star["mu_star"], float(spec.get("refresh_alpha", 0.3)),
    )


def _post_step_clamp(proj, hyperparams, u_params, spec) -> None:
    clamp_projection(proj, spec.get("sigma_max"))
    hyperparams["var_w"].clamp_(min=0.1)
    hyperparams["lengthscales"].clamp_(min=1e-6, max=1e2)
    for u in u_params:
        u["sigma_k"].clamp_(min=0.1)


@torch.no_grad()
def _test_anchor_correction_metrics(
    m_j: torch.Tensor,
    zeta: torch.Tensor,
    y_work: torch.Tensor,
    *,
    scaler_y,
) -> dict[str, float]:
    """Test-anchor residual alignment: ``e0 = y* - m_j``, correction ``zeta_*``."""
    e0 = y_work - m_j
    sse0 = e0.pow(2).sum()
    sse1 = (e0 - zeta).pow(2).sum()
    gain = (sse0 - sse1) / sse0.clamp_min(1e-8)
    cos = (e0 * zeta).sum() / (e0.norm() * zeta.norm()).clamp_min(1e-8)
    out = {
        "test_sse_gain_zeta": float(gain.item()),
        "test_e0_dot_zeta_sum": float((e0 * zeta).sum().item()),
        "test_e0_dot_zeta_mean": float((e0 * zeta).mean().item()),
        "test_zeta_sq_mean": float(zeta.pow(2).mean().item()),
        "test_e0_sq_mean": float(e0.pow(2).mean().item()),
        "test_zeta_over_e0_norm": float(zeta.norm().item() / e0.norm().clamp_min(1e-8).item()),
        "test_cos_e0_zeta": float(cos.item()),
        "rmse_mj_only_work": float(torch.sqrt(e0.pow(2).mean()).item()),
    }
    if scaler_y is not None:
        scale = float(scaler_y.scale_[0])
        mean = float(scaler_y.mean_[0])
        e0_raw = e0 * scale
        zeta_raw = zeta * scale
        y_raw = y_work * scale + mean
        m_j_raw = m_j * scale + mean
        sse0r = e0_raw.pow(2).sum()
        sse1r = (y_raw - m_j_raw - zeta_raw).pow(2).sum()
        out["test_sse_gain_zeta_raw"] = float(((sse0r - sse1r) / sse0r.clamp_min(1e-8)).item())
        out["rmse_mj_only_raw"] = float(torch.sqrt(e0_raw.pow(2).mean()).item())
    return out


@torch.no_grad()
def _predict_samplew_cem_mj(
    regions,
    proj,
    hyperparams,
    *,
    M: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``W`` from ``q(W)`` and JumpGP-CEM per anchor (work-scale ``y`` in regions)."""
    W_samples = sample_W_from_proj(proj, hyperparams, M=M, seed=seed)
    X_query = hyperparams["X_test"]
    out = run_projected_jumpgp_ensemble(
        W_samples,
        X_query,
        [r["X"] for r in regions],
        [r["y"] for r in regions],
        device=X_query.device,
        progress=False,
        desc="sampleW_cem_mj",
    )
    mu = torch.as_tensor(out["mu"], device=X_query.device, dtype=X_query.dtype)
    var = torch.as_tensor(out["sigma"], device=X_query.device, dtype=X_query.dtype).pow(2)
    return mu, var


@torch.no_grad()
def _rmse_cem_work(
    regions,
    proj,
    hyperparams,
    y_test_work: np.ndarray,
    *,
    M: int,
    seed: int,
) -> float:
    """Cheap work-scale RMSE from sample-W + CEM (for acceptance / monitoring)."""
    if proj.get("kind") != "lowrank":
        return float("nan")
    mu_cem, _ = _predict_samplew_cem_mj(regions, proj, hyperparams, M=M, seed=seed)
    device = mu_cem.device
    T = len(regions)
    yw = torch.from_numpy(y_test_work.astype(np.float32)).to(device)[:T]
    return float(torch.sqrt(((mu_cem - yw) ** 2).mean()).item())


@torch.no_grad()
def _direct_pred_rmse_work(
    regions,
    proj,
    u_params,
    hyperparams,
    V_dummy,
    y_test_work: np.ndarray,
    spec: dict[str, Any],
) -> float:
    """Direct analytic head RMSE on standardized ``y`` (work scale)."""
    mu_y, _, _ = predict_inlier_mj(
        regions, V_dummy, u_params, hyperparams,
        proj=proj,
        use_mc_kfu=bool(spec.get("use_mc_kfu")),
        mc_samples=int(spec.get("mc_samples", 5)),
    )
    device = mu_y.device
    T = len(regions)
    yw = torch.from_numpy(y_test_work.astype(np.float32)).to(device)[:T]
    return float(torch.sqrt(((mu_y - yw) ** 2).mean()).item())


def _line_tau_grid(spec: dict[str, Any], step: int, num_steps: int) -> tuple[float, ...]:
    decay_at = spec.get("line_tau_decay_step")
    if decay_at is not None and step > int(decay_at):
        g = spec.get("line_tau_grid_late")
        if g is not None:
            return tuple(float(x) for x in g)
    g = spec.get("line_tau_grid_early", (0.0, 0.25, 0.5, 1.0, 2.0, 4.0))
    return tuple(float(x) for x in g)


def _line_w_base(**over: Any) -> dict[str, Any]:
    """W-gradient line search on ``mu_B``/``sigma_B`` + inner ``mu_u`` adaptation."""
    base = dict(
        rho_blend_end_step=600,
        refresh_star="tilde",
        refresh_alpha=0.3,
        refresh_mu_u_every=0,
        w_line_search=True,
        line_inner_mu=20,
        line_cand_inner_mu=0,
        line_eta_W_scale=1.0,
        line_score_mode="elbo",
        line_tau_grid_early=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
        line_tau_grid_late=(0.0, 0.25, 0.5, 1.0),
        line_tau_decay_step=None,
        lambda_W_trust=0.0,
        rho_blend_alpha_cap=0.3,
        rho_blend_alpha_cap_until_step=200,
        eval_sampleW_cem=True,
        sampleW_M=8,
        slow_update_every=1,
    )
    base.update(over)
    return _warm120_blend_uniform(0.5, **base)


def _explore_w_profile_schedules(spec: dict[str, Any], step: int, num_steps: int) -> dict[str, Any]:
    """Piecewise phase A/B/C: inner ``mu_u`` count, ``lr_W`` multiplier, freeze flags."""
    t_a = int(spec.get("explore_phase_a_end", 80))
    t_b = int(spec.get("explore_phase_b_end", 300))
    ia = int(spec.get("explore_inner_mu_a", 20))
    ib = int(spec.get("explore_inner_mu_b", 8))
    ic = int(spec.get("explore_inner_mu_c", 5))
    wa = float(spec.get("explore_lr_W_a", 1.0))
    wb = float(spec.get("explore_lr_W_b", 0.1))
    wc = float(spec.get("explore_lr_W_c", 0.03))
    freeze_w_c = bool(spec.get("explore_freeze_W_phase_c", True))
    if step <= t_a:
        return {
            "inner": ia,
            "lr_W_mult": wa,
            "freeze_mj": bool(spec.get("explore_phase_a_freeze_mj", True)),
            "freeze_gate": bool(spec.get("explore_phase_a_freeze_gate", True)),
            "freeze_w": False,
            "phase": "A",
        }
    if step <= t_b:
        span = max(t_b - t_a, 1)
        u = (step - t_a) / span
        inner = int(round(ia + u * (ib - ia)))
        lr_w_mult = wa + u * (wb - wa)
        return {
            "inner": max(inner, 1),
            "lr_W_mult": float(lr_w_mult),
            "freeze_mj": False,
            "freeze_gate": False,
            "freeze_w": False,
            "phase": "B",
        }
    return {
        "inner": ic,
        "lr_W_mult": wc if not freeze_w_c else 0.0,
        "freeze_mj": bool(spec.get("explore_phase_c_freeze_mj", False)),
        "freeze_gate": bool(spec.get("explore_phase_c_freeze_gate", False)),
        "freeze_w": freeze_w_c,
        "phase": "C",
    }


def _step_spec_for_explore(spec: dict[str, Any], sched: dict[str, Any], step: int) -> dict[str, Any]:
    out = dict(spec)
    out["freeze_mj"] = sched["freeze_mj"]
    out["freeze_gate"] = sched["freeze_gate"]
    if sched["freeze_w"] or _freeze_W_at_step(spec, step):
        out["freeze_W"] = True
    if sched["phase"] == "A" and spec.get("explore_phase_a_freeze_sigma_n", True):
        out["freeze_sigma_n"] = True
    return out


def _clone_proj(proj):
    out = {"kind": proj["kind"], "W0": proj["W0"].detach().clone()}
    if proj.get("sigma_max") is not None:
        out["sigma_max"] = proj["sigma_max"]
    if proj["kind"] in ("full", "full_centered"):
        out["mu_V"] = proj["mu_V"].detach().clone().requires_grad_(True)
        out["sigma_V"] = proj["sigma_V"].detach().clone().requires_grad_(True)
    else:
        out["U"] = proj["U"].detach().clone()
        out["mu_B"] = proj["mu_B"].detach().clone().requires_grad_(True)
        out["sigma_B"] = proj["sigma_B"].detach().clone().requires_grad_(True)
    return out


def _train_variant(
    variant,
    spec,
    regions,
    proj,
    u_params,
    hyperparams,
    *,
    lr,
    num_steps,
    eval_every,
    y_test_work,
    y_test_raw,
    scaler_y,
    seed: int = 0,
):
    if spec.get("w_line_search"):
        return _train_variant_w_line_search(
            variant, spec, regions, proj, u_params, hyperparams,
            lr=lr, num_steps=num_steps, eval_every=eval_every,
            y_test_work=y_test_work, y_test_raw=y_test_raw, scaler_y=scaler_y, seed=seed,
        )
    if spec.get("explore_w_profile"):
        return _train_variant_explore_w_profile(
            variant, spec, regions, proj, u_params, hyperparams,
            lr=lr, num_steps=num_steps, eval_every=eval_every,
            y_test_work=y_test_work, y_test_raw=y_test_raw, scaler_y=scaler_y, seed=seed,
        )
    if spec.get("two_timescale"):
        return _train_variant_two_timescale(
            variant, spec, regions, proj, u_params, hyperparams,
            lr=lr, num_steps=num_steps, eval_every=eval_every,
            y_test_work=y_test_work, y_test_raw=y_test_raw, scaler_y=scaler_y, seed=seed,
        )
    return _train_variant_joint(
        variant, spec, regions, proj, u_params, hyperparams,
        lr=lr, num_steps=num_steps, eval_every=eval_every,
        y_test_work=y_test_work, y_test_raw=y_test_raw, scaler_y=scaler_y, seed=seed,
    )


def _train_variant_joint(
    variant,
    spec,
    regions,
    proj,
    u_params,
    hyperparams,
    *,
    lr,
    num_steps,
    eval_every,
    y_test_work,
    y_test_raw,
    scaler_y,
    seed: int = 0,
):
    opt = None
    freeze_w_prev: bool | None = None
    records = []
    y_std = float(np.std(y_test_work))
    V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}
    _print_rho0_quality(hyperparams)

    for step in range(1, num_steps + 1):
        freeze_w_now = _freeze_W_at_step(spec, step)
        if opt is None or freeze_w_now != freeze_w_prev:
            params = _collect_trainable_params(proj, u_params, hyperparams, spec, step)
            opt = torch.optim.Adam(params, lr=lr)
            freeze_w_prev = freeze_w_now
            if step > 1:
                print(f"  [{variant}] optimizer rebuilt (freeze_W={freeze_w_now})", flush=True)
        opt.zero_grad()
        elbo, sp, align_parts = _compute_elbo(
            regions, proj, u_params, hyperparams, spec, step, num_steps, V_dummy,
        )
        (-elbo).backward()
        opt.step()
        with torch.no_grad():
            every = int(spec.get("refresh_mu_u_every", 0) or 0)
            if every > 0 and step % every == 0:
                _refresh_mu_u_star(
                    regions, proj, u_params, hyperparams, V_dummy, spec, step, num_steps,
                )
            _post_step_clamp(proj, hyperparams, u_params, spec)
            apply_sigma_n_control(u_params, spec, step)

        _maybe_log_step(variant, step, num_steps, spec, elbo, sp, align_parts, eval_every)
        if step == 1 or step % eval_every == 0 or step == num_steps:
            records.append(
                _evaluate(
                    variant, regions, proj, u_params, hyperparams, V_dummy,
                    y_test_work, y_test_raw, scaler_y, step, y_std, spec, seed=seed,
                ),
            )
    return records


def _train_variant_two_timescale(
    variant,
    spec,
    regions,
    proj,
    u_params,
    hyperparams,
    *,
    lr,
    num_steps,
    eval_every,
    y_test_work,
    y_test_raw,
    scaler_y,
    seed: int = 0,
):
    """Block coordinate: inner ``mu_u`` steps, then slow ``W``, then occasional slow params."""
    records = []
    y_std = float(np.std(y_test_work))
    V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}
    inner_n = int(spec.get("inner_mu_steps", 5))
    lr_w = lr * float(spec.get("lr_W_scale", 0.1))
    slow_every = int(spec.get("slow_update_every", 1) or 1)
    opt_mu = opt_w = opt_slow = None
    freeze_w_prev: bool | None = None
    _print_rho0_quality(hyperparams)
    print(
        f"  [{variant}] two-timescale: inner_mu={inner_n}  lr_W={lr_w:.4g}  "
        f"trust={spec.get('lambda_W_trust', 0.0)}  slow_every={slow_every}",
        flush=True,
    )

    for step in range(1, num_steps + 1):
        freeze_w_now = _freeze_W_at_step(spec, step)
        if freeze_w_now != freeze_w_prev:
            mu_f, w_s, slow = _collect_param_groups(
                proj, u_params, hyperparams, spec, step,
            )
            opt_mu = torch.optim.Adam(mu_f, lr=lr) if mu_f else None
            opt_w = torch.optim.Adam(w_s, lr=lr_w) if w_s else None
            opt_slow = torch.optim.Adam(slow, lr=lr) if slow else None
            freeze_w_prev = freeze_w_now
            if step > 1:
                print(f"  [{variant}] TT optimizers rebuilt (freeze_W={freeze_w_now})", flush=True)
        elif opt_mu is None:
            mu_f, w_s, slow = _collect_param_groups(
                proj, u_params, hyperparams, spec, step,
            )
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
    return records


def _train_variant_explore_w_profile(
    variant,
    spec,
    regions,
    proj,
    u_params,
    hyperparams,
    *,
    lr,
    num_steps,
    eval_every,
    y_test_work,
    y_test_raw,
    scaler_y,
    seed: int = 0,
):
    """Phase A/B/C W exploration with inner ``mu_u`` adaptation, decaying ``lr_W``, optional CEM accept."""
    records = []
    y_std = float(np.std(y_test_work))
    V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}
    slow_every = int(spec.get("slow_update_every", 1) or 1)
    opt_mu = opt_w = opt_slow = None
    prev_sig: tuple[Any, ...] | None = None
    _print_rho0_quality(hyperparams)
    print(
        f"  [{variant}] explore_w_profile: A<= {spec.get('explore_phase_a_end', 80)}  "
        f"B<= {spec.get('explore_phase_b_end', 300)}  rho_cap={spec.get('rho_blend_alpha_cap')}",
        flush=True,
    )

    for step in range(1, num_steps + 1):
        sched = _explore_w_profile_schedules(spec, step, num_steps)
        step_spec = _step_spec_for_explore(spec, sched, step)
        inner_n = int(sched["inner"])
        lr_w_mult = float(sched["lr_W_mult"])
        sig = (
            inner_n,
            lr_w_mult,
            bool(step_spec.get("freeze_W")),
            bool(step_spec.get("freeze_mj")),
            bool(step_spec.get("freeze_gate")),
            sched["phase"],
        )
        if sig != prev_sig or opt_mu is None:
            mu_f, w_s, slow = _collect_param_groups(
                proj, u_params, hyperparams, step_spec, step,
            )
            lr_w = lr * lr_w_mult
            opt_mu = torch.optim.Adam(mu_f, lr=lr) if mu_f else None
            opt_w = torch.optim.Adam(w_s, lr=lr_w) if w_s else None
            opt_slow = torch.optim.Adam(slow, lr=lr) if slow else None
            prev_sig = sig
            if step > 1:
                print(
                    f"  [{variant}] EX rebuild phase={sched['phase']} inner={inner_n}  "
                    f"lr_W={lr_w:.4g}  freeze_W={step_spec.get('freeze_W')}",
                    flush=True,
                )

        elbo_last = None
        sp_last = 0.0
        align_parts: dict[str, float] = {}

        for _ in range(inner_n):
            if opt_mu is None:
                break
            opt_mu.zero_grad()
            elbo, sp_last, align_parts = _compute_elbo(
                regions, proj, u_params, hyperparams, step_spec, step, num_steps, V_dummy,
            )
            (-elbo).backward()
            opt_mu.step()
            elbo_last = elbo

        with torch.no_grad():
            if spec.get("refresh_before_W", True):
                _refresh_mu_u_star(
                    regions, proj, u_params, hyperparams, V_dummy, step_spec, step, num_steps,
                )

        if opt_w is not None:
            do_accept = bool(spec.get("explore_accept_cem")) and (
                step % int(spec.get("explore_accept_every", 20)) == 0
            )
            mu_b_bak = sig_b_bak = None
            rmse_before = float("nan")
            if do_accept:
                rmse_before = _rmse_cem_work(
                    regions, proj, hyperparams, y_test_work,
                    M=int(spec.get("explore_accept_MC", 3)),
                    seed=seed + step * 7919,
                )
                mu_b_bak = proj["mu_B"].detach().clone()
                sig_b_bak = proj["sigma_B"].detach().clone()
            opt_w.zero_grad()
            elbo, sp_last, align_parts = _compute_elbo(
                regions, proj, u_params, hyperparams, step_spec, step, num_steps, V_dummy,
            )
            (-elbo).backward()
            gclip = spec.get("explore_grad_clip_norm_W")
            if gclip is not None and proj["kind"] == "lowrank":
                torch.nn.utils.clip_grad_norm_(proj_trainable_tensors(proj), float(gclip))
            opt_w.step()
            elbo_last = elbo
            if do_accept and mu_b_bak is not None:
                rmse_after = _rmse_cem_work(
                    regions, proj, hyperparams, y_test_work,
                    M=int(spec.get("explore_accept_MC", 3)),
                    seed=seed + step * 7919,
                )
                tol = float(spec.get("explore_accept_tolerance", 0.0))
                if rmse_after == rmse_after and rmse_before == rmse_before and rmse_after > rmse_before + tol:
                    proj["mu_B"].copy_(mu_b_bak)
                    proj["sigma_B"].copy_(sig_b_bak)
                    print(
                        f"  [{variant}] EX CEM-reject step={step}  "
                        f"RMSE_cem {rmse_before:.4f} -> {rmse_after:.4f}",
                        flush=True,
                    )

        with torch.no_grad():
            if spec.get("refresh_after_W"):
                _refresh_mu_u_star(
                    regions, proj, u_params, hyperparams, V_dummy, step_spec, step, num_steps,
                )

        if opt_slow is not None and step % slow_every == 0:
            opt_slow.zero_grad()
            elbo, sp_last, align_parts = _compute_elbo(
                regions, proj, u_params, hyperparams, step_spec, step, num_steps, V_dummy,
            )
            (-elbo).backward()
            opt_slow.step()
            elbo_last = elbo

        with torch.no_grad():
            _post_step_clamp(proj, hyperparams, u_params, spec)
            apply_sigma_n_control(u_params, spec, step)

        if elbo_last is not None:
            _maybe_log_step(
                variant, step, num_steps, step_spec, elbo_last, sp_last, align_parts, eval_every,
                tt_tag=f"  EX(phase={sched['phase']},inner={inner_n},lrW_mult={lr_w_mult:.2f})",
            )

        if step == 1 or step % eval_every == 0 or step == num_steps:
            records.append(
                _evaluate(
                    variant, regions, proj, u_params, hyperparams, V_dummy,
                    y_test_work, y_test_raw, scaler_y, step, y_std, spec, seed=seed,
                ),
            )
    return records


def _snapshot_u_mu_sigma(u_params) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(u["mu_u"].detach().clone(), u["Sigma_u"].detach().clone()) for u in u_params]


def _restore_u_mu_sigma(u_params, snap: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    with torch.no_grad():
        for u, (m, s) in zip(u_params, snap):
            u["mu_u"].copy_(m)
            u["Sigma_u"].copy_(s)


def _train_variant_w_line_search(
    variant,
    spec,
    regions,
    proj,
    u_params,
    hyperparams,
    *,
    lr,
    num_steps,
    eval_every,
    y_test_work,
    y_test_raw,
    scaler_y,
    seed: int = 0,
):
    """Inner ``mu_u`` adaptation + line search along ``(mu_B,sigma_B)`` ELBO gradient."""
    records = []
    y_std = float(np.std(y_test_work))
    V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}
    inner_n = int(spec.get("line_inner_mu", 20))
    cand_inner = int(spec.get("line_cand_inner_mu", 0))
    post_inner = int(spec.get("line_post_best_inner", 2))
    slow_every = int(spec.get("slow_update_every", 1) or 1)
    inner_spec = dict(spec)
    inner_spec["freeze_W"] = True
    opt_mu = None
    opt_slow = None
    ls_taus: list[float] = []
    _print_rho0_quality(hyperparams)
    print(
        f"  [{variant}] w_line_search: inner_mu={inner_n}  score={spec.get('line_score_mode', 'elbo')}  "
        f"trust={spec.get('lambda_W_trust', 0.0)}",
        flush=True,
    )

    for step in range(1, num_steps + 1):
        step_spec = dict(spec)
        mu_f, _, _ = _collect_param_groups(proj, u_params, hyperparams, inner_spec, step)
        if opt_mu is None and mu_f:
            opt_mu = torch.optim.Adam(mu_f, lr=lr)

        elbo_last = None
        sp_last = 0.0
        align_parts: dict[str, float] = {}

        for _ in range(inner_n):
            if opt_mu is None:
                break
            opt_mu.zero_grad()
            elbo, sp_last, align_parts = _compute_elbo(
                regions, proj, u_params, hyperparams, inner_spec, step, num_steps, V_dummy,
            )
            (-elbo).backward()
            opt_mu.step()
            elbo_last = elbo

        with torch.no_grad():
            if spec.get("refresh_before_W", True):
                _refresh_mu_u_star(
                    regions, proj, u_params, hyperparams, V_dummy, inner_spec, step, num_steps,
                )

        u_ref = _snapshot_u_mu_sigma(u_params)
        mu_b0 = proj["mu_B"].detach().clone()
        sig0 = proj["sigma_B"].detach().clone()

        cem_every = spec.get("line_cem_accept_every")
        cem_before = float("nan")
        if cem_every is not None and step % int(cem_every) == 0:
            cem_before = _rmse_cem_work(
                regions, proj, hyperparams, y_test_work,
                M=int(spec.get("line_cem_accept_MC", spec.get("sampleW_M", 3))),
                seed=seed + step * 10007,
            )

        elbo_g, sp_g, al_g = _compute_elbo(
            regions, proj, u_params, hyperparams, step_spec, step, num_steps, V_dummy,
        )
        g_pair = torch.autograd.grad(
            elbo_g,
            (proj["mu_B"], proj["sigma_B"]),
            retain_graph=False,
            allow_unused=True,
        )
        gB, gS = g_pair
        if gB is None:
            gB = torch.zeros_like(proj["mu_B"])
        if gS is None:
            gS = torch.zeros_like(proj["sigma_B"])

        eta = lr * float(spec.get("line_eta_W_scale", 1.0))
        tau_grid = _line_tau_grid(spec, step, num_steps)
        best_tau = 0.0
        best_score = -1e38
        best_mu = mu_b0.clone()
        best_sig = sig0.clone()
        mu_b0_ref = hyperparams.get("mu_B_init")
        lam_tr = float(spec.get("lambda_W_trust", 0.0))
        mode = str(spec.get("line_score_mode", "elbo"))

        for tau in tau_grid:
            _restore_u_mu_sigma(u_params, u_ref)
            with torch.no_grad():
                proj["mu_B"].copy_(mu_b0 + tau * eta * gB)
                proj["sigma_B"].copy_(sig0 + tau * eta * gS)
            clamp_projection(proj, spec.get("sigma_max"))
            with torch.no_grad():
                _refresh_mu_u_star(
                    regions, proj, u_params, hyperparams, V_dummy, step_spec, step, num_steps,
                )
            for _ in range(max(cand_inner, 0)):
                if opt_mu is None:
                    break
                opt_mu.zero_grad()
                elbo_c, _, _ = _compute_elbo(
                    regions, proj, u_params, hyperparams, inner_spec, step, num_steps, V_dummy,
                )
                (-elbo_c).backward()
                opt_mu.step()
            with torch.no_grad():
                _refresh_mu_u_star(
                    regions, proj, u_params, hyperparams, V_dummy, step_spec, step, num_steps,
                )
            with torch.no_grad():
                if mode == "heldout":
                    rm = _direct_pred_rmse_work(
                        regions, proj, u_params, hyperparams, V_dummy, y_test_work, step_spec,
                    )
                    sc = -rm
                else:
                    sc = float(_compute_elbo(
                        regions, proj, u_params, hyperparams, step_spec, step, num_steps, V_dummy,
                    )[0].item())
                    if lam_tr > 0.0 and mu_b0_ref is not None and proj.get("kind") == "lowrank":
                        sc -= lam_tr * (proj["mu_B"] - mu_b0_ref).pow(2).sum().item()
            if sc > best_score:
                best_score = sc
                best_tau = float(tau)
                best_mu = proj["mu_B"].detach().clone()
                best_sig = proj["sigma_B"].detach().clone()

        ls_taus.append(best_tau)
        _restore_u_mu_sigma(u_params, u_ref)
        with torch.no_grad():
            proj["mu_B"].copy_(best_mu)
            proj["sigma_B"].copy_(best_sig)
        clamp_projection(proj, spec.get("sigma_max"))
        with torch.no_grad():
            _refresh_mu_u_star(
                regions, proj, u_params, hyperparams, V_dummy, step_spec, step, num_steps,
            )
        for _ in range(max(post_inner, 0)):
            if opt_mu is None:
                break
            opt_mu.zero_grad()
            elbo_p, sp_last, align_parts = _compute_elbo(
                regions, proj, u_params, hyperparams, inner_spec, step, num_steps, V_dummy,
            )
            (-elbo_p).backward()
            opt_mu.step()
            elbo_last = elbo_p
        with torch.no_grad():
            _refresh_mu_u_star(
                regions, proj, u_params, hyperparams, V_dummy, step_spec, step, num_steps,
            )

        if cem_every is not None and step % int(cem_every) == 0 and cem_before == cem_before:
            cem_after = _rmse_cem_work(
                regions, proj, hyperparams, y_test_work,
                M=int(spec.get("line_cem_accept_MC", spec.get("sampleW_M", 3))),
                seed=seed + step * 10007,
            )
            tol = float(spec.get("line_cem_accept_tolerance", 0.0))
            if cem_after == cem_after and cem_after > cem_before + tol:
                with torch.no_grad():
                    proj["mu_B"].copy_(mu_b0)
                    proj["sigma_B"].copy_(sig0)
                _restore_u_mu_sigma(u_params, u_ref)
                clamp_projection(proj, spec.get("sigma_max"))
                _refresh_mu_u_star(
                    regions, proj, u_params, hyperparams, V_dummy, step_spec, step, num_steps,
                )
                print(
                    f"  [{variant}] LINE CEM-reject step={step}  "
                    f"RMSE_cem {cem_before:.4f} -> {cem_after:.4f}",
                    flush=True,
                )

        slow_spec = dict(spec)
        slow_spec["freeze_mu_u"] = True
        slow_spec["freeze_W"] = True
        _, _, slowp = _collect_param_groups(proj, u_params, hyperparams, slow_spec, step)
        if opt_slow is None and slowp:
            opt_slow = torch.optim.Adam(slowp, lr=lr)
        if opt_slow is not None and step % slow_every == 0:
            opt_slow.zero_grad()
            elbo_s, sp_last, align_parts = _compute_elbo(
                regions, proj, u_params, hyperparams, step_spec, step, num_steps, V_dummy,
            )
            (-elbo_s).backward()
            opt_slow.step()
            elbo_last = elbo_s

        with torch.no_grad():
            _post_step_clamp(proj, hyperparams, u_params, spec)
            apply_sigma_n_control(u_params, spec, step)

        if elbo_last is not None:
            _maybe_log_step(
                variant, step, num_steps, step_spec, elbo_last, sp_last, align_parts, eval_every,
                tt_tag=f"  LINE(tau={best_tau:.2f})",
            )

        if step == 1 or step % eval_every == 0 or step == num_steps:
            records.append(
                _evaluate(
                    variant, regions, proj, u_params, hyperparams, V_dummy,
                    y_test_work, y_test_raw, scaler_y, step, y_std, spec, seed=seed,
                ),
            )

    if ls_taus:
        pz = sum(1 for t in ls_taus if t == 0.0) / len(ls_taus)
        print(
            f"  [{variant}] line_tau summary: mean={float(np.mean(ls_taus)):.4f}  "
            f"P(tau=0)={pz:.3f}  n={len(ls_taus)}",
            flush=True,
        )
    return records


def _print_rho0_quality(hyperparams) -> None:
    rq = hyperparams.get("rho0_quality")
    if rq:
        print(
            f"  rho0 quality: B0/Bp={rq.get('phiTr_rho0_over_pure', float('nan')):.3f}  "
            f"z*0/r={rq.get('ratio_zeta_star_rho0_r', float('nan')):.4f}  "
            f"corr0(|r|)={rq.get('corr_rho0_abs_r', float('nan')):.3f}",
            flush=True,
        )


def _maybe_log_step(
    variant,
    step,
    num_steps,
    spec,
    elbo,
    sp,
    align_parts,
    eval_every,
    *,
    tt_tag: str = "",
) -> None:
    if step % max(1, eval_every // 3) != 0 and step != 1:
        return
    extra = ""
    if align_parts:
        extra = "  " + "  ".join(f"{k}={v:.4f}" for k, v in align_parts.items())
    beta_u = training_beta_u_kl(spec, step, num_steps)
    bu_msg = f"  beta_u={beta_u:.3f}" if beta_u != 1.0 else ""
    warm_msg = "  pure_warm" if use_pure_inlier_warmup(spec, step) else ""
    anch = lambda_rho_anchor_schedule(spec, step, num_steps)
    anch_msg = f"  lam_rho_anc={anch:.1f}" if anch > 0 else ""
    ab = rho_blend_alpha(spec, step, num_steps)
    cap = spec.get("rho_blend_alpha_cap")
    cap_until = spec.get("rho_blend_alpha_cap_until_step")
    if cap is not None and cap_until is not None and step <= int(cap_until):
        ab = min(float(ab), float(cap))
    blend_msg = f"  alpha={ab:.2f}" if (
        spec.get("rho_blend") or spec.get("rho_point_blend_gamma") is not None
    ) else ""
    print(
        f"  [{variant} {step}/{num_steps}] ELBO={elbo.item():.4f} s_pen={sp:.4f}"
        f"{tt_tag}{bu_msg}{warm_msg}{blend_msg}{anch_msg}{extra}",
        flush=True,
    )


@torch.no_grad()
def _evaluate(
    variant,
    regions,
    proj,
    u_params,
    hyperparams,
    V_dummy,
    y_test_work,
    y_test_raw,
    scaler_y,
    step,
    y_std,
    spec,
    *,
    seed: int = 0,
):
    device = hyperparams["Z"].device
    T = len(regions)
    proj_eval = _proj_eval_copy(proj, use_W0=bool(spec.get("eval_with_W0")))
    mu_y, var_y, zeta = predict_inlier_mj(
        regions, V_dummy, u_params, hyperparams,
        proj=proj,
        use_mc_kfu=bool(spec.get("use_mc_kfu")),
        mc_samples=int(spec.get("mc_samples", 5)),
    )
    m_j = stack_m_j(u_params)
    y_work = torch.from_numpy(y_test_work.astype(np.float32)).to(device)[:T]
    anchor_diag = _test_anchor_correction_metrics(
        m_j, zeta, y_work, scaler_y=scaler_y,
    )
    w_drift = projection_w_drift_metrics(proj, hyperparams)
    if spec.get("eval_with_W0") and proj["kind"] == "lowrank":
        mu_w0, var_w0, _ = predict_inlier_mj(
            regions, V_dummy, u_params, hyperparams,
            proj=proj_eval,
            use_mc_kfu=bool(spec.get("use_mc_kfu")),
            mc_samples=int(spec.get("mc_samples", 5)),
        )
    else:
        mu_w0, var_w0 = None, None
    y_raw = torch.from_numpy(y_test_raw.astype(np.float32)).to(device)[:T]
    sigma_y = torch.sqrt(var_y.clamp_min(1e-12))
    if scaler_y is not None:
        y_scale = float(scaler_y.scale_[0])
        y_mean = float(scaler_y.mean_[0])
        mu_eval = mu_y * y_scale + y_mean
        sigma_eval = sigma_y * y_scale
    else:
        mu_eval = mu_y
        sigma_eval = sigma_y
    rmse = float(torch.sqrt(((mu_eval - y_raw) ** 2).mean()).item())
    crps = float(mean_gaussian_crps_torch(y_raw, mu_eval, sigma_eval).item())
    rmse_work = float(torch.sqrt(((mu_y - y_work) ** 2).mean()).item())
    crps_work = float(mean_gaussian_crps_torch(y_work, mu_y, sigma_y).item())
    rmse_raw = rmse
    if mu_w0 is not None:
        if scaler_y is not None:
            y_scale = float(scaler_y.scale_[0])
            y_mean = float(scaler_y.mean_[0])
            mu_w0_eval = mu_w0 * y_scale + y_mean
        else:
            mu_w0_eval = mu_w0
        rmse_eval_W0 = float(torch.sqrt(((mu_w0_eval - y_raw) ** 2).mean()).item())
    else:
        rmse_eval_W0 = float("nan")

    rmse_cem_work = float("nan")
    rmse_cem_raw = float("nan")
    crps_cem_raw = float("nan")
    crps_cem_work = float("nan")
    if spec.get("eval_sampleW_cem") and proj["kind"] == "lowrank":
        M = int(spec.get("sampleW_M", 3))
        mu_cem, var_cem = _predict_samplew_cem_mj(
            regions, proj, hyperparams, M=M, seed=seed + step,
        )
        sigma_cem = torch.sqrt(var_cem.clamp_min(1e-12))
        rmse_cem_work = float(torch.sqrt(((mu_cem - y_work) ** 2).mean()).item())
        crps_cem_work = float(mean_gaussian_crps_torch(y_work, mu_cem, sigma_cem).item())
        if scaler_y is not None:
            y_scale = float(scaler_y.scale_[0])
            y_mean = float(scaler_y.mean_[0])
            mu_cem_raw = mu_cem * y_scale + y_mean
            sigma_cem_raw = sigma_cem * y_scale
        else:
            mu_cem_raw = mu_cem
            sigma_cem_raw = sigma_cem
        rmse_cem_raw = float(torch.sqrt(((mu_cem_raw - y_raw) ** 2).mean()).item())
        crps_cem_raw = float(mean_gaussian_crps_torch(y_raw, mu_cem_raw, sigma_cem_raw).item())

    mom = gp_residual_moments(
        regions, V_dummy, u_params, hyperparams, proj=proj, at_test=True,
        use_mc_kfu=bool(spec.get("use_mc_kfu")),
        mc_samples=int(spec.get("mc_samples", 5)),
    )
    X_nb = torch.stack([r["X"] for r in regions], dim=0)
    psi = psi1_diagnostics(
        mom["mu_W"], X_nb, torch.stack([r["C"] for r in regions]),
        torch.stack([u["sigma_k"] for u in u_params]).view(-1),
    )
    ratio_z = float(zeta.std().item() / max(y_std, 1e-8))
    mom_tr = gp_residual_moments(
        regions, V_dummy, u_params, hyperparams, proj=proj, at_test=False,
        use_mc_kfu=bool(spec.get("use_mc_kfu")),
        mc_samples=int(spec.get("mc_samples", 5)),
    )
    zeta_nb = mom_tr["zeta"]
    y_train = torch.stack([r["y"] for r in regions], dim=0)
    r_elbo = y_train - m_j.view(-1, 1)
    y_res_std = float(r_elbo.std().item())
    sse_r = float(r_elbo.pow(2).sum().item())
    sse_res = float((r_elbo - zeta_nb).pow(2).sum().item())
    sse_gain = float((sse_r - sse_res) / max(sse_r, 1e-8))

    align: dict[str, float] = {}
    if spec.get("diag_tau", True):
        align = region_alignment_diagnostics(
            regions, proj, hyperparams, u_params,
            ridge_lambda=float(spec.get("ridge_lambda", 1e-2)),
            use_mc_kfu=bool(spec.get("use_mc_kfu")),
            mc_samples=int(spec.get("mc_samples", 5)),
        )
    shrink: dict[str, float] = {}
    if spec.get("diag_shrinkage", False):
        bu_eval = training_beta_u_kl(spec, step, max(step, 1))
        bu_star = bu_eval if (
            spec.get("beta_u_kl") is not None or spec.get("beta_u_warmup")
        ) else None
        shrink = elbo_star_metrics(
            regions, proj, hyperparams, u_params,
            beta_u_train=bu_star,
            use_mc_kfu=bool(spec.get("use_mc_kfu")),
            mc_samples=int(spec.get("mc_samples", 5)),
        )
        shrink.update(qu_diagnostics(
            regions, proj, hyperparams, u_params,
            use_mc_kfu=bool(spec.get("use_mc_kfu")),
            mc_samples=int(spec.get("mc_samples", 5)),
        ))
        if spec.get("use_mixture_elbo"):
            gate_cfg = spec.get("gate", {"gate_mode": "learned"})
            rho_tilde_eval = None
            if spec.get("rho_blend") or spec.get("rho_point_blend_gamma") is not None:
                _, _, rho_el = mixture_scores_rho_mj(
                    regions, V_dummy, u_params, hyperparams,
                    proj=proj, gate_config=gate_cfg,
                    use_mc_kfu=bool(spec.get("use_mc_kfu")),
                    mc_samples=int(spec.get("mc_samples", 5)),
                )
                alpha_e = rho_blend_alpha(spec, step, max(step, 1))
                rho_tilde_eval = compute_rho_tilde(
                    rho_el, hyperparams.get("rho0"),
                    alpha=alpha_e,
                    point_gamma=spec.get("rho_point_blend_gamma"),
                )
            mix_star = mixture_elbo_star_metrics(
                regions, proj, hyperparams, u_params, V_dummy,
                gate_config=gate_cfg,
                rho_tilde=rho_tilde_eval,
                use_mc_kfu=bool(spec.get("use_mc_kfu")),
                mc_samples=int(spec.get("mc_samples", 5)),
            )
            shrink.update(mix_star)
            shrink["ratio_zeta_star_pure_r"] = shrink.get(
                "ratio_zeta_star_r", mix_star["ratio_zeta_star_pure_r"],
            )

    mix_diag: dict[str, float] = {}
    if spec.get("diag_mixture", False):
        gate_cfg = spec.get("gate", {"gate_mode": "learned"})
        mix_diag = mixture_diagnostics_mj(
            regions, V_dummy, u_params, hyperparams,
            proj=proj, gate_config=gate_cfg,
        )
        mix_diag["std_m_j"] = float(m_j.std().item())
        mix_diag["rho_batch_mean"] = mix_diag.get("r_mean", float("nan"))
        rho_min = spec.get("rho_floor_min")
        if rho_min is not None:
            mix_diag["rho_floor_gap"] = max(
                0.0, float(rho_min) - float(mix_diag.get("r_mean", 0.0)),
            )

    rec = {
        "variant": variant,
        "step": step,
        "rmse": rmse,
        "rmse_raw": rmse_raw,
        "rmse_work": rmse_work,
        "crps": crps,
        "crps_work": crps_work,
        "pred_scale": "y_raw",
        "elbo_scale": "per_region",
        "std_zeta": float(zeta.std().item()),
        "std_mu_y": float(mu_y.std().item()),
        "ratio_zeta_yres": float(zeta.std().item() / max(y_res_std, 1e-8)),
        "ratio_zeta_ywork": ratio_z,
        "sse_gain_zeta": sse_gain,
        "mean_m_j": float(m_j.mean().item()),
        "std_m_W": float(mom["mu_W"].std().item()),
        "rmse_eval_W0": rmse_eval_W0,
        "rmse_cem_work": rmse_cem_work,
        "rmse_cem_raw": rmse_cem_raw,
        "crps_cem_raw": crps_cem_raw,
        "crps_cem_work": crps_cem_work,
        **anchor_diag,
        **w_drift,
        **psi,
        **align,
        **shrink,
        **mix_diag,
    }
    zbu = rec.get("ratio_zeta_star_beta_u_r", float("nan"))
    zbu_s = f"  z*_bu/r={zbu:.4f}" if zbu == zbu else ""
    z_pure = rec.get("ratio_zeta_star_pure_r", rec.get("ratio_zeta_elbo_star_r", float("nan")))
    z_rho = rec.get("ratio_zeta_star_rho_r", float("nan"))
    z_tilde = rec.get("ratio_zeta_star_rho_tilde_r", float("nan"))
    z_rho_s = f"  z*_rho/r={z_rho:.4f}" if z_rho == z_rho else ""
    z_tilde_s = f"  z*_tilde/r={z_tilde:.4f}" if z_tilde == z_tilde else ""
    rho_sig_s = ""
    if spec.get("use_mixture_elbo") and rec.get("phiTr_rho_over_rhobar_pure") == rec.get("phiTr_rho_over_rhobar_pure"):
        rho_sig_s = (
            f"  Br/Bp={rec.get('phiTr_rho_over_pure', float('nan')):.3f}"
            f" corr(rho,|r|)={rec.get('corr_rho_abs_r', float('nan')):.3f}"
            f" corr(rho,zr)={rec.get('corr_rho_abs_zridge', float('nan')):.3f}"
        )
    cem_s = ""
    if rec.get("rmse_cem_raw") == rec.get("rmse_cem_raw"):
        cem_s = (
            f"  RMSE_cem={rec['rmse_cem_raw']:.3f}  CRPS_cem={rec.get('crps_cem_raw', float('nan')):.3f}"
        )
    w0_s = ""
    if rec.get("rmse_eval_W0") == rec.get("rmse_eval_W0"):
        w0_s = f"  RMSE@W0={rec['rmse_eval_W0']:.3f}"
    print(
        f"    >>> [{variant} step={step}] RMSE_raw={rmse:.4f} CRPS_raw={crps:.4f}  "
        f"RMSE_mj={rec.get('rmse_mj_only_raw', rec.get('rmse_mj_only_work', float('nan'))):.3f}  "
        f"test_SSEg={rec.get('test_sse_gain_zeta_raw', rec.get('test_sse_gain_zeta', float('nan'))):.4f}  "
        f"cos(e0,z)={rec.get('test_cos_e0_zeta', float('nan')):.3f}  "
        f"W_drift={rec.get('W_drift_frob_mean', float('nan')):.4f}  "
        f"RMSE_work={rmse_work:.4f}  "
        f"train_SSEg={rec.get('sse_gain_zeta', float('nan')):.4f}  "
        f"sig_n={rec.get('sigma_n_mean', float('nan')):.3f}  "
        f"z_learn/r={rec.get('ratio_zeta_learned_r_elbo', float('nan')):.4f}  "
        f"z*_pure/r={z_pure:.4f}{z_tilde_s}{z_rho_s}{rho_sig_s}  "
        f"cos_u(pure)={rec.get('mu_u_cos_star_pure', float('nan')):.3f}  "
        f"rho={mix_diag.get('r_mean', float('nan')):.3f}  "
        f"dT={mix_diag.get('delta_T_mean', float('nan')):.1f}  "
        f"|K^-1mu|={rec.get('Kinvt_mu_learned_norm_mean', float('nan')):.3f}"
        f"{w0_s}{cem_s}",
        flush=True,
    )
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description="Remedy D structured projection VI.")
    p.add_argument("--setting", default="lh_q5_train500")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--num_steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--variants", default="C0_mj,D1,D2,D3,D4")
    p.add_argument("--standardize_y", action="store_true", default=True)
    p.add_argument("--no_standardize_y", action="store_true")
    p.add_argument("--w0_init", default=None, help="Override W0 mode for all D variants.")
    p.add_argument("--out_dir", default="experiments/synthetic/remedy_d_structured_vi")
    p.add_argument(
        "--eval_sampleW_cem",
        action="store_true",
        help="At checkpoints, also run sample-W + JumpGP-CEM prediction.",
    )
    p.add_argument("--MC_num", type=int, default=3, help="W samples for --eval_sampleW_cem.")
    p.add_argument(
        "--eval_with_W0",
        action="store_true",
        help="Also eval direct head with mu_B=0 (E[W]=W0) at checkpoints.",
    )
    args = p.parse_args()
    if args.no_standardize_y:
        args.standardize_y = False

    all_specs = _variant_specs()
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    cfg = SETTING_PRESETS[args.setting]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    _set_seed(args.seed)
    data = _generate_paper_data(cfg, args.seed, device)
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(data["X_train"])
    X_test_s = scaler_x.transform(data["X_test"])
    y_train_raw = np.asarray(data["y_train"], dtype=np.float64)
    y_test_raw = np.asarray(data["y_test"], dtype=np.float64)[: args.max_test]
    X_test_s = X_test_s[: args.max_test]

    y_train = y_train_raw.copy()
    y_test_work = y_test_raw.copy()
    scaler_y = None
    if args.standardize_y:
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_work = scaler_y.transform(y_test_work.reshape(-1, 1)).ravel()

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)
    Q, n = int(cfg["Q"]), int(cfg["n"])

    all_records = []
    for vname in want:
        spec = dict(all_specs[vname])
        if args.w0_init and "w0_init" in spec:
            spec["w0_init"] = args.w0_init
        if args.eval_sampleW_cem:
            spec["eval_sampleW_cem"] = True
        if spec.get("eval_sampleW_cem"):
            spec["sampleW_M"] = args.MC_num
        if args.eval_with_W0:
            spec["eval_with_W0"] = True
        print(f"\n{'='*70}\n  {vname}: {json.dumps(spec)}\n{'='*70}", flush=True)
        regions, proj, u_params, hyperparams, _ = _build_base(
            X_train_t, y_train_t, X_test_t,
            Q=Q, n=n, m1=args.m1, m2=args.m2, device=device, seed=args.seed,
            y_train_np=y_train,
            spec=spec,
        )
        recs = _train_variant(
            vname, spec, _clone_regions(regions), _clone_proj(proj), u_params, hyperparams,
            lr=args.lr, num_steps=args.num_steps, eval_every=args.eval_every,
            y_test_work=y_test_work, y_test_raw=y_test_raw, scaler_y=scaler_y,
            seed=args.seed,
        )
        all_records.extend(recs)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_records for k in r})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_records)

    print(f"\n{'='*80}\nFINAL (z ratios on r_elbo = y - m_j)\n{'='*80}")
    hdr = (
        f"{'Variant':<36s} {'z_learn':>8s} {'tSSEg':>6s} {'trSSEg':>7s} "
        f"{'RMSE':>6s} {'RMSEmj':>7s} {'Wdrft':>6s} {'rho':>6s}"
    )
    print(hdr)
    for v in want:
        r = next(x for x in reversed(all_records) if x["variant"] == v)
        rho_m = r.get("rho_tilde_mean", r.get("rho_batch_mean", r.get("r_mean", float("nan"))))
        tsse = r.get("test_sse_gain_zeta_raw", r.get("test_sse_gain_zeta", float("nan")))
        print(
            f"{v:<36s} "
            f"{r.get('ratio_zeta_learned_r_elbo', float('nan')):8.4f} "
            f"{tsse:6.3f} "
            f"{r.get('sse_gain_zeta', float('nan')):7.3f} "
            f"{r.get('rmse', float('nan')):6.3f} "
            f"{r.get('rmse_mj_only_raw', r.get('rmse_mj_only_work', float('nan'))):7.3f} "
            f"{r.get('W_drift_frob_mean', float('nan')):6.4f} "
            f"{rho_m:6.3f}"
        )
    _print_rmse_crps_checkpoints(all_records, want)
    print(f"\nSaved {out / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()

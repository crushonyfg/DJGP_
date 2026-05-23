r"""Run MC-CEM q(R) M-step variants on paper-style LH synthetic data.

How to run from the repository root:

1. `conda activate jumpGP`
2. `cd C:\Users\yxu59\files\spring2026\park\DJGP`
3. Smoke test:

   `python experiments/synthetic/compare_mc_cem_qr_lh.py --max_test 8 --mc_steps 1 --mc_samples 2 --cem_updates 1 --hyper_steps 1 --gate_steps 1 --final_ensemble_samples 2 --variants A_uniform,B_std --out_dir experiments/synthetic/mc_cem_qr_lh_smoke`

4. Pilot LH run:

   `python experiments/synthetic/compare_mc_cem_qr_lh.py --max_test 40 --mc_steps 4 --mc_samples 3 --cem_updates 1 --hyper_steps 4 --gate_steps 6 --final_ensemble_samples 3 --out_dir experiments/synthetic/mc_cem_qr_lh_seed0_20260518`

5. Per-step optimization diagnostic with batched hyperparameter updates:

   `python experiments/synthetic/compare_mc_cem_qr_lh.py --max_test 40 --mc_steps 5 --mc_samples 3 --cem_updates 1 --hyper_steps 3 --gate_steps 4 --eval_each_step --eval_ensemble_samples 1 --variants B --no-include_baseline --out_dir experiments/synthetic/mc_cem_qr_lh_b_steps5_eval_seed0_20260518`

6. Learn q(R) inducing locations and per-anchor constant outlier density:

   `python experiments/synthetic/compare_mc_cem_qr_lh.py --max_test 40 --mc_steps 3 --mc_samples 3 --cem_updates 1 --hyper_steps 1 --gate_steps 1 --variants B --no-include_baseline --mc_lr 0.01 --r_init pca --learn_inducing --inducing_lr_mult 0.1 --inducing_l2 0.001 --outlier_mode constant --learn_outlier_constant --out_dir experiments/synthetic/mc_cem_qr_lh_b_pca_learnZ_constU_seed0`

7. Quadratic gate, supervised route-B reward, or projected-neighborhood ablations:

   `python experiments/synthetic/compare_mc_cem_qr_lh.py --max_test 40 --mc_steps 3 --mc_samples 3 --cem_updates 1 --hyper_steps 1 --gate_steps 1 --variants B --no-include_baseline --mc_lr 0.01 --r_init pca --outlier_mode constant --learn_outlier_constant --gate_basis quadratic --mc_reward_mode supervised_anchor --reward_anchor_count 200 --reselect_neighbors_after_train anchor --out_dir experiments/synthetic/mc_cem_qr_lh_ablation`

The variants are route A (`A_uniform`, `A_softmax`, `A_softmax_entropy`) and
route B (`B`, `B_std`).  The entropy route uses a decaying exploration bonus;
`B_std` standardizes rewards across MC samples within each anchor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.projections.uncertain_w_jgp import (  # noqa: E402
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
    MCCEMQRMstepConfig,
    SelfCEMConfig,
    run_uncertain_w_self_cem,
    sparse_gp_uncertain_w_moments_for_mode,
    train_qr_mc_cem_mstep,
    uncertain_w_cem_predict_from_labels,
)
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import (  # noqa: E402
    SETTING_PRESETS,
    _aggregate,
    _generate_paper_data,
    _kmeans_medoids,
    _median_distance,
    _metrics_pack,
    _neighbor_stack,
    _projection_W0,
    _run_meanw_cem_init,
    _set_seed,
    _write_csv,
    sanitize_for_json,
)


VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "A_uniform": {
        "objective": "stopgrad_reparam",
        "weighting": "uniform",
        "entropy_beta_start": 0.0,
        "entropy_beta_end": 0.0,
        "standardize_rewards": False,
    },
    "A_softmax": {
        "objective": "stopgrad_reparam",
        "weighting": "softmax",
        "entropy_beta_start": 0.0,
        "entropy_beta_end": 0.0,
        "standardize_rewards": False,
    },
    "A_softmax_entropy": {
        "objective": "stopgrad_reparam",
        "weighting": "softmax",
        "entropy_beta_start": 0.01,
        "entropy_beta_end": 0.0,
        "standardize_rewards": False,
    },
    "B": {
        "objective": "score_function",
        "weighting": "uniform",
        "entropy_beta_start": 0.0,
        "entropy_beta_end": 0.0,
        "standardize_rewards": False,
        "reward_advantage": "center",
        "anchor_weight": "uniform",
    },
    "B_std": {
        "objective": "score_function",
        "weighting": "uniform",
        "entropy_beta_start": 0.0,
        "entropy_beta_end": 0.0,
        "standardize_rewards": True,
        "reward_advantage": "center_std",
        "anchor_weight": "uniform",
    },
}


def _sample_anchor_w(w_moments: dict[str, torch.Tensor], *, n_samples: int, seed: int) -> torch.Tensor:
    W_mu = w_moments["W_mu_anchor"]
    W_var = w_moments["W_var_anchor"].clamp_min(0.0)
    try:
        gen = torch.Generator(device=W_mu.device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(seed))
    eps = torch.randn((int(n_samples), *tuple(W_mu.shape)), device=W_mu.device, dtype=W_mu.dtype, generator=gen)
    return W_mu.unsqueeze(0) + eps * W_var.sqrt().unsqueeze(0)


def _repeat_first(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.unsqueeze(0).expand(int(n), *tuple(x.shape)).reshape(int(n) * x.shape[0], *tuple(x.shape[1:])).contiguous()


def _projected_neighbor_stack(
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    n_neighbors: int,
    mode: str,
    inducing_X: torch.Tensor,
    R_mu: torch.Tensor,
    R_log_std: torch.Tensor,
    lengthscale: float,
    signal_var: float,
    include_conditional_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dummy_anchor_neighbors = X_anchor[:, None, :]
    wm_anchor = sparse_gp_uncertain_w_moments_for_mode(
        mode="anchor",
        X_anchor=X_anchor,
        X_neighbors=dummy_anchor_neighbors,
        inducing_X=inducing_X,
        R_mu=R_mu,
        R_log_std=R_log_std,
        lengthscale=float(lengthscale),
        signal_var=float(signal_var),
        include_conditional_residual=bool(include_conditional_residual),
    )
    W_anchor = wm_anchor["W_mu_anchor"]
    if mode == "anchor":
        delta = X_pool.unsqueeze(0) - X_anchor.unsqueeze(1)
        proj = torch.einsum("tqd,tnd->tnq", W_anchor, delta)
        dist2 = proj.pow(2).sum(dim=-1)
    elif mode == "pointwise":
        dummy_pool_neighbors = X_pool[:, None, :]
        wm_pool = sparse_gp_uncertain_w_moments_for_mode(
            mode="anchor",
            X_anchor=X_pool,
            X_neighbors=dummy_pool_neighbors,
            inducing_X=inducing_X,
            R_mu=R_mu,
            R_log_std=R_log_std,
            lengthscale=float(lengthscale),
            signal_var=float(signal_var),
            include_conditional_residual=bool(include_conditional_residual),
        )
        z_anchor = torch.einsum("tqd,td->tq", W_anchor, X_anchor)
        z_pool = torch.einsum("nqd,nd->nq", wm_pool["W_mu_anchor"], X_pool)
        dist2 = (z_pool.unsqueeze(0) - z_anchor.unsqueeze(1)).pow(2).sum(dim=-1)
    else:
        raise ValueError(f"Unknown projected neighbor mode {mode!r}.")
    k = min(int(n_neighbors), X_pool.shape[0])
    idx = torch.topk(dist2, k=k, largest=False).indices
    Xn = X_pool[idx]
    yn = y_pool[idx]
    return Xn.contiguous(), yn.contiguous()


def _self_cem_config(
    args: argparse.Namespace,
    *,
    hyper_steps: int | None = None,
    gate_steps: int | None = None,
) -> SelfCEMConfig:
    return SelfCEMConfig(
        cem_updates=int(args.cem_updates),
        hyperopt=FixedLabelHyperoptConfig(
            steps=int(args.hyper_steps if hyper_steps is None else hyper_steps),
            lr=float(args.hyper_lr),
            batched=bool(args.batch_hyperopt),
            min_noise_var=float(args.noise_std_floor) ** 2,
            max_noise_var=10.0,
            min_signal_var=1e-4,
            max_signal_var=100.0,
            min_lengthscale=1e-3,
            max_lengthscale=20.0,
        ),
        gate=FixedLabelGateConfig(
            steps=int(args.gate_steps if gate_steps is None else gate_steps),
            lr=float(args.gate_lr),
            basis=str(args.gate_basis),
            l2=float(args.gate_l2),
            max_norm=float(args.gate_max_norm),
            gh_points=int(args.gh_points),
        ),
        outlier_mode=str(args.outlier_mode),
        outlier_sigma_mult=float(args.outlier_sigma_mult),
        background_scale_mult=float(args.background_scale_mult),
        learn_outlier_constant=bool(args.learn_outlier_constant),
        outlier_constant_steps=int(args.outlier_constant_steps),
        outlier_constant_lr=float(args.outlier_constant_lr),
        outlier_constant_l2=float(args.outlier_constant_l2),
        outlier_constant_min_logp=float(args.outlier_constant_min_logp),
        outlier_constant_max_logp=float(args.outlier_constant_max_logp),
        min_inliers=int(args.min_inliers),
        max_inlier_frac=float(args.max_inlier_frac),
        refit_after_update=True,
    )


def _final_sampled_self_cem_predict(
    args: argparse.Namespace,
    *,
    seed: int,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors_std: torch.Tensor,
    init_state: dict[str, torch.Tensor],
    w_moments: dict[str, torch.Tensor],
    y_test: np.ndarray,
    y_mean: float,
    y_std: float,
    n_samples_override: int | None = None,
    hyper_steps_override: int | None = None,
    gate_steps_override: int | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    n_samples = int(args.final_ensemble_samples if n_samples_override is None else n_samples_override)
    T, n, D = X_neighbors.shape
    Q = w_moments["W_mu_anchor"].shape[1]
    W_samples = _sample_anchor_w(w_moments, n_samples=n_samples, seed=int(seed) + 17011)
    W_flat = W_samples.reshape(n_samples * T, Q, D).contiguous()
    zero = torch.zeros_like(W_flat)
    X_anchor_flat = _repeat_first(X_anchor, n_samples)
    X_neighbors_flat = _repeat_first(X_neighbors, n_samples)
    y_flat = _repeat_first(y_neighbors_std, n_samples)
    labels0 = _repeat_first(init_state["labels"].bool(), n_samples)
    ell0 = _repeat_first(init_state["lengthscale"], n_samples)
    sig0 = _repeat_first(init_state["signal_var"], n_samples)
    noise0 = _repeat_first(init_state["noise_var"], n_samples)
    gate0 = _repeat_first(init_state["gate"], n_samples)

    t0 = time.perf_counter()
    sample_state = run_uncertain_w_self_cem(
        X_anchor_flat,
        X_neighbors_flat,
        y_flat,
        labels0,
        mode="anchor",
        W_mu_anchor=W_flat,
        W_var_anchor=zero,
        init_lengthscale=ell0,
        init_signal_var=sig0,
        init_noise_var=noise0,
        gate_init=gate0,
        config=_self_cem_config(args, hyper_steps=hyper_steps_override, gate_steps=gate_steps_override),
    )
    cem_sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    out = uncertain_w_cem_predict_from_labels(
        X_anchor_flat,
        X_neighbors_flat,
        y_flat,
        sample_state["labels"],
        mode="anchor",
        W_mu_anchor=W_flat,
        W_var_anchor=zero,
        lengthscale=sample_state["lengthscale"],
        signal_var=sample_state["signal_var"],
        noise_var=sample_state["noise_var"],
        mean_const=sample_state["mean_const"],
        correction_weight=0.0,
        min_inliers=int(args.min_inliers),
    )
    pred_sec = time.perf_counter() - t1
    mu_samples = out.mu.detach().reshape(n_samples, T)
    var_samples = out.var.detach().reshape(n_samples, T).clamp_min(1e-12)
    mu_std = mu_samples.mean(dim=0)
    total_var_std = (var_samples + mu_samples.pow(2)).mean(dim=0) - mu_std.pow(2)
    total_var_std = total_var_std.clamp_min(1e-12)
    scale = abs(float(y_std))
    mu = mu_std.detach().cpu().numpy().reshape(-1) * float(y_std) + float(y_mean)
    sigma = total_var_std.sqrt().detach().cpu().numpy().reshape(-1) * scale
    pack = _metrics_pack(y_test, mu, sigma, cem_sec + pred_sec)
    row = {key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)}
    diag = {
        "final_ensemble_samples": int(n_samples),
        "final_hyper_steps": int(args.hyper_steps if hyper_steps_override is None else hyper_steps_override),
        "final_gate_steps": int(args.gate_steps if gate_steps_override is None else gate_steps_override),
        "final_ensemble_cem_sec": float(cem_sec),
        "final_ensemble_pred_sec": float(pred_sec),
        "final_ensemble_total_sec": float(cem_sec + pred_sec),
        "final_ensemble_mean_n_inliers": float(out.diagnostics["mean_n_inliers"]),
        "final_ensemble_fallback_count": int(out.diagnostics["fallback_count"]),
        "final_ensemble_within_sigma_mean": float((var_samples.mean(dim=0).sqrt() * scale).mean().cpu().item()),
        "final_ensemble_between_sigma_mean": float(mu_samples.var(dim=0, unbiased=False).sqrt().mul(scale).mean().cpu().item()),
    }
    return row, diag


def _baseline_self_cem_row(
    args: argparse.Namespace,
    *,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors_std: torch.Tensor,
    cem_state: dict[str, torch.Tensor],
    W0: torch.Tensor,
    y_test: np.ndarray,
    y_mean: float,
    y_std: float,
) -> dict[str, Any]:
    zero = torch.zeros((X_anchor.shape[0], W0.shape[0], W0.shape[1]), device=X_anchor.device, dtype=X_anchor.dtype)
    W_anchor = W0.unsqueeze(0).expand_as(zero).contiguous()
    t0 = time.perf_counter()
    state = run_uncertain_w_self_cem(
        X_anchor,
        X_neighbors,
        y_neighbors_std,
        cem_state["labels"],
        mode="anchor",
        W_mu_anchor=W_anchor,
        W_var_anchor=zero,
        init_lengthscale=cem_state["lengthscale"],
        init_signal_var=cem_state["signal_var"],
        init_noise_var=cem_state["noise_var"],
        gate_init=cem_state["gate"],
        config=_self_cem_config(args),
    )
    train_sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    out = uncertain_w_cem_predict_from_labels(
        X_anchor,
        X_neighbors,
        y_neighbors_std,
        state["labels"],
        mode="anchor",
        W_mu_anchor=W_anchor,
        W_var_anchor=zero,
        lengthscale=state["lengthscale"],
        signal_var=state["signal_var"],
        noise_var=state["noise_var"],
        mean_const=state["mean_const"],
        correction_weight=float(args.kcov),
        min_inliers=int(args.min_inliers),
    )
    pred_sec = time.perf_counter() - t1
    mu = out.mu.detach().cpu().numpy().reshape(-1) * float(y_std) + float(y_mean)
    sigma = out.sigma.detach().cpu().numpy().reshape(-1) * abs(float(y_std))
    pack = _metrics_pack(y_test, mu, sigma, pred_sec)
    row = {key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)}
    row.update(
        {
            "variant": "baseline_self_cem",
            "status": "ok",
            "train_sec": float(train_sec),
            "pred_sec": float(pred_sec),
            "mean_n_inliers": float(out.diagnostics["mean_n_inliers"]),
            "fallback_count": int(out.diagnostics["fallback_count"]),
        }
    )
    return row


def _run_variant(
    args: argparse.Namespace,
    variant: str,
    *,
    seed: int,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors_std: torch.Tensor,
    cem_state: dict[str, torch.Tensor],
    W0: torch.Tensor,
    R_init_W: torch.Tensor,
    X_inducing: torch.Tensor,
    qr_w_lengthscale: float,
    y_anchor_std: torch.Tensor | None = None,
    X_anchor_eval: torch.Tensor | None = None,
    X_neighbors_eval: torch.Tensor | None = None,
    y_neighbors_eval_std: torch.Tensor | None = None,
    cem_state_eval: dict[str, torch.Tensor] | None = None,
    X_pool: torch.Tensor | None = None,
    y_pool_std: torch.Tensor | None = None,
    y_test: np.ndarray,
    y_mean: float,
    y_std: float,
) -> dict[str, Any]:
    X_anchor_pred = X_anchor if X_anchor_eval is None else X_anchor_eval
    X_neighbors_pred = X_neighbors if X_neighbors_eval is None else X_neighbors_eval
    y_neighbors_pred_std = y_neighbors_std if y_neighbors_eval_std is None else y_neighbors_eval_std
    cem_state_pred = cem_state if cem_state_eval is None else cem_state_eval
    spec = VARIANT_SPECS[variant]
    cfg = MCCEMQRMstepConfig(
        objective=spec["objective"],
        steps=int(args.mc_steps),
        lr=float(args.mc_lr),
        beta_kl=float(args.mc_beta_kl),
        kl_warmup_steps=int(args.mc_kl_warmup_steps),
        n_samples=int(args.mc_samples),
        weighting=spec["weighting"],
        softmax_temperature_start=float(args.softmax_temperature_start),
        softmax_temperature_end=float(args.softmax_temperature_end),
        entropy_beta_start=float(spec["entropy_beta_start"]),
        entropy_beta_end=float(spec["entropy_beta_end"]),
        standardize_rewards=bool(spec["standardize_rewards"]),
        reward_mode=str(args.mc_reward_mode),
        init_log_std=float(args.mc_init_log_std),
        train_log_std=bool(args.mc_train_log_std),
        w_lengthscale=float(qr_w_lengthscale),
        w_signal_var=float(args.mc_w_signal_var),
        include_conditional_residual=bool(args.mc_include_residual),
        outlier_mode=str(args.outlier_mode),
        outlier_sigma_mult=float(args.outlier_sigma_mult),
        background_scale_mult=float(args.background_scale_mult),
        learn_inducing=bool(args.learn_inducing),
        inducing_lr_mult=float(args.inducing_lr_mult),
        inducing_l2=float(args.inducing_l2),
        ortho_weight=float(args.mc_ortho_weight),
        ortho_target_scale=float(args.mc_ortho_target_scale),
        ortho_normalize=bool(args.mc_ortho_normalize),
        early_stopping=bool(args.early_stopping),
        early_stopping_metric=str(args.early_stopping_metric),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
        restore_state=str(args.mc_restore_state),
        gh_points=int(args.gh_points),
        grad_clip_norm=float(args.mc_grad_clip_norm),
        min_inliers=int(args.min_inliers),
        log_interval=int(args.mc_log_interval),
        log_sample_rewards=bool(spec.get("log_sample_rewards", False)),
        reward_advantage=str(spec.get("reward_advantage", "center_std")),
        anchor_weight=str(spec.get("anchor_weight", "uniform")),
        anchor_weight_top_frac=float(spec.get("anchor_weight_top_frac", 0.3)),
        anchor_weight_spread_tau=float(spec.get("anchor_weight_spread_tau", 0.0)),
        anchor_weight_max=float(spec.get("anchor_weight_max", 5.0)),
        anchor_weight_curriculum_steps=int(spec.get("anchor_weight_curriculum_steps", 0)),
        seed=int(seed) + 2309,
    )
    eval_callback = None
    if bool(args.eval_each_step) or bool(args.early_stopping):
        def eval_callback(
            step: int,
            R_mu: torch.Tensor,
            R_log_std: torch.Tensor,
            inducing_X_eval: torch.Tensor | None = None,
        ) -> dict[str, float]:
            t_eval = time.perf_counter()
            step_w_moments = sparse_gp_uncertain_w_moments_for_mode(
                mode="anchor",
                X_anchor=X_anchor_pred,
                X_neighbors=X_neighbors_pred,
                inducing_X=X_inducing if inducing_X_eval is None else inducing_X_eval,
                R_mu=R_mu,
                R_log_std=R_log_std,
                lengthscale=float(qr_w_lengthscale),
                signal_var=float(args.mc_w_signal_var),
                include_conditional_residual=bool(args.mc_include_residual),
            )
            pred_row_step, _pred_diag_step = _final_sampled_self_cem_predict(
                args,
                seed=seed + 1000 * int(step),
                X_anchor=X_anchor_pred,
                X_neighbors=X_neighbors_pred,
                y_neighbors_std=y_neighbors_pred_std,
                init_state=cem_state_pred,
                w_moments=step_w_moments,
                y_test=y_test,
                y_mean=y_mean,
                y_std=y_std,
                n_samples_override=int(args.eval_ensemble_samples),
                hyper_steps_override=int(args.eval_hyper_steps) if args.eval_hyper_steps is not None else None,
                gate_steps_override=int(args.eval_gate_steps) if args.eval_gate_steps is not None else None,
            )
            return {
                "eval_rmse": float(pred_row_step["rmse"]),
                "eval_mean_crps": float(pred_row_step["mean_crps"]),
                "eval_coverage_90": float(pred_row_step["coverage_90"]),
                "eval_mean_width_90": float(pred_row_step["mean_width_90"]),
                "eval_sec": float(time.perf_counter() - t_eval),
            }
    t0 = time.perf_counter()
    qr = train_qr_mc_cem_mstep(
        X_anchor,
        X_neighbors,
        y_neighbors_std,
        cem_state["labels"],
        y_anchor=y_anchor_std,
        inducing_X=X_inducing,
        init_W=R_init_W,
        init_lengthscale=cem_state["lengthscale"],
        init_signal_var=cem_state["signal_var"],
        init_noise_var=cem_state["noise_var"],
        gate_init=cem_state["gate"],
        cem_config=_self_cem_config(args),
        config=cfg,
        eval_callback=eval_callback,
    )
    train_sec = time.perf_counter() - t0
    if str(args.reselect_neighbors_after_train) != "none":
        if X_pool is None or y_pool_std is None:
            raise ValueError("Projected neighbor reselection requires X_pool/y_pool_std.")
        X_neighbors_pred, y_neighbors_pred_std = _projected_neighbor_stack(
            X_anchor_pred,
            X_pool,
            y_pool_std,
            n_neighbors=int(args.n_neighbors),
            mode=str(args.reselect_neighbors_after_train),
            inducing_X=qr.inducing_X,
            R_mu=qr.R_mu,
            R_log_std=qr.R_log_std,
            lengthscale=float(qr_w_lengthscale),
            signal_var=float(args.mc_w_signal_var),
            include_conditional_residual=bool(args.mc_include_residual),
        )
        cem_state_pred, _reselect_cem_diag = _run_meanw_cem_init(X_anchor_pred, X_neighbors_pred, y_neighbors_pred_std, W0)
    pred_w_moments = sparse_gp_uncertain_w_moments_for_mode(
        mode="anchor",
        X_anchor=X_anchor_pred,
        X_neighbors=X_neighbors_pred,
        inducing_X=qr.inducing_X,
        R_mu=qr.R_mu,
        R_log_std=qr.R_log_std,
        lengthscale=float(qr_w_lengthscale),
        signal_var=float(args.mc_w_signal_var),
        include_conditional_residual=bool(args.mc_include_residual),
    )
    pred_row, pred_diag = _final_sampled_self_cem_predict(
        args,
        seed=seed,
        X_anchor=X_anchor_pred,
        X_neighbors=X_neighbors_pred,
        y_neighbors_std=y_neighbors_pred_std,
        init_state=cem_state_pred,
        w_moments=pred_w_moments,
        y_test=y_test,
        y_mean=y_mean,
        y_std=y_std,
        hyper_steps_override=int(args.final_hyper_steps) if args.final_hyper_steps is not None else None,
        gate_steps_override=int(args.final_gate_steps) if args.final_gate_steps is not None else None,
    )
    row: dict[str, Any] = {
        "variant": variant,
        "status": "ok",
        "objective": spec["objective"],
        "mc_reward_mode": str(args.mc_reward_mode),
        "reselect_neighbors_after_train": str(args.reselect_neighbors_after_train),
        "mc_weighting": spec["weighting"],
        "mc_standardize_rewards": bool(spec["standardize_rewards"]),
        "mc_entropy_beta_start": float(spec["entropy_beta_start"]),
        "mc_entropy_beta_end": float(spec["entropy_beta_end"]),
        "mc_ortho_weight": float(args.mc_ortho_weight),
        "mc_early_stopping": bool(args.early_stopping),
        "mc_restore_state": str(args.mc_restore_state),
        "learn_inducing": bool(args.learn_inducing),
        "inducing_lr_mult": float(args.inducing_lr_mult),
        "inducing_l2": float(args.inducing_l2),
        "inducing_shift_mean": float(qr.diagnostics.get("inducing_shift_mean", 0.0)),
        "inducing_shift_max": float(qr.diagnostics.get("inducing_shift_max", 0.0)),
        "train_sec": float(train_sec),
        "qr_train_sec": float(qr.train_sec),
        "qr_best_loss": float(qr.diagnostics["best_loss"]),
        "qr_kl_per_anchor": float(qr.diagnostics["kl_per_anchor"]),
        "qr_R_std_median": float(qr.diagnostics["R_std_median"]),
        "qr_best_eval_metric": float(qr.diagnostics.get("best_eval_metric", float("inf"))),
        "qr_best_eval_step": int(qr.diagnostics.get("best_eval_step", 0)),
        "qr_history_json": json.dumps(sanitize_for_json(qr.history), sort_keys=True),
    }
    row.update(pred_row)
    row.update(pred_diag)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MC-CEM qR M-step variants on paper-style LH data.")
    p.add_argument("--variants", default="A_uniform,A_softmax,A_softmax_entropy,B,B_std")
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--max_test", type=int, default=40)
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--m_inducing", type=int, default=40)
    p.add_argument("--projection_init", choices=["pls", "pca", "pca_scaled", "pairwise"], default="pls")
    p.add_argument(
        "--r_init",
        choices=["same", "pls", "pca", "pca_scaled", "pairwise"],
        default="same",
        help="Initialization for q(R) inducing means; 'same' reuses --projection_init.",
    )
    p.add_argument("--include_baseline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mc_steps", type=int, default=4)
    p.add_argument("--mc_samples", type=int, default=3)
    p.add_argument("--mc_lr", type=float, default=0.005)
    p.add_argument("--mc_beta_kl", type=float, default=0.05)
    p.add_argument("--mc_kl_warmup_steps", type=int, default=2)
    p.add_argument("--mc_init_log_std", type=float, default=-3.0)
    p.add_argument("--mc_train_log_std", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mc_w_signal_var", type=float, default=1.0)
    p.add_argument("--mc_include_residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mc_grad_clip_norm", type=float, default=10.0)
    p.add_argument("--mc_log_interval", type=int, default=1)
    p.add_argument("--mc_reward_mode", choices=["transductive", "supervised_anchor"], default="transductive")
    p.add_argument("--reward_anchor_count", type=int, default=None)
    p.add_argument("--reselect_neighbors_after_train", choices=["none", "anchor", "pointwise"], default="none")
    p.add_argument("--mc_ortho_weight", type=float, default=0.0)
    p.add_argument("--mc_ortho_target_scale", type=float, default=1.0)
    p.add_argument("--mc_ortho_normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--learn_inducing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--inducing_lr_mult", type=float, default=0.1)
    p.add_argument("--inducing_l2", type=float, default=0.0)
    p.add_argument("--softmax_temperature_start", type=float, default=2.0)
    p.add_argument("--softmax_temperature_end", type=float, default=0.7)
    p.add_argument("--final_ensemble_samples", type=int, default=3)
    p.add_argument("--final_hyper_steps", type=int, default=None)
    p.add_argument("--final_gate_steps", type=int, default=None)
    p.add_argument("--eval_each_step", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eval_ensemble_samples", type=int, default=1)
    p.add_argument("--eval_hyper_steps", type=int, default=None)
    p.add_argument("--eval_gate_steps", type=int, default=None)
    p.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--early_stopping_metric", choices=["eval_rmse", "eval_mean_crps"], default="eval_rmse")
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    p.add_argument("--mc_restore_state", choices=["best_loss", "best_eval", "final"], default="best_loss")
    p.add_argument("--cem_updates", type=int, default=1)
    p.add_argument("--hyper_steps", type=int, default=4)
    p.add_argument("--hyper_lr", type=float, default=0.04)
    p.add_argument("--batch_hyperopt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gate_steps", type=int, default=6)
    p.add_argument("--gate_basis", choices=["linear", "quadratic"], default="linear")
    p.add_argument("--gate_lr", type=float, default=0.05)
    p.add_argument("--gate_l2", type=float, default=1e-3)
    p.add_argument("--gate_max_norm", type=float, default=8.0)
    p.add_argument("--gh_points", type=int, default=20)
    p.add_argument("--noise_std_floor", type=float, default=0.1)
    p.add_argument("--kcov", type=float, default=0.1)
    p.add_argument("--min_inliers", type=int, default=2)
    p.add_argument("--max_inlier_frac", type=float, default=0.95)
    p.add_argument("--outlier_mode", choices=["background_normal", "constant"], default="background_normal")
    p.add_argument("--outlier_sigma_mult", type=float, default=2.5)
    p.add_argument("--background_scale_mult", type=float, default=3.0)
    p.add_argument("--learn_outlier_constant", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--outlier_constant_steps", type=int, default=10)
    p.add_argument("--outlier_constant_lr", type=float, default=0.05)
    p.add_argument("--outlier_constant_l2", type=float, default=5.0)
    p.add_argument("--outlier_constant_min_logp", type=float, default=-12.0)
    p.add_argument("--outlier_constant_max_logp", type=float, default=-1e-4)
    p.add_argument("--out_dir", default="experiments/synthetic/mc_cem_qr_lh_seed0_20260518")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    unknown = sorted(set(variants) - set(VARIANT_SPECS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setting = "lh_q5_train1000"
    cfg = dict(SETTING_PRESETS[setting])
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
        _set_seed(seed)
        data = _generate_paper_data(cfg, seed, device)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(data["X_train"]).astype(np.float64)
        X_test = scaler.transform(data["X_test"]).astype(np.float64)
        y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
        y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)
        if args.max_test is not None:
            X_test = X_test[: int(args.max_test)]
            y_test = y_test[: int(args.max_test)]
        y_mean = float(y_train.mean())
        y_std = float(y_train.std() if y_train.std() > 1e-8 else 1.0)
        y_train_std = ((y_train - y_mean) / y_std).astype(np.float64)
        Q = int(cfg["Q"])
        W0_np = _projection_W0(X_train, y_train_std, Q, str(args.projection_init), seed=seed + 513)
        W0 = torch.as_tensor(W0_np, device=device, dtype=torch.float64)
        r_init_mode = str(args.projection_init) if str(args.r_init) == "same" else str(args.r_init)
        R_init_np = (
            W0_np
            if r_init_mode == str(args.projection_init)
            else _projection_W0(X_train, y_train_std, Q, r_init_mode, seed=seed + 1543)
        )
        R_init_W = torch.as_tensor(R_init_np, device=device, dtype=torch.float64)
        X_train_t = torch.as_tensor(X_train, device=device, dtype=torch.float64)
        X_test_t = torch.as_tensor(X_test, device=device, dtype=torch.float64)
        y_train_std_t = torch.as_tensor(y_train_std, device=device, dtype=torch.float64)
        X_neighbors, y_neighbors_std, _neighbor_idx = _neighbor_stack(
            X_test_t,
            X_train_t,
            y_train_std_t,
            n_neighbors=int(args.n_neighbors),
        )
        inducing_idx = _kmeans_medoids(X_train, int(args.m_inducing), seed + 7919)
        X_inducing = X_train_t[torch.as_tensor(inducing_idx, device=device, dtype=torch.long)]
        qr_w_lengthscale = float(_median_distance(X_inducing))
        cem_state, cem_diag = _run_meanw_cem_init(X_test_t, X_neighbors, y_neighbors_std, W0)
        common = {
            "setting": setting,
            "family": "LH",
            "seed": int(seed),
            "projection_init": str(args.projection_init),
            "r_init": str(args.r_init),
            "r_init_resolved": str(r_init_mode),
            "N_train": int(X_train.shape[0]),
            "N_test": int(data["X_test"].shape[0]),
            "n_test_eval": int(X_test.shape[0]),
            "Q": int(Q),
            "observed_dim": int(X_train.shape[1]),
            "latent_dim": int(cfg["d"]),
            "expansion": "rff",
            "standardize_x": True,
            "standardize_y_for_head": True,
            "n_neighbors": int(args.n_neighbors),
            "m_inducing": int(args.m_inducing),
            "qr_w_lengthscale": float(qr_w_lengthscale),
            "mc_steps": int(args.mc_steps),
            "mc_samples": int(args.mc_samples),
            "mc_lr": float(args.mc_lr),
            "mc_beta_kl": float(args.mc_beta_kl),
            "mc_reward_mode": str(args.mc_reward_mode),
            "reward_anchor_count": "" if args.reward_anchor_count is None else int(args.reward_anchor_count),
            "reselect_neighbors_after_train": str(args.reselect_neighbors_after_train),
            "mc_ortho_weight": float(args.mc_ortho_weight),
            "learn_inducing": bool(args.learn_inducing),
            "inducing_lr_mult": float(args.inducing_lr_mult),
            "inducing_l2": float(args.inducing_l2),
            "early_stopping": bool(args.early_stopping),
            "early_stopping_metric": str(args.early_stopping_metric),
            "mc_restore_state": str(args.mc_restore_state),
            "cem_updates": int(args.cem_updates),
            "hyper_steps": int(args.hyper_steps),
            "batch_hyperopt": bool(args.batch_hyperopt),
            "gate_steps": int(args.gate_steps),
            "gate_basis": str(args.gate_basis),
            "outlier_mode": str(args.outlier_mode),
            "learn_outlier_constant": bool(args.learn_outlier_constant),
            "outlier_constant_l2": float(args.outlier_constant_l2),
            "final_ensemble_samples": int(args.final_ensemble_samples),
            "final_hyper_steps": "" if args.final_hyper_steps is None else int(args.final_hyper_steps),
            "final_gate_steps": "" if args.final_gate_steps is None else int(args.final_gate_steps),
            "eval_each_step": bool(args.eval_each_step),
            "eval_ensemble_samples": int(args.eval_ensemble_samples),
            "eval_hyper_steps": "" if args.eval_hyper_steps is None else int(args.eval_hyper_steps),
            "eval_gate_steps": "" if args.eval_gate_steps is None else int(args.eval_gate_steps),
            "init_sec": float(cem_diag.get("init_sec", 0.0)),
        }
        if bool(args.include_baseline):
            print(f"run seed={seed} variant=baseline_self_cem", flush=True)
            row = dict(common)
            row.update(
                _baseline_self_cem_row(
                    args,
                    X_anchor=X_test_t,
                    X_neighbors=X_neighbors,
                    y_neighbors_std=y_neighbors_std,
                    cem_state=cem_state,
                    W0=W0,
                    y_test=y_test,
                    y_mean=y_mean,
                    y_std=y_std,
                )
            )
            rows.append(row)
            _write_csv(out_dir / "metrics.csv", rows)
            _write_csv(out_dir / "aggregate.csv", _aggregate(rows))
        train_anchor_args: dict[str, Any] = {}
        if str(args.mc_reward_mode) == "supervised_anchor":
            reward_count = int(args.max_test if args.reward_anchor_count is None else args.reward_anchor_count)
            reward_count = min(reward_count, X_train.shape[0])
            reward_idx_np = _kmeans_medoids(X_train, reward_count, seed + 4271)
            reward_idx = torch.as_tensor(reward_idx_np, device=device, dtype=torch.long)
            X_reward_t = X_train_t[reward_idx]
            y_reward_std_t = y_train_std_t[reward_idx]
            X_reward_neighbors, y_reward_neighbors_std, _reward_neighbor_idx = _neighbor_stack(
                X_reward_t,
                X_train_t,
                y_train_std_t,
                n_neighbors=int(args.n_neighbors),
            )
            reward_cem_state, _reward_cem_diag = _run_meanw_cem_init(X_reward_t, X_reward_neighbors, y_reward_neighbors_std, W0)
            train_anchor_args = {
                "X_anchor": X_reward_t,
                "X_neighbors": X_reward_neighbors,
                "y_neighbors_std": y_reward_neighbors_std,
                "cem_state": reward_cem_state,
                "y_anchor_std": y_reward_std_t,
                "X_anchor_eval": X_test_t,
                "X_neighbors_eval": X_neighbors,
                "y_neighbors_eval_std": y_neighbors_std,
                "cem_state_eval": cem_state,
            }
        else:
            train_anchor_args = {
                "X_anchor": X_test_t,
                "X_neighbors": X_neighbors,
                "y_neighbors_std": y_neighbors_std,
                "cem_state": cem_state,
            }
        for variant in variants:
            print(f"run seed={seed} variant={variant}", flush=True)
            row = dict(common)
            try:
                row.update(
                    _run_variant(
                        args,
                        variant,
                        seed=seed,
                        **train_anchor_args,
                        W0=W0,
                        R_init_W=R_init_W,
                        X_inducing=X_inducing,
                        qr_w_lengthscale=qr_w_lengthscale,
                        X_pool=X_train_t,
                        y_pool_std=y_train_std_t,
                        y_test=y_test,
                        y_mean=y_mean,
                        y_std=y_std,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                row.update({
                    "variant": variant,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                print(f"FAILED seed={seed} variant={variant}: {row['error']}", flush=True)
            rows.append(row)
            _write_csv(out_dir / "metrics.csv", rows)
            _write_csv(out_dir / "aggregate.csv", _aggregate(rows))

    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "args": vars(args),
        "setting": cfg,
        "variants": variants,
        "n_rows": len(rows),
        "n_ok": int(sum(1 for r in rows if r.get("status") == "ok")),
        "n_failed": int(sum(1 for r in rows if r.get("status") != "ok")),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    readme = (
        "# MC-CEM qR LH Pilot\n\n"
        "Paper-style LH `lh_q5_train1000`: RFF expansion, latent dimension 5, "
        "observed dimension 30, train/test 1000/200, train-only X scaling and "
        "train-only y standardization inside the head.\n\n"
        "Files: `metrics.csv`, `aggregate.csv`, `summary.json`.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved results to {out_dir}", flush=True)
    return {"rows": rows, "aggregate": _aggregate(rows), "summary": summary}


if __name__ == "__main__":
    main()

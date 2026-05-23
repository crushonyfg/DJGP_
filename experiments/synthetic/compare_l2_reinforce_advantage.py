"""Compare supervised REINFORCE advantage / anchor-weight variants on L2.

Priority variants (see ``l2_reinforce_advantage_specs.DEFAULT_COMPARE_VARIANTS``):
  center, rank, center_std (baseline), hard_center_spread, hard_rank_spread,
  hard_top30_rank, hard_curriculum

Avoid ``center_std`` + ``spread_std`` — they partially cancel.

Outputs ``metrics.csv`` with RMSE_cem, W drift, reward spread, rank stability,
top-anchor overlap, and mean ΔR vs W0.

```powershell
cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP
python experiments/synthetic/compare_l2_reinforce_advantage.py --phase2_steps 25 --phase3_steps 8
```
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.projections.uncertain_w_jgp import MCCEMQRMstepConfig, train_qr_mc_cem_mstep  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import SETTING_PRESETS  # noqa: E402
from experiments.synthetic.compare_mc_cem_qr_lh import (  # noqa: E402
    _aggregate,
    _final_sampled_self_cem_predict,
    _self_cem_config,
    sanitize_for_json,
)
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import (  # noqa: E402
    _generate_paper_data,
    _kmeans_medoids,
    _median_distance,
    _neighbor_stack,
    _projection_W0,
    _run_meanw_cem_init,
    _set_seed,
    _write_csv,
)
from experiments.synthetic.l2_reinforce_advantage_specs import (  # noqa: E402
    ADVANTAGE_VARIANTS,
    DEFAULT_COMPARE_VARIANTS,
)
from experiments.synthetic.l2_reinforce_diagnostics import (  # noqa: E402
    summarize_reward_trace,
    w_drift_frobenius_mean,
    write_spread_histogram,
)
from experiments.synthetic.train_supervised_reinforce_l2 import (  # noqa: E402
    _export_sample_reward_trace,
    _frozen_cem_config,
    _mc_config as _base_mc_config,
)


def _mc_config_for_variant(
    args: argparse.Namespace,
    spec: dict[str, Any],
    *,
    phase: str,
    seed: int,
    qr_w_lengthscale: float,
) -> MCCEMQRMstepConfig:
    steps = int(args.phase2_steps if phase == "explore" else args.phase3_steps)
    base = _base_mc_config(args, steps=steps, phase=phase, seed=seed, qr_w_lengthscale=qr_w_lengthscale)
    return replace(
        base,
        log_sample_rewards=bool(args.log_sample_rewards),
        reward_advantage=str(spec.get("reward_advantage", base.reward_advantage)),
        anchor_weight=str(spec.get("anchor_weight", base.anchor_weight)),
        anchor_weight_top_frac=float(spec.get("anchor_weight_top_frac", base.anchor_weight_top_frac)),
        anchor_weight_spread_tau=float(spec.get("anchor_weight_spread_tau", base.anchor_weight_spread_tau)),
        anchor_weight_max=float(spec.get("anchor_weight_max", base.anchor_weight_max)),
        anchor_weight_curriculum_steps=int(spec.get("anchor_weight_curriculum_steps", base.anchor_weight_curriculum_steps)),
        standardize_rewards=bool(spec.get("standardize_rewards", base.standardize_rewards)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variants",
        default=",".join(DEFAULT_COMPARE_VARIANTS),
        help="Comma-separated keys from l2_reinforce_advantage_specs.",
    )
    p.add_argument("--setting", default="l2_q2_train500")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=80)
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--m_inducing", type=int, default=40)
    p.add_argument("--reward_anchor_count", type=int, default=50)
    p.add_argument("--phase2_steps", type=int, default=25)
    p.add_argument("--phase3_steps", type=int, default=8)
    p.add_argument("--mc_samples", type=int, default=5)
    p.add_argument("--mc_lr", type=float, default=0.008)
    p.add_argument("--mc_train_log_std", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mc_w_signal_var", type=float, default=1.0)
    p.add_argument("--mc_include_residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mc_grad_clip_norm", type=float, default=10.0)
    p.add_argument("--learn_inducing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--inducing_lr_mult", type=float, default=0.1)
    p.add_argument("--min_inliers", type=int, default=2)
    p.add_argument("--outlier_mode", default="background_normal")
    p.add_argument("--outlier_sigma_mult", type=float, default=2.5)
    p.add_argument("--background_scale_mult", type=float, default=3.0)
    p.add_argument("--learn_outlier_constant", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--outlier_constant_steps", type=int, default=10)
    p.add_argument("--outlier_constant_lr", type=float, default=0.05)
    p.add_argument("--outlier_constant_l2", type=float, default=5.0)
    p.add_argument("--outlier_constant_min_logp", type=float, default=-12.0)
    p.add_argument("--outlier_constant_max_logp", type=float, default=-1e-4)
    p.add_argument("--hyper_lr", type=float, default=0.04)
    p.add_argument("--batch_hyperopt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gate_basis", choices=["linear", "quadratic"], default="linear")
    p.add_argument("--gate_lr", type=float, default=0.05)
    p.add_argument("--gate_l2", type=float, default=1e-3)
    p.add_argument("--gate_max_norm", type=float, default=8.0)
    p.add_argument("--gh_points", type=int, default=20)
    p.add_argument("--noise_std_floor", type=float, default=0.1)
    p.add_argument("--kcov", type=float, default=0.1)
    p.add_argument("--max_inlier_frac", type=float, default=0.95)
    p.add_argument("--explore_beta_kl", type=float, default=0.01)
    p.add_argument("--refine_beta_kl", type=float, default=0.1)
    p.add_argument("--explore_kl_warmup", type=int, default=4)
    p.add_argument("--refine_kl_warmup", type=int, default=6)
    p.add_argument("--explore_init_log_std", type=float, default=-2.5)
    p.add_argument("--refine_init_log_std", type=float, default=-3.0)
    p.add_argument("--explore_entropy_beta", type=float, default=0.02)
    p.add_argument("--hyper_steps", type=int, default=4)
    p.add_argument("--gate_steps", type=int, default=6)
    p.add_argument("--cem_updates", type=int, default=1)
    p.add_argument("--noise_std", type=float, default=None)
    p.add_argument("--standardize_y", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--log_sample_rewards", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--refine_freeze_cem", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--final_ensemble_samples", type=int, default=5)
    p.add_argument("--out_dir", default="experiments/synthetic/l2_reinforce_advantage_cmp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    unknown = sorted(set(variants) - set(ADVANTAGE_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(SETTING_PRESETS[args.setting])
    if args.noise_std is not None:
        ns = float(args.noise_std)
        cfg["noise_std"] = ns
        cfg["noise_var"] = ns * ns

    rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"\n=== variant {variant} ===", flush=True)
        _set_seed(args.seed)
        spec = ADVANTAGE_VARIANTS[variant]
        data = _generate_paper_data(cfg, args.seed, device)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(data["X_train"]).astype(np.float64)
        X_test = scaler.transform(data["X_test"]).astype(np.float64)[: int(args.max_test)]
        y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
        y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)[: int(args.max_test)]
        if bool(args.standardize_y):
            y_mean = float(y_train.mean())
            y_std = float(y_train.std() if y_train.std() > 1e-8 else 1.0)
            y_head = ((y_train - y_mean) / y_std).astype(np.float64)
        else:
            y_mean, y_std = 0.0, 1.0
            y_head = y_train.astype(np.float64)
        Q = int(cfg["Q"])
        X_train_t = torch.as_tensor(X_train, device=device, dtype=torch.float64)
        X_test_t = torch.as_tensor(X_test, device=device, dtype=torch.float64)
        y_train_head_t = torch.as_tensor(y_head, device=device, dtype=torch.float64)
        W0 = torch.as_tensor(_projection_W0(X_train, y_head, Q, "pls", seed=args.seed + 513), device=device, dtype=torch.float64)
        X_neighbors, y_neighbors_head, _ = _neighbor_stack(X_test_t, X_train_t, y_train_head_t, n_neighbors=int(args.n_neighbors))
        X_inducing = X_train_t[torch.as_tensor(_kmeans_medoids(X_train, int(args.m_inducing), args.seed + 7919), device=device, dtype=torch.long)]
        qr_w_lengthscale = float(_median_distance(X_inducing))
        reward_count = min(int(args.reward_anchor_count), X_train.shape[0])
        reward_idx = _kmeans_medoids(X_train, reward_count, args.seed + 4271)
        X_reward_t = X_train_t[torch.as_tensor(reward_idx, device=device, dtype=torch.long)]
        y_reward_head_t = y_train_head_t[torch.as_tensor(reward_idx, device=device, dtype=torch.long)]
        X_reward_nb, y_reward_nb_head, _ = _neighbor_stack(X_reward_t, X_train_t, y_train_head_t, n_neighbors=int(args.n_neighbors))
        cem_test, _ = _run_meanw_cem_init(X_test_t, X_neighbors, y_neighbors_head, W0)
        cem_reward, _ = _run_meanw_cem_init(X_reward_t, X_reward_nb, y_reward_nb_head, W0)

        var_dir = out_dir / variant
        var_dir.mkdir(parents=True, exist_ok=True)
        explore_cfg = _mc_config_for_variant(args, spec, phase="explore", seed=args.seed + 11, qr_w_lengthscale=qr_w_lengthscale)
        qr_explore = train_qr_mc_cem_mstep(
            X_reward_t, X_reward_nb, y_reward_nb_head, cem_reward["labels"],
            y_anchor=y_reward_head_t, inducing_X=X_inducing, init_W=W0.clone(),
            init_lengthscale=cem_reward["lengthscale"], init_signal_var=cem_reward["signal_var"],
            init_noise_var=cem_reward["noise_var"], gate_init=cem_reward["gate"],
            cem_config=_self_cem_config(args), config=explore_cfg,
        )
        explore_diag: dict[str, float] = {}
        if bool(args.log_sample_rewards):
            _export_sample_reward_trace(qr_explore.sample_reward_trace, phase="explore", out_dir=var_dir)
            explore_diag = summarize_reward_trace(qr_explore.sample_reward_trace)
            write_spread_histogram(
                qr_explore.sample_reward_trace,
                var_dir / "spread_histogram_explore.json",
            )

        refine_cem_cfg = _frozen_cem_config(args) if bool(args.refine_freeze_cem) else _self_cem_config(args)
        refine_cfg = _mc_config_for_variant(args, spec, phase="refine", seed=args.seed + 29, qr_w_lengthscale=qr_w_lengthscale)
        qr_refine = train_qr_mc_cem_mstep(
            X_reward_t, X_reward_nb, y_reward_nb_head, cem_reward["labels"],
            y_anchor=y_reward_head_t, inducing_X=qr_explore.inducing_X,
            init_R_mu=qr_explore.R_mu.detach(), init_R_log_std=qr_explore.R_log_std.detach(),
            init_lengthscale=cem_reward["lengthscale"], init_signal_var=cem_reward["signal_var"],
            init_noise_var=cem_reward["noise_var"], gate_init=cem_reward["gate"],
            cem_config=refine_cem_cfg, config=refine_cfg,
        )
        if bool(args.log_sample_rewards):
            _export_sample_reward_trace(qr_refine.sample_reward_trace, phase="refine", out_dir=var_dir)

        from djgp.projections.uncertain_w_jgp import sparse_gp_uncertain_w_moments_for_mode  # noqa: WPS433

        w_moments_test = sparse_gp_uncertain_w_moments_for_mode(
            mode="anchor", X_anchor=X_test_t, X_neighbors=X_neighbors, inducing_X=qr_refine.inducing_X,
            R_mu=qr_refine.R_mu, R_log_std=qr_refine.R_log_std, lengthscale=qr_w_lengthscale,
            signal_var=float(args.mc_w_signal_var),
            include_conditional_residual=True,
        )
        w_moments_reward = sparse_gp_uncertain_w_moments_for_mode(
            mode="anchor",
            X_anchor=X_reward_t,
            X_neighbors=X_reward_nb,
            inducing_X=qr_refine.inducing_X,
            R_mu=qr_refine.R_mu,
            R_log_std=qr_refine.R_log_std,
            lengthscale=qr_w_lengthscale,
            signal_var=float(args.mc_w_signal_var),
            include_conditional_residual=True,
        )
        w_drift_test = w_drift_frobenius_mean(W0, w_moments_test["W_mu_anchor"])
        w_drift_reward = w_drift_frobenius_mean(W0, w_moments_reward["W_mu_anchor"])
        pred_row, pred_diag = _final_sampled_self_cem_predict(
            args, seed=args.seed, X_anchor=X_test_t, X_neighbors=X_neighbors,
            y_neighbors_std=y_neighbors_head, init_state=cem_test, w_moments=w_moments_test,
            y_test=y_test, y_mean=y_mean, y_std=y_std, n_samples_override=int(args.final_ensemble_samples),
        )
        row = {
            "variant": variant,
            "reward_advantage": str(spec.get("reward_advantage", "")),
            "anchor_weight": str(spec.get("anchor_weight", "uniform")),
            "noise_std": float(cfg["noise_std"]),
            "rmse_cem": float(pred_row.get("rmse", float("nan"))),
            "crps_cem": float(pred_row.get("mean_crps", float("nan"))),
            "w_drift_test": w_drift_test,
            "w_drift_reward": w_drift_reward,
            "anchor_weight_active_frac_last": float(
                qr_explore.history[-1].get("anchor_weight_active_frac", float("nan")) if qr_explore.history else float("nan")
            ),
        }
        for k, v in explore_diag.items():
            row[f"explore_{k}"] = float(v)
        row.update({k: float(pred_row.get(k, float("nan"))) for k in RESULT_ROW_KEYS})
        row.update(pred_diag)
        rows.append(row)
        with (var_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(row), f, indent=2)

    _write_csv(out_dir / "metrics.csv", rows)
    _write_csv(out_dir / "aggregate.csv", _aggregate(rows))
    print(f"\nSaved {out_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()

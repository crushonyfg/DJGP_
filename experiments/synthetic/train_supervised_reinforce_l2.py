"""L2 explore-then-refine training with supervised anchor REINFORCE (Route B).

Uses existing ``train_qr_mc_cem_mstep`` + ``reward_mode=supervised_anchor`` (leave-anchor-out
predictive log-likelihood).  Phases:

1. **warm** — fixed ``W0`` self-CEM init (no q(R) update).
2. **explore** — Route ``B_std`` + low ``beta_kl`` + **full** self-CEM per sampled ``W``.
3. **refine** — higher ``beta_kl`` + full self-CEM per sampled ``W`` (optional freeze via ``--refine_freeze_cem``).
4. **eval** — full self-CEM ensemble on test anchors.

L2 2D projection panels are written every ``--viz_every`` MC steps.

```powershell
cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP
python experiments/synthetic/train_supervised_reinforce_l2.py --max_test 80 --phase2_steps 40 --phase3_steps 20 --reward_anchor_count 60 --viz_every 10
```
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
from djgp.projections.uncertain_w_jgp import (  # noqa: E402
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
    MCCEMQRMstepConfig,
    SelfCEMConfig,
    train_qr_mc_cem_mstep,
)
from experiments.synthetic.compare_mc_cem_qr_lh import (  # noqa: E402
    _aggregate,
    _baseline_self_cem_row,
    _final_sampled_self_cem_predict,
    _self_cem_config,
    sanitize_for_json,
)
from experiments.synthetic.l2_reinforce_advantage_specs import ADVANTAGE_VARIANTS
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _kmeans_medoids,
    _median_distance,
    _neighbor_stack,
    _projection_W0,
    _run_meanw_cem_init,
    _set_seed,
    _write_csv,
)
from experiments.synthetic.l2_reinforce_viz import (  # noqa: E402
    fit_gate_for_projection,
    plot_l2_projection_panel,
    plot_l2_step_diagnostics,
    regime_labels_from_phys,
)


def _export_sample_reward_trace(
    trace: list[dict[str, Any]],
    *,
    phase: str,
    out_dir: Path,
) -> None:
    """Write per-step per-W-sample supervised rewards to CSV + JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"reward_trace_{phase}.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(trace), f, indent=2)

    csv_path = out_dir / f"reward_trace_{phase}.csv"
    fieldnames = [
        "phase", "step", "sample", "reward_mean", "reward_std", "reward_min", "reward_max",
        "adv_mean", "adv_std", "reward_spread_mean_over_anchors", "reward_spread_max_over_anchors",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in trace:
            step = int(row["step"])
            spread_m = float(row.get("reward_spread_mean_over_anchors", float("nan")))
            spread_x = float(row.get("reward_spread_max_over_anchors", float("nan")))
            adv_means = row.get("adv_mean_per_sample") or [float("nan")] * len(row["reward_mean_per_sample"])
            adv_stds = row.get("adv_std_per_sample") or [float("nan")] * len(row["reward_std_per_sample"])
            for s, (rm, rs, rmin, rmax) in enumerate(zip(
                row["reward_mean_per_sample"],
                row["reward_std_per_sample"],
                row["reward_min_per_sample"],
                row["reward_max_per_sample"],
            )):
                w.writerow({
                    "phase": phase,
                    "step": step,
                    "sample": s,
                    "reward_mean": float(rm),
                    "reward_std": float(rs),
                    "reward_min": float(rmin),
                    "reward_max": float(rmax),
                    "adv_mean": float(adv_means[s]) if s < len(adv_means) else float("nan"),
                    "adv_std": float(adv_stds[s]) if s < len(adv_stds) else float("nan"),
                    "reward_spread_mean_over_anchors": spread_m,
                    "reward_spread_max_over_anchors": spread_x,
                })

    # Step-level summary: is there signal across W samples?
    summary_rows: list[dict[str, Any]] = []
    for row in trace:
        means = np.asarray(row["reward_mean_per_sample"], dtype=np.float64)
        rec = {
            "phase": phase,
            "step": int(row["step"]),
            "reward_mean_over_samples": float(means.mean()),
            "reward_std_over_samples": float(means.std()),
            "reward_range_over_samples": float(means.max() - means.min()),
            "reward_spread_mean_over_anchors": float(row.get("reward_spread_mean_over_anchors", float("nan"))),
        }
        if "delta_reward_per_sample_anchor" in row:
            delta = np.asarray(row["delta_reward_per_sample_anchor"], dtype=np.float64)
            rec["delta_reward_mean"] = float(delta.mean())
            rec["delta_reward_best_mean"] = float(delta.max(axis=0).mean())
            rec["frac_delta_positive"] = float((delta > 0.0).mean())
        if "top_anchor_indices" in row:
            rec["top_anchor_count"] = int(len(row["top_anchor_indices"]))
        summary_rows.append(rec)
    with (out_dir / f"reward_trace_{phase}_step_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)


def _frozen_cem_config(args: argparse.Namespace) -> SelfCEMConfig:
    """CEM with frozen hyper + gate (optional refine phase)."""
    return SelfCEMConfig(
        cem_updates=int(args.cem_updates),
        hyperopt=FixedLabelHyperoptConfig(
            steps=0,
            lr=float(args.hyper_lr),
            batched=bool(args.batch_hyperopt),
            min_noise_var=float(args.noise_std_floor) ** 2,
        ),
        gate=FixedLabelGateConfig(
            steps=0,
            lr=float(args.gate_lr),
            basis=str(args.gate_basis),
            l2=float(args.gate_l2),
        ),
        outlier_mode=str(args.outlier_mode),
        outlier_sigma_mult=float(args.outlier_sigma_mult),
        background_scale_mult=float(args.background_scale_mult),
        learn_outlier_constant=bool(args.learn_outlier_constant),
        min_inliers=int(args.min_inliers),
        max_inlier_frac=float(args.max_inlier_frac),
        refit_after_update=False,
    )


def _mc_config(
    args: argparse.Namespace,
    *,
    steps: int,
    phase: str,
    seed: int,
    qr_w_lengthscale: float,
) -> MCCEMQRMstepConfig:
    if phase == "explore":
        beta = float(args.explore_beta_kl)
        kl_warm = int(args.explore_kl_warmup)
        init_log_std = float(args.explore_init_log_std)
        entropy_start, entropy_end = float(args.explore_entropy_beta), 0.0
    else:
        beta = float(args.refine_beta_kl)
        kl_warm = int(args.refine_kl_warmup)
        init_log_std = float(args.refine_init_log_std)
        entropy_start, entropy_end = 0.0, 0.0
    spec = ADVANTAGE_VARIANTS[str(getattr(args, "advantage_variant", "center_std"))]
    base = MCCEMQRMstepConfig(
        objective="score_function",
        steps=int(steps),
        lr=float(args.mc_lr),
        beta_kl=beta,
        kl_warmup_steps=kl_warm,
        n_samples=int(args.mc_samples),
        weighting="uniform",
        entropy_beta_start=entropy_start,
        entropy_beta_end=entropy_end,
        standardize_rewards=bool(spec.get("standardize_rewards", True)),
        reward_advantage=str(spec.get("reward_advantage", "center_std")),
        anchor_weight=str(spec.get("anchor_weight", "uniform")),
        anchor_weight_top_frac=float(spec.get("anchor_weight_top_frac", 0.3)),
        anchor_weight_curriculum_steps=int(spec.get("anchor_weight_curriculum_steps", 0)),
        reward_mode="supervised_anchor",
        init_log_std=init_log_std,
        train_log_std=bool(args.mc_train_log_std),
        w_lengthscale=float(qr_w_lengthscale),
        w_signal_var=float(args.mc_w_signal_var),
        include_conditional_residual=bool(args.mc_include_residual),
        outlier_mode=str(args.outlier_mode),
        learn_inducing=bool(args.learn_inducing),
        inducing_lr_mult=float(args.inducing_lr_mult),
        grad_clip_norm=float(args.mc_grad_clip_norm),
        min_inliers=int(args.min_inliers),
        log_interval=1,
        log_sample_rewards=bool(args.log_sample_rewards),
        seed=int(seed),
        restore_state="final",
    )
    return replace(
        base,
        reward_advantage=str(spec.get("reward_advantage", base.reward_advantage)),
        anchor_weight=str(spec.get("anchor_weight", base.anchor_weight)),
        anchor_weight_top_frac=float(spec.get("anchor_weight_top_frac", base.anchor_weight_top_frac)),
        anchor_weight_curriculum_steps=int(spec.get("anchor_weight_curriculum_steps", base.anchor_weight_curriculum_steps)),
        standardize_rewards=bool(spec.get("standardize_rewards", base.standardize_rewards)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L2 supervised REINFORCE explore-refine trainer.")
    p.add_argument("--setting", default="l2_q2_train500")
    p.add_argument(
        "--noise_std",
        type=float,
        default=None,
        help="L2 phantom observation noise (overrides preset). Preset default is now 0.5.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--m_inducing", type=int, default=40)
    p.add_argument("--projection_init", default="pls")
    p.add_argument("--reward_anchor_count", type=int, default=80)
    p.add_argument("--viz_anchors", default="0,10,20,30", help="Comma-separated test anchor indices for panels.")
    p.add_argument("--viz_every", type=int, default=10)
    p.add_argument(
        "--viz_gate_steps",
        type=int,
        default=None,
        help="Gate refit steps for plots only; default = --gate_steps.",
    )
    p.add_argument("--viz_gate_lr", type=float, default=0.05)
    p.add_argument("--phase2_steps", type=int, default=80, help="Explore MC steps.")
    p.add_argument("--phase3_steps", type=int, default=30, help="Refine MC steps.")
    p.add_argument("--explore_beta_kl", type=float, default=0.01)
    p.add_argument("--refine_beta_kl", type=float, default=0.1)
    p.add_argument("--explore_kl_warmup", type=int, default=4)
    p.add_argument("--refine_kl_warmup", type=int, default=6)
    p.add_argument("--explore_init_log_std", type=float, default=-2.5)
    p.add_argument("--refine_init_log_std", type=float, default=-3.0)
    p.add_argument("--explore_entropy_beta", type=float, default=0.02)
    p.add_argument(
        "--advantage_variant",
        default="center_std",
        choices=sorted(ADVANTAGE_VARIANTS),
        help="Per-anchor REINFORCE advantage + anchor weight preset.",
    )
    p.add_argument("--mc_samples", type=int, default=5)
    p.add_argument("--mc_lr", type=float, default=0.008)
    p.add_argument("--mc_train_log_std", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mc_w_signal_var", type=float, default=1.0)
    p.add_argument("--mc_include_residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mc_grad_clip_norm", type=float, default=10.0)
    p.add_argument("--learn_inducing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--inducing_lr_mult", type=float, default=0.1)
    p.add_argument(
        "--refine_freeze_cem",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, refine phase uses 0 hyper/gate steps (old behavior). Default: full CEM each W sample.",
    )
    p.add_argument("--final_hyper_steps", type=int, default=None)
    p.add_argument("--final_gate_steps", type=int, default=None)
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
    p.add_argument("--outlier_constant_steps", type=int, default=10)
    p.add_argument("--outlier_constant_lr", type=float, default=0.05)
    p.add_argument("--outlier_constant_l2", type=float, default=5.0)
    p.add_argument("--outlier_constant_min_logp", type=float, default=-12.0)
    p.add_argument("--outlier_constant_max_logp", type=float, default=-1e-4)
    p.add_argument("--qr_w_lengthscale", type=float, default=None, help="Override; default = median inducing distance.")
    p.add_argument("--min_inliers", type=int, default=2)
    p.add_argument("--max_inlier_frac", type=float, default=0.95)
    p.add_argument("--outlier_mode", default="background_normal")
    p.add_argument("--outlier_sigma_mult", type=float, default=2.5)
    p.add_argument("--background_scale_mult", type=float, default=3.0)
    p.add_argument("--learn_outlier_constant", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--final_ensemble_samples", type=int, default=5)
    p.add_argument("--include_baseline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--standardize_y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Z-score y with train mean/std before CEM/PLS/reward. Default: raw y.",
    )
    p.add_argument("--out_dir", default="experiments/synthetic/l2_supervised_reinforce_seed0")
    p.add_argument(
        "--log_sample_rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log per-iteration per-W supervised_anchor reward to reward_trace_*.csv/json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(SETTING_PRESETS[args.setting])
    if args.noise_std is not None:
        ns = float(args.noise_std)
        cfg["noise_std"] = ns
        cfg["noise_var"] = ns * ns
    _set_seed(args.seed)
    print(f"phantom noise_std={cfg.get('noise_std')}  noise_var={cfg.get('noise_var')}", flush=True)

    data = _generate_paper_data(cfg, args.seed, device)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data["X_train"]).astype(np.float64)
    X_test = scaler.transform(data["X_test"]).astype(np.float64)[: int(args.max_test)]
    y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
    y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)[: int(args.max_test)]
    z_test_phys = np.asarray(data["z_test"], dtype=np.float64)[: int(args.max_test)]
    if bool(args.standardize_y):
        y_mean = float(y_train.mean())
        y_std = float(y_train.std() if y_train.std() > 1e-8 else 1.0)
        y_train_head = ((y_train - y_mean) / y_std).astype(np.float64)
    else:
        y_mean, y_std = 0.0, 1.0
        y_train_head = y_train.astype(np.float64)
    Q = int(cfg["Q"])
    print(
        f"standardize_y={bool(args.standardize_y)}  "
        f"y_train range=[{y_train.min():.4g}, {y_train.max():.4g}]  "
        f"head range=[{y_train_head.min():.4g}, {y_train_head.max():.4g}]",
        flush=True,
    )

    X_train_t = torch.as_tensor(X_train, device=device, dtype=torch.float64)
    X_test_t = torch.as_tensor(X_test, device=device, dtype=torch.float64)
    y_train_head_t = torch.as_tensor(y_train_head, device=device, dtype=torch.float64)

    W0_np = _projection_W0(X_train, y_train_head, Q, str(args.projection_init), seed=args.seed + 513)
    W0 = torch.as_tensor(W0_np, device=device, dtype=torch.float64)
    R_init_W = W0.clone()

    X_neighbors, y_neighbors_head, neighbor_idx = _neighbor_stack(
        X_test_t, X_train_t, y_train_head_t, n_neighbors=int(args.n_neighbors),
    )
    inducing_idx = _kmeans_medoids(X_train, int(args.m_inducing), args.seed + 7919)
    X_inducing = X_train_t[torch.as_tensor(inducing_idx, device=device, dtype=torch.long)]
    qr_w_lengthscale = (
        float(args.qr_w_lengthscale)
        if args.qr_w_lengthscale is not None
        else float(_median_distance(X_inducing))
    )
    z_train_phys = np.asarray(data.get("z_train", data["z_test"]), dtype=np.float64)

    reward_count = min(int(args.reward_anchor_count), X_train.shape[0])
    reward_idx = _kmeans_medoids(X_train, reward_count, args.seed + 4271)
    X_reward_t = X_train_t[torch.as_tensor(reward_idx, device=device, dtype=torch.long)]
    y_reward_head_t = y_train_head_t[torch.as_tensor(reward_idx, device=device, dtype=torch.long)]
    X_reward_nb, y_reward_nb_head, _ = _neighbor_stack(
        X_reward_t, X_train_t, y_train_head_t, n_neighbors=int(args.n_neighbors),
    )

    print("Phase 0: W0 self-CEM init on test + reward anchors", flush=True)
    cem_test, _ = _run_meanw_cem_init(X_test_t, X_neighbors, y_neighbors_head, W0)
    cem_reward, _ = _run_meanw_cem_init(X_reward_t, X_reward_nb, y_reward_nb_head, W0)

    rows: list[dict[str, Any]] = []
    if args.include_baseline:
        rows.append(_baseline_self_cem_row(
            args,
            X_anchor=X_test_t,
            X_neighbors=X_neighbors,
            y_neighbors_std=y_neighbors_head,
            cem_state=cem_test,
            W0=W0,
            y_test=y_test,
            y_mean=y_mean,
            y_std=y_std,
        ))
        rows[-1]["variant"] = "baseline_W0_test"
        rows[-1]["setting"] = str(args.setting)
        rows[-1]["standardize_y"] = bool(args.standardize_y)

    viz_anchor_ids = [int(x) for x in str(args.viz_anchors).split(",") if x.strip()]
    viz_gate_steps = int(args.gate_steps if args.viz_gate_steps is None else args.viz_gate_steps)
    history_all: list[dict[str, float]] = []

    def _neighbor_regime_labels(j: int) -> np.ndarray:
        idx = neighbor_idx[j].detach().cpu().numpy().astype(np.int64)
        return regime_labels_from_phys(z_train_phys[idx])

    def make_viz_callback(
        phase: str,
        cem_state_ref: dict[str, torch.Tensor],
        inducing_X_fixed: torch.Tensor,
    ):
        def _cb(step: int, R_mu: torch.Tensor, R_log_std: torch.Tensor, inducing_X_eval=None):
            if int(args.viz_every) <= 0 or step % int(args.viz_every) != 0:
                return {}
            U = inducing_X_fixed if inducing_X_eval is None else inducing_X_eval
            for j in viz_anchor_ids:
                if j < 0 or j >= X_test_t.shape[0]:
                    continue
                plot_l2_step_diagnostics(
                    out_dir=viz_dir,
                    step=step,
                    phase=phase,
                    anchor_idx=j,
                    label=f"test{j}",
                    X_anchor=X_test_t[j],
                    X_neighbors=X_neighbors[j],
                    y_neighbors=y_neighbors_head[j],
                    W0=W0,
                    R_mu=R_mu,
                    R_log_std=R_log_std,
                    inducing_X=U,
                    cem_state=cem_state_ref,
                    cem_anchor_idx=j,
                    qr_w_lengthscale=qr_w_lengthscale,
                    mc_w_signal_var=float(args.mc_w_signal_var),
                    z_phys_anchor=z_test_phys[j],
                    z_phys_neighbors=z_train_phys[neighbor_idx[j].cpu().numpy()],
                    regime_labels=_neighbor_regime_labels(j),
                    n_W_samples=3,
                    seed=args.seed,
                    viz_gate_steps=viz_gate_steps,
                    viz_gate_lr=float(args.viz_gate_lr),
                )
            return {}
        return _cb

    if int(args.viz_every) >= 0:
        for j in viz_anchor_ids:
            if 0 <= j < X_test_t.shape[0]:
                lab_j = cem_test["labels"][j] if cem_test["labels"].dim() > 1 else cem_test["labels"]
                g = cem_test["gate"]
                gate_j = g[j] if g.dim() > 1 else g
                with torch.enable_grad():
                    gate_w0 = fit_gate_for_projection(
                        X_test_t[j], X_neighbors[j], lab_j, W0,
                        gate_init=gate_j,
                        gate_steps=viz_gate_steps,
                        gate_lr=float(args.viz_gate_lr),
                    )
                plot_l2_projection_panel(
                    out_path=viz_dir / f"step0000_warm_j{j}_test{j}_W0.png",
                    X_anchor=X_test_t[j],
                    X_neighbors=X_neighbors[j],
                    y_neighbors=y_neighbors_head[j],
                    W_plot=W0,
                    W0=W0,
                    gate=gate_w0,
                    labels=_neighbor_regime_labels(j),
                    title=f"warm W0 baseline  j={j}",
                    z_phys_anchor=z_test_phys[j],
                    z_phys_neighbors=z_train_phys[neighbor_idx[j].cpu().numpy()],
                )

    # Phase 2: explore
    print(f"Phase 2: explore REINFORCE ({args.phase2_steps} steps)", flush=True)
    explore_cem_cfg = _self_cem_config(args)
    print(
        f"  inner CEM per W sample: cem_updates={args.cem_updates}, "
        f"hyper_steps={args.hyper_steps}, gate_steps={args.gate_steps}, refit_after_update=True",
        flush=True,
    )
    qr_explore = train_qr_mc_cem_mstep(
        X_reward_t,
        X_reward_nb,
        y_reward_nb_head,
        cem_reward["labels"],
        y_anchor=y_reward_head_t,
        inducing_X=X_inducing,
        init_W=R_init_W,
        init_lengthscale=cem_reward["lengthscale"],
        init_signal_var=cem_reward["signal_var"],
        init_noise_var=cem_reward["noise_var"],
        gate_init=cem_reward["gate"],
        cem_config=explore_cem_cfg,
        config=_mc_config(args, steps=int(args.phase2_steps), phase="explore", seed=args.seed + 11, qr_w_lengthscale=qr_w_lengthscale),
        eval_callback=make_viz_callback("explore", cem_test, X_inducing),
    )
    for h in qr_explore.history:
        h["phase"] = "explore"
        history_all.append(h)
    if bool(args.log_sample_rewards):
        _export_sample_reward_trace(qr_explore.sample_reward_trace, phase="explore", out_dir=out_dir)

    # Phase 3: refine
    if bool(args.refine_freeze_cem):
        print(f"Phase 3: refine (frozen CEM nuisances, {args.phase3_steps} MC steps)", flush=True)
        refine_cem_cfg = _frozen_cem_config(args)
    else:
        print(f"Phase 3: refine (full CEM per W sample, {args.phase3_steps} MC steps)", flush=True)
        refine_cem_cfg = _self_cem_config(args)
    qr_refine = train_qr_mc_cem_mstep(
        X_reward_t,
        X_reward_nb,
        y_reward_nb_head,
        cem_reward["labels"],
        y_anchor=y_reward_head_t,
        inducing_X=qr_explore.inducing_X,
        init_R_mu=qr_explore.R_mu.detach(),
        init_R_log_std=qr_explore.R_log_std.detach(),
        init_lengthscale=cem_reward["lengthscale"],
        init_signal_var=cem_reward["signal_var"],
        init_noise_var=cem_reward["noise_var"],
        gate_init=cem_reward["gate"],
        cem_config=refine_cem_cfg,
        config=_mc_config(args, steps=int(args.phase3_steps), phase="refine", seed=args.seed + 29, qr_w_lengthscale=qr_w_lengthscale),
        eval_callback=make_viz_callback("refine", cem_test, qr_explore.inducing_X),
    )
    for h in qr_refine.history:
        h["phase"] = "refine"
        history_all.append(h)
    if bool(args.log_sample_rewards):
        _export_sample_reward_trace(qr_refine.sample_reward_trace, phase="refine", out_dir=out_dir)

    # Phase 4: final eval on test
    print("Phase 4: final ensemble eval on test anchors", flush=True)
    from djgp.projections.uncertain_w_jgp import sparse_gp_uncertain_w_moments_for_mode

    w_moments = sparse_gp_uncertain_w_moments_for_mode(
        mode="anchor",
        X_anchor=X_test_t,
        X_neighbors=X_neighbors,
        inducing_X=qr_refine.inducing_X,
        R_mu=qr_refine.R_mu,
        R_log_std=qr_refine.R_log_std,
        lengthscale=qr_w_lengthscale,
        signal_var=float(args.mc_w_signal_var),
        include_conditional_residual=bool(args.mc_include_residual),
    )
    pred_row, pred_diag = _final_sampled_self_cem_predict(
        args,
        seed=args.seed,
        X_anchor=X_test_t,
        X_neighbors=X_neighbors,
        y_neighbors_std=y_neighbors_head,
        init_state=cem_test,
        w_moments=w_moments,
        y_test=y_test,
        y_mean=y_mean,
        y_std=y_std,
        hyper_steps_override=int(args.final_hyper_steps) if args.final_hyper_steps is not None else None,
        gate_steps_override=int(args.final_gate_steps) if args.final_gate_steps is not None else None,
    )
    row = {
        "variant": "supervised_reinforce_final",
        "setting": str(args.setting),
        "status": "ok",
        "phase2_steps": int(args.phase2_steps),
        "phase3_steps": int(args.phase3_steps),
        "reward_anchor_count": int(reward_count),
        "mc_samples": int(args.mc_samples),
        "explore_beta_kl": float(args.explore_beta_kl),
        "refine_beta_kl": float(args.refine_beta_kl),
        "standardize_y": bool(args.standardize_y),
        "qr_explore_train_sec": float(qr_explore.train_sec),
        "qr_refine_train_sec": float(qr_refine.train_sec),
    }
    row.update(pred_row)
    row.update(pred_diag)
    rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "metrics.csv", rows)
    _write_csv(out_dir / "aggregate.csv", _aggregate(rows))
    with (out_dir / "mc_history.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(history_all), f, indent=2)
    summary = {
        "setting": args.setting,
        "seed": args.seed,
        "device": str(device),
        "Q": Q,
        "n_test": int(X_test_t.shape[0]),
        "reward_anchor_count": reward_count,
        "final_rmse": float(pred_row.get("rmse", float("nan"))),
        "baseline_rmse": float(rows[0].get("rmse", float("nan"))) if rows else float("nan"),
        "standardize_y": bool(args.standardize_y),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

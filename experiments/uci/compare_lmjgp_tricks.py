"""
LMJGP training-time tricks ablation on UCI.

For each ``(dataset, seed)`` we re-train LMJGP under several variants that share the same
data, neighbourhood structure, ``Q``, ``m1``, ``m2``, ``n``, learning rate, and number of
optimisation steps.  Every variant ends with the same evaluation pipeline (``predict_vi``
+ Gaussian PI calibration + Mahalanobis-metric posterior diagnostics) so differences come
purely from training-side knobs.

Variants implemented:

- ``baseline``      — random ``mu_V``, full KL, no row-norm, no aux loss (matches
                      :mod:`experiments.uci.compare_metric_variants`).
- ``kl_warmup``     — linear ``beta_W: 0.01 -> 1.0`` over the first ``warmup_steps`` steps.
- ``pls_init``      — ``mu_V`` initialised from PLS (raw-X), full KL.
- ``sir_init``      — ``mu_V`` initialised from SIR (raw-X composed with the standardiser),
                      full KL.
- ``row_norm``      — soft penalty on the **second moment**::

                          lambda * mean((E[||W_{t,q,:}||^2] - 1)^2)
                              with E[||W_{t,q,:}||^2] = ||mu_W,k:||^2 + sum_d Var(W,k:d).
                      (Use ``--row_norm_mode mu`` for the legacy first-moment-only
                      variant whose gradient vanishes at ``mu_W = 0``.)
- ``aux_metric``    — soft penalty on full Mahalanobis distance::

                          -lambda * mean_pearson(d^2_{A_t}(x_i,x_j), (y_i-y_j)^2)
                              with d^2_{A_t}(x_i,x_j) = (x_i-x_j)^T A_t (x_i-x_j),
                              A_t = E_q[W_t^T W_t] = mu_W^T mu_W + sum_q cov_W_{q}.
                      (Use ``--aux_metric_mode mu`` for the older ``mu_W``-only variant.)
- ``pls_kl``        — ``pls_init`` + ``kl_warmup`` (no row-norm, no aux). Recommended as
                      the first head-to-head comparison against ``baseline``.
- ``combo``         — ``pls_init`` + ``kl_warmup`` + new ``row_norm`` + new ``aux_metric``.
- ``combo_full``    — same as ``combo`` but using ``row_norm_mode="mu"`` and
                      ``aux_metric_mode="mu"`` (legacy aggressive form).

For each variant we report (per seed):

- the standard 7-tuple metric row used elsewhere
  ``[rmse, mean_crps, wall_sec, cov90, w90, cov95, w95]`` from ``predict_vi``;
- a small posterior-metric diagnostics block from ``A_j = E_q[W_j^T W_j]`` so we can see
  whether the trick actually changed the learned Mahalanobis metric (median trace,
  median effective rank of ``A_j``, median ``E[||mu_W_q,:||^2]``, etc.);
- training history (loss, ``beta_W``, KL_V/T, row-norm, aux corr) sampled at
  ``log_interval``;
- step-by-step ``q(W)`` snapshots at ``--snapshot_steps`` (default ``0,10,50`` plus the
  final step). Each snapshot reports::

      median ||mu_W||_F      median ||mu_W,k:||^2
      median sum_d Var(W,k:d)   median E[||W,k:||^2]
      median trace(A_j)         median eff_rank(A_j)

  This lets us tell ``init failed (step 0 already ≈ 0)`` apart from
  ``init worked but training pulled mean back to 0``.

Existing entry points (``compare_uci_baselines.py``, ``compare_metric_variants.py``,
``run_projection_w_diagnostics.py``) are unchanged.

How to run (repository root, conda env: ``jumpGP``):

  conda activate jumpGP

  # Smoke test
  python experiments/uci/compare_lmjgp_tricks.py \\
      --num_exp 1 --num_steps 500 --max_anchors 64 \\
      --datasets "Wine Quality" \\
      --variants baseline,pls_init,kl_warmup,combo

    python experiments/uci/compare_lmjgp_tricks.py ^
      --num_exp 1 --num_steps 500 --max_anchors 64 ^
      --datasets "Wine Quality" ^
      --variants baseline,pls_init,kl_warmup,combo

  # Full sweep on all UCI datasets we routinely report
  python experiments/uci/compare_lmjgp_tricks.py \\
      --num_exp 3 --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"

Agents: update this docstring when CLI defaults or variant names change.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from experiments.uci.uci_data import load_uci_split  # noqa: E402
from shared.jumpgp_runner import compute_metrics, find_neighborhoods  # noqa: E402
from djgp.diagnostics.projection_posterior import (  # noqa: E402
    metric_A_from_moment,
    sanitize_for_json,
    sym_matrix_stats,
)
from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402
from djgp.variational import predict_vi, qW_from_qV, train_vi  # noqa: E402
from djgp.variational_tricks import (  # noqa: E402
    KLWarmupSchedule,
    calibrate_mu_V_to_target_mu_W,
    init_lengthscales_median_heuristic,
    init_mu_V_supervised,
    train_vi_with_tricks,
)


VARIANT_NAMES = (
    "baseline",
    "kl_warmup",
    "pls_init",
    "sir_init",
    "row_norm",
    "aux_metric",
    "pls_kl",      # PLS init + KL warm-up only — recommended starting point.
    "combo",       # PLS init + KL warm-up + row_norm(second_moment) + aux(A).
    "combo_full",  # combo + extra (kept for completeness; matches the previous "combo").
)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_regions(X_test, X_train, Y_train, device, m1: int, Q: int, n: int):
    neighborhoods = find_neighborhoods(X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n)
    regions = []
    T = X_test.shape[0]
    for i in range(T):
        X_nb = neighborhoods[i]["X_neighbors"].to(device)
        y_nb = neighborhoods[i]["y_neighbors"].to(device)
        regions.append({"X": X_nb, "y": y_nb, "C": torch.randn(m1, Q, device=device)})
    return regions


def _make_initial_params(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    Q: int,
    m1: int,
    m2: int,
    T: int,
    D: int,
    device: torch.device,
    mu_V_mode: str,
    init_seed: int,
) -> tuple[dict, list[dict], dict, dict[str, Any]]:
    """Build ``(V_params, u_params, hyperparams, info)`` with optional supervised mu_V."""
    info: dict[str, Any] = {"mu_V_mode": mu_V_mode}
    W_init_tensor: torch.Tensor | None = None
    if mu_V_mode == "random":
        mu_V_init = torch.randn(m2, Q, D, device=device)
    elif mu_V_mode in ("pls", "sir"):
        mu_V_init, sup_info, W_init_tensor = init_mu_V_supervised(
            X_train,
            Y_train,
            m2=m2,
            Q=Q,
            mode=mu_V_mode,
            sir_H=10,
            perturb_std=1e-3,
            seed=init_seed,
            device=device,
            return_W=True,
        )
        info["supervised_init"] = sup_info
    else:
        raise ValueError(f"Unknown mu_V_mode={mu_V_mode!r}.")

    V_params = {
        "mu_V": mu_V_init.detach().clone().requires_grad_(True),
        "sigma_V": torch.rand(m2, Q, D, device=device).requires_grad_(True),
    }
    u_params = []
    for _ in range(T):
        u_params.append(
            {
                "U_logit": torch.zeros(1, device=device, requires_grad=True),
                "mu_u": torch.randn(m1, device=device, requires_grad=True),
                "Sigma_u": torch.eye(m1, device=device, requires_grad=True),
                "sigma_noise": torch.tensor(0.5, device=device, requires_grad=True),
                "sigma_k": torch.tensor(0.5, device=device, requires_grad=True),
                "omega": torch.randn(Q + 1, device=device, requires_grad=True),
            }
        )
    X_train_mean = X_train.mean(dim=0)
    X_train_std = X_train.std(dim=0)
    Z = X_train_mean + torch.randn(m2, D, device=device) * X_train_std
    # Median-heuristic lengthscales so K_xz isn't tiny at init (used by ALL variants for
    # apples-to-apples comparison; without this, supervised init damps to mu_W ≈ 0).
    ls_init = init_lengthscales_median_heuristic(X_test, Z, Q=Q)
    hyperparams = {
        "Z": Z.requires_grad_(True),
        "X_test": X_test,
        "lengthscales": ls_init.clone().requires_grad_(True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    info["lengthscale_init_median_heuristic"] = float(ls_init[0].item())

    if W_init_tensor is not None:
        # GP interpolation A = K_xz K_zz^{-1} is not row-stochastic; rescale mu_V so
        # mu_W at step 0 has unit row magnitude in the supervised direction.
        info["calibration"] = calibrate_mu_V_to_target_mu_W(
            V_params,
            hyperparams,
            X_test,
            W_init_tensor,
            per_row=True,
            max_alpha=5.0,
            target_row_norm=1.0,
        )
    return V_params, u_params, hyperparams, info


def _post_train_metric_block(
    V_params: dict,
    hyperparams: dict,
    X_test: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``A_j`` median diagnostics summarising the learned Mahalanobis metric."""
    with torch.no_grad():
        mu_W, cov_W = qW_from_qV(
            X_test,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
    A = metric_A_from_moment(mu_W.detach(), cov_W.detach())
    per_anchor_stats = [sym_matrix_stats(A[t]) for t in range(A.shape[0])]
    medians = {
        "median_trace_A": float(np.nanmedian([s["trace"] for s in per_anchor_stats])),
        "median_eff_rank_A": float(np.nanmedian([s["eff_rank"] for s in per_anchor_stats])),
        "median_top_eig_A": float(np.nanmedian([s["top_eig"] for s in per_anchor_stats])),
        "median_min_eig_A_pos": float(np.nanmedian([s["min_eig_pos"] for s in per_anchor_stats])),
    }
    EW_row_sq = (mu_W * mu_W).sum(dim=-1)
    medians["median_row_norm_sq_mu_W"] = float(EW_row_sq.median().item())
    medians["mean_row_norm_sq_mu_W"] = float(EW_row_sq.mean().item())
    medians["median_frob_mu_W"] = float(torch.linalg.norm(mu_W, dim=(-2, -1)).median().item())
    return A, medians


def _evaluate_predict_vi(
    regions,
    V_params,
    hyperparams,
    Y_test: torch.Tensor,
    *,
    MC_num: int,
) -> tuple[list[float], dict[str, float]]:
    t0 = time.perf_counter()
    mu_pred, var_pred = predict_vi(regions, V_params, hyperparams, M=MC_num)
    elapsed = time.perf_counter() - t0
    sigmas = torch.sqrt(torch.clamp(var_pred, min=1e-12))
    rmse, crps = compute_metrics(mu_pred, sigmas, Y_test)
    rmse_v, crps_v, elapsed_v = float(rmse.detach().item()), float(crps.detach().item()), float(elapsed)
    pack = pack_gaussian_uq_list(
        rmse_v,
        crps_v,
        elapsed_v,
        Y_test.detach().cpu().numpy(),
        mu_pred.detach().cpu().numpy(),
        sigmas.detach().cpu().numpy(),
    )
    return pack, {"rmse": rmse_v, "crps": crps_v, "wall_sec": elapsed_v}


def _variant_config(
    variant: str,
    *,
    num_steps: int,
    warmup_frac: float,
    row_norm_weight: float,
    row_norm_mode: str,
    aux_metric_weight: float,
    aux_metric_mode: str,
) -> dict[str, Any]:
    """Resolve variant name -> trick switches."""
    warm = KLWarmupSchedule(
        beta_init=0.01,
        beta_final=1.0,
        warmup_steps=max(1, int(num_steps * warmup_frac)),
        shape="linear",
    )
    cfg = {
        "mu_V_mode": "random",
        "kl_warmup": None,
        "row_norm_weight": 0.0,
        "row_norm_mode": row_norm_mode,
        "aux_metric_weight": 0.0,
        "aux_metric_mode": aux_metric_mode,
    }
    if variant == "baseline":
        return cfg
    if variant == "kl_warmup":
        cfg["kl_warmup"] = warm
        return cfg
    if variant == "pls_init":
        cfg["mu_V_mode"] = "pls"
        return cfg
    if variant == "sir_init":
        cfg["mu_V_mode"] = "sir"
        return cfg
    if variant == "row_norm":
        cfg["row_norm_weight"] = float(row_norm_weight)
        return cfg
    if variant == "aux_metric":
        cfg["aux_metric_weight"] = float(aux_metric_weight)
        return cfg
    if variant == "pls_kl":
        cfg["mu_V_mode"] = "pls"
        cfg["kl_warmup"] = warm
        return cfg
    if variant == "combo":
        cfg["mu_V_mode"] = "pls"
        cfg["kl_warmup"] = warm
        cfg["row_norm_weight"] = float(row_norm_weight)
        cfg["aux_metric_weight"] = float(aux_metric_weight)
        return cfg
    if variant == "combo_full":
        cfg["mu_V_mode"] = "pls"
        cfg["kl_warmup"] = warm
        cfg["row_norm_weight"] = float(row_norm_weight)
        cfg["row_norm_mode"] = "mu"  # legacy first-moment penalty
        cfg["aux_metric_weight"] = float(aux_metric_weight)
        cfg["aux_metric_mode"] = "mu"  # legacy mu-based aux
        return cfg
    raise ValueError(f"Unknown variant {variant!r}.")


def _train_one_variant(
    variant: str,
    *,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    Q: int,
    m1: int,
    m2: int,
    n: int,
    num_steps: int,
    lr: float,
    log_interval: int,
    MC_num: int,
    warmup_frac: float,
    row_norm_weight: float,
    row_norm_mode: str,
    aux_metric_weight: float,
    aux_metric_mode: str,
    aux_max_pairs: int,
    snapshot_steps: list[int],
    init_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    T, D = X_test.shape
    cfg = _variant_config(
        variant,
        num_steps=num_steps,
        warmup_frac=warmup_frac,
        row_norm_weight=row_norm_weight,
        row_norm_mode=row_norm_mode,
        aux_metric_weight=aux_metric_weight,
        aux_metric_mode=aux_metric_mode,
    )
    mu_V_mode = cfg["mu_V_mode"]
    kl_warmup = cfg["kl_warmup"]
    rn_w = cfg["row_norm_weight"]
    rn_mode = cfg["row_norm_mode"]
    ax_w = cfg["aux_metric_weight"]
    ax_mode = cfg["aux_metric_mode"]

    _seed_all(init_seed)
    regions = _build_regions(X_test, X_train, Y_train, device, m1, Q, n)
    try:
        V_params, u_params, hyperparams, init_info = _make_initial_params(
            X_train,
            Y_train,
            X_test,
            Q=Q,
            m1=m1,
            m2=m2,
            T=T,
            D=D,
            device=device,
            mu_V_mode=mu_V_mode,
            init_seed=init_seed,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "variant": variant,
            "error": f"init_failed: {type(exc).__name__}: {exc}",
        }

    history: list[dict[str, float]] = []
    snapshots: list[dict[str, float]] = []
    t0 = time.perf_counter()
    # Always go through train_vi_with_tricks so we get snapshots and a uniform breakdown.
    V_params, u_params, hyperparams, history, snapshots = train_vi_with_tricks(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=lr,
        num_steps=num_steps,
        log_interval=log_interval,
        snapshot_steps=snapshot_steps,
        kl_warmup=kl_warmup,
        row_norm_weight=rn_w,
        row_norm_mode=rn_mode,
        aux_metric_weight=ax_w,
        aux_metric_mode=ax_mode,
        aux_max_pairs=aux_max_pairs,
        aux_seed=init_seed,
    )
    train_time = time.perf_counter() - t0

    _A, metric_block = _post_train_metric_block(V_params, hyperparams, X_test)
    pack, raw_eval = _evaluate_predict_vi(regions, V_params, hyperparams, Y_test, MC_num=MC_num)

    return {
        "variant": variant,
        "init_info": init_info,
        "hyperparams_summary": {
            "kl_warmup": None if kl_warmup is None else {
                "beta_init": kl_warmup.beta_init,
                "beta_final": kl_warmup.beta_final,
                "warmup_steps": kl_warmup.warmup_steps,
                "shape": kl_warmup.shape,
            },
            "row_norm_weight": rn_w,
            "row_norm_mode": rn_mode,
            "aux_metric_weight": ax_w,
            "aux_metric_mode": ax_mode,
        },
        "train_time_sec": float(train_time),
        "metric_A_summary": metric_block,
        "predict_vi_pack": pack,
        "predict_vi_raw": raw_eval,
        "snapshots": snapshots,
        "history_tail": history[-5:] if history else [],
    }


def _per_dataset_run(
    dataset_name: str,
    seed: int,
    *,
    variants: list[str],
    num_steps: int,
    log_interval: int,
    MC_num: int,
    Q_override: int | None,
    max_anchors: int | None,
    warmup_frac: float,
    row_norm_weight: float,
    row_norm_mode: str,
    aux_metric_weight: float,
    aux_metric_mode: str,
    aux_max_pairs: int,
    snapshot_steps: list[int],
    device: torch.device,
) -> dict[str, Any]:
    X_train_np, X_test_np, y_train_np, y_test_np, K = load_uci_split(dataset_name, seed)
    Q = Q_override if Q_override is not None else K
    X_train = torch.from_numpy(X_train_np).float().to(device)
    Y_train = torch.from_numpy(y_train_np).float().squeeze(-1).to(device)
    X_test = torch.from_numpy(X_test_np).float().to(device)
    Y_test = torch.from_numpy(y_test_np).float().squeeze(-1).to(device)
    if max_anchors is not None:
        k = min(max_anchors, X_test.shape[0])
        X_test = X_test[:k]
        Y_test = Y_test[:k]

    init_seed = int(seed) * 1009 + 7
    out_variants: dict[str, Any] = {}
    for variant in variants:
        try:
            out_variants[variant] = _train_one_variant(
                variant,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                Q=Q,
                m1=2,
                m2=40,
                n=20,
                num_steps=num_steps,
                lr=0.01,
                log_interval=log_interval,
                MC_num=MC_num,
                warmup_frac=warmup_frac,
                row_norm_weight=row_norm_weight,
                row_norm_mode=row_norm_mode,
                aux_metric_weight=aux_metric_weight,
                aux_metric_mode=aux_metric_mode,
                aux_max_pairs=aux_max_pairs,
                snapshot_steps=snapshot_steps,
                init_seed=init_seed,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            out_variants[variant] = {
                "variant": variant,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "label": f"uci:{dataset_name}",
        "dataset": dataset_name,
        "seed": int(seed),
        "Q": Q,
        "max_anchors_if_subsampled": max_anchors,
        "n_test_anchors_used": int(X_test.shape[0]),
        "variants": out_variants,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="LMJGP training-time tricks ablation on UCI."
    )
    p.add_argument("--num_exp", type=int, default=3, help="Number of seeds per dataset.")
    p.add_argument(
        "--datasets",
        type=str,
        default="Wine Quality",
        help="Comma-separated subset of: Wine Quality, Parkinsons Telemonitoring, Appliances Energy Prediction",
    )
    p.add_argument(
        "--variants",
        type=str,
        default=",".join(VARIANT_NAMES),
        help=f"Comma-separated subset of {VARIANT_NAMES}.",
    )
    p.add_argument("--num_steps", type=int, default=300, help="train_vi steps for LMJGP.")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--MC_num", type=int, default=3, help="MC samples used by predict_vi.")
    p.add_argument("--Q", type=int, default=None, help="Override latent dimension Q.")
    p.add_argument(
        "--max_anchors",
        type=int,
        default=None,
        help="Use only the first K test anchors (faster).",
    )
    p.add_argument(
        "--warmup_frac",
        type=float,
        default=0.5,
        help="Fraction of num_steps used for KL warm-up (only when kl_warmup is enabled).",
    )
    p.add_argument(
        "--row_norm_weight",
        type=float,
        default=0.1,
        help="lambda for row-norm penalty (only used by row_norm / combo / combo_full).",
    )
    p.add_argument(
        "--row_norm_mode",
        type=str,
        default="second_moment",
        choices=("mu", "second_moment"),
        help="Penalise (||mu_W||^2 - 1)^2 ('mu') or (E[||W||^2] - 1)^2 ('second_moment').",
    )
    p.add_argument(
        "--aux_metric_weight",
        type=float,
        default=0.1,
        help="lambda for auxiliary metric loss (only used by aux_metric / combo / combo_full).",
    )
    p.add_argument(
        "--aux_metric_mode",
        type=str,
        default="A",
        choices=("mu", "A"),
        help="Aux loss uses pairwise distances under mu_W ('mu') or full A=E[W^TW] ('A').",
    )
    p.add_argument(
        "--aux_max_pairs",
        type=int,
        default=64,
        help="Sampled pairs per region per step for auxiliary metric loss.",
    )
    p.add_argument(
        "--snapshot_steps",
        type=str,
        default="0,10,50",
        help=(
            "Comma-separated step indices at which to snapshot q(W) diagnostics."
            " The final step (num_steps) is always added automatically."
        ),
    )
    p.add_argument("--out", type=str, default="", help="Pickle output path.")
    p.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional JSON output path (NaN/Inf sanitised to null).",
    )
    return p.parse_args()


def _fmt(x: float) -> str:
    return f"{x:7.4f}" if isinstance(x, float) and np.isfinite(x) else "    nan"


def _print_summary_row(dataset: str, seed: int, res: dict[str, Any]) -> None:
    print(f"\n[{dataset}] seed={seed}")
    print(
        "  variant         rmse    crps    cov90   w90     cov95   w95    "
        "med_tr_A  med_eff_A  med_||muW||^2  med_E||W||^2"
    )
    for variant, info in res["variants"].items():
        if "error" in info:
            print(f"  {variant:<14}  ERROR: {info['error']}")
            continue
        pack = info["predict_vi_pack"]
        rmse, crps, _wall, c90, w90, c95, w95 = pack
        m = info["metric_A_summary"]
        snaps = info.get("snapshots", [])
        e_norm_final = float("nan")
        if snaps:
            e_norm_final = snaps[-1].get("median_E_row_norm_sq", float("nan"))

        print(
            f"  {variant:<14} {_fmt(rmse)} {_fmt(crps)} {_fmt(c90)} {_fmt(w90)} {_fmt(c95)} {_fmt(w95)} "
            f"{_fmt(m['median_trace_A'])} {_fmt(m['median_eff_rank_A'])} "
            f"{_fmt(m['median_row_norm_sq_mu_W'])} {_fmt(e_norm_final)}"
        )


def _print_snapshots(res: dict[str, Any]) -> None:
    """For each variant print the (step, ||muW||, ||muW,k:||^2, E||W||^2, tr(A), eff(A))."""
    print("  step-by-step q(W) snapshots")
    print(
        "    variant         step    frob_muW  med_muW^2  med_var_sum  med_E_W_sq  "
        "med_tr_A  med_eff_A"
    )
    for variant, info in res["variants"].items():
        if "error" in info:
            continue
        for snap in info.get("snapshots", []):
            print(
                f"    {variant:<14} {int(snap['step']):>5d}  "
                f"{_fmt(snap['median_frob_mu_W'])} "
                f"{_fmt(snap['median_row_norm_sq_mu_W'])} "
                f"{_fmt(snap['median_row_var_sum'])} "
                f"{_fmt(snap['median_E_row_norm_sq'])} "
                f"{_fmt(snap['median_trace_A'])} "
                f"{_fmt(snap['median_eff_rank_A'])}"
            )


def main():
    args = parse_args()
    dataset_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    variant_list = [s.strip() for s in args.variants.split(",") if s.strip()]
    unknown = [v for v in variant_list if v not in VARIANT_NAMES]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Choose from {VARIANT_NAMES}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, variants={variant_list}")

    snapshot_steps = [int(s) for s in args.snapshot_steps.split(",") if s.strip()]

    total_res: dict = {"args": vars(args), "results": {}}
    for dataset_name in dataset_list:
        total_res["results"][dataset_name] = {}
        for seed in range(args.num_exp):
            res = _per_dataset_run(
                dataset_name,
                seed,
                variants=variant_list,
                num_steps=args.num_steps,
                log_interval=args.log_interval,
                MC_num=args.MC_num,
                Q_override=args.Q,
                max_anchors=args.max_anchors,
                warmup_frac=args.warmup_frac,
                row_norm_weight=args.row_norm_weight,
                row_norm_mode=args.row_norm_mode,
                aux_metric_weight=args.aux_metric_weight,
                aux_metric_mode=args.aux_metric_mode,
                aux_max_pairs=args.aux_max_pairs,
                snapshot_steps=snapshot_steps,
                device=device,
            )
            total_res["results"][dataset_name][seed] = res
            _print_summary_row(dataset_name, seed, res)
            _print_snapshots(res)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"uci_lmjgp_tricks_{stamp}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(total_res, f)
    print(f"Saved pickle to {out_path}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(total_res), f, indent=2)
        print(f"Saved JSON to {args.out_json}")

    return total_res


if __name__ == "__main__":
    main()

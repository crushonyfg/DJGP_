"""
LMJGP first-layer metric ablation on UCI: ``learned vs isotropic vs random``.

For each (dataset, seed):

1. Train LMJGP with the existing :func:`djgp.variational.train_vi` API (defaults match
   the LMJGP block in :mod:`experiments.uci.compare_uci_baselines`). Also report the
   ``predict_vi`` numbers for reference.
2. Build ``A_j = E_q[W_j^T W_j]`` per test anchor from ``mu_W, cov_W``.
3. Construct three per-anchor ``Q x D`` projections with matched trace
   (``tr(L_j L_j^T) = tr(A_j)``):

       * ``learned``   : top-``Q`` eigendecomposition of ``A_j``
       * ``isotropic`` : random row-orthonormal scaled by ``sqrt(tr(A_j)/Q)``
       * ``random``    : random row-normalized scaled by ``sqrt(tr(A_j)/Q)``

4. Run JumpGP (CEM) on the projected neighborhood ``Z = X L_j^T`` per test point and
   report the same metrics row used elsewhere::

       [rmse, mean_crps, wall_sec, coverage_90, mean_width_90, coverage_95, mean_width_95]

Existing entry points (``compare_uci_baselines.py``, ``run_projection_w_diagnostics.py``)
are unchanged.

How to run (repository root, conda env: ``jumpGP``):

  conda activate jumpGP

  # Smoke test: 1 seed, 32 anchors, fewer VI steps
  python experiments/uci/compare_metric_variants.py --num_exp 1 --num_steps 500 --max_anchors 64 --datasets "Wine Quality"

  # Full run (still per-anchor JumpGP — slow on Appliances)
  python experiments/uci/compare_metric_variants.py --num_exp 1 --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"

Agents: update this docstring when CLI or defaults change.
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
)
from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402
from djgp.projections import (  # noqa: E402
    PROJECTION_MODES,
    build_projection_per_anchor,
    projection_quality_diagnostics,
    run_projected_jumpgp,
)
from djgp.variational import predict_vi, qW_from_qV, train_vi  # noqa: E402


def _build_regions(X_test, X_train, Y_train, device, m1: int, Q: int, n: int):
    neighborhoods = find_neighborhoods(X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n)
    regions = []
    X_nb_list, y_nb_list = [], []
    T = X_test.shape[0]
    for i in range(T):
        X_nb = neighborhoods[i]["X_neighbors"].to(device)
        y_nb = neighborhoods[i]["y_neighbors"].to(device)
        regions.append({"X": X_nb, "y": y_nb, "C": torch.randn(m1, Q, device=device)})
        X_nb_list.append(X_nb)
        y_nb_list.append(y_nb)
    return regions, X_nb_list, y_nb_list


def _train_lmjgp(
    X_train,
    Y_train,
    X_test,
    Y_test,
    *,
    Q: int,
    m1: int,
    m2: int,
    n: int,
    num_steps: int,
    lr: float,
    MC_num: int,
    device: torch.device,
):
    T, D = X_test.shape
    regions, X_nb_list, y_nb_list = _build_regions(X_test, X_train, Y_train, device, m1, Q, n)

    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
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
    hyperparams = {
        "Z": Z,
        "X_test": X_test,
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }

    t_train = time.perf_counter()
    V_params, u_params, hyperparams = train_vi(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=lr,
        num_steps=num_steps,
        log_interval=max(50, num_steps // 5),
    )
    train_time = time.perf_counter() - t_train

    mu_W, cov_W = qW_from_qV(
        X_test,
        hyperparams["Z"],
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
    )

    t_pred = time.perf_counter()
    mu_pred, var_pred = predict_vi(regions, V_params, hyperparams, M=MC_num)
    pred_time = time.perf_counter() - t_pred
    sigmas_predvi = torch.sqrt(torch.clamp(var_pred, min=1e-12))
    rmse_pv, crps_pv = compute_metrics(mu_pred, sigmas_predvi, Y_test)

    return {
        "regions": regions,
        "X_neighbors_list": X_nb_list,
        "y_neighbors_list": y_nb_list,
        "V_params": V_params,
        "u_params": u_params,
        "hyperparams": hyperparams,
        "mu_W": mu_W.detach(),
        "cov_W": cov_W.detach(),
        "predict_vi": {
            "mu": mu_pred.detach(),
            "sigma": sigmas_predvi.detach(),
            "rmse": float(rmse_pv.detach().item()),
            "mean_crps": float(crps_pv.detach().item()),
            "wall_sec": float(pred_time),
        },
        "train_time_sec": float(train_time),
    }


def _per_dataset_run(
    dataset_name: str,
    seed: int,
    *,
    num_steps: int,
    MC_num: int,
    Q_override: int | None,
    max_anchors: int | None,
    proj_seed_offset: int,
    device: torch.device,
):
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

    trained = _train_lmjgp(
        X_train,
        Y_train,
        X_test,
        Y_test,
        Q=Q,
        m1=2,
        m2=40,
        n=20,
        num_steps=num_steps,
        lr=0.01,
        MC_num=MC_num,
        device=device,
    )

    A = metric_A_from_moment(trained["mu_W"], trained["cov_W"])

    L_dict: dict[str, torch.Tensor] = {}
    for mode in PROJECTION_MODES:
        L_dict[mode] = build_projection_per_anchor(
            A,
            Q,
            mode=mode,
            seed=int(seed) * 7919 + proj_seed_offset + hash(mode) % 1000,
        )

    diagnostics = projection_quality_diagnostics(A, L_dict, Q=Q)

    y_test_np_arr = Y_test.detach().cpu().numpy()
    pv = trained["predict_vi"]
    pv_pack = pack_gaussian_uq_list(
        pv["rmse"],
        pv["mean_crps"],
        pv["wall_sec"],
        y_test_np_arr,
        pv["mu"].cpu().numpy(),
        pv["sigma"].cpu().numpy(),
    )

    method_results: dict[str, list[float]] = {"lmjgp_predict_vi": pv_pack}

    for mode in PROJECTION_MODES:
        out = run_projected_jumpgp(
            L_dict[mode],
            X_test,
            trained["X_neighbors_list"],
            trained["y_neighbors_list"],
            device=device,
            progress=True,
            desc=f"{dataset_name}/seed={seed}/{mode}",
        )
        mu_np = out["mu"]
        sig_np = out["sigma"]
        rmse = float(np.sqrt(np.mean((mu_np - y_test_np_arr) ** 2)))

        from djgp.evaluation import mean_gaussian_crps_torch

        crps = float(
            mean_gaussian_crps_torch(
                torch.from_numpy(y_test_np_arr).float(),
                torch.from_numpy(mu_np).float(),
                torch.from_numpy(sig_np).float(),
            ).item()
        )
        method_results[f"jgp_metric_{mode}"] = pack_gaussian_uq_list(
            rmse,
            crps,
            float(out["elapsed_sec"]),
            y_test_np_arr,
            mu_np,
            sig_np,
        )

    return {
        "label": f"uci:{dataset_name}",
        "dataset": dataset_name,
        "seed": int(seed),
        "Q": Q,
        "max_anchors_if_subsampled": max_anchors,
        "n_test_anchors_used": int(X_test.shape[0]),
        "lmjgp_train_time_sec": trained["train_time_sec"],
        "metrics": method_results,
        "diagnostics": diagnostics,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="LMJGP first-layer metric ablation on UCI: learned / isotropic / random projections."
    )
    p.add_argument("--num_exp", type=int, default=3, help="Number of seeds per dataset.")
    p.add_argument(
        "--datasets",
        type=str,
        default="Wine Quality",
        help="Comma-separated subset of: Wine Quality, Parkinsons Telemonitoring, Appliances Energy Prediction",
    )
    p.add_argument("--num_steps", type=int, default=300, help="train_vi steps for LMJGP.")
    p.add_argument("--MC_num", type=int, default=3, help="MC samples used by predict_vi.")
    p.add_argument("--Q", type=int, default=None, help="Override latent dimension Q.")
    p.add_argument(
        "--max_anchors",
        type=int,
        default=None,
        help="Use only the first K test anchors (faster: predict_vi + per-anchor JumpGP scale linearly).",
    )
    p.add_argument(
        "--proj_seed_offset",
        type=int,
        default=0,
        help="Offset added to projection RNG seed (for paired-randomness ablations).",
    )
    p.add_argument("--out", type=str, default="", help="Pickle output path.")
    p.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional JSON path for diagnostics + metrics (NaN/Inf sanitised to null).",
    )
    return p.parse_args()


def _print_summary_row(dataset: str, seed: int, methods: dict[str, list[float]]) -> None:
    keys = ("rmse", "crps", "wall", "cov90", "w90", "cov95", "w95")
    print(f"\n[{dataset}] seed={seed}")
    print("  method                rmse    crps    wall    cov90   w90    cov95   w95")
    for name, row in methods.items():
        vals = [f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else "  nan " for v in row]
        formatted = "  ".join(vals)
        print(f"  {name:<22}{formatted}")


def main():
    args = parse_args()
    dataset_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    total_res: dict = {}
    for dataset_name in dataset_list:
        total_res[dataset_name] = {}
        for seed in range(args.num_exp):
            res = _per_dataset_run(
                dataset_name,
                seed,
                num_steps=args.num_steps,
                MC_num=args.MC_num,
                Q_override=args.Q,
                max_anchors=args.max_anchors,
                proj_seed_offset=args.proj_seed_offset,
                device=device,
            )
            total_res[dataset_name][seed] = res
            _print_summary_row(dataset_name, seed, res["metrics"])
            print("  diagnostics.metric_A:", json.dumps(sanitize_for_json(res["diagnostics"]["metric_A"]), indent=None))
            for mode in PROJECTION_MODES:
                print(f"  diagnostics.{mode:<9}:", json.dumps(sanitize_for_json(res["diagnostics"][mode]), indent=None))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"uci_metric_variants_{stamp}.pkl"
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

"""Try CEM-EM MAP projections as W0 for transductive low-rank residual LMJGP.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_lmjgp_lowrank_cem_w0_paper.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --seed 0 --out_dir experiments/synthetic/lmjgp_lowrank_cem_w0_paper_train1000_seed0

For each paper-synthetic setting, this runner:

1. trains transductive CEM-EM MAP projections ``W_j^CEM`` on test anchors
   using training-neighbor labels only;
2. evaluates CEM-MAP directly with projected JumpGP-CEM;
3. trains transductive low-rank residual LMJGP with fixed
   ``W0 = mean_j W_j^CEM`` and either PLS/PCA ``V`` or CEM-SVD ``V``.

This tests whether a non-trivial CEM-derived projection mean can prevent the
mean-zero full LMJGP posterior from collapsing to prior-like random projections.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    compute_pls_W0,
    run_projected_jumpgp,
    train_projection_cem_em,
)
from djgp.variational_lowrank import predict_vi_lowrank_residual, train_vi_lowrank_residual  # noqa: E402
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import _metrics_pack  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from experiments.uci.compare_lowrank_residual_lmjgp import _init_lowrank_state  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting",
        "family",
        "seed",
        "method",
        "N_train",
        "N_test",
        "latent_dim",
        "observed_dim",
        "Q",
        "n",
        *RESULT_ROW_KEYS,
        "cem_train_sec",
        "lmjgp_train_sec",
        "pred_sec",
        "best_elbo",
        "W0_source",
        "V_source",
    ]
    keys = [k for k in preferred if any(k in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _build_neighbor_lists(
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    n: int,
    Q: int,
    m1: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], list[torch.Tensor], list[torch.Tensor]]:
    nbr = find_neighborhoods(X_anchor.cpu(), X_pool.cpu(), y_pool.cpu(), M=int(n))
    regions = []
    X_nb_list = []
    y_nb_list = []
    for item in nbr:
        X_nb = item["X_neighbors"].to(device)
        y_nb = item["y_neighbors"].to(device)
        X_nb_list.append(X_nb)
        y_nb_list.append(y_nb)
        regions.append({"X": X_nb, "y": y_nb, "C": torch.randn(m1, Q, device=device)})
    return regions, X_nb_list, y_nb_list


def _cem_svd_basis(W: torch.Tensor, R: int, fallback: torch.Tensor) -> torch.Tensor:
    D = W.shape[-1]
    rows = W.detach().reshape(-1, D)
    rows = rows - rows.mean(dim=0, keepdim=True)
    try:
        _u, _s, vh = torch.linalg.svd(rows, full_matrices=False)
        V = vh[: min(R, vh.shape[0])].T
    except RuntimeError:
        V = fallback[:, : min(R, fallback.shape[1])]
    if V.shape[1] < R:
        V = torch.cat([V, fallback[:, : R - V.shape[1]]], dim=1)
    V, _ = torch.linalg.qr(V[:, :R], mode="reduced")
    return V


def _pack_row(
    *,
    base: dict[str, Any],
    method: str,
    y_test: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    wall_sec: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    pack = _metrics_pack(y_test, mu, sigma, wall_sec)
    row = dict(base)
    row["method"] = method
    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
    row.update(extra)
    return row


def _run_lowrank_variant(
    *,
    method: str,
    W0_t: torch.Tensor,
    V_t: torch.Tensor,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    regions: list[dict[str, torch.Tensor]],
    y_test: np.ndarray,
    Q: int,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    C_params, u_params, hyperparams = _init_lowrank_state(
        X_train_t=X_train_t,
        X_test_t=X_test_t,
        Q=Q,
        R=int(V_t.shape[1]),
        m1=int(args.m1),
        m2=int(args.m2),
        W0_t=W0_t,
        V_t=V_t,
        residual_scale=float(args.residual_scale),
        device=device,
    )
    C_params, _u, hyperparams, diag = train_vi_lowrank_residual(
        regions,
        C_params,
        u_params,
        hyperparams,
        lr=float(args.lr),
        num_steps=int(args.lmjgp_steps),
        log_interval=max(1, int(args.lmjgp_steps)),
    )
    mu, var, pred_time = predict_vi_lowrank_residual(
        regions,
        C_params,
        hyperparams,
        M=int(args.MC_num),
        seed=int(seed) + 101,
    )
    sigma = torch.sqrt(var.clamp_min(1e-12)).detach().cpu().numpy()
    info = {
        "lmjgp_train_sec": float(diag["train_time_sec"]),
        "pred_sec": float(pred_time),
        "best_elbo": float(diag["best_elbo"]),
        "R": int(V_t.shape[1]),
        "residual_scale": float(args.residual_scale),
    }
    return mu.detach().cpu().numpy(), sigma, info


def run_setting(args: argparse.Namespace, setting: str, device: torch.device) -> list[dict[str, Any]]:
    cfg = SETTING_PRESETS[setting]
    _set_seed(int(args.seed))
    data = _generate_paper_data(cfg, int(args.seed), device)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data["X_train"])
    X_test_s = scaler.transform(data["X_test"])
    y_train = data["y_train"]
    y_test = data["y_test"]

    X_train_t = torch.from_numpy(X_train_s).float().to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).float().to(device)
    X_test_t = torch.from_numpy(X_test_s).float().to(device)
    Q = int(cfg["Q"])
    n = int(cfg["n"])
    regions, X_nb_list, y_nb_list = _build_neighbor_lists(
        X_test_t,
        X_train_t,
        y_train_t,
        n=n,
        Q=Q,
        m1=int(args.m1),
        device=device,
    )

    V_np, _ = build_V_basis(
        X_train_s,
        y_train,
        n_pls=int(args.n_pls),
        n_pca=int(args.n_pca),
        n_random=int(args.n_random),
        seed=int(args.seed),
    )
    W0_np = compute_pls_W0(X_train_s, y_train, Q=Q)
    V_pls_pca = torch.from_numpy(V_np).float().to(device)
    W0_pls = torch.from_numpy(W0_np).float().to(device)

    cfg_cem = CemEmConfig(
        n_outer=int(args.cem_outer),
        n_m_steps=int(args.cem_m_steps),
        lr=float(args.lr),
        lambda_gate=float(args.lambda_gate),
        lambda_smooth=float(args.lambda_smooth),
        lambda_scale=0.0,
        lambda_anchor_pred=0.0,
        normalize_smoothness=True,
        track_best_W=True,
        snapshot_after_each_outer=False,
        cem_progress=False,
        verbose=False,
    )
    model = CoeffProjection(
        T=X_test_t.shape[0],
        Q=Q,
        D=X_test_t.shape[1],
        V=V_pls_pca,
        W0_init=W0_pls,
        learn_W0=True,
        init_C_scale=float(args.init_C_scale),
        coeff_scale=float(args.coeff_scale),
        row_normalize=True,
    ).to(device)
    print(f"[{setting}] train CEM MAP", flush=True)
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        X_anchors=X_test_t,
        y_anchors=None,
        device=device,
        cfg=cfg_cem,
        label=f"cem_map_{setting}",
    )
    rows: list[dict[str, Any]] = []
    base = {
        "setting": setting,
        "family": cfg["family"],
        "seed": int(args.seed),
        "N_train": int(cfg["N_train"]),
        "N_test": int(cfg["N_test"]),
        "latent_dim": int(cfg["d"]),
        "observed_dim": int(cfg["H"]),
        "Q": Q,
        "n": n,
        "standardize_x": True,
        "cem_train_sec": float(res.train_time_sec),
    }

    print(f"[{setting}] evaluate CEM MAP", flush=True)
    out = run_projected_jumpgp(
        res.W.detach(),
        X_test_t,
        X_nb_list,
        y_nb_list,
        device=device,
        progress=False,
        desc=f"{setting}/cem_map",
    )
    rows.append(
        _pack_row(
            base=base,
            method="cem_em_transductive_map",
            y_test=y_test,
            mu=out["mu"],
            sigma=out["sigma"],
            wall_sec=float(out["elapsed_sec"]),
            extra={
                "pred_sec": float(out["elapsed_sec"]),
                "W0_source": "cem_anchor_map",
                "V_source": "pls_pca",
                "best_outer": int(res.best_outer),
            },
        )
    )

    W_cem_mean = res.W.detach().mean(dim=0)
    variants = [
        ("lmjgp_lowrank_cem_mean_pls_pca", W_cem_mean, V_pls_pca, "cem_mean", "pls_pca"),
        (
            "lmjgp_lowrank_cem_mean_cem_svd",
            W_cem_mean,
            _cem_svd_basis(res.W.detach(), int(V_pls_pca.shape[1]), V_pls_pca),
            "cem_mean",
            "cem_svd",
        ),
    ]
    for method, W0_t, V_t, w_src, v_src in variants:
        print(f"[{setting}] train {method}", flush=True)
        mu, sigma, info = _run_lowrank_variant(
            method=method,
            W0_t=W0_t,
            V_t=V_t,
            X_train_t=X_train_t,
            y_train_t=y_train_t,
            X_test_t=X_test_t,
            regions=regions,
            y_test=y_test,
            Q=Q,
            args=args,
            seed=int(args.seed),
            device=device,
        )
        rows.append(
            _pack_row(
                base=base,
                method=method,
                y_test=y_test,
                mu=mu,
                sigma=sigma,
                wall_sec=float(info["lmjgp_train_sec"] + info["pred_sec"]),
                extra={**info, "W0_source": w_src, "V_source": v_src},
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CEM MAP W0 + low-rank residual LMJGP on paper synthetic data.")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--cem_outer", type=int, default=5)
    p.add_argument("--cem_m_steps", type=int, default=80)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lambda_gate", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--init_C_scale", type=float, default=0.0)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--out_dir", default="experiments/synthetic/lmjgp_lowrank_cem_w0_paper_train1000_seed0")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    unknown = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for setting in settings:
        rows.extend(run_setting(args, setting, device))
        _write_csv(out_dir / "metrics_partial.csv", rows)
    _write_csv(out_dir / "metrics.csv", rows)
    summary = {"args": vars(args), "device": str(device), "n_rows": len(rows)}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    readme = f"""# CEM MAP W0 + Low-Rank Residual LMJGP on Paper Synthetic Data

This experiment uses transductive CEM-EM MAP projections to build a fixed
global projection mean for low-rank residual LMJGP.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved {len(rows)} rows to {out_dir}", flush=True)
    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    main()

"""Compare LMJGP transductive prediction heads on paper synthetic data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_lmjgp_prediction_heads_paper.py ^
        --settings l2_q2_train500,lh_q5_train500 ^
        --seed 0 --out_dir experiments/synthetic/lmjgp_prediction_heads_paper_seed0

This runner trains one transductive LMJGP model per setting, then evaluates two
prediction heads on the same trained variational state:

1. ``sampleW_jumpgp_refit``: ``predict_vi`` samples ``W_*`` from ``q(W_*)`` and
   calls JumpGP-CEM for each sample/anchor before moment matching.
2. ``direct_variational_analytic``: ``predict_vi_analytic`` uses the learned
   local variational ``u_params`` directly, without JumpGP-CEM refit.

The data are generated through the paper-style L2/LH generator wrapper in
``compare_paper_synthetic_baselines.py`` and X is standardized using train-only
statistics after the RFF expansion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.variational import (  # noqa: E402
    expected_Kfu,
    expected_KufKfu,
    expected_sigmoid_gh,
    qW_from_qV,
    predict_vi,
    train_vi,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    preferred = [
        "setting",
        "family",
        "seed",
        "head",
        "N_train",
        "N_test",
        "latent_dim",
        "observed_dim",
        "Q",
        "n",
        *RESULT_ROW_KEYS,
        "bad_var_count",
        "nan_mu_count",
        "nan_sigma_count",
        "train_wall_sec",
        "pred_wall_sec",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _metrics_pack(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> list[float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, wall_sec, y, mu, sigma)


def _stack_psd_sigma_u(u_params: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    raw = torch.stack([u["Sigma_u"] for u in u_params], dim=0)
    L = torch.tril(raw)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    pos_diag = F.softplus(diag) + 1e-6
    L = L - torch.diag_embed(diag) + torch.diag_embed(pos_diag)
    return L @ L.transpose(-2, -1)


def _predict_direct_variational_consistent(
    regions: list[dict[str, torch.Tensor]],
    V_params: dict[str, torch.Tensor],
    u_params: list[dict[str, torch.Tensor]],
    hyperparams: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct VI predictive moments matching the training ELBO conventions.

    This is intentionally separate from the legacy ``predict_vi_analytic``
    helper, whose current implementation has stale shape/scale conventions.
    """
    device = hyperparams["Z"].device
    T = len(regions)
    C = torch.stack([r["C"] for r in regions], dim=0)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)
    Sigma_u = _stack_psd_sigma_u(u_params)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(T).clamp_min(1e-6)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(T).clamp_min(1e-6)
    omega = torch.stack([u["omega"] for u in u_params], dim=0)
    U = torch.sigmoid(torch.stack([u["U_logit"] for u in u_params], dim=0)).view(T)

    X_test = hyperparams["X_test"]
    mu_W, cov_W = qW_from_qV(
        X_test,
        hyperparams["Z"],
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
    )
    Xn = X_test.unsqueeze(1)
    m_W = expected_Kfu(mu_W, cov_W, Xn, C, sigma_k).squeeze(1)
    S_W = expected_KufKfu(mu_W, cov_W, Xn, C, sigma_k).squeeze(1)

    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    eye = torch.eye(m1, device=device, dtype=C.dtype).unsqueeze(0)
    Kuu = torch.exp(-0.5 * d2) + 1e-6 * eye
    Luu = torch.linalg.cholesky(0.5 * (Kuu + Kuu.transpose(-1, -2)))
    Kinv = torch.cholesky_inverse(Luu)

    a = torch.einsum("tij,tj->ti", Kinv, mu_u)
    mu_f = (m_W * a).sum(dim=1)
    V1 = sigma_k.pow(2) - torch.einsum("tij,tij->t", Kinv, S_W)
    T3 = torch.einsum("tij,tjk,tkm,tmi->t", Kinv, S_W, Kinv, Sigma_u)
    centered_S = S_W - m_W.unsqueeze(2) * m_W.unsqueeze(1)
    V_W = torch.einsum("ti,tij,tj->t", a, centered_S, a)
    var_f = (V1 + T3 + V_W).clamp_min(1e-12)

    pi = expected_sigmoid_gh(omega, mu_W, cov_W, Xn).squeeze(1).clamp(0.0, 1.0)
    inlier_second = mu_f.pow(2) + var_f + sigma_noise.pow(2)
    mu_y = pi * mu_f + (1.0 - pi) * U
    Ey2 = pi * inlier_second + (1.0 - pi) * U.pow(2)
    var_y = (Ey2 - mu_y.pow(2)).clamp_min(1e-12)
    return mu_y, var_y


def _train_lmjgp_state(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    Q: int,
    n: int,
    m1: int,
    m2: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor], list[dict[str, torch.Tensor]], dict[str, torch.Tensor], float]:
    t0 = time.perf_counter()
    neighborhoods = find_neighborhoods(X_test.cpu(), X_train.cpu(), y_train.cpu(), M=int(n))
    regions: list[dict[str, torch.Tensor]] = []
    for item in neighborhoods:
        regions.append(
            {
                "X": item["X_neighbors"].to(device),
                "y": item["y_neighbors"].to(device),
                "C": torch.randn(m1, Q, device=device),
            }
        )

    D = X_train.shape[1]
    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
    }
    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0).clamp_min(1e-6)
    hyperparams = {
        "Z": X_mean + torch.randn(m2, D, device=device) * X_std,
        "X_test": X_test,
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    u_params: list[dict[str, torch.Tensor]] = []
    for _ in range(X_test.shape[0]):
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

    V_params, u_params, hyperparams = train_vi(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=float(lr),
        num_steps=int(steps),
        log_interval=max(50, int(steps)),
    )
    return regions, V_params, u_params, hyperparams, float(time.perf_counter() - t0)


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
    regions, V_params, u_params, hyperparams, train_sec = _train_lmjgp_state(
        X_train_t,
        y_train_t,
        X_test_t,
        Q=Q,
        n=n,
        m1=int(args.m1),
        m2=int(args.m2),
        steps=int(args.lmjgp_steps),
        lr=float(args.lr),
        device=device,
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
        "train_wall_sec": train_sec,
    }

    t0 = time.perf_counter()
    mu_jgp, var_jgp = predict_vi(regions, V_params, hyperparams, M=int(args.MC_num))
    pred_sec = float(time.perf_counter() - t0)
    sigma_jgp = torch.sqrt(torch.clamp(var_jgp, min=1e-12))
    pack = _metrics_pack(
        y_test,
        mu_jgp.detach().cpu().numpy(),
        sigma_jgp.detach().cpu().numpy(),
        pred_sec,
    )
    row = dict(base)
    row.update({"head": "sampleW_jumpgp_refit", "pred_wall_sec": pred_sec})
    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
    row.update(
        {
            "bad_var_count": int((var_jgp.detach().cpu() <= 0).sum().item()),
            "nan_mu_count": int(torch.isnan(mu_jgp).sum().item()),
            "nan_sigma_count": int(torch.isnan(sigma_jgp).sum().item()),
        }
    )
    rows.append(row)

    t0 = time.perf_counter()
    mu_vi, var_vi = _predict_direct_variational_consistent(regions, V_params, u_params, hyperparams)
    pred_sec = float(time.perf_counter() - t0)
    var_raw = var_vi.detach().cpu()
    bad_var = int((~torch.isfinite(var_raw) | (var_raw <= 0)).sum().item())
    mu_vi = torch.nan_to_num(mu_vi, nan=0.0, posinf=0.0, neginf=0.0)
    var_vi = torch.nan_to_num(var_vi, nan=1e-12, posinf=1e12, neginf=1e-12).clamp_min(1e-12)
    sigma_vi = torch.sqrt(var_vi)
    pack = _metrics_pack(
        y_test,
        mu_vi.detach().cpu().numpy(),
        sigma_vi.detach().cpu().numpy(),
        pred_sec,
    )
    row = dict(base)
    row.update({"head": "direct_variational_consistent", "pred_wall_sec": pred_sec})
    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
    row.update(
        {
            "bad_var_count": bad_var,
            "nan_mu_count": int(torch.isnan(mu_vi).sum().item()),
            "nan_sigma_count": int(torch.isnan(sigma_vi).sum().item()),
        }
    )
    rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two LMJGP transductive prediction heads.")
    p.add_argument("--settings", default="l2_q2_train500,lh_q5_train500")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out_dir", default="experiments/synthetic/lmjgp_prediction_heads_paper_seed0")
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
        print(f"running setting={setting} seed={args.seed}", flush=True)
        rows.extend(run_setting(args, setting, device))
        _write_csv(out_dir / "metrics_partial.csv", rows)
    _write_csv(out_dir / "metrics.csv", rows)
    summary = {"args": vars(args), "device": str(device), "n_rows": len(rows)}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    readme = f"""# LMJGP Prediction Head Comparison on Paper Synthetic Data

This compares the same trained transductive LMJGP state with two prediction
heads:

- `sampleW_jumpgp_refit`: `predict_vi`, samples `q(W_*)` and refits JumpGP-CEM.
- `direct_variational_analytic`: `predict_vi_analytic`, uses learned local
  `u_params` directly without JumpGP-CEM refit.

Settings:

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved {len(rows)} rows to {out_dir}", flush=True)
    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    main()

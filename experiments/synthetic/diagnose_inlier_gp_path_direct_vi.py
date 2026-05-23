"""Diagnose inlier GP path vs gate collapse for Original ELBO + direct VI.

Priority fixes under test (see module doc in conversation):

- **``--standardize_y``** (train-only): align ``y`` scale with ``sigma_noise,sigma_k ~ O(1)``.
- **``--local_center_y``**: train residual GP on ``y - mean(N_j)``; predict ``mean(N_j) + mu_f``.

Modes:

1. **mixture_ab** — full mixture + A/B (reference).
2. **pure_inlier** — ``log p_in`` only; eval ``pi=1``.
3. **fixedW_inlier** — point ``W=mu_W``, pure inlier.
4. **fixedW_predict** — eval-only point kernel from mixture_ab weights.
5. **oracle_local_mean** — eval-only ``y_hat = mean(neighbors)`` (no GP).

Success on scale: ``std(mu_f) / y_std`` should rise from ~0.002 toward 0.1–0.5+ after fixes.

```powershell
# Explicit m_j in Q (empirical init, learnable)
python experiments/synthetic/diagnose_inlier_gp_path_direct_vi.py ^
    --standardize_y --modes pure_inlier,pure_inlier_mj,fixedW_inlier_mj,mixture_mj ^
    --max_test 200 --num_steps 300

# Fixed empirical m_j only
python experiments/synthetic/diagnose_inlier_gp_path_direct_vi.py ^
    --standardize_y --modes pure_inlier_mj --no_learn_mj --mj_init median

# Step 2 (LH region jumps)
python experiments/synthetic/diagnose_inlier_gp_path_direct_vi.py ^
    --local_center_y --modes pure_inlier,fixedW_inlier,oracle_local_mean --max_test 200
```
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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

from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.variational import (  # noqa: E402
    expected_Kfu,
    expected_KufKfu,
    kl_qp,
    kl_q_u_batch,
    predict_vi,
    qW_from_qV,
    train_vi,
)
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import (  # noqa: E402
    _predict_direct_variational_consistent,
    _stack_psd_sigma_u,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from experiments.synthetic.local_gp_elbo_mj import (  # noqa: E402
    compute_elbo_mixture_mj,
    compute_elbo_pure_inlier_mj,
    init_m_j_from_regions,
    predict_inlier_mj,
    stack_m_j,
)
from experiments.synthetic.test_ab_remedies_direct_vi import (  # noqa: E402
    _clone_regions,
    _deep_clone_state,
    _predict_direct_vi_forced_inlier,
    compute_elbo_augmented,
    compute_s_penalty,
    sigma_v_max_schedule,
)
from shared.jumpgp_runner import find_neighborhoods  # noqa: E402

_LOG_2PI = math.log(2 * math.pi)


def _build_state_diagnose(
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    *,
    Q: int,
    n: int,
    m1: int,
    m2: int,
    device: torch.device,
    seed: int,
    local_center_y: bool = False,
    explicit_mj: bool = False,
    mj_init: str = "mean",
    learn_mj: bool = True,
) -> tuple[list, dict, list, dict]:
    """Build LMJGP state; optional per-region ``y`` centering or explicit learnable ``m_j``."""
    if explicit_mj and local_center_y:
        print(
            "NOTE: explicit_mj set — training uses full y with m_j in Q; "
            "ignoring --local_center_y on response.",
            flush=True,
        )
        local_center_y = False
    _set_seed(seed)
    neighborhoods = find_neighborhoods(
        X_test_t.cpu(), X_train_t.cpu(), y_train_t.cpu(), M=n,
    )
    T = X_test_t.shape[0]
    D = X_train_t.shape[1]

    local_means: list[torch.Tensor] = []
    regions = []
    for item in neighborhoods:
        y_nb = item["y_neighbors"].to(device).clone()
        m_loc = y_nb.mean()
        local_means.append(m_loc)
        if local_center_y:
            y_nb = y_nb - m_loc
        regions.append({
            "X": item["X_neighbors"].to(device),
            "y": y_nb,
            "C": torch.randn(m1, Q, device=device),
        })

    X_mean = X_train_t.mean(dim=0)
    X_std = X_train_t.std(dim=0).clamp_min(1e-6)
    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
    }
    hyperparams: dict[str, Any] = {
        "Z": (X_mean + torch.randn(m2, D, device=device) * X_std).requires_grad_(True),
        "X_test": X_test_t,
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
        "y_local_mean": torch.stack(local_means),
        "local_center_y": local_center_y,
    }
    u_params = []
    m_j_list = init_m_j_from_regions(
        regions, device=device, init=mj_init, learn=learn_mj,
    ) if explicit_mj else [None] * T
    for t in range(T):
        entry: dict[str, Any] = {
            "U_logit": torch.zeros(1, device=device, requires_grad=True),
            "mu_u": torch.randn(m1, device=device, requires_grad=True),
            "Sigma_u": torch.eye(m1, device=device, requires_grad=True),
            "sigma_noise": torch.tensor(0.5, device=device, requires_grad=True),
            "sigma_k": torch.tensor(0.5, device=device, requires_grad=True),
            "omega": torch.randn(Q + 1, device=device, requires_grad=True),
        }
        if explicit_mj:
            entry["m_j"] = m_j_list[t]
        u_params.append(entry)

    hyperparams["explicit_mj"] = explicit_mj
    hyperparams["learn_mj"] = learn_mj
    return regions, V_params, u_params, hyperparams


def _gp_mean_offset(hyperparams: dict[str, Any]) -> torch.Tensor:
    return hyperparams["y_local_mean"]


def _add_gp_offset(mu_resid: torch.Tensor, hyperparams: dict[str, Any]) -> torch.Tensor:
    return mu_resid + _gp_mean_offset(hyperparams)


def _inducing_distance_stats(
    regions,
    V_params,
    hyperparams,
) -> dict[str, float]:
    """Mean/min distance from projected neighbours ``mu_W x`` to inducing ``C``."""
    device = hyperparams["Z"].device
    X = torch.stack([r["X"] for r in regions], dim=0)
    C = torch.stack([r["C"] for r in regions], dim=0)
    mu_W, _ = qW_from_qV(
        hyperparams["X_test"],
        hyperparams["Z"],
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
    )
    # mu_proj[t,n,q] = mu_W[t,q] · x_{t,n}
    proj = torch.einsum("tqd,tnd->tnq", mu_W, X)
    # C[t,m1,q] -> distance per (t,n,m1)
    diff = proj.unsqueeze(2) - C.unsqueeze(1)
    dist = diff.pow(2).sum(-1).sqrt()
    return {
        "inducing_dist_mean": float(dist.mean().item()),
        "inducing_dist_min_mean": float(dist.min(dim=2).values.mean().item()),
    }


# ---------------------------------------------------------------------------
# Pure inlier ELBO (no mixture / gate / outlier)
# ---------------------------------------------------------------------------


def compute_elbo_pure_inlier(
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    fixed_w: bool = False,
) -> torch.Tensor:
    """ELBO = sum log N(y | GP mean, sigma_noise) - KL_V - KL_u (no pi/U terms)."""
    T = len(regions)
    device = regions[0]["X"].device

    X = torch.stack([r["X"] for r in regions], dim=0)
    y = torch.stack([r["y"] for r in regions], dim=0)
    C = torch.stack([r["C"] for r in regions], dim=0)

    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(-1)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)

    eps = 1e-6
    raw = torch.stack([u["Sigma_u"] for u in u_params], dim=0)
    L = torch.tril(raw)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    pos_diag = F.softplus(diag) + eps
    L = L - torch.diag_embed(diag) + torch.diag_embed(pos_diag)
    Sigma_u = L @ L.transpose(-1, -2)

    Z = hyperparams["Z"]
    X_test = hyperparams["X_test"]
    lengthscales = hyperparams["lengthscales"]
    var_w = hyperparams["var_w"]

    KL_V = kl_qp(Z, V_params["mu_V"], V_params["sigma_V"], lengthscales, var_w)
    mu_W, cov_W = qW_from_qV(
        X_test, Z, V_params["mu_V"], V_params["sigma_V"], lengthscales, var_w,
    )
    if fixed_w:
        cov_W = torch.zeros_like(cov_W)

    Kfu = expected_Kfu(mu_W, cov_W, X, C, sigma_k)
    KufKfu = expected_KufKfu(mu_W, cov_W, X, C, sigma_k)
    KL_u = kl_q_u_batch(C, mu_u, Sigma_u, torch.tensor(1.0, device=device)).sum()

    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    Kuu = torch.exp(-0.5 * d2)
    Kuu_inv = torch.cholesky_inverse(torch.linalg.cholesky(Kuu))

    V1 = sigma_k.view(-1, 1) ** 2 - torch.einsum("tij,tnji->tn", Kuu_inv, KufKfu)
    v = torch.einsum("tij,tj->ti", Kuu_inv, mu_u)
    E_fu = torch.einsum("tni,ti->tn", Kfu, v)

    Sigma_u_tilde = Sigma_u + torch.einsum("ti,tj->tij", mu_u, mu_u)
    T3 = torch.einsum("tij,tnjk,tkm,tmi->tn", Kuu_inv, KufKfu, Kuu_inv, Sigma_u_tilde)
    quad = (y.pow(2) - 2 * y * E_fu + (V1 + T3)) / (2 * sigma_noise.view(-1, 1) ** 2)

    loglik = (
        -0.5 * _LOG_2PI
        - 0.5 * torch.log(sigma_noise.view(-1, 1) ** 2)
        - quad
    ).sum()

    return (loglik - KL_V - KL_u) / T


@torch.no_grad()
def predict_fixed_w_inlier(
    regions,
    V_params,
    u_params,
    hyperparams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct VI predict with cov_W=0 (point W=mu_W), pi=1."""
    device = hyperparams["Z"].device
    T = len(regions)
    C = torch.stack([r["C"] for r in regions], dim=0)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)
    Sigma_u = _stack_psd_sigma_u(u_params)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(T).clamp_min(1e-6)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(T).clamp_min(1e-6)

    mu_W, cov_W = qW_from_qV(
        hyperparams["X_test"],
        hyperparams["Z"],
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
    )
    cov_W = torch.zeros_like(cov_W)

    Xn = hyperparams["X_test"].unsqueeze(1)
    m_W = expected_Kfu(mu_W, cov_W, Xn, C, sigma_k).squeeze(1)
    S_W = expected_KufKfu(mu_W, cov_W, Xn, C, sigma_k).squeeze(1)

    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    eye = torch.eye(m1, device=device, dtype=C.dtype).unsqueeze(0)
    Kuu = torch.exp(-0.5 * d2) + 1e-6 * eye
    Kinv = torch.cholesky_inverse(
        torch.linalg.cholesky(0.5 * (Kuu + Kuu.transpose(-1, -2)))
    )

    a = torch.einsum("tij,tj->ti", Kinv, mu_u)
    mu_f = (m_W * a).sum(dim=1)
    V1 = sigma_k.pow(2) - torch.einsum("tij,tij->t", Kinv, S_W)
    T3 = torch.einsum("tij,tjk,tkm,tmi->t", Kinv, S_W, Kinv, Sigma_u)
    centered_S = S_W - m_W.unsqueeze(2) * m_W.unsqueeze(1)
    V_W = torch.einsum("ti,tij,tj->t", a, centered_S, a)
    var_f = (V1 + T3 + V_W).clamp_min(1e-12)
    var_y = (var_f + sigma_noise.pow(2)).clamp_min(1e-12)
    return mu_f, var_y, mu_f


@torch.no_grad()
def gp_path_diagnostics(
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    fixed_w: bool = False,
) -> dict[str, float]:
    """Intermediate GP quantities at test anchors and training neighbours."""
    device = hyperparams["Z"].device
    T = len(regions)

    X = torch.stack([r["X"] for r in regions], dim=0)
    C = torch.stack([r["C"] for r in regions], dim=0)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params], dim=0).view(-1)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(-1)

    mu_W, cov_W = qW_from_qV(
        hyperparams["X_test"],
        hyperparams["Z"],
        V_params["mu_V"],
        V_params["sigma_V"],
        hyperparams["lengthscales"],
        hyperparams["var_w"],
    )
    if fixed_w:
        cov_W = torch.zeros_like(cov_W)

    X_test = hyperparams["X_test"]
    Xn = X_test.unsqueeze(1)

    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    eye = torch.eye(m1, device=device, dtype=C.dtype).unsqueeze(0)
    Kuu = torch.exp(-0.5 * d2) + 1e-6 * eye
    Kinv = torch.cholesky_inverse(
        torch.linalg.cholesky(0.5 * (Kuu + Kuu.transpose(-1, -2)))
    )
    a = torch.einsum("tij,tj->ti", Kinv, mu_u)

    m_W_test = expected_Kfu(mu_W, cov_W, Xn, C, sigma_k).squeeze(1)
    Kfu_train = expected_Kfu(mu_W, cov_W, X, C, sigma_k)
    E_fu_train = torch.einsum("tni,ti->tn", Kfu_train, a)

    mu_f_test = (m_W_test * a).sum(dim=1)
    if hyperparams.get("explicit_mj"):
        m_j = stack_m_j(u_params)
        mu_y_test = m_j + mu_f_test
        loc_mean = m_j
    else:
        mu_y_test = _add_gp_offset(mu_f_test, hyperparams)
        loc_mean = _gp_mean_offset(hyperparams)
    sigma_W = torch.sqrt(cov_W.diagonal(dim1=2, dim2=3))

    out = {
        "std_m_W": float(m_W_test.std().item()),
        "std_E_Kfu_train": float(E_fu_train.std().item()),
        "mean_a_norm": float(a.norm(dim=1).mean().item()),
        "max_a_norm": float(a.norm(dim=1).max().item()),
        "std_mu_f": float(mu_f_test.std().item()),
        "mean_mu_f": float(mu_f_test.mean().item()),
        "std_mu_y": float(mu_y_test.std().item()),
        "mean_mu_y": float(mu_y_test.mean().item()),
        "std_local_mean": float(loc_mean.std().item()),
        "mean_local_mean": float(loc_mean.mean().item()),
        "std_m_j": float(loc_mean.std().item()) if hyperparams.get("explicit_mj") else 0.0,
        "mean_m_j": float(loc_mean.mean().item()) if hyperparams.get("explicit_mj") else 0.0,
        "sigma_k_mean": float(sigma_k.mean().item()),
        "sigma_k_max": float(sigma_k.max().item()),
        "sigma_noise_mean": float(sigma_noise.mean().item()),
        "sigma_noise_max": float(sigma_noise.max().item()),
        "lengthscale_mean": float(hyperparams["lengthscales"].mean().item()),
        "lengthscale_max": float(hyperparams["lengthscales"].max().item()),
        "mu_u_norm_mean": float(mu_u.norm(dim=1).mean().item()),
        "sigma_w_mean": float(sigma_W.mean().item()),
        "sigma_v_mean": float(V_params["sigma_V"].mean().item()),
        "std_mu_W": float(mu_W.std().item()),
    }
    out.update(_inducing_distance_stats(regions, V_params, hyperparams))
    return out


def _deep_clone_state_diag(V_params, u_params, hyperparams):
    V2, u2, h2 = _deep_clone_state(V_params, u_params, hyperparams)
    h2["y_local_mean"] = hyperparams["y_local_mean"].detach().clone()
    h2["local_center_y"] = hyperparams["local_center_y"]
    h2["explicit_mj"] = hyperparams.get("explicit_mj", False)
    h2["learn_mj"] = hyperparams.get("learn_mj", False)
    if hyperparams.get("explicit_mj"):
        for u, u2i in zip(u_params, u2):
            u2i["m_j"] = u["m_j"].detach().clone().requires_grad_(hyperparams.get("learn_mj", True))
    return V2, u2, h2


def _mode_uses_mj(mode: str) -> bool:
    return mode.endswith("_mj") or "inlier_mj" in mode


def _mode_fixed_w(mode: str) -> bool:
    return "fixedW" in mode


@torch.no_grad()
def _evaluate_oracle_local_mean(
    hyperparams: dict[str, Any],
    y_test_eval: np.ndarray,
    y_test_working: np.ndarray,
    *,
    step: int,
    y_std_working: float,
    scaler_y: StandardScaler | None,
) -> dict[str, Any]:
    """Brutal baseline: predict neighborhood mean only (no GP residual)."""
    device = hyperparams["Z"].device
    T = hyperparams["y_local_mean"].shape[0]
    mu_y = _gp_mean_offset(hyperparams)
    y_work = torch.from_numpy(y_test_working.astype(np.float32)).to(device)[:T]
    rmse = float(torch.sqrt(((mu_y - y_work) ** 2).mean()).item())
    if scaler_y is not None:
        mu_raw = scaler_y.inverse_transform(
            mu_y.detach().cpu().numpy().reshape(-1, 1),
        ).ravel()
        y_raw = y_test_eval[:T]
        rmse_raw = float(np.sqrt(np.mean((mu_raw - y_raw) ** 2)))
    else:
        rmse_raw = rmse
    std_lm = float(mu_y.std().item())
    rec = {
        "mode": "oracle_local_mean",
        "step": step,
        "rmse": rmse,
        "rmse_raw": rmse_raw,
        "crps": float("nan"),
        "y_test_std": float(np.std(y_test_eval)),
        "y_working_std": y_std_working,
        "std_mu_f": 0.0,
        "std_mu_y": std_lm,
        "mean_local_mean": float(mu_y.mean().item()),
        "mu_f_to_y_std_ratio": 0.0,
        "mu_y_to_y_std_ratio": float(std_lm / max(y_std_working, 1e-8)),
    }
    print(
        f"    >>> [oracle_local_mean] RMSE={rmse:.4f} raw={rmse_raw:.2f}  "
        f"std(local_mean)={std_lm:.4f}",
        flush=True,
    )
    return rec


def _train_mode(
    mode: str,
    regions,
    V_params,
    u_params,
    hyperparams,
    *,
    lr: float,
    num_steps: int,
    eval_every: int,
    y_test: np.ndarray,
    lambda_s: float,
    anneal_sigma_v: bool,
    sigma_v_start: float,
    sigma_v_end: float,
    warmup_frac: float,
    freeze_sigma_v: bool = False,
) -> list[dict[str, Any]]:
    use_mj = _mode_uses_mj(mode)
    pure = use_mj or mode in ("pure_inlier", "pure_inlier_ab", "fixedW_inlier")
    fixed_w = _mode_fixed_w(mode)

    params = [V_params["mu_V"], V_params["sigma_V"]]
    for u in u_params:
        params += [u["mu_u"], u["Sigma_u"], u["sigma_noise"], u["sigma_k"]]
        if use_mj and hyperparams.get("learn_mj", True):
            params.append(u["m_j"])
        if not pure:
            params += [u["U_logit"], u["omega"]]
    params += [hyperparams["lengthscales"], hyperparams["var_w"], hyperparams["Z"]]
    optimizer = torch.optim.Adam(params, lr=lr)

    records: list[dict[str, Any]] = []
    y_std = float(np.std(y_test))  # working-scale std for ratio denominators

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        if use_mj and mode.startswith("mixture"):
            elbo = compute_elbo_mixture_mj(
                regions, V_params, u_params, hyperparams, fixed_w=fixed_w,
            )
        elif use_mj:
            elbo = compute_elbo_pure_inlier_mj(
                regions, V_params, u_params, hyperparams, fixed_w=fixed_w,
            )
        elif pure:
            elbo = compute_elbo_pure_inlier(
                regions, V_params, u_params, hyperparams, fixed_w=fixed_w,
            )
        else:
            elbo = compute_elbo_augmented(regions, V_params, u_params, hyperparams)

        s_pen = 0.0
        if lambda_s > 0:
            mu_W, cov_W = qW_from_qV(
                hyperparams["X_test"], hyperparams["Z"],
                V_params["mu_V"], V_params["sigma_V"],
                hyperparams["lengthscales"], hyperparams["var_w"],
            )
            X_stacked = torch.stack([r["X"] for r in regions], dim=0)
            s_pen = float((lambda_s * compute_s_penalty(mu_W, cov_W, X_stacked)).item())
            elbo = elbo - lambda_s * compute_s_penalty(mu_W, cov_W, X_stacked)

        (-elbo).backward()
        optimizer.step()

        with torch.no_grad():
            if freeze_sigma_v or fixed_w:
                V_params["sigma_V"].fill_(0.01)
            elif anneal_sigma_v:
                sv_max = sigma_v_max_schedule(
                    step, num_steps,
                    sigma_start=sigma_v_start, sigma_end=sigma_v_end,
                    warmup_frac=warmup_frac,
                )
                V_params["sigma_V"].clamp_(min=0.01, max=sv_max)
            else:
                V_params["sigma_V"].clamp_(min=0.1)

            hyperparams["var_w"].clamp_(min=0.1)
            hyperparams["lengthscales"].clamp_(min=1e-6, max=1e2)
            for u in u_params:
                u["sigma_noise"].clamp_(min=0.1)
                u["sigma_k"].clamp_(min=0.1)

        if step % max(1, eval_every // 3) == 0 or step == 1:
            print(f"  [{mode} {step}/{num_steps}] ELBO={elbo.item():.4f} s_pen={s_pen:.4f}", flush=True)

        if step % eval_every == 0 or step == num_steps:
            records.append(
                _evaluate_mode(
                    mode, regions, V_params, u_params, hyperparams,
                    hyperparams.get("y_test_working", y_test),
                    step=step,
                    y_std_working=y_std,
                    y_test_raw=hyperparams.get("y_test_raw"),
                    fixed_w=fixed_w,
                    scaler_y=hyperparams.get("scaler_y"),
                )
            )

    return records


@torch.no_grad()
def _evaluate_mode(
    mode: str,
    regions,
    V_params,
    u_params,
    hyperparams,
    y_test_working: np.ndarray,
    *,
    step: int,
    y_std_working: float,
    y_test_raw: np.ndarray | None = None,
    fixed_w: bool = False,
    scaler_y: StandardScaler | None = None,
) -> dict[str, Any]:
    device = hyperparams["Z"].device
    T = len(regions)
    y_t = torch.from_numpy(y_test_working.astype(np.float32)).to(device)[:T]
    y_raw_np = y_test_raw if y_test_raw is not None else y_test_working

    use_mj = _mode_uses_mj(mode)
    fixed_w = _mode_fixed_w(mode)

    if use_mj:
        mu_y, var_y, mu_f = predict_inlier_mj(
            regions, V_params, u_params, hyperparams, fixed_w=fixed_w,
        )
        mu_resid = mu_y
    elif mode == "fixedW_predict":
        mu_resid, var_y, mu_f = predict_fixed_w_inlier(
            regions, V_params, u_params, hyperparams,
        )
    elif mode in ("pure_inlier", "pure_inlier_ab", "fixedW_inlier"):
        if fixed_w:
            mu_resid, var_y, mu_f = predict_fixed_w_inlier(
                regions, V_params, u_params, hyperparams,
            )
        else:
            mu_resid, var_y, mu_f = _predict_direct_vi_forced_inlier(
                regions, V_params, u_params, hyperparams,
            )
    else:
        mu_mix, var_y = _predict_direct_variational_consistent(
            regions, V_params, u_params, hyperparams,
        )
        mu_resid, _, mu_f = _predict_direct_vi_forced_inlier(
            regions, V_params, u_params, hyperparams,
        )
        _ = mu_mix

    if use_mj:
        mu_y = torch.nan_to_num(mu_y, nan=0.0)
    else:
        mu_y = _add_gp_offset(torch.nan_to_num(mu_resid, nan=0.0), hyperparams)
    var_y = torch.nan_to_num(var_y, nan=1e-12).clamp_min(1e-12)
    rmse = float(torch.sqrt(((mu_y - y_t) ** 2).mean()).item())
    crps = float(mean_gaussian_crps_torch(y_t, mu_y, torch.sqrt(var_y)).item())

    rmse_working = rmse
    if scaler_y is not None:
        mu_np = mu_y.detach().cpu().numpy().reshape(-1, 1)
        mu_raw = scaler_y.inverse_transform(mu_np).ravel()
        y_raw = y_raw_np[:T]
        rmse_raw = float(np.sqrt(np.mean((mu_raw - y_raw) ** 2)))
    else:
        rmse_raw = rmse

    path = gp_path_diagnostics(
        regions, V_params, u_params, hyperparams, fixed_w=fixed_w,
    )
    ratio_f = float(path["std_mu_f"] / max(y_std_working, 1e-8))
    ratio_y = float(path["std_mu_y"] / max(y_std_working, 1e-8))

    rec: dict[str, Any] = {
        "mode": mode,
        "step": step,
        "rmse": rmse_working,
        "rmse_eval_scale": rmse,
        "rmse_raw": rmse_raw,
        "crps": crps,
        "y_test_std": float(np.std(y_raw_np)),
        "y_working_std": y_std_working,
        "mu_f_to_y_std_ratio": ratio_f,
        "mu_y_to_y_std_ratio": ratio_y,
        **path,
    }

    if mode == "mixture_ab":
        mu_pi1, _, _ = _predict_direct_vi_forced_inlier(
            regions, V_params, u_params, hyperparams,
        )
        mu_pi1 = _add_gp_offset(mu_pi1, hyperparams)
        rec["rmse_pi1"] = float(torch.sqrt(((mu_pi1 - y_t) ** 2).mean()).item())

    mj_s = ""
    if hyperparams.get("explicit_mj"):
        mj_s = f"  m_j={path.get('mean_m_j', 0):.3f}±{path.get('std_m_j', 0):.3f}"
    print(
        f"    >>> [{mode} step={step}] RMSE_work={rmse_working:.4f} RMSE_raw={rmse_raw:.2f}  "
        f"y_work_std={y_std_working:.4f}  std(zeta)/y_std={ratio_f:.4f}  "
        f"std(mu_y)/y_std={ratio_y:.4f}  std(m_W)={path['std_m_W']:.4f}  "
        f"|a|={path['mean_a_norm']:.3f}  sig_k={path['sigma_k_mean']:.3f}  "
        f"sig_n={path['sigma_noise_mean']:.3f}{mj_s}",
        flush=True,
    )
    return rec


def _run_jumpgp_baseline(
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    y_test: np.ndarray,
    *,
    Q: int,
    n: int,
    m1: int,
    m2: int,
    num_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """train_vi + predict_vi (sample W + JumpGP-CEM), same graph setup."""
    _set_seed(seed)
    neighborhoods = find_neighborhoods(
        X_test_t.cpu(), X_train_t.cpu(), y_train_t.cpu(), M=n,
    )
    regions = []
    for item in neighborhoods:
        regions.append({
            "X": item["X_neighbors"].to(device),
            "y": item["y_neighbors"].to(device),
            "C": torch.randn(m1, Q, device=device),
        })
    T = len(regions)
    D = X_train_t.shape[1]
    X_mean = X_train_t.mean(dim=0)
    X_std = X_train_t.std(dim=0).clamp_min(1e-6)

    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
    }
    hyperparams = {
        "Z": (X_mean + torch.randn(m2, D, device=device) * X_std).requires_grad_(True),
        "X_test": X_test_t,
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    u_params = []
    for _ in range(T):
        u_params.append({
            "U_logit": torch.zeros(1, device=device, requires_grad=True),
            "mu_u": torch.randn(m1, device=device, requires_grad=True),
            "Sigma_u": torch.eye(m1, device=device, requires_grad=True),
            "sigma_noise": torch.tensor(0.5, device=device, requires_grad=True),
            "sigma_k": torch.tensor(0.5, device=device, requires_grad=True),
            "omega": torch.randn(Q + 1, device=device, requires_grad=True),
        })

    train_vi(regions, V_params, u_params, hyperparams, lr=lr, num_steps=num_steps)
    mu_y, var_y = predict_vi(regions, V_params, u_params, hyperparams)

    y_t = torch.from_numpy(y_test.astype(np.float32)).to(device)[:T]
    mu_y = torch.nan_to_num(mu_y[:T], nan=0.0)
    var_y = torch.nan_to_num(var_y[:T], nan=1e-12).clamp_min(1e-12)
    rmse = float(torch.sqrt(((mu_y - y_t) ** 2).mean()).item())
    return {
        "mode": "jumpgp_sampleW",
        "step": num_steps,
        "rmse": rmse,
        "y_test_std": float(np.std(y_test)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Inlier GP path diagnostics for direct VI.")
    p.add_argument("--setting", default="lh_q5_train500")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--num_steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument(
        "--modes",
        default="mixture_ab,pure_inlier,fixedW_inlier,fixedW_predict",
        help="Comma-separated: mixture_ab, pure_inlier, pure_inlier_ab, fixedW_inlier, "
        "fixedW_predict (eval only, uses mixture_ab weights), jumpgp_sampleW",
    )
    p.add_argument("--lambda_s", type=float, default=0.1)
    p.add_argument("--no_ab", action="store_true")
    p.add_argument("--run_jumpgp", action="store_true", help="Also run train_vi+predict_vi.")
    p.add_argument(
        "--standardize_y",
        action="store_true",
        help="Train-only StandardScaler on y (matches stdxy LMJGP runs).",
    )
    p.add_argument(
        "--local_center_y",
        action="store_true",
        help="Per-region y <- y - mean(neighbors); predict mean(N_j) + GP residual.",
    )
    p.add_argument(
        "--mj_init",
        choices=("mean", "median", "zero"),
        default="mean",
        help="Initial m_j per region when using *_mj modes.",
    )
    p.add_argument(
        "--no_learn_mj",
        action="store_true",
        help="Keep m_j fixed at empirical init (no gradient on m_j).",
    )
    p.add_argument("--out_dir", default="experiments/synthetic/inlier_gp_path_diag")
    args = p.parse_args()

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
    scaler_y: StandardScaler | None = None
    if args.standardize_y:
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_work = scaler_y.transform(y_test_work.reshape(-1, 1)).ravel()

    print(
        f"y_raw_test std={float(np.std(y_test_raw)):.4f}  "
        f"y_working std={float(np.std(y_test_work)):.4f}  "
        f"standardize_y={args.standardize_y}  local_center_y={args.local_center_y}",
        flush=True,
    )

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)

    Q = int(cfg["Q"])
    n = int(cfg["n"])
    use_ab = not args.no_ab

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_records: list[dict[str, Any]] = []

    need_mj = any(_mode_uses_mj(m) for m in modes)
    regions_base, V_base, u_base, hyp_base = _build_state_diagnose(
        X_train_t, y_train_t, X_test_t,
        Q=Q, n=n, m1=args.m1, m2=args.m2, device=device, seed=args.seed,
        local_center_y=args.local_center_y,
        explicit_mj=need_mj,
        mj_init=args.mj_init,
        learn_mj=not args.no_learn_mj,
    )
    hyp_base["y_test_raw"] = y_test_raw
    hyp_base["y_test_working"] = y_test_work
    hyp_base["scaler_y"] = scaler_y
    y_std_working = float(np.std(y_test_work))

    mixture_state: tuple | None = None

    for mode in modes:
        if mode == "oracle_local_mean":
            print(f"\n--- oracle_local_mean (eval only) ---", flush=True)
            rec = _evaluate_oracle_local_mean(
                hyp_base, y_test_raw, y_test_work,
                step=0, y_std_working=y_std_working, scaler_y=scaler_y,
            )
            all_records.append(rec)
            continue

        if mode == "fixedW_predict":
            if mixture_state is None:
                print("SKIP fixedW_predict: run mixture_ab first.", flush=True)
                continue
            V, u, h, regions = mixture_state
            print(f"\n--- fixedW_predict (eval only, from mixture_ab) ---", flush=True)
            rec = _evaluate_mode(
                "fixedW_predict", regions, V, u, h, y_test_work,
                step=args.num_steps,
                y_std_working=y_std_working,
                y_test_raw=y_test_raw,
                fixed_w=True,
                scaler_y=scaler_y,
            )
            all_records.append(rec)
            continue

        print(f"\n{'='*70}\n  MODE: {mode}\n{'='*70}", flush=True)
        V, u, h = _deep_clone_state_diag(V_base, u_base, hyp_base)
        regions = _clone_regions(regions_base)

        ab_modes = ("mixture_ab", "pure_inlier_ab", "mixture_mj")
        lam_s = args.lambda_s if use_ab and mode in ab_modes else 0.0
        anneal = use_ab and mode in ab_modes
        freeze_sv = _mode_fixed_w(mode)

        h["y_test_raw"] = y_test_raw
        h["y_test_working"] = y_test_work
        h["scaler_y"] = scaler_y

        recs = _train_mode(
            mode, regions, V, u, h,
            lr=args.lr,
            num_steps=args.num_steps,
            eval_every=args.eval_every,
            y_test=y_test_work,
            lambda_s=lam_s,
            anneal_sigma_v=anneal,
            sigma_v_start=0.2,
            sigma_v_end=2.0,
            warmup_frac=0.5,
            freeze_sigma_v=freeze_sv,
        )
        all_records.extend(recs)

        if mode == "mixture_ab":
            mixture_state = (V, u, h, _clone_regions(regions))

    if args.run_jumpgp or "jumpgp_sampleW" in modes:
        print(f"\n{'='*70}\n  MODE: jumpgp_sampleW\n{'='*70}", flush=True)
        rec = _run_jumpgp_baseline(
            X_train_t, y_train_t, X_test_t, y_test_work,
            Q=Q, n=n, m1=args.m1, m2=args.m2,
            num_steps=args.num_steps, lr=args.lr, device=device, seed=args.seed,
        )
        if scaler_y is not None:
            rec["rmse_raw"] = rec["rmse"]
        else:
            rec["rmse_raw"] = rec["rmse"]
        print(f"    >>> jumpgp_sampleW RMSE={rec['rmse']:.4f}", flush=True)
        all_records.append(rec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_records for k in r})
    with (out_dir / "diag_records.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_records)

    print(f"\n{'='*90}\nFINAL (last checkpoint per mode)\n{'='*90}")
    print(
        f"{'Mode':<20s} {'RMSE':>8s} {'RMSEraw':>8s} {'ratio_f':>8s} "
        f"{'std_mW':>8s} {'std_muf':>8s} {'sig_k':>7s}"
    )
    print("-" * 85)
    for mode in sorted(set(r["mode"] for r in all_records)):
        rec = next(r for r in reversed(all_records) if r["mode"] == mode)
        print(
            f"{mode:<20s} {rec.get('rmse', float('nan')):8.4f} "
            f"{rec.get('rmse_raw', rec.get('rmse', float('nan'))):8.2f} "
            f"{rec.get('mu_f_to_y_std_ratio', float('nan')):8.4f} "
            f"{rec.get('std_m_W', float('nan')):8.4f} "
            f"{rec.get('std_mu_f', float('nan')):8.4f} "
            f"{rec.get('sigma_k_mean', float('nan')):7.3f}"
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "n_records": len(all_records)}, f, indent=2)

    print(f"\nSaved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

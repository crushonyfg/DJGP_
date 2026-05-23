"""Diagnose why Original ELBO direct-VI prediction RMSE does not improve.

How to run from the repository root with conda env ``jumpGP``::

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/diagnose_orig_elbo_vi_predict.py ^
        --setting lh_q5_train500 --seed 0 --max_test 50 ^
        --num_steps 300 --eval_every 50 --lr 0.01

This script trains the original ELBO and dumps diagnostic statistics at each
eval checkpoint to show exactly WHY direct-VI prediction degenerates:

- q(W) posterior variance  →  does sigma_W stay large?
- Expected kernel E_q[K_fu]  →  does it collapse to a constant?
- Gate pi = E_q[sigmoid(...)]  →  does it degenerate to ~0.5?
- Predictive mu_f statistics  →  is mu_f extreme or constant?
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.variational import (  # noqa: E402
    compute_ELBO,
    expected_Kfu,
    expected_KufKfu,
    expected_sigmoid_gh,
    qW_from_qV,
    train_vi,
)
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import (  # noqa: E402
    _predict_direct_variational_consistent,
    _stack_psd_sigma_u,
    _train_lmjgp_state,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from shared.jumpgp_runner import find_neighborhoods  # noqa: E402


def diagnose_prediction_state(
    regions, V_params, u_params, hyperparams, y_test, step_label: str,
):
    """Print diagnostic statistics about the prediction pathway."""
    device = hyperparams["Z"].device
    T = len(regions)
    Q = V_params["mu_V"].shape[1]

    X_test = hyperparams["X_test"]
    mu_W, cov_W = qW_from_qV(
        X_test, hyperparams["Z"],
        V_params["mu_V"], V_params["sigma_V"],
        hyperparams["lengthscales"], hyperparams["var_w"],
    )
    sigma_W = torch.sqrt(cov_W.diagonal(dim1=2, dim2=3))  # [T, Q, D]

    # 1) q(W) posterior spread
    mu_W_norm = mu_W.norm(dim=-1).mean().item()       # avg norm of mean
    sigma_W_mean = sigma_W.mean().item()               # avg std per element
    sigma_W_max = sigma_W.max().item()

    # 2) Expected kernel: E_q[K_fu] at the test point
    C = torch.stack([r["C"] for r in regions], dim=0)
    sigma_k = torch.stack([u["sigma_k"] for u in u_params], dim=0).view(T).clamp_min(1e-6)
    Xn = X_test.unsqueeze(1)  # [T, 1, D]
    m_W = expected_Kfu(mu_W, cov_W, Xn, C, sigma_k).squeeze(1)  # [T, m1]
    kfu_mean = m_W.mean().item()
    kfu_std = m_W.std().item()
    kfu_max = m_W.abs().max().item()

    # s = x^T Sigma_W x: the "variance inflation" term in expected kernel
    s = torch.einsum("tnd,tqde,tne->tnq", Xn, cov_W, Xn)  # [T, 1, Q]
    s_mean = s.mean().item()
    s_max = s.max().item()

    # 3) Gate: pi = E_q[sigmoid(omega^T [1, Wx*])]
    omega = torch.stack([u["omega"] for u in u_params], dim=0)
    pi = expected_sigmoid_gh(omega, mu_W, cov_W, Xn).squeeze(1)  # [T]
    pi_mean = pi.mean().item()
    pi_std = pi.std().item()
    pi_min = pi.min().item()
    pi_max = pi.max().item()

    # 4) Predictive mu_f
    S_W = expected_KufKfu(mu_W, cov_W, Xn, C, sigma_k).squeeze(1)
    m1 = C.shape[1]
    d2 = (C.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(-1)
    eye = torch.eye(m1, device=device, dtype=C.dtype).unsqueeze(0)
    Kuu = torch.exp(-0.5 * d2) + 1e-6 * eye
    Luu = torch.linalg.cholesky(0.5 * (Kuu + Kuu.transpose(-1, -2)))
    Kinv = torch.cholesky_inverse(Luu)
    mu_u = torch.stack([u["mu_u"] for u in u_params], dim=0)
    a = torch.einsum("tij,tj->ti", Kinv, mu_u)
    mu_f = (m_W * a).sum(dim=1)  # [T]

    U = torch.sigmoid(torch.stack([u["U_logit"] for u in u_params], dim=0)).view(T)
    mu_y = pi * mu_f + (1.0 - pi) * U

    y_t = torch.from_numpy(y_test.astype(np.float32)).to(device)[:T]
    rmse = torch.sqrt(((mu_y - y_t) ** 2).mean()).item()

    # 5) Hyperparams
    ls = hyperparams["lengthscales"].detach().cpu().numpy()
    vw = hyperparams["var_w"].item()
    sigma_noise = torch.stack([u["sigma_noise"] for u in u_params]).detach()

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTICS at {step_label}")
    print(f"{'='*70}")
    print(f"  Hyperparams: lengthscales={ls}, var_w={vw:.4f}")
    print(f"  sigma_noise: mean={sigma_noise.mean().item():.4f}, range=[{sigma_noise.min().item():.4f}, {sigma_noise.max().item():.4f}]")
    print(f"  sigma_k:     mean={sigma_k.mean().item():.4f}")
    print()
    print(f"  q(W) posterior:")
    print(f"    |mu_W|_mean     = {mu_W_norm:.4f}")
    print(f"    sigma_W_mean    = {sigma_W_mean:.4f}")
    print(f"    sigma_W_max     = {sigma_W_max:.4f}")
    print(f"    ratio sigma/|mu|= {sigma_W_mean / (mu_W_norm + 1e-8):.4f}")
    print()
    print(f"  Variance inflation s = x^T Sigma_W x:")
    print(f"    s_mean = {s_mean:.4f}   s_max = {s_max:.4f}")
    print(f"    (denominator in E[Kfu] is sqrt(s+1) → {np.sqrt(s_mean+1):.4f})")
    print()
    print(f"  E_q[K_fu] (expected kernel at test point):")
    print(f"    mean = {kfu_mean:.6f}   std = {kfu_std:.6f}   max = {kfu_max:.6f}")
    print(f"    {'*** COLLAPSED: near-constant kernel!' if kfu_std < 0.01 * kfu_mean else 'OK: kernel has variation'}")
    print()
    print(f"  Gate pi = E_q[sigmoid(omega^T [1,Wx])]:")
    print(f"    mean = {pi_mean:.4f}   std = {pi_std:.4f}")
    print(f"    range = [{pi_min:.4f}, {pi_max:.4f}]")
    print(f"    {'*** DEGENERATE: pi ≈ 0.5!' if abs(pi_mean - 0.5) < 0.1 and pi_std < 0.1 else 'OK: gate is discriminating'}")
    print()
    print(f"  Predictive moments:")
    print(f"    mu_f:  mean={mu_f.mean().item():.4f}, std={mu_f.std().item():.4f}, range=[{mu_f.min().item():.4f}, {mu_f.max().item():.4f}]")
    print(f"    U:     mean={U.mean().item():.4f}")
    print(f"    mu_y:  mean={mu_y.mean().item():.4f}, std={mu_y.std().item():.4f}")
    print(f"    y_test:mean={y_t.mean().item():.4f}, std={y_t.std().item():.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"{'='*70}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", default="lh_q5_train500")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=50)
    p.add_argument("--num_steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    args = p.parse_args()

    cfg = SETTING_PRESETS[args.setting]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _set_seed(args.seed)
    data = _generate_paper_data(cfg, args.seed, device)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data["X_train"])
    X_test_s = scaler.transform(data["X_test"])
    y_train = data["y_train"]
    y_test = data["y_test"][:args.max_test]
    X_test_s = X_test_s[:args.max_test]

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)

    Q = int(cfg["Q"])
    n = int(cfg["n"])
    m1 = args.m1
    m2 = args.m2
    T = X_test_t.shape[0]

    # Build regions and params
    neighborhoods = find_neighborhoods(X_test_t.cpu(), X_train_t.cpu(), y_train_t.cpu(), M=n)
    regions = []
    for item in neighborhoods:
        regions.append({
            "X": item["X_neighbors"].to(device),
            "y": item["y_neighbors"].to(device),
            "C": torch.randn(m1, Q, device=device),
        })

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

    # Diagnose at init (step 0)
    with torch.no_grad():
        diagnose_prediction_state(regions, V_params, u_params, hyperparams, y_test, "step=0 (init)")

    # Training with periodic diagnosis
    params = [V_params["mu_V"], V_params["sigma_V"]]
    for u in u_params:
        params += [u["U_logit"], u["mu_u"], u["Sigma_u"],
                   u["sigma_noise"], u["sigma_k"], u["omega"]]
    params += [hyperparams["lengthscales"], hyperparams["var_w"], hyperparams["Z"]]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    for step in range(1, args.num_steps + 1):
        optimizer.zero_grad()
        elbo = compute_ELBO(regions, V_params, u_params, hyperparams)
        (-elbo).backward()
        optimizer.step()

        with torch.no_grad():
            V_params["sigma_V"].clamp_(min=0.1)
            hyperparams["var_w"].clamp_(min=0.1)
            hyperparams["lengthscales"].clamp_(min=1e-6)
            for u in u_params:
                u["sigma_noise"].clamp_(min=0.1)
                u["sigma_k"].clamp_(min=0.1)

        if step % args.eval_every == 0 or step == 1:
            print(f"\n[train step {step}/{args.num_steps}] ELBO = {elbo.item():.4f}")
            with torch.no_grad():
                diagnose_prediction_state(
                    regions, V_params, u_params, hyperparams,
                    y_test, f"step={step}",
                )


if __name__ == "__main__":
    main()

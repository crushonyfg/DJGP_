"""ELBO shrinkage decomposition on D3 (pure inlier + m_j).

Trains D3 (optional), then sweeps ``beta_G``, ``beta_K``, and ``sigma_n`` caps.

KPI: raise ``z_star/r`` from ~0.005 toward 0.05--0.1.

```powershell
python experiments/synthetic/diagnose_elbo_shrinkage_d3.py ^
    --max_test 200 --num_steps 300 --eval_every 100
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

from djgp.elbo_shrinkage import (
    beta_shrinkage_grid,
    elbo_star_metrics,
    format_beta_grid_table,
    sigma_n_shrinkage_sweep,
)
from djgp.structured_projection import (
    build_U_numpy,
    build_W0_numpy,
    build_proj_params,
    clamp_projection,
    compute_s_penalty_proj,
    init_inducing_from_w0x,
    proj_trainable_tensors,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from experiments.synthetic.diagnose_inlier_gp_path_direct_vi import _build_state_diagnose
from experiments.synthetic.local_gp_elbo_mj import (
    apply_sigma_n_control,
    compute_elbo_pure_inlier_mj,
    init_sigma_n_from_spec,
    predict_inlier_mj,
    sigma_n_learnable,
    training_beta_u_kl,
)
from experiments.synthetic.test_remedy_d_structured_vi import _clone_proj, _d3_base


def _variant_specs() -> dict[str, dict[str, Any]]:
    base = {**_d3_base(), "diag_shrinkage": True, "diag_mixture": True}
    return {
        "D3": dict(base),
        "D3_snmax0.5": {**base, "sigma_n_max": 0.5},
        "D3_snmax0.8": {**base, "sigma_n_max": 0.8},
        "D3_snfix0.2": {**base, "sigma_n_fixed": 0.2},
        "D3_snfix0.3": {**base, "sigma_n_fixed": 0.3},
        "D3_snfix0.5": {**base, "sigma_n_fixed": 0.5},
        "D3_snwarm0.2_release": {
            **base,
            "sigma_n_fixed": 0.2,
            "sigma_n_freeze_steps": 150,
            "sigma_n_max": 0.8,
        },
        "D3_snprior_strong": {
            **base,
            "lambda_sigma_prior": 10.0,
            "log_sigma_n0": -1.6094379124345703,
        },
        "D3_betaU0.3": {**base, "beta_u_kl": 0.3},
    }


def _build_d3(
    X_train_t, y_train_t, X_test_t, *, Q, n, m1, m2, device, seed, y_train_np, spec,
):
    regions, _V, u_params, hyperparams = _build_state_diagnose(
        X_train_t, y_train_t, X_test_t,
        Q=Q, n=n, m1=m1, m2=m2, device=device, seed=seed,
        explicit_mj=True, mj_init="mean", learn_mj=True,
    )
    W0_np = build_W0_numpy(
        X_train_t.cpu().numpy(), y_train_np, Q, spec.get("w0_init", "pls"), seed,
    )
    rank = int(spec.get("rank", 3))
    U_np = build_U_numpy(
        X_train_t.cpu().numpy(), y_train_np, W0_np, rank,
        spec.get("basis_mode", "pca_residual"), seed + 17,
    )
    proj = build_proj_params(
        kind=spec["proj_kind"], device=device, m2=m2, Q=Q, D=X_train_t.shape[1],
        r=U_np.shape[1], W0_np=W0_np, U_np=U_np,
        sigma_init=0.05, sigma_max=spec.get("sigma_max"),
    )
    init_inducing_from_w0x(regions, proj["W0"], seed=seed + 3)
    init_sigma_n_from_spec(u_params, spec)
    return regions, proj, u_params, hyperparams


def _leaf_param(t: torch.Tensor) -> torch.Tensor:
    if t.is_leaf:
        return t
    return t.detach().clone().requires_grad_(True)


def _train(
    regions, proj, u_params, hyperparams, spec, *, lr, num_steps, sigma_n_max,
):
    device = hyperparams["Z"].device
    hyperparams["Z"] = _leaf_param(hyperparams["Z"])
    for key in ("mu_B", "sigma_B", "mu_V", "sigma_V"):
        if key in proj:
            proj[key] = _leaf_param(proj[key])
    params = proj_trainable_tensors(proj) + [
        hyperparams["lengthscales"], hyperparams["var_w"], hyperparams["Z"],
    ]
    for u in u_params:
        params += [u["mu_u"], u["Sigma_u"], u["sigma_k"]]
        if u["sigma_noise"].requires_grad:
            params.append(u["sigma_noise"])
        if u.get("m_j") is not None and u["m_j"].requires_grad:
            params.append(u["m_j"])
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}
    sn_max = sigma_n_max

    for step in range(1, num_steps + 1):
        opt.zero_grad()
        beta_u = training_beta_u_kl(spec, step, num_steps)
        elbo = compute_elbo_pure_inlier_mj(
            regions, V_dummy, u_params, hyperparams, proj=proj,
            beta_u_kl=beta_u,
            lambda_sigma_prior=float(spec.get("lambda_sigma_prior", 0.0)),
            log_sigma_n0=float(spec.get("log_sigma_n0", 0.0)),
        )
        lam_s = float(spec.get("lambda_s", 0.0))
        if lam_s > 0:
            elbo = elbo - lam_s * compute_s_penalty_proj(regions, proj, hyperparams)
        (-elbo).backward()
        opt.step()
        with torch.no_grad():
            clamp_projection(proj, spec.get("sigma_max"))
            hyperparams["var_w"].clamp_(min=0.1)
            hyperparams["lengthscales"].clamp_(min=1e-6, max=1e2)
            apply_sigma_n_control(u_params, spec, step)
            freeze_sn = int(spec.get("sigma_n_freeze_steps", 0) or 0)
            if freeze_sn > 0 and step == freeze_sn + 1:
                opt.add_param_group({"params": [u["sigma_noise"] for u in u_params]})
            for u in u_params:
                u["sigma_k"].clamp_(min=0.1)
        if step == 1 or step == num_steps or step % max(num_steps // 5, 1) == 0:
            sn = float(torch.stack([u["sigma_noise"] for u in u_params]).mean().item())
            print(f"  step {step}/{num_steps} ELBO={elbo.item():.4f}  mean_sigma_n={sn:.4f}", flush=True)


@torch.no_grad()
def _run_shrinkage_diagnostics(regions, proj, u_params, hyperparams, tag: str) -> list[dict]:
    betas = (0.0, 0.1, 0.3, 1.0)
    rows: list[dict[str, float]] = []

    base = elbo_star_metrics(regions, proj, hyperparams, u_params)
    base["variant"] = tag
    base["diag"] = "baseline"
    rows.append(base)

    for row in beta_shrinkage_grid(regions, proj, hyperparams, u_params, betas=betas):
        row["variant"] = tag
        row["diag"] = "beta_grid"
        rows.append(row)

    for row in sigma_n_shrinkage_sweep(regions, proj, hyperparams, u_params):
        row["variant"] = tag
        row["diag"] = "sigma_n"
        rows.append(row)

    print(f"\n--- {tag} beta_G x beta_K  (z_star/r) ---", flush=True)
    print(format_beta_grid_table(rows, betas), flush=True)
    print(
        f"  baseline: z_star/r={base['ratio_zeta_star_r']:.4f}  "
        f"z_learn/r={base['ratio_zeta_learned_r']:.4f}  "
        f"z_ridge/r={base['ratio_zeta_ridge_r']:.4f}  "
        f"sigma_n={base['sigma_n_mean']:.4f}  "
        f"|K^-1 mu|={base['Kinvt_mu_learned_norm_mean']:.4f}",
        flush=True,
    )
    sn_rows = [r for r in rows if r["diag"] == "sigma_n" and r.get("sweep") != "learned_sigma_n"]
    best = max(sn_rows, key=lambda r: r["ratio_zeta_star_r"], default=None)
    if best:
        print(
            f"  best sigma_n tweak: {best.get('sweep')}  z_star/r={best['ratio_zeta_star_r']:.4f}  "
            f"(sigma_n_mean={best['sigma_n_mean']:.4f})",
            flush=True,
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="ELBO G vs sigma_n shrinkage diagnostics (D3).")
    p.add_argument("--setting", default="lh_q5_train500")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--num_steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument(
        "--variants",
        default="D3_snfix0.2,D3_snfix0.3,D3_snfix0.5,D3_snmax0.8",
    )
    p.add_argument("--skip_train", action="store_true", help="Only run diagnostics at init.")
    p.add_argument("--out_dir", default="experiments/synthetic/elbo_shrinkage_d3_seed0")
    args = p.parse_args()

    cfg = SETTING_PRESETS[args.setting]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    _set_seed(args.seed)

    data = _generate_paper_data(cfg, args.seed, device)
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(data["X_train"])
    X_test_s = scaler_x.transform(data["X_test"])[: args.max_test]
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(np.asarray(data["y_train"]).reshape(-1, 1)).ravel()
    y_test_work = scaler_y.transform(np.asarray(data["y_test"])[: args.max_test].reshape(-1, 1)).ravel()
    y_test_raw = np.asarray(data["y_test"])[: args.max_test]

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)
    Q, n = int(cfg["Q"]), int(cfg["n"])

    all_rows: list[dict] = []
    want = [v.strip() for v in args.variants.split(",") if v.strip()]

    for vname in want:
        spec = _variant_specs()[vname]
        print(f"\n{'='*70}\n  {vname}: {json.dumps(spec)}\n{'='*70}", flush=True)
        regions, proj, u_params, hyperparams = _build_d3(
            X_train_t, y_train_t, X_test_t,
            Q=Q, n=n, m1=args.m1, m2=args.m2, device=device, seed=args.seed,
            y_train_np=y_train, spec=spec,
        )
        if not args.skip_train and args.num_steps > 0:
            _train(
                regions, proj, u_params, hyperparams, spec,
                lr=args.lr, num_steps=args.num_steps,
                sigma_n_max=spec.get("sigma_n_max"),
            )
        all_rows.extend(_run_shrinkage_diagnostics(regions, proj, u_params, hyperparams, vname))

        if args.num_steps > 0 and not args.skip_train:
            V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}
            mu_y, _, zeta = predict_inlier_mj(
                regions, V_dummy, u_params, hyperparams, proj=proj,
            )
            y_t = torch.from_numpy(y_test_work.astype(np.float32)).to(device)[: len(regions)]
            rmse = float(torch.sqrt(((mu_y - y_t) ** 2).mean()).item())
            print(f"  RMSE_work={rmse:.4f}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_rows for k in r})
    with (out / "shrinkage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {out / 'shrinkage.csv'}", flush=True)
    print(
        f"\n{'='*80}\nKPI summary (z_star/r at beta_G=beta_K=1)\n{'='*80}",
        flush=True,
    )
    for v in want:
        hits = [r for r in all_rows if r["variant"] == v and r.get("diag") == "baseline"]
        if hits:
            h = hits[-1]
            print(
                f"  {v:<18s} z_star/r={h['ratio_zeta_star_r']:.4f}  "
                f"z_learn/r={h['ratio_zeta_learned_r']:.4f}  "
                f"sigma_n={h['sigma_n_mean']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

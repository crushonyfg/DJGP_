"""Fixed gate slope experiments on D3 structured projection + mixture ELBO + m_j.

Variants:
- ``G0``: learned ``omega`` (full ``nu``)
- ``G1``: ``v = 1/Q``, learn ``omega0`` only
- ``G2``: ``v = e_1``, learn ``omega0`` only
- ``G3``: ``v = e_1`` + learnable gate temperature ``tau_g``

```powershell
python experiments/synthetic/test_gate_fixed_slope_d3.py ^
    --variants G0,G1,G2,G3 --max_test 200 --num_steps 300 --eval_every 100
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

from djgp.gate_fixed_slope import gate_trainable_params, init_gate_params
from djgp.gp_feature_alignment import region_alignment_diagnostics
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
from experiments.synthetic.diagnose_inlier_gp_path_direct_vi import (
    _build_state_diagnose,
    _clone_regions,
)
from experiments.synthetic.local_gp_elbo_mj import (
    compute_elbo_mixture_mj,
    mixture_diagnostics_mj,
    predict_inlier_mj,
)
from shared.jumpgp_runner import find_neighborhoods  # noqa: E402


def _d3_proj_spec() -> dict[str, Any]:
    return {
        "proj_kind": "lowrank",
        "rank": 3,
        "lambda_s": 0.1,
        "sigma_max": 0.05,
        "w0_init": "pls",
        "basis_mode": "pca_residual",
        "init_c_w0": True,
    }


def _variant_specs() -> dict[str, dict[str, Any]]:
    base = {
        **_d3_proj_spec(),
        "explicit_mj": True,
        "learn_mj": True,
        "omega0_init": 1.0,
        "omega0_floor": None,
        "diag_align": True,
    }
    return {
        "G0": {**base, "gate": {"gate_mode": "learned", "omega0_init": 1.0}},
        "G1": {**base, "gate": {"gate_mode": "fixed_ones", "omega0_init": 1.0}},
        "G2": {**base, "gate": {"gate_mode": "fixed_e1", "omega0_init": 1.0}},
        "G3": {
            **base,
            "gate": {"gate_mode": "fixed_e1_temp", "omega0_init": 1.0, "gate_temp_init": 1.0},
        },
    }


def _build_base_gate(
    X_train_t,
    y_train_t,
    X_test_t,
    *,
    Q,
    n,
    m1,
    m2,
    device,
    seed,
    y_train_np,
    spec,
):
    regions, _V, u_params, hyperparams = _build_state_diagnose(
        X_train_t, y_train_t, X_test_t,
        Q=Q, n=n, m1=m1, m2=m2, device=device, seed=seed,
        explicit_mj=True,
        mj_init="mean",
        learn_mj=spec.get("learn_mj", True),
    )
    gate_cfg = spec["gate"]
    new_u = []
    for u in u_params:
        entry = {k: v for k, v in u.items() if k not in ("omega",)}
        entry.update(init_gate_params(Q, device, gate_cfg))
        new_u.append(entry)
    u_params = new_u

    W0_np = build_W0_numpy(
        X_train_t.cpu().numpy(), y_train_np, Q, spec.get("w0_init", "pls"), seed,
    )
    rank = int(spec.get("rank", 3))
    U_np = build_U_numpy(
        X_train_t.cpu().numpy(), y_train_np, W0_np, rank,
        spec.get("basis_mode", "pca_residual"), seed + 17,
    )
    proj = build_proj_params(
        kind=spec["proj_kind"],
        device=device,
        m2=m2,
        Q=Q,
        D=X_train_t.shape[1],
        r=U_np.shape[1],
        W0_np=W0_np,
        U_np=U_np,
        sigma_init=0.05,
        sigma_max=spec.get("sigma_max"),
    )
    if spec.get("init_c_w0"):
        init_inducing_from_w0x(regions, proj["W0"], seed=seed + 3)
    return regions, proj, u_params, hyperparams


def _clone_proj(proj):
    out = {"kind": proj["kind"], "W0": proj["W0"].detach().clone()}
    if proj.get("sigma_max") is not None:
        out["sigma_max"] = proj["sigma_max"]
    if proj["kind"] in ("full", "full_centered"):
        out["mu_V"] = proj["mu_V"].detach().clone().requires_grad_(True)
        out["sigma_V"] = proj["sigma_V"].detach().clone().requires_grad_(True)
    else:
        out["U"] = proj["U"].detach().clone()
        out["mu_B"] = proj["mu_B"].detach().clone().requires_grad_(True)
        out["sigma_B"] = proj["sigma_B"].detach().clone().requires_grad_(True)
    return out


def _train_variant(
    variant,
    spec,
    regions,
    proj,
    u_params,
    hyperparams,
    *,
    lr,
    num_steps,
    eval_every,
    y_test_work,
    y_test_raw,
    scaler_y,
):
    device = hyperparams["Z"].device
    gate_cfg = spec["gate"]
    params = proj_trainable_tensors(proj) + [
        hyperparams["lengthscales"], hyperparams["var_w"], hyperparams["Z"],
    ]
    for u in u_params:
        params += [
            u["mu_u"], u["Sigma_u"], u["sigma_noise"], u["sigma_k"], u["U_logit"],
        ]
        params += gate_trainable_params(u, gate_cfg)
        if "m_j" in u and u["m_j"].requires_grad:
            params.append(u["m_j"])

    opt = torch.optim.Adam(params, lr=lr)
    records = []
    V_dummy = {"mu_V": proj.get("mu_V"), "sigma_V": proj.get("sigma_V")}
    omega0_floor = spec.get("omega0_floor")

    for step in range(1, num_steps + 1):
        opt.zero_grad()
        elbo = compute_elbo_mixture_mj(
            regions, V_dummy, u_params, hyperparams,
            proj=proj, gate_config=gate_cfg,
        )
        sp = 0.0
        lam_s = float(spec.get("lambda_s", 0.0))
        if lam_s > 0:
            sp_t = compute_s_penalty_proj(regions, proj, hyperparams)
            sp = float((lam_s * sp_t).item())
            elbo = elbo - lam_s * sp_t
        (-elbo).backward()
        opt.step()
        with torch.no_grad():
            clamp_projection(proj, spec.get("sigma_max"))
            hyperparams["var_w"].clamp_(min=0.1)
            hyperparams["lengthscales"].clamp_(min=1e-6, max=1e2)
            for u in u_params:
                u["sigma_noise"].clamp_(min=0.1)
                u["sigma_k"].clamp_(min=0.1)
                if omega0_floor is not None:
                    if "omega0" in u:
                        u["omega0"].clamp_(min=float(omega0_floor))
                    elif "omega" in u:
                        u["omega"][0].clamp_(min=float(omega0_floor))

        if step % max(1, eval_every // 3) == 0 or step == 1:
            print(
                f"  [{variant} {step}/{num_steps}] ELBO={elbo.item():.4f} s_pen={sp:.4f}",
                flush=True,
            )
        if step % eval_every == 0 or step == num_steps:
            records.append(
                _evaluate(
                    variant, regions, proj, u_params, hyperparams, V_dummy,
                    y_test_work, y_test_raw, scaler_y, step, spec,
                ),
            )
    return records


@torch.no_grad()
def _evaluate(variant, regions, proj, u_params, hyperparams, V_dummy,
              y_test_work, y_test_raw, scaler_y, step, spec):
    device = hyperparams["Z"].device
    T = len(regions)
    gate_cfg = spec["gate"]
    mix = mixture_diagnostics_mj(
        regions, V_dummy, u_params, hyperparams, proj=proj, gate_config=gate_cfg,
    )
    mu_y, _var_y, zeta = predict_inlier_mj(
        regions, V_dummy, u_params, hyperparams, proj=proj,
    )
    y_t = torch.from_numpy(y_test_work.astype(np.float32)).to(device)[:T]
    rmse = float(torch.sqrt(((mu_y - y_t) ** 2).mean()).item())
    if scaler_y is not None:
        mu_raw = scaler_y.inverse_transform(mu_y.cpu().numpy().reshape(-1, 1)).ravel()
        rmse_raw = float(np.sqrt(np.mean((mu_raw - y_test_raw[:T]) ** 2)))
    else:
        rmse_raw = rmse

    align = {}
    if spec.get("diag_align", True):
        align = region_alignment_diagnostics(
            regions, proj, hyperparams, u_params,
        )

    rec = {
        "variant": variant,
        "step": step,
        "gate_mode": gate_cfg["gate_mode"],
        "rmse": rmse,
        "rmse_raw": rmse_raw,
        **mix,
        **align,
    }
    print(
        f"    >>> [{variant} step={step}] RMSE_work={rmse:.4f} raw={rmse_raw:.1f}  "
        f"mu_xi_std={rec.get('mu_xi_std', float('nan')):.4f}  "
        f"pi_std={rec.get('pi_gate_std', float('nan')):.4f}  "
        f"r={rec.get('r_mean', float('nan')):.4f}  "
        f"dT={rec.get('delta_T_mean', float('nan')):.2f}  "
        f"zeta/y_res={rec.get('ratio_zeta_yres', float('nan')):.4f}  "
        f"z_learn/r_elbo={rec.get('ratio_zeta_learned_r_elbo', float('nan')):.4f}  "
        f"z_star/r_elbo={rec.get('ratio_zeta_elbo_star_r', float('nan')):.4f}",
        flush=True,
    )
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed gate slope on D3 mixture+m_j.")
    p.add_argument("--setting", default="lh_q5_train500")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--num_steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--variants", default="G0,G1,G2,G3")
    p.add_argument("--standardize_y", action="store_true", default=True)
    p.add_argument("--no_standardize_y", action="store_true")
    p.add_argument("--out_dir", default="experiments/synthetic/gate_fixed_slope_d3_seed0")
    args = p.parse_args()
    if args.no_standardize_y:
        args.standardize_y = False

    all_specs = _variant_specs()
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
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
    scaler_y = None
    if args.standardize_y:
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_work = scaler_y.transform(y_test_work.reshape(-1, 1)).ravel()

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)
    Q, n = int(cfg["Q"]), int(cfg["n"])

    all_records = []
    for vname in want:
        spec = dict(all_specs[vname])
        print(f"\n{'='*70}\n  {vname}: {json.dumps(spec)}\n{'='*70}", flush=True)
        regions, proj, u_params, hyperparams = _build_base_gate(
            X_train_t, y_train_t, X_test_t,
            Q=Q, n=n, m1=args.m1, m2=args.m2, device=device, seed=args.seed,
            y_train_np=y_train, spec=spec,
        )
        recs = _train_variant(
            vname, spec, _clone_regions(regions), _clone_proj(proj), u_params, hyperparams,
            lr=args.lr, num_steps=args.num_steps, eval_every=args.eval_every,
            y_test_work=y_test_work, y_test_raw=y_test_raw, scaler_y=scaler_y,
        )
        all_records.extend(recs)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_records for k in r})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_records)

    print(f"\n{'='*80}\nFINAL\n{'='*80}")
    print(
        f"{'Var':<6s} {'mu_xi_std':>10s} {'pi_std':>8s} {'r':>8s} "
        f"{'z/y_res':>8s} {'z_learn':>8s} {'z_star':>8s} {'RMSE':>8s}"
    )
    for v in want:
        r = next(x for x in reversed(all_records) if x["variant"] == v)
        print(
            f"{v:<6s} {r.get('mu_xi_std', float('nan')):10.4f} "
            f"{r.get('pi_gate_std', float('nan')):8.4f} "
            f"{r.get('r_mean', float('nan')):8.4f} "
            f"{r.get('ratio_zeta_yres', float('nan')):8.4f} "
            f"{r.get('ratio_zeta_learned_r_elbo', float('nan')):8.4f} "
            f"{r.get('ratio_zeta_elbo_star_r', float('nan')):8.4f} "
            f"{r.get('rmse', float('nan')):8.4f}"
        )
    print(f"\nSaved {out / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()

"""Compare MC-ELBO vs Original ELBO on LH synthetic data (4 variants).

How to run from the repository root with conda env ``jumpGP``::

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Quick smoke test (few test points, few steps)
    python experiments/synthetic/compare_mc_elbo_vs_original.py ^
        --setting lh_q5_train200 --seed 0 --max_test 30 ^
        --orig_steps 150 --mc_steps 80 --MC_num 3 --mc_S 2 ^
        --inner_steps 8 --inner_lr 0.05 ^
        --out_dir experiments/synthetic/mc_elbo_vs_original_smoke

    # Full comparison
    python experiments/synthetic/compare_mc_elbo_vs_original.py ^
        --setting lh_q5_train500 --seed 0 --max_test 200 ^
        --orig_steps 300 --mc_steps 150 --MC_num 5 --mc_S 3 ^
        --inner_steps 15 --inner_lr 0.05 ^
        --out_dir experiments/synthetic/mc_elbo_vs_original_lh500_seed0

This runner trains both the original LMJGP ELBO and the new MC-ELBO on the
same LH synthetic data, then evaluates 4 prediction variants:

1. Original ELBO → ``predict_vi_analytic`` (direct VI predict)
2. Original ELBO → ``predict_vi`` (sample W + JGP CEM)
3. MC-ELBO → ``predict_mc_vi_direct`` (per-sample-optimized direct predict)
4. MC-ELBO → ``predict_mc_vi`` (sample W + JGP CEM)

Data generator: paper-style LH from ``new_highdata_gen_utils.generate_data``
with RFF expansion and train-only X standardization.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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

from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.variational import (  # noqa: E402
    predict_vi,
    qW_from_qV,
    train_vi,
)
from djgp.variational_mc import (  # noqa: E402
    predict_mc_vi,
    predict_mc_vi_direct,
    train_mc_vi,
)
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import (  # noqa: E402
    _predict_direct_variational_consistent,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from shared.jumpgp_runner import find_neighborhoods  # noqa: E402


def _metrics_pack(y, mu, sigma, wall_sec):
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting", "family", "seed", "variant",
        "N_train", "N_test", "n_test_eval",
        "latent_dim", "observed_dim", "Q", "n_neighbors",
        *RESULT_ROW_KEYS,
        "train_sec", "pred_sec", "error",
    ]
    keys = [k for k in preferred if any(k in r for r in rows)]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _build_regions(
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    *,
    n: int,
    m1: int,
    Q: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
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
    return regions


def _init_V_and_hyp(
    X_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    *,
    m2: int,
    Q: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
    return V_params, hyperparams


def _init_u_params(T: int, m1: int, Q: int, device: torch.device) -> list[dict[str, torch.Tensor]]:
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
    return u_params


def run_comparison(args: argparse.Namespace) -> list[dict[str, Any]]:
    setting = args.setting
    cfg = SETTING_PRESETS[setting]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _set_seed(args.seed)
    data = _generate_paper_data(cfg, args.seed, device)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data["X_train"])
    X_test_s = scaler.transform(data["X_test"])
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Limit test points
    if args.max_test > 0 and args.max_test < len(y_test):
        X_test_s = X_test_s[: args.max_test]
        y_test = y_test[: args.max_test]

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)

    Q = int(cfg["Q"])
    n = int(cfg["n"])
    m1 = args.m1
    m2 = args.m2
    T = X_test_t.shape[0]

    base_row = {
        "setting": setting,
        "family": cfg["family"],
        "seed": args.seed,
        "N_train": int(cfg["N_train"]),
        "N_test": len(y_test),
        "n_test_eval": T,
        "latent_dim": int(cfg["d"]),
        "observed_dim": int(cfg["H"]),
        "Q": Q,
        "n_neighbors": n,
    }
    rows: list[dict[str, Any]] = []

    # ===================================================================
    # Part A: Original ELBO training
    # ===================================================================
    print("\n" + "=" * 60)
    print("Training ORIGINAL ELBO...")
    print("=" * 60)

    _set_seed(args.seed)
    regions_orig = _build_regions(X_train_t, y_train_t, X_test_t, n=n, m1=m1, Q=Q, device=device)
    V_orig, hyp_orig = _init_V_and_hyp(X_train_t, X_test_t, m2=m2, Q=Q, device=device)
    u_orig = _init_u_params(T, m1, Q, device)

    t0 = time.perf_counter()
    V_orig, u_orig, hyp_orig = train_vi(
        regions=regions_orig,
        V_params=V_orig,
        u_params=u_orig,
        hyperparams=hyp_orig,
        lr=args.lr,
        num_steps=args.orig_steps,
        log_interval=max(20, args.orig_steps // 10),
    )
    orig_train_sec = time.perf_counter() - t0
    print(f"Original ELBO training done in {orig_train_sec:.1f}s")

    # --- Variant 1: Original ELBO → direct VI predict ---
    print("\nVariant 1: Original ELBO → direct VI predict")
    t0 = time.perf_counter()
    try:
        mu_vi, var_vi = _predict_direct_variational_consistent(
            regions_orig, V_orig, u_orig, hyp_orig,
        )
        pred_sec = time.perf_counter() - t0
        mu_vi = torch.nan_to_num(mu_vi, nan=0.0)
        var_vi = torch.nan_to_num(var_vi, nan=1e-6).clamp_min(1e-12)
        sigma_vi = torch.sqrt(var_vi)
        pack = _metrics_pack(y_test, mu_vi.detach().cpu().numpy(),
                             sigma_vi.detach().cpu().numpy(), pred_sec)
        row = dict(base_row)
        row.update({"variant": "orig_elbo_vi_predict", "train_sec": orig_train_sec, "pred_sec": pred_sec})
        row.update({k: float(pack[i]) for i, k in enumerate(RESULT_ROW_KEYS)})
        rows.append(row)
        print(f"  RMSE = {pack[0]:.4f}")
    except Exception as e:
        row = dict(base_row)
        row.update({"variant": "orig_elbo_vi_predict", "error": str(e)})
        rows.append(row)
        print(f"  ERROR: {e}")

    # --- Variant 2: Original ELBO → sample W + JGP CEM ---
    print("\nVariant 2: Original ELBO → sample W + JGP CEM")
    t0 = time.perf_counter()
    try:
        mu_cem, var_cem = predict_vi(regions_orig, V_orig, hyp_orig, M=args.MC_num)
        pred_sec = time.perf_counter() - t0
        sigma_cem = torch.sqrt(var_cem.clamp_min(1e-12))
        pack = _metrics_pack(y_test, mu_cem.detach().cpu().numpy(),
                             sigma_cem.detach().cpu().numpy(), pred_sec)
        row = dict(base_row)
        row.update({"variant": "orig_elbo_sampleW_cem", "train_sec": orig_train_sec, "pred_sec": pred_sec})
        row.update({k: float(pack[i]) for i, k in enumerate(RESULT_ROW_KEYS)})
        rows.append(row)
        print(f"  RMSE = {pack[0]:.4f}")
    except Exception as e:
        row = dict(base_row)
        row.update({"variant": "orig_elbo_sampleW_cem", "error": str(e)})
        rows.append(row)
        print(f"  ERROR: {e}")

    # ===================================================================
    # Part B: MC-ELBO training
    # ===================================================================
    print("\n" + "=" * 60)
    print("Training MC-ELBO...")
    print("=" * 60)

    _set_seed(args.seed)
    regions_mc = _build_regions(X_train_t, y_train_t, X_test_t, n=n, m1=m1, Q=Q, device=device)
    V_mc, hyp_mc = _init_V_and_hyp(X_train_t, X_test_t, m2=m2, Q=Q, device=device)

    t0 = time.perf_counter()
    V_mc, hyp_mc, last_locals = train_mc_vi(
        regions=regions_mc,
        V_params=V_mc,
        hyperparams=hyp_mc,
        S=args.mc_S,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        m1=m1,
        lr=args.lr,
        num_steps=args.mc_steps,
        log_interval=max(10, args.mc_steps // 10),
    )
    mc_train_sec = time.perf_counter() - t0
    print(f"MC-ELBO training done in {mc_train_sec:.1f}s")

    # --- Variant 3: MC-ELBO → direct predict (per-sample optimized locals) ---
    print("\nVariant 3: MC-ELBO → direct VI predict (per-sample locals)")
    t0 = time.perf_counter()
    try:
        mu_mc_direct, var_mc_direct = predict_mc_vi_direct(
            regions_mc, V_mc, hyp_mc,
            M=args.MC_num, inner_steps=args.inner_steps, inner_lr=args.inner_lr, m1=m1,
        )
        pred_sec = time.perf_counter() - t0
        sigma_mc_direct = torch.sqrt(var_mc_direct.clamp_min(1e-12))
        pack = _metrics_pack(y_test, mu_mc_direct.detach().cpu().numpy(),
                             sigma_mc_direct.detach().cpu().numpy(), pred_sec)
        row = dict(base_row)
        row.update({"variant": "mc_elbo_vi_predict", "train_sec": mc_train_sec, "pred_sec": pred_sec})
        row.update({k: float(pack[i]) for i, k in enumerate(RESULT_ROW_KEYS)})
        rows.append(row)
        print(f"  RMSE = {pack[0]:.4f}")
    except Exception as e:
        row = dict(base_row)
        row.update({"variant": "mc_elbo_vi_predict", "error": str(e)})
        rows.append(row)
        print(f"  ERROR: {e}")

    # --- Variant 4: MC-ELBO → sample W + JGP CEM ---
    print("\nVariant 4: MC-ELBO → sample W + JGP CEM")
    t0 = time.perf_counter()
    try:
        mu_mc_cem, var_mc_cem = predict_mc_vi(
            regions_mc, V_mc, hyp_mc, M=args.MC_num,
        )
        pred_sec = time.perf_counter() - t0
        sigma_mc_cem = torch.sqrt(var_mc_cem.clamp_min(1e-12))
        pack = _metrics_pack(y_test, mu_mc_cem.detach().cpu().numpy(),
                             sigma_mc_cem.detach().cpu().numpy(), pred_sec)
        row = dict(base_row)
        row.update({"variant": "mc_elbo_sampleW_cem", "train_sec": mc_train_sec, "pred_sec": pred_sec})
        row.update({k: float(pack[i]) for i, k in enumerate(RESULT_ROW_KEYS)})
        rows.append(row)
        print(f"  RMSE = {pack[0]:.4f}")
    except Exception as e:
        row = dict(base_row)
        row.update({"variant": "mc_elbo_sampleW_cem", "error": str(e)})
        rows.append(row)
        print(f"  ERROR: {e}")

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare MC-ELBO vs Original ELBO on LH synthetic data.")
    p.add_argument("--setting", default="lh_q5_train200",
                   choices=list(SETTING_PRESETS.keys()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=50,
                   help="Max test points to evaluate (0=all)")
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.01)
    # Original ELBO settings
    p.add_argument("--orig_steps", type=int, default=300)
    # MC-ELBO settings
    p.add_argument("--mc_steps", type=int, default=100,
                   help="Outer steps for MC-ELBO (fewer needed since inner loop also optimizes)")
    p.add_argument("--mc_S", type=int, default=3,
                   help="Number of W samples per MC-ELBO step")
    p.add_argument("--inner_steps", type=int, default=10,
                   help="Inner optimization steps for local params per W sample")
    p.add_argument("--inner_lr", type=float, default=0.05)
    # Prediction
    p.add_argument("--MC_num", type=int, default=3,
                   help="Number of W samples for prediction (both methods)")
    p.add_argument("--out_dir", default="experiments/synthetic/mc_elbo_vs_original_smoke")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_comparison(args)
    _write_csv(out_dir / "metrics.csv", rows)

    summary = {"args": vars(args), "n_rows": len(rows)}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<35s} {'RMSE':>8s} {'CRPS':>8s} {'Train(s)':>9s} {'Pred(s)':>8s}")
    print("-" * 70)
    for row in rows:
        variant = row.get("variant", "?")
        rmse = row.get("rmse", row.get("error", "err"))
        crps = row.get("mean_crps", "")
        train_s = row.get("train_sec", "")
        pred_s = row.get("pred_sec", "")
        if isinstance(rmse, float):
            print(f"{variant:<35s} {rmse:8.4f} {crps:8.4f} {train_s:9.1f} {pred_s:8.1f}")
        else:
            print(f"{variant:<35s} {str(rmse):>8s}")
    print("=" * 70)

    readme = f"""# MC-ELBO vs Original ELBO Comparison

Setting: {args.setting}, seed={args.seed}, max_test={args.max_test}

Variants:
1. `orig_elbo_vi_predict` — Original ELBO training → direct VI predict
2. `orig_elbo_sampleW_cem` — Original ELBO training → sample W + JGP CEM
3. `mc_elbo_vi_predict` — MC-ELBO training → per-sample-optimized direct predict
4. `mc_elbo_sampleW_cem` — MC-ELBO training → sample W + JGP CEM

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"\nSaved {len(rows)} rows to {out_dir}")
    return rows


if __name__ == "__main__":
    main()

"""Train Original ELBO and track direct-VI predict RMSE/CRPS during training.

How to run from the repository root with conda env ``jumpGP``::

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/validate_orig_elbo_direct_vi_curve.py ^
        --setting lh_q5_train500 --seed 0 --max_test 200 ^
        --num_steps 900 --eval_every 150 --lr 0.01 ^
        --out_dir experiments/synthetic/orig_elbo_direct_vi_curve_lh500_seed0

The script trains ``train_vi``-style Original ELBO (``compute_ELBO``) and every
``eval_every`` steps evaluates ``_predict_direct_variational_consistent``
(direct VI predict, ELBO-consistent).  It writes ``curve_metrics.csv`` and
``rmse_crps_vs_step.png`` under ``out_dir``.
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

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.variational import compute_ELBO  # noqa: E402
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import (  # noqa: E402
    _predict_direct_variational_consistent,
)
from experiments.synthetic.compare_mc_elbo_vs_original import (  # noqa: E402
    _build_regions,
    _init_V_and_hyp,
    _init_u_params,
    _metrics_pack,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


@torch.no_grad()
def _eval_direct_vi(
    regions: list[dict[str, torch.Tensor]],
    V_params: dict[str, torch.Tensor],
    u_params: list[dict[str, torch.Tensor]],
    hyperparams: dict[str, torch.Tensor],
    y_test: np.ndarray,
) -> tuple[list[float], float]:
    """Return ``pack_gaussian_uq_list`` metrics and prediction wall time."""
    t0 = time.perf_counter()
    mu_vi, var_vi = _predict_direct_variational_consistent(
        regions, V_params, u_params, hyperparams,
    )
    pred_sec = float(time.perf_counter() - t0)
    mu_vi = torch.nan_to_num(mu_vi, nan=0.0)
    var_vi = torch.nan_to_num(var_vi, nan=1e-6).clamp_min(1e-12)
    sigma_vi = torch.sqrt(var_vi)
    pack = _metrics_pack(
        y_test,
        mu_vi.detach().cpu().numpy(),
        sigma_vi.detach().cpu().numpy(),
        pred_sec,
    )
    return pack, pred_sec


def _train_with_periodic_eval(
    *,
    regions: list[dict[str, torch.Tensor]],
    V_params: dict[str, torch.Tensor],
    u_params: list[dict[str, torch.Tensor]],
    hyperparams: dict[str, torch.Tensor],
    y_test: np.ndarray,
    num_steps: int,
    eval_every: int,
    lr: float,
    log_interval: int,
) -> list[dict[str, Any]]:
    """Original ELBO training loop with periodic direct-VI evaluation."""
    params = [V_params["mu_V"], V_params["sigma_V"]]
    for u in u_params:
        params += [
            u["U_logit"],
            u["mu_u"],
            u["Sigma_u"],
            u["sigma_noise"],
            u["omega"],
            u["sigma_k"],
        ]
    params += [
        hyperparams["lengthscales"],
        hyperparams["var_w"],
        hyperparams["Z"],
    ]
    optimizer = torch.optim.Adam(params, lr=float(lr))

    eval_steps = sorted(
        {s for s in range(int(eval_every), int(num_steps) + 1, int(eval_every))}
        | {int(num_steps)}
    )
    history: list[dict[str, Any]] = []

    for step in range(1, int(num_steps) + 1):
        optimizer.zero_grad()
        elbo = compute_ELBO(regions, V_params, u_params, hyperparams)
        (-elbo).backward()
        optimizer.step()

        with torch.no_grad():
            V_params["sigma_V"].clamp_(min=1e-1)
            hyperparams["var_w"].clamp_(min=1e-1)
            hyperparams["lengthscales"].clamp_(min=1e-6, max=1e2)
            for u in u_params:
                u["sigma_noise"].clamp_(min=1e-1)
                u["sigma_k"].clamp_(min=1e-1)

        elbo_val = float(elbo.detach().cpu().item())
        if step % int(log_interval) == 0 or step == 1:
            print(f"[train step {step}/{num_steps}] ELBO={elbo_val:.4f}", flush=True)

        if step in eval_steps:
            pack, pred_sec = _eval_direct_vi(
                regions, V_params, u_params, hyperparams, y_test,
            )
            row: dict[str, Any] = {
                "step": int(step),
                "elbo": elbo_val,
                "pred_sec": float(pred_sec),
            }
            row.update({k: float(pack[i]) for i, k in enumerate(RESULT_ROW_KEYS)})
            history.append(row)
            print(
                f"  [eval step {step}] RMSE={row['rmse']:.4f} "
                f"CRPS={row['mean_crps']:.4f} "
                f"Cov90={row.get('coverage_90', float('nan')):.3f}",
                flush=True,
            )

    return history


def _plot_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    steps = [int(r["step"]) for r in rows]
    rmse = [float(r["rmse"]) for r in rows]
    crps = [float(r["mean_crps"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, rmse, "o-", color="tab:blue", linewidth=2, markersize=7)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Direct VI predict — RMSE vs step")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, crps, "s-", color="tab:orange", linewidth=2, markersize=7)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("CRPS")
    axes[1].set_title("Direct VI predict — CRPS vs step")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Original ELBO training with periodic direct-VI RMSE/CRPS curve."
    )
    p.add_argument("--setting", default="lh_q5_train500", choices=list(SETTING_PRESETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=200, help="0 = use all test points")
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--num_steps", type=int, default=900)
    p.add_argument("--eval_every", type=int, default=150)
    p.add_argument("--log_interval", type=int, default=50, help="Print ELBO every N steps")
    p.add_argument(
        "--out_dir",
        default="experiments/synthetic/orig_elbo_direct_vi_curve_lh500_seed0",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SETTING_PRESETS[args.setting]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    _set_seed(args.seed)
    data = _generate_paper_data(cfg, args.seed, device)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data["X_train"])
    X_test_s = scaler.transform(data["X_test"])
    y_train = np.asarray(data["y_train"], dtype=np.float64)
    y_test = np.asarray(data["y_test"], dtype=np.float64)

    if args.max_test > 0 and args.max_test < len(y_test):
        X_test_s = X_test_s[: int(args.max_test)]
        y_test = y_test[: int(args.max_test)]

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)

    Q = int(cfg["Q"])
    n = int(cfg["n"])
    T = int(X_test_t.shape[0])

    regions = _build_regions(
        X_train_t, y_train_t, X_test_t, n=n, m1=int(args.m1), Q=Q, device=device,
    )
    V_params, hyperparams = _init_V_and_hyp(
        X_train_t, X_test_t, m2=int(args.m2), Q=Q, device=device,
    )
    u_params = _init_u_params(T, int(args.m1), Q, device)

    print(
        f"Training Original ELBO for {args.num_steps} steps; "
        f"eval direct VI every {args.eval_every} steps "
        f"(setting={args.setting}, seed={args.seed}, n_test={T})",
        flush=True,
    )
    t0 = time.perf_counter()
    history = _train_with_periodic_eval(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        y_test=y_test,
        num_steps=int(args.num_steps),
        eval_every=int(args.eval_every),
        lr=float(args.lr),
        log_interval=int(args.log_interval),
    )
    train_sec = float(time.perf_counter() - t0)

    _write_csv(out_dir / "curve_metrics.csv", history)
    _plot_curves(history, out_dir / "rmse_crps_vs_step.png")

    summary = {
        "args": vars(args),
        "device": str(device),
        "train_wall_sec": train_sec,
        "eval_steps": [int(r["step"]) for r in history],
        "final_rmse": float(history[-1]["rmse"]) if history else None,
        "final_crps": float(history[-1]["mean_crps"]) if history else None,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    readme = f"""# Original ELBO → Direct VI predict — training curve

- Setting: `{args.setting}`, seed={args.seed}
- Train steps: {args.num_steps}, eval every {args.eval_every}
- Predictor: `_predict_direct_variational_consistent` (ELBO-consistent direct VI)

## Files

- `curve_metrics.csv`: RMSE / CRPS / ELBO at each eval step
- `rmse_crps_vs_step.png`: RMSE and CRPS vs training step
- `summary.json`: run metadata

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"\nDone in {train_sec:.1f}s. Artifacts: {out_dir}", flush=True)
    print(f"  curve_metrics.csv  ({len(history)} eval points)", flush=True)
    print(f"  rmse_crps_vs_step.png", flush=True)


if __name__ == "__main__":
    main()

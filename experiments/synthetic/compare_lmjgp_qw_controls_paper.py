"""Compare learned LMJGP q(W) against projection-posterior controls.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_lmjgp_qw_controls_paper.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --seed 0 --out_dir experiments/synthetic/lmjgp_qw_controls_paper_train1000_seed0

The runner trains one transductive LMJGP state per paper-synthetic setting and
then evaluates several projection distributions with the same two-stage
prediction head:

    W_* samples -> projected JumpGP-CEM refit -> moment-matched mixture.

Controls:

- ``learned_qw``: the learned ``q(W_*)`` induced by ``q(V)``.
- ``prior_qw``: independent marginal prior samples ``N(0, var_w I)``.
- ``moment_matched_isotropic``: zero-mean isotropic Gaussian whose global
  coordinate second moment matches learned ``E_q[W^2]``.
- ``feature_permuted_learned_qw``: learned samples with a fixed feature-axis
  permutation applied to destroy learned feature alignment while preserving
  row norms and sample magnitudes.
- ``anchor_shuffled_learned_qw``: learned samples with a fixed test-anchor
  permutation applied to preserve the empirical distribution of ``W_j`` while
  breaking the learned ``x_j -> W_j`` assignment.
- ``global_mean_learned_qw``: learned samples averaged over test anchors and
  broadcast back to every anchor, removing anchor-specific variation while
  retaining Monte Carlo sample variation.
- ``zero_mean_learned_sigma``: zero mean with the learned diagonal posterior
  standard deviation, preserving variance anisotropy but removing mean.

All samples are row-normalized before JumpGP prediction, matching
``djgp.variational.predict_vi``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.projections import run_projected_jumpgp_ensemble  # noqa: E402
from djgp.variational import qW_from_qV  # noqa: E402
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import (  # noqa: E402
    _metrics_pack,
    _train_lmjgp_state,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


CONTROL_ORDER = (
    "learned_qw",
    "prior_qw",
    "moment_matched_isotropic",
    "feature_permuted_learned_qw",
    "anchor_shuffled_learned_qw",
    "global_mean_learned_qw",
    "zero_mean_learned_sigma",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting",
        "family",
        "seed",
        "control",
        "N_train",
        "N_test",
        "latent_dim",
        "observed_dim",
        "Q",
        "n",
        *RESULT_ROW_KEYS,
        "train_wall_sec",
        "pred_wall_sec",
        "sigma_w_mean",
        "mu_w_norm_mean",
        "matched_second_moment",
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


def _row_normalize(W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(eps)


def _sample_qw_controls(
    *,
    mu_W: torch.Tensor,
    cov_W: torch.Tensor,
    var_w: torch.Tensor,
    M: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Return row-normalized W samples for each q(W) control."""
    T, Q, D = mu_W.shape
    sigma_W = torch.sqrt(torch.diagonal(cov_W, dim1=-2, dim2=-1).clamp_min(1e-12))
    gen = torch.Generator(device=mu_W.device)
    gen.manual_seed(int(seed))
    eps = torch.randn((M, T, Q, D), device=mu_W.device, dtype=mu_W.dtype, generator=gen)

    learned = mu_W.unsqueeze(0) + sigma_W.unsqueeze(0) * eps

    var_scalar = torch.as_tensor(var_w, device=mu_W.device, dtype=mu_W.dtype).mean().clamp_min(1e-12)
    prior = torch.sqrt(var_scalar) * eps

    second = mu_W.pow(2) + sigma_W.pow(2)
    matched = second.mean().clamp_min(1e-12)
    iso = torch.sqrt(matched) * eps

    gen_perm = torch.Generator(device=mu_W.device)
    gen_perm.manual_seed(int(seed) + 999)
    perm = torch.randperm(D, device=mu_W.device, generator=gen_perm)
    permuted = learned[..., perm]

    gen_anchor = torch.Generator(device=mu_W.device)
    gen_anchor.manual_seed(int(seed) + 1777)
    anchor_perm = torch.randperm(T, device=mu_W.device, generator=gen_anchor)
    anchor_shuffled = learned[:, anchor_perm, :, :]

    global_mean = learned.mean(dim=1, keepdim=True).expand(-1, T, -1, -1)

    zero_mean_sigma = sigma_W.unsqueeze(0) * eps

    return {
        "learned_qw": _row_normalize(learned),
        "prior_qw": _row_normalize(prior),
        "moment_matched_isotropic": _row_normalize(iso),
        "feature_permuted_learned_qw": _row_normalize(permuted),
        "anchor_shuffled_learned_qw": _row_normalize(anchor_shuffled),
        "global_mean_learned_qw": _row_normalize(global_mean),
        "zero_mean_learned_sigma": _row_normalize(zero_mean_sigma),
    }


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
    regions, V_params, _u_params, hyperparams, train_sec = _train_lmjgp_state(
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
    with torch.no_grad():
        mu_W, cov_W = qW_from_qV(
            X_test_t,
            hyperparams["Z"],
            V_params["mu_V"],
            V_params["sigma_V"],
            hyperparams["lengthscales"],
            hyperparams["var_w"],
        )
        sigma_W = torch.sqrt(torch.diagonal(cov_W, dim1=-2, dim2=-1).clamp_min(1e-12))
        matched = float((mu_W.pow(2) + sigma_W.pow(2)).mean().detach().cpu().item())
        controls = _sample_qw_controls(
            mu_W=mu_W,
            cov_W=cov_W,
            var_w=hyperparams["var_w"],
            M=int(args.MC_num),
            seed=int(args.seed) + 1701,
        )

    X_nb_list = [r["X"] for r in regions]
    y_nb_list = [r["y"] for r in regions]
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
        "train_wall_sec": float(train_sec),
        "sigma_w_mean": float(sigma_W.mean().detach().cpu().item()),
        "mu_w_norm_mean": float(torch.linalg.norm(mu_W, dim=-1).mean().detach().cpu().item()),
        "matched_second_moment": matched,
    }
    for control in CONTROL_ORDER:
        print(f"predict setting={setting} control={control}", flush=True)
        out = run_projected_jumpgp_ensemble(
            controls[control],
            X_test_t,
            X_nb_list,
            y_nb_list,
            device=device,
            progress=False,
            desc=f"{setting}/{control}",
        )
        pack = _metrics_pack(y_test, out["mu"], out["sigma"], float(out["elapsed_sec"]))
        row = dict(base)
        row["control"] = control
        row["pred_wall_sec"] = float(out["elapsed_sec"])
        row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare LMJGP q(W) projection controls.")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out_dir", default="experiments/synthetic/lmjgp_qw_controls_paper_train1000_seed0")
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
    summary = {"args": vars(args), "device": str(device), "controls": CONTROL_ORDER, "n_rows": len(rows)}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    readme = f"""# LMJGP q(W) Controls on Paper Synthetic Data

This trains transductive LMJGP once per setting and evaluates learned
projection samples against prior/isotropic/permuted/zero-mean controls using
the same projected JumpGP-CEM ensemble head.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved {len(rows)} rows to {out_dir}", flush=True)
    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    main()

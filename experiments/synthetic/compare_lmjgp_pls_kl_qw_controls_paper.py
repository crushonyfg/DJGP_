"""Diagnose whether PLS-KL LMJGP learns nontrivial projection information.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_lmjgp_pls_kl_qw_controls_paper.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --seed 0 --out_dir experiments/synthetic/lmjgp_pls_kl_qw_controls_paper_train1000_seed0

This uses the paper-style L2/LH synthetic generators from
``compare_paper_synthetic_baselines.py`` and trains the same transductive
``pls_kl`` LMJGP variant used in the synthetic baseline tables.  After training,
it keeps the local neighborhoods fixed and evaluates prediction-time projection
controls through the same projected JumpGP-CEM ensemble head:

    W_* samples -> projected JumpGP-CEM refit -> moment-matched mixture.

Controls include learned PLS-KL ``q(W_*)``, prior/isotropic/feature-permuted/
anchor-shuffled/global-mean/zero-mean controls, and a deterministic frozen PLS
projection broadcast to every anchor.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.projections import run_projected_jumpgp_ensemble  # noqa: E402
from djgp.variational import qW_from_qV  # noqa: E402
from djgp.variational_tricks import KLWarmupSchedule, train_vi_with_tricks  # noqa: E402
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import _metrics_pack  # noqa: E402
from experiments.synthetic.compare_lmjgp_qw_controls_paper import (  # noqa: E402
    _row_normalize,
    _sample_qw_controls,
)
from experiments.synthetic.compare_lmjgp_tricks_synth import (  # noqa: E402
    _build_regions,
    _make_initial_params,
    _seed_all,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


CONTROL_ORDER = (
    "pls_kl_learned_qw",
    "prior_qw",
    "moment_matched_isotropic",
    "feature_permuted_learned_qw",
    "anchor_shuffled_learned_qw",
    "global_mean_learned_qw",
    "zero_mean_learned_sigma",
    "frozen_pls_static",
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
        "mean_abs_cos_muW_to_pls",
        "mean_anchor_muW_frob_std",
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


def _pls_W(X_train_s: np.ndarray, y_train: np.ndarray, Q: int) -> torch.Tensor:
    pls = PLSRegression(n_components=Q, scale=False)
    pls.fit(np.asarray(X_train_s), np.asarray(y_train).reshape(-1, 1))
    return torch.from_numpy(np.asarray(pls.x_weights_).T.astype(np.float32))


def _mean_abs_cos_muW_to_pls(mu_W: torch.Tensor, W_pls: torch.Tensor) -> float:
    W = W_pls.to(device=mu_W.device, dtype=mu_W.dtype)
    mu_n = mu_W / torch.linalg.norm(mu_W, dim=-1, keepdim=True).clamp_min(1e-8)
    pls_n = W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(1e-8)
    cos = (mu_n * pls_n.unsqueeze(0)).sum(dim=-1).abs()
    return float(cos.mean().detach().cpu().item())


def _train_pls_kl_state(
    *,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    Q: int,
    n: int,
    m1: int,
    m2: int,
    steps: int,
    lr: float,
    seed: int,
    device: torch.device,
):
    T, D = X_test_t.shape
    init_seed = int(seed) * 1019 + 11
    _seed_all(init_seed)
    regions, X_nb_list, y_nb_list = _build_regions(
        X_test_t, X_train_t, y_train_t, device, m1, Q, n
    )
    V_params, u_params, hyperparams, init_info = _make_initial_params(
        X_train_t,
        y_train_t,
        X_test_t,
        Q=Q,
        m1=m1,
        m2=m2,
        T=T,
        D=D,
        device=device,
        mu_V_mode="pls",
        init_seed=init_seed,
    )
    warm = KLWarmupSchedule(
        beta_init=0.01,
        beta_final=1.0,
        warmup_steps=max(1, int(steps * 0.5)),
        shape="linear",
    )
    t0 = time.perf_counter()
    V_params, u_params, hyperparams, history, _snapshots = train_vi_with_tricks(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=lr,
        num_steps=steps,
        log_interval=max(50, steps),
        snapshot_steps=[0, min(50, steps), steps],
        kl_warmup=warm,
        row_norm_weight=0.0,
        row_norm_mode="second_moment",
        aux_metric_weight=0.0,
        aux_metric_mode="A",
        aux_max_pairs=64,
        aux_seed=init_seed,
    )
    return {
        "regions": regions,
        "X_nb_list": X_nb_list,
        "y_nb_list": y_nb_list,
        "V_params": V_params,
        "u_params": u_params,
        "hyperparams": hyperparams,
        "init_info": init_info,
        "history_tail": history[-5:] if history else [],
        "train_wall_sec": time.perf_counter() - t0,
    }


def run_setting(args: argparse.Namespace, setting: str, device: torch.device) -> list[dict[str, Any]]:
    cfg = SETTING_PRESETS[setting]
    _set_seed(int(args.seed))
    data = _generate_paper_data(cfg, int(args.seed), device)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data["X_train"])
    X_test_s = scaler.transform(data["X_test"])
    y_train = np.asarray(data["y_train"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.float32)

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32)).to(device)

    Q = int(cfg["Q"])
    n = int(cfg["n"])
    print(f"[{setting}] train PLS-KL LMJGP", flush=True)
    state = _train_pls_kl_state(
        X_train_t=X_train_t,
        y_train_t=y_train_t,
        X_test_t=X_test_t,
        Q=Q,
        n=n,
        m1=int(args.m1),
        m2=int(args.m2),
        steps=int(args.lmjgp_steps),
        lr=float(args.lr),
        seed=int(args.seed),
        device=device,
    )

    with torch.no_grad():
        mu_W, cov_W = qW_from_qV(
            X_test_t,
            state["hyperparams"]["Z"],
            state["V_params"]["mu_V"],
            state["V_params"]["sigma_V"],
            state["hyperparams"]["lengthscales"],
            state["hyperparams"]["var_w"],
        )
        sigma_W = torch.sqrt(torch.diagonal(cov_W, dim1=-2, dim2=-1).clamp_min(1e-12))
        matched = float((mu_W.pow(2) + sigma_W.pow(2)).mean().detach().cpu().item())
        controls = _sample_qw_controls(
            mu_W=mu_W,
            cov_W=cov_W,
            var_w=state["hyperparams"]["var_w"],
            M=int(args.MC_num),
            seed=int(args.seed) + 2701,
        )
        controls["pls_kl_learned_qw"] = controls.pop("learned_qw")

        W_pls = _pls_W(X_train_s, y_train, Q).to(device=device, dtype=X_test_t.dtype)
        frozen = W_pls.unsqueeze(0).unsqueeze(0).expand(int(args.MC_num), X_test_t.shape[0], -1, -1)
        controls["frozen_pls_static"] = _row_normalize(frozen)

        mean_abs_cos = _mean_abs_cos_muW_to_pls(mu_W, W_pls)
        anchor_frob = torch.linalg.norm(mu_W, dim=(-2, -1))
        anchor_frob_std = float(anchor_frob.std().detach().cpu().item())

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
        "train_wall_sec": float(state["train_wall_sec"]),
        "sigma_w_mean": float(sigma_W.mean().detach().cpu().item()),
        "mu_w_norm_mean": float(torch.linalg.norm(mu_W, dim=-1).mean().detach().cpu().item()),
        "matched_second_moment": matched,
        "mean_abs_cos_muW_to_pls": mean_abs_cos,
        "mean_anchor_muW_frob_std": anchor_frob_std,
    }
    for control in CONTROL_ORDER:
        print(f"[{setting}] predict control={control}", flush=True)
        out = run_projected_jumpgp_ensemble(
            controls[control],
            X_test_t,
            state["X_nb_list"],
            state["y_nb_list"],
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
    p = argparse.ArgumentParser(description="Diagnose PLS-KL LMJGP q(W) controls.")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out_dir", default="experiments/synthetic/lmjgp_pls_kl_qw_controls_paper_train1000_seed0")
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
    summary = {"args": vars(args), "device": str(device), "controls": CONTROL_ORDER, "n_rows": len(rows)}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    readme = f"""# PLS-KL LMJGP q(W) Controls on Paper Synthetic Data

This trains the PLS-KL LMJGP variant once per paper-synthetic setting and
evaluates whether the learned projection samples outperform controls that
destroy feature alignment, anchor assignment, or the learned mean.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved {len(rows)} rows to {out_dir}", flush=True)
    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    main()

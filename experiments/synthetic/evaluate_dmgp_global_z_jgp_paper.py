"""Evaluate DeepMahalanobisGP global mapped features with JumpGP.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/evaluate_dmgp_global_z_jgp_paper.py ^
        --split_dir experiments/synthetic/deep_mahalanobis_gp_paper_train1000_5seeds ^
        --w_dir experiments/synthetic/deep_mahalanobis_gp_global_z_jgp_train1000_5seeds ^
        --out_dir experiments/synthetic/deep_mahalanobis_gp_global_z_jgp_train1000_5seeds ^
        --settings l2_q2_train1000,lh_q5_train1000 --seeds 0,1,2,3,4

This is different from ``evaluate_dmgp_w_with_jgp_paper.py``.  That script
uses each test anchor's ``W(x_*)`` as a local projection for its raw-X
neighborhood.  This script first maps the entire dataset pointwise,
``z_i = W(x_i)x_i``, then finds neighborhoods and fits JumpGP in the global
mapped ``Z`` space.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import _metrics_pack  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _aggregate,
    _metric_keys,
)
from shared.utils1 import jumpgp_ld_wrapper  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = _metric_keys(rows) if rows else []
    for extra in ("z_norm_mean", "z_norm_std", "w_train_anchor_frob_std", "w_test_anchor_frob_std"):
        if any(extra in row for row in rows) and extra not in keys:
            keys.append(extra)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _infer_n(setting: str) -> int:
    cfg = SETTING_PRESETS.get(setting, {})
    return int(cfg.get("n", 25 if setting.startswith("l2_") else 35))


def _map_global_z(X: np.ndarray, W: np.ndarray, *, row_normalize_w: bool = False) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 3 or W.shape[0] != X.shape[0] or W.shape[2] != X.shape[1]:
        raise ValueError(f"Expected W [N,Q,D] aligned with X [N,D], got W={W.shape}, X={X.shape}.")
    if row_normalize_w:
        norms = np.linalg.norm(W, axis=-1, keepdims=True)
        W = W / np.maximum(norms, 1e-8)
    return np.einsum("nqd,nd->nq", W, X)


def _run_global_z_jgp(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n: int,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    Z_train_t = torch.as_tensor(Z_train, dtype=torch.float32, device=device)
    Z_test_t = torch.as_tensor(Z_test, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    neighborhoods = find_neighborhoods(Z_test_t.cpu(), Z_train_t.cpu(), y_train_t.cpu(), M=n)

    mus = np.empty(Z_test.shape[0], dtype=np.float64)
    sigmas = np.empty(Z_test.shape[0], dtype=np.float64)
    t0 = time.perf_counter()
    for j, nb in enumerate(tqdm(neighborhoods, desc="global-Z JumpGP", leave=False)):
        Z_nb = nb["X_neighbors"].to(device)
        y_nb = nb["y_neighbors"].to(device).view(-1, 1)
        z_star = Z_test_t[j].view(1, -1)
        mu_j, sig2_j, _, _ = jumpgp_ld_wrapper(
            Z_nb,
            y_nb,
            z_star,
            mode="CEM",
            flag=False,
            device=device,
        )
        mus[j] = float(mu_j.detach().cpu().reshape(-1)[0].item())
        sigmas[j] = float(torch.sqrt(torch.clamp(sig2_j, min=1e-12)).detach().cpu().reshape(-1)[0].item())
    wall_sec = time.perf_counter() - t0
    pack = _metrics_pack(y_test, mus, sigmas, wall_sec)
    return {key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)}, wall_sec


def run_one(args: argparse.Namespace, setting: str, seed: int, device: torch.device) -> dict[str, Any]:
    split_dir = Path(args.split_dir)
    w_dir = Path(args.w_dir)
    arr = np.load(split_dir / f"{setting}_seed{seed}.npz")
    X_train = np.asarray(arr["X_train"], dtype=np.float64)
    X_test = np.asarray(arr["X_test"], dtype=np.float64)
    y_train = np.asarray(arr["y_train"], dtype=np.float64).reshape(-1)
    y_test = np.asarray(arr["y_test"], dtype=np.float64).reshape(-1)
    W_train = np.load(w_dir / f"{setting}_seed{seed}_W_train_mean.npy")
    W_test = np.load(w_dir / f"{setting}_seed{seed}_W_test_mean.npy")

    Z_train = _map_global_z(X_train, W_train, row_normalize_w=bool(args.row_normalize_w))
    Z_test = _map_global_z(X_test, W_test, row_normalize_w=bool(args.row_normalize_w))
    n = int(args.n) if int(args.n) > 0 else _infer_n(setting)
    metrics, _ = _run_global_z_jgp(Z_train, y_train, Z_test, y_test, n=n, device=device)

    cfg = dict(SETTING_PRESETS.get(setting, {}))
    method = "dmgp_global_z_mean_jgp"
    if args.row_normalize_w:
        method += "_rownormW"
    row: dict[str, Any] = {
        "setting": setting,
        "family": cfg.get("family", ""),
        "expansion": cfg.get("expansion", ""),
        "caseno": cfg.get("caseno", ""),
        "seed": int(seed),
        "method": method,
        "N_train": int(X_train.shape[0]),
        "N_test": int(X_test.shape[0]),
        "latent_dim": int(cfg.get("d", W_train.shape[1])),
        "observed_dim": int(X_train.shape[1]),
        "Q_true": int(cfg.get("Q_true", W_train.shape[1])),
        "Q": int(W_train.shape[1]),
        "n": int(n),
        "standardize_x": True,
        "standardize_y_for_dmgp_training": True,
        "row_normalize_w": bool(args.row_normalize_w),
        "noise_std": cfg.get("noise_std", ""),
        "noise_var": cfg.get("noise_var", ""),
        "lengthscale": cfg.get("lengthscale", ""),
        "kernel_var": cfg.get("kernel_var", ""),
        "status": "ok",
        "z_norm_mean": float(np.linalg.norm(Z_train, axis=1).mean()),
        "z_norm_std": float(np.linalg.norm(Z_train, axis=1).std()),
        "w_train_anchor_frob_std": float(np.linalg.norm(W_train, axis=(1, 2)).std()),
        "w_test_anchor_frob_std": float(np.linalg.norm(W_test, axis=(1, 2)).std()),
    }
    row.update(metrics)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate global z_i=W(x_i)x_i DeepMahalanobisGP features with JumpGP.")
    p.add_argument("--split_dir", default="experiments/synthetic/deep_mahalanobis_gp_paper_train1000_5seeds")
    p.add_argument("--w_dir", default="experiments/synthetic/deep_mahalanobis_gp_global_z_jgp_train1000_5seeds")
    p.add_argument("--out_dir", default="experiments/synthetic/deep_mahalanobis_gp_global_z_jgp_train1000_5seeds")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="", help="Optional comma-separated seed list. Overrides --seed when set.")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--row_normalize_w", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "global_z_jgp_metrics.csv"
    rows: list[dict[str, Any]] = []
    if args.resume and metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {(str(r.get("setting")), int(r.get("seed", -1)), str(r.get("method"))) for r in rows}

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [int(args.seed)]
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = "dmgp_global_z_mean_jgp" + ("_rownormW" if args.row_normalize_w else "")
    for seed in seeds:
        for setting in settings:
            if (setting, seed, method) in done:
                print(f"[skip] {setting} seed={seed} method={method}", flush=True)
                continue
            print(f"[run] {setting} seed={seed} method={method}", flush=True)
            try:
                row = run_one(args, setting, seed, device)
            except Exception as exc:  # noqa: BLE001
                row = {"setting": setting, "seed": seed, "method": method, "status": "failed", "error": repr(exc)}
            rows.append(row)
            _write_csv(metrics_path, rows)
            _write_csv(out_dir / "global_z_jgp_aggregate.csv", _aggregate(rows))

    summary = {"args": vars(args), "device": str(device), "n_rows": len(rows)}
    (out_dir / "global_z_jgp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(metrics_path, rows)
    _write_csv(out_dir / "global_z_jgp_aggregate.csv", _aggregate(rows))
    readme = f"""# DeepMahalanobisGP Global Z JumpGP

This folder evaluates global pointwise mapped features from DeepMahalanobisGP:
`z_i = W(x_i)x_i`, followed by ordinary JumpGP in the mapped `Z` space.

It is not the local per-anchor `W(x_*)` projection diagnostic.

```json
{json.dumps(summary, indent=2)}
```

Files:

- `global_z_jgp_metrics.csv`: per-seed metrics.
- `global_z_jgp_aggregate.csv`: mean/std over successful seeds.
- `*_W_train_mean.npy`, `*_W_test_mean.npy`: exported by
  `run_deep_mahalanobis_gp_paper.py` during the matching DMGP retrain.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

"""Evaluate DeepMahalanobisGP learned W with the DJGP projected JumpGP head.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/evaluate_dmgp_w_with_jgp_paper.py ^
        --data_dir experiments/synthetic/deep_mahalanobis_gp_paper_train1000_seed0 ^
        --settings l2_q2_train1000,lh_q5_train1000 --seed 0

This script loads exported paper-style synthetic data plus ``W`` arrays saved by
``run_deep_mahalanobis_gp_paper.py``.  It evaluates the external model's
test-side ``W`` through our standard projected JumpGP-CEM refit head and adds
simple controls that destroy feature alignment or anchor assignment.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from djgp.projections import run_projected_jumpgp_ensemble  # noqa: E402
from experiments.synthetic.compare_lmjgp_prediction_heads_paper import _metrics_pack  # noqa: E402


CONTROL_ORDER = (
    "dmgp_W_mean_jgp",
    "dmgp_W_samples_jgp",
    "dmgp_W_mean_feature_permuted",
    "dmgp_W_mean_anchor_shuffled",
    "dmgp_W_mean_global_mean",
    "random_isotropic_jgp",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    preferred = [
        "setting",
        "seed",
        "method",
        "N_train",
        "N_test",
        "Q",
        "D",
        "n",
        *RESULT_ROW_KEYS,
        "W_mean_norm_mean",
        "W_mean_anchor_frob_std",
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


def _row_normalize(W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return W / torch.linalg.norm(W, dim=-1, keepdim=True).clamp_min(eps)


def _infer_n(setting: str) -> int:
    return 25 if setting.startswith("l2_") else 35


def _make_controls(W_mean: torch.Tensor, W_samples: torch.Tensor | None, seed: int) -> dict[str, torch.Tensor]:
    T, Q, D = W_mean.shape
    M = 3 if W_samples is None else int(W_samples.shape[0])
    base = W_mean.unsqueeze(0).expand(M, -1, -1, -1).clone()
    if W_samples is not None:
        sample = W_samples
    else:
        sample = base

    gen = torch.Generator(device=W_mean.device)
    gen.manual_seed(int(seed) + 5001)
    perm = torch.randperm(D, device=W_mean.device, generator=gen)
    feature_perm = base[..., perm]

    gen_anchor = torch.Generator(device=W_mean.device)
    gen_anchor.manual_seed(int(seed) + 5007)
    anchor_perm = torch.randperm(T, device=W_mean.device, generator=gen_anchor)
    anchor_shuffle = base[:, anchor_perm, :, :]

    global_mean = base.mean(dim=1, keepdim=True).expand(-1, T, -1, -1)

    gen_rand = torch.Generator(device=W_mean.device)
    gen_rand.manual_seed(int(seed) + 5011)
    random = torch.randn((M, T, Q, D), device=W_mean.device, dtype=W_mean.dtype, generator=gen_rand)

    return {
        "dmgp_W_mean_jgp": _row_normalize(base),
        "dmgp_W_samples_jgp": _row_normalize(sample),
        "dmgp_W_mean_feature_permuted": _row_normalize(feature_perm),
        "dmgp_W_mean_anchor_shuffled": _row_normalize(anchor_shuffle),
        "dmgp_W_mean_global_mean": _row_normalize(global_mean),
        "random_isotropic_jgp": _row_normalize(random),
    }


def run_setting(args: argparse.Namespace, setting: str, device: torch.device) -> list[dict[str, Any]]:
    data_dir = Path(args.data_dir)
    arr = np.load(data_dir / f"{setting}_seed{args.seed}.npz")
    X_train = np.asarray(arr["X_train"], dtype=np.float32)
    X_test = np.asarray(arr["X_test"], dtype=np.float32)
    y_train = np.asarray(arr["y_train"], dtype=np.float32).reshape(-1)
    y_test = np.asarray(arr["y_test"], dtype=np.float64).reshape(-1)
    W_mean_np = np.load(data_dir / f"{setting}_seed{args.seed}_W_mean.npy")
    W_samples_path = data_dir / f"{setting}_seed{args.seed}_W_samples.npy"
    W_samples_np = np.load(W_samples_path) if W_samples_path.exists() else None

    X_train_t = torch.from_numpy(X_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    W_mean = torch.from_numpy(W_mean_np.astype(np.float32)).to(device)
    W_samples = None if W_samples_np is None else torch.from_numpy(W_samples_np.astype(np.float32)).to(device)

    n = int(args.n) if int(args.n) > 0 else _infer_n(setting)
    neighborhoods = find_neighborhoods(X_test_t.cpu(), X_train_t.cpu(), y_train_t.cpu(), M=n)
    X_nb_list = [nb["X_neighbors"].to(device) for nb in neighborhoods]
    y_nb_list = [nb["y_neighbors"].to(device) for nb in neighborhoods]

    controls = _make_controls(W_mean, W_samples, int(args.seed))
    rows: list[dict[str, Any]] = []
    base = {
        "setting": setting,
        "seed": int(args.seed),
        "N_train": int(X_train.shape[0]),
        "N_test": int(X_test.shape[0]),
        "Q": int(W_mean.shape[1]),
        "D": int(W_mean.shape[2]),
        "n": n,
        "W_mean_norm_mean": float(torch.linalg.norm(W_mean, dim=-1).mean().detach().cpu().item()),
        "W_mean_anchor_frob_std": float(torch.linalg.norm(W_mean, dim=(-2, -1)).std().detach().cpu().item()),
    }
    for method in CONTROL_ORDER:
        if method == "dmgp_W_samples_jgp" and W_samples is None:
            continue
        print(f"[{setting}] projected JGP method={method}", flush=True)
        out = run_projected_jumpgp_ensemble(
            controls[method],
            X_test_t,
            X_nb_list,
            y_nb_list,
            device=device,
            progress=False,
            desc=f"{setting}/{method}",
        )
        pack = _metrics_pack(y_test, out["mu"], out["sigma"], float(out["elapsed_sec"]))
        row = dict(base)
        row["method"] = method
        row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DeepMahalanobisGP W with projected JumpGP.")
    p.add_argument("--data_dir", default="experiments/synthetic/deep_mahalanobis_gp_paper_train1000_seed0")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=0)
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    out_dir = Path(args.data_dir)
    for setting in [s.strip() for s in args.settings.split(",") if s.strip()]:
        rows.extend(run_setting(args, setting, device))
        _write_csv(out_dir / "dmgp_w_jgp_metrics_partial.csv", rows)
    _write_csv(out_dir / "dmgp_w_jgp_metrics.csv", rows)
    summary = {"args": vars(args), "device": str(device), "n_rows": len(rows)}
    (out_dir / "dmgp_w_jgp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

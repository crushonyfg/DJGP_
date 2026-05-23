"""Export paper-style synthetic splits for DeepMahalanobisGP runs.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/export_paper_synthetic_for_dmgp.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --seed 0 --out_dir experiments/synthetic/deep_mahalanobis_gp_paper_train1000_seed0

    python experiments/synthetic/export_paper_synthetic_for_dmgp.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --seeds 0,1,2,3,4 --out_dir experiments/synthetic/deep_mahalanobis_gp_paper_train1000_5seeds

The exported ``.npz`` files contain the same paper-style RFF-expanded L2/LH
synthetic data used by ``compare_paper_synthetic_baselines.py``.  X is
standardized with train-only statistics.  Y is also standardized for the
external DeepMahalanobisGP model because its original notebook standardizes the
response before training; the original-scale Y and scaler parameters are saved
so metrics can be reported on the original scale.
"""

from __future__ import annotations

import argparse
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

from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)


def export_setting(setting: str, seed: int, out_dir: Path, device: torch.device) -> dict[str, Any]:
    cfg = dict(SETTING_PRESETS[setting])
    _set_seed(seed)
    data = _generate_paper_data(cfg, seed, device)
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(data["X_train"])
    X_test_s = x_scaler.transform(data["X_test"])
    y_scaler = StandardScaler()
    y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1, 1)
    y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1, 1)
    y_train_s = y_scaler.fit_transform(y_train)
    y_test_s = y_scaler.transform(y_test)

    path = out_dir / f"{setting}_seed{seed}.npz"
    np.savez_compressed(
        path,
        X_train=X_train_s.astype(np.float64),
        X_test=X_test_s.astype(np.float64),
        y_train=y_train.reshape(-1).astype(np.float64),
        y_test=y_test.reshape(-1).astype(np.float64),
        y_train_std=y_train_s.astype(np.float64),
        y_test_std=y_test_s.astype(np.float64),
        z_train=np.asarray(data["z_train"], dtype=np.float64),
        z_test=np.asarray(data["z_test"], dtype=np.float64),
        x_mean=x_scaler.mean_.astype(np.float64),
        x_scale=x_scaler.scale_.astype(np.float64),
        y_mean=y_scaler.mean_.astype(np.float64),
        y_scale=y_scaler.scale_.astype(np.float64),
    )
    return {
        "setting": setting,
        "seed": seed,
        "path": str(path),
        "family": cfg["family"],
        "N_train": int(cfg["N_train"]),
        "N_test": int(cfg["N_test"]),
        "latent_dim": int(cfg["d"]),
        "observed_dim": int(cfg["H"]),
        "Q": int(cfg["Q"]),
        "n": int(cfg["n"]),
        "standardize_x": True,
        "standardize_y_for_dmgp": True,
        "expansion": cfg.get("expansion"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export paper synthetic splits for DeepMahalanobisGP.")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="", help="Optional comma-separated seed list. Overrides --seed when set.")
    p.add_argument("--out_dir", default="experiments/synthetic/deep_mahalanobis_gp_paper_train1000_seed0")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [int(args.seed)]
    unknown = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    rows = [export_setting(s, seed, out_dir, device) for seed in seeds for s in settings]
    summary = {"args": vars(args), "device": str(device), "exports": rows}
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

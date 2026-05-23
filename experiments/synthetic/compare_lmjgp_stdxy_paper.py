"""Run LMJGP baseline/PLS-KL on paper synthetic data with X and Y standardized.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_lmjgp_stdxy_paper.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --num_exp 5 --out_dir experiments/synthetic/paper_l2_lh_stdxy_lmjgp_train1000_5seeds

This is a small compatibility run for comparing against DeepMahalanobisGP,
whose original notebook standardizes both X and Y.  The existing paper
synthetic LMJGP/CEM table standardized X only.  Here, LMJGP trains and predicts
on standardized Y; RMSE, CRPS, and interval widths are scaled back to original
Y units before writing CSV rows, while coverage is unchanged by the affine
inverse transform.
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

from djgp.evaluation.calibration import RESULT_ROW_KEYS  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _run_lmjgp_baseline,
    _run_lmjgp_pls_kl,
    _set_seed,
)


METHODS = ("lmjgp_baseline", "lmjgp_pls_kl")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "setting",
        "family",
        "seed",
        "method",
        "status",
        "N_train",
        "N_test",
        "latent_dim",
        "observed_dim",
        "Q_true",
        "Q",
        "n",
        "standardize_x",
        "standardize_y",
        "y_scale",
        *RESULT_ROW_KEYS,
        "error",
    ]
    keys = [k for k in preferred if any(k in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _scale_pack_from_std_y(pack: list[float], y_scale: float) -> list[float]:
    out = list(pack)
    scale = float(y_scale)
    for idx in (0, 1, 4, 6):
        out[idx] = float(out[idx]) * scale
    return out


def _write_aggregate(path: Path, rows: list[dict[str, Any]]) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    metrics = ["rmse", "mean_crps", "coverage_90", "mean_width_90", "coverage_95", "mean_width_95", "wall_sec"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ok:
        groups.setdefault((str(row["setting"]), str(row["method"])), []).append(row)
    out_rows: list[dict[str, Any]] = []
    for (setting, method), items in sorted(groups.items()):
        agg: dict[str, Any] = {
            "setting": setting,
            "method": method,
            "n_success": len(items),
            "seeds": ",".join(str(int(row["seed"])) for row in items),
        }
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in items], dtype=np.float64)
            agg[f"{metric}_mean"] = float(vals.mean())
            agg[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out_rows.append(agg)
    _write_csv(path, out_rows)


def run_one(args: argparse.Namespace, setting: str, seed: int, method: str, device: torch.device) -> dict[str, Any]:
    cfg = dict(SETTING_PRESETS[setting])
    _set_seed(seed)
    data = _generate_paper_data(cfg, seed, device)
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(data["X_train"])
    X_test_s = x_scaler.transform(data["X_test"])
    y_scaler = StandardScaler()
    y_train_std = y_scaler.fit_transform(np.asarray(data["y_train"], dtype=np.float64).reshape(-1, 1)).reshape(-1)
    y_test_std = y_scaler.transform(np.asarray(data["y_test"], dtype=np.float64).reshape(-1, 1)).reshape(-1)
    y_scale = float(y_scaler.scale_[0])

    row: dict[str, Any] = {
        "setting": setting,
        "family": cfg["family"],
        "expansion": cfg["expansion"],
        "caseno": cfg["caseno"],
        "seed": int(seed),
        "method": method,
        "N_train": int(cfg["N_train"]),
        "N_test": int(cfg["N_test"]),
        "latent_dim": int(cfg["d"]),
        "observed_dim": int(cfg["H"]),
        "Q_true": int(cfg["Q_true"]),
        "Q": int(cfg["Q"]),
        "n": int(cfg["n"]),
        "standardize_x": True,
        "standardize_y": True,
        "y_scale": y_scale,
        "lmjgp_steps": int(args.lmjgp_steps),
        "MC_num": int(args.MC_num),
    }
    try:
        if method == "lmjgp_baseline":
            pack_std = _run_lmjgp_baseline(
                X_train_s,
                y_train_std,
                X_test_s,
                y_test_std,
                Q=int(cfg["Q"]),
                n_neighbors=int(cfg["n"]),
                steps=int(args.lmjgp_steps),
                MC_num=int(args.MC_num),
                device=device,
            )
        elif method == "lmjgp_pls_kl":
            pack_std = _run_lmjgp_pls_kl(
                X_train_s,
                y_train_std,
                X_test_s,
                y_test_std,
                seed=seed,
                Q=int(cfg["Q"]),
                n_neighbors=int(cfg["n"]),
                steps=int(args.lmjgp_steps),
                MC_num=int(args.MC_num),
                device=device,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        pack = _scale_pack_from_std_y(pack_std, y_scale)
        row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR setting={setting} seed={seed} method={method}: {row['error']}", flush=True)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paper synthetic LMJGP variants with std_x and std_y.")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--num_exp", type=int, default=5)
    p.add_argument("--methods", default="lmjgp_baseline,lmjgp_pls_kl")
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--out_dir", default="experiments/synthetic/paper_l2_lh_stdxy_lmjgp_train1000_5seeds")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_settings:
        raise ValueError(f"Unknown settings: {unknown_settings}")
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    metrics_path = out_dir / "metrics.csv"
    if args.resume and metrics_path.exists():
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    completed = {
        (str(row.get("setting")), int(row.get("seed")), str(row.get("method")))
        for row in rows
        if row.get("status") == "ok" and str(row.get("seed", "")).strip()
    }

    for seed in range(int(args.num_exp)):
        for setting in settings:
            for method in methods:
                if (setting, seed, method) in completed:
                    print(f"skip completed setting={setting} seed={seed} method={method}", flush=True)
                    continue
                print(f"setting={setting} seed={seed} method={method}", flush=True)
                rows.append(run_one(args, setting, seed, method, device))
                _write_csv(out_dir / "metrics_partial.csv", rows)
                _write_csv(out_dir / "metrics.csv", rows)
                _write_aggregate(out_dir / "aggregate.csv", rows)

    _write_csv(out_dir / "metrics.csv", rows)
    _write_aggregate(out_dir / "aggregate.csv", rows)
    summary = {"args": vars(args), "device": str(device), "n_rows": len(rows)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = f"""# LMJGP Paper Synthetic with X/Y Standardization

This run trains `lmjgp_baseline` and `lmjgp_pls_kl` on standardized X and
standardized Y for paper-style L2/LH train=1000 synthetic data.  Metrics are
reported on the original Y scale.

See `docs/experiments_log.md` for the headline table.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

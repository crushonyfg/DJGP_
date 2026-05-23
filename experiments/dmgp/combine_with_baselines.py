"""Combine DeepMahalanobisGP aggregate rows with existing baseline aggregates.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/dmgp/combine_with_baselines.py ^
        --baseline experiments/uci/smalltrain_multiseed_bestpreset_summary/aggregate_train200_500_1000_bestpresets.csv ^
        --dmgp experiments/uci/dmgp_smalltrain_5seeds_20260517/aggregate.csv ^
        --out_dir experiments/uci/dmgp_smalltrain_5seeds_20260517/combined

    python experiments/dmgp/combine_with_baselines.py ^
        --baseline experiments/pdebench/burgers_train200_500_1000_xgb_dkl_lmjgp_5seeds_20260517/aggregate.csv ^
        --dmgp experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517/aggregate.csv ^
        --out_dir experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517/combined
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


METRIC_MEAN_KEYS = (
    "rmse_mean",
    "mean_crps_mean",
    "coverage_90_mean",
    "mean_width_90_mean",
    "coverage_95_mean",
    "mean_width_95_mean",
    "wall_sec_mean",
)
METRIC_STD_KEYS = (
    "rmse_std",
    "mean_crps_std",
    "coverage_90_std",
    "mean_width_90_std",
    "coverage_95_std",
    "mean_width_95_std",
    "wall_sec_std",
)


def _read_csv(path: Path, source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out = dict(row)
            out.setdefault("subset", "all")
            if not out.get("subset"):
                out["subset"] = "all"
            out.setdefault("benchmark", "")
            out.setdefault("dataset", "")
            out.setdefault("train_size", "")
            out.setdefault("source", source)
            if not out.get("source"):
                out["source"] = source
            rows.append(out)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    preferred = [
        "benchmark",
        "dataset",
        "train_size",
        "subset",
        "method",
        "n_seeds",
        "seeds",
        *METRIC_MEAN_KEYS,
        *METRIC_STD_KEYS,
        "source",
        "best_rmse_method",
        "best_rmse",
        "best_crps_method",
        "best_crps",
        "dmgp_rmse_delta_vs_best",
        "dmgp_crps_delta_vs_best",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _float(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def _best_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("train_size", "")), str(row.get("subset", "all")))
        groups.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, train_size, subset), items in sorted(groups.items()):
        rmse_items = [r for r in items if np.isfinite(_float(r, "rmse_mean"))]
        crps_items = [r for r in items if np.isfinite(_float(r, "mean_crps_mean"))]
        best_rmse = min(rmse_items, key=lambda r: _float(r, "rmse_mean")) if rmse_items else None
        best_crps = min(crps_items, key=lambda r: _float(r, "mean_crps_mean")) if crps_items else None
        dmgp = next((r for r in items if r.get("method") == "deep_mahalanobis_gp"), None)
        row: Dict[str, Any] = {
            "dataset": dataset,
            "train_size": train_size,
            "subset": subset,
            "best_rmse_method": best_rmse.get("method") if best_rmse else "",
            "best_rmse": _float(best_rmse, "rmse_mean") if best_rmse else "",
            "best_crps_method": best_crps.get("method") if best_crps else "",
            "best_crps": _float(best_crps, "mean_crps_mean") if best_crps else "",
        }
        if dmgp is not None and best_rmse is not None:
            row["dmgp_rmse_delta_vs_best"] = _float(dmgp, "rmse_mean") - _float(best_rmse, "rmse_mean")
        if dmgp is not None and best_crps is not None:
            row["dmgp_crps_delta_vs_best"] = _float(dmgp, "mean_crps_mean") - _float(best_crps, "mean_crps_mean")
        out.append(row)
    return out


def _write_readme(out_dir: Path, combined: List[Dict[str, Any]], best: List[Dict[str, Any]]) -> None:
    lines = [
        "# DeepMahalanobisGP Combined Comparison",
        "",
        "Produced by `experiments/dmgp/combine_with_baselines.py`.",
        "",
        "## Best Methods",
        "",
        "| dataset | train | subset | best RMSE | RMSE | best CRPS | CRPS | DMGP RMSE delta | DMGP CRPS delta |",
        "|---|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row.get('dataset', '')} | {row.get('train_size', '')} | {row.get('subset', 'all')} | "
            f"{row.get('best_rmse_method', '')} | {float(row['best_rmse']):.4f} | "
            f"{row.get('best_crps_method', '')} | {float(row['best_crps']):.4f} | "
            f"{float(row.get('dmgp_rmse_delta_vs_best', 0.0)):.4f} | "
            f"{float(row.get('dmgp_crps_delta_vs_best', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `aggregate_with_dmgp.csv`: all aggregate rows.",
            "- `best_by_metric.csv`: best RMSE/CRPS method per dataset/train/subset.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine DeepMahalanobisGP and baseline aggregate CSV files.")
    p.add_argument("--baseline", required=True)
    p.add_argument("--dmgp", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    baseline = _read_csv(Path(args.baseline), "baseline")
    dmgp = _read_csv(Path(args.dmgp), "deep_mahalanobis_gp")
    combined = baseline + dmgp
    best = _best_rows(combined)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "aggregate_with_dmgp.csv", combined)
    _write_csv(out_dir / "best_by_metric.csv", best)
    (out_dir / "summary.json").write_text(
        json.dumps({"baseline": args.baseline, "dmgp": args.dmgp, "n_rows": len(combined)}, indent=2),
        encoding="utf-8",
    )
    _write_readme(out_dir, combined, best)
    print(f"Saved combined comparison -> {out_dir}", flush=True)
    return {"combined": combined, "best": best}


if __name__ == "__main__":
    main()

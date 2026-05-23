"""Aggregate the Parkinsons CEM-EM grid into a markdown table."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_tag(name: str) -> tuple[float, float]:
    m = re.match(r"a(\d+)_l(\d+)p(\d+)", name)
    if not m:
        raise ValueError(f"bad tag: {name}")
    alpha = float(m.group(1))
    lam = float(f"{m.group(2)}.{m.group(3)}")
    return alpha, lam


def _row_from_metrics(metrics: list[float]) -> dict[str, float]:
    keys = ["rmse", "crps", "wall", "cov90", "w90", "cov95", "w95"]
    out = {}
    for k, v in zip(keys, metrics):
        out[k] = float(v) if v is not None else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="experiments/uci/cem_em_park_grid")
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for jpath in sorted(root.glob("a*_l*.json")):
        tag = jpath.stem
        try:
            alpha, lam = _parse_tag(tag)
        except ValueError:
            continue
        data = json.loads(jpath.read_text())
        # JSON shape: { dataset: { seed: {...} } } or top-level depending on writer.
        if "Parkinsons Telemonitoring" in data:
            container = data["Parkinsons Telemonitoring"]
        else:
            container = data
        # take seed 0
        seed_keys = [k for k in container.keys() if str(k).isdigit() or k == "0" or k == 0]
        if not seed_keys:
            seed_keys = list(container.keys())
        seed_block = container[seed_keys[0]]
        methods = seed_block.get("methods", seed_block.get("metrics", {}))
        diag = seed_block.get("diagnostics", {})
        pls = _row_from_metrics(methods.get("pls_only", [float("nan")] * 7))
        coeff = _row_from_metrics(methods.get("cem_em_coeff", [float("nan")] * 7))
        cdiag = diag.get("cem_em_coeff", {}) if isinstance(diag, dict) else {}
        rows.append({
            "tag": tag,
            "alpha": alpha,
            "lam": lam,
            "pls_rmse": pls["rmse"],
            "coeff_rmse": coeff["rmse"],
            "rmse_drop_%": (1 - coeff["rmse"] / pls["rmse"]) * 100 if pls["rmse"] > 0 else float("nan"),
            "pls_crps": pls["crps"],
            "coeff_crps": coeff["crps"],
            "crps_drop_%": (1 - coeff["crps"] / pls["crps"]) * 100 if pls["crps"] > 0 else float("nan"),
            "coeff_cov95": coeff["cov95"],
            "coeff_w95": coeff["w95"],
            "anchor_disp": cdiag.get("anchor_dispersion_frob", float("nan")),
            "eff_rank": cdiag.get("median_eff_rank_WWT", float("nan")),
            "cos_to_pls": cdiag.get("cosine_to_ref_median", float("nan")),
            "best_outer": cdiag.get("best_outer", -1),
            "best_loss": cdiag.get("best_metric_value", float("nan")),
            "train_sec": cdiag.get("train_time_sec", float("nan")),
        })

    rows.sort(key=lambda r: (r["alpha"], r["lam"]))
    cols = ["tag", "alpha", "lam", "pls_rmse", "coeff_rmse", "rmse_drop_%",
            "pls_crps", "coeff_crps", "crps_drop_%",
            "coeff_cov95", "coeff_w95", "anchor_disp", "eff_rank", "cos_to_pls",
            "best_outer", "best_loss", "train_sec"]
    # Markdown
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")

    # Best by RMSE and CRPS
    if rows:
        best_rmse = min(rows, key=lambda r: r["coeff_rmse"])
        best_crps = min(rows, key=lambda r: r["coeff_crps"])
        print(f"\nBest RMSE: {best_rmse['tag']}  -> {best_rmse['coeff_rmse']:.3f} (drop {best_rmse['rmse_drop_%']:.1f}%)")
        print(f"Best CRPS: {best_crps['tag']}  -> {best_crps['coeff_crps']:.3f} (drop {best_crps['crps_drop_%']:.1f}%)")


if __name__ == "__main__":
    main()

"""Aggregate 3-way 10-seed results into mean ± std markdown tables.

Usage::

    python experiments/uci/compare_3way_10seeds/summarize.py \
        --csv experiments/uci/compare_3way_10seeds/results.csv

Produces a markdown table per (dataset, metric) pair.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="experiments/uci/compare_3way_10seeds/results.csv")
    ap.add_argument("--metrics", type=str, default="rmse,crps,wall_sec,cov95,w95")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows: list[dict] = []
    with Path(args.csv).open("r") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                r["value"] = float(r["value"])
            except ValueError:
                r["value"] = float("nan")
            rows.append(r)

    by_dsmm: dict = defaultdict(list)
    for r in rows:
        by_dsmm[(r["dataset"], r["method"], r["metric"])].append(r["value"])

    methods_seen = sorted({r["method"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})

    method_order_pref = ["xgb_bootstrap", "dkl_svgp", "pls_only", "cem_em_coeff"]
    methods = [m for m in method_order_pref if m in methods_seen] + \
              [m for m in methods_seen if m not in method_order_pref]

    def _mean_std(values):
        clean = [v for v in values if v is not None and not math.isnan(v)]
        if not clean:
            return float("nan"), float("nan"), 0
        m = sum(clean) / len(clean)
        if len(clean) > 1:
            var = sum((v - m) ** 2 for v in clean) / (len(clean) - 1)
            s = math.sqrt(var)
        else:
            s = 0.0
        return m, s, len(clean)

    def _cell(values):
        m, s, n = _mean_std(values)
        if n == 0:
            return " — "
        return f"{m:.3f} ± {s:.3f}"

    # One table per metric: rows = datasets, columns = methods
    for metric in metrics:
        print(f"\n## metric: `{metric}`  (mean ± std over seeds; lower-is-better unless noted)\n")
        header = "| dataset | " + " | ".join(f"`{m}`" for m in methods) + " | best |"
        sep = "|" + "|".join(["---"] * (len(methods) + 2)) + "|"
        print(header)
        print(sep)
        for ds in datasets:
            cells = []
            mean_only = []
            for m in methods:
                vals = by_dsmm.get((ds, m, metric), [])
                cells.append(_cell(vals))
                mu, _, _ = _mean_std(vals)
                mean_only.append(mu)
            higher_is_better = metric in {"cov90", "cov95"}
            finite_means = [(i, v) for i, v in enumerate(mean_only) if v is not None and not math.isnan(v)]
            if finite_means:
                if higher_is_better:
                    best_idx = max(finite_means, key=lambda iv: iv[1])[0]
                else:
                    best_idx = min(finite_means, key=lambda iv: iv[1])[0]
                best_method = methods[best_idx]
            else:
                best_method = "-"
            print("| " + ds + " | " + " | ".join(cells) + f" | **{best_method}** |")
        print()


if __name__ == "__main__":
    main()

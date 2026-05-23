"""Sweep train-anchor q(R) hyperparameters for uncertain-W CEM on L2/LH.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # LH seed0 train-anchor q(R) sweep, 200 test points.
    python experiments/synthetic/sweep_uncertain_w_cem_qr_trainanchors.py ^
        --settings lh_q5_train1000 ^
        --variant qr_alt_lr2_refresh10_self_cem2_nstd0.1_kcov0.1 ^
        --n_neighbors_grid 50,100,200 ^
        --m_inducing_grid 100,200 ^
        --qr_train_anchor_count_grid 50,100,200 ^
        --max_test 200 ^
        --out_dir experiments/synthetic/uncertain_w_cem_qr_trainanchor_sweep_seed0_20260515

    # Faster smoke.
    python experiments/synthetic/sweep_uncertain_w_cem_qr_trainanchors.py ^
        --settings l2_q2_train1000 ^
        --variant qr_alt_lr1_refresh2_self_cem1_nstd0.1_kcov0.1 ^
        --n_neighbors_grid 20 --m_inducing_grid 8 ^
        --qr_train_anchor_count_grid 8 --max_test 5 --qr_steps 2 --qr_outer_cycles 2 ^
        --refresh_hyper_steps 1 --refresh_gate_steps 1 ^
        --out_dir experiments/synthetic/uncertain_w_cem_qr_trainanchor_sweep_smoke

This driver reuses ``compare_uncertain_w_cem_l2_lh._run_one`` but makes the
variant name unique per hyperparameter tuple, so aggregate rows are meaningful.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from djgp.diagnostics import sanitize_for_json  # noqa: E402
from experiments.synthetic import compare_uncertain_w_cem_l2_lh as base  # noqa: E402


def _parse_int_grid(text: str) -> list[int]:
    return [int(s.strip()) for s in str(text).split(",") if s.strip()]


def _default_base_args() -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["compare_uncertain_w_cem_l2_lh.py"]
        return base.parse_args()
    finally:
        sys.argv = old_argv


def _sweep_variant_name(variant: str, *, n: int, m: int, a: int, an: int, strategy: str) -> str:
    return f"{variant}_nn{n}_M{m}_A{a}_an{an}_{strategy}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep train-anchor qR hyperparameters for uncertain-W CEM.")
    p.add_argument("--settings", default="lh_q5_train1000")
    p.add_argument("--variant", default="qr_alt_lr2_refresh10_self_cem2_nstd0.1_kcov0.1")
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--n_neighbors_grid", default="50,100,200")
    p.add_argument("--m_inducing_grid", default="100,200")
    p.add_argument("--qr_train_anchor_count_grid", default="50,100,200")
    p.add_argument(
        "--qr_train_n_neighbors_grid",
        default="",
        help="Defaults to the same values as --n_neighbors_grid.",
    )
    p.add_argument("--qr_train_anchor_strategy", default="kmeans_yvar")
    p.add_argument("--projection_init", default="pls", choices=["pls", "pca", "pca_scaled"])
    p.add_argument("--lowrank_basis", default="pca_residual", choices=["pca_residual", "pca", "random"])
    p.add_argument("--lowrank_qr_w_signal_var", type=float, default=0.1)
    p.add_argument("--qr_steps", type=int, default=40)
    p.add_argument("--qr_outer_cycles", type=int, default=4)
    p.add_argument("--qr_beta_kl", type=float, default=0.05)
    p.add_argument("--qr_lr", type=float, default=0.01)
    p.add_argument("--refresh_hyper_steps", type=int, default=10)
    p.add_argument("--refresh_gate_steps", type=int, default=20)
    p.add_argument("--hyper_steps", type=int, default=30)
    p.add_argument("--gate_steps", type=int, default=60)
    p.add_argument("--include_original", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out_dir", default="experiments/synthetic/uncertain_w_cem_qr_trainanchor_sweep_seed0_20260515")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in str(args.settings).split(",") if s.strip()]
    n_grid = _parse_int_grid(args.n_neighbors_grid)
    m_grid = _parse_int_grid(args.m_inducing_grid)
    a_grid = _parse_int_grid(args.qr_train_anchor_count_grid)
    an_grid = _parse_int_grid(args.qr_train_n_neighbors_grid) if str(args.qr_train_n_neighbors_grid).strip() else n_grid
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = base._read_csv(out_dir / "metrics.csv") if args.resume else []
    done = {
        (str(r.get("setting")), int(r.get("seed", -1)), str(r.get("variant")))
        for r in rows
        if r.get("status") == "ok"
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
        for setting in settings:
            if bool(args.include_original):
                key = (setting, seed, "original_meanw_jumpgp_cem")
                if key not in done:
                    base_args = _default_base_args()
                    base_args.settings = setting
                    base_args.max_test = int(args.max_test)
                    base_args.include_original = True
                    print(f"run setting={setting} seed={seed} variant=original_meanw_jumpgp_cem", flush=True)
                    rows.append(base._run_one(base_args, setting, seed, "__original__", device))
                    base._write_csv(out_dir / "metrics.csv", rows)
                    base._write_csv(out_dir / "aggregate.csv", base._aggregate(rows))
                    done.add(key)
            for n in n_grid:
                for an in an_grid:
                    for m in m_grid:
                        for a in a_grid:
                            sweep_name = _sweep_variant_name(
                                args.variant,
                                n=n,
                                m=m,
                                a=a,
                                an=an,
                                strategy=str(args.qr_train_anchor_strategy),
                            )
                            key = (setting, seed, sweep_name)
                            if key in done:
                                print(f"skip completed setting={setting} seed={seed} variant={sweep_name}", flush=True)
                                continue
                            base_args = _default_base_args()
                            base_args.settings = setting
                            base_args.variants = args.variant
                            base_args.seed_start = seed
                            base_args.num_exp = 1
                            base_args.max_test = int(args.max_test)
                            base_args.n_neighbors = int(n)
                            base_args.m_inducing = int(m)
                            base_args.qr_train_anchor_count = int(a)
                            base_args.qr_train_n_neighbors = int(an)
                            base_args.qr_train_anchor_strategy = str(args.qr_train_anchor_strategy)
                            base_args.projection_init = str(args.projection_init)
                            base_args.lowrank_basis = str(args.lowrank_basis)
                            base_args.lowrank_qr_w_signal_var = float(args.lowrank_qr_w_signal_var)
                            base_args.qr_steps = int(args.qr_steps)
                            base_args.qr_outer_cycles = int(args.qr_outer_cycles)
                            base_args.qr_beta_kl = float(args.qr_beta_kl)
                            base_args.qr_lr = float(args.qr_lr)
                            base_args.refresh_hyper_steps = int(args.refresh_hyper_steps)
                            base_args.refresh_gate_steps = int(args.refresh_gate_steps)
                            base_args.hyper_steps = int(args.hyper_steps)
                            base_args.gate_steps = int(args.gate_steps)
                            base_args.include_original = False
                            print(f"run setting={setting} seed={seed} variant={sweep_name}", flush=True)
                            row = base._run_one(base_args, setting, seed, str(args.variant), device)
                            row["variant"] = sweep_name
                            row["sweep_base_variant"] = str(args.variant)
                            row["sweep_n_neighbors"] = int(n)
                            row["sweep_m_inducing"] = int(m)
                            row["sweep_qr_train_anchor_count"] = int(a)
                            row["sweep_qr_train_n_neighbors"] = int(an)
                            row["sweep_qr_train_anchor_strategy"] = str(args.qr_train_anchor_strategy)
                            rows.append(row)
                            base._write_csv(out_dir / "metrics.csv", rows)
                            base._write_csv(out_dir / "aggregate.csv", base._aggregate(rows))
                            if row.get("status") == "ok":
                                done.add(key)

    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "args": vars(args),
        "settings": {name: base.SETTING_PRESETS[name] for name in settings},
        "n_rows": len(rows),
        "n_ok": int(sum(1 for r in rows if r.get("status") == "ok")),
        "n_failed": int(sum(1 for r in rows if r.get("status") != "ok")),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    base._write_readme(out_dir, sanitize_for_json(summary))
    print(f"Saved sweep results to {out_dir}", flush=True)
    return {"rows": rows, "aggregate": base._aggregate(rows), "summary": summary}


if __name__ == "__main__":
    main()

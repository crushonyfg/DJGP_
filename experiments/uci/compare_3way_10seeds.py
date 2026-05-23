"""
Cross-method UCI benchmark: XGBoost-bootstrap, DKL-SVGP, and CEM-EM
(coeff-form, recommended defaults from `docs/cem_em_projection.md`) on
Wine Quality, Parkinsons Telemonitoring, and Appliances Energy Prediction
across multiple seeds.

For fair comparison **all methods evaluate on the same `max_anchors` test
slice** (subsampled deterministically from the dataset's test split). All
methods share the seed used by `load_uci_split`. PLS-only is included as a
free reference line because the CEM-EM pipeline already computes `W_pls`.

Each method's output row:

    [rmse, mean_crps, wall_sec, coverage_90, mean_width_90,
     coverage_95, mean_width_95]

Outputs
-------
- ``--out_csv``: long-form CSV (dataset, seed, method, metric, value); easy
  to load into pandas / matplotlib for plots.
- ``--out_json``: nested ``{dataset: {seed: {method: {metrics, diagnostics}}}}``
  with full per-seed information (CEM-EM diagnostics + snapshots inclusive).
- Per-seed JSON checkpoint ``<out_json>.partial`` is written after each seed
  finishes so a crash mid-run does not lose previous seeds.

How to run (from repository root; conda env: jumpGP)::

    conda activate jumpGP
    cd <path-to-DJGP-repo>

    # Headline run: 3 datasets x 10 seeds, 200 anchors per seed.
    python experiments/uci/compare_3way_10seeds.py \
        --num_exp 10 \
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" \
        --max_anchors 200 \
        --out_csv experiments/uci/compare_3way_10seeds/results.csv \
        --out_json experiments/uci/compare_3way_10seeds/results.json

    # Smoke (1 seed, Wine Quality, 64 anchors)
    python experiments/uci/compare_3way_10seeds.py --num_exp 1 \
        --datasets "Wine Quality" --max_anchors 64

Resume after interruption::

    # Same paths as the original run; loads ``results.json.partial`` (if present)
    # and skips every (dataset, seed) that already has all requested methods.
    conda activate jumpGP
    cd <path-to-DJGP-repo>
    python experiments/uci/compare_3way_10seeds.py --resume \
        --num_exp 10 \
        --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" \
        --max_anchors 200 \
        --out_csv experiments/uci/compare_3way_10seeds/results.csv \
        --out_json experiments/uci/compare_3way_10seeds/results.json

When you change CLI defaults or add methods, update the docstring above and
also update ``docs/experiments_log.md`` with a new entry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402

from djgp.baselines import (  # noqa: E402
    run_dkl_svgp_regression,
    run_xgboost_bootstrap_regression,
)
from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    compute_pls_W0,
    per_anchor_W_diagnostics,
    run_projected_jumpgp,
    train_projection_cem_em,
)
from experiments.uci.uci_data import load_uci_split  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
METRIC_KEYS = ["rmse", "crps", "wall_sec", "cov90", "w90", "cov95", "w95"]


def _subsample_test(X_test_np, y_test_np, max_anchors: int | None, seed: int):
    """Deterministic subsample of the test split. Returns the subset plus the
    indices used (so different methods evaluate on identical points)."""
    n = X_test_np.shape[0]
    if max_anchors is None or max_anchors >= n:
        return X_test_np, y_test_np, np.arange(n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_anchors, replace=False))
    return X_test_np[idx], y_test_np[idx], idx


def _row(metrics_list) -> dict[str, float]:
    out = {}
    for k, v in zip(METRIC_KEYS, metrics_list):
        out[k] = float(v) if v is not None and np.isfinite(float(v)) else float("nan")
    return out


def _seed_complete(seed_block: dict, methods: list[str]) -> bool:
    m = seed_block.get("methods") or {}
    for name in methods:
        if name not in m:
            return False
        row = m[name]
        if not isinstance(row, list) or len(row) != 7:
            return False
    return True


def _long_rows_from_nested(nested: dict, methods: list[str]) -> list[dict]:
    rows: list[dict] = []
    for ds, seeds in nested.items():
        if ds == "_meta" or not isinstance(seeds, dict):
            continue
        for seed_str, block in seeds.items():
            if not isinstance(block, dict):
                continue
            for mname, ml in (block.get("methods") or {}).items():
                if mname not in methods:
                    continue
                rows.append({"dataset": ds, "seed": int(seed_str), "method": mname, "metrics": _row(ml)})
    return rows


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------
def _run_xgb(X_train_np, y_train_np, X_test_np, y_test_np, *, seed: int, n_boots: int):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_np)
    X_te = scaler.transform(X_test_np)
    rmse, crps, elapsed, details = run_xgboost_bootstrap_regression(
        X_tr, y_train_np, X_te, y_test_np,
        n_bootstrap=n_boots, random_state=seed,
    )
    return pack_gaussian_uq_list(
        float(rmse), float(crps), float(elapsed),
        y_test_np.ravel(), details["mu"], details["sigma"],
    )


def _run_dkl(X_train_np, y_train_np, X_test_np, y_test_np, *, seed: int, epochs: int, device: torch.device):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_np)
    X_te = scaler.transform(X_test_np)
    rmse, crps, dt, mu, sig = run_dkl_svgp_regression(
        X_tr, y_train_np, X_te, y_test_np,
        device=device, epochs=epochs, seed=seed,
    )
    return pack_gaussian_uq_list(float(rmse), float(crps), float(dt), y_test_np.ravel(), mu, sig)


def _build_neighborhoods(X_test_t, X_train_t, Y_train_t, n: int):
    nbs = find_neighborhoods(X_test_t.cpu(), X_train_t.cpu(), Y_train_t.cpu(), M=n)
    X_nb_list = [nb["X_neighbors"].to(X_test_t.device) for nb in nbs]
    y_nb_list = [nb["y_neighbors"].to(X_test_t.device) for nb in nbs]
    return X_nb_list, y_nb_list


def _eval_with_w(W_t, X_anchors_t, X_nb_list, y_nb_list, y_test_np, device, label: str):
    out = run_projected_jumpgp(
        W_t, X_anchors_t, X_nb_list, y_nb_list,
        device=device, progress=False, desc=label,
    )
    mu_np = out["mu"]
    sig_np = out["sigma"]
    rmse = float(np.sqrt(np.mean((mu_np - y_test_np.ravel()) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.from_numpy(y_test_np.ravel()).float(),
            torch.from_numpy(mu_np).float(),
            torch.from_numpy(sig_np).float(),
        ).item()
    )
    return pack_gaussian_uq_list(
        rmse, crps, float(out["elapsed_sec"]), y_test_np.ravel(), mu_np, sig_np,
    ), {"mu": mu_np, "sigma": sig_np}


def _run_pls_and_cem_em(
    *, X_train_np, y_train_np, X_test_np, y_test_np, K, args, device,
):
    """Builds neighborhoods once and runs PLS-only + CEM-EM on the same anchors."""
    X_train = torch.from_numpy(X_train_np).float().to(device)
    Y_train = torch.from_numpy(y_train_np).float().squeeze(-1).to(device)
    X_test = torch.from_numpy(X_test_np).float().to(device)

    X_nb_list, y_nb_list = _build_neighborhoods(X_test, X_train, Y_train, n=args.n)

    Q = args.Q if args.Q is not None else int(K)
    V_np, V_info = build_V_basis(
        X_train_np, y_train_np.ravel(),
        n_pls=args.n_pls, n_pca=args.n_pca, n_random=args.n_random, seed=0,
    )
    V_t = torch.from_numpy(V_np).float().to(device)
    W_pls_np = compute_pls_W0(X_train_np, y_train_np.ravel(), Q=Q)
    W_pls_t = torch.from_numpy(W_pls_np).float().to(device)
    T = X_test.shape[0]

    pls_metrics, _ = _eval_with_w(
        W_pls_t.unsqueeze(0).expand(T, -1, -1).contiguous(),
        X_test, X_nb_list, y_nb_list, y_test_np, device, label="pls_only",
    )

    model = CoeffProjection(
        T=T, Q=Q, D=X_test.shape[1], V=V_t, W0_init=W_pls_t,
        learn_W0=True, init_C_scale=args.init_C_scale,
        coeff_scale=args.coeff_scale, row_normalize=bool(args.row_normalize_W),
    ).to(device)
    cfg = CemEmConfig(
        n_outer=args.n_outer,
        n_m_steps=args.n_m_steps,
        lr=args.lr,
        lambda_gate=args.lambda_gate,
        lambda_smooth=args.lambda_smooth,
        lambda_scale=args.lambda_scale,
        normalize_smoothness=bool(args.normalize_smoothness),
        track_best_W=bool(args.track_best_W),
        snapshot_after_each_outer=False,
        cem_progress=False,
        verbose=False,
    )
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        X_anchors=X_test,
        device=device,
        cfg=cfg,
        label="cem_em_coeff",
    )
    cem_metrics, _ = _eval_with_w(
        res.W, X_test, X_nb_list, y_nb_list, y_test_np, device, label="cem_em_coeff",
    )

    diag = per_anchor_W_diagnostics(res.W, W_ref=W_pls_t)
    diag.update({
        "best_outer": int(res.best_outer),
        "best_metric_value": float(res.best_metric_value),
        "train_time_sec": float(res.train_time_sec),
        "history": res.history,
        "Q": int(Q),
        "Rv": int(V_t.shape[1]),
        "D": int(X_test.shape[1]),
        "n_anchors": int(T),
    })
    return pls_metrics, cem_metrics, diag


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="3-way (XGB + DKL + CEM-EM) UCI benchmark over multiple seeds.")
    p.add_argument("--num_exp", type=int, default=10)
    p.add_argument("--datasets", type=str,
                   default="Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction",
                   help="Comma-separated UCI dataset names.")
    p.add_argument("--max_anchors", type=int, default=200,
                   help="Subsample size of the test split (used by every method).")
    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--out_json", type=str, default="")

    # XGB
    p.add_argument("--xgb_boots", type=int, default=30)

    # DKL
    p.add_argument("--dkl_epochs", type=int, default=80)

    # CEM-EM (frozen at the recommended winning grid config)
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--n", type=int, default=35)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_m_steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lambda_gate", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_scale", type=float, default=0.0)
    p.add_argument("--init_C_scale", type=float, default=0.0)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--row_normalize_W", type=int, default=1)
    p.add_argument("--normalize_smoothness", type=int, default=1)
    p.add_argument("--track_best_W", type=int, default=1)

    p.add_argument("--methods", type=str, default="xgb_bootstrap,dkl_svgp,pls_only,cem_em_coeff",
                   help="Comma-separated subset of {xgb_bootstrap,dkl_svgp,pls_only,cem_em_coeff}.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Load existing checkpoint (--out_json).partial if present; skip completed "
        "(dataset, seed) pairs. Rebuild long CSV rows from JSON so outputs stay consistent.",
    )
    return p.parse_args()


def _save_csv(rows: list[dict], path: Path):
    """Long-form CSV: dataset, seed, method, metric, value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seed", "method", "metric", "value"])
        for r in rows:
            for m in METRIC_KEYS:
                w.writerow([r["dataset"], r["seed"], r["method"], m, r["metrics"][m]])


def _save_csv_atomic(rows: list[dict], path: Path):
    """Avoid PermissionError on Windows when another program holds ``path`` open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    _save_csv(rows, tmp)
    tmp.replace(path)


def _save_json(blob: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with path.open("w") as f:
        json.dump(blob, f, indent=2, default=_convert)


def main():
    args = parse_args()
    dataset_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(args.out_csv) if args.out_csv else Path(
        f"experiments/uci/compare_3way_10seeds/results_{timestamp}.csv"
    )
    out_json = Path(args.out_json) if args.out_json else Path(
        f"experiments/uci/compare_3way_10seeds/results_{timestamp}.json"
    )
    out_partial = out_json.with_suffix(out_json.suffix + ".partial")

    long_rows: list[dict] = []
    nested: dict = {"_meta": {
        "num_exp": args.num_exp,
        "datasets": dataset_list,
        "methods": methods,
        "max_anchors": args.max_anchors,
        "cem_em": {
            "n": args.n, "n_pls": args.n_pls, "n_pca": args.n_pca, "n_random": args.n_random,
            "n_outer": args.n_outer, "n_m_steps": args.n_m_steps, "lr": args.lr,
            "lambda_gate": args.lambda_gate, "lambda_smooth": args.lambda_smooth,
            "lambda_scale": args.lambda_scale, "coeff_scale": args.coeff_scale,
            "row_normalize_W": bool(args.row_normalize_W),
            "normalize_smoothness": bool(args.normalize_smoothness),
            "track_best_W": bool(args.track_best_W),
        },
        "xgb": {"n_boots": args.xgb_boots},
        "dkl": {"epochs": args.dkl_epochs},
        "started_at": timestamp,
    }}
    for ds in dataset_list:
        nested[ds] = {}

    if args.resume:
        ckpt_path = out_partial if out_partial.exists() else out_json
        if ckpt_path.exists():
            with ckpt_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            meta_old = loaded.get("_meta", {})
            if meta_old.get("max_anchors") != args.max_anchors or meta_old.get("num_exp") != args.num_exp:
                print(
                    f"[resume] WARNING: checkpoint meta num_exp/max_anchors differs from CLI; "
                    f"continuing anyway. ckpt={meta_old.get('num_exp')}, {meta_old.get('max_anchors')} "
                    f"vs args={args.num_exp}, {args.max_anchors}",
                )
            nested = loaded
            # Drop incomplete (dataset, seed) blocks so they are recomputed from scratch.
            dropped: list[tuple[str, str]] = []
            for ds in dataset_list:
                if ds not in nested or not isinstance(nested.get(ds), dict):
                    continue
                for seed_str, block in list(nested[ds].items()):
                    if isinstance(block, dict) and not _seed_complete(block, methods):
                        del nested[ds][seed_str]
                        dropped.append((ds, seed_str))
            if dropped:
                print(f"[resume] dropped incomplete checkpoints (will re-run): {dropped}", flush=True)
            long_rows = _long_rows_from_nested(nested, methods)
            nested["_meta"] = meta_old
            nested["_meta"]["resumed_at"] = timestamp
            nested["_meta"]["resume_checkpoint"] = str(ckpt_path)
            # Ensure every dataset key exists for in-place updates
            for ds in dataset_list:
                nested.setdefault(ds, {})
            n_skip = sum(
                1
                for ds in dataset_list
                for s in range(args.num_exp)
                if str(s) in nested.get(ds, {})
                and _seed_complete(nested[ds][str(s)], methods)
            )
            print(
                f"[resume] loaded {ckpt_path} — rebuilt {len(long_rows)} CSV rows; "
                f"will skip {n_skip} complete (dataset, seed) pairs.",
                flush=True,
            )
        else:
            print("[resume] no checkpoint found; starting fresh.", flush=True)

    t_total = time.perf_counter()
    for ds in dataset_list:
        for seed in range(args.num_exp):
            if (
                args.resume
                and str(seed) in nested.get(ds, {})
                and _seed_complete(nested[ds][str(seed)], methods)
            ):
                print(f"\n=== {ds}  seed={seed} === [resume: skip, already complete]", flush=True)
                continue
            t_seed = time.perf_counter()
            print(f"\n=== {ds}  seed={seed} ===", flush=True)
            try:
                X_train_np, X_test_np, y_train_np, y_test_np_full, K = load_uci_split(ds, seed)
            except Exception as exc:
                print(f"[load_uci_split error] {exc}; skipping seed.")
                continue
            X_te_sub, y_te_sub, idx_used = _subsample_test(
                X_test_np, y_test_np_full, args.max_anchors, seed=seed,
            )
            print(f"  test_full={X_test_np.shape[0]} -> test_used={X_te_sub.shape[0]}, "
                  f"D={X_test_np.shape[1]}, K={K}", flush=True)

            seed_block: dict = {
                "K": int(K), "D": int(X_test_np.shape[1]),
                "n_test_used": int(X_te_sub.shape[0]),
                "n_test_full": int(X_test_np.shape[0]),
                "test_indices_used": idx_used.tolist(),
                "methods": {},
                "diagnostics": {},
            }

            # --- XGBoost bootstrap ---
            if "xgb_bootstrap" in methods:
                try:
                    print("  xgb_bootstrap...", flush=True)
                    row = _run_xgb(X_train_np, y_train_np, X_te_sub, y_te_sub,
                                   seed=seed, n_boots=args.xgb_boots)
                    seed_block["methods"]["xgb_bootstrap"] = list(row)
                    long_rows.append({"dataset": ds, "seed": seed, "method": "xgb_bootstrap", "metrics": _row(row)})
                    print(f"    rmse={row[0]:.4f} crps={row[1]:.4f} wall={row[2]:.2f}s", flush=True)
                except Exception as exc:
                    print(f"  [xgb error] {exc}")
                    seed_block["methods"]["xgb_bootstrap"] = [float("nan")] * 7

            # --- DKL-SVGP ---
            if "dkl_svgp" in methods:
                try:
                    print("  dkl_svgp...", flush=True)
                    row = _run_dkl(X_train_np, y_train_np, X_te_sub, y_te_sub,
                                   seed=seed, epochs=args.dkl_epochs, device=device)
                    seed_block["methods"]["dkl_svgp"] = list(row)
                    long_rows.append({"dataset": ds, "seed": seed, "method": "dkl_svgp", "metrics": _row(row)})
                    print(f"    rmse={row[0]:.4f} crps={row[1]:.4f} wall={row[2]:.2f}s", flush=True)
                except Exception as exc:
                    print(f"  [dkl error] {exc}")
                    seed_block["methods"]["dkl_svgp"] = [float("nan")] * 7

            # --- PLS-only + CEM-EM (share neighbors) ---
            if "cem_em_coeff" in methods or "pls_only" in methods:
                try:
                    print("  pls_only + cem_em_coeff...", flush=True)
                    pls_row, cem_row, cem_diag = _run_pls_and_cem_em(
                        X_train_np=X_train_np, y_train_np=y_train_np,
                        X_test_np=X_te_sub, y_test_np=y_te_sub,
                        K=K, args=args, device=device,
                    )
                    if "pls_only" in methods:
                        seed_block["methods"]["pls_only"] = list(pls_row)
                        long_rows.append({"dataset": ds, "seed": seed, "method": "pls_only", "metrics": _row(pls_row)})
                        print(f"    pls   rmse={pls_row[0]:.4f} crps={pls_row[1]:.4f} wall={pls_row[2]:.2f}s", flush=True)
                    if "cem_em_coeff" in methods:
                        seed_block["methods"]["cem_em_coeff"] = list(cem_row)
                        seed_block["diagnostics"]["cem_em_coeff"] = cem_diag
                        long_rows.append({"dataset": ds, "seed": seed, "method": "cem_em_coeff", "metrics": _row(cem_row)})
                        print(f"    cem   rmse={cem_row[0]:.4f} crps={cem_row[1]:.4f} wall={cem_row[2]:.2f}s "
                              f"(train={cem_diag['train_time_sec']:.1f}s best_outer={cem_diag['best_outer']})",
                              flush=True)
                except Exception as exc:
                    import traceback
                    print(f"  [cem-em / pls error] {exc}\n{traceback.format_exc()}")
                    if "pls_only" in methods:
                        seed_block["methods"]["pls_only"] = [float("nan")] * 7
                    if "cem_em_coeff" in methods:
                        seed_block["methods"]["cem_em_coeff"] = [float("nan")] * 7

            nested[ds][str(seed)] = seed_block
            t_elapsed = time.perf_counter() - t_seed
            print(f"  --- seed wall: {t_elapsed:.1f}s ---", flush=True)

            # checkpoint after every seed in case of crash later
            try:
                _save_json(nested, out_partial)
                _save_csv_atomic(long_rows, out_csv.with_suffix(".partial.csv"))
            except Exception as exc:
                print(f"  [checkpoint warn] {exc}")

    # Final outputs
    _save_csv(long_rows, out_csv)
    _save_json(nested, out_json)
    if out_partial.exists():
        try:
            out_partial.unlink()
        except OSError:
            pass
    partial_csv = out_csv.with_suffix(".partial.csv")
    if partial_csv.exists():
        try:
            partial_csv.unlink()
        except OSError:
            pass
    print(f"\nSaved CSV  -> {out_csv}")
    print(f"Saved JSON -> {out_json}")
    print(f"Total wall: {(time.perf_counter() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main()

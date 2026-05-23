"""
LMJGP (VI) vs CEM-EM on three UCI datasets with **one seed**, comparing input scaling:

  * ``raw``      : no standardization (current default behaviour).
  * ``std_x``    : ``StandardScaler`` fit on train **X** only.
  * ``std_xy``   : standardize **X** and **y** (fit on train); RMSE / CRPS / coverage
    are computed on the **original y scale** (inverse transform of ``y`` and affine
    map of Gaussian ``(mu, sigma)`` in scaled space: ``mu_y = mu_s * scale + mean``,
    ``sigma_y = sigma_s * |scale|``).

Same neighbourhoods and hyperparameters for LMJGP and CEM-EM within each run; only
the numeric values of ``X, y`` change with the scaling mode.

How to run (repo root, conda env ``jumpGP``)::

    conda activate jumpGP
    cd <path-to-DJGP>

    python experiments/uci/compare_lmjgp_cem_standardization.py \\
        --seed 0 --max_anchors 200 \\
        --out_json experiments/uci/compare_lmjgp_cem_std/results.json \\
        --out_csv experiments/uci/compare_lmjgp_cem_std/results.csv

    # Optional: LMJGP MC predictive variance = within (local GP) + between (W disagreement)
    # summaries in JSON under ``lmjgp.variance_decomposition``; also printed.
    python experiments/uci/compare_lmjgp_cem_standardization.py --lmjgp_var_diag ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import compute_metrics  # noqa: E402

from djgp.evaluation import mean_gaussian_crps_torch  # noqa: E402
from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    compute_pls_W0,
    run_projected_jumpgp,
    train_projection_cem_em,
)
from experiments.uci.compare_cem_em_projection import (  # noqa: E402
    _build_neighborhoods,
    _train_lmjgp_reference,
)
from experiments.uci.uci_data import load_uci_split  # noqa: E402


DATASETS_DEFAULT = (
    "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
)


def _subsample_test(X_test_np, y_test_np, max_anchors: int | None, seed: int):
    n = X_test_np.shape[0]
    if max_anchors is None or max_anchors >= n:
        return X_test_np, y_test_np
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_anchors, replace=False))
    return X_test_np[idx], y_test_np[idx]


def _apply_scaling(
    X_train_np: np.ndarray,
    X_test_np: np.ndarray,
    y_train_np: np.ndarray,
    y_test_np: np.ndarray,
    *,
    standardize_x: bool,
    standardize_y: bool,
):
    """Returns transformed arrays and scalers (None if unused)."""
    x_scaler = StandardScaler() if standardize_x else None
    y_scaler = StandardScaler() if standardize_y else None

    if x_scaler is not None:
        X_tr = x_scaler.fit_transform(X_train_np)
        X_te = x_scaler.transform(X_test_np)
    else:
        X_tr = np.asarray(X_train_np, dtype=np.float64)
        X_te = np.asarray(X_test_np, dtype=np.float64)

    y_tr = np.asarray(y_train_np, dtype=np.float64).ravel()
    y_te = np.asarray(y_test_np, dtype=np.float64).ravel()

    if y_scaler is not None:
        y_tr = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
        y_te = y_scaler.transform(y_te.reshape(-1, 1)).ravel()

    return X_tr, X_te, y_tr, y_te, x_scaler, y_scaler


def _pack_metrics(
    y_eval: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
    *,
    wall_sec: float,
) -> list[float]:
    y_eval = np.asarray(y_eval, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sig = np.asarray(sig, dtype=np.float64).ravel()
    rmse = float(np.sqrt(np.mean((mu - y_eval) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.from_numpy(y_eval).float(),
            torch.from_numpy(mu).float(),
            torch.from_numpy(sig).float(),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, wall_sec, y_eval, mu, sig)


def _metrics_original_y_scale(
    y_scaler: StandardScaler,
    y_test_scaled: np.ndarray,
    mu_scaled: np.ndarray,
    sig_scaled: np.ndarray,
    *,
    wall_sec: float,
) -> list[float]:
    """Map Gaussian predictive in scaled-y space back to original train y scale.

    Training uses ``y_std = (y - mean_) / scale_`` (``StandardScaler``). If the
    model marginal is ``N(μ_s, σ_s²)`` in *standardized* y, then in original units
    ``y = scale_ · y_std + mean_`` implies ``y | data ~ N(μ_s·scale_+mean_, (σ_s·|scale_|)²)``.

    So: ``μ_raw = inverse_transform(μ_s)`` elementwise, ``σ_raw = σ_s * |scale_|``.
    We use ``inverse_transform`` for ``μ`` (and ``y_true``) to match sklearn exactly
    for vector-valued edge cases; for 1-D it equals ``μ_s * scale_ + mean_``.

    **Sanity:** Gaussian CRPS is equivariant: ``CRPS(y,μ,σ)`` at raw scale equals
    ``|scale_| * CRPS(y_std, μ_s, σ_s)`` (same z-score). See ``_debug_std_xy_checks``.
    """
    scale = float(np.squeeze(y_scaler.scale_))
    if not np.isfinite(scale) or scale < 1e-20:
        scale = 1.0
    y_orig = y_scaler.inverse_transform(np.asarray(y_test_scaled).reshape(-1, 1)).ravel()
    mu_orig = y_scaler.inverse_transform(np.asarray(mu_scaled).reshape(-1, 1)).ravel()
    sig_orig = np.asarray(sig_scaled, dtype=np.float64).ravel() * abs(scale)
    return _pack_metrics(y_orig, mu_orig, sig_orig, wall_sec=wall_sec)


def _debug_std_xy_checks(
    tag: str,
    y_scaler: StandardScaler,
    y_std: np.ndarray,
    mu_s: np.ndarray,
    sig_s: np.ndarray,
    *,
    first_k: int,
) -> None:
    """Print row-level samples + assert affine Gaussian CRPS identity (catches σ doubled, etc.)."""
    mean_y = float(np.squeeze(y_scaler.mean_))
    scale = float(np.squeeze(y_scaler.scale_))
    y_std = np.asarray(y_std, dtype=np.float64).ravel()
    mu_s = np.asarray(mu_s, dtype=np.float64).ravel()
    sig_s = np.asarray(sig_s, dtype=np.float64).ravel()

    y_orig = y_scaler.inverse_transform(y_std.reshape(-1, 1)).ravel()
    mu_o = y_scaler.inverse_transform(mu_s.reshape(-1, 1)).ravel()
    sig_o = sig_s * abs(scale)

    assert np.allclose(y_orig, y_std * scale + mean_y, rtol=1e-5, atol=1e-5), f"{tag}: y_orig affine"
    assert np.allclose(mu_o, mu_s * scale + mean_y, rtol=1e-5, atol=1e-5), f"{tag}: mu_orig affine"
    assert np.allclose(sig_o, sig_s * abs(scale), rtol=1e-6, atol=1e-8), f"{tag}: sigma_orig affine"

    crps_std = mean_gaussian_crps_torch(
        torch.from_numpy(y_std).float(),
        torch.from_numpy(mu_s).float(),
        torch.from_numpy(sig_s).float(),
    ).item()
    crps_orig = mean_gaussian_crps_torch(
        torch.from_numpy(y_orig).float(),
        torch.from_numpy(mu_o).float(),
        torch.from_numpy(sig_o).float(),
    ).item()
    crps_via_scale = abs(scale) * crps_std
    if abs(crps_orig - crps_via_scale) > 5e-4 * max(1.0, crps_orig):
        raise AssertionError(
            f"{tag}: CRPS inconsistency: orig={crps_orig:.6f} vs |scale|*CRPS_std={crps_via_scale:.6f} "
            f"(scale={scale:.6f}) — check that σ is std not variance.",
        )

    k = min(first_k, len(y_std))
    print(f"\n[debug std_xy] {tag}  y_scaler mean={mean_y:.6g} scale={scale:.6g}  (first {k} points)", flush=True)
    print("  i   y_raw      y_std      mu_std     sig_std    mu_raw     sig_raw", flush=True)
    for i in range(k):
        print(
            f"  {i}  {y_orig[i]:.5f}  {y_std[i]:.5f}  {mu_s[i]:.5f}  {sig_s[i]:.5f}  {mu_o[i]:.5f}  {sig_o[i]:.5f}",
            flush=True,
        )
    print(
        f"  mean CRPS std-space={crps_std:.6f}  |scale|*CRPS_std={crps_via_scale:.6f}  "
        f"CRPS recomputed orig={crps_orig:.6f} (match OK)",
        flush=True,
    )


def _summarize_lmjgp_var_decomposition(
    sig: np.ndarray,
    var_within: np.ndarray,
    var_between: np.ndarray,
    *,
    y_scaler: StandardScaler | None,
) -> dict:
    """Summaries in training (model) y-space and, if ``y_scaler`` is set, original y-space."""
    sig = np.asarray(sig, dtype=np.float64).ravel()
    vw = np.asarray(var_within, dtype=np.float64).ravel()
    vb = np.asarray(var_between, dtype=np.float64).ravel()
    scale = 1.0
    if y_scaler is not None:
        scale = float(np.squeeze(y_scaler.scale_))
        if not np.isfinite(scale) or scale < 1e-20:
            scale = 1.0
    out = {
        "median_sigma_train": float(np.median(sig)),
        "p90_sigma_train": float(np.percentile(sig, 90)),
        "median_var_within_train": float(np.median(vw)),
        "median_var_between_train": float(np.median(vb)),
        "mean_var_within_train": float(np.mean(vw)),
        "mean_var_between_train": float(np.mean(vb)),
    }
    if y_scaler is not None:
        s = abs(scale)
        out["median_sigma_orig"] = float(np.median(sig) * s)
        out["p90_sigma_orig"] = float(np.percentile(sig, 90) * s)
        out["median_var_within_orig"] = float(np.median(vw) * (scale ** 2))
        out["median_var_between_orig"] = float(np.median(vb) * (scale ** 2))
    else:
        out["median_sigma_orig"] = out["median_sigma_train"]
        out["p90_sigma_orig"] = out["p90_sigma_train"]
        out["median_var_within_orig"] = out["median_var_within_train"]
        out["median_var_between_orig"] = out["median_var_between_train"]
    return out


def _run_lmjgp(
    X_train_np,
    y_train_np,
    X_test_np,
    y_test_np,
    *,
    K: int,
    n: int,
    m1: int,
    m2: int,
    num_steps: int,
    MC_num: int,
    lr: float,
    device: torch.device,
    vi_decomposition: bool = False,
):
    X_train = torch.from_numpy(X_train_np).float().to(device)
    Y_train = torch.from_numpy(y_train_np).float().squeeze(-1).to(device)
    X_test = torch.from_numpy(X_test_np).float().to(device)
    Y_test = torch.from_numpy(y_test_np).float().squeeze(-1).to(device)
    block = _train_lmjgp_reference(
        X_train, Y_train, X_test, Y_test,
        Q=K, m1=m1, m2=m2, n=n, num_steps=num_steps, lr=lr, MC_num=MC_num,
        device=device,
        vi_decomposition=vi_decomposition,
    )
    pv = block["predict_vi"]
    mu = pv["mu"].detach().cpu().numpy()
    sig = pv["sigma"].detach().cpu().numpy()
    wall = float(pv["wall_sec"])
    decomp = None
    if vi_decomposition:
        decomp = {
            "var_within": pv["var_within"].detach().cpu().numpy(),
            "var_between": pv["var_between"].detach().cpu().numpy(),
        }
    return mu, sig, wall, block["train_time_sec"], decomp


def _run_cem_em(
    X_train_np,
    y_train_np,
    X_test_np,
    y_test_np,
    *,
    K: int,
    n: int,
    cfg: CemEmConfig,
    device: torch.device,
):
    X_train = torch.from_numpy(X_train_np).float().to(device)
    Y_train = torch.from_numpy(y_train_np).float().squeeze(-1).to(device)
    X_test = torch.from_numpy(X_test_np).float().to(device)

    _, X_nb_list, y_nb_list = _build_neighborhoods(X_test, X_train, Y_train, device, m1=2, Q=K, n=n)

    V_np, _ = build_V_basis(
        X_train_np, np.asarray(y_train_np).ravel(),
        n_pls=5, n_pca=5, n_random=0, seed=0,
    )
    V_t = torch.from_numpy(V_np).float().to(device)
    W_pls_np = compute_pls_W0(X_train_np, np.asarray(y_train_np).ravel(), Q=K)
    W_pls_t = torch.from_numpy(W_pls_np).float().to(device)
    T = X_test.shape[0]

    model = CoeffProjection(
        T=T, Q=K, D=X_test.shape[1], V=V_t, W0_init=W_pls_t,
        learn_W0=True, init_C_scale=0.0,
        coeff_scale=1.0, row_normalize=True,
    ).to(device)
    cfg_local = CemEmConfig(**{**cfg.__dict__})
    res = train_projection_cem_em(
        model,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        X_anchors=X_test,
        device=device,
        cfg=cfg_local,
        label="cem_em_coeff",
    )
    t0 = time.perf_counter()
    out = run_projected_jumpgp(
        res.W, X_test, X_nb_list, y_nb_list,
        device=device, progress=True, desc="cem_em_coeff",
    )
    jgp_wall = time.perf_counter() - t0
    mu = out["mu"]
    sig = out["sigma"]
    return mu, sig, float(jgp_wall), float(res.train_time_sec)


def parse_args():
    p = argparse.ArgumentParser(description="LMJGP vs CEM-EM with X/y standardization ablation (one seed).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--datasets", type=str, default=DATASETS_DEFAULT)
    p.add_argument("--max_anchors", type=int, default=200)
    p.add_argument("--n", type=int, default=35)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_m_steps", type=int, default=80)
    p.add_argument("--lambda_gate", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--out_csv", type=str, default="")
    p.add_argument(
        "--debug_std_xy",
        type=int,
        default=0,
        help="If >0, for std_xy mode print this many test rows and assert CRPS affine identity.",
    )
    p.add_argument(
        "--lmjgp_var_diag",
        action="store_true",
        help="LMJGP only: run predict_vi with within/between variance decomposition and log summaries.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    dataset_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    modes = [
        ("raw", False, False),
        ("std_x", True, False),
        ("std_xy", True, True),
    ]

    cem_cfg = CemEmConfig(
        n_outer=args.n_outer,
        n_m_steps=args.n_m_steps,
        lr=args.lr,
        lambda_gate=args.lambda_gate,
        lambda_smooth=args.lambda_smooth,
        lambda_scale=0.0,
        normalize_smoothness=True,
        track_best_W=True,
        snapshot_after_each_outer=False,
        cem_progress=False,
        verbose=True,
    )

    results: dict = {
        "_meta": {
            "seed": args.seed,
            "max_anchors": args.max_anchors,
            "n": args.n,
            "lmjgp_steps": args.lmjgp_steps,
            "MC_num": args.MC_num,
            "modes": [m[0] for m in modes],
            "debug_std_xy": int(args.debug_std_xy),
            "lmjgp_var_diag": bool(args.lmjgp_var_diag),
        }
    }
    csv_rows: list[list] = []

    for ds in dataset_list:
        results[ds] = {}
        X_tr_full, X_te_full, y_tr_full, y_te_full, K = load_uci_split(ds, args.seed)
        X_te_sub, y_te_sub = _subsample_test(X_te_full, y_te_full, args.max_anchors, args.seed)

        for mode_name, sx, sy in modes:
            print(f"\n{'='*60}\n{ds} | mode={mode_name} | std_x={sx} std_y={sy}\n{'='*60}", flush=True)
            X_tr, X_te, y_tr, y_te, _, y_scaler = _apply_scaling(
                X_tr_full, X_te_sub, y_tr_full, y_te_sub,
                standardize_x=sx, standardize_y=sy,
            )

            block: dict = {"mode": mode_name, "standardize_x": sx, "standardize_y": sy, "K": int(K)}

            # LMJGP
            t0 = time.perf_counter()
            mu_l, sig_l, wall_l, train_l, lmjgp_decomp = _run_lmjgp(
                X_tr, y_tr, X_te, y_te, K=K, n=args.n, m1=2, m2=40,
                num_steps=args.lmjgp_steps, MC_num=args.MC_num, lr=args.lr, device=device,
                vi_decomposition=args.lmjgp_var_diag,
            )
            total_l = time.perf_counter() - t0
            if sy:
                assert y_scaler is not None
                if args.debug_std_xy > 0:
                    _debug_std_xy_checks(
                        f"{ds}/lmjgp", y_scaler, y_te, mu_l, sig_l, first_k=args.debug_std_xy,
                    )
                pack_l = _metrics_original_y_scale(
                    y_scaler, y_te, mu_l, sig_l, wall_sec=wall_l,
                )
            else:
                pack_l = _pack_metrics(y_te, mu_l, sig_l, wall_sec=wall_l)
            block["lmjgp"] = {
                "metrics_7": pack_l,
                "train_time_sec": float(train_l),
                "predict_wall_sec": float(wall_l),
                "total_wall_sec": float(total_l),
                "metrics_on_original_y_scale": bool(sy),
            }
            if args.lmjgp_var_diag and lmjgp_decomp is not None:
                var_summ = _summarize_lmjgp_var_decomposition(
                    sig_l,
                    lmjgp_decomp["var_within"],
                    lmjgp_decomp["var_between"],
                    y_scaler=y_scaler if sy else None,
                )
                block["lmjgp"]["variance_decomposition"] = var_summ
                ysl = y_scaler if sy else None
                print(
                    f"  [LMJGP var decomp] sigma_train median={var_summ['median_sigma_train']:.5g} "
                    f"p90={var_summ['p90_sigma_train']:.5g} | "
                    f"within_med={var_summ['median_var_within_train']:.5g} "
                    f"between_med={var_summ['median_var_between_train']:.5g}",
                    flush=True,
                )
                print(
                    f"  [LMJGP var decomp] on original y scale: sigma median={var_summ['median_sigma_orig']:.5g} "
                    f"p90={var_summ['p90_sigma_orig']:.5g} | "
                    f"within_med={var_summ['median_var_within_orig']:.5g} "
                    f"between_med={var_summ['median_var_between_orig']:.5g}",
                    flush=True,
                )
                if ysl is not None:
                    print(
                        "    (train space = standardized y; orig scale: sigma *= |scale|, variances *= scale^2)",
                        flush=True,
                    )

            # CEM-EM
            t0 = time.perf_counter()
            mu_c, sig_c, wall_c, train_c = _run_cem_em(
                X_tr, y_tr, X_te, y_te, K=K, n=args.n, cfg=cem_cfg, device=device,
            )
            total_c = time.perf_counter() - t0
            if sy:
                assert y_scaler is not None
                if args.debug_std_xy > 0:
                    _debug_std_xy_checks(
                        f"{ds}/cem_em", y_scaler, y_te, mu_c, sig_c, first_k=args.debug_std_xy,
                    )
                pack_c = _metrics_original_y_scale(
                    y_scaler, y_te, mu_c, sig_c, wall_sec=wall_c,
                )
            else:
                pack_c = _pack_metrics(y_te, mu_c, sig_c, wall_sec=wall_c)
            block["cem_em_coeff"] = {
                "metrics_7": pack_c,
                "train_time_sec": float(train_c),
                "predict_wall_sec": float(wall_c),
                "total_wall_sec": float(total_c),
                "metrics_on_original_y_scale": bool(sy),
            }

            results[ds][mode_name] = block
            for method, pack in [("lmjgp", pack_l), ("cem_em_coeff", pack_c)]:
                for i, name in enumerate(["rmse", "crps", "wall", "cov90", "w90", "cov95", "w95"]):
                    csv_rows.append([ds, mode_name, method, name, pack[i], sx, sy])

    out_dir = PROJECT_ROOT / "experiments" / "uci" / "compare_lmjgp_cem_std"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else out_dir / "results.json"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / "results.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "mode", "method", "metric", "value", "std_x", "std_y"])
        w.writerows(csv_rows)

    print(f"\nSaved JSON -> {out_json}\nSaved CSV  -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()

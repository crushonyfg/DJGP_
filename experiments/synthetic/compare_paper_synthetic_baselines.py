"""Compare baselines on the paper-style L2/LH synthetic generators.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Corrected one-seed pilot on the paper synthetic families.
    python experiments/synthetic/compare_paper_synthetic_baselines.py ^
        --num_exp 1 ^
        --settings l2_q2_train200,l2_q2_train500,lh_q5_train200,lh_q5_train500 ^
        --out_dir experiments/synthetic/paper_l2_lh_xstd_train200_500_seed1

    # Five-seed table, resumable.
    python experiments/synthetic/compare_paper_synthetic_baselines.py ^
        --num_exp 5 ^
        --settings l2_q2_train200,l2_q2_train500,lh_q5_train200,lh_q5_train500 ^
        --resume ^
        --out_dir experiments/synthetic/paper_l2_lh_xstd_train200_500_5seeds

    # Train=1000 extension on the same paper synthetic families.
    python experiments/synthetic/compare_paper_synthetic_baselines.py ^
        --num_exp 5 ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --resume ^
        --out_dir experiments/synthetic/paper_l2_lh_xstd_train1000_5seeds

This runner is intentionally separate from ``compare_synthetic_baselines.py``.
It uses the historical paper data-generation code:

- L2/phantom: ``new_highdata_gen.generate_data_phantom`` with image boundary
  cases from ``data_generate.generate_Y_from_image``.
- LH: ``new_highdata_gen_utils.generate_data`` with ``caseno=0`` behavior.
- High-dimensional observed features: the Appendix-C RFF expansion from
  ``new_highdata_gen.sample_rff_params`` and ``make_rff_features``.

All reported methods use train-only X standardization after the expansion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.baselines import run_dkl_svgp_regression, run_xgboost_bootstrap_regression  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from experiments.synthetic.compare_lmjgp_tricks_synth import _train_one_variant  # noqa: E402
from experiments.synthetic.compare_synthetic_baselines import (  # noqa: E402
    _run_cem_transductive_vi_fullcov,
    _run_jgp_pca,
)
from experiments.uci.compare_uci_baselines import _run_djgp_block  # noqa: E402
from data_gen.highdata import generate_data_phantom, make_rff_features, sample_rff_params  # noqa: E402
from data_gen.highdata_utils import generate_data as generate_lh_data  # noqa: E402


METHOD_ORDER = (
    "xgb_bootstrap",
    "dkl_svgp",
    "jgp_pca",
    "lmjgp_baseline",
    "lmjgp_pls_kl",
    "cem_em_transductive_vi_fullcov_ensemble",
)

SETTING_PRESETS: dict[str, dict[str, Any]] = {
    "l2_q2_train200": {
        "family": "L2_phantom",
        "N_train": 200,
        "N_test": 200,
        "d": 2,
        "H": 20,
        "Q_true": 2,
        "Q": 2,
        "n": 25,
        "caseno": 6,
        "noise_std": 0.5,
        "noise_var": 0.25,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
    "l2_q2_train500": {
        "family": "L2_phantom",
        "N_train": 500,
        "N_test": 200,
        "d": 2,
        "H": 20,
        "Q_true": 2,
        "Q": 2,
        "n": 25,
        "caseno": 6,
        "noise_std": 0.5,
        "noise_var": 0.25,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
    "l2_q2_train1000": {
        "family": "L2_phantom",
        "N_train": 1000,
        "N_test": 200,
        "d": 2,
        "H": 20,
        "Q_true": 2,
        "Q": 2,
        "n": 25,
        "caseno": 6,
        "noise_std": 0.5,
        "noise_var": 0.25,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
    "lh_q5_train200": {
        "family": "LH",
        "N_train": 200,
        "N_test": 200,
        "d": 5,
        "H": 30,
        "Q_true": 5,
        "Q": 5,
        "n": 35,
        "caseno": 0,
        "noise_std": 2.0,
        "noise_var": 4.0,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
    "lh_q5_train500": {
        "family": "LH",
        "N_train": 500,
        "N_test": 200,
        "d": 5,
        "H": 30,
        "Q_true": 5,
        "Q": 5,
        "n": 35,
        "caseno": 0,
        "noise_std": 2.0,
        "noise_var": 4.0,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
    "lh_q5_train1000": {
        "family": "LH",
        "N_train": 1000,
        "N_test": 200,
        "d": 5,
        "H": 30,
        "Q_true": 5,
        "Q": 5,
        "n": 35,
        "caseno": 0,
        "noise_std": 2.0,
        "noise_var": 4.0,
        "expansion": "rff",
        "lengthscale": 0.5,
        "kernel_var": 9.0,
    },
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_numpy_1d(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _as_numpy_2d(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _generate_paper_data(cfg: dict[str, Any], seed: int, device: torch.device) -> dict[str, np.ndarray]:
    """Generate one split from the paper-style synthetic code path."""
    _set_seed(seed)
    family = str(cfg["family"])
    n_train = int(cfg["N_train"])
    n_test = int(cfg["N_test"])
    d = int(cfg["d"])
    h = int(cfg["H"])

    if family == "LH":
        x_train, y_train, x_test, y_test = generate_lh_data(
            d=d,
            N=n_train,
            Nt=n_test,
            noise_var=float(cfg["noise_var"]),
            device=str(device),
        )
    elif family == "L2_phantom":
        # data_generate.generate_Y_from_image reads bound*.png from CWD.
        old_cwd = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT / "src" / "JumpGaussianProcess")
            x_train, y_train, x_test, y_test = generate_data_phantom(
                n_train,
                n_test,
                str(device),
                caseno=int(cfg["caseno"]),
                noise_std=float(cfg["noise_std"]),
            )
        finally:
            os.chdir(old_cwd)
    else:
        raise ValueError(f"Unknown paper synthetic family: {family}")

    expansion = str(cfg.get("expansion", "rff"))
    if expansion != "rff":
        raise NotImplementedError("This corrected runner currently supports the paper RFF expansion only.")

    omega, phases = sample_rff_params(d, h, float(cfg["lengthscale"]), device)
    X_train = make_rff_features(x_train.to(device), omega, phases, float(cfg["kernel_var"]))
    X_test = make_rff_features(x_test.to(device), omega, phases, float(cfg["kernel_var"]))

    return {
        "X_train": _as_numpy_2d(X_train),
        "y_train": _as_numpy_1d(y_train),
        "X_test": _as_numpy_2d(X_test),
        "y_test": _as_numpy_1d(y_test),
        "z_train": _as_numpy_2d(x_train),
        "z_test": _as_numpy_2d(x_test),
    }


def _run_lmjgp_baseline(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    *,
    Q: int,
    n_neighbors: int,
    steps: int,
    MC_num: int,
    device: torch.device,
) -> list[float]:
    args_ns = Namespace(Q=Q, m1=2, m2=40, n=n_neighbors, num_steps=steps, MC_num=MC_num, lr=0.01)
    return _run_djgp_block(
        torch.from_numpy(X_train_s).float().to(device),
        torch.from_numpy(np.asarray(y_train, dtype=np.float32)).float().to(device),
        torch.from_numpy(X_test_s).float().to(device),
        torch.from_numpy(np.asarray(y_test, dtype=np.float32)).float().to(device),
        args_ns,
        device,
    )


def _dummy_truth(Q: int, D: int) -> np.ndarray:
    W = np.zeros((Q, D), dtype=np.float64)
    for q in range(Q):
        W[q, q % D] = 1.0
    return W


def _run_lmjgp_pls_kl(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    Q: int,
    n_neighbors: int,
    steps: int,
    MC_num: int,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    t0 = time.perf_counter()
    info = _train_one_variant(
        "pls_kl",
        X_train=torch.from_numpy(X_train_s).float().to(device),
        Y_train=torch.from_numpy(np.asarray(y_train, dtype=np.float32)).float().to(device),
        X_test=torch.from_numpy(X_test_s).float().to(device),
        Y_test=torch.from_numpy(np.asarray(y_test, dtype=np.float32)).float().to(device),
        W_true=_dummy_truth(Q, X_train_s.shape[1]),
        Q=Q,
        Q_true=Q,
        m1=2,
        m2=40,
        n=n_neighbors,
        num_steps=steps,
        lr=0.01,
        log_interval=max(50, steps),
        MC_num=MC_num,
        warmup_frac=0.5,
        row_norm_weight=0.1,
        row_norm_mode="second_moment",
        aux_metric_weight=0.1,
        aux_metric_mode="A",
        aux_max_pairs=64,
        snapshot_steps=[0, min(50, steps), steps],
        init_seed=int(seed) * 1019 + 11,
        device=device,
        with_projection_ablation=False,
        proj_seed_offset=0,
    )
    if "error" in info:
        raise RuntimeError(info["error"])
    pack = list(info["method_results"]["lmjgp_predict_vi"])
    pack[2] = time.perf_counter() - t0
    pointfiltered = dict(info.get("pointfiltered_results", {}).get("lmjgp_predict_vi", {}))
    if pointfiltered:
        pointfiltered["wall_sec"] = float(pack[2])
    return pack, pointfiltered


def _write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metric_keys(rows: list[dict]) -> list[str]:
    preferred = [
        "setting",
        "family",
        "expansion",
        "caseno",
        "seed",
        "method",
        "N_train",
        "N_test",
        "latent_dim",
        "observed_dim",
        "Q_true",
        "Q",
        "n",
        "standardize_x",
        "noise_std",
        "noise_var",
        "lengthscale",
        "kernel_var",
        *RESULT_ROW_KEYS,
        "n_seeds",
        "status",
        "error",
    ]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    return [k for k in preferred if k in all_keys] + sorted(all_keys - set(preferred))


def _float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, str) and x.strip() == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["setting"]), str(row["method"])), []).append(row)
    out: list[dict] = []
    for (setting, method), group in sorted(groups.items()):
        agg: dict[str, Any] = {"setting": setting, "method": method, "n_seeds": len(group)}
        for meta in (
            "family",
            "expansion",
            "caseno",
            "N_train",
            "N_test",
            "latent_dim",
            "observed_dim",
            "Q_true",
            "Q",
            "n",
            "standardize_x",
            "noise_std",
            "noise_var",
            "lengthscale",
            "kernel_var",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in RESULT_ROW_KEYS:
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        for key in ("pca_explained_variance_sum", "vi_kl", "vi_best_loss"):
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            if np.isfinite(vals).any():
                agg[f"{key}_mean"] = float(np.nanmean(vals))
        out.append(agg)
    return out


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Paper Synthetic L2/LH X-Standardized Baselines

Purpose: corrected baseline comparison on the paper-style synthetic generators,
not the direct low-rank Gaussian-X toy generator.

Settings:

```json
{json.dumps(summary, indent=2)}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method metrics.
- `aggregate.csv`: means/stds over seeds.
- `summary.json`: exact runner arguments and setting metadata.

All methods use train-only X standardization after the RFF expansion.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-style synthetic L2/LH baseline comparison.")
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument(
        "--settings",
        type=str,
        default="l2_q2_train200,l2_q2_train500,lh_q5_train200,lh_q5_train500",
    )
    p.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--dkl_epochs", type=int, default=80)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--cem_outer", type=int, default=5)
    p.add_argument("--cem_m_steps", type=int, default=80)
    p.add_argument("--cem_vi_steps", type=int, default=80)
    p.add_argument("--cem_vi_lr", type=float, default=0.005)
    p.add_argument("--cem_vi_mc", type=int, default=2)
    p.add_argument("--cem_vi_kl_weight", type=float, default=1.0)
    p.add_argument("--cem_vi_init_std", type=float, default=0.03)
    p.add_argument("--out_dir", type=str, default="experiments/synthetic/paper_l2_lh_xstd_train200_500_seed1")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown_settings:
        raise ValueError(f"Unknown settings {unknown_settings}; choose from {sorted(SETTING_PRESETS)}")
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    unknown_methods = sorted(set(methods) - set(METHOD_ORDER))
    if unknown_methods:
        raise ValueError(f"Unknown methods {unknown_methods}; choose from {METHOD_ORDER}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = _read_csv(out_dir / "metrics_partial.csv") if args.resume else []
    if args.resume and not metric_rows:
        metric_rows = _read_csv(out_dir / "metrics.csv")
    completed = {
        (str(r.get("setting")), int(r.get("seed")), str(r.get("method")))
        for r in metric_rows
        if str(r.get("status")) == "ok" and str(r.get("seed", "")).strip() != ""
    }

    pointfiltered_rows: list[dict[str, Any]] = []

    for setting in settings:
        cfg = dict(SETTING_PRESETS[setting])
        for seed in range(int(args.num_exp)):
            data = _generate_paper_data(cfg, seed, device)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(data["X_train"])
            X_test_s = scaler.transform(data["X_test"])
            y_train = data["y_train"]
            y_test = data["y_test"]

            for method in methods:
                if (setting, seed, method) in completed:
                    print(f"skip completed setting={setting} seed={seed} method={method}", flush=True)
                    continue
                print(f"setting={setting} seed={seed} method={method}", flush=True)
                row: dict[str, Any] = {
                    "setting": setting,
                    "family": cfg["family"],
                    "expansion": cfg["expansion"],
                    "caseno": cfg["caseno"],
                    "seed": seed,
                    "method": method,
                    "N_train": int(cfg["N_train"]),
                    "N_test": int(cfg["N_test"]),
                    "latent_dim": int(cfg["d"]),
                    "observed_dim": int(cfg["H"]),
                    "Q_true": int(cfg["Q_true"]),
                    "Q": int(cfg["Q"]),
                    "n": int(cfg["n"]),
                    "standardize_x": True,
                    "noise_std": float(cfg["noise_std"]),
                    "noise_var": float(cfg["noise_var"]),
                    "lengthscale": float(cfg["lengthscale"]),
                    "kernel_var": float(cfg["kernel_var"]),
                }
                try:
                    if method == "xgb_bootstrap":
                        rmse, crps, elapsed, details = run_xgboost_bootstrap_regression(
                            X_train_s,
                            y_train,
                            X_test_s,
                            y_test,
                            n_bootstrap=int(args.xgb_boots),
                            random_state=seed,
                        )
                        pack = pack_gaussian_uq_list(rmse, crps, elapsed, y_test, details["mu"], details["sigma"])
                        diag = {"xgb_boots": int(args.xgb_boots)}
                    elif method == "dkl_svgp":
                        rmse, crps, elapsed, mu, sigma = run_dkl_svgp_regression(
                            X_train_s,
                            y_train,
                            X_test_s,
                            y_test,
                            device=device,
                            epochs=int(args.dkl_epochs),
                            seed=seed,
                        )
                        pack = pack_gaussian_uq_list(rmse, crps, elapsed, y_test, mu, sigma)
                        diag = {"dkl_epochs": int(args.dkl_epochs)}
                    elif method == "jgp_pca":
                        pack, diag = _run_jgp_pca(
                            X_train_s,
                            y_train,
                            X_test_s,
                            y_test,
                            Q=int(cfg["Q"]),
                            n_neighbors=int(cfg["n"]),
                        )
                    elif method == "lmjgp_baseline":
                        pack = _run_lmjgp_baseline(
                            X_train_s,
                            y_train,
                            X_test_s,
                            y_test,
                            Q=int(cfg["Q"]),
                            n_neighbors=int(cfg["n"]),
                            steps=int(args.lmjgp_steps),
                            MC_num=int(args.MC_num),
                            device=device,
                        )
                        diag = {"lmjgp_steps": int(args.lmjgp_steps)}
                    elif method == "lmjgp_pls_kl":
                        pack, pointfiltered = _run_lmjgp_pls_kl(
                            X_train_s,
                            y_train,
                            X_test_s,
                            y_test,
                            seed=seed,
                            Q=int(cfg["Q"]),
                            n_neighbors=int(cfg["n"]),
                            steps=int(args.lmjgp_steps),
                            MC_num=int(args.MC_num),
                            device=device,
                        )
                        diag = {"lmjgp_steps": int(args.lmjgp_steps), "truth_diag_skipped": True}
                    elif method == "cem_em_transductive_vi_fullcov_ensemble":
                        pack, diag = _run_cem_transductive_vi_fullcov(
                            X_train_s,
                            y_train,
                            X_test_s,
                            y_test,
                            None,
                            seed=seed,
                            Q=int(cfg["Q"]),
                            Q_true=int(cfg["Q_true"]),
                            n_neighbors=int(cfg["n"]),
                            MC_num=int(args.MC_num),
                            cem_outer=int(args.cem_outer),
                            cem_m_steps=int(args.cem_m_steps),
                            cem_vi_steps=int(args.cem_vi_steps),
                            cem_vi_lr=float(args.cem_vi_lr),
                            cem_vi_mc=int(args.cem_vi_mc),
                            cem_vi_kl_weight=float(args.cem_vi_kl_weight),
                            cem_vi_init_std=float(args.cem_vi_init_std),
                            device=device,
                        )
                    else:
                        raise ValueError(f"Unhandled method: {method}")

                    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
                    row.update(diag)
                    row["status"] = "ok"
                    if method == "lmjgp_pls_kl" and pointfiltered:
                        pf_row = dict(row)
                        pf_row.update(pointfiltered)
                        pf_row["status"] = "ok"
                        pointfiltered_rows.append(pf_row)
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"ERROR setting={setting} seed={seed} method={method}: {row['error']}", flush=True)
                metric_rows.append(row)
                _write_csv(out_dir / "metrics_partial.csv", metric_rows, _metric_keys(metric_rows))

    aggregate_rows = _aggregate(metric_rows)
    _write_csv(out_dir / "metrics.csv", metric_rows, _metric_keys(metric_rows))
    _write_csv(out_dir / "aggregate.csv", aggregate_rows, _metric_keys(aggregate_rows))
    if pointfiltered_rows:
        existing_pf = _read_csv(out_dir / "metrics_pointfiltered.csv") if args.resume else []
        pf_by_key = {
            (str(r.get("setting")), int(r.get("seed", -1)), str(r.get("method"))): r
            for r in existing_pf
            if str(r.get("seed", "")).strip() != ""
        }
        for r in pointfiltered_rows:
            pf_by_key[(str(r.get("setting")), int(r.get("seed", -1)), str(r.get("method")))] = r
        merged_pf = list(pf_by_key.values())
        _write_csv(out_dir / "metrics_pointfiltered.csv", merged_pf, _metric_keys(merged_pf))
        _write_csv(out_dir / "aggregate_pointfiltered.csv", _aggregate(merged_pf), _metric_keys(_aggregate(merged_pf)))
    summary = {
        "args": vars(args),
        "device": str(device),
        "settings": {name: SETTING_PRESETS[name] for name in settings},
        "n_rows": len(metric_rows),
        "n_ok": int(sum(1 for r in metric_rows if r.get("status") == "ok")),
        "n_errors": int(sum(1 for r in metric_rows if r.get("status") == "error")),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_readme(out_dir, summary)
    print(f"Saved results to {out_dir}", flush=True)
    return {"rows": metric_rows, "aggregate": aggregate_rows, "summary": summary}


if __name__ == "__main__":
    main()

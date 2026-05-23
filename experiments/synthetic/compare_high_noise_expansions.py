"""Compare high-noise orthogonal-mixing expansions for L2/LH synthetic data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Seed-0 pilot with X/Y standardization and four non-DMGP baselines.
    python experiments/synthetic/compare_high_noise_expansions.py ^
        --num_exp 1 ^
        --settings l2_linear_noise3_train1000,l2_nonlinear_noise3_train1000,lh_linear_noise3_train1000,lh_nonlinear_noise3_train1000 ^
        --methods xgb_bootstrap,dkl_svgp,lmjgp_baseline,lmjgp_pls_kl ^
        --standardize_y ^
        --out_dir experiments/synthetic/high_noise_expansions_seed0

    # Export the same standardized splits for DeepMahalanobisGP.
    python experiments/synthetic/compare_high_noise_expansions.py ^
        --num_exp 1 --only_export_dmgp --standardize_y ^
        --out_dir experiments/synthetic/high_noise_expansions_seed0

Then run DMGP from its TensorFlow-1 environment:

    conda activate dmgp_tf1
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP
    python experiments/synthetic/run_deep_mahalanobis_gp_paper.py ^
        --data_dir experiments/synthetic/high_noise_expansions_seed0/dmgp_data ^
        --out_dir experiments/synthetic/high_noise_expansions_seed0/dmgp ^
        --settings l2_linear_noise3_train1000,l2_nonlinear_noise3_train1000,lh_linear_noise3_train1000,lh_nonlinear_noise3_train1000 ^
        --seed 0 --steps1 300 --steps2 700

This stress-test keeps the paper L2/LH latent generators and responses, but
replaces the Appendix-C RFF expansion with noisy observed features:

``linear_noise``:
    ``x = Q [A z, eta]``.

``nonlinear_noise``:
    ``x = Q [z, z^2, sin(pi z), eta]``.

Here ``Q`` is a random orthogonal mixing matrix and
``eta ~ N(0, eta_sigma^2 I)``.  Metrics are reported in standardized-y units
when ``--standardize_y`` is enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS as PAPER_PRESETS,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    _generate_paper_data,
    _run_lmjgp_baseline,
    _run_lmjgp_pls_kl,
    _set_seed,
)


METHOD_ORDER = ("xgb_bootstrap", "dkl_svgp", "lmjgp_baseline", "lmjgp_pls_kl")


def _make_setting_presets() -> dict[str, dict[str, Any]]:
    presets: dict[str, dict[str, Any]] = {}
    for family_key, base_name in (("l2", "l2_q2_train1000"), ("lh", "lh_q5_train1000")):
        base = dict(PAPER_PRESETS[base_name])
        for expansion in ("linear_noise", "nonlinear_noise"):
            name = f"{family_key}_{'linear' if expansion == 'linear_noise' else 'nonlinear'}_noise3_train1000"
            cfg = dict(base)
            cfg.update(
                {
                    "base_setting": base_name,
                    "expansion": expansion,
                    "eta_sigma": 3.0,
                    "signal_scale": 1.0,
                    "observed_dim": int(base["H"]),
                }
            )
            presets[name] = cfg
    return presets


SETTING_PRESETS = _make_setting_presets()


def _random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(mat)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def _linear_noise_expansion(
    z: np.ndarray,
    *,
    observed_dim: int,
    eta_sigma: float,
    signal_scale: float,
    rng: np.random.Generator,
    A: np.ndarray | None = None,
    Qmix: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    z = np.asarray(z, dtype=np.float64)
    n, latent_dim = z.shape
    if observed_dim <= latent_dim:
        raise ValueError("observed_dim must exceed latent_dim for noisy expansion.")
    if A is None:
        A = _random_orthogonal(latent_dim, rng)
    if Qmix is None:
        Qmix = _random_orthogonal(observed_dim, rng)
    signal = float(signal_scale) * (z @ A.T)
    eta = float(eta_sigma) * rng.standard_normal((n, observed_dim - signal.shape[1]))
    raw = np.concatenate([signal, eta], axis=1)
    return raw @ Qmix.T, {"A": A, "Qmix": Qmix, "signal_dim": int(signal.shape[1])}


def _nonlinear_noise_expansion(
    z: np.ndarray,
    *,
    observed_dim: int,
    eta_sigma: float,
    signal_scale: float,
    rng: np.random.Generator,
    Qmix: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    z = np.asarray(z, dtype=np.float64)
    n, latent_dim = z.shape
    signal = np.concatenate([z, z**2, np.sin(np.pi * z)], axis=1)
    signal = float(signal_scale) * signal
    if observed_dim <= signal.shape[1]:
        raise ValueError("observed_dim must exceed 3 * latent_dim for nonlinear noisy expansion.")
    if Qmix is None:
        Qmix = _random_orthogonal(observed_dim, rng)
    eta = float(eta_sigma) * rng.standard_normal((n, observed_dim - signal.shape[1]))
    raw = np.concatenate([signal, eta], axis=1)
    return raw @ Qmix.T, {"Qmix": Qmix, "signal_dim": int(signal.shape[1])}


def _generate_high_noise_data(cfg: dict[str, Any], seed: int, device: torch.device) -> dict[str, Any]:
    base_cfg = dict(PAPER_PRESETS[str(cfg["base_setting"])])
    latent = _generate_paper_data(base_cfg, seed, device)
    z_train = np.asarray(latent["z_train"], dtype=np.float64)
    z_test = np.asarray(latent["z_test"], dtype=np.float64)
    rng = np.random.default_rng(seed * 1009 + 17)
    kwargs = {
        "observed_dim": int(cfg["observed_dim"]),
        "eta_sigma": float(cfg["eta_sigma"]),
        "signal_scale": float(cfg["signal_scale"]),
        "rng": rng,
    }
    if cfg["expansion"] == "linear_noise":
        X_train, params = _linear_noise_expansion(z_train, **kwargs)
        X_test, _ = _linear_noise_expansion(
            z_test,
            observed_dim=int(cfg["observed_dim"]),
            eta_sigma=float(cfg["eta_sigma"]),
            signal_scale=float(cfg["signal_scale"]),
            rng=rng,
            A=params["A"],
            Qmix=params["Qmix"],
        )
    elif cfg["expansion"] == "nonlinear_noise":
        X_train, params = _nonlinear_noise_expansion(z_train, **kwargs)
        X_test, _ = _nonlinear_noise_expansion(
            z_test,
            observed_dim=int(cfg["observed_dim"]),
            eta_sigma=float(cfg["eta_sigma"]),
            signal_scale=float(cfg["signal_scale"]),
            rng=rng,
            Qmix=params["Qmix"],
        )
    else:
        raise ValueError(f"Unknown expansion {cfg['expansion']!r}")
    return {
        "X_train": X_train.astype(np.float64),
        "X_test": X_test.astype(np.float64),
        "y_train": np.asarray(latent["y_train"], dtype=np.float64).reshape(-1),
        "y_test": np.asarray(latent["y_test"], dtype=np.float64).reshape(-1),
        "z_train": z_train,
        "z_test": z_test,
        "expansion_params": {k: v for k, v in params.items() if k not in {"A", "Qmix"}},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "setting",
        "family",
        "base_setting",
        "expansion",
        "eta_sigma",
        "seed",
        "method",
        "status",
        "N_train",
        "N_test",
        "latent_dim",
        "observed_dim",
        "Q",
        "n",
        "standardize_x",
        "standardize_y",
        *RESULT_ROW_KEYS,
        "error",
    ]
    keys_seen: set[str] = set()
    for row in rows:
        keys_seen.update(row.keys())
    keys = [key for key in preferred if key in keys_seen]
    keys.extend(sorted(keys_seen - set(keys)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, str) and x.strip() == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["setting"]), str(row["method"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (setting, method), group in sorted(groups.items()):
        agg: dict[str, Any] = {"setting": setting, "method": method, "n_seeds": len(group)}
        for meta in (
            "family",
            "base_setting",
            "expansion",
            "eta_sigma",
            "N_train",
            "N_test",
            "latent_dim",
            "observed_dim",
            "Q",
            "n",
            "standardize_x",
            "standardize_y",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in RESULT_ROW_KEYS:
            vals = np.asarray([_float_or_nan(row.get(key)) for row in group], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _base_row(setting: str, cfg: dict[str, Any], seed: int, method: str, *, standardize_y: bool) -> dict[str, Any]:
    return {
        "setting": setting,
        "family": cfg["family"],
        "base_setting": cfg["base_setting"],
        "expansion": cfg["expansion"],
        "eta_sigma": float(cfg["eta_sigma"]),
        "seed": int(seed),
        "method": method,
        "N_train": int(cfg["N_train"]),
        "N_test": int(cfg["N_test"]),
        "latent_dim": int(cfg["d"]),
        "observed_dim": int(cfg["observed_dim"]),
        "Q": int(cfg["Q"]),
        "n": int(cfg["n"]),
        "standardize_x": True,
        "standardize_y": bool(standardize_y),
    }


def _standardize_split(data: dict[str, Any], *, standardize_y: bool) -> dict[str, Any]:
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(data["X_train"])
    X_test_s = x_scaler.transform(data["X_test"])
    y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
    y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)
    y_scaler = StandardScaler() if standardize_y else None
    if y_scaler is not None:
        y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
    else:
        y_train_s = y_train
        y_test_s = y_test
    return {
        "X_train": X_train_s.astype(np.float64),
        "X_test": X_test_s.astype(np.float64),
        "y_train": y_train_s.astype(np.float64),
        "y_test": y_test_s.astype(np.float64),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "y_original_train": y_train,
        "y_original_test": y_test,
    }


def _export_dmgp_npz(out_dir: Path, setting: str, seed: int, split: dict[str, Any], data: dict[str, Any]) -> None:
    path = out_dir / "dmgp_data" / f"{setting}_seed{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_train=np.asarray(split["X_train"], dtype=np.float64),
        X_test=np.asarray(split["X_test"], dtype=np.float64),
        y_train=np.asarray(split["y_train"], dtype=np.float64).reshape(-1),
        y_test=np.asarray(split["y_test"], dtype=np.float64).reshape(-1),
        y_train_std=np.asarray(split["y_train"], dtype=np.float64).reshape(-1, 1),
        y_test_std=np.asarray(split["y_test"], dtype=np.float64).reshape(-1, 1),
        y_original_train=np.asarray(data["y_train"], dtype=np.float64).reshape(-1),
        y_original_test=np.asarray(data["y_test"], dtype=np.float64).reshape(-1),
        y_mean=np.asarray([0.0], dtype=np.float64),
        y_scale=np.asarray([1.0], dtype=np.float64),
        z_train=np.asarray(data["z_train"], dtype=np.float64),
        z_test=np.asarray(data["z_test"], dtype=np.float64),
    )


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# High-Noise Orthogonal Synthetic Expansions

Purpose: seed-level stress-test of noisy observed expansions for paper L2/LH
latent synthetic generators.

Settings and args:

```json
{json.dumps(summary, indent=2)}
```

Files:

- `metrics.csv`: non-DMGP per-method rows.
- `aggregate.csv`: aggregate over completed seeds.
- `dmgp_data/*.npz`: standardized X/Y exports for DeepMahalanobisGP.
- `dmgp/`: optional output from `run_deep_mahalanobis_gp_paper.py`.

Metrics are in standardized-y units when `standardize_y=true`.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-noise orthogonal expansion synthetic comparison.")
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--settings", type=str, default=",".join(SETTING_PRESETS))
    p.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    p.add_argument("--xgb_boots", type=int, default=30)
    p.add_argument("--dkl_epochs", type=int, default=80)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--standardize_y", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--only_export_dmgp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out_dir", type=str, default="experiments/synthetic/high_noise_expansions_seed0")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    unknown_methods = sorted(set(methods) - set(METHOD_ORDER))
    if unknown_settings:
        raise ValueError(f"Unknown settings: {unknown_settings}. Choices: {sorted(SETTING_PRESETS)}")
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}. Choices: {METHOD_ORDER}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _read_csv(out_dir / "metrics_partial.csv") if args.resume else []
    if args.resume and not rows:
        rows = _read_csv(out_dir / "metrics.csv")
    completed = {
        (str(row.get("setting")), int(row.get("seed")), str(row.get("method")))
        for row in rows
        if row.get("status") == "ok" and str(row.get("seed", "")).strip()
    }

    export_rows: list[dict[str, Any]] = []
    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
        _set_seed(seed)
        for setting in settings:
            cfg = dict(SETTING_PRESETS[setting])
            print(f"generate setting={setting} seed={seed}", flush=True)
            data = _generate_high_noise_data(cfg, seed, device)
            split = _standardize_split(data, standardize_y=bool(args.standardize_y))
            _export_dmgp_npz(out_dir, setting, seed, split, data)
            export_rows.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "path": str(out_dir / "dmgp_data" / f"{setting}_seed{seed}.npz"),
                    "standardize_x": True,
                    "standardize_y": bool(args.standardize_y),
                }
            )
            if args.only_export_dmgp:
                continue
            for method in methods:
                if (setting, seed, method) in completed:
                    print(f"skip completed setting={setting} seed={seed} method={method}", flush=True)
                    continue
                print(f"setting={setting} seed={seed} method={method}", flush=True)
                row = _base_row(setting, cfg, seed, method, standardize_y=bool(args.standardize_y))
                try:
                    if method == "xgb_bootstrap":
                        rmse, crps, elapsed, details = run_xgboost_bootstrap_regression(
                            split["X_train"],
                            split["y_train"],
                            split["X_test"],
                            split["y_test"],
                            n_bootstrap=int(args.xgb_boots),
                            random_state=seed,
                        )
                        pack = pack_gaussian_uq_list(
                            rmse,
                            crps,
                            elapsed,
                            split["y_test"],
                            details["mu"],
                            details["sigma"],
                        )
                    elif method == "dkl_svgp":
                        rmse, crps, elapsed, mu, sigma = run_dkl_svgp_regression(
                            split["X_train"],
                            split["y_train"],
                            split["X_test"],
                            split["y_test"],
                            device=device,
                            epochs=int(args.dkl_epochs),
                            seed=seed,
                        )
                        pack = pack_gaussian_uq_list(rmse, crps, elapsed, split["y_test"], mu, sigma)
                    elif method == "lmjgp_baseline":
                        pack = _run_lmjgp_baseline(
                            split["X_train"],
                            split["y_train"],
                            split["X_test"],
                            split["y_test"],
                            Q=int(cfg["Q"]),
                            n_neighbors=int(cfg["n"]),
                            steps=int(args.lmjgp_steps),
                            MC_num=int(args.MC_num),
                            device=device,
                        )
                    elif method == "lmjgp_pls_kl":
                        pack = _run_lmjgp_pls_kl(
                            split["X_train"],
                            split["y_train"],
                            split["X_test"],
                            split["y_test"],
                            seed=seed,
                            Q=int(cfg["Q"]),
                            n_neighbors=int(cfg["n"]),
                            steps=int(args.lmjgp_steps),
                            MC_num=int(args.MC_num),
                            device=device,
                        )
                    else:
                        raise ValueError(method)
                    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
                    row["status"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"ERROR setting={setting} seed={seed} method={method}: {row['error']}", flush=True)
                rows.append(row)
                _write_csv(out_dir / "metrics_partial.csv", rows)

    agg = _aggregate(rows)
    _write_csv(out_dir / "metrics.csv", rows)
    _write_csv(out_dir / "aggregate.csv", agg)
    summary = {
        "args": vars(args),
        "device": str(device),
        "settings": {name: SETTING_PRESETS[name] for name in settings},
        "exports": export_rows,
        "n_rows": len(rows),
        "n_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "n_errors": sum(1 for row in rows if row.get("status") == "error"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_readme(out_dir, summary)
    print(f"saved artifacts to {out_dir}", flush=True)
    return {"rows": rows, "aggregate": agg, "summary": summary}


if __name__ == "__main__":
    main()

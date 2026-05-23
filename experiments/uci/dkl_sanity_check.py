"""Run DKL-SVGP preprocessing and hyperparameter diagnostics for UCI splits.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Parkinsons grouped seed-0 raw-vs-std_x diagnostic.
    python experiments/uci/dkl_sanity_check.py ^
        --dataset "Parkinsons Telemonitoring" ^
        --train_sizes 200,500,1000 --seed 0 --epochs 80 ^
        --out_dir experiments/uci/dkl_sanity_parkinsons_grouped_seed0

This diagnostic intentionally mirrors ``djgp.baselines.run_dkl_svgp_regression``
but records internal quantities such as learned kernel lengthscales, likelihood
noise, predictive sigma scale, and train-only scaler checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import gpytorch
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.baselines.dkl import _FeatureMap, _SVGPOnFeatures  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from experiments.uci.compare_smalltrain_multiseed_grouped import _load_dataset_for_seed  # noqa: E402


def _write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _run_dkl_with_diagnostics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    device: torch.device,
    d_hidden: int,
    d_latent: int,
    num_inducing: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[list[float], dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).reshape(-1)

    y_mean = float(y_train.mean())
    y_std = float(max(y_train.std(), 1e-6))
    y_train_z = (y_train - y_mean) / y_std
    y_test_z = (y_test - y_mean) / y_std

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train_z).to(device)
    Xv = torch.from_numpy(X_test).to(device)
    yv = torch.from_numpy(y_test_z).to(device)

    features = _FeatureMap(X_train.shape[1], d_hidden, d_latent).to(device)
    with torch.no_grad():
        z_init = features(Xt[: min(512, len(Xt))])
        idx = torch.randperm(z_init.size(0), device=device)[:num_inducing]
        inducing = z_init[idx].clone()

    svgp = _SVGPOnFeatures(inducing).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.noise = torch.tensor(0.01, device=device)
    mll = gpytorch.mlls.VariationalELBO(likelihood, svgp, num_data=len(Xt))
    optimizer = torch.optim.Adam(
        list(features.parameters()) + list(svgp.parameters()) + list(likelihood.parameters()),
        lr=lr,
    )
    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=min(batch_size, len(Xt)),
        shuffle=True,
    )

    losses: list[float] = []
    t0 = time.perf_counter()
    features.train()
    svgp.train()
    likelihood.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            hb = features(xb)
            with gpytorch.settings.cholesky_jitter(1e-3):
                out = svgp(hb)
                loss = -mll(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    features.eval()
    svgp.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        h_train = features(Xt)
        h_test = features(Xv)
        pred = likelihood(svgp(h_test))

    mu_z = pred.mean
    var_z = pred.variance
    sigma_z = torch.sqrt(torch.clamp(var_z, min=1e-12))
    mu = mu_z * y_std + y_mean
    sigma = sigma_z * y_std
    y_orig = yv * y_std + y_mean
    rmse = torch.sqrt(torch.mean((mu - y_orig) ** 2))
    crps = mean_gaussian_crps_torch(y_orig, mu, sigma)
    elapsed = time.perf_counter() - t0

    mu_np = mu.detach().cpu().numpy()
    sigma_np = sigma.detach().cpu().numpy()
    metrics = pack_gaussian_uq_list(
        float(rmse.item()),
        float(crps.item()),
        float(elapsed),
        y_test,
        mu_np,
        sigma_np,
    )

    lengthscale = svgp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
    h_train_np = h_train.detach().cpu().numpy()
    h_test_np = h_test.detach().cpu().numpy()
    inducing_np = svgp.variational_strategy.inducing_points.detach().cpu().numpy()
    sigma_np = np.asarray(sigma_np, dtype=np.float64)
    diag = {
        "y_train_mean": y_mean,
        "y_train_std": y_std,
        "loss_first": float(losses[0]) if losses else float("nan"),
        "loss_last": float(losses[-1]) if losses else float("nan"),
        "lengthscale_mean": float(np.mean(lengthscale)),
        "lengthscale_min": float(np.min(lengthscale)),
        "lengthscale_max": float(np.max(lengthscale)),
        "outputscale": float(svgp.covar_module.outputscale.detach().cpu().item()),
        "likelihood_noise": float(likelihood.noise.detach().cpu().item()),
        "latent_train_std_mean": float(np.mean(np.std(h_train_np, axis=0))),
        "latent_test_std_mean": float(np.mean(np.std(h_test_np, axis=0))),
        "inducing_std_mean": float(np.mean(np.std(inducing_np, axis=0))),
        "pred_sigma_mean": float(np.mean(sigma_np)),
        "pred_sigma_min": float(np.min(sigma_np)),
        "pred_sigma_max": float(np.max(sigma_np)),
    }
    return metrics, diag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DKL-SVGP sanity diagnostics for UCI small-train splits.")
    p.add_argument("--dataset", type=str, default="Parkinsons Telemonitoring")
    p.add_argument("--train_sizes", type=str, default="200,500,1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test_anchors", type=int, default=200)
    p.add_argument("--train_subject_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--d_hidden", type=int, default=64)
    p.add_argument("--d_latent", type=int, default=4)
    p.add_argument("--num_inducing", type=int, default=128)
    p.add_argument("--out_dir", type=str, default="experiments/uci/dkl_sanity_parkinsons_grouped_seed0")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    train_sizes = [int(x.strip()) for x in args.train_sizes.split(",") if x.strip()]

    for train_size in train_sizes:
        data = _load_dataset_for_seed(
            args.dataset,
            args.seed,
            max_total_train=train_size,
            max_test_anchors=args.max_test_anchors,
            train_subject_frac=args.train_subject_frac,
        )
        for standardize_x in (False, True):
            x_scaler = StandardScaler() if standardize_x else None
            if x_scaler is None:
                X_train_s = np.asarray(data["X_train"], dtype=np.float64)
                X_test_s = np.asarray(data["X_test"], dtype=np.float64)
                scaler_train_mean_abs = float(np.mean(np.abs(np.mean(X_train_s, axis=0))))
                scaler_test_mean_abs = float(np.mean(np.abs(np.mean(X_test_s, axis=0))))
            else:
                x_scaler.fit(data["X_train"])
                X_train_s = x_scaler.transform(data["X_train"])
                X_test_s = x_scaler.transform(data["X_test"])
                scaler_train_mean_abs = float(np.mean(np.abs(np.mean(X_train_s, axis=0))))
                scaler_test_mean_abs = float(np.mean(np.abs(np.mean(X_test_s, axis=0))))

            metrics, diag = _run_dkl_with_diagnostics(
                X_train_s,
                data["y_train"],
                X_test_s,
                data["y_test"],
                device=device,
                d_hidden=args.d_hidden,
                d_latent=args.d_latent,
                num_inducing=args.num_inducing,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
            )
            row = {
                "dataset": args.dataset,
                "seed": int(args.seed),
                "train_size": int(train_size),
                "standardize_x": bool(standardize_x),
                "x_scaler_fit": "train_only" if standardize_x else "none",
                "x_train_mean_abs_after_scaling": scaler_train_mean_abs,
                "x_test_mean_abs_after_scaling": scaler_test_mean_abs,
                "split_type": data["meta"]["split_type"],
            }
            row.update({key: float(metrics[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
            row.update(diag)
            rows.append(row)
            print(
                f"train={train_size} std_x={standardize_x} "
                f"rmse={row['rmse']:.4f} crps={row['mean_crps']:.4f} "
                f"cov90={row['coverage_90']:.3f} noise={row['likelihood_noise']:.4f}",
                flush=True,
            )

    keys = [
        "dataset",
        "seed",
        "train_size",
        "standardize_x",
        "x_scaler_fit",
        "x_train_mean_abs_after_scaling",
        "x_test_mean_abs_after_scaling",
        "split_type",
        *RESULT_ROW_KEYS,
        "y_train_mean",
        "y_train_std",
        "loss_first",
        "loss_last",
        "lengthscale_mean",
        "lengthscale_min",
        "lengthscale_max",
        "outputscale",
        "likelihood_noise",
        "latent_train_std_mean",
        "latent_test_std_mean",
        "inducing_std_mean",
        "pred_sigma_mean",
        "pred_sigma_min",
        "pred_sigma_max",
    ]
    _write_csv(out_dir / "diagnostics.csv", rows, keys)
    summary = {
        "args": vars(args),
        "device": str(device),
        "diagnostics_csv": "diagnostics.csv",
        "note": "Feature scalers are fit on training rows only; DKL y-standardization is internal and train-only.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# DKL Sanity Check\n\n"
        "Diagnostics for DKL-SVGP preprocessing and learned GP hyperparameters. "
        "`diagnostics.csv` compares raw X and train-only `std_x` using the same seed and split.\n",
        encoding="utf-8",
    )
    return {"rows": rows}


if __name__ == "__main__":
    main()

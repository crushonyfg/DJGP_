"""Compare sequential vs batch JGP-CEM on L2 data with PCA-2D reduction.

How to run from the repository root with conda env ``jumpGP``::

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Quick smoke test (20 test points):
    python experiments/synthetic/compare_jgp_cem_batch_l2_pca.py ^
        --max_test 20 --seed 0 ^
        --out_dir experiments/synthetic/jgp_batch_pca_l2_smoke

    # Full run (200 test points, default):
    python experiments/synthetic/compare_jgp_cem_batch_l2_pca.py ^
        --seed 0 ^
        --out_dir experiments/synthetic/jgp_batch_pca_l2_seed0

    # Specify number of worker processes for batch CEM:
    python experiments/synthetic/compare_jgp_cem_batch_l2_pca.py ^
        --seed 0 --n_jobs 4 ^
        --out_dir experiments/synthetic/jgp_batch_pca_l2_4jobs_seed0

Purpose
-------
The default JGP in this codebase is ``JumpGP_LD`` with ``mode='CEM'``
(Classification EM), executed **sequentially** – one test point at a time.
This script benchmarks a new ``JumpGPBatch`` class that dispatches each
per-test-point CEM fit to a separate worker process via
``concurrent.futures.ProcessPoolExecutor``.

Both methods run on the same data: the paper ``l2_q2_train1000`` synthetic
dataset (2-D latent space, RFF-expanded to 20-D), with the 20-D features
further reduced to 2-D by PCA before JGP is applied.

Generator provenance
--------------------
- Family: L2_phantom (``new_highdata_gen.generate_data_phantom``, caseno=6)
- Expansion: RFF (``new_highdata_gen.make_rff_features``)
- Latent dim d=2, observed dim H=20, PCA target dim=2
- Train / test: 1 000 / 200
- X standardised with train-only StandardScaler
- y left on raw scale (JGP handles its own offset internally)
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── repo root on path ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from data_gen.highdata import generate_data_phantom, make_rff_features, sample_rff_params  # noqa: E402
from JumpGaussianProcess.jumpgp import JumpGP, JumpGPBatch  # noqa: E402

# ── dataset preset (mirrors compare_paper_synthetic_baselines.py) ──────────
L2_PRESET = {
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
}

PCA_DIM = 2          # target dimension after PCA
N_NEIGHBORS = 25     # neighbourhood size for JGP (same as preset 'n')


def _set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _generate_l2_data(seed: int) -> dict:
    """Generate L2 phantom data with RFF expansion; return numpy arrays."""
    _set_seed(seed)
    cfg = L2_PRESET

    import torch
    device = "cpu"

    # generate_data_phantom reads bound*.png from JumpGaussianProcess/
    old_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT / "JumpGaussianProcess")
        x_train_t, y_train_t, x_test_t, y_test_t = generate_data_phantom(
            cfg["N_train"],
            cfg["N_test"],
            device,
            caseno=cfg["caseno"],
            noise_std=cfg["noise_std"],
        )
    finally:
        os.chdir(old_cwd)

    omega, phases = sample_rff_params(
        cfg["d"], cfg["H"], cfg["lengthscale"], device
    )
    X_train_rff = make_rff_features(x_train_t, omega, phases, cfg["kernel_var"])
    X_test_rff  = make_rff_features(x_test_t,  omega, phases, cfg["kernel_var"])

    def _np2d(t):
        return np.asarray(t.detach().cpu() if hasattr(t, "detach") else t).reshape(len(t), -1)

    def _np1d(t):
        return np.asarray(t.detach().cpu() if hasattr(t, "detach") else t).ravel()

    return {
        "X_train": _np2d(X_train_rff),   # (N_train, H=20)
        "y_train": _np1d(y_train_t),
        "X_test":  _np2d(X_test_rff),    # (N_test,  H=20)
        "y_test":  _np1d(y_test_t),
        "z_train": _np2d(x_train_t),     # true 2-D latent
        "z_test":  _np2d(x_test_t),
    }


def _apply_pca(data: dict, n_components: int) -> dict:
    """Standardise X then apply train-fit PCA; return new data dict."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data["X_train"])
    X_test_s  = scaler.transform(data["X_test"])

    pca = PCA(n_components=n_components, random_state=0)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca  = pca.transform(X_test_s)

    explained = float(pca.explained_variance_ratio_.sum())
    print(
        f"PCA {data['X_train'].shape[1]}D → {n_components}D  "
        f"(explained variance {explained:.1%})"
    )

    return {
        **data,
        "X_train_pca": X_train_pca,
        "X_test_pca":  X_test_pca,
        "pca_explained_variance": explained,
    }


def _run_sequential(data: dict, max_test: int, n_neighbors: int) -> dict:
    """Run the standard sequential JGP-CEM and return metrics dict."""
    X_tr = data["X_train_pca"]
    y_tr = data["y_train"]
    X_te = data["X_test_pca"][:max_test]
    y_te = data["y_test"][:max_test]

    t0 = time.perf_counter()
    jgp = JumpGP(X_tr, y_tr, X_te, L=1, M=n_neighbors, mode="CEM", bVerbose=False)
    jgp.fit()
    wall = time.perf_counter() - t0

    rmse, crps = jgp.metrics(y_te)
    mu = np.asarray(jgp.jump_results["mu"]).ravel()
    sig2 = np.asarray(jgp.jump_results["sig2"]).ravel()

    return {
        "method": "jgp_cem_sequential",
        "wall_s": wall,
        "rmse": float(rmse),
        "crps": float(crps),
        "n_test": int(max_test),
        "n_train": int(X_tr.shape[0]),
        "n_neighbors": int(n_neighbors),
        "pca_dim": int(X_tr.shape[1]),
        "mu": mu.tolist(),
        "sig2": sig2.tolist(),
    }


def _run_batch(data: dict, max_test: int, n_neighbors: int, n_jobs) -> dict:
    """Run the parallel batch JGP-CEM and return metrics dict."""
    X_tr = data["X_train_pca"]
    y_tr = data["y_train"]
    X_te = data["X_test_pca"][:max_test]
    y_te = data["y_test"][:max_test]

    t0 = time.perf_counter()
    jgp_b = JumpGPBatch(
        X_tr, y_tr, X_te,
        L=1, M=n_neighbors, mode="CEM",
        n_jobs=n_jobs, bVerbose=False,
    )
    jgp_b.fit()
    wall = time.perf_counter() - t0

    rmse, crps = jgp_b.metrics(y_te)
    mu = np.asarray(jgp_b.jump_results["mu"]).ravel()
    sig2 = np.asarray(jgp_b.jump_results["sig2"]).ravel()

    return {
        "method": "jgp_cem_batch",
        "wall_s": wall,
        "rmse": float(rmse),
        "crps": float(crps),
        "n_test": int(max_test),
        "n_train": int(X_tr.shape[0]),
        "n_neighbors": int(n_neighbors),
        "pca_dim": int(X_tr.shape[1]),
        "n_jobs": n_jobs if n_jobs is not None else "auto",
        "mu": mu.tolist(),
        "sig2": sig2.tolist(),
    }


def _write_results(results: list[dict], out_dir: Path, seed: int,
                   pca_explained: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics CSV (no per-point arrays)
    csv_keys = ["method", "wall_s", "rmse", "crps", "n_test", "n_train",
                "n_neighbors", "pca_dim"]
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in csv_keys if k in r})
    print(f"Wrote {csv_path}")

    # summary JSON
    summary = {
        "seed": seed,
        "setting": "l2_q2_train1000",
        "pca_dim": PCA_DIM,
        "pca_explained_variance": pca_explained,
        "n_neighbors": N_NEIGHBORS,
        "results": [
            {k: v for k, v in r.items() if k not in ("mu", "sig2")}
            for r in results
        ],
    }
    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")

    # predictions npz (for downstream analysis)
    npz_data: dict = {}
    for r in results:
        tag = r["method"]
        npz_data[f"{tag}_mu"]   = np.array(r["mu"])
        npz_data[f"{tag}_sig2"] = np.array(r["sig2"])
    npz_path = out_dir / "predictions.npz"
    np.savez(npz_path, **npz_data)
    print(f"Wrote {npz_path}")

    # human-readable table
    print("\n" + "=" * 62)
    print(f"{'Method':<28} {'Wall(s)':>8} {'RMSE':>8} {'CRPS':>8}")
    print("-" * 62)
    for r in results:
        print(
            f"{r['method']:<28} "
            f"{r['wall_s']:>8.2f} "
            f"{r['rmse']:>8.4f} "
            f"{r['crps']:>8.4f}"
        )
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential vs batch JGP-CEM on L2 data with PCA-2D"
    )
    parser.add_argument("--seed",      type=int,  default=0)
    parser.add_argument("--max_test",  type=int,  default=200,
                        help="Number of test points to evaluate (max 200).")
    parser.add_argument("--n_neighbors", type=int, default=N_NEIGHBORS,
                        help="Neighbourhood size for JGP.")
    parser.add_argument("--n_jobs",    type=int,  default=None,
                        help="Worker processes for batch CEM (None=all cores).")
    parser.add_argument("--out_dir",   type=str,
                        default="experiments/synthetic/jgp_batch_pca_l2_seed0")
    args = parser.parse_args()

    out_dir  = REPO_ROOT / args.out_dir
    max_test = min(args.max_test, L2_PRESET["N_test"])

    print(f"Seed: {args.seed}  |  max_test: {max_test}  |  "
          f"n_neighbors: {args.n_neighbors}  |  n_jobs: {args.n_jobs}")
    print("Generating L2 phantom data ...")
    data = _generate_l2_data(args.seed)
    print(f"  X_train shape (RFF): {data['X_train'].shape}")

    print(f"Applying PCA to {PCA_DIM}D ...")
    data = _apply_pca(data, PCA_DIM)

    results = []

    print("\n--- Sequential JGP-CEM ---")
    seq_res = _run_sequential(data, max_test, args.n_neighbors)
    print(f"  wall={seq_res['wall_s']:.2f}s  RMSE={seq_res['rmse']:.4f}  "
          f"CRPS={seq_res['crps']:.4f}")
    results.append(seq_res)

    print("\n--- Batch JGP-CEM ---")
    bat_res = _run_batch(data, max_test, args.n_neighbors, args.n_jobs)
    print(f"  wall={bat_res['wall_s']:.2f}s  RMSE={bat_res['rmse']:.4f}  "
          f"CRPS={bat_res['crps']:.4f}")
    results.append(bat_res)

    # Sanity check: both methods should give identical predictions
    mu_seq = np.array(seq_res["mu"])
    mu_bat = np.array(bat_res["mu"])
    max_abs_diff = float(np.max(np.abs(mu_seq - mu_bat)))
    print(f"\nMax |mu_seq - mu_batch| = {max_abs_diff:.6f}  "
          f"(should be ~0 – same CEM, just parallel)")

    _write_results(results, out_dir, args.seed, data["pca_explained_variance"])

    # Write README
    readme = (
        f"# JGP-CEM Sequential vs Batch – L2 PCA-2D\n\n"
        f"Seed: {args.seed}, setting: l2_q2_train1000, PCA dim: {PCA_DIM}, "
        f"n_test: {max_test}, n_neighbors: {args.n_neighbors}\n\n"
        f"See `docs/experiments_log.md` for the corresponding entry.\n\n"
        f"Files:\n"
        f"- `metrics.csv` – per-method wall time, RMSE, CRPS\n"
        f"- `summary.json` – full config + results\n"
        f"- `predictions.npz` – per-point mu / sig2 for both methods\n"
    )
    (out_dir / "README.md").write_text(readme)
    print(f"Wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    # ProcessPoolExecutor requires the if-name-main guard on Windows.
    main()

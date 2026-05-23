"""
Low-rank tensor factorisation ablation on **synthetic** data with a known
``W_true ∈ R^{Q_true × D}`` jump structure.

We reuse ``generate_lowrank_jump`` and ``subspace_alignment_to_truth`` from
``experiments.synthetic.compare_lmjgp_tricks_synth`` so the data generator is
identical to our trick experiments. Three low-rank variants are run::

    pls_only       : W_j = W_pls (frozen)
    global_C       : W_j = C_0 V^T (learned globally)
    local_C_smooth : W_j = (C_0 + ΔC_j) V^T (per-anchor + Laplacian smoothness)

For each variant we report:

* the standard 7-tuple ``[rmse, mean_crps, wall_sec, cov90, w90, cov95, w95]``,
* per-anchor ``W`` diagnostics (``trace(W W^T)``, ``eff_rank(W W^T)``,
  ``anchor_dispersion_frob``, cosine to the fixed PLS reference),
* **ground-truth alignment** of ``A_j = W_j^T W_j`` with the row-span of
  ``W_true`` (``capture_ratio_truth``, ``mean_cos_principal_angles``).

How to run (repository root, conda env: ``jumpGP``)::

    conda activate jumpGP

    # Smoke test
    python experiments/synthetic/compare_lowrank_factorization_synth.py \
        --num_seeds 1 --N_train 800 --N_test 64 --D 30 --Q_true 2 --num_steps 200

    # Wider sweep
    python experiments/synthetic/compare_lowrank_factorization_synth.py \
        --num_seeds 3 --N_train 1500 --N_test 128 --D 50 --Q_true 3 \
        --n_pls 5 --n_pca 5 --n_random 0 --num_steps 300

Agents: keep this aligned with ``experiments/uci/compare_lowrank_factorization.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from djgp.diagnostics.projection_posterior import sanitize_for_json  # noqa: E402
from djgp.evaluation import mean_gaussian_crps_torch  # noqa: E402
from djgp.evaluation.calibration import pack_gaussian_uq_list  # noqa: E402
from djgp.projections import (  # noqa: E402
    LOWRANK_MODES,
    LowRankConfig,
    LowRankProjectionModel,
    build_V_basis,
    compute_pls_W0,
    per_anchor_W_diagnostics,
    run_projected_jumpgp,
    train_lowrank_projection,
)
from experiments.synthetic.compare_lmjgp_tricks_synth import (  # noqa: E402
    SyntheticDataset,
    generate_lowrank_jump,
    subspace_alignment_to_truth,
)


# =====================================================================
# Helpers
# =====================================================================
def _build_regions(X_test, X_train, Y_train, device, n: int):
    neighborhoods = find_neighborhoods(X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n)
    X_nb_list, y_nb_list = [], []
    for i in range(X_test.shape[0]):
        X_nb = neighborhoods[i]["X_neighbors"].to(device)
        y_nb = neighborhoods[i]["y_neighbors"].to(device)
        X_nb_list.append(X_nb)
        y_nb_list.append(y_nb)
    return X_nb_list, y_nb_list


def _stack_neighborhoods(X_nb_list, y_nb_list, device):
    T = len(X_nb_list)
    n = X_nb_list[0].shape[0]
    D = X_nb_list[0].shape[1]
    X_stack = torch.empty(T, n, D, device=device)
    y_stack = torch.empty(T, n, device=device)
    for t in range(T):
        X_stack[t] = X_nb_list[t]
        y_stack[t] = y_nb_list[t].view(-1)
    return X_stack, y_stack


def _A_from_W(W: torch.Tensor) -> torch.Tensor:
    """``A_j = W_j^T W_j ∈ R^{D × D}`` from a deterministic per-anchor W."""
    return torch.einsum("tqd,tqe->tde", W, W)


# =====================================================================
# Per-(seed) orchestration
# =====================================================================
def _per_seed_run(
    seed: int,
    *,
    N_train: int,
    N_test: int,
    D: int,
    Q_true: int,
    Q: int,
    n_pls: int,
    n_pca: int,
    n_random: int,
    num_steps: int,
    lr: float,
    smoothness_lambda: float,
    snapshot_steps: list[int],
    jump_amp: float,
    noise_std: float,
    n: int,
    device: torch.device,
):
    data: SyntheticDataset = generate_lowrank_jump(
        N_train=N_train, N_test=N_test, D=D, Q_true=Q_true,
        jump_amp=jump_amp, noise_std=noise_std, seed=seed,
    )
    X_train = torch.from_numpy(data.X_train).float().to(device)
    Y_train = torch.from_numpy(data.y_train).float().to(device)
    X_test = torch.from_numpy(data.X_test).float().to(device)
    Y_test = torch.from_numpy(data.y_test).float().to(device)
    W_true_t = torch.from_numpy(data.W_true).float().to(device)

    X_nb_list, y_nb_list = _build_regions(X_test, X_train, Y_train, device, n)
    X_stack, y_stack = _stack_neighborhoods(X_nb_list, y_nb_list, device)

    V_np, V_info = build_V_basis(
        data.X_train, data.y_train,
        n_pls=n_pls, n_pca=n_pca, n_random=n_random, seed=seed,
    )
    V_t = torch.from_numpy(V_np).float().to(device)

    W_pls_np = compute_pls_W0(data.X_train, data.y_train, Q=Q)
    W_pls_t = torch.from_numpy(W_pls_np).float().to(device)

    y_test_np = data.y_test.astype(np.float64)

    metrics: dict[str, list[float]] = {}
    diagnostics: dict[str, object] = {
        "data": data.info,
        "Q": int(Q),
        "Rv": int(V_t.shape[1]),
        "V_basis": V_info,
    }

    # PLS reference subspace alignment (for reporting only).
    A_pls = _A_from_W(W_pls_t.unsqueeze(0).expand(X_test.shape[0], -1, -1).contiguous())
    diagnostics["pls_reference_truth_alignment"] = subspace_alignment_to_truth(A_pls, W_true_t, Q_true)

    for mode in LOWRANK_MODES:
        cfg = LowRankConfig(
            mode=mode,
            Q=Q,
            Rv=V_t.shape[1],
            smoothness_lambda=float(smoothness_lambda),
        )
        model = LowRankProjectionModel(
            cfg, V_t, T=X_test.shape[0], D=D, W_pls=W_pls_t, device=device,
        )
        train_res = train_lowrank_projection(
            model,
            X_neighbors_stack=X_stack,
            y_neighbors_stack=y_stack,
            X_anchors=X_test,
            num_steps=num_steps if mode != "pls_only" else min(num_steps, 50),
            lr=lr,
            log_interval=max(50, num_steps // 5),
            snapshot_steps=snapshot_steps,
            verbose=True,
        )
        W = train_res.W

        out = run_projected_jumpgp(
            W, X_test, X_nb_list, y_nb_list,
            device=device, progress=True, desc=f"synth/seed={seed}/{mode}",
        )
        mu_np = out["mu"]
        sig_np = out["sigma"]
        rmse = float(np.sqrt(np.mean((mu_np - y_test_np) ** 2)))
        crps = float(
            mean_gaussian_crps_torch(
                torch.from_numpy(y_test_np).float(),
                torch.from_numpy(mu_np).float(),
                torch.from_numpy(sig_np).float(),
            ).item()
        )
        metrics[f"lowrank_{mode}"] = pack_gaussian_uq_list(
            rmse, crps, float(out["elapsed_sec"]), y_test_np, mu_np, sig_np,
        )
        diag = per_anchor_W_diagnostics(W, W_ref=W_pls_t)
        diag["truth_alignment"] = subspace_alignment_to_truth(_A_from_W(W), W_true_t, Q_true)
        diag["train_time_sec"] = float(train_res.train_time_sec)
        diag["num_trainable_params"] = int(train_res.num_trainable_params)
        diag["snapshots"] = train_res.snapshots
        diag["smoothness_lambda"] = float(train_res.smoothness_lambda)
        diagnostics[f"lowrank_{mode}"] = diag

    return {
        "label": "synthetic:lowrank_jump",
        "seed": int(seed),
        "Q": int(Q),
        "Q_true": int(Q_true),
        "D": int(D),
        "Rv": int(V_t.shape[1]),
        "n_test_anchors_used": int(X_test.shape[0]),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


# =====================================================================
# CLI / printing
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Low-rank tensor factorisation ablation on synthetic jump data.")
    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--N_test", type=int, default=64)
    p.add_argument("--D", type=int, default=30)
    p.add_argument("--Q_true", type=int, default=2, help="True latent dimension (rows of W_true).")
    p.add_argument("--Q", type=int, default=None, help="Override learned latent dimension Q (default: Q_true).")
    p.add_argument("--jump_amp", type=float, default=1.5)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--n", type=int, default=20, help="Neighborhood size for JumpGP.")
    p.add_argument("--n_pls", type=int, default=4)
    p.add_argument("--n_pca", type=int, default=4)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--smoothness_lambda", type=float, default=0.05)
    p.add_argument("--snapshot_steps", type=str, default="0,10,50,200")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--out_json", type=str, default="")
    return p.parse_args()


def _print_summary_row(seed: int, methods: dict[str, list[float]]) -> None:
    print(f"\n[synth] seed={seed}")
    print("  method                rmse    crps    wall    cov90   w90    cov95   w95")
    for name, row in methods.items():
        vals = [f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else "  nan " for v in row]
        formatted = "  ".join(vals)
        print(f"  {name:<22}{formatted}")


def _print_diag(seed: int, diagnostics: dict) -> None:
    print(f"\n[synth] seed={seed} diagnostics")
    skip_keys = {"data", "V_basis", "Q", "Rv"}
    for name, diag in diagnostics.items():
        if name in skip_keys:
            continue
        if not isinstance(diag, dict):
            continue
        printable = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in diag.items()
            if k not in {"snapshots", "truth_alignment"}
        }
        if "truth_alignment" in diag:
            ta = diag["truth_alignment"]
            ta_round = {k: round(v, 4) if isinstance(v, float) else v for k, v in ta.items()}
            printable["truth_alignment"] = ta_round
        print(f"  {name:<28} {json.dumps(sanitize_for_json(printable), indent=None)}")


def _print_snapshots(seed: int, diagnostics: dict) -> None:
    relevant = {k: v for k, v in diagnostics.items() if k.startswith("lowrank_") and isinstance(v, dict) and v.get("snapshots")}
    if not relevant:
        return
    print(f"\n[synth] seed={seed} snapshots")
    for name, diag in relevant.items():
        for snap in diag.get("snapshots", []):
            row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in snap.items()}
            print(f"  {name:<22} step={int(row.get('step', -1)):>4d}  {json.dumps(row)}")


def main():
    args = parse_args()
    snapshot_steps = sorted({int(s.strip()) for s in args.snapshot_steps.split(",") if s.strip()})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    Q = args.Q if args.Q is not None else args.Q_true

    total_res: dict[int, dict] = {}
    for seed in range(args.num_seeds):
        res = _per_seed_run(
            seed,
            N_train=args.N_train,
            N_test=args.N_test,
            D=args.D,
            Q_true=args.Q_true,
            Q=Q,
            n_pls=args.n_pls,
            n_pca=args.n_pca,
            n_random=args.n_random,
            num_steps=args.num_steps,
            lr=args.lr,
            smoothness_lambda=args.smoothness_lambda,
            snapshot_steps=snapshot_steps,
            jump_amp=args.jump_amp,
            noise_std=args.noise_std,
            n=args.n,
            device=device,
        )
        total_res[seed] = res
        _print_summary_row(seed, res["metrics"])
        _print_diag(seed, res["diagnostics"])
        _print_snapshots(seed, res["diagnostics"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"synth_lowrank_factorization_{stamp}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(total_res, f)
    print(f"\nSaved pickle to {out_path}")

    if args.out_json:
        def _strip(d):
            if isinstance(d, dict):
                return {k: _strip(v) for k, v in d.items() if k not in {"W"}}
            if isinstance(d, list):
                return [_strip(x) for x in d]
            return d

        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(_strip(total_res)), f, indent=2)
        print(f"Saved JSON to {args.out_json}")

    return total_res


if __name__ == "__main__":
    main()

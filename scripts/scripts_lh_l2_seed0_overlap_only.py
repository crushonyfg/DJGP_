"""Part 1: compute z vs X KNN overlap on LH and L2 paper synthetic seed=0 splits.

Uses pre-exported npz from `experiments/synthetic/deep_mahalanobis_gp_paper_*`
which were produced by
`experiments/synthetic/export_paper_synthetic_for_dmgp.py` calling
`compare_paper_synthetic_baselines._generate_paper_data`. So:
- z is the paper latent (LH d=5, L2 d=2)
- X is the paper RFF expansion (LH H=30, L2 H=20), standardized by train mean/std
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("/sessions/elegant-eloquent-carson/mnt/DJGP")


def cross_overlap(z_test, X_test, z_train, X_train, k):
    Nt = z_test.shape[0]

    def knn(Atest, Atrain):
        d2 = (
            np.sum(Atest ** 2, axis=1, keepdims=True)
            + np.sum(Atrain ** 2, axis=1)[None, :]
            - 2.0 * (Atest @ Atrain.T)
        )
        return np.argpartition(d2, kth=k, axis=1)[:, :k]

    zi = knn(z_test, z_train)
    xi = knn(X_test, X_train)
    ov = np.array([len(set(zi[i].tolist()) & set(xi[i].tolist())) for i in range(Nt)])
    return {
        "k": int(k),
        "anchors": int(Nt),
        "train_pool": int(z_train.shape[0]),
        "mean": float(np.mean(ov)),
        "std": float(np.std(ov)),
        "median": float(np.median(ov)),
        "min": int(np.min(ov)),
        "max": int(np.max(ov)),
        "frac_mean": float(np.mean(ov) / k),
        "hist_buckets_le5_le10_le15_le20_le25_le30_le35": [
            int(np.sum(ov <= 5)),
            int(np.sum((ov > 5) & (ov <= 10))),
            int(np.sum((ov > 10) & (ov <= 15))),
            int(np.sum((ov > 15) & (ov <= 20))),
            int(np.sum((ov > 20) & (ov <= 25))),
            int(np.sum((ov > 25) & (ov <= 30))),
            int(np.sum((ov > 30) & (ov <= 35))),
        ],
    }


def within_train_overlap(z, X, k):
    """Within-train: each train point as anchor, candidate pool = train \ {self}."""
    N = z.shape[0]

    def knn_self(M):
        d2 = np.sum(M ** 2, axis=1, keepdims=True) + np.sum(M ** 2, axis=1)[None, :] - 2.0 * (M @ M.T)
        np.fill_diagonal(d2, np.inf)
        return np.argpartition(d2, kth=k, axis=1)[:, :k]

    zi = knn_self(z)
    xi = knn_self(X)
    ov = np.array([len(set(zi[i].tolist()) & set(xi[i].tolist())) for i in range(N)])
    return {
        "k": int(k),
        "anchors": int(N),
        "mean": float(np.mean(ov)),
        "std": float(np.std(ov)),
        "median": float(np.median(ov)),
        "min": int(np.min(ov)),
        "max": int(np.max(ov)),
        "frac_mean": float(np.mean(ov) / k),
    }


def main():
    base = PROJECT_ROOT / "experiments" / "synthetic"
    files = {
        "lh_q5_train200_seed0": base / "deep_mahalanobis_gp_paper_train200_500_5seeds" / "lh_q5_train200_seed0.npz",
        "lh_q5_train500_seed0": base / "deep_mahalanobis_gp_paper_train200_500_5seeds" / "lh_q5_train500_seed0.npz",
        "lh_q5_train1000_seed0": base / "deep_mahalanobis_gp_paper_train1000_seed0" / "lh_q5_train1000_seed0.npz",
        "l2_q2_train200_seed0": base / "deep_mahalanobis_gp_paper_train200_500_5seeds" / "l2_q2_train200_seed0.npz",
        "l2_q2_train500_seed0": base / "deep_mahalanobis_gp_paper_train200_500_5seeds" / "l2_q2_train500_seed0.npz",
        "l2_q2_train1000_seed0": base / "deep_mahalanobis_gp_paper_train1000_seed0" / "l2_q2_train1000_seed0.npz",
    }
    rep = {}
    for name, p in files.items():
        if not p.exists():
            print(f"missing: {p}")
            continue
        d = np.load(p)
        z_train, z_test, X_train, X_test = d["z_train"], d["z_test"], d["X_train"], d["X_test"]
        entry = {
            "file": str(p),
            "shapes": {
                "z_train": list(z_train.shape),
                "z_test": list(z_test.shape),
                "X_train": list(X_train.shape),
                "X_test": list(X_test.shape),
            },
            "test_to_train_k35": cross_overlap(z_test, X_test, z_train, X_train, k=35),
            "within_train_k35": within_train_overlap(z_train, X_train, k=35),
        }
        if name.startswith("l2_"):
            entry["test_to_train_k25"] = cross_overlap(z_test, X_test, z_train, X_train, k=25)
            entry["within_train_k25"] = within_train_overlap(z_train, X_train, k=25)
        rep[name] = entry

    out = Path("/sessions/elegant-eloquent-carson/mnt/outputs/overlap_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

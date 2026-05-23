"""
Compare **raw-X** vs **train-fitted StandardScaler(X)** KNN neighborhoods.

Motivation: local / projected methods (LMJGP, CEM-EM, JumpGP) depend on
``KNN(x_*, X_train)`` geometry. If raw feature scales encode useful implicit
weights (e.g. subject / time-like columns with large variance), standardizing X
can change neighbor sets and worsen neighborhood ``y`` homogeneity — without
any bug in the model.

For **Parkinsons Telemonitoring**, ``load_uci_split(..., meta=UciSplitMeta())``
records ``subject#`` (UCI feature column 0). We report the fraction of each
query's ``n`` neighbors that share the same subject id, for raw vs std-X KNN.

Run from repo root::

    python experiments/uci/knn_scale_diagnostics.py \\
        --dataset "Parkinsons Telemonitoring" --seed 0 --n_neighbors 35 --max_anchors 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.uci.uci_data import UciSplitMeta, load_uci_split


def _subsample_query(
    X_te: np.ndarray,
    y_te: np.ndarray,
    max_anchors: int | None,
    seed: int,
):
    n = X_te.shape[0]
    if max_anchors is None or max_anchors >= n:
        sub = np.arange(n, dtype=np.int64)
        return X_te, y_te, sub
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(n, size=max_anchors, replace=False))
    return X_te[sub], y_te[sub], sub


def knn_indices(X_train: np.ndarray, X_query: np.ndarray, n_neighbors: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_train)
    _, idx = nn.kneighbors(X_query, n_neighbors=n_neighbors)
    return np.asarray(idx, dtype=np.int64)


def jaccard_stats(idx_a: np.ndarray, idx_b: np.ndarray) -> dict:
    j_list = []
    for i in range(idx_a.shape[0]):
        A = set(idx_a[i].tolist())
        B = set(idx_b[i].tolist())
        inter = len(A & B)
        union = len(A | B)
        j_list.append(inter / union if union else 0.0)
    j = np.asarray(j_list, dtype=np.float64)
    return {"mean_jaccard": float(j.mean()), "median_jaccard": float(np.median(j))}


def y_neighborhood_stats(y_train: np.ndarray, y_query: np.ndarray, idx: np.ndarray) -> dict:
    v = []
    mad = []
    y_tr = np.asarray(y_train, dtype=np.float64).ravel()
    y_q = np.asarray(y_query, dtype=np.float64).ravel()
    for i in range(idx.shape[0]):
        yn = y_tr[idx[i]]
        v.append(float(np.var(yn)))
        mad.append(float(np.mean(np.abs(yn - y_q[i]))))
    v = np.asarray(v, dtype=np.float64)
    mad = np.asarray(mad, dtype=np.float64)
    return {
        "mean_var_y_nb": float(v.mean()),
        "median_var_y_nb": float(np.median(v)),
        "mean_mean_abs_y_diff": float(mad.mean()),
        "median_mean_abs_y_diff": float(np.median(mad)),
    }


def same_subject_neighbor_fraction(
    subject_train: np.ndarray,
    subject_query: np.ndarray,
    idx: np.ndarray,
) -> dict:
    st = np.asarray(subject_train).ravel()
    sq = np.asarray(subject_query).ravel()
    fracs = []
    for i in range(idx.shape[0]):
        sid = sq[i]
        nb = st[idx[i]]
        fracs.append(float(np.mean(nb == sid)))
    f = np.asarray(fracs, dtype=np.float64)
    return {
        "mean_frac_same_subject": float(f.mean()),
        "median_frac_same_subject": float(np.median(f)),
    }


def parse_args():
    p = argparse.ArgumentParser(description="KNN raw vs std-X diagnostics for UCI splits.")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--max_anchors", type=int, default=500)
    p.add_argument("--query_subsample_seed", type=int, default=12345)
    p.add_argument("--out_json", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    meta = UciSplitMeta()
    X_tr, X_te, y_tr, y_te, K = load_uci_split(args.dataset, args.seed, meta=meta)

    X_q, y_q, sub = _subsample_query(X_te, y_te, args.max_anchors, args.query_subsample_seed)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_q_s = scaler.transform(X_q)

    idx_raw = knn_indices(X_tr, X_q, args.n_neighbors)
    idx_std = knn_indices(X_tr_s, X_q_s, args.n_neighbors)

    out: dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "n_neighbors": args.n_neighbors,
        "n_train": int(X_tr.shape[0]),
        "n_query": int(X_q.shape[0]),
        "K_sir": int(K),
        "neighbor_set_jaccard_raw_vs_stdX": jaccard_stats(idx_raw, idx_std),
        "raw_X_knn_y_stats": y_neighborhood_stats(y_tr, y_q, idx_raw),
        "stdX_knn_y_stats": y_neighborhood_stats(y_tr, y_q, idx_std),
    }

    if meta.parkinsons_subject_train is not None and meta.parkinsons_subject_test is not None:
        subj_q = meta.parkinsons_subject_test[sub]
        out["parkinsons_subject_in_neighbors_raw"] = same_subject_neighbor_fraction(
            meta.parkinsons_subject_train, subj_q, idx_raw,
        )
        out["parkinsons_subject_in_neighbors_stdX"] = same_subject_neighbor_fraction(
            meta.parkinsons_subject_train, subj_q, idx_std,
        )

    print(json.dumps(out, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_json).open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()

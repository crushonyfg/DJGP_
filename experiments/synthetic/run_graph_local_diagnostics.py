"""Generate graph-local synthetic data and Stage-0 diagnostics.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Default one-seed Dataset B-style jump diagnostic.
    python experiments/synthetic/run_graph_local_diagnostics.py

    # Smooth Dataset A diagnostic.
    python experiments/synthetic/run_graph_local_diagnostics.py ^
        --preset dataset_a ^
        --out_dir experiments/synthetic/graph_local_stage0_dataset_a_seed0_20260518

    # Misleading feature-kNN diagnostic with explicit graph/feature decoupling.
    python experiments/synthetic/run_graph_local_diagnostics.py ^
        --preset dataset_e ^
        --seed 0 ^
        --out_dir experiments/synthetic/graph_local_stage0_dataset_e_seed0_20260518

This is the first milestone from ``docs/graph_local_djgp_experiment_plan.md``:
it generates data and diagnostics only. It does not train DJGP/JGP or any
baseline model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


PRESETS: dict[str, dict[str, Any]] = {
    "dataset_a": {
        "description": "Graph-local smooth latent synthetic, D=30, q=3.",
        "D": 30,
        "q": 3,
        "jump": False,
        "misleading_features": False,
        "latent_type": "linear",
    },
    "dataset_b": {
        "description": "Graph-local hard-jump latent synthetic, D=30, q=3.",
        "D": 30,
        "q": 3,
        "jump": True,
        "misleading_features": False,
        "latent_type": "linear",
    },
    "dataset_d": {
        "description": "Graph-local hard-jump latent synthetic, D=100, q=5.",
        "D": 100,
        "q": 5,
        "jump": True,
        "misleading_features": False,
        "latent_type": "linear",
    },
    "dataset_e": {
        "description": "Misleading feature-kNN graph-local smooth synthetic, D=100, q=5.",
        "D": 100,
        "q": 5,
        "jump": False,
        "misleading_features": True,
        "latent_type": "linear",
    },
}


def _as_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _as_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_builtin(v) for v in value]
    return value


def _balanced_block_ids(n_nodes: int, n_blocks: int) -> np.ndarray:
    block_id = np.arange(n_nodes, dtype=np.int64) * n_blocks // n_nodes
    return np.minimum(block_id, n_blocks - 1)


def _generate_sbm(
    rng: np.random.Generator,
    *,
    n_nodes: int,
    n_blocks: int,
    p_in: float,
    p_out: float,
) -> tuple[np.ndarray, np.ndarray]:
    block_id = _balanced_block_ids(n_nodes, n_blocks)
    same_block = block_id[:, None] == block_id[None, :]
    prob = np.where(same_block, p_in, p_out)
    upper = rng.random((n_nodes, n_nodes)) < prob
    upper = np.triu(upper, k=1)
    adj = upper | upper.T

    # Make each block internally connected by a light ring, and connect blocks
    # with one bridge edge so graph distances stay finite for diagnostics.
    for b in range(n_blocks):
        nodes = np.flatnonzero(block_id == b)
        for i, node in enumerate(nodes):
            nxt = nodes[(i + 1) % len(nodes)]
            if node != nxt:
                adj[node, nxt] = True
                adj[nxt, node] = True
    for b in range(n_blocks - 1):
        left = np.flatnonzero(block_id == b)[0]
        right = np.flatnonzero(block_id == b + 1)[0]
        adj[left, right] = True
        adj[right, left] = True
    return adj.astype(bool), block_id


def _shortest_path_hops(adj: np.ndarray) -> np.ndarray:
    n_nodes = adj.shape[0]
    neighbors = [np.flatnonzero(adj[i]).astype(np.int64) for i in range(n_nodes)]
    dist = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    for start in range(n_nodes):
        dist[start, start] = 0.0
        q: deque[int] = deque([start])
        while q:
            node = q.popleft()
            next_dist = dist[start, node] + 1.0
            for nb in neighbors[node]:
                if math.isinf(dist[start, nb]):
                    dist[start, nb] = next_dist
                    q.append(int(nb))
    return dist


def _neighborhood_weights(graph_dist: np.ndarray, radius: int, lam: float) -> np.ndarray:
    mask = graph_dist <= float(radius)
    weights = np.where(mask, np.exp(-lam * graph_dist), 0.0)
    denom = weights.sum(axis=1, keepdims=True)
    if np.any(denom <= 0):
        raise RuntimeError("Every node should have at least the self-loop neighborhood.")
    return weights / denom


def _standardize_columns(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / np.maximum(std, 1e-8)


def generate_graph_local_surrogate(
    *,
    n_nodes: int = 300,
    n_time: int = 20,
    D: int = 30,
    q: int = 3,
    graph_type: str = "sbm",
    n_blocks: int = 8,
    p_in: float = 0.15,
    p_out: float = 0.01,
    graph_radius: int = 1,
    graph_lambda: float = 1.0,
    jump: bool = True,
    soft_jump: bool = False,
    misleading_features: bool = False,
    latent_type: str = "linear",
    noise_x: float = 0.5,
    noise_y: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Generate one graph-local surrogate dataset and metadata."""
    if graph_type != "sbm":
        raise NotImplementedError("Stage-0 runner currently implements SBM only.")
    if latent_type != "linear":
        raise NotImplementedError("Stage-0 runner currently implements linear latent features only.")

    rng = np.random.default_rng(seed)
    adj, block_id = _generate_sbm(rng, n_nodes=n_nodes, n_blocks=n_blocks, p_in=p_in, p_out=p_out)
    graph_dist = _shortest_path_hops(adj)
    weights = _neighborhood_weights(graph_dist, graph_radius, graph_lambda)

    block_scale = 0.10 if misleading_features else 0.35
    block_mu = rng.normal(0.0, block_scale, size=(n_blocks, q))
    S = np.empty((n_time, n_nodes, q), dtype=np.float64)
    S[0] = block_mu[block_id] + rng.normal(0.0, 1.0, size=(n_nodes, q))
    for t in range(1, n_time):
        innovations = rng.normal(0.0, 1.0, size=(n_nodes, q))
        S[t] = 0.7 * S[t - 1] + 0.3 * block_mu[block_id] + innovations

    A = rng.normal(0.0, 1.0 / math.sqrt(q), size=(D, q))
    X_3d = np.einsum("tnq,dq->tnd", S, A) + rng.normal(0.0, noise_x, size=(n_time, n_nodes, D))
    W_star = np.linalg.pinv(A)
    Z_3d = np.einsum("tnd,qd->tnq", X_3d, W_star)
    M_3d = np.einsum("vu,tuq->tvq", weights, Z_3d)

    a0 = rng.normal(0.0, 0.45, size=q)
    b0 = rng.normal(0.0, 0.45, size=q)
    c0 = rng.normal(0.0, 0.20, size=q)
    d0 = rng.normal(0.0, 0.20, size=q)
    a = a0 + rng.normal(0.0, 0.20, size=(n_blocks, q))
    b = b0 + rng.normal(0.0, 0.20, size=(n_blocks, q))
    c = c0 + rng.normal(0.0, 0.12, size=(n_blocks, q))
    d = d0 + rng.normal(0.0, 0.12, size=(n_blocks, q))

    z_score = np.einsum("tnq,nq->tn", Z_3d, a[block_id])
    m_score = np.einsum("tnq,nq->tn", M_3d, b[block_id])
    linear_z = np.einsum("tnq,nq->tn", Z_3d, c[block_id])
    linear_m = np.einsum("tnq,nq->tn", M_3d, d[block_id])
    signal = np.sin(z_score) + 0.25 * np.square(m_score) + linear_z + linear_m

    jump_indicator = np.zeros((n_time, n_nodes), dtype=bool)
    jump_term = np.zeros((n_time, n_nodes), dtype=np.float64)
    thresholds = np.full(n_blocks, np.nan, dtype=np.float64)
    jump_delta = np.zeros(n_blocks, dtype=np.float64)
    if jump:
        for block in range(n_blocks):
            nodes = np.flatnonzero(block_id == block)
            scores = m_score[:, nodes].reshape(-1)
            thresholds[block] = float(np.quantile(scores, 0.68))
            jump_delta[block] = float(rng.uniform(1.0, 2.5))
            block_jump = m_score[:, nodes] > thresholds[block]
            jump_indicator[:, nodes] = block_jump
            if soft_jump:
                jump_term[:, nodes] = jump_delta[block] / (1.0 + np.exp(-20.0 * (m_score[:, nodes] - thresholds[block])))
            else:
                jump_term[:, nodes] = jump_delta[block] * block_jump

    clean_y = signal + jump_term
    y_3d = clean_y + rng.normal(0.0, noise_y, size=(n_time, n_nodes))

    node_grid = np.tile(np.arange(n_nodes, dtype=np.int64), n_time)
    time_grid = np.repeat(np.arange(n_time, dtype=np.int64), n_nodes)
    X = X_3d.reshape(n_time * n_nodes, D)
    y = y_3d.reshape(-1)
    clean_y_flat = clean_y.reshape(-1)
    Z_true = Z_3d.reshape(n_time * n_nodes, q)
    M_true = M_3d.reshape(n_time * n_nodes, q)
    regime_id = block_id[node_grid]

    edge_index = np.column_stack(np.nonzero(adj)).astype(np.int64).T
    return {
        "X": X,
        "y": y,
        "clean_y": clean_y_flat,
        "node_id": node_grid,
        "time_id": time_grid,
        "edge_index": edge_index,
        "graph_dist": graph_dist,
        "block_id": block_id,
        "regime_id": regime_id,
        "jump_indicator": jump_indicator.reshape(-1),
        "Z_true": Z_true,
        "M_true": M_true,
        "W_star": W_star,
        "metadata": {
            "generator_family": "graph_local_sbm_latent_surrogate",
            "base_low_dimensional_generator": "temporal AR latent state with SBM block means",
            "expansion_method": "linear_latent_to_observed",
            "latent_dimension": q,
            "observed_dimension": D,
            "n_nodes": n_nodes,
            "n_time": n_time,
            "train_test_sizes": "diagnostic masks only; see split diagnostics",
            "x_standardization": "not applied in raw saved data",
            "y_standardization": "not applied in raw saved data",
            "graph_type": graph_type,
            "n_blocks": n_blocks,
            "graph_radius": graph_radius,
            "jump": jump,
            "soft_jump": soft_jump,
            "misleading_features": misleading_features,
            "noise_x": noise_x,
            "noise_y": noise_y,
            "seed": seed,
            "thresholds": thresholds,
            "jump_delta": jump_delta,
        },
    }


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    aa = a[finite]
    bb = b[finite]
    if aa.std() <= 1e-12 or bb.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _pair_distance_diagnostic(
    rng: np.random.Generator,
    X: np.ndarray,
    node_id: np.ndarray,
    graph_dist: np.ndarray,
    *,
    max_pairs: int,
) -> dict[str, float]:
    n = X.shape[0]
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    keep = i != j
    i = i[keep]
    j = j[keep]
    feature_dist = np.linalg.norm(X[i] - X[j], axis=1)
    graph_hops = graph_dist[node_id[i], node_id[j]]
    return {
        "feature_graph_distance_corr": _corrcoef_safe(feature_dist, graph_hops),
        "sampled_pair_count": int(len(i)),
        "sampled_pair_finite_graph_fraction": float(np.isfinite(graph_hops).mean()),
        "sampled_feature_distance_mean": float(feature_dist.mean()),
        "sampled_graph_hop_mean": float(np.mean(graph_hops[np.isfinite(graph_hops)])),
    }


def _knn_overlap_diagnostic(
    rng: np.random.Generator,
    X: np.ndarray,
    n_nodes: int,
    n_time: int,
    graph_dist: np.ndarray,
    *,
    k: int,
    max_anchors: int,
) -> dict[str, float]:
    X_3d = X.reshape(n_time, n_nodes, X.shape[1])
    n_anchors = min(max_anchors, n_nodes * n_time)
    anchor_flat = rng.choice(n_nodes * n_time, size=n_anchors, replace=False)
    overlaps: list[float] = []
    for flat in anchor_flat:
        t = int(flat // n_nodes)
        v = int(flat % n_nodes)
        feature_dist = np.linalg.norm(X_3d[t] - X_3d[t, v][None, :], axis=1)
        feature_order = np.argsort(feature_dist)
        feature_knn = [int(idx) for idx in feature_order if int(idx) != v][:k]
        graph_order = np.lexsort((np.arange(n_nodes), graph_dist[v]))
        graph_knn = [int(idx) for idx in graph_order if int(idx) != v][:k]
        overlaps.append(len(set(feature_knn).intersection(graph_knn)) / float(k))
    return {
        "feature_graph_knn_overlap_at_k": float(np.mean(overlaps)),
        "feature_graph_knn_overlap_std_at_k": float(np.std(overlaps)),
        "knn_k": int(k),
        "knn_anchor_count": int(n_anchors),
        "random_overlap_reference": float(k / max(n_nodes - 1, 1)),
    }


def _split_diagnostics(node_id: np.ndarray, time_id: np.ndarray, regime_id: np.ndarray, n_time: int) -> dict[str, Any]:
    n = len(node_id)
    rng = np.random.default_rng(12345)
    random_train = rng.random(n) < 0.7
    future_train = time_id < int(math.floor(0.7 * n_time))
    heldout_blocks = np.unique(regime_id)[-max(1, len(np.unique(regime_id)) // 4) :]
    heldout_train = ~np.isin(regime_id, heldout_blocks)
    return {
        "random_split_train": int(random_train.sum()),
        "random_split_test": int((~random_train).sum()),
        "future_split_train": int(future_train.sum()),
        "future_split_test": int((~future_train).sum()),
        "heldout_region_split_train": int(heldout_train.sum()),
        "heldout_region_split_test": int((~heldout_train).sum()),
        "heldout_region_ids": [int(x) for x in heldout_blocks],
    }


def compute_diagnostics(data: dict[str, Any], *, seed: int, max_pairs: int, knn_k: int, knn_anchors: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1009)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    clean_y = np.asarray(data["clean_y"], dtype=np.float64)
    node_id = np.asarray(data["node_id"], dtype=np.int64)
    time_id = np.asarray(data["time_id"], dtype=np.int64)
    graph_dist = np.asarray(data["graph_dist"], dtype=np.float64)
    block_id = np.asarray(data["block_id"], dtype=np.int64)
    regime_id = np.asarray(data["regime_id"], dtype=np.int64)
    jump_indicator = np.asarray(data["jump_indicator"], dtype=bool)
    degrees = np.isfinite(graph_dist).sum(axis=1)  # overwritten below with adjacency degree proxy from hop=1
    degrees = (graph_dist == 1.0).sum(axis=1)

    out: dict[str, Any] = {
        "n_samples": int(X.shape[0]),
        "n_nodes": int(len(block_id)),
        "n_time": int(len(np.unique(time_id))),
        "observed_dimension": int(X.shape[1]),
        "latent_dimension": int(np.asarray(data["Z_true"]).shape[1]),
        "n_blocks": int(len(np.unique(block_id))),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "y_min": float(y.min()),
        "y_p05": float(np.quantile(y, 0.05)),
        "y_p50": float(np.quantile(y, 0.50)),
        "y_p95": float(np.quantile(y, 0.95)),
        "y_max": float(y.max()),
        "degree_min": int(degrees.min()),
        "degree_mean": float(degrees.mean()),
        "degree_p50": float(np.quantile(degrees, 0.50)),
        "degree_max": int(degrees.max()),
        "jump_proportion": float(jump_indicator.mean()),
        "signal_variance": float(clean_y.var()),
        "noise_variance_empirical": float((y - clean_y).var()),
        "signal_to_noise_variance_ratio": float(clean_y.var() / max((y - clean_y).var(), 1e-12)),
    }
    for block in np.unique(regime_id):
        mask = regime_id == block
        out[f"block_{int(block)}_jump_proportion"] = float(jump_indicator[mask].mean())
        out[f"block_{int(block)}_sample_count"] = int(mask.sum())
    out.update(_pair_distance_diagnostic(rng, _standardize_columns(X), node_id, graph_dist, max_pairs=max_pairs))
    out.update(
        _knn_overlap_diagnostic(
            rng,
            _standardize_columns(X),
            int(len(block_id)),
            int(len(np.unique(time_id))),
            graph_dist,
            k=knn_k,
            max_anchors=knn_anchors,
        )
    )
    out.update(_split_diagnostics(node_id, time_id, regime_id, int(len(np.unique(time_id)))))
    return out


def _write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Graph-Local Stage-0 Diagnostics

This folder contains a generator-only pilot for
`docs/graph_local_djgp_experiment_plan.md`.

Synthetic provenance:

- Base low-dimensional generator: temporal AR latent state with SBM block means.
- Expansion: linear latent-to-observed features.
- Graph family: stochastic block model with explicit connected block/ring edges.
- Standardization: raw `dataset.npz` is not standardized; diagnostics standardize
  X only for feature-distance comparisons.
- Models: none trained.

Files:

- `dataset.npz`: generated arrays and graph metadata.
- `diagnostics.json`: full Stage-0 diagnostics.
- `diagnostics.csv`: one-row flat diagnostics table.
- `summary.json`: CLI settings and output manifest.

Runner summary:

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph-local synthetic Stage-0 diagnostic generator.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="dataset_b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_nodes", type=int, default=300)
    p.add_argument("--n_time", type=int, default=20)
    p.add_argument("--n_blocks", type=int, default=8)
    p.add_argument("--p_in", type=float, default=0.15)
    p.add_argument("--p_out", type=float, default=0.01)
    p.add_argument("--graph_radius", type=int, default=1)
    p.add_argument("--graph_lambda", type=float, default=1.0)
    p.add_argument("--noise_x", type=float, default=0.5)
    p.add_argument("--noise_y", type=float, default=0.1)
    p.add_argument("--soft_jump", action="store_true")
    p.add_argument("--max_pairs", type=int, default=10000)
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--knn_anchors", type=int, default=1000)
    p.add_argument(
        "--out_dir",
        type=str,
        default="experiments/synthetic/graph_local_stage0_dataset_b_seed0_20260518",
    )
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    preset = dict(PRESETS[args.preset])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_graph_local_surrogate(
        n_nodes=args.n_nodes,
        n_time=args.n_time,
        D=int(preset["D"]),
        q=int(preset["q"]),
        graph_type="sbm",
        n_blocks=args.n_blocks,
        p_in=args.p_in,
        p_out=args.p_out,
        graph_radius=args.graph_radius,
        graph_lambda=args.graph_lambda,
        jump=bool(preset["jump"]),
        soft_jump=bool(args.soft_jump),
        misleading_features=bool(preset["misleading_features"]),
        latent_type=str(preset["latent_type"]),
        noise_x=args.noise_x,
        noise_y=args.noise_y,
        seed=args.seed,
    )
    diagnostics = compute_diagnostics(
        data,
        seed=args.seed,
        max_pairs=args.max_pairs,
        knn_k=args.knn_k,
        knn_anchors=args.knn_anchors,
    )
    diagnostics = {"preset": args.preset, "description": preset["description"], **diagnostics}

    np.savez_compressed(
        out_dir / "dataset.npz",
        X=data["X"],
        y=data["y"],
        clean_y=data["clean_y"],
        node_id=data["node_id"],
        time_id=data["time_id"],
        edge_index=data["edge_index"],
        graph_dist=data["graph_dist"],
        block_id=data["block_id"],
        regime_id=data["regime_id"],
        jump_indicator=data["jump_indicator"],
        Z_true=data["Z_true"],
        M_true=data["M_true"],
        W_star=data["W_star"],
    )
    (out_dir / "diagnostics.json").write_text(json.dumps(_as_builtin(diagnostics), indent=2), encoding="utf-8")
    _write_flat_csv(out_dir / "diagnostics.csv", [_as_builtin(diagnostics)])

    summary = {
        "args": vars(args),
        "preset": preset,
        "metadata": _as_builtin(data["metadata"]),
        "artifacts": ["dataset.npz", "diagnostics.json", "diagnostics.csv", "README.md", "summary.json"],
    }
    (out_dir / "summary.json").write_text(json.dumps(_as_builtin(summary), indent=2), encoding="utf-8")
    _write_readme(out_dir, _as_builtin(summary))
    print(json.dumps(_as_builtin(diagnostics), indent=2), flush=True)
    return {"summary": summary, "diagnostics": diagnostics}


if __name__ == "__main__":
    main()

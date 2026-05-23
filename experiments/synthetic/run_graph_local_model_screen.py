"""Run a seed-0 model screen on graph-local synthetic data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Default pilot: XGBoost, DKL, LMJGP sampled-W CEM, and fixed-neighborhood self-CEM.
    python experiments/synthetic/run_graph_local_model_screen.py

    # Add local rows that also receive graph-augmented features.
    python experiments/synthetic/run_graph_local_model_screen.py ^
        --methods lmjgp_feature_graphfeat,lmjgp_graph_graphfeat,selfcem_feature_graphfeat,selfcem_graph_graphfeat ^
        --out_dir experiments/synthetic/graph_local_model_screen_dataset_b_seed0_graphfeat_local_20260518

    # Faster smoke with fewer anchors and no DKL.
    python experiments/synthetic/run_graph_local_model_screen.py ^
        --methods xgb_raw,xgb_graph,lmjgp_feature,lmjgp_graph,selfcem_feature,selfcem_graph ^
        --max_train 300 --max_test 20 --lmjgp_steps 40 --self_cem_hyper_steps 2 --self_cem_gate_steps 2 ^
        --out_dir experiments/synthetic/graph_local_model_screen_smoke_20260518

This runner tests the first graph-local benchmark signal from
``docs/graph_local_djgp_experiment_plan.md``.  LMJGP uses
``djgp.variational.predict_vi``, i.e. sampled ``W_*`` followed by per-sample
JumpGP-CEM refits.  The self-CEM rows keep neighborhoods fixed after the
initial feature- or graph-neighborhood retrieval; they do not reselect
neighbors with ``W``.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.baselines import run_dkl_svgp_regression, run_xgboost_bootstrap_regression  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.projections import (  # noqa: E402
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
    SelfCEMConfig,
    compute_pls_W0,
    run_uncertain_w_self_cem,
    uncertain_w_cem_predict_from_labels,
)
from djgp.variational import predict_vi, train_vi  # noqa: E402
from experiments.synthetic.run_graph_local_diagnostics import (  # noqa: E402
    PRESETS,
    generate_graph_local_surrogate,
)
from shared.utils1 import jumpgp_ld_wrapper  # noqa: E402


ALL_METHODS = (
    "xgb_raw",
    "xgb_graph",
    "dkl_raw",
    "dkl_graph",
    "lmjgp_feature",
    "lmjgp_graph",
    "lmjgp_feature_graphfeat",
    "lmjgp_graph_graphfeat",
    "selfcem_feature",
    "selfcem_graph",
    "selfcem_feature_graphfeat",
    "selfcem_graph_graphfeat",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _as_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_builtin(v) for v in value]
    return value


def _graph_aggregate_features(data: dict[str, Any]) -> np.ndarray:
    X = np.asarray(data["X"], dtype=np.float64)
    n_time = int(len(np.unique(data["time_id"])))
    n_nodes = int(len(data["block_id"]))
    D = X.shape[1]
    graph_dist = np.asarray(data["graph_dist"], dtype=np.float64)
    weights = np.where(graph_dist <= 1.0, np.exp(-graph_dist), 0.0)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    X_3d = X.reshape(n_time, n_nodes, D)
    agg = np.einsum("vu,tud->tvd", weights, X_3d).reshape(n_time * n_nodes, D)
    block_id = np.asarray(data["block_id"], dtype=np.int64)
    n_blocks = int(block_id.max()) + 1
    onehot_node = np.eye(n_blocks, dtype=np.float64)[block_id]
    onehot = onehot_node[np.asarray(data["node_id"], dtype=np.int64)]
    return np.concatenate([X, onehot, agg], axis=1)


def _make_split(
    data: dict[str, Any],
    *,
    seed: int,
    train_frac: float,
    max_train: int,
    max_test: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 17)
    n = len(data["y"])
    train_mask = rng.random(n) < train_frac
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(~train_mask)
    if max_train > 0 and len(train_idx) > max_train:
        train_idx = np.sort(rng.choice(train_idx, size=max_train, replace=False))
    if max_test > 0 and len(test_idx) > max_test:
        test_idx = np.sort(rng.choice(test_idx, size=max_test, replace=False))
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def _feature_neighbor_indices(X_train: np.ndarray, X_test: np.ndarray, n_neighbors: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X_train)), metric="euclidean")
    nn.fit(X_train)
    return nn.kneighbors(X_test, return_distance=False).astype(np.int64)


def _graph_neighbor_indices(
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_node: np.ndarray,
    train_time: np.ndarray,
    test_node: np.ndarray,
    test_time: np.ndarray,
    graph_dist: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    out = np.empty((len(X_test), min(n_neighbors, len(X_train))), dtype=np.int64)
    for i in range(len(X_test)):
        hops = graph_dist[int(test_node[i]), train_node]
        time_gap = np.abs(train_time - int(test_time[i]))
        feat_dist = np.linalg.norm(X_train - X_test[i][None, :], axis=1)
        order = np.lexsort((np.arange(len(X_train)), feat_dist, time_gap, hops))
        out[i] = order[: out.shape[1]]
    return out


def _pack_subset_metrics(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    elapsed: float,
    mask: np.ndarray,
) -> dict[str, float]:
    if int(mask.sum()) == 0:
        return {key: float("nan") for key in RESULT_ROW_KEYS}
    y = np.asarray(y_true[mask], dtype=np.float64)
    m = np.asarray(mu[mask], dtype=np.float64)
    s = np.maximum(np.asarray(sigma[mask], dtype=np.float64), 1e-8)
    rmse = float(np.sqrt(np.mean((m - y) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(m, dtype=torch.float32),
            torch.as_tensor(s, dtype=torch.float32),
        ).item()
    )
    vals = pack_gaussian_uq_list(rmse, crps, elapsed, y, m, s)
    return {key: float(vals[i]) for i, key in enumerate(RESULT_ROW_KEYS)}


def _rows_for_prediction(
    *,
    method: str,
    y_test: np.ndarray,
    jump_test: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    elapsed: float,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    masks = {
        "all": np.ones_like(y_test, dtype=bool),
        "jump": jump_test.astype(bool),
        "nonjump": ~jump_test.astype(bool),
    }
    rows: list[dict[str, Any]] = []
    for subset, mask in masks.items():
        row = dict(meta)
        row.update({"method": method, "subset": subset, "n_eval": int(mask.sum()), "status": "ok"})
        row.update(_pack_subset_metrics(y_test, mu, sigma, elapsed, mask))
        rows.append(row)
    return rows


def _build_regions(
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    neighbor_idx: np.ndarray,
    *,
    m1: int,
    Q: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    regions = []
    idx_t = torch.as_tensor(neighbor_idx, dtype=torch.long, device=device)
    for j in range(idx_t.shape[0]):
        regions.append(
            {
                "X": X_train_t[idx_t[j]],
                "y": y_train_t[idx_t[j]],
                "C": torch.randn(m1, Q, device=device),
            }
        )
    return regions


def _run_lmjgp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    neighbor_idx: np.ndarray,
    *,
    Q: int,
    m1: int,
    m2: int,
    num_steps: int,
    MC_num: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    _set_seed(seed)
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device).view(-1)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    regions = _build_regions(X_train_t, y_train_t, neighbor_idx, m1=m1, Q=Q, device=device)
    D = X_train_t.shape[1]
    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
    }
    u_params = [
        {
            "U_logit": torch.zeros(1, device=device, requires_grad=True),
            "mu_u": torch.randn(m1, device=device, requires_grad=True),
            "Sigma_u": torch.eye(m1, device=device, requires_grad=True),
            "sigma_noise": torch.tensor(0.5, device=device, requires_grad=True),
            "sigma_k": torch.tensor(0.5, device=device, requires_grad=True),
            "omega": torch.randn(Q + 1, device=device, requires_grad=True),
        }
        for _ in regions
    ]
    X_train_std = X_train_t.std(dim=0).clamp_min(1e-6)
    hyperparams = {
        "Z": X_train_t.mean(dim=0) + torch.randn(m2, D, device=device) * X_train_std,
        "X_test": X_test_t,
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    t0 = time.perf_counter()
    V_params, _u_params, hyperparams = train_vi(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=lr,
        num_steps=num_steps,
        log_interval=max(50, num_steps),
    )
    mu_t, var_t = predict_vi(regions, V_params, hyperparams, M=MC_num)
    elapsed = time.perf_counter() - t0
    mu = mu_t.detach().cpu().numpy().reshape(-1)
    sigma = torch.sqrt(var_t.clamp_min(1e-12)).detach().cpu().numpy().reshape(-1)
    diag = {"lmjgp_steps": int(num_steps), "MC_num": int(MC_num)}
    return mu, sigma, elapsed, diag


def _initial_cem_labels(
    X_anchor_z: torch.Tensor,
    X_neighbors_z: torch.Tensor,
    y_neighbors: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.empty(X_neighbors_z.shape[:2], dtype=torch.float32, device=device)
    for i in range(X_neighbors_z.shape[0]):
        try:
            _mu, _var, model, _h = jumpgp_ld_wrapper(
                X_neighbors_z[i].to(torch.float64),
                y_neighbors[i].view(-1, 1).to(torch.float64),
                X_anchor_z[i].view(1, -1).to(torch.float64),
                mode="CEM",
                flag=False,
                device=device,
            )
            r = model.get("r")
            if r is None:
                raise RuntimeError("JumpGP model did not return r labels.")
            r = r.reshape(-1).to(dtype=torch.float32, device=device)
            labels[i] = (r > 0.5).to(dtype=torch.float32)
        except Exception:
            d = torch.linalg.norm(X_neighbors_z[i] - X_anchor_z[i].view(1, -1), dim=1)
            k = max(2, int(0.7 * len(d)))
            keep = torch.topk(-d, k=k).indices
            labels[i].zero_()
            labels[i, keep] = 1.0
    return labels


def _run_selfcem_fixed_neighborhood(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    neighbor_idx: np.ndarray,
    *,
    Q: int,
    cem_updates: int,
    hyper_steps: int,
    gate_steps: int,
    kcov: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    _set_seed(seed)
    W0 = compute_pls_W0(X_train, y_train, Q=Q).astype(np.float64)
    W_mu = np.repeat(W0[None, :, :], len(X_test), axis=0)
    W_var = np.zeros_like(W_mu)
    X_anchor_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    X_neighbors_np = X_train[neighbor_idx]
    y_neighbors_np = y_train[neighbor_idx]
    X_neighbors_t = torch.as_tensor(X_neighbors_np, dtype=torch.float32, device=device)
    y_neighbors_t = torch.as_tensor(y_neighbors_np, dtype=torch.float32, device=device)
    W_mu_t = torch.as_tensor(W_mu, dtype=torch.float32, device=device)
    W_var_t = torch.as_tensor(W_var, dtype=torch.float32, device=device)
    X_anchor_z = torch.einsum("td,tqd->tq", X_anchor_t, W_mu_t)
    X_neighbors_z = torch.einsum("tnd,tqd->tnq", X_neighbors_t, W_mu_t)

    t0 = time.perf_counter()
    init_labels = _initial_cem_labels(X_anchor_z, X_neighbors_z, y_neighbors_t, device=device)
    state = run_uncertain_w_self_cem(
        X_anchor_t,
        X_neighbors_t,
        y_neighbors_t,
        init_labels,
        mode="anchor",
        W_mu_anchor=W_mu_t,
        W_var_anchor=W_var_t,
        init_lengthscale=1.0,
        init_signal_var=1.0,
        init_noise_var=0.05,
        config=SelfCEMConfig(
            cem_updates=cem_updates,
            hyperopt=FixedLabelHyperoptConfig(steps=hyper_steps, lr=0.04, min_noise_var=1e-4),
            gate=FixedLabelGateConfig(steps=gate_steps, lr=0.05, l2=1e-3),
            outlier_mode="background_normal",
            min_inliers=2,
            max_inlier_frac=0.95,
            refit_after_update=True,
        ),
    )
    pred = uncertain_w_cem_predict_from_labels(
        X_anchor_t,
        X_neighbors_t,
        y_neighbors_t,
        state["labels"],
        mode="anchor",
        W_mu_anchor=W_mu_t,
        W_var_anchor=W_var_t,
        lengthscale=state["lengthscale"],
        signal_var=state["signal_var"],
        noise_var=state["noise_var"],
        mean_const=state["mean_const"],
        correction_weight=kcov,
        min_inliers=2,
    )
    elapsed = time.perf_counter() - t0
    mu = pred.mu.detach().cpu().numpy().reshape(-1)
    sigma = pred.sigma.detach().cpu().numpy().reshape(-1)
    diag = {
        "self_cem_updates": int(cem_updates),
        "self_cem_hyper_steps": int(hyper_steps),
        "self_cem_gate_steps": int(gate_steps),
        "self_cem_kcov": float(kcov),
        "mean_n_inliers": pred.diagnostics.get("mean_n_inliers"),
        "history": _as_builtin(state.get("history", [])),
    }
    return mu, sigma, elapsed, diag


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    preferred = [
        "preset",
        "seed",
        "method",
        "subset",
        "status",
        "neighborhood",
        "feature_set",
        "n_train",
        "n_test",
        "n_eval",
        "D",
        "Q",
        "n_neighbors",
        *RESULT_ROW_KEYS,
        "error",
    ]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    keys = [k for k in preferred if k in all_keys] + sorted(all_keys - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _write_readme(out_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    all_rows = [r for r in rows if r.get("subset") == "all" and r.get("status") == "ok"]
    lines = [
        "# Graph-Local Model Screen",
        "",
        "Pilot model screen for `docs/graph_local_djgp_experiment_plan.md`.",
        "",
        "LMJGP uses sampled-W `predict_vi` with JumpGP-CEM refits. Self-CEM keeps the",
        "retrieved feature/graph neighborhoods fixed and does not reselect neighbors by W.",
        "",
        "| method | neighborhood | feature set | RMSE | CRPS | Cov90 | Width90 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in sorted(all_rows, key=lambda x: float(x.get("rmse", "nan"))):
        lines.append(
            f"| `{r['method']}` | {r.get('neighborhood', '')} | {r.get('feature_set', '')} | "
            f"{float(r['rmse']):.4f} | {float(r['mean_crps']):.4f} | "
            f"{float(r['coverage_90']):.3f} | {float(r['mean_width_90']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Runner summary:",
            "",
            "```json",
            json.dumps(_as_builtin(summary), indent=2),
            "```",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph-local seed0 model screen.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="dataset_b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    p.add_argument("--max_train", type=int, default=1000)
    p.add_argument("--max_test", type=int, default=80)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--n_nodes", type=int, default=300)
    p.add_argument("--n_time", type=int, default=20)
    p.add_argument("--n_blocks", type=int, default=8)
    p.add_argument("--noise_x", type=float, default=0.5)
    p.add_argument("--noise_y", type=float, default=0.1)
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--Q", type=int, default=3)
    p.add_argument("--m1", type=int, default=2)
    p.add_argument("--m2", type=int, default=40)
    p.add_argument("--lmjgp_steps", type=int, default=120)
    p.add_argument("--MC_num", type=int, default=3)
    p.add_argument("--lmjgp_lr", type=float, default=0.01)
    p.add_argument("--xgb_boots", type=int, default=10)
    p.add_argument("--dkl_epochs", type=int, default=40)
    p.add_argument("--dkl_inducing", type=int, default=64)
    p.add_argument("--self_cem_updates", type=int, default=2)
    p.add_argument("--self_cem_hyper_steps", type=int, default=10)
    p.add_argument("--self_cem_gate_steps", type=int, default=20)
    p.add_argument("--self_cem_kcov", type=float, default=0.0)
    p.add_argument(
        "--out_dir",
        default="experiments/synthetic/graph_local_model_screen_dataset_b_seed0_20260518",
    )
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choose from {ALL_METHODS}")
    preset = dict(PRESETS[args.preset])
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_graph_local_surrogate(
        n_nodes=args.n_nodes,
        n_time=args.n_time,
        D=int(preset["D"]),
        q=int(preset["q"]),
        n_blocks=args.n_blocks,
        jump=bool(preset["jump"]),
        misleading_features=bool(preset["misleading_features"]),
        noise_x=args.noise_x,
        noise_y=args.noise_y,
        seed=args.seed,
    )
    train_idx, test_idx = _make_split(
        data,
        seed=args.seed,
        train_frac=args.train_frac,
        max_train=args.max_train,
        max_test=args.max_test,
    )

    X_raw = np.asarray(data["X"], dtype=np.float64)
    X_graph = _graph_aggregate_features(data)
    y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
    jump = np.asarray(data["jump_indicator"], dtype=bool).reshape(-1)
    node_id = np.asarray(data["node_id"], dtype=np.int64)
    time_id = np.asarray(data["time_id"], dtype=np.int64)
    graph_dist = np.asarray(data["graph_dist"], dtype=np.float64)

    raw_scaler = StandardScaler()
    X_train_raw = raw_scaler.fit_transform(X_raw[train_idx])
    X_test_raw = raw_scaler.transform(X_raw[test_idx])
    graph_scaler = StandardScaler()
    X_train_graph = graph_scaler.fit_transform(X_graph[train_idx])
    X_test_graph = graph_scaler.transform(X_graph[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]
    jump_test = jump[test_idx]

    feature_neighbors = _feature_neighbor_indices(X_train_raw, X_test_raw, int(args.n_neighbors))
    feature_neighbors_graphfeat = _feature_neighbor_indices(X_train_graph, X_test_graph, int(args.n_neighbors))
    graph_neighbors = _graph_neighbor_indices(
        X_train_raw,
        X_test_raw,
        node_id[train_idx],
        time_id[train_idx],
        node_id[test_idx],
        time_id[test_idx],
        graph_dist,
        int(args.n_neighbors),
    )
    graph_neighbors_graphfeat = _graph_neighbor_indices(
        X_train_graph,
        X_test_graph,
        node_id[train_idx],
        time_id[train_idx],
        node_id[test_idx],
        time_id[test_idx],
        graph_dist,
        int(args.n_neighbors),
    )

    common_meta = {
        "preset": args.preset,
        "seed": int(args.seed),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "D": int(X_raw.shape[1]),
        "Q": int(args.Q),
        "n_neighbors": int(args.n_neighbors),
        "generator_family": "graph_local_sbm_latent_surrogate",
        "base_low_dimensional_generator": "temporal AR latent state with SBM block means",
        "expansion_method": "linear_latent_to_observed",
        "x_standardization": "train-only StandardScaler for model inputs",
        "y_standardization": "method-internal only where applicable",
    }

    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    for method in methods:
        print(f"method={method}", flush=True)
        method_meta = dict(common_meta)
        try:
            if method == "xgb_raw":
                rmse, crps, elapsed, details = run_xgboost_bootstrap_regression(
                    X_train_raw,
                    y_train,
                    X_test_raw,
                    y_test,
                    n_bootstrap=int(args.xgb_boots),
                    random_state=int(args.seed),
                )
                mu = details["mu"]
                sigma = details["sigma"]
                method_meta.update({"feature_set": "raw_x", "neighborhood": "none", "xgb_boots": int(args.xgb_boots)})
            elif method == "xgb_graph":
                rmse, crps, elapsed, details = run_xgboost_bootstrap_regression(
                    X_train_graph,
                    y_train,
                    X_test_graph,
                    y_test,
                    n_bootstrap=int(args.xgb_boots),
                    random_state=int(args.seed),
                )
                mu = details["mu"]
                sigma = details["sigma"]
                method_meta.update({"feature_set": "raw_x+block_onehot+graph_agg_x", "neighborhood": "none", "xgb_boots": int(args.xgb_boots)})
            elif method == "dkl_raw":
                rmse, crps, elapsed, mu, sigma = run_dkl_svgp_regression(
                    X_train_raw,
                    y_train,
                    X_test_raw,
                    y_test,
                    device=device,
                    epochs=int(args.dkl_epochs),
                    num_inducing=int(args.dkl_inducing),
                    seed=int(args.seed),
                )
                method_meta.update({"feature_set": "raw_x", "neighborhood": "none", "dkl_epochs": int(args.dkl_epochs)})
            elif method == "dkl_graph":
                rmse, crps, elapsed, mu, sigma = run_dkl_svgp_regression(
                    X_train_graph,
                    y_train,
                    X_test_graph,
                    y_test,
                    device=device,
                    epochs=int(args.dkl_epochs),
                    num_inducing=int(args.dkl_inducing),
                    seed=int(args.seed),
                )
                method_meta.update({"feature_set": "raw_x+block_onehot+graph_agg_x", "neighborhood": "none", "dkl_epochs": int(args.dkl_epochs)})
            elif method == "lmjgp_feature":
                mu, sigma, elapsed, diag = _run_lmjgp(
                    X_train_raw,
                    y_train,
                    X_test_raw,
                    y_test,
                    feature_neighbors,
                    Q=int(args.Q),
                    m1=int(args.m1),
                    m2=int(args.m2),
                    num_steps=int(args.lmjgp_steps),
                    MC_num=int(args.MC_num),
                    lr=float(args.lmjgp_lr),
                    seed=int(args.seed),
                    device=device,
                )
                method_meta.update({"feature_set": "raw_x", "neighborhood": "feature_knn", **diag})
            elif method == "lmjgp_graph":
                mu, sigma, elapsed, diag = _run_lmjgp(
                    X_train_raw,
                    y_train,
                    X_test_raw,
                    y_test,
                    graph_neighbors,
                    Q=int(args.Q),
                    m1=int(args.m1),
                    m2=int(args.m2),
                    num_steps=int(args.lmjgp_steps),
                    MC_num=int(args.MC_num),
                    lr=float(args.lmjgp_lr),
                    seed=int(args.seed),
                    device=device,
                )
                method_meta.update({"feature_set": "raw_x", "neighborhood": "graph_hop_then_time_then_feature", **diag})
            elif method == "lmjgp_feature_graphfeat":
                mu, sigma, elapsed, diag = _run_lmjgp(
                    X_train_graph,
                    y_train,
                    X_test_graph,
                    y_test,
                    feature_neighbors_graphfeat,
                    Q=int(args.Q),
                    m1=int(args.m1),
                    m2=int(args.m2),
                    num_steps=int(args.lmjgp_steps),
                    MC_num=int(args.MC_num),
                    lr=float(args.lmjgp_lr),
                    seed=int(args.seed),
                    device=device,
                )
                method_meta.update({"feature_set": "raw_x+block_onehot+graph_agg_x", "neighborhood": "feature_knn_graphfeat", **diag})
            elif method == "lmjgp_graph_graphfeat":
                mu, sigma, elapsed, diag = _run_lmjgp(
                    X_train_graph,
                    y_train,
                    X_test_graph,
                    y_test,
                    graph_neighbors_graphfeat,
                    Q=int(args.Q),
                    m1=int(args.m1),
                    m2=int(args.m2),
                    num_steps=int(args.lmjgp_steps),
                    MC_num=int(args.MC_num),
                    lr=float(args.lmjgp_lr),
                    seed=int(args.seed),
                    device=device,
                )
                method_meta.update({"feature_set": "raw_x+block_onehot+graph_agg_x", "neighborhood": "graph_hop_then_time_then_graphfeat", **diag})
            elif method == "selfcem_feature":
                mu, sigma, elapsed, diag = _run_selfcem_fixed_neighborhood(
                    X_train_raw,
                    y_train,
                    X_test_raw,
                    feature_neighbors,
                    Q=int(args.Q),
                    cem_updates=int(args.self_cem_updates),
                    hyper_steps=int(args.self_cem_hyper_steps),
                    gate_steps=int(args.self_cem_gate_steps),
                    kcov=float(args.self_cem_kcov),
                    seed=int(args.seed),
                    device=device,
                )
                method_meta.update({"feature_set": "raw_x", "neighborhood": "feature_knn_fixed", **diag})
            elif method == "selfcem_graph":
                mu, sigma, elapsed, diag = _run_selfcem_fixed_neighborhood(
                    X_train_raw,
                    y_train,
                    X_test_raw,
                    graph_neighbors,
                    Q=int(args.Q),
                    cem_updates=int(args.self_cem_updates),
                    hyper_steps=int(args.self_cem_hyper_steps),
                    gate_steps=int(args.self_cem_gate_steps),
                    kcov=float(args.self_cem_kcov),
                    seed=int(args.seed),
                    device=device,
                )
                method_meta.update({"feature_set": "raw_x", "neighborhood": "graph_hop_then_time_then_feature_fixed", **diag})
            elif method == "selfcem_feature_graphfeat":
                mu, sigma, elapsed, diag = _run_selfcem_fixed_neighborhood(
                    X_train_graph,
                    y_train,
                    X_test_graph,
                    feature_neighbors_graphfeat,
                    Q=int(args.Q),
                    cem_updates=int(args.self_cem_updates),
                    hyper_steps=int(args.self_cem_hyper_steps),
                    gate_steps=int(args.self_cem_gate_steps),
                    kcov=float(args.self_cem_kcov),
                    seed=int(args.seed),
                    device=device,
                )
                method_meta.update({"feature_set": "raw_x+block_onehot+graph_agg_x", "neighborhood": "feature_knn_graphfeat_fixed", **diag})
            elif method == "selfcem_graph_graphfeat":
                mu, sigma, elapsed, diag = _run_selfcem_fixed_neighborhood(
                    X_train_graph,
                    y_train,
                    X_test_graph,
                    graph_neighbors_graphfeat,
                    Q=int(args.Q),
                    cem_updates=int(args.self_cem_updates),
                    hyper_steps=int(args.self_cem_hyper_steps),
                    gate_steps=int(args.self_cem_gate_steps),
                    kcov=float(args.self_cem_kcov),
                    seed=int(args.seed),
                    device=device,
                )
                method_meta.update({"feature_set": "raw_x+block_onehot+graph_agg_x", "neighborhood": "graph_hop_then_time_then_graphfeat_fixed", **diag})
            else:
                raise ValueError(f"Unhandled method {method}")

            rows.extend(
                _rows_for_prediction(
                    method=method,
                    y_test=y_test,
                    jump_test=jump_test,
                    mu=np.asarray(mu, dtype=np.float64),
                    sigma=np.asarray(sigma, dtype=np.float64),
                    elapsed=float(elapsed),
                    meta=method_meta,
                )
            )
            predictions[method] = {"mu": np.asarray(mu), "sigma": np.asarray(sigma)}
        except Exception as exc:  # noqa: BLE001
            row = dict(method_meta)
            row.update({"method": method, "subset": "all", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            print(f"ERROR method={method}: {row['error']}", flush=True)
        _write_csv(out_dir / "metrics_partial.csv", rows)

    _write_csv(out_dir / "metrics.csv", rows)
    np.savez_compressed(
        out_dir / "predictions.npz",
        y_test=y_test,
        jump_test=jump_test,
        train_idx=train_idx,
        test_idx=test_idx,
        **{f"{m}_mu": v["mu"] for m, v in predictions.items()},
        **{f"{m}_sigma": v["sigma"] for m, v in predictions.items()},
    )
    summary = {
        "args": vars(args),
        "device": str(device),
        "preset": preset,
        "n_rows": len(rows),
        "n_ok_all": int(sum(1 for r in rows if r.get("subset") == "all" and r.get("status") == "ok")),
        "n_errors": int(sum(1 for r in rows if r.get("status") == "error")),
        "artifacts": ["metrics.csv", "metrics_partial.csv", "predictions.npz", "summary.json", "README.md"],
    }
    (out_dir / "summary.json").write_text(json.dumps(_as_builtin(summary), indent=2), encoding="utf-8")
    _write_readme(out_dir, summary, rows)
    print(f"Saved graph-local model screen to {out_dir}", flush=True)
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    main()

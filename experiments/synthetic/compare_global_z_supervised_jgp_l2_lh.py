"""Global supervised z=W(x)x + JGP on paper L2/LH synthetic data.

Implements the experiment from
``docs/global_z_supervised_jgp_plan_cn.md``: instead of reranking inside a raw
candidate pool (hybrid retrieval), learn a projection field ``W_theta(x)`` so
that the entire training/test dataset is mapped to ``z_i = W_theta(x_i) x_i``,
and rebuild neighborhoods directly in the global ``Z``-space.  ``W_theta`` is
trained with a leave-anchor-out supervised anchor predictive NLL using a
differentiable local RBF GP surrogate, while final test predictions still go
through JumpGP CEM in the learned ``Z``-space.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke check for the global-Z self-CEM alternating flow:
    python experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py ^
        --settings lh_q5_train1000 --seed 0 ^
        --variants pls_selfcem,lowrank_selfcem_supervised,lowrank_selfcem_shuffled,lowrank_selfcem_random ^
        --max_test 8 --anchor_count 12 --epochs 2 --inner_steps 2 ^
        --self_cem_hyper_steps 1 --self_cem_gate_steps 1 ^
        --out_dir experiments/synthetic/global_z_selfcem_supervised_smoke_20260517

    # Stage 1 self-CEM loop (LH seed0):
    python experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py ^
        --settings lh_q5_train1000 --seed 0 ^
        --variants raw_jgp,pls_selfcem,pca_selfcem,static_selfcem_supervised,lowrank_selfcem_supervised,lowrank_selfcem_shuffled,lowrank_selfcem_random ^
        --max_test 200 --anchor_count 100 --anchor_strategy kmeans_yvar ^
        --epochs 4 --inner_steps 20 ^
        --self_cem_updates 2 --self_cem_hyper_steps 10 --self_cem_gate_steps 20 ^
        --out_dir experiments/synthetic/global_z_selfcem_supervised_lh_seed0_20260517

    # Projected-dataset transductive self-CEM loop (no test-y leakage):
    python experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py ^
        --settings lh_q5_train1000 --seed 0 ^
        --variants pd_raw_selfcem,pd_pca_selfcem,pd_pls_selfcem,pd_static_transductive,pd_lowrank_transductive,pd_lowrank_transductive_shuffled,pd_lowrank_random ^
        --max_test 200 --epochs 4 --inner_steps 20 ^
        --self_cem_updates 2 --self_cem_hyper_steps 10 --self_cem_gate_steps 20 ^
        --out_dir experiments/synthetic/projected_dataset_transductive_selfcem_lh_seed0_20260517

    # Older surrogate-only smoke check (fast, 10 test points, very few training steps):
    python experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py ^
        --settings lh_q5_train1000 --seed 0 ^
        --variants raw_jgp,pca_jgp,pls_jgp,static_supervised,lowrank_supervised,lowrank_shuffled,lowrank_random ^
        --max_test 10 --epochs 10 --outer_refresh 2 --inner_steps 3 ^
        --out_dir experiments/synthetic/global_z_supervised_jgp_smoke_20260516

    # Stage 1 (LH seed0, all controls, 200 test points):
    python experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py ^
        --settings lh_q5_train1000 --seed 0 ^
        --variants raw_jgp,pca_jgp,pls_jgp,static_supervised,lowrank_supervised,lowrank_shuffled,lowrank_random ^
        --max_test 200 --epochs 200 --outer_refresh 10 --inner_steps 50 ^
        --out_dir experiments/synthetic/global_z_supervised_jgp_lh_seed0_20260516

This runner is intentionally separate from the hybrid retrieval workflow in
``compare_uncertain_w_cem_l2_lh.py``.  Do **not** mix the two: this experiment
must use whole-dataset projected KNN, not a raw candidate pool.  The clearer
name for this line is "projected-dataset self-CEM"; older comments may still
say "global-Z".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import find_neighborhoods  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import _run_meanw_cem_init  # noqa: E402
from djgp.projections.uncertain_w_jgp import (  # noqa: E402
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
    SelfCEMConfig,
    run_uncertain_w_self_cem,
    uncertain_w_cem_predict_from_labels,
)
from shared.utils1 import jumpgp_ld_wrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Metric utilities (shared format with prediction-head runners).
# ---------------------------------------------------------------------------

def _metrics_pack(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> list[float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, wall_sec, y, mu, sigma)


def _row_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = _row_keys(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


# ---------------------------------------------------------------------------
# Projection initialisations (W0) and residual bases (V).
# ---------------------------------------------------------------------------

def _pca_W0(X: np.ndarray, Q: int) -> np.ndarray:
    """Top-Q PCA loadings in raw-X scale, returned as W0 of shape [Q, D]."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=Q)
    pca.fit(X)
    return np.asarray(pca.components_, dtype=np.float64)  # [Q, D]


def _pls_W0(X: np.ndarray, y: np.ndarray, Q: int) -> np.ndarray:
    """PLS x_weights (raw-X scale) as W0 of shape [Q, D]."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_comp = max(1, min(Q, Xs.shape[1]))
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(Xs, y.reshape(-1, 1))
    V_std = np.asarray(pls.x_weights_, dtype=np.float64)  # [D, K]
    inv = 1.0 / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    W_raw = (inv[:, None] * V_std).T  # [K, D]
    if W_raw.shape[0] < Q:
        # pad with PCA directions if PLS dim < Q (rare here).
        pca_dirs = _pca_W0(X, Q - W_raw.shape[0])
        W_raw = np.concatenate([W_raw, pca_dirs], axis=0)
    return W_raw[:Q]


def _residual_V_basis(X: np.ndarray, W0: np.ndarray, r: int) -> np.ndarray:
    r"""PCA on the residual ``X - X W0^T W0 / ||W0 row||^2`` projection.

    Returns ``V \in R^{D x r}`` whose columns are orthonormal in raw-X space.
    The residual is intentionally not whitened so that V captures additional
    directions beyond W0's row span.
    """
    D = X.shape[1]
    # Orthonormalise W0 rows for the residual.
    W0n = W0.copy()
    norms = np.linalg.norm(W0n, axis=1, keepdims=True)
    W0n /= np.maximum(norms, 1e-12)
    # Gram-Schmidt over W0 rows to get an orthonormal basis of row span.
    q, _ = np.linalg.qr(W0n.T)  # q: [D, Q] columns orthonormal
    # Residual: X minus projection onto q.
    coef = X @ q  # [N, Q]
    X_res = X - coef @ q.T  # [N, D]
    from sklearn.decomposition import PCA

    n_comp = max(1, min(r, D - q.shape[1]))
    pca = PCA(n_components=n_comp)
    pca.fit(X_res)
    V = np.asarray(pca.components_, dtype=np.float64).T  # [D, n_comp]
    if V.shape[1] < r:
        pad = np.zeros((D, r - V.shape[1]), dtype=np.float64)
        V = np.concatenate([V, pad], axis=1)
    return V


# ---------------------------------------------------------------------------
# W_theta network definitions.
# ---------------------------------------------------------------------------

class StaticW(nn.Module):
    """``W_theta(x) = W_0`` (single global linear projection, learnable)."""

    def __init__(self, W0_init: torch.Tensor) -> None:
        super().__init__()
        self.W0 = nn.Parameter(W0_init.clone())  # [Q, D]
        self.Q = int(W0_init.shape[0])
        self.D = int(W0_init.shape[1])

    def project(self, X: torch.Tensor) -> torch.Tensor:
        # z_i = W0 x_i  => [N, Q]
        return X @ self.W0.t()


class LowRankW(nn.Module):
    """``W_theta(x) = W_0 + C_theta(x) V^T``.

    ``W_0`` and ``V`` are buffers (fixed PCA/PLS init), ``C_theta`` is an MLP
    that maps ``x`` to a ``Q*r`` residual coefficient vector.  ``W_0`` is
    optionally learnable; default keeps it fixed so the residual is the only
    learnable contribution from the input-dependent net.
    """

    def __init__(
        self,
        W0_init: torch.Tensor,
        V_init: torch.Tensor,
        *,
        hidden: int = 64,
        learn_W0: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        Q, D = W0_init.shape
        r = V_init.shape[1]
        self.Q = Q
        self.D = D
        self.r = r
        if learn_W0:
            self.W0 = nn.Parameter(W0_init.clone())
        else:
            self.register_buffer("W0", W0_init.clone())
        self.register_buffer("V", V_init.clone())
        self.net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, Q * r),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def project(self, X: torch.Tensor) -> torch.Tensor:
        C = self.net(X).view(-1, self.Q, self.r)  # [N, Q, r]
        # z_i = W_0 x_i + C(x_i) (V^T x_i)
        z_static = X @ self.W0.t()  # [N, Q]
        VtX = X @ self.V  # [N, r]
        z_res = torch.einsum("nqr,nr->nq", C, VtX)  # [N, Q]
        return z_static + z_res


# ---------------------------------------------------------------------------
# Differentiable batched local RBF GP NLL surrogate.
# ---------------------------------------------------------------------------

class LocalRBFGPSurrogate(nn.Module):
    """Anchor-batched RBF GP NLL for leave-anchor-out supervised W training.

    Parameters:
        log_ell, log_sf, log_sn (scalar): RBF hyperparameters.
    """

    def __init__(self, init_ell: float = 1.0, init_sf: float = 1.0, init_sn: float = 0.1) -> None:
        super().__init__()
        self.log_ell = nn.Parameter(torch.tensor(math.log(init_ell), dtype=torch.float64))
        self.log_sf = nn.Parameter(torch.tensor(math.log(init_sf), dtype=torch.float64))
        self.log_sn = nn.Parameter(torch.tensor(math.log(init_sn), dtype=torch.float64))

    def forward(
        self,
        Z_anchor: torch.Tensor,
        Z_neighbors: torch.Tensor,
        y_anchor: torch.Tensor,
        y_neighbors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched leave-anchor-out NLL.

        Inputs:
            Z_anchor      [B, Q]
            Z_neighbors   [B, K, Q]
            y_anchor      [B]
            y_neighbors   [B, K]

        Returns:
            nll_mean (scalar), mu [B], var [B].
        """
        Z_anchor = Z_anchor.to(dtype=torch.float64)
        Z_neighbors = Z_neighbors.to(dtype=torch.float64)
        y_anchor = y_anchor.to(dtype=torch.float64)
        y_neighbors = y_neighbors.to(dtype=torch.float64)

        ell = torch.exp(self.log_ell).clamp(min=1e-3, max=1e3)
        sf2 = torch.exp(2.0 * self.log_sf).clamp(min=1e-8, max=1e8)
        sn2 = torch.exp(2.0 * self.log_sn).clamp(min=1e-8, max=1e8)

        B, K, _ = Z_neighbors.shape
        # Pairwise squared distances inside each neighborhood.
        d2_nn = torch.cdist(Z_neighbors, Z_neighbors, p=2.0).pow(2)  # [B, K, K]
        K_nn = sf2 * torch.exp(-0.5 * d2_nn / (ell ** 2))
        I = torch.eye(K, dtype=K_nn.dtype, device=K_nn.device).unsqueeze(0)
        K_nn = K_nn + (sn2 + 1e-6) * I

        # K_an: [B, K]
        d2_an = torch.cdist(Z_anchor.unsqueeze(1), Z_neighbors, p=2.0).pow(2).squeeze(1)
        K_an = sf2 * torch.exp(-0.5 * d2_an / (ell ** 2))  # [B, K]

        # Local mean subtraction for numerical stability.
        mean_y = y_neighbors.mean(dim=1, keepdim=True)  # [B, 1]
        y_centered = y_neighbors - mean_y  # [B, K]

        L = torch.linalg.cholesky(K_nn)  # [B, K, K]
        alpha = torch.cholesky_solve(y_centered.unsqueeze(-1), L).squeeze(-1)  # [B, K]
        mu = (K_an * alpha).sum(dim=-1) + mean_y.squeeze(-1)  # [B]
        v = torch.cholesky_solve(K_an.unsqueeze(-1), L).squeeze(-1)  # [B, K]
        var = sf2 - (K_an * v).sum(dim=-1) + sn2
        var = torch.clamp(var, min=1e-6)

        diff = y_anchor - mu
        nll = 0.5 * torch.log(2.0 * math.pi * var) + 0.5 * diff.pow(2) / var
        return nll.mean(), mu.detach(), var.detach()


# ---------------------------------------------------------------------------
# Z-space KNN refresh.
# ---------------------------------------------------------------------------

def _z_knn_indices(Z_anchor: torch.Tensor, Z_pool: torch.Tensor, K: int, exclude_self: torch.Tensor | None) -> torch.Tensor:
    """Return [B, K] indices of K nearest pool points to each anchor in Z space.

    ``exclude_self`` is an optional ``[B]`` long tensor of indices to exclude
    from the pool for each anchor (used for leave-anchor-out training).
    """
    dists = torch.cdist(Z_anchor, Z_pool, p=2.0)  # [B, N]
    if exclude_self is not None:
        rows = torch.arange(Z_anchor.shape[0], device=Z_anchor.device)
        dists[rows, exclude_self] = float("inf")
    _, idx = torch.topk(dists, K, dim=1, largest=False)
    return idx  # [B, K]


# ---------------------------------------------------------------------------
# Supervised W training loop.
# ---------------------------------------------------------------------------

def _train_supervised_W(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    W_net: nn.Module,
    K_neighbor: int,
    anchor_idx: torch.Tensor,
    epochs: int,
    outer_refresh: int,
    inner_steps: int,
    lr_w: float,
    lr_gp: float,
    weight_decay: float,
    w_l2: float,
    anchor_batch: int,
    init_ell: float,
    init_sf: float,
    init_sn: float,
    device: torch.device,
    verbose: bool = False,
) -> tuple[nn.Module, LocalRBFGPSurrogate, dict[str, Any]]:
    """Alternating KNN-refresh + differentiable surrogate optimization.

    Returns the trained ``W_net``, the GP surrogate, and a diagnostics dict.
    """
    surrogate = LocalRBFGPSurrogate(init_ell=init_ell, init_sf=init_sf, init_sn=init_sn).to(device)
    # Separate parameter groups so the GP hypers can move faster than the W net.
    w_params = list(W_net.parameters())
    gp_params = list(surrogate.parameters())
    optim = torch.optim.Adam([
        {"params": w_params, "lr": lr_w, "weight_decay": weight_decay},
        {"params": gp_params, "lr": lr_gp, "weight_decay": 0.0},
    ])

    diag: dict[str, Any] = {
        "train_nll_history": [],
        "ell_history": [],
        "sf_history": [],
        "sn_history": [],
        "label_change_history": [],
        "z_norm_mean_history": [],
        "raw_overlap_history": [],
    }

    n_anchors = int(anchor_idx.shape[0])
    K = int(K_neighbor)

    # Pre-compute raw-X KNN of each anchor inside the training set, for overlap diagnostics.
    with torch.no_grad():
        raw_dists = torch.cdist(X_train[anchor_idx], X_train, p=2.0)
        rows = torch.arange(n_anchors, device=device)
        raw_dists[rows, anchor_idx] = float("inf")
        _, raw_knn = torch.topk(raw_dists, K, dim=1, largest=False)
    raw_knn_sets = [set(map(int, raw_knn[i].cpu().tolist())) for i in range(n_anchors)]

    prev_neighbor_sets: list[set[int]] | None = None

    total_outer = int(epochs)
    iterator = range(total_outer)
    if verbose:
        iterator = tqdm(iterator, desc="W train", leave=False)
    for outer in iterator:
        # ---------- E-step: refresh Z-KNN ----------
        W_net.eval()
        with torch.no_grad():
            Z_train_full = W_net.project(X_train).detach()
            knn_idx = _z_knn_indices(Z_train_full[anchor_idx], Z_train_full, K, exclude_self=anchor_idx)
        W_net.train()

        cur_sets = [set(map(int, knn_idx[i].cpu().tolist())) for i in range(n_anchors)]
        if prev_neighbor_sets is None:
            label_change = 1.0
        else:
            denom = max(1, K * n_anchors)
            symdiff = 0
            for new_set, old_set in zip(cur_sets, prev_neighbor_sets):
                symdiff += len(new_set.symmetric_difference(old_set))
            label_change = symdiff / (2.0 * denom)  # normalized [0,1]
        prev_neighbor_sets = cur_sets

        z_norm_mean = float(torch.linalg.vector_norm(Z_train_full, dim=1).mean().item())
        overlaps = [len(cur_sets[i] & raw_knn_sets[i]) / K for i in range(n_anchors)]
        raw_overlap = float(np.mean(overlaps))

        diag["z_norm_mean_history"].append(z_norm_mean)
        diag["raw_overlap_history"].append(raw_overlap)
        diag["label_change_history"].append(label_change)

        # ---------- M-step: refresh inner gradient steps with current KNN ----------
        last_nll = float("nan")
        for inner in range(int(inner_steps)):
            # Shuffle anchors and form mini-batches.
            perm = torch.randperm(n_anchors, device=device)
            anchor_perm = anchor_idx[perm]
            knn_perm = knn_idx[perm]
            nll_running = 0.0
            n_batches = 0
            for start in range(0, n_anchors, int(anchor_batch)):
                end = min(start + int(anchor_batch), n_anchors)
                a_idx = anchor_perm[start:end]
                n_idx = knn_perm[start:end]
                # Project anchor and neighbors through current W_theta.
                X_a = X_train[a_idx]
                X_n = X_train[n_idx.reshape(-1)].view(*n_idx.shape, X_train.shape[1])
                # Flatten for projection, then reshape.
                B = X_a.shape[0]
                Z_a = W_net.project(X_a)  # [B, Q]
                Z_n_flat = W_net.project(X_n.reshape(-1, X_train.shape[1]))  # [B*K, Q]
                Z_n = Z_n_flat.view(B, K, -1)
                y_a = y_train[a_idx]
                y_n = y_train[n_idx.reshape(-1)].view(B, K)

                nll_mean, _, _ = surrogate(Z_a, Z_n, y_a, y_n)
                reg = torch.tensor(0.0, device=device, dtype=nll_mean.dtype)
                if w_l2 > 0:
                    if hasattr(W_net, "W0"):
                        W0 = W_net.W0
                        reg = reg + w_l2 * (W0 ** 2).sum()
                loss = nll_mean + reg
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(w_params + gp_params, max_norm=5.0)
                optim.step()
                nll_running += float(nll_mean.detach().item())
                n_batches += 1
            last_nll = nll_running / max(1, n_batches)

        diag["train_nll_history"].append(last_nll)
        diag["ell_history"].append(float(torch.exp(surrogate.log_ell).item()))
        diag["sf_history"].append(float(torch.exp(surrogate.log_sf).item()))
        diag["sn_history"].append(float(torch.exp(surrogate.log_sn).item()))

        if verbose:
            iterator.set_postfix(  # type: ignore[attr-defined]
                nll=last_nll,
                ell=diag["ell_history"][-1],
                sf=diag["sf_history"][-1],
                sn=diag["sn_history"][-1],
                lab=label_change,
                ov=raw_overlap,
            )

    diag["final_ell"] = diag["ell_history"][-1] if diag["ell_history"] else float("nan")
    diag["final_sf"] = diag["sf_history"][-1] if diag["sf_history"] else float("nan")
    diag["final_sn"] = diag["sn_history"][-1] if diag["sn_history"] else float("nan")
    diag["final_train_nll"] = diag["train_nll_history"][-1] if diag["train_nll_history"] else float("nan")
    diag["final_label_change"] = diag["label_change_history"][-1] if diag["label_change_history"] else float("nan")
    diag["final_z_norm_mean"] = diag["z_norm_mean_history"][-1] if diag["z_norm_mean_history"] else float("nan")
    diag["final_raw_overlap"] = diag["raw_overlap_history"][-1] if diag["raw_overlap_history"] else float("nan")
    diag["raw_knn_train"] = raw_knn.detach().cpu()
    return W_net, surrogate, diag


# ---------------------------------------------------------------------------
# JumpGP prediction in a given feature space.
# ---------------------------------------------------------------------------

def _jumpgp_predict_in_feature_space(
    F_train: np.ndarray,
    y_train: np.ndarray,
    F_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_neighbors: int,
    device: torch.device,
    desc: str,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    F_train_t = torch.as_tensor(F_train, dtype=torch.float32)
    F_test_t = torch.as_tensor(F_test, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    neighborhoods = find_neighborhoods(F_test_t, F_train_t, y_train_t, M=int(n_neighbors))

    mus = np.empty(F_test.shape[0], dtype=np.float64)
    sigmas = np.empty(F_test.shape[0], dtype=np.float64)
    bad_sig2 = 0
    t0 = time.perf_counter()
    for j, nb in enumerate(tqdm(neighborhoods, desc=desc, leave=False)):
        F_nb = nb["X_neighbors"].to(device)
        y_nb = nb["y_neighbors"].to(device).view(-1, 1)
        f_star = F_test_t[j].view(1, -1).to(device)
        mu_j, sig2_j, _, _ = jumpgp_ld_wrapper(
            F_nb,
            y_nb,
            f_star,
            mode="CEM",
            flag=False,
            device=device,
        )
        mu_val = float(mu_j.detach().cpu().reshape(-1)[0].item())
        sig2_val = float(sig2_j.detach().cpu().reshape(-1)[0].item())
        if not math.isfinite(sig2_val) or sig2_val < 0.0:
            bad_sig2 += 1
            sig2_val = 1e-12
        mus[j] = mu_val
        sigmas[j] = math.sqrt(max(sig2_val, 1e-12))
    wall_sec = time.perf_counter() - t0
    pack = _metrics_pack(y_test, mus, sigmas, wall_sec)
    metrics = {key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)}
    extras = {"n_bad_sig2": int(bad_sig2)}
    return metrics, wall_sec, extras


# ---------------------------------------------------------------------------
# Current self-CEM head in global-Z space.
# ---------------------------------------------------------------------------

def _identity_w_moments(T: int, Q: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    eye = torch.eye(int(Q), device=device, dtype=dtype).reshape(1, int(Q), int(Q)).expand(int(T), -1, -1).contiguous()
    return {
        "W_mu_anchor": eye,
        "W_var_anchor": torch.zeros_like(eye),
    }


def _self_cem_config(args: argparse.Namespace) -> SelfCEMConfig:
    noise_floor = float(args.self_cem_noise_std_floor) ** 2
    return SelfCEMConfig(
        cem_updates=int(args.self_cem_updates),
        hyperopt=FixedLabelHyperoptConfig(
            steps=int(args.self_cem_hyper_steps),
            lr=float(args.self_cem_hyper_lr),
            min_noise_var=float(noise_floor),
            max_noise_var=10.0,
            min_signal_var=1e-4,
            max_signal_var=100.0,
            min_lengthscale=1e-3,
            max_lengthscale=20.0,
        ),
        gate=FixedLabelGateConfig(
            steps=int(args.self_cem_gate_steps),
            lr=float(args.self_cem_gate_lr),
            l2=float(args.self_cem_gate_l2),
            max_norm=float(args.self_cem_gate_max_norm),
            gh_points=int(args.self_cem_gh_points),
        ),
        outlier_mode="background_normal",
        outlier_sigma_mult=2.5,
        background_scale_mult=3.0,
        min_inliers=int(args.min_inliers),
        max_inlier_frac=0.95,
        refit_after_update=True,
    )


def _fit_selfcem_on_feature_neighbors(
    F_anchor: torch.Tensor,
    F_neighbors: torch.Tensor,
    y_neighbors_std: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Fit the current uncertain-W self-CEM head on deterministic Z features.

    The learned global projection has already produced ``F``.  Inside this
    head, ``W`` is the deterministic identity map in feature space, so the
    labels and hyperparameters are exactly the current self-CEM flow applied to
    the reprojected dataset.
    """
    T, _K, Q = F_neighbors.shape
    device = F_neighbors.device
    dtype = F_neighbors.dtype
    W_eye = torch.eye(Q, device=device, dtype=dtype)
    init_state, init_diag = _run_meanw_cem_init(F_anchor, F_neighbors, y_neighbors_std, W_eye)
    moments = _identity_w_moments(T, Q, device=device, dtype=dtype)
    state = run_uncertain_w_self_cem(
        F_anchor,
        F_neighbors,
        y_neighbors_std,
        init_state["labels"].bool(),
        mode="anchor",
        init_lengthscale=init_state["lengthscale"],
        init_signal_var=init_state["signal_var"],
        init_noise_var=init_state["noise_var"].clamp_min(float(args.self_cem_noise_std_floor) ** 2),
        gate_init=init_state["gate"],
        config=_self_cem_config(args),
        **moments,
    )
    labels = state["labels"].bool()
    init_labels = init_state["labels"].bool()
    diag = {
        **init_diag,
        "selfcem_inlier_rate": float(labels.to(dtype=torch.float64).mean().detach().cpu().item()),
        "selfcem_label_change": float((labels != init_labels).to(dtype=torch.float64).mean().detach().cpu().item()),
        "selfcem_mean_lengthscale": float(state["lengthscale"].detach().mean().cpu().item()),
        "selfcem_mean_signal_var": float(state["signal_var"].detach().mean().cpu().item()),
        "selfcem_mean_noise_var": float(state["noise_var"].detach().mean().cpu().item()),
    }
    return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}, diag


def _selfcem_predict_in_feature_space(
    F_train: np.ndarray,
    y_train: np.ndarray,
    F_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_neighbors: int,
    args: argparse.Namespace,
    device: torch.device,
    desc: str,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    F_train_t = torch.as_tensor(F_train, dtype=torch.float32, device=device)
    F_test_t = torch.as_tensor(F_test, dtype=torch.float32, device=device)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if not math.isfinite(y_std) or y_std <= 1e-12:
        y_std = 1.0
    y_train_std_t = torch.as_tensor((y_train - y_mean) / y_std, dtype=torch.float32, device=device)

    k = min(int(n_neighbors), F_train_t.shape[0])
    with torch.no_grad():
        d = torch.cdist(F_test_t, F_train_t, p=2.0)
        idx = torch.topk(d, k=k, dim=1, largest=False).indices
        F_neighbors = F_train_t[idx].contiguous()
        y_neighbors_std = y_train_std_t[idx].contiguous()

    t0 = time.perf_counter()
    state, fit_diag = _fit_selfcem_on_feature_neighbors(
        F_test_t,
        F_neighbors,
        y_neighbors_std,
        args=args,
    )
    moments = _identity_w_moments(F_test_t.shape[0], F_test_t.shape[1], device=device, dtype=F_test_t.dtype)
    out = uncertain_w_cem_predict_from_labels(
        F_test_t,
        F_neighbors,
        y_neighbors_std,
        state["labels"],
        mode="anchor",
        lengthscale=state["lengthscale"],
        signal_var=state["signal_var"],
        noise_var=state["noise_var"],
        mean_const=state["mean_const"],
        correction_weight=float(args.self_cem_kcov),
        min_inliers=int(args.min_inliers),
        **moments,
    )
    wall_sec = time.perf_counter() - t0
    mu = out.mu.detach().cpu().numpy().reshape(-1) * y_std + y_mean
    sigma = out.sigma.detach().cpu().numpy().reshape(-1) * abs(y_std)
    pack = _metrics_pack(y_test, mu, sigma, wall_sec)
    metrics = {key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)}
    extras = {
        "selfcem_mean_inlier_rate": float(fit_diag.get("selfcem_inlier_rate", float("nan"))),
        "selfcem_label_change": float(fit_diag.get("selfcem_label_change", float("nan"))),
        "selfcem_mean_lengthscale": float(fit_diag.get("selfcem_mean_lengthscale", float("nan"))),
        "selfcem_mean_signal_var": float(fit_diag.get("selfcem_mean_signal_var", float("nan"))),
        "selfcem_mean_noise_var": float(fit_diag.get("selfcem_mean_noise_var", float("nan"))),
        "selfcem_init_failed": int(fit_diag.get("meanw_cem_failed", 0)),
        "mean_n_inliers": float(out.diagnostics["mean_n_inliers"]),
        "fallback_count": int(out.diagnostics["fallback_count"]),
        "desc": desc,
    }
    return metrics, wall_sec, extras


def _rbf_kernel_scaled(Z1: torch.Tensor, Z2: torch.Tensor, ell: torch.Tensor, signal_var: torch.Tensor) -> torch.Tensor:
    diff = (Z1.unsqueeze(1) - Z2.unsqueeze(0)) / ell.reshape(1, 1, -1).clamp_min(1e-4)
    d2 = diff.pow(2).sum(dim=-1)
    return signal_var.clamp_min(1e-8) * torch.exp(-0.5 * d2)


def _masked_selfcem_anchor_nll(
    X_train: torch.Tensor,
    y_train_std: torch.Tensor,
    *,
    W_net: nn.Module,
    anchor_idx: torch.Tensor,
    neighbor_idx: torch.Tensor,
    labels: torch.Tensor,
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
    noise_var: torch.Tensor,
    mean_const: torch.Tensor,
    min_inliers: int,
) -> torch.Tensor:
    """Differentiable supervised anchor NLL under fixed Z-KNN and CEM labels."""
    X_a = X_train[anchor_idx]
    X_n = X_train[neighbor_idx.reshape(-1)].view(*neighbor_idx.shape, X_train.shape[1])
    B, K, _D = X_n.shape
    Z_a = W_net.project(X_a).to(dtype=torch.float64)
    Z_n = W_net.project(X_n.reshape(-1, X_train.shape[1])).view(B, K, -1).to(dtype=torch.float64)
    y_a = y_train_std[anchor_idx].to(dtype=torch.float64)
    y_n = y_train_std[neighbor_idx.reshape(-1)].view(B, K).to(dtype=torch.float64)
    labels = labels.bool()
    lengthscale = lengthscale.to(dtype=torch.float64)
    signal_var = signal_var.to(dtype=torch.float64)
    noise_var = noise_var.to(dtype=torch.float64)
    mean_const = mean_const.to(dtype=torch.float64)

    losses: list[torch.Tensor] = []
    for b in range(B):
        mask = labels[b]
        if int(mask.sum().item()) < int(min_inliers):
            k = min(max(int(min_inliers), 1), K)
            top = torch.topk(labels[b].to(dtype=torch.float64), k=k, largest=True).indices
            mask = torch.zeros(K, device=labels.device, dtype=torch.bool)
            mask[top] = True
        Zi = Z_n[b, mask]
        yi = y_n[b, mask]
        ell = lengthscale[b]
        sf = signal_var[b]
        sn = noise_var[b].clamp_min(1e-8)
        m = mean_const[b]
        Kii = _rbf_kernel_scaled(Zi, Zi, ell, sf)
        Kii = Kii + (sn + 1e-6) * torch.eye(Kii.shape[0], device=Kii.device, dtype=Kii.dtype)
        kai = _rbf_kernel_scaled(Z_a[b : b + 1], Zi, ell, sf).reshape(-1)
        L = torch.linalg.cholesky(Kii)
        alpha = torch.cholesky_solve((yi - m).unsqueeze(-1), L).reshape(-1)
        mu = m + torch.dot(kai, alpha)
        v = torch.cholesky_solve(kai.unsqueeze(-1), L).reshape(-1)
        var = (sf - torch.dot(kai, v) + sn).clamp_min(1e-8)
        diff = y_a[b] - mu
        losses.append(0.5 * torch.log(2.0 * math.pi * var) + 0.5 * diff.pow(2) / var)
    return torch.stack(losses).mean()


def _fixed_cem_transductive_loss(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train_std: torch.Tensor,
    *,
    W_net: nn.Module,
    test_idx: torch.Tensor,
    neighbor_idx: torch.Tensor,
    labels: torch.Tensor,
    lengthscale: torch.Tensor,
    signal_var: torch.Tensor,
    noise_var: torch.Tensor,
    mean_const: torch.Tensor,
    gate: torch.Tensor,
    min_inliers: int,
    gate_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transductive fixed-CEM objective for test anchors.

    This uses no test responses.  For fixed test-anchor neighborhoods and CEM
    partitions, it updates ``W_theta`` by making the selected inlier training
    responses likely under the local GP and by preserving the CEM gate labels
    around the projected test anchor.
    """
    X_a = X_test[test_idx]
    X_n = X_train[neighbor_idx.reshape(-1)].view(*neighbor_idx.shape, X_train.shape[1])
    B, K, _D = X_n.shape
    Z_a = W_net.project(X_a).to(dtype=torch.float64)
    Z_n = W_net.project(X_n.reshape(-1, X_train.shape[1])).view(B, K, -1).to(dtype=torch.float64)
    y_n = y_train_std[neighbor_idx.reshape(-1)].view(B, K).to(dtype=torch.float64)
    labels_b = labels.bool()
    lengthscale = lengthscale.to(dtype=torch.float64)
    signal_var = signal_var.to(dtype=torch.float64)
    noise_var = noise_var.to(dtype=torch.float64)
    mean_const = mean_const.to(dtype=torch.float64)
    gate = gate.to(dtype=torch.float64)

    gp_losses: list[torch.Tensor] = []
    gate_losses: list[torch.Tensor] = []
    for b in range(B):
        mask = labels_b[b]
        if int(mask.sum().item()) < int(min_inliers):
            k = min(max(int(min_inliers), 1), K)
            top = torch.topk(labels_b[b].to(dtype=torch.float64), k=k, largest=True).indices
            mask = torch.zeros(K, device=labels_b.device, dtype=torch.bool)
            mask[top] = True
        Zi = Z_n[b, mask]
        yi = y_n[b, mask]
        ell = lengthscale[b]
        sf = signal_var[b]
        sn = noise_var[b].clamp_min(1e-8)
        m = mean_const[b]
        Kii = _rbf_kernel_scaled(Zi, Zi, ell, sf)
        Kii = Kii + (sn + 1e-6) * torch.eye(Kii.shape[0], device=Kii.device, dtype=Kii.dtype)
        L = torch.linalg.cholesky(Kii)
        yc = yi - m
        alpha = torch.cholesky_solve(yc.unsqueeze(-1), L).reshape(-1)
        logdet = 2.0 * torch.diagonal(L).log().sum()
        n_i = float(yi.numel())
        gp_nll = 0.5 * torch.dot(yc, alpha) + 0.5 * logdet + 0.5 * n_i * math.log(2.0 * math.pi)
        gp_losses.append(gp_nll / max(n_i, 1.0))

        delta = Z_n[b] - Z_a[b].reshape(1, -1)
        logits = gate[b, 0] + (delta * gate[b, 1:].reshape(1, -1)).sum(dim=1)
        target = labels_b[b].to(dtype=torch.float64)
        gate_losses.append(F.binary_cross_entropy_with_logits(logits, target))

    gp_loss = torch.stack(gp_losses).mean()
    gate_loss = torch.stack(gate_losses).mean()
    total = gp_loss + float(gate_weight) * gate_loss
    return total, gp_loss.detach(), gate_loss.detach()


def _train_selfcem_supervised_W(
    X_train: torch.Tensor,
    y_train_std: torch.Tensor,
    *,
    W_net: nn.Module,
    K_neighbor: int,
    anchor_idx: torch.Tensor,
    epochs: int,
    inner_steps: int,
    lr_w: float,
    weight_decay: float,
    w_l2: float,
    anchor_batch: int,
    args: argparse.Namespace,
    device: torch.device,
    verbose: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Alternating global-Z KNN, self-CEM refit, and fixed-label W M-step."""
    w_params = list(W_net.parameters())
    optim = torch.optim.Adam([{"params": w_params, "lr": lr_w, "weight_decay": weight_decay}])
    n_anchors = int(anchor_idx.shape[0])
    K = int(K_neighbor)
    diag: dict[str, Any] = {
        "train_nll_history": [],
        "neighbor_change_history": [],
        "label_change_history": [],
        "raw_overlap_history": [],
        "inlier_rate_history": [],
        "noise_var_history": [],
        "z_norm_mean_history": [],
    }
    with torch.no_grad():
        raw_dists = torch.cdist(X_train[anchor_idx], X_train, p=2.0)
        rows = torch.arange(n_anchors, device=device)
        raw_dists[rows, anchor_idx] = float("inf")
        raw_knn = torch.topk(raw_dists, K, dim=1, largest=False).indices
    raw_knn_sets = [set(map(int, raw_knn[i].detach().cpu().tolist())) for i in range(n_anchors)]
    prev_neighbor_sets: list[set[int]] | None = None
    prev_labels: torch.Tensor | None = None

    iterator = range(int(epochs))
    if verbose:
        iterator = tqdm(iterator, desc="global-Z self-CEM W train", leave=False)

    last_state: dict[str, torch.Tensor] | None = None
    last_knn_idx: torch.Tensor | None = None
    for outer in iterator:
        W_net.eval()
        with torch.no_grad():
            Z_train_full = W_net.project(X_train).detach()
            knn_idx = _z_knn_indices(Z_train_full[anchor_idx], Z_train_full, K, exclude_self=anchor_idx)
            Z_anchor = Z_train_full[anchor_idx]
            Z_neighbors = Z_train_full[knn_idx.reshape(-1)].view(n_anchors, K, -1)
            y_neighbors_std = y_train_std[knn_idx.reshape(-1)].view(n_anchors, K)
        state, cem_diag = _fit_selfcem_on_feature_neighbors(Z_anchor, Z_neighbors, y_neighbors_std, args=args)
        W_net.train()

        cur_sets = [set(map(int, knn_idx[i].detach().cpu().tolist())) for i in range(n_anchors)]
        if prev_neighbor_sets is None:
            neighbor_change = 1.0
        else:
            symdiff = sum(len(new.symmetric_difference(old)) for new, old in zip(cur_sets, prev_neighbor_sets))
            neighbor_change = symdiff / float(max(1, 2 * K * n_anchors))
        prev_neighbor_sets = cur_sets
        raw_overlap = float(np.mean([len(cur_sets[i] & raw_knn_sets[i]) / K for i in range(n_anchors)]))
        labels = state["labels"].bool()
        if prev_labels is None:
            label_change = 1.0
        else:
            label_change = float((labels != prev_labels).to(dtype=torch.float64).mean().detach().cpu().item())
        prev_labels = labels.detach().clone()

        diag["neighbor_change_history"].append(float(neighbor_change))
        diag["label_change_history"].append(float(label_change))
        diag["raw_overlap_history"].append(float(raw_overlap))
        diag["inlier_rate_history"].append(float(labels.to(dtype=torch.float64).mean().detach().cpu().item()))
        diag["noise_var_history"].append(float(state["noise_var"].detach().mean().cpu().item()))
        diag["z_norm_mean_history"].append(float(torch.linalg.vector_norm(Z_train_full, dim=1).mean().cpu().item()))

        last_nll = float("nan")
        for _inner in range(int(inner_steps)):
            perm = torch.randperm(n_anchors, device=device)
            nll_running = 0.0
            n_batches = 0
            for start in range(0, n_anchors, int(anchor_batch)):
                sel = perm[start : min(start + int(anchor_batch), n_anchors)]
                nll = _masked_selfcem_anchor_nll(
                    X_train,
                    y_train_std,
                    W_net=W_net,
                    anchor_idx=anchor_idx[sel],
                    neighbor_idx=knn_idx[sel],
                    labels=labels[sel],
                    lengthscale=state["lengthscale"][sel],
                    signal_var=state["signal_var"][sel],
                    noise_var=state["noise_var"][sel],
                    mean_const=state["mean_const"][sel],
                    min_inliers=int(args.min_inliers),
                )
                reg = nll.new_zeros(())
                if w_l2 > 0 and hasattr(W_net, "W0"):
                    reg = reg + float(w_l2) * (W_net.W0.to(dtype=nll.dtype) ** 2).sum()
                loss = nll + reg
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(w_params, max_norm=5.0)
                optim.step()
                nll_running += float(nll.detach().cpu().item())
                n_batches += 1
            last_nll = nll_running / max(1, n_batches)
        diag["train_nll_history"].append(float(last_nll))
        last_state = state
        last_knn_idx = knn_idx.detach()

        if verbose:
            iterator.set_postfix(  # type: ignore[attr-defined]
                nll=last_nll,
                nb=neighbor_change,
                lab=label_change,
                ov=raw_overlap,
                inr=diag["inlier_rate_history"][-1],
            )

    diag["final_train_nll"] = diag["train_nll_history"][-1] if diag["train_nll_history"] else float("nan")
    diag["final_neighbor_change"] = diag["neighbor_change_history"][-1] if diag["neighbor_change_history"] else float("nan")
    diag["final_label_change"] = diag["label_change_history"][-1] if diag["label_change_history"] else float("nan")
    diag["final_raw_overlap"] = diag["raw_overlap_history"][-1] if diag["raw_overlap_history"] else float("nan")
    diag["final_inlier_rate"] = diag["inlier_rate_history"][-1] if diag["inlier_rate_history"] else float("nan")
    diag["final_noise_var"] = diag["noise_var_history"][-1] if diag["noise_var_history"] else float("nan")
    diag["final_z_norm_mean"] = diag["z_norm_mean_history"][-1] if diag["z_norm_mean_history"] else float("nan")
    diag["last_knn_idx"] = None if last_knn_idx is None else last_knn_idx.detach().cpu()
    if last_state is not None:
        diag["final_lengthscale"] = float(last_state["lengthscale"].detach().mean().cpu().item())
        diag["final_signal_var"] = float(last_state["signal_var"].detach().mean().cpu().item())
    return W_net, diag


def _train_transductive_selfcem_W(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train_std: torch.Tensor,
    *,
    W_net: nn.Module,
    K_neighbor: int,
    epochs: int,
    inner_steps: int,
    lr_w: float,
    weight_decay: float,
    w_l2: float,
    anchor_batch: int,
    args: argparse.Namespace,
    device: torch.device,
    verbose: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Alternate projected-dataset test-anchor CEM refresh and W M-step."""
    w_params = list(W_net.parameters())
    optim = torch.optim.Adam([{"params": w_params, "lr": lr_w, "weight_decay": weight_decay}])
    T = int(X_test.shape[0])
    K = int(K_neighbor)
    test_all = torch.arange(T, device=device, dtype=torch.long)
    diag: dict[str, Any] = {
        "train_loss_history": [],
        "gp_loss_history": [],
        "gate_loss_history": [],
        "neighbor_change_history": [],
        "label_change_history": [],
        "raw_overlap_history": [],
        "inlier_rate_history": [],
        "noise_var_history": [],
        "z_norm_mean_history": [],
    }

    with torch.no_grad():
        raw_dists = torch.cdist(X_test, X_train, p=2.0)
        raw_knn = torch.topk(raw_dists, K, dim=1, largest=False).indices
    raw_knn_sets = [set(map(int, raw_knn[i].detach().cpu().tolist())) for i in range(T)]
    prev_neighbor_sets: list[set[int]] | None = None
    prev_labels: torch.Tensor | None = None

    iterator = range(int(epochs))
    if verbose:
        iterator = tqdm(iterator, desc="projected-dataset transductive CEM", leave=False)

    last_state: dict[str, torch.Tensor] | None = None
    last_knn_idx: torch.Tensor | None = None
    for _outer in iterator:
        W_net.eval()
        with torch.no_grad():
            Z_train_full = W_net.project(X_train).detach()
            Z_test_full = W_net.project(X_test).detach()
            d = torch.cdist(Z_test_full, Z_train_full, p=2.0)
            knn_idx = torch.topk(d, K, dim=1, largest=False).indices
            Z_neighbors = Z_train_full[knn_idx.reshape(-1)].view(T, K, -1)
            y_neighbors_std = y_train_std[knn_idx.reshape(-1)].view(T, K)
        state, _cem_diag = _fit_selfcem_on_feature_neighbors(Z_test_full, Z_neighbors, y_neighbors_std, args=args)
        W_net.train()

        cur_sets = [set(map(int, knn_idx[i].detach().cpu().tolist())) for i in range(T)]
        if prev_neighbor_sets is None:
            neighbor_change = 1.0
        else:
            symdiff = sum(len(new.symmetric_difference(old)) for new, old in zip(cur_sets, prev_neighbor_sets))
            neighbor_change = symdiff / float(max(1, 2 * K * T))
        prev_neighbor_sets = cur_sets
        raw_overlap = float(np.mean([len(cur_sets[i] & raw_knn_sets[i]) / K for i in range(T)]))
        labels = state["labels"].bool()
        if prev_labels is None:
            label_change = 1.0
        else:
            label_change = float((labels != prev_labels).to(dtype=torch.float64).mean().detach().cpu().item())
        prev_labels = labels.detach().clone()

        diag["neighbor_change_history"].append(float(neighbor_change))
        diag["label_change_history"].append(float(label_change))
        diag["raw_overlap_history"].append(float(raw_overlap))
        diag["inlier_rate_history"].append(float(labels.to(dtype=torch.float64).mean().detach().cpu().item()))
        diag["noise_var_history"].append(float(state["noise_var"].detach().mean().cpu().item()))
        diag["z_norm_mean_history"].append(float(torch.linalg.vector_norm(Z_train_full, dim=1).mean().cpu().item()))

        last_loss = float("nan")
        last_gp = float("nan")
        last_gate = float("nan")
        for _inner in range(int(inner_steps)):
            perm = torch.randperm(T, device=device)
            loss_running = 0.0
            gp_running = 0.0
            gate_running = 0.0
            n_batches = 0
            for start in range(0, T, int(anchor_batch)):
                sel = perm[start : min(start + int(anchor_batch), T)]
                loss_base, gp_loss, gate_loss = _fixed_cem_transductive_loss(
                    X_train,
                    X_test,
                    y_train_std,
                    W_net=W_net,
                    test_idx=test_all[sel],
                    neighbor_idx=knn_idx[sel],
                    labels=labels[sel],
                    lengthscale=state["lengthscale"][sel],
                    signal_var=state["signal_var"][sel],
                    noise_var=state["noise_var"][sel],
                    mean_const=state["mean_const"][sel],
                    gate=state["gate"][sel],
                    min_inliers=int(args.min_inliers),
                    gate_weight=float(args.transductive_gate_weight),
                )
                reg = loss_base.new_zeros(())
                if w_l2 > 0 and hasattr(W_net, "W0"):
                    reg = reg + float(w_l2) * (W_net.W0.to(dtype=loss_base.dtype) ** 2).sum()
                loss = loss_base + reg
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(w_params, max_norm=5.0)
                optim.step()
                loss_running += float(loss_base.detach().cpu().item())
                gp_running += float(gp_loss.cpu().item())
                gate_running += float(gate_loss.cpu().item())
                n_batches += 1
            last_loss = loss_running / max(1, n_batches)
            last_gp = gp_running / max(1, n_batches)
            last_gate = gate_running / max(1, n_batches)
        diag["train_loss_history"].append(float(last_loss))
        diag["gp_loss_history"].append(float(last_gp))
        diag["gate_loss_history"].append(float(last_gate))
        last_state = state
        last_knn_idx = knn_idx.detach()

        if verbose:
            iterator.set_postfix(  # type: ignore[attr-defined]
                loss=last_loss,
                gp=last_gp,
                gate=last_gate,
                nb=neighbor_change,
                ov=raw_overlap,
                inr=diag["inlier_rate_history"][-1],
            )

    diag["final_train_loss"] = diag["train_loss_history"][-1] if diag["train_loss_history"] else float("nan")
    diag["final_gp_loss"] = diag["gp_loss_history"][-1] if diag["gp_loss_history"] else float("nan")
    diag["final_gate_loss"] = diag["gate_loss_history"][-1] if diag["gate_loss_history"] else float("nan")
    diag["final_neighbor_change"] = diag["neighbor_change_history"][-1] if diag["neighbor_change_history"] else float("nan")
    diag["final_label_change"] = diag["label_change_history"][-1] if diag["label_change_history"] else float("nan")
    diag["final_raw_overlap"] = diag["raw_overlap_history"][-1] if diag["raw_overlap_history"] else float("nan")
    diag["final_inlier_rate"] = diag["inlier_rate_history"][-1] if diag["inlier_rate_history"] else float("nan")
    diag["final_noise_var"] = diag["noise_var_history"][-1] if diag["noise_var_history"] else float("nan")
    diag["final_z_norm_mean"] = diag["z_norm_mean_history"][-1] if diag["z_norm_mean_history"] else float("nan")
    diag["last_knn_idx"] = None if last_knn_idx is None else last_knn_idx.detach().cpu()
    if last_state is not None:
        diag["final_lengthscale"] = float(last_state["lengthscale"].detach().mean().cpu().item())
        diag["final_signal_var"] = float(last_state["signal_var"].detach().mean().cpu().item())
    return W_net, diag


# ---------------------------------------------------------------------------
# Variant-level entry points.
# ---------------------------------------------------------------------------

def _standardize_X(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    info = {"x_mean_nrm": float(np.linalg.norm(scaler.mean_)), "x_std_nrm": float(np.linalg.norm(scaler.scale_))}
    return Xtr.astype(np.float64), Xte.astype(np.float64), info


def _make_w_net(
    variant: str,
    *,
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    Q: int,
    lowrank_r: int,
    init_mode: str,
    hidden: int,
    device: torch.device,
) -> nn.Module:
    if init_mode == "pls":
        W0 = _pls_W0(X_train_np, y_train_np, Q)
    elif init_mode == "pca":
        W0 = _pca_W0(X_train_np, Q)
    else:
        raise ValueError(f"Unknown init_mode={init_mode}")
    W0_t = torch.as_tensor(W0, dtype=torch.float32, device=device)

    if variant == "static":
        return StaticW(W0_t).to(device=device, dtype=torch.float32)
    if variant in {"lowrank", "lowrank_random", "lowrank_shuffled"}:
        V = _residual_V_basis(X_train_np, W0, lowrank_r)
        V_t = torch.as_tensor(V, dtype=torch.float32, device=device)
        return LowRankW(W0_t, V_t, hidden=hidden, learn_W0=True).to(device=device, dtype=torch.float32)
    raise ValueError(f"Unknown w_variant={variant}")


def _project_dataset(
    W_net: nn.Module,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    shuffle_inputs: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Project train/test to Z-space, optionally with shuffled inputs.

    ``shuffle_inputs`` permutes which ``x`` is fed into ``W_theta`` while keeping
    the multiplication by the unshuffled ``x``: this breaks the spatial mapping
    ``x -> W(x)`` while leaving the dataset of ``x`` unchanged.
    """
    W_net.eval()
    with torch.no_grad():
        if shuffle_inputs:
            n_train = X_train.shape[0]
            perm = torch.as_tensor(rng.permutation(n_train), dtype=torch.long, device=X_train.device)
            # For LowRankW, we want z_i = W0 x_i + C(x_{perm(i)}) (V^T x_i) for train,
            # and z*_i = W0 x*_i + C(x*_{perm(i)}) (V^T x*_i) for test.
            if isinstance(W_net, LowRankW):
                C_train = W_net.net(X_train[perm]).view(-1, W_net.Q, W_net.r)
                z_static_train = X_train @ W_net.W0.t()
                VtX_train = X_train @ W_net.V
                Z_train = z_static_train + torch.einsum("nqr,nr->nq", C_train, VtX_train)

                test_perm = torch.as_tensor(rng.permutation(X_test.shape[0]), dtype=torch.long, device=X_test.device)
                C_test = W_net.net(X_test[test_perm]).view(-1, W_net.Q, W_net.r)
                z_static_test = X_test @ W_net.W0.t()
                VtX_test = X_test @ W_net.V
                Z_test = z_static_test + torch.einsum("nqr,nr->nq", C_test, VtX_test)
            else:
                # StaticW has no input-dependent part; shuffling is a no-op.
                Z_train = W_net.project(X_train)
                Z_test = W_net.project(X_test)
        else:
            Z_train = W_net.project(X_train)
            Z_test = W_net.project(X_test)
    return Z_train.detach().cpu().numpy().astype(np.float64), Z_test.detach().cpu().numpy().astype(np.float64)


def _resolve_anchor_indices(
    n_train: int,
    anchor_count: int | None,
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    *,
    strategy: str,
    seed: int,
) -> np.ndarray:
    if anchor_count is None or anchor_count <= 0 or anchor_count >= n_train:
        return np.arange(n_train, dtype=np.int64)
    if strategy == "uniform":
        rng = np.random.default_rng(seed)
        return rng.choice(n_train, size=anchor_count, replace=False)
    if strategy == "kmeans_yvar":
        # Pick K-medoid-like anchors weighted by local y variance: simple proxy = top |y - mean|.
        # Fallback to kmeans over (X, y) joint, then nearest train indices to centers.
        try:
            from sklearn.cluster import KMeans
        except Exception:  # pragma: no cover
            return _resolve_anchor_indices(n_train, anchor_count, X_train_s, y_train, strategy="uniform", seed=seed)
        X_aug = np.concatenate([X_train_s, y_train.reshape(-1, 1)], axis=1)
        km = KMeans(n_clusters=int(anchor_count), n_init=4, random_state=int(seed))
        km.fit(X_aug)
        idx_out = []
        for c in range(int(anchor_count)):
            dists = np.linalg.norm(X_aug - km.cluster_centers_[c], axis=1)
            idx_out.append(int(np.argmin(dists)))
        return np.unique(np.asarray(idx_out, dtype=np.int64))
    raise ValueError(f"Unknown anchor strategy={strategy}")


def _run_variant(
    variant: str,
    *,
    setting: str,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    Q = int(cfg["Q"])
    n_neighbors = int(args.n_neighbors) if int(args.n_neighbors) > 0 else int(cfg.get("n", 35))
    X_train_raw = data["X_train"]
    X_test_raw = data["X_test"]
    y_train_np = data["y_train"].astype(np.float64).reshape(-1)
    y_test_np = data["y_test"].astype(np.float64).reshape(-1)

    if args.max_test > 0 and args.max_test < X_test_raw.shape[0]:
        X_test_raw = X_test_raw[: int(args.max_test)]
        y_test_np = y_test_np[: int(args.max_test)]

    X_train_s, X_test_s, scale_info = _standardize_X(X_train_raw, X_test_raw)
    n_train = int(X_train_s.shape[0])

    row: dict[str, Any] = {
        "setting": setting,
        "family": cfg.get("family", ""),
        "expansion": cfg.get("expansion", ""),
        "caseno": cfg.get("caseno", ""),
        "seed": int(seed),
        "method": variant,
        "N_train": n_train,
        "N_test": int(X_test_raw.shape[0]),
        "latent_dim": int(cfg.get("d", Q)),
        "observed_dim": int(X_train_s.shape[1]),
        "Q_true": int(cfg.get("Q_true", Q)),
        "Q": int(Q),
        "n": int(n_neighbors),
        "standardize_x": True,
        "noise_std": cfg.get("noise_std", ""),
        "noise_var": cfg.get("noise_var", ""),
        "lengthscale": cfg.get("lengthscale", ""),
        "kernel_var": cfg.get("kernel_var", ""),
        "x_mean_nrm": scale_info["x_mean_nrm"],
        "x_std_nrm": scale_info["x_std_nrm"],
        "status": "ok",
    }

    rng = np.random.default_rng(int(seed))

    try:
        if variant == "raw_jgp":
            metrics, wall, extras = _jumpgp_predict_in_feature_space(
                X_train_s, y_train_np, X_test_s, y_test_np,
                n_neighbors=n_neighbors, device=device, desc="raw_jgp",
            )
            row.update(metrics)
            row.update({"feature_dim": int(X_train_s.shape[1]), "wall_total_sec": float(wall)})
            row.update({f"diag_{k}": v for k, v in extras.items()})
            return row

        if variant == "pca_jgp":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=Q)
            Z_train = pca.fit_transform(X_train_s)
            Z_test = pca.transform(X_test_s)
            metrics, wall, extras = _jumpgp_predict_in_feature_space(
                Z_train, y_train_np, Z_test, y_test_np,
                n_neighbors=n_neighbors, device=device, desc="pca_jgp",
            )
            row.update(metrics)
            row.update({
                "feature_dim": int(Q),
                "wall_total_sec": float(wall),
                "diag_pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
                "diag_z_norm_mean": float(np.linalg.norm(Z_train, axis=1).mean()),
            })
            row.update({f"diag_{k}": v for k, v in extras.items()})
            return row

        if variant == "pls_jgp":
            W_pls = _pls_W0(X_train_s, y_train_np, Q)  # [Q, D]
            Z_train = X_train_s @ W_pls.T
            Z_test = X_test_s @ W_pls.T
            metrics, wall, extras = _jumpgp_predict_in_feature_space(
                Z_train, y_train_np, Z_test, y_test_np,
                n_neighbors=n_neighbors, device=device, desc="pls_jgp",
            )
            row.update(metrics)
            row.update({
                "feature_dim": int(Q),
                "wall_total_sec": float(wall),
                "diag_z_norm_mean": float(np.linalg.norm(Z_train, axis=1).mean()),
            })
            row.update({f"diag_{k}": v for k, v in extras.items()})
            return row

        if variant in {"raw_selfcem", "pca_selfcem", "pls_selfcem", "pd_raw_selfcem", "pd_pca_selfcem", "pd_pls_selfcem"}:
            base_variant = variant.replace("pd_", "", 1)
            if base_variant == "raw_selfcem":
                F_train = X_train_s
                F_test = X_test_s
                feature_dim = int(X_train_s.shape[1])
                extra_static: dict[str, Any] = {}
            elif base_variant == "pca_selfcem":
                from sklearn.decomposition import PCA
                pca = PCA(n_components=Q)
                F_train = pca.fit_transform(X_train_s)
                F_test = pca.transform(X_test_s)
                feature_dim = int(Q)
                extra_static = {"diag_pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_))}
            else:
                W_pls = _pls_W0(X_train_s, y_train_np, Q)
                F_train = X_train_s @ W_pls.T
                F_test = X_test_s @ W_pls.T
                feature_dim = int(Q)
                extra_static = {}
            metrics, wall, extras = _selfcem_predict_in_feature_space(
                F_train,
                y_train_np,
                F_test,
                y_test_np,
                n_neighbors=n_neighbors,
                args=args,
                device=device,
                desc=variant,
            )
            row.update(metrics)
            row.update({
                "feature_dim": int(feature_dim),
                "head": "selfcem",
                "wall_total_sec": float(wall),
                "diag_z_norm_mean": float(np.linalg.norm(F_train, axis=1).mean()),
                "diag_z_norm_std": float(np.linalg.norm(F_train, axis=1).std()),
            })
            row.update(extra_static)
            row.update({f"diag_{k}": v for k, v in extras.items()})
            return row

        if variant in {
            "pd_static_transductive",
            "pd_lowrank_transductive",
            "pd_lowrank_transductive_shuffled",
            "pd_lowrank_random",
        }:
            w_arch = "static" if variant == "pd_static_transductive" else "lowrank"
            W_net = _make_w_net(
                w_arch,
                X_train_np=X_train_s,
                y_train_np=y_train_np,
                Q=Q,
                lowrank_r=int(args.lowrank_r),
                init_mode=args.init_mode,
                hidden=int(args.hidden),
                device=device,
            )
            X_train_t = torch.as_tensor(X_train_s, dtype=torch.float32, device=device)
            X_test_t = torch.as_tensor(X_test_s, dtype=torch.float32, device=device)
            y_mean = float(y_train_np.mean())
            y_std = float(y_train_np.std())
            if not math.isfinite(y_std) or y_std <= 1e-12:
                y_std = 1.0
            y_train_std_t = torch.as_tensor((y_train_np - y_mean) / y_std, dtype=torch.float32, device=device)

            train_wall_start = time.perf_counter()
            if variant == "pd_lowrank_random":
                diag_train = {}
            else:
                _, diag_train = _train_transductive_selfcem_W(
                    X_train_t,
                    X_test_t,
                    y_train_std_t,
                    W_net=W_net,
                    K_neighbor=int(n_neighbors),
                    epochs=int(args.epochs),
                    inner_steps=int(args.inner_steps),
                    lr_w=float(args.lr_w),
                    weight_decay=float(args.weight_decay),
                    w_l2=float(args.w_l2),
                    anchor_batch=int(args.anchor_batch),
                    args=args,
                    device=device,
                    verbose=bool(args.verbose),
                )
            train_wall = time.perf_counter() - train_wall_start

            shuffle_inputs = variant == "pd_lowrank_transductive_shuffled"
            Z_train, Z_test = _project_dataset(
                W_net,
                X_train_t,
                X_test_t,
                shuffle_inputs=shuffle_inputs,
                rng=rng,
            )
            metrics, predict_wall, extras = _selfcem_predict_in_feature_space(
                Z_train,
                y_train_np,
                Z_test,
                y_test_np,
                n_neighbors=n_neighbors,
                args=args,
                device=device,
                desc=variant,
            )
            row.update(metrics)
            row.update({
                "feature_dim": int(Q),
                "head": "projected_dataset_transductive_selfcem",
                "wall_total_sec": float(train_wall + predict_wall),
                "wall_train_sec": float(train_wall),
                "wall_predict_sec": float(predict_wall),
                "diag_anchor_count": int(X_test_s.shape[0]),
                "diag_z_norm_mean": float(np.linalg.norm(Z_train, axis=1).mean()),
                "diag_z_norm_std": float(np.linalg.norm(Z_train, axis=1).std()),
            })
            for k in (
                "final_train_loss",
                "final_gp_loss",
                "final_gate_loss",
                "final_neighbor_change",
                "final_label_change",
                "final_raw_overlap",
                "final_inlier_rate",
                "final_noise_var",
                "final_z_norm_mean",
                "final_lengthscale",
                "final_signal_var",
            ):
                if k in diag_train:
                    row[f"diag_{k}"] = float(diag_train[k])
            row.update({f"diag_{k}": v for k, v in extras.items()})
            return row

        if variant in {
            "static_selfcem_supervised",
            "lowrank_selfcem_supervised",
            "lowrank_selfcem_shuffled",
            "lowrank_selfcem_random",
        }:
            w_arch = "static" if variant == "static_selfcem_supervised" else "lowrank"
            W_net = _make_w_net(
                w_arch,
                X_train_np=X_train_s,
                y_train_np=y_train_np,
                Q=Q,
                lowrank_r=int(args.lowrank_r),
                init_mode=args.init_mode,
                hidden=int(args.hidden),
                device=device,
            )
            anchor_idx_np = _resolve_anchor_indices(
                n_train,
                anchor_count=int(args.anchor_count) if int(args.anchor_count) > 0 else None,
                X_train_s=X_train_s,
                y_train=y_train_np,
                strategy=args.anchor_strategy,
                seed=int(seed),
            )
            anchor_idx = torch.as_tensor(anchor_idx_np, dtype=torch.long, device=device)
            X_train_t = torch.as_tensor(X_train_s, dtype=torch.float32, device=device)
            X_test_t = torch.as_tensor(X_test_s, dtype=torch.float32, device=device)
            y_mean = float(y_train_np.mean())
            y_std = float(y_train_np.std())
            if not math.isfinite(y_std) or y_std <= 1e-12:
                y_std = 1.0
            y_train_std_t = torch.as_tensor((y_train_np - y_mean) / y_std, dtype=torch.float32, device=device)

            train_wall_start = time.perf_counter()
            if variant == "lowrank_selfcem_random":
                diag_train = {}
            else:
                _, diag_train = _train_selfcem_supervised_W(
                    X_train_t,
                    y_train_std_t,
                    W_net=W_net,
                    K_neighbor=int(n_neighbors),
                    anchor_idx=anchor_idx,
                    epochs=int(args.epochs),
                    inner_steps=int(args.inner_steps),
                    lr_w=float(args.lr_w),
                    weight_decay=float(args.weight_decay),
                    w_l2=float(args.w_l2),
                    anchor_batch=int(args.anchor_batch),
                    args=args,
                    device=device,
                    verbose=bool(args.verbose),
                )
            train_wall = time.perf_counter() - train_wall_start

            shuffle_inputs = variant == "lowrank_selfcem_shuffled"
            Z_train, Z_test = _project_dataset(
                W_net,
                X_train_t,
                X_test_t,
                shuffle_inputs=shuffle_inputs,
                rng=rng,
            )
            metrics, predict_wall, extras = _selfcem_predict_in_feature_space(
                Z_train,
                y_train_np,
                Z_test,
                y_test_np,
                n_neighbors=n_neighbors,
                args=args,
                device=device,
                desc=variant,
            )
            row.update(metrics)
            row.update({
                "feature_dim": int(Q),
                "head": "selfcem",
                "wall_total_sec": float(train_wall + predict_wall),
                "wall_train_sec": float(train_wall),
                "wall_predict_sec": float(predict_wall),
                "diag_anchor_count": int(anchor_idx.shape[0]),
                "diag_z_norm_mean": float(np.linalg.norm(Z_train, axis=1).mean()),
                "diag_z_norm_std": float(np.linalg.norm(Z_train, axis=1).std()),
            })
            for k in (
                "final_train_nll",
                "final_neighbor_change",
                "final_label_change",
                "final_raw_overlap",
                "final_inlier_rate",
                "final_noise_var",
                "final_z_norm_mean",
                "final_lengthscale",
                "final_signal_var",
            ):
                if k in diag_train:
                    row[f"diag_{k}"] = float(diag_train[k])
            row.update({f"diag_{k}": v for k, v in extras.items()})
            return row

        if variant in {"static_supervised", "lowrank_supervised", "lowrank_shuffled", "lowrank_random"}:
            w_arch = "static" if variant == "static_supervised" else "lowrank"
            W_net = _make_w_net(
                w_arch,
                X_train_np=X_train_s,
                y_train_np=y_train_np,
                Q=Q,
                lowrank_r=int(args.lowrank_r),
                init_mode=args.init_mode,
                hidden=int(args.hidden),
                device=device,
            )
            anchor_idx_np = _resolve_anchor_indices(
                n_train,
                anchor_count=int(args.anchor_count) if int(args.anchor_count) > 0 else None,
                X_train_s=X_train_s,
                y_train=y_train_np,
                strategy=args.anchor_strategy,
                seed=int(seed),
            )
            anchor_idx = torch.as_tensor(anchor_idx_np, dtype=torch.long, device=device)
            X_train_t = torch.as_tensor(X_train_s, dtype=torch.float32, device=device)
            X_test_t = torch.as_tensor(X_test_s, dtype=torch.float32, device=device)
            y_train_t = torch.as_tensor(y_train_np, dtype=torch.float32, device=device)

            train_wall_start = time.perf_counter()
            if variant == "lowrank_random":
                # Skip training; use the freshly-initialised network.
                diag_train: dict[str, Any] = {}
                surrogate = LocalRBFGPSurrogate(
                    init_ell=float(args.init_ell), init_sf=float(args.init_sf), init_sn=float(args.init_sn),
                ).to(device)
            else:
                _, surrogate, diag_train = _train_supervised_W(
                    X_train_t, y_train_t,
                    W_net=W_net,
                    K_neighbor=int(n_neighbors),
                    anchor_idx=anchor_idx,
                    epochs=int(args.epochs),
                    outer_refresh=int(args.outer_refresh),
                    inner_steps=int(args.inner_steps),
                    lr_w=float(args.lr_w),
                    lr_gp=float(args.lr_gp),
                    weight_decay=float(args.weight_decay),
                    w_l2=float(args.w_l2),
                    anchor_batch=int(args.anchor_batch),
                    init_ell=float(args.init_ell),
                    init_sf=float(args.init_sf),
                    init_sn=float(args.init_sn),
                    device=device,
                    verbose=bool(args.verbose),
                )
            train_wall = time.perf_counter() - train_wall_start

            shuffle_inputs = variant == "lowrank_shuffled"
            Z_train, Z_test = _project_dataset(
                W_net, X_train_t, X_test_t,
                shuffle_inputs=shuffle_inputs, rng=rng,
            )

            metrics, predict_wall, extras = _jumpgp_predict_in_feature_space(
                Z_train, y_train_np, Z_test, y_test_np,
                n_neighbors=n_neighbors, device=device, desc=variant,
            )
            row.update(metrics)
            row.update({
                "feature_dim": int(Q),
                "wall_total_sec": float(train_wall + predict_wall),
                "wall_train_sec": float(train_wall),
                "wall_predict_sec": float(predict_wall),
                "diag_z_norm_mean": float(np.linalg.norm(Z_train, axis=1).mean()),
                "diag_z_norm_std": float(np.linalg.norm(Z_train, axis=1).std()),
            })
            for k in ("final_ell", "final_sf", "final_sn", "final_train_nll",
                      "final_label_change", "final_z_norm_mean", "final_raw_overlap"):
                if k in diag_train:
                    row[f"diag_{k}"] = float(diag_train[k]) if isinstance(diag_train[k], (int, float)) else diag_train[k]
            row.update({f"diag_{k}": v for k, v in extras.items()})
            return row

        raise ValueError(f"Unknown variant={variant}")
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


# ---------------------------------------------------------------------------
# CLI driver.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global supervised z=W(x)x + JGP runner.")
    p.add_argument("--settings", default="lh_q5_train1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="", help="Optional comma-separated seed list; overrides --seed.")
    p.add_argument("--variants",
                   default="raw_jgp,pca_jgp,pls_jgp,static_supervised,lowrank_supervised,lowrank_shuffled,lowrank_random")
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--n_neighbors", type=int, default=0,
                   help="Override per-setting n; 0 keeps SETTING_PRESETS['n'].")

    p.add_argument("--init_mode", choices=["pls", "pca"], default="pls")
    p.add_argument("--lowrank_r", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)

    p.add_argument("--anchor_count", type=int, default=0,
                   help="Number of training anchors; 0 = all training points.")
    p.add_argument("--anchor_strategy", choices=["uniform", "kmeans_yvar"], default="uniform")
    p.add_argument("--anchor_batch", type=int, default=128)

    p.add_argument("--epochs", type=int, default=200,
                   help="Number of outer cycles (Z-KNN refreshes).")
    p.add_argument("--outer_refresh", type=int, default=10,
                   help="Deprecated alias kept for header docs; ignored if --epochs is set.")
    p.add_argument("--inner_steps", type=int, default=50,
                   help="Gradient steps per Z-KNN refresh.")

    p.add_argument("--lr_w", type=float, default=1e-3)
    p.add_argument("--lr_gp", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--w_l2", type=float, default=1e-4)

    p.add_argument("--init_ell", type=float, default=1.0)
    p.add_argument("--init_sf", type=float, default=1.0)
    p.add_argument("--init_sn", type=float, default=0.1)

    p.add_argument("--self_cem_updates", type=int, default=2)
    p.add_argument("--self_cem_hyper_steps", type=int, default=10)
    p.add_argument("--self_cem_hyper_lr", type=float, default=0.04)
    p.add_argument("--self_cem_gate_steps", type=int, default=20)
    p.add_argument("--self_cem_gate_lr", type=float, default=0.05)
    p.add_argument("--self_cem_gate_l2", type=float, default=1e-3)
    p.add_argument("--self_cem_gate_max_norm", type=float, default=8.0)
    p.add_argument("--self_cem_gh_points", type=int, default=20)
    p.add_argument("--self_cem_noise_std_floor", type=float, default=0.1)
    p.add_argument("--self_cem_kcov", type=float, default=0.1)
    p.add_argument("--min_inliers", type=int, default=2)
    p.add_argument("--transductive_gate_weight", type=float, default=0.1)

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out_dir", default="experiments/synthetic/global_z_supervised_jgp_lh_seed0_20260516")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "global_z_supervised_jgp_metrics.csv"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rows: list[dict[str, Any]] = []
    if args.resume and metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {(str(r.get("setting")), int(r.get("seed", -1)), str(r.get("method")))
            for r in rows if str(r.get("status", "ok")) == "ok"}

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [int(args.seed)]

    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown_settings:
        raise ValueError(f"Unknown settings {unknown_settings}; choose from {sorted(SETTING_PRESETS)}")

    for setting in settings:
        cfg = dict(SETTING_PRESETS[setting])
        for seed in seeds:
            _set_seed(int(seed))
            data = _generate_paper_data(cfg, int(seed), device)
            for variant in variants:
                key = (setting, int(seed), variant)
                if key in done:
                    print(f"[skip] {setting} seed={seed} variant={variant}", flush=True)
                    continue
                print(f"[run]  {setting} seed={seed} variant={variant}", flush=True)
                t0 = time.perf_counter()
                row = _run_variant(variant, setting=setting, cfg=cfg, args=args,
                                   data=data, device=device, seed=int(seed))
                row["elapsed_sec"] = float(time.perf_counter() - t0)
                rows.append(row)
                _write_csv(metrics_path, rows)
                print(
                    f"      -> status={row.get('status')} rmse={row.get('rmse')} crps={row.get('mean_crps')}"
                    f" cov90={row.get('coverage_90')} width90={row.get('mean_width_90')}",
                    flush=True,
                )

    summary = {
        "args": vars(args),
        "device": str(device),
        "n_rows": len(rows),
        "settings": settings,
        "seeds": seeds,
        "variants": variants,
    }
    (out_dir / "global_z_supervised_jgp_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(metrics_path, rows)

    readme = f"""# Global Supervised z=W(x)x + JGP

Runner: ``experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py``.
Plan:   ``docs/global_z_supervised_jgp_plan_cn.md``.

This experiment implements the global-Z supervised projection variant: instead
of hybrid raw+W reranking inside a candidate pool, the whole train/test dataset
is mapped through ``z_i = W_theta(x_i) x_i`` and JumpGP runs in the global
``Z``-space.  ``W_theta`` is trained with a leave-anchor-out supervised anchor
predictive NLL.  Variants ending in ``_selfcem`` or containing
``_selfcem_`` use the current self-CEM head in the alternating loop: after each
global-Z KNN refresh, they refit self-CEM labels/hyperparameters in ``Z`` space
and use those fixed neighborhoods/partitions for the next supervised W M-step.
The older non-self-CEM variants use a differentiable local RBF GP surrogate and
plain JumpGP CEM only at inference.

```json
{json.dumps(summary, indent=2)}
```

Files:

- ``global_z_supervised_jgp_metrics.csv`` — per-variant metrics.
- ``global_z_supervised_jgp_summary.json`` — run configuration.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

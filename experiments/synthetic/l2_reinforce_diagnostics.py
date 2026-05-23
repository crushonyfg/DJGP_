"""Diagnostics for supervised REINFORCE reward traces and W drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    c = np.corrcoef(ra.astype(np.float64), rb.astype(np.float64))[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return float("nan")
    union = a | b
    if not union:
        return float("nan")
    return float(len(a & b) / len(union))


def w_drift_frobenius_mean(W0: torch.Tensor, W_mu_anchor: torch.Tensor) -> float:
    """Mean Frobenius norm ``||W_mu_j - W0||_F`` over anchors."""
    w0 = W0.detach().cpu().to(dtype=torch.float64)
    wmu = W_mu_anchor.detach().cpu().to(dtype=torch.float64)
    if w0.dim() == 2:
        diff = wmu - w0.unsqueeze(0)
    else:
        diff = wmu - w0
    return float(diff.pow(2).sum(dim=(-2, -1)).sqrt().mean().item())


def summarize_reward_trace(
    trace: list[dict[str, Any]],
    *,
    top_frac: float = 0.3,
    spread_eps: float = 1e-8,
) -> dict[str, float]:
    """Aggregate spread / stability / ΔR-vs-W0 from ``sample_reward_trace`` rows."""
    if not trace:
        return {}

    spreads_by_step: list[np.ndarray] = []
    top_sets: list[set[int]] = []
    spread_hist_all: list[float] = []
    delta_all: list[float] = []
    delta_top_all: list[float] = []
    delta_best_all: list[float] = []

    for row in trace:
        if "anchor_spread_std" in row:
            spread = np.asarray(row["anchor_spread_std"], dtype=np.float64)
        else:
            rpa = np.asarray(row["reward_per_sample_anchor"], dtype=np.float64)
            spread = rpa.std(axis=0, unbiased=False)
        spreads_by_step.append(spread)
        spread_hist_all.extend(spread.tolist())

        t_count = int(spread.shape[0])
        k = max(1, int(round(float(top_frac) * t_count)))
        top_idx = set(np.argsort(spread)[-k:].astype(int).tolist())
        top_sets.append(top_idx)

        if "delta_reward_per_sample_anchor" in row:
            delta = np.asarray(row["delta_reward_per_sample_anchor"], dtype=np.float64)
            delta_all.extend(delta.reshape(-1).tolist())
            delta_best = delta.max(axis=0)
            delta_best_all.extend(delta_best.tolist())
            if top_idx:
                delta_top_all.extend(delta[:, sorted(top_idx)].reshape(-1).tolist())

    spread_arr = np.asarray(spread_hist_all, dtype=np.float64)
    out: dict[str, float] = {
        "reward_spread_mean": float(np.nanmean(spread_arr)),
        "reward_spread_median": float(np.nanmedian(spread_arr)),
        "reward_spread_p90": float(np.nanpercentile(spread_arr, 90)) if spread_arr.size else float("nan"),
        "reward_spread_frac_near_zero": float(np.mean(spread_arr < spread_eps)) if spread_arr.size else float("nan"),
    }

    if len(top_sets) >= 2:
        overlaps = [_jaccard(top_sets[i - 1], top_sets[i]) for i in range(1, len(top_sets))]
        out["top_anchor_overlap_mean"] = float(np.nanmean(overlaps))
        out["top_anchor_overlap_last"] = float(overlaps[-1])
    else:
        out["top_anchor_overlap_mean"] = float("nan")
        out["top_anchor_overlap_last"] = float("nan")

    if len(spreads_by_step) >= 2:
        rank_corrs = [
            _spearman_corr(spreads_by_step[i - 1], spreads_by_step[i])
            for i in range(1, len(spreads_by_step))
        ]
        out["rank_stability_spread_mean"] = float(np.nanmean(rank_corrs))
        out["rank_stability_spread_last"] = float(rank_corrs[-1])
    else:
        out["rank_stability_spread_mean"] = float("nan")
        out["rank_stability_spread_last"] = float("nan")

    if delta_all:
        d = np.asarray(delta_all, dtype=np.float64)
        out["mean_delta_reward_vs_w0"] = float(np.nanmean(d))
        out["frac_delta_reward_positive"] = float(np.mean(d > 0.0))
    if delta_best_all:
        db = np.asarray(delta_best_all, dtype=np.float64)
        out["mean_best_delta_reward_vs_w0"] = float(np.nanmean(db))
        out["frac_best_delta_positive"] = float(np.mean(db > 0.0))
    if delta_top_all:
        dt = np.asarray(delta_top_all, dtype=np.float64)
        out["mean_delta_reward_top_anchors"] = float(np.nanmean(dt))
        out["frac_delta_top_positive"] = float(np.mean(dt > 0.0))

    last = trace[-1]
    if "delta_reward_per_sample_anchor" in last:
        dlast = np.asarray(last["delta_reward_per_sample_anchor"], dtype=np.float64)
        spread_last = spreads_by_step[-1]
        k = max(1, int(round(float(top_frac) * spread_last.shape[0])))
        top_last = set(np.argsort(spread_last)[-k:].astype(int).tolist())
        if top_last:
            cols = sorted(top_last)
            out["mean_delta_reward_top_anchors_last"] = float(dlast[:, cols].mean())
            out["mean_best_delta_top_anchors_last"] = float(dlast.max(axis=0)[cols].mean())

    return out


def write_spread_histogram(trace: list[dict[str, Any]], path: Path, *, bins: int = 20) -> None:
    """Save per-step anchor spread histogram (pooled over steps)."""
    vals: list[float] = []
    for row in trace:
        if "anchor_spread_std" in row:
            vals.extend(float(x) for x in row["anchor_spread_std"])
        else:
            rpa = np.asarray(row["reward_per_sample_anchor"], dtype=np.float64)
            vals.extend(rpa.std(axis=0, unbiased=False).tolist())
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return
    counts, edges = np.histogram(arr, bins=int(bins))
    payload = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "frac_near_zero": float(np.mean(arr < 1e-8)),
        "bin_edges": edges.tolist(),
        "bin_counts": counts.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

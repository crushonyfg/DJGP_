"""Shared REINFORCE advantage / anchor-weight presets for L2 supervised training.

Important: do **not** pair ``center_std`` advantage with ``spread_std`` weights — they partially
cancel (w_j * A_{s,j} ~ R_{s,j} - mean_s R_{s,j}). Use ``center`` + spread weighting instead.
"""

from __future__ import annotations

from typing import Any

ADVANTAGE_VARIANTS: dict[str, dict[str, Any]] = {
    # --- core comparisons (user priority) ---
    "center": {
        "reward_advantage": "center",
        "anchor_weight": "uniform",
        "standardize_rewards": False,
    },
    "rank": {
        "reward_advantage": "rank",
        "anchor_weight": "uniform",
        "standardize_rewards": False,
    },
    "hard_center_spread": {
        "reward_advantage": "center",
        "anchor_weight": "spread_std",
        "standardize_rewards": False,
    },
    "hard_rank_spread": {
        "reward_advantage": "rank",
        "anchor_weight": "spread_std",
        "standardize_rewards": False,
    },
    "hard_top30_rank": {
        "reward_advantage": "rank",
        "anchor_weight": "top_fraction",
        "anchor_weight_top_frac": 0.3,
        "standardize_rewards": False,
    },
    # --- baselines / optional ---
    "center_std": {
        "reward_advantage": "center_std",
        "anchor_weight": "uniform",
        "standardize_rewards": True,
    },
    "hard_curriculum": {
        "reward_advantage": "center",
        "anchor_weight": "spread_std",
        "anchor_weight_curriculum_steps": 20,
        "standardize_rewards": False,
    },
    # deprecated aliases (old smoke names)
    "hard_spread_std": {
        "reward_advantage": "center",
        "anchor_weight": "spread_std",
        "standardize_rewards": False,
    },
    "hard_top30": {
        "reward_advantage": "rank",
        "anchor_weight": "top_fraction",
        "anchor_weight_top_frac": 0.3,
        "standardize_rewards": False,
    },
}

# Default 25+8 comparison set (excludes deprecated aliases).
DEFAULT_COMPARE_VARIANTS: tuple[str, ...] = (
    "center",
    "rank",
    "center_std",
    "hard_center_spread",
    "hard_rank_spread",
    "hard_top30_rank",
    "hard_curriculum",
)

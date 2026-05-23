"""Plot W0 projection + gate boundary for every test anchor; summarize L2 jump scale.

```powershell
cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP
python experiments/synthetic/plot_l2_w0_all_test.py --seed 0 --standardize_y
python experiments/synthetic/plot_l2_w0_all_test.py --seed 0 --no-standardize_y --out_dir experiments/synthetic/l2_w0_all_test_rawy
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.synthetic.compare_paper_synthetic_baselines import SETTING_PRESETS  # noqa: E402
from experiments.synthetic.compare_uncertain_w_cem_l2_lh import (  # noqa: E402
    _generate_paper_data,
    _neighbor_stack,
    _projection_W0,
    _run_meanw_cem_init,
    _set_seed,
)
from experiments.synthetic.l2_reinforce_viz import (  # noqa: E402
    fit_gate_for_projection,
    plot_l2_projection_panel,
    regime_labels_from_phys,
)


def _local_jump_stats(y_nb: np.ndarray, labels_phys: np.ndarray) -> dict[str, float]:
    """Contrast of y and W0-z spread between physical-regime halves in one neighborhood."""
    y_nb = np.asarray(y_nb, dtype=np.float64).reshape(-1)
    lab = np.asarray(labels_phys, dtype=np.int32).reshape(-1)
    out: dict[str, float] = {}
    for r in (0, 1):
        m = lab == r
        out[f"n_regime{r}"] = float(m.sum())
        if m.any():
            out[f"y_mean_regime{r}"] = float(y_nb[m].mean())
            out[f"y_std_regime{r}"] = float(y_nb[m].std() if m.sum() > 1 else 0.0)
    if out.get("n_regime0", 0) > 0 and out.get("n_regime1", 0) > 0:
        out["y_jump_abs"] = abs(out["y_mean_regime1"] - out["y_mean_regime0"])
    else:
        out["y_jump_abs"] = float("nan")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--setting", default="l2_q2_train500")
    p.add_argument("--noise_std", type=float, default=None, help="Override L2 phantom noise_std (preset default 0.5).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=None, help="Default: all test in preset (200).")
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--projection_init", default="pls")
    p.add_argument("--standardize_y", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gate_steps", type=int, default=6)
    p.add_argument("--gate_lr", type=float, default=0.05)
    p.add_argument("--out_dir", default="experiments/synthetic/l2_w0_all_test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "panels"
    plot_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(SETTING_PRESETS[args.setting])
    if args.noise_std is not None:
        ns = float(args.noise_std)
        cfg["noise_std"] = ns
        cfg["noise_var"] = ns * ns
    _set_seed(args.seed)
    print(f"phantom noise_std={cfg.get('noise_std')}", flush=True)

    data = _generate_paper_data(cfg, args.seed, device)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data["X_train"]).astype(np.float64)
    X_test = scaler.transform(data["X_test"]).astype(np.float64)
    if args.max_test is not None:
        X_test = X_test[: int(args.max_test)]
    y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
    y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)[: X_test.shape[0]]
    z_train = np.asarray(data["z_train"], dtype=np.float64)
    z_test = np.asarray(data["z_test"], dtype=np.float64)[: X_test.shape[0]]

    if bool(args.standardize_y):
        y_mean = float(y_train.mean())
        y_std = float(y_train.std() if y_train.std() > 1e-8 else 1.0)
        y_train_head = ((y_train - y_mean) / y_std).astype(np.float64)
    else:
        y_mean, y_std = 0.0, 1.0
        y_train_head = y_train.astype(np.float64)

    Q = int(cfg["Q"])
    X_train_t = torch.as_tensor(X_train, device=device, dtype=torch.float64)
    X_test_t = torch.as_tensor(X_test, device=device, dtype=torch.float64)
    y_train_head_t = torch.as_tensor(y_train_head, device=device, dtype=torch.float64)

    W0 = torch.as_tensor(
        _projection_W0(X_train, y_train_head, Q, str(args.projection_init), seed=args.seed + 513),
        device=device,
        dtype=torch.float64,
    )
    X_neighbors, y_neighbors_head, neighbor_idx = _neighbor_stack(
        X_test_t, X_train_t, y_train_head_t, n_neighbors=int(args.n_neighbors),
    )
    print(f"Running JumpGP init CEM for {X_test_t.shape[0]} test anchors...", flush=True)
    cem_test, cem_diag = _run_meanw_cem_init(X_test_t, X_neighbors, y_neighbors_head, W0)

    # Global phantom jump reference (caseno 6: +5 / -5)
    global_jump = 10.0
    noise_std_cfg = float(cfg.get("noise_std", 2.0))

    rows: list[dict] = []
    T = int(X_test_t.shape[0])
    for j in range(T):
        lab_j = cem_test["labels"][j] if cem_test["labels"].dim() > 1 else cem_test["labels"]
        gate_j = cem_test["gate"][j] if cem_test["gate"].dim() > 1 else cem_test["gate"]
        nb_idx = neighbor_idx[j].detach().cpu().numpy().astype(np.int64)
        y_nb = y_neighbors_head[j].detach().cpu().numpy()
        labels_phys = regime_labels_from_phys(z_train[nb_idx])
        jst = _local_jump_stats(y_nb, labels_phys)
        inlier_frac = float(lab_j.to(dtype=torch.float64).mean().cpu().item())

        with torch.enable_grad():
            gate_plot = fit_gate_for_projection(
                X_test_t[j],
                X_neighbors[j],
                lab_j,
                W0,
                gate_init=gate_j,
                gate_steps=int(args.gate_steps),
                gate_lr=float(args.gate_lr),
            ).detach()

        plot_l2_projection_panel(
            out_path=plot_dir / f"j{j:04d}_W0.png",
            X_anchor=X_test_t[j],
            X_neighbors=X_neighbors[j],
            y_neighbors=y_neighbors_head[j],
            W_plot=W0,
            W0=W0,
            gate=gate_plot,
            labels=labels_phys,
            title=f"W0 test j={j}  inlier={inlier_frac:.2f}",
            z_phys_anchor=z_test[j],
            z_phys_neighbors=z_train[nb_idx],
        )

        rows.append({
            "j": j,
            "inlier_frac": inlier_frac,
            "gate_w0": float(gate_j[0].cpu()),
            "gate_w1": float(gate_j[1].cpu()),
            "gate_w2": float(gate_j[2].cpu()),
            "z_anchor_x": float(z_test[j, 0]),
            "z_anchor_y": float(z_test[j, 1]),
            "standardize_y": bool(args.standardize_y),
            **jst,
        })
        if (j + 1) % 20 == 0 or j + 1 == T:
            print(f"  plotted {j + 1}/{T}", flush=True)

    y_jumps = np.asarray([r["y_jump_abs"] for r in rows], dtype=np.float64)
    y_jumps_f = y_jumps[np.isfinite(y_jumps)]
    summary = {
        "setting": args.setting,
        "seed": args.seed,
        "n_test": T,
        "n_neighbors": int(args.n_neighbors),
        "standardize_y": bool(args.standardize_y),
        "global_phantom_jump": global_jump,
        "config_noise_std": noise_std_cfg,
        "y_train_min": float(y_train.min()),
        "y_train_max": float(y_train.max()),
        "z_train_range": [float(z_train.min()), float(z_train.max())],
        "meanw_cem_failed": int(cem_diag.get("meanw_cem_failed", 0)),
        "meanw_cem_inlier_rate": float(cem_diag.get("meanw_cem_inlier_rate", float("nan"))),
        "local_y_jump_mean": float(np.mean(y_jumps_f)) if y_jumps_f.size else float("nan"),
        "local_y_jump_median": float(np.median(y_jumps_f)) if y_jumps_f.size else float("nan"),
        "local_y_jump_min": float(np.min(y_jumps_f)) if y_jumps_f.size else float("nan"),
        "local_y_jump_max": float(np.max(y_jumps_f)) if y_jumps_f.size else float("nan"),
        "frac_local_jump_below_3": float((y_jumps_f < 3.0).mean()) if y_jumps_f.size else float("nan"),
        "frac_local_jump_below_5": float((y_jumps_f < 5.0).mean()) if y_jumps_f.size else float("nan"),
        "mean_inlier_frac": float(np.mean([r["inlier_frac"] for r in rows])),
    }
    with (out_dir / "per_anchor_stats.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (out_dir / "jump_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Panels: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()

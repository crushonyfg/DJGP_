r"""Compare baselines on a processed PDEBench-Burgers surrogate dataset.

How to run from the repository root:

    conda activate jumpGP
    cd C:\Users\yxu59\files\spring2026\park\DJGP

Default one-seed run:

    python experiments\pdebench\compare_pdebench_burgers.py ^
      --dataset-folder data\processed\pdebench_burgers_shockjump ^
      --out results\pdebench_burgers_compare.pkl ^
      --out_json results\pdebench_burgers_compare.json

Balanced shock-threshold run continuing from seed 1:

    python experiments\pdebench\compare_pdebench_burgers.py ^
      --dataset-folder data\processed\pdebench_burgers_shockjump_balanced_delta01_n2000 ^
      --seed_start 1 --num_exp 4 --Q 3 --neighbors 25 --lmjgp_steps 300 ^
      --include_xgb 1 --include_dkl 1 --include_lmjgp 1 --include_cemem 0 ^
      --out results\pdebench_burgers_balanced_delta01_n2000_xgb_dkl_lmjgp_q3_n25_seeds1to4.pkl ^
      --out_json results\pdebench_burgers_balanced_delta01_n2000_xgb_dkl_lmjgp_q3_n25_seeds1to4.json

JGP-PCA only, matching the balanced LMJGP Q/n setting:

    python experiments\pdebench\compare_pdebench_burgers.py ^
      --dataset-folder data\processed\pdebench_burgers_shockjump_balanced_delta01_n2000 ^
      --num_exp 5 --Q 3 --neighbors 25 ^
      --include_xgb 0 --include_dkl 0 --include_lmjgp 0 --include_cemem 0 --include_jgp_pca 1 ^
      --out results\pdebench_burgers_balanced_delta01_n2000_jgp_pca_q3_n25_seeds0to4.pkl ^
      --out_json results\pdebench_burgers_balanced_delta01_n2000_jgp_pca_q3_n25_seeds0to4.json
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
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from shared.jumpgp_runner import compute_metrics, find_neighborhoods  # noqa: E402
from djgp.baselines import run_dkl_svgp_regression, run_xgboost_bootstrap_regression  # noqa: E402
from djgp.evaluation import mean_gaussian_crps_torch, pack_gaussian_uq_list  # noqa: E402
from djgp.projections import (  # noqa: E402
    CemEmConfig,
    CoeffProjection,
    build_V_basis,
    compute_pls_W0,
    per_anchor_W_diagnostics,
    run_projected_jumpgp,
    train_projection_cem_em,
)
from djgp.variational import predict_vi, train_vi  # noqa: E402
from jumpgp import JumpGP  # noqa: E402


RESULT_SPLITS = ("all", "near_threshold", "far_from_threshold")


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_dataset(folder: str | Path) -> dict[str, Any]:
    path = Path(folder) / "dataset.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset.pkl at {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _dataset_arrays(dataset: dict[str, Any], max_anchors: int | None = None):
    X_train = _as_numpy(dataset["X_train"]).astype(np.float32)
    y_train = _as_numpy(dataset["Y_train"]).astype(np.float32).reshape(-1)
    X_test = _as_numpy(dataset["X_test"]).astype(np.float32)
    y_test = _as_numpy(dataset["Y_test"]).astype(np.float32).reshape(-1)
    if max_anchors is not None:
        k = min(max_anchors, len(X_test))
        X_test = X_test[:k]
        y_test = y_test[:k]
    return X_train, y_train, X_test, y_test


def _threshold_masks(dataset: dict[str, Any], n_test_full: int, n_used: int):
    meta = dataset.get("args", {}) or {}
    threshold = float(meta.get("jump_threshold", 0.5))
    distance = meta.get("threshold_distance")
    shock = meta.get("shock_location")

    values = None
    if distance is not None:
        values = np.asarray(distance, dtype=np.float64).reshape(-1)
    elif shock is not None:
        values = np.abs(np.asarray(shock, dtype=np.float64).reshape(-1) - threshold)
    else:
        return {"all": np.ones(n_used, dtype=bool)}

    test_indices = meta.get("test_indices")
    if test_indices is not None and len(values) >= max(test_indices) + 1:
        values = values[np.asarray(test_indices, dtype=int)]
    elif len(values) != n_test_full:
        return {"all": np.ones(n_used, dtype=bool)}

    values = values[:n_used]
    delta = meta.get("near_threshold_delta")
    if delta is None:
        delta = float(np.quantile(values, 0.25)) if len(values) else 0.0
        delta = max(delta, 1e-8)
    near = values <= float(delta)
    far = ~near
    return {
        "all": np.ones(n_used, dtype=bool),
        "near_threshold": near,
        "far_from_threshold": far,
        "threshold_distance": values.astype(float),
        "near_threshold_delta": float(delta),
    }


def _empty_row() -> list[float]:
    return [float("nan")] * 7


def _row_for_mask(y, mu, sigma, elapsed, mask: np.ndarray) -> list[float]:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return _empty_row()
    y_m = np.asarray(y).reshape(-1)[mask]
    mu_m = np.asarray(mu).reshape(-1)[mask]
    sig_m = np.asarray(sigma).reshape(-1)[mask]
    rmse = float(np.sqrt(np.mean((mu_m - y_m) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y_m, dtype=torch.float32),
            torch.as_tensor(mu_m, dtype=torch.float32),
            torch.as_tensor(sig_m, dtype=torch.float32),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, elapsed, y_m, mu_m, sig_m)


def _pack_split_metrics(y, mu, sigma, elapsed, masks: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for split in RESULT_SPLITS:
        if split in masks:
            out[split] = _row_for_mask(y, mu, sigma, elapsed, masks[split])
            out[f"n_{split}"] = int(np.asarray(masks[split], dtype=bool).sum())
    return out


def _build_neighborhoods(X_test, X_train, Y_train, device, *, m1: int, Q: int, n: int):
    neighborhoods = find_neighborhoods(X_test.cpu(), X_train.cpu(), Y_train.cpu(), M=n)
    regions, X_nb_list, y_nb_list = [], [], []
    for i in range(X_test.shape[0]):
        X_nb = neighborhoods[i]["X_neighbors"].to(device)
        y_nb = neighborhoods[i]["y_neighbors"].to(device)
        regions.append({"X": X_nb, "y": y_nb, "C": torch.randn(m1, Q, device=device)})
        X_nb_list.append(X_nb)
        y_nb_list.append(y_nb)
    return regions, X_nb_list, y_nb_list


def _run_lmjgp_predict_vi(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    args: argparse.Namespace,
    device: torch.device,
):
    Q_default = args.n_pls if args.n_pls > 0 else 5
    Q = min(args.Q if args.Q is not None else Q_default, X_train.shape[1])
    m1 = 4
    m2 = min(40, max(8, len(X_train)))
    n = min(args.n, len(X_train))
    MC_num = 3

    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    Xv = torch.from_numpy(X_test).float().to(device)
    yv = torch.from_numpy(y_test).float().to(device)
    T, D = Xv.shape
    regions, X_nb_list, y_nb_list = _build_neighborhoods(Xv, Xt, yt, device, m1=m1, Q=Q, n=n)

    V_params = {
        "mu_V": torch.randn(m2, Q, D, device=device, requires_grad=True),
        "sigma_V": torch.rand(m2, Q, D, device=device, requires_grad=True),
    }
    u_params = []
    for _ in range(T):
        u_params.append(
            {
                "U_logit": torch.zeros(1, device=device, requires_grad=True),
                "mu_u": torch.randn(m1, device=device, requires_grad=True),
                "Sigma_u": torch.eye(m1, device=device, requires_grad=True),
                "sigma_noise": torch.tensor(0.5, device=device, requires_grad=True),
                "sigma_k": torch.tensor(0.5, device=device, requires_grad=True),
                "omega": torch.randn(Q + 1, device=device, requires_grad=True),
            }
        )
    X_train_mean = Xt.mean(dim=0)
    X_train_std = Xt.std(dim=0)
    Z = X_train_mean + torch.randn(m2, D, device=device) * X_train_std
    hyperparams = {
        "Z": Z,
        "X_test": Xv,
        "lengthscales": torch.rand(Q, device=device, requires_grad=True),
        "var_w": torch.tensor(1.0, device=device, requires_grad=True),
    }
    t0 = time.perf_counter()
    V_params, u_params, hyperparams = train_vi(
        regions=regions,
        V_params=V_params,
        u_params=u_params,
        hyperparams=hyperparams,
        lr=0.01,
        num_steps=args.lmjgp_steps,
        log_interval=max(50, args.lmjgp_steps // 5),
    )
    mu, var = predict_vi(regions, V_params, hyperparams, M=MC_num)
    elapsed = time.perf_counter() - t0
    sigma = torch.sqrt(torch.clamp(var, min=1e-12))
    return {
        "mu": mu.detach().cpu().numpy(),
        "sigma": sigma.detach().cpu().numpy(),
        "elapsed": float(elapsed),
        "X_neighbors_list": X_nb_list,
        "y_neighbors_list": y_nb_list,
        "Q": Q,
        "m1": m1,
        "n": n,
    }


def _run_jgp_pca(
    X_train,
    y_train,
    X_test,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    Q_default = args.n_pls if args.n_pls > 0 else 5
    Q = min(args.Q if args.Q is not None else Q_default, X_train.shape[1])
    n = min(args.n, len(X_train))
    t0 = time.perf_counter()
    pca = PCA(n_components=Q, random_state=seed)
    X_train_proj = pca.fit_transform(np.asarray(X_train, dtype=np.float64))
    X_test_proj = pca.transform(np.asarray(X_test, dtype=np.float64))
    jgp = JumpGP(
        X_train_proj,
        np.asarray(y_train, dtype=np.float64).reshape(-1),
        X_test_proj,
        L=1,
        M=int(n),
        mode="CEM",
        bVerbose=False,
    )
    jgp.fit()
    elapsed = time.perf_counter() - t0
    mu = np.asarray(jgp.jump_results["mu"], dtype=np.float64).reshape(-1)
    sig2 = np.asarray(jgp.jump_results["sig2"], dtype=np.float64).reshape(-1)
    n_bad_sig2 = int(np.sum(~np.isfinite(sig2) | (sig2 < 0.0)))
    sig2 = np.where(np.isfinite(sig2) & (sig2 >= 0.0), sig2, 1e-12)
    sigma = np.sqrt(np.maximum(sig2, 1e-12))
    return {
        "mu": mu,
        "sigma": sigma,
        "elapsed": float(elapsed),
        "diagnostics": {
            "projection": "pca",
            "projection_fit": "train_only",
            "Q": int(Q),
            "n": int(n),
            "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
            "n_negative_or_nonfinite_sig2": n_bad_sig2,
        },
    }


def _run_cemem(X_train, y_train, X_test, y_test, *, args: argparse.Namespace, device: torch.device):
    Q_default = args.n_pls if args.n_pls > 0 else 5
    Q = min(args.Q if args.Q is not None else Q_default, X_train.shape[1])
    m1 = 4
    n = min(args.n, len(X_train))
    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    Xv = torch.from_numpy(X_test).float().to(device)
    _, X_nb_list, y_nb_list = _build_neighborhoods(Xv, Xt, yt, device, m1=m1, Q=Q, n=n)
    y_test_np = np.asarray(y_test, dtype=np.float64)

    W_pls = compute_pls_W0(X_train, y_train, Q=Q)
    W_pls_t = torch.from_numpy(W_pls).float().to(device)
    W_pls_all = W_pls_t.unsqueeze(0).expand(Xv.shape[0], -1, -1).contiguous()

    out_pls = run_projected_jumpgp(W_pls_all, Xv, X_nb_list, y_nb_list, device=device, progress=False)
    results = {
        "cemem_pls_only": {
            "mu": out_pls["mu"],
            "sigma": out_pls["sigma"],
            "elapsed": float(out_pls["elapsed_sec"]),
        }
    }

    V, V_info = build_V_basis(
        X_train,
        y_train,
        n_pls=args.n_pls,
        n_pca=args.n_pca,
        n_random=args.n_random,
        seed=0,
    )
    V_t = torch.from_numpy(V).float().to(device)
    model = CoeffProjection(
        T=Xv.shape[0],
        Q=Q,
        D=Xv.shape[1],
        V=V_t,
        W0_init=W_pls_t,
        learn_W0=True,
        init_C_scale=args.init_C_scale,
        coeff_scale=args.coeff_scale,
        row_normalize=bool(args.row_normalize_W),
    ).to(device)
    cfg = CemEmConfig(
        n_outer=args.n_outer,
        n_m_steps=args.n_m_steps,
        lr=args.lr,
        lambda_gate=args.lambda_gate,
        lambda_smooth=args.lambda_smooth,
        lambda_scale=args.lambda_scale,
        normalize_smoothness=bool(args.normalize_smoothness),
        track_best_W=bool(args.track_best_W),
        snapshot_after_each_outer=False,
        cem_progress=False,
        verbose=False,
    )
    trained = train_projection_cem_em(
        model,
        X_neighbors_list=X_nb_list,
        y_neighbors_list=y_nb_list,
        X_anchors=Xv,
        device=device,
        cfg=cfg,
        label="pdebench_cem_em_coeff",
    )
    out_coeff = run_projected_jumpgp(
        trained.W, Xv, X_nb_list, y_nb_list, device=device, progress=False, desc="cem_em_coeff"
    )
    results["cemem_coeff"] = {
        "mu": out_coeff["mu"],
        "sigma": out_coeff["sigma"],
        "elapsed": float(trained.train_time_sec + out_coeff["elapsed_sec"]),
        "diagnostics": {
            "V_basis": V_info,
            "W": per_anchor_W_diagnostics(trained.W, W_ref=W_pls_t),
            "best_outer": int(trained.best_outer),
            "best_metric_value": float(trained.best_metric_value),
            "train_time_sec": float(trained.train_time_sec),
            "history": trained.history,
            "Q": int(Q),
            "Rv": int(V_t.shape[1]),
            "D": int(Xv.shape[1]),
            "n_neighbors": int(n),
            "cem_em_defaults_source": "experiments/uci/compare_3way_10seeds.py",
        },
    }
    return results


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _add_bool_flag(parser: argparse.ArgumentParser, name: str, *, default: bool = True) -> None:
    parser.add_argument(f"--{name}", nargs="?", const=True, default=default, type=_parse_bool)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")


def run_one_seed(args: argparse.Namespace, seed: int, device: torch.device) -> dict[str, Any]:
    _seed_all(seed)
    dataset = load_dataset(args.dataset_folder)
    X_train, y_train, X_test_full, y_test_full = _dataset_arrays(dataset, max_anchors=None)
    X_train, y_train, X_test, y_test = _dataset_arrays(dataset, max_anchors=args.max_anchors)
    masks = _threshold_masks(dataset, len(X_test_full), len(X_test))

    seed_res: dict[str, Any] = {
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split_counts": {
            k: int(np.asarray(v, dtype=bool).sum())
            for k, v in masks.items()
            if k in RESULT_SPLITS
        },
        "metrics": {},
    }
    if "near_threshold_delta" in masks:
        seed_res["near_threshold_delta"] = masks["near_threshold_delta"]

    if args.include_xgb:
        rmse, crps, elapsed, details = run_xgboost_bootstrap_regression(
            X_train,
            y_train,
            X_test,
            y_test,
            n_bootstrap=10,
            random_state=seed,
            xgb_params={"n_estimators": 80, "max_depth": 4},
        )
        seed_res["metrics"]["xgb_bootstrap"] = _pack_split_metrics(
            y_test, details["mu"], details["sigma"], elapsed, masks
        )

    if args.include_dkl:
        rmse, crps, elapsed, mu, sigma = run_dkl_svgp_regression(
            X_train,
            y_train,
            X_test,
            y_test,
            device=device,
            epochs=30,
            num_inducing=min(64, len(X_train)),
            seed=seed,
        )
        seed_res["metrics"]["dkl_svgp"] = _pack_split_metrics(y_test, mu, sigma, elapsed, masks)

    if args.include_lmjgp:
        lmjgp = _run_lmjgp_predict_vi(
            X_train, y_train, X_test, y_test, args=args, device=device
        )
        seed_res["metrics"]["lmjgp_predict_vi"] = _pack_split_metrics(
            y_test, lmjgp["mu"], lmjgp["sigma"], lmjgp["elapsed"], masks
        )

    if args.include_jgp_pca:
        jgp_pca = _run_jgp_pca(X_train, y_train, X_test, args=args, seed=seed)
        seed_res["metrics"]["jgp_pca"] = _pack_split_metrics(
            y_test, jgp_pca["mu"], jgp_pca["sigma"], jgp_pca["elapsed"], masks
        )
        seed_res.setdefault("diagnostics", {})["jgp_pca"] = jgp_pca["diagnostics"]

    if args.include_cemem:
        cem = _run_cemem(X_train, y_train, X_test, y_test, args=args, device=device)
        for name, out in cem.items():
            seed_res["metrics"][name] = _pack_split_metrics(
                y_test, out["mu"], out["sigma"], out["elapsed"], masks
            )
            if "diagnostics" in out:
                seed_res.setdefault("diagnostics", {})[name] = out["diagnostics"]

    return seed_res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baselines on processed PDEBench-Burgers surrogate data.")
    p.add_argument("--dataset-folder", default="data/processed/pdebench_burgers_shockjump")
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_anchors", type=int, default=None)
    p.add_argument("--lmjgp_steps", type=int, default=300)
    _add_bool_flag(p, "include_xgb", default=True)
    _add_bool_flag(p, "include_dkl", default=True)
    _add_bool_flag(p, "include_lmjgp", default=True)
    _add_bool_flag(p, "include_jgp_pca", default=False)
    _add_bool_flag(p, "include_cemem", default=True)
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--n", "--neighbors", dest="n", type=int, default=35)
    p.add_argument("--n_pls", type=int, default=5)
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--n_random", type=int, default=0)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_m_steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lambda_gate", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_scale", type=float, default=0.0)
    p.add_argument("--init_C_scale", type=float, default=0.0)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--row_normalize_W", type=int, default=1)
    p.add_argument("--normalize_smoothness", type=int, default=1)
    p.add_argument("--track_best_W", type=int, default=1)
    p.add_argument("--out", default="")
    p.add_argument("--out_json", default="")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    total = {
        "dataset_folder": str(Path(args.dataset_folder)),
        "result_row": [
            "rmse",
            "mean_crps",
            "wall_sec",
            "coverage_90",
            "width_90",
            "coverage_95",
            "width_95",
        ],
        "runs": {},
    }
    for seed in range(args.seed_start, args.seed_start + args.num_exp):
        print(f"[PDEBench-Burgers] seed={seed}")
        total["runs"][seed] = run_one_seed(args, seed, device)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(total, f)
        print(f"Wrote pickle: {out_path}")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(f"pdebench_burgers_compare_{stamp}.pkl")
        with out.open("wb") as f:
            pickle.dump(total, f)
        print(f"Wrote pickle: {out}")

    if args.out_json:
        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with out_json_path.open("w", encoding="utf-8") as f:
            json.dump(_sanitize(total), f, indent=2)
        print(f"Wrote json: {out_json_path}")
    return total


if __name__ == "__main__":
    main()

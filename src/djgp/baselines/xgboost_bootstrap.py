"""Bootstrap ensemble of XGBoost regressors for epistemic uncertainty (+ homoscedastic aleatoric)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

try:
    import xgboost as xgb
except ImportError as e:  # pragma: no cover
    xgb = None
    _XGB_IMPORT_ERROR = e
else:
    _XGB_IMPORT_ERROR = None


def run_xgboost_bootstrap_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_bootstrap: int = 30,
    random_state: int = 0,
    xgb_params: dict[str, Any] | None = None,
) -> tuple[float, float, float, dict[str, np.ndarray]]:
    """
    Train `n_bootstrap` XGBoost models on bootstrap samples of the training set.

    Predictive distribution per test point: Gaussian with mean = ensemble mean prediction
    and variance = Var_b[f_b(x)] + sigma_a^2, where sigma_a^2 is a homoscedastic aleatoric
    term estimated from training residuals of the ensemble mean.

    Returns
    -------
    rmse : float
    mean_crps : float
    elapsed_sec : float
    details : dict with 'mu', 'sigma', 'preds_boot' (n_test, n_bootstrap)
    """
    if _XGB_IMPORT_ERROR is not None:
        raise ImportError(
            "xgboost is required for this baseline. Install with: pip install xgboost"
        ) from _XGB_IMPORT_ERROR

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64).ravel()

    rng = np.random.default_rng(random_state)
    t0 = time.perf_counter()

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if xgb_params:
        params.update(xgb_params)

    n = X_train.shape[0]
    preds_train = np.zeros((n, n_bootstrap), dtype=np.float64)
    preds_test = np.zeros((X_test.shape[0], n_bootstrap), dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train[idx], y_train[idx])
        preds_train[:, b] = model.predict(X_train)
        preds_test[:, b] = model.predict(X_test)

    mu_train = preds_train.mean(axis=1)
    sigma_a2 = float(np.mean((y_train - mu_train) ** 2))
    sigma_a2 = max(sigma_a2, 1e-8)

    mu_test = preds_test.mean(axis=1)
    var_epi = preds_test.var(axis=1, ddof=1)
    var_epi = np.maximum(var_epi, 1e-8)
    sigma_test = np.sqrt(var_epi + sigma_a2)

    rmse = float(np.sqrt(np.mean((mu_test - y_test) ** 2)))

    from djgp.evaluation.crps import mean_gaussian_crps_torch

    mean_crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y_test, dtype=torch.float32),
            torch.as_tensor(mu_test, dtype=torch.float32),
            torch.as_tensor(sigma_test, dtype=torch.float32),
        ).item()
    )

    elapsed = time.perf_counter() - t0
    details = {
        "mu": mu_test,
        "sigma": sigma_test,
        "preds_boot": preds_test,
        "sigma_a2": sigma_a2,
    }
    return rmse, mean_crps, elapsed, details

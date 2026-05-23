"""Non-DJGP baselines: Deep Kernel Learning, bootstrap XGBoost, global GPs."""

from djgp.baselines.dkl import run_dkl_svgp_regression
from djgp.baselines.global_sparse_gp import (
    BACKENDS as GLOBAL_GP_BACKENDS,
    GlobalGPConfig,
    GlobalGPModel,
    GlobalGPTrainResult,
    predict_oob_ensemble,
    predict_oob_kfold,
)
from djgp.baselines.xgboost_bootstrap import run_xgboost_bootstrap_regression

__all__ = [
    "GLOBAL_GP_BACKENDS",
    "GlobalGPConfig",
    "GlobalGPModel",
    "GlobalGPTrainResult",
    "predict_oob_ensemble",
    "predict_oob_kfold",
    "run_dkl_svgp_regression",
    "run_xgboost_bootstrap_regression",
]

"""
Deep Kernel Learning: learned feature map + sparse variational GP (GPyTorch).

Features are computed outside the `ApproximateGP` so inducing points and batches
live in the same latent space (GPyTorch `VariationalStrategy` concatenates
inducing points with inputs before `forward`).
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class _FeatureMap(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SVGPOnFeatures(ApproximateGP):
    """SVGP where `forward(h)` only builds the prior on latent features h."""

    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=inducing_points.size(-1)),
        )

    def forward(self, h: torch.Tensor) -> MultivariateNormal:
        mean_h = self.mean_module(h)
        covar_h = self.covar_module(h)
        return MultivariateNormal(mean_h, covar_h)


def run_dkl_svgp_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    device: torch.device | None = None,
    d_hidden: int = 64,
    d_latent: int = 4,
    num_inducing: int = 128,
    epochs: int = 80,
    batch_size: int = 1024,
    lr: float = 0.02,
    seed: int = 0,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Train DKL-SVGP and return (rmse, mean_crps, elapsed_sec, mu_test, sigma_test) on original y scale.
    Last two arrays are length n_test for Gaussian PI / coverage metrics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).ravel()

    y_mean = y_train.mean()
    y_std = float(max(y_train.std(), 1e-6))
    y_train_z = (y_train - y_mean) / y_std
    y_test_z = (y_test - y_mean) / y_std

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train_z).to(device)
    Xv = torch.from_numpy(X_test).to(device)
    yv = torch.from_numpy(y_test_z).to(device)

    d_in = X_train.shape[1]
    features = _FeatureMap(d_in, d_hidden, d_latent).to(device)

    with torch.no_grad():
        z_init = features(Xt[: min(512, len(Xt))])
        idx = torch.randperm(z_init.size(0), device=device)[:num_inducing]
        inducing = z_init[idx].clone()

    svgp = _SVGPOnFeatures(inducing).to(device)
    likelihood = GaussianLikelihood().to(device)
    likelihood.noise = torch.tensor(0.01, device=device)

    mll = VariationalELBO(likelihood, svgp, num_data=len(Xt))
    optimizer = torch.optim.Adam(
        list(features.parameters()) + list(svgp.parameters()) + list(likelihood.parameters()),
        lr=lr,
    )

    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=min(batch_size, len(Xt)),
        shuffle=True,
    )

    t0 = time.perf_counter()
    features.train()
    svgp.train()
    likelihood.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            hb = features(xb)
            with gpytorch.settings.cholesky_jitter(1e-3):
                out = svgp(hb)
                loss = -mll(out, yb)
            loss.backward()
            optimizer.step()

    features.eval()
    svgp.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        hv = features(Xv)
        pred = likelihood(svgp(hv))

    mu_z = pred.mean
    var_z = pred.variance
    sigma_z = torch.sqrt(torch.clamp(var_z, min=1e-12))

    mu = mu_z * y_std + y_mean
    sigma = sigma_z * y_std

    from djgp.evaluation.crps import mean_gaussian_crps_torch

    y_orig = yv * y_std + y_mean
    rmse = torch.sqrt(torch.mean((mu - y_orig) ** 2))
    mean_crps = mean_gaussian_crps_torch(y_orig, mu, sigma)
    elapsed = time.perf_counter() - t0

    mu_np = mu.detach().cpu().numpy()
    sigma_np = sigma.detach().cpu().numpy()
    return float(rmse.item()), float(mean_crps.item()), float(elapsed), mu_np, sigma_np

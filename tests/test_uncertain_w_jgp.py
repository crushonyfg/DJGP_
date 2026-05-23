import torch

from djgp.projections.uncertain_w_jgp import (
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
    SelfCEMConfig,
    UncertainWQRMstepConfig,
    UncertainWJGPConfig,
    expected_se_kernel_anchor,
    expected_se_kernel_pointwise,
    fit_uncertain_gate_from_labels,
    fit_uncertain_kernel_hyperparams_from_labels,
    kernel_vector_covariance_anchor,
    kernel_vector_covariance_pointwise,
    run_uncertain_w_self_cem,
    sparse_gp_lowrank_residual_w_moments_for_mode,
    sparse_gp_uncertain_w_moments_for_mode,
    sparse_gp_w_moments_diag,
    train_qr_uncertain_w_cem_mstep,
    uncertain_w_cem_update_labels,
    uncertain_w_cem_predict_from_labels,
    uncertain_w_jgp_vem_predict,
)


def test_anchor_expected_kernel_matches_deterministic_when_variance_zero():
    torch.manual_seed(0)
    T, n, D, Q = 2, 5, 3, 2
    X = torch.randn(T, n, D, dtype=torch.float64)
    W = torch.randn(T, Q, D, dtype=torch.float64)
    W_var = torch.zeros_like(W)
    ell = 1.3
    signal_var = 1.7
    K_unc = expected_se_kernel_anchor(X, X, W, W_var, lengthscale=ell, signal_var=signal_var)
    Z = torch.einsum("tnd,tqd->tnq", X, W)
    d2 = (Z.unsqueeze(2) - Z.unsqueeze(1)).pow(2).sum(-1)
    K_det = signal_var * torch.exp(-0.5 * d2 / (ell**2))
    assert torch.allclose(K_unc, K_det, atol=1e-10, rtol=1e-10)


def test_pointwise_expected_kernel_matches_deterministic_when_covariance_zero():
    torch.manual_seed(1)
    T, m, D, Q = 2, 6, 3, 2
    X = torch.randn(T, m, D, dtype=torch.float64)
    W = torch.randn(T, m, Q, D, dtype=torch.float64)
    W_cov = torch.zeros(T, Q, D, m, m, dtype=torch.float64)
    ell = 0.9
    signal_var = 1.2
    K_unc = expected_se_kernel_pointwise(X, W, W_cov, lengthscale=ell, signal_var=signal_var)
    Z = torch.einsum("tmqd,tmd->tmq", W, X)
    d2 = (Z.unsqueeze(2) - Z.unsqueeze(1)).pow(2).sum(-1)
    K_det = signal_var * torch.exp(-0.5 * d2 / (ell**2))
    assert torch.allclose(K_unc, K_det, atol=1e-10, rtol=1e-10)


def test_kernel_vector_covariance_zero_when_w_is_deterministic():
    torch.manual_seed(2)
    T, n, D, Q = 3, 4, 2, 1
    Xa = torch.randn(T, D, dtype=torch.float64)
    Xn = torch.randn(T, n, D, dtype=torch.float64)
    W = torch.randn(T, Q, D, dtype=torch.float64)
    W_var = torch.zeros_like(W)
    k_star = expected_se_kernel_anchor(
        Xa.unsqueeze(1), Xn, W, W_var, lengthscale=1.0, signal_var=1.0
    ).squeeze(1)
    cov = kernel_vector_covariance_anchor(
        Xa, Xn, W, W_var, k_star, lengthscale=1.0, signal_var=1.0
    )
    assert torch.allclose(cov, torch.zeros_like(cov), atol=1e-10, rtol=1e-10)


def test_kernel_vector_covariance_pointwise_zero_when_w_is_deterministic():
    torch.manual_seed(3)
    T, n, D, Q = 2, 5, 2, 1
    Xall = torch.randn(T, n + 1, D, dtype=torch.float64)
    W = torch.randn(T, n + 1, Q, D, dtype=torch.float64)
    W_cov = torch.zeros(T, Q, D, n + 1, n + 1, dtype=torch.float64)
    Kall = expected_se_kernel_pointwise(Xall, W, W_cov, lengthscale=1.0, signal_var=1.0)
    cov = kernel_vector_covariance_pointwise(
        Xall, W, W_cov, Kall[:, 0, 1:], lengthscale=1.0, signal_var=1.0
    )
    assert torch.allclose(cov, torch.zeros_like(cov), atol=1e-10, rtol=1e-10)


def test_uncertain_anchor_vem_adds_variance_over_mean_w_case():
    torch.manual_seed(4)
    T, n, D, Q = 8, 18, 2, 1
    Xa = torch.linspace(-0.4, 0.4, T, dtype=torch.float64).unsqueeze(1)
    Xa = torch.cat([Xa, torch.zeros_like(Xa)], dim=1)
    Xn = Xa.unsqueeze(1) + 0.35 * torch.randn(T, n, D, dtype=torch.float64)
    y = torch.sin(3.0 * Xn[..., 0]) + 1.5 * (Xn[..., 0] > 0).to(torch.float64)
    y = y + 0.05 * torch.randn_like(y)
    W = torch.zeros(T, Q, D, dtype=torch.float64)
    W[..., 0] = 1.0
    gate = torch.zeros(T, Q + 1, dtype=torch.float64)
    gate[:, 1] = 4.0
    cfg_base = UncertainWJGPConfig(
        mode="anchor",
        lengthscale=0.35,
        noise_var=0.05,
        vem_iters=5,
        kernel_vector_correction=True,
    )
    det = uncertain_w_jgp_vem_predict(
        Xa, Xn, y, config=cfg_base, W_mu_anchor=W, W_var_anchor=torch.zeros_like(W), gate=gate
    )
    unc = uncertain_w_jgp_vem_predict(
        Xa, Xn, y, config=cfg_base, W_mu_anchor=W, W_var_anchor=0.12 * torch.ones_like(W), gate=gate
    )
    assert torch.isfinite(unc.mu).all()
    assert torch.isfinite(unc.var).all()
    assert unc.var.mean() > det.var.mean()
    assert unc.correction.mean() > 0.0


def test_fixed_label_cem_prediction_matches_direct_projected_gp_when_w_deterministic():
    torch.manual_seed(6)
    T, n, D, Q = 3, 9, 2, 1
    Xa = torch.randn(T, D, dtype=torch.float64)
    Xn = Xa.unsqueeze(1) + 0.3 * torch.randn(T, n, D, dtype=torch.float64)
    W = torch.zeros(T, Q, D, dtype=torch.float64)
    W[:, 0, 0] = 1.0
    z = torch.einsum("tnd,tqd->tnq", Xn, W)
    za = torch.einsum("td,tqd->tq", Xa, W)
    y = torch.sin(2.0 * z[..., 0]) + 0.05 * torch.randn(T, n, dtype=torch.float64)
    labels = Xn[..., 0] > Xa[:, None, 0] - 0.15
    mean_const = torch.stack([y[t, labels[t]].mean() for t in range(T)])
    ell = 0.7
    signal = 1.4
    noise = 0.03
    out = uncertain_w_cem_predict_from_labels(
        Xa,
        Xn,
        y,
        labels,
        mode="anchor",
        W_mu_anchor=W,
        W_var_anchor=torch.zeros_like(W),
        lengthscale=ell,
        signal_var=signal,
        noise_var=noise,
        mean_const=mean_const,
        correction_weight=1.0,
    )
    mus = []
    vars_ = []
    for t in range(T):
        I = labels[t]
        Zi = z[t, I]
        diff = Zi.unsqueeze(1) - Zi.unsqueeze(0)
        K = signal * torch.exp(-0.5 * diff.pow(2).sum(-1) / (ell**2))
        kt = signal * torch.exp(-0.5 * (za[t].view(1, Q) - Zi).pow(2).sum(-1) / (ell**2))
        Ky = K + noise * torch.eye(int(I.sum()), dtype=torch.float64)
        L = torch.linalg.cholesky(Ky + 1e-5 * torch.eye(int(I.sum()), dtype=torch.float64))
        alpha = torch.cholesky_solve((y[t, I] - mean_const[t]).unsqueeze(-1), L).squeeze(-1)
        mus.append(mean_const[t] + kt @ alpha)
        v = torch.cholesky_solve(kt.unsqueeze(-1), L).squeeze(-1)
        vars_.append(signal - kt @ v + noise)
    assert torch.allclose(out.mu, torch.stack(mus), atol=1e-5, rtol=1e-5)
    assert torch.allclose(out.var, torch.stack(vars_), atol=1e-5, rtol=1e-5)
    assert torch.allclose(out.correction, torch.zeros_like(out.correction), atol=1e-10, rtol=1e-10)


def test_fixed_label_hyperopt_returns_finite_params():
    torch.manual_seed(7)
    T, n, D, Q = 3, 10, 2, 1
    Xa = torch.randn(T, D, dtype=torch.float64)
    Xn = Xa.unsqueeze(1) + 0.25 * torch.randn(T, n, D, dtype=torch.float64)
    W = torch.zeros(T, Q, D, dtype=torch.float64)
    W[:, 0, 0] = 1.0
    y = torch.sin(2.0 * Xn[..., 0])
    labels = torch.ones(T, n, dtype=torch.bool)
    params = fit_uncertain_kernel_hyperparams_from_labels(
        Xa,
        Xn,
        y,
        labels,
        mode="anchor",
        W_mu_anchor=W,
        W_var_anchor=0.05 * torch.ones_like(W),
        init_lengthscale=0.8,
        init_signal_var=1.0,
        init_noise_var=0.05,
    )
    assert params["lengthscale"].shape == (T, Q)
    assert torch.isfinite(params["lengthscale"]).all()
    assert torch.isfinite(params["signal_var"]).all()
    assert torch.isfinite(params["noise_var"]).all()


def test_gate_learning_and_self_cem_run_on_small_anchor_case():
    torch.manual_seed(8)
    T, n, D, Q = 2, 8, 2, 1
    Xa = torch.randn(T, D, dtype=torch.float64)
    Xn = Xa.unsqueeze(1) + 0.35 * torch.randn(T, n, D, dtype=torch.float64)
    W = torch.zeros(T, Q, D, dtype=torch.float64)
    W[:, 0, 0] = 1.0
    labels = (Xn[..., 0] > Xa[:, None, 0]).to(torch.float64)
    y = torch.sin(2.0 * Xn[..., 0]) + labels
    gate_fit = fit_uncertain_gate_from_labels(
        Xa,
        Xn,
        labels,
        mode="anchor",
        W_mu_anchor=W,
        W_var_anchor=0.02 * torch.ones_like(W),
        config=FixedLabelGateConfig(steps=3, lr=0.05, max_norm=4.0),
    )
    assert gate_fit["gate"].shape == (T, Q + 1)
    assert torch.isfinite(gate_fit["gate"]).all()
    upd = uncertain_w_cem_update_labels(
        Xa,
        Xn,
        y,
        labels,
        mode="anchor",
        W_mu_anchor=W,
        W_var_anchor=0.02 * torch.ones_like(W),
        gate=gate_fit["gate"],
        lengthscale=0.7,
        signal_var=1.0,
        noise_var=0.05,
        mean_const=y.mean(dim=1),
        min_inliers=2,
    )
    assert upd["labels"].shape == (T, n)
    state = run_uncertain_w_self_cem(
        Xa,
        Xn,
        y,
        labels,
        mode="anchor",
        W_mu_anchor=W,
        W_var_anchor=0.02 * torch.ones_like(W),
        init_lengthscale=0.7,
        init_signal_var=1.0,
        init_noise_var=0.05,
        config=SelfCEMConfig(
            cem_updates=1,
            hyperopt=FixedLabelHyperoptConfig(steps=2, lr=0.03),
            gate=FixedLabelGateConfig(steps=2, lr=0.03),
        ),
    )
    assert state["labels"].shape == (T, n)
    assert torch.isfinite(state["lengthscale"]).all()


def test_sparse_gp_w_moments_shapes_and_pointwise_vem_runs():
    torch.manual_seed(5)
    T, n, Dx, D, Q, M = 4, 8, 2, 2, 1, 6
    Xa = torch.randn(T, Dx, dtype=torch.float64)
    Xn = Xa.unsqueeze(1) + 0.2 * torch.randn(T, n, Dx, dtype=torch.float64)
    Xall = torch.cat([Xa.unsqueeze(1), Xn], dim=1)
    U = torch.randn(M, Dx, dtype=torch.float64)
    R_mu = torch.randn(M, Q, D, dtype=torch.float64) * 0.5
    R_var = torch.full((M, Q, D), 0.05, dtype=torch.float64)
    W_mu, W_cov = sparse_gp_w_moments_diag(
        Xall, U, R_mu, R_var, lengthscale=0.8, include_conditional_residual=True
    )
    assert W_mu.shape == (T, n + 1, Q, D)
    assert W_cov.shape == (T, Q, D, n + 1, n + 1)
    y = torch.sin(Xn[..., 0])
    cfg = UncertainWJGPConfig(
        mode="pointwise",
        lengthscale=0.6,
        noise_var=0.05,
        vem_iters=3,
        kernel_vector_correction=True,
    )
    out = uncertain_w_jgp_vem_predict(Xa, Xn, y, config=cfg, W_mu_all=W_mu, W_cov_all=W_cov)
    assert out.mu.shape == (T,)
    assert out.sigma.shape == (T,)
    assert torch.isfinite(out.sigma).all()


def test_qr_mstep_runs_and_returns_sparse_w_moments():
    torch.manual_seed(9)
    T, n, D, Q, M = 2, 7, 2, 1, 5
    Xa = torch.randn(T, D, dtype=torch.float64)
    Xn = Xa.unsqueeze(1) + 0.25 * torch.randn(T, n, D, dtype=torch.float64)
    y = torch.sin(2.0 * Xn[..., 0])
    labels = torch.ones(T, n, dtype=torch.bool)
    U = torch.randn(M, D, dtype=torch.float64)
    W0 = torch.zeros(Q, D, dtype=torch.float64)
    W0[0, 0] = 1.0
    gate = torch.zeros(T, Q + 1, dtype=torch.float64)
    res = train_qr_uncertain_w_cem_mstep(
        Xa,
        Xn,
        y,
        labels,
        mode="anchor",
        inducing_X=U,
        init_W=W0,
        gate=gate,
        lengthscale=torch.full((T, Q), 0.8, dtype=torch.float64),
        signal_var=torch.ones(T, dtype=torch.float64),
        noise_var=0.05 * torch.ones(T, dtype=torch.float64),
        mean_const=y.mean(dim=1),
        config=UncertainWQRMstepConfig(steps=2, lr=0.01, beta_kl=0.01, log_interval=1),
    )
    assert res.R_mu.shape == (M, Q, D)
    assert res.R_log_std.shape == (M, Q, D)
    assert len(res.history) == 2
    assert "grad_data_over_scaled_KL_R_mu" in res.history[-1]
    moments = sparse_gp_uncertain_w_moments_for_mode(
        mode="anchor",
        X_anchor=Xa,
        X_neighbors=Xn,
        inducing_X=U,
        R_mu=res.R_mu,
        R_log_std=res.R_log_std,
        lengthscale=0.8,
    )
    assert moments["W_mu_anchor"].shape == (T, Q, D)
    assert moments["W_var_anchor"].shape == (T, Q, D)
    assert torch.isfinite(moments["W_var_anchor"]).all()


def test_qr_mstep_orthogonality_penalty_records_diagnostics():
    torch.manual_seed(11)
    T, n, D, Q, M = 2, 7, 4, 2, 5
    Xa = torch.randn(T, D, dtype=torch.float64)
    Xn = Xa.unsqueeze(1) + 0.25 * torch.randn(T, n, D, dtype=torch.float64)
    y = torch.sin(Xn[..., 0]) + 0.1 * Xn[..., 1]
    labels = torch.ones(T, n, dtype=torch.bool)
    U = torch.randn(M, D, dtype=torch.float64)
    W0 = torch.zeros(Q, D, dtype=torch.float64)
    W0[0, 0] = 1.0
    W0[1, 1] = 1.0
    gate = torch.zeros(T, Q + 1, dtype=torch.float64)
    res = train_qr_uncertain_w_cem_mstep(
        Xa,
        Xn,
        y,
        labels,
        mode="anchor",
        inducing_X=U,
        init_W=W0,
        gate=gate,
        lengthscale=torch.full((T, Q), 0.8, dtype=torch.float64),
        signal_var=torch.ones(T, dtype=torch.float64),
        noise_var=0.05 * torch.ones(T, dtype=torch.float64),
        mean_const=y.mean(dim=1),
        config=UncertainWQRMstepConfig(
            steps=2,
            lr=0.01,
            beta_kl=0.01,
            ortho_weight=0.1,
            log_interval=1,
        ),
    )
    assert "ortho_penalty" in res.diagnostics
    assert "ortho_term" in res.history[-1]
    assert "grad_ortho_R_mu" in res.history[-1]
    assert res.diagnostics["ortho_penalty"] >= 0.0


def test_qr_mstep_shared_svd_mean_parameterization_runs():
    torch.manual_seed(12)
    T, n, D, Q, M = 2, 7, 4, 2, 5
    Xa = torch.randn(T, D, dtype=torch.float64)
    Xn = Xa.unsqueeze(1) + 0.25 * torch.randn(T, n, D, dtype=torch.float64)
    y = torch.sin(Xn[..., 0]) + 0.1 * Xn[..., 1]
    labels = torch.ones(T, n, dtype=torch.bool)
    U = torch.randn(M, D, dtype=torch.float64)
    W0 = torch.zeros(Q, D, dtype=torch.float64)
    W0[0, 0] = 1.0
    W0[1, 1] = 1.0
    gate = torch.zeros(T, Q + 1, dtype=torch.float64)
    res = train_qr_uncertain_w_cem_mstep(
        Xa,
        Xn,
        y,
        labels,
        mode="anchor",
        inducing_X=U,
        init_W=W0,
        gate=gate,
        lengthscale=torch.full((T, Q), 0.8, dtype=torch.float64),
        signal_var=torch.ones(T, dtype=torch.float64),
        noise_var=0.05 * torch.ones(T, dtype=torch.float64),
        mean_const=y.mean(dim=1),
        config=UncertainWQRMstepConfig(
            steps=2,
            lr=0.01,
            beta_kl=0.01,
            mean_parameterization="shared_svd",
            log_interval=1,
        ),
    )
    assert res.R_mu.shape == (M, Q, D)
    assert res.diagnostics["mean_parameterization"] == "shared_svd"
    assert res.diagnostics["svd_left_orth_error"] < 1e-8
    assert res.diagnostics["svd_right_orth_error"] < 1e-8
    assert torch.isfinite(res.R_mu).all()


def test_lowrank_qr_mstep_group_penalty_and_basis_scale_run():
    torch.manual_seed(10)
    T, n, D, Q, r, M = 2, 7, 4, 2, 2, 5
    Xa = torch.randn(T, D, dtype=torch.float64)
    Xn = Xa.unsqueeze(1) + 0.25 * torch.randn(T, n, D, dtype=torch.float64)
    y = torch.sin(Xn[..., 0]) + 0.2 * Xn[..., 1]
    labels = torch.ones(T, n, dtype=torch.bool)
    U = torch.randn(M, D, dtype=torch.float64)
    W0 = torch.zeros(Q, D, dtype=torch.float64)
    W0[0, 0] = 1.0
    W0[1, 1] = 1.0
    V = torch.linalg.qr(torch.randn(D, r, dtype=torch.float64)).Q[:, :r]
    gate = torch.zeros(T, Q + 1, dtype=torch.float64)
    res = train_qr_uncertain_w_cem_mstep(
        Xa,
        Xn,
        y,
        labels,
        mode="anchor",
        inducing_X=U,
        init_W=W0,
        residual_basis=V,
        gate=gate,
        lengthscale=torch.full((T, Q), 0.8, dtype=torch.float64),
        signal_var=torch.ones(T, dtype=torch.float64),
        noise_var=0.05 * torch.ones(T, dtype=torch.float64),
        mean_const=y.mean(dim=1),
        config=UncertainWQRMstepConfig(
            steps=2,
            lr=0.01,
            beta_kl=0.01,
            residual_group_weight=1e-3,
            train_basis_scale=True,
            basis_scale_l1_weight=1e-3,
            log_interval=1,
        ),
    )
    assert res.R_mu.shape == (M, Q, r)
    assert res.basis_scale is not None
    assert res.basis_scale.shape == (r,)
    assert "residual_group_penalty_final" in res.diagnostics
    V_eff = V * res.basis_scale.reshape(1, -1)
    moments = sparse_gp_lowrank_residual_w_moments_for_mode(
        mode="anchor",
        X_anchor=Xa,
        X_neighbors=Xn,
        inducing_X=U,
        C_mu=res.R_mu,
        C_log_std=res.R_log_std,
        base_W=W0,
        basis_V=V_eff,
        lengthscale=0.8,
    )
    assert moments["W_cov_anchor"].shape == (T, Q, D, D)
    assert torch.isfinite(moments["W_cov_anchor"]).all()

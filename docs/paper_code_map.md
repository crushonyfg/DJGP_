# Paper to Code Map

This file maps the DJGP paper and appendix to the current code. It is intentionally pragmatic: names in the code do not always match the notation in the paper.

## Paper Sections

| Paper concept | Paper notation | Code location | Code names |
| --- | --- | --- | --- |
| Jump Gaussian Process baseline | JGP, local labels `v_i`, boundary `h(x; nu)` | `src/JumpGaussianProcess/JumpGP_LD.py`, `src/JumpGaussianProcess/maximize_PD.py`, `src/JumpGaussianProcess/variationalEM.py`, `src/JumpGaussianProcess/stochasticEM.py`, `src/JumpGaussianProcess/jumpgp.py` | `JumpGP_LD`, `maximize_PD`, `variationalEM`, `stochasticEM`, `JumpGP` |
| Local neighborhood per test point | `D_n^(j)`, anchor/test point `x_*^(j)` | `src/shared/jumpgp_runner.py`, `src/shared/djgp_runner.py`, `src/djgp/active_learning.py` | `find_neighborhoods`, `regions[i]['X']`, `regions[i]['y']` |
| Locally linear projection | `W_j`, latent `z_i^(j) = W_j x_i` | `src/djgp/variational.py` | `qW_from_qV`, `mu_W`, `cov_W`, `predict_vi` |
| Global inducing variables for projection process | Paper uses `R`, inducing inputs `x_tilde_l` | `src/djgp/variational.py`, `src/shared/djgp_runner.py` | `V_params['mu_V']`, `V_params['sigma_V']`, `hyperparams['Z']` |
| Projection GP hyperparameters | row lengthscales, variance | `src/djgp/variational.py` | `hyperparams['lengthscales']`, `hyperparams['var_w']` |
| Local inducing variables for latent function | Paper uses `r^(j)` at `z_tilde_l^(j)` | `src/shared/djgp_runner.py`, `src/djgp/variational.py` | `regions[j]['C']`, `u_params[j]['mu_u']`, `u_params[j]['Sigma_u']` |
| Local JGP/gating parameters | `rho_i^(j)`, `omega`, local noise/kernel | `src/djgp/variational.py` | `U_logit`, `omega`, `sigma_noise`, `sigma_k` |
| ELBO in Appendix A / Eq. 21 | expected likelihood plus KL terms | `src/djgp/variational.py` | `compute_ELBO`, `expected_Kfu`, `expected_KufKfu`, `expected_log_sigmoid_gh_batch`, `kl_q_u_batch`, `kl_qp` |
| Variational training algorithm | Algorithm 1 training loop | `src/djgp/variational.py`, `src/djgp/minibatch.py`, `src/shared/djgp_runner.py` | `train_vi`, `train_vi_minibatch`, `main` |
| Prediction by sampling projections | Monte Carlo over `W_j ~ q(W_j)` | `src/djgp/variational.py`, `src/djgp/active_learning.py` | `predict_vi`, `sample_W_for_point` |
| Active learning acquisition | Not core Algorithm 1, later experiment extension | `src/djgp/active_learning.py`, `experiments/active_learning/run_synth_al4jgp.py` | `compute_acquisition_all_with_JGP`, `acquisition_one_x_with_JGP`, `run_al_loop` |

Preferred import paths:

- `djgp.variational` contains the current full-batch variational implementation.
- `djgp.minibatch` contains the minibatch implementation.
- `djgp.active_learning` contains active-learning utilities.
- `jumpgp` wraps the canonical vendored Jump GP implementation in `JumpGaussianProcess/`.

## Experiments

| Paper/appendix experiment | Likely code |
| --- | --- |
| Synthetic datasets with latent-to-observed expansion | `src/data_gen/synthetic.py`, `src/data_gen/highdata.py`, `src/data_gen/highdata_utils.py`, `legacy/old_data_gen/test_datagen.py` |
| Appendix C expansion methods: RP, RF/RFF, PE, AE | `src/data_gen/synthetic.py`, `src/data_gen/highdata.py`, `src/data_gen/autoencoder.py`, `src/data_gen/lh_autoencoder.py`, `src/data_gen/uci_autoencoder.py` |
| Hyperparameter sensitivity for `L1`, `L2`, `n`, `Q` | `experiments/erosion/erosion_exp_alone.py`, `experiments/erosion/erosion_new_dataset.py`, `experiments/synthetic/minibatch_LH.py`, `experiments/synthetic/experiment_new.py`, `experiments/synthetic/compare_K.py` |
| UCI benchmark datasets | `experiments/uci/new_dataset.py`, `experiments/uci/plot_uci_pca.py`, `experiments/uci/plot_uci_pca_nonstationary.py` |
| OHT benchmark | `experiments/oht/OHTDataset_analysis.py`, `experiments/oht/OHT_dataset_test.py`, `legacy/r_code/main_OHT4D_TGP_R1.R` |
| Active-learning synthetic plots | `src/djgp/active_learning.py`, `experiments/active_learning/run_synth_al4jgp.py`, `experiments/active_learning/run_synth_and_plot_acq.py`, `experiments/active_learning/run_synth_and_plot_acq_v2.py` |

## Important Name Mismatches

| Paper | Code |
| --- | --- |
| `K`, latent dimension | Often `Q` in code. `K` is also used for SIR selected dimension in some scripts. |
| `L1`, local inducing count | `m1` |
| `L2`, global inducing count | `m2` |
| `R`, global inducing outputs for projection GP | `V_params` |
| `x_tilde`, global inducing inputs | `hyperparams['Z']` |
| `z_tilde`, local inducing inputs | `regions[j]['C']` |
| `DJGP` | Sometimes written as `LMJGP` or `lmjgp` in result dictionaries. |

## Practical Entry Points

After `pip install -e .` in the `jumpGP` conda env, modules are importable as top-level packages.

Run DJGP on a generated dataset folder:

```powershell
python -m shared.djgp_runner --folder_name <folder-with-dataset-pkl> --Q 5 --m1 4 --m2 40 --n 35 --num_steps 300 --MC_num 5
```

Generate a synthetic dataset:

```powershell
python -m data_gen.synthetic --N 1000 --T 200 --D 30 --Q 5 --caseno 5 --device cpu
```

Run a UCI benchmark script:

```powershell
python -m experiments.uci.new_dataset
```

These scripts may require optional dependencies such as `torch`, `gpytorch`, `pyro`, `ucimlrepo`, `scikit-learn`, `scipy`, `skimage`, and `matplotlib`.

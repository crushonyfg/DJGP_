# L2/LH LMJGP ELBO Gradient Diagnostics

This folder records training-time q(V) gradient diagnostics for the original
LMJGP ELBO on paper-style synthetic L2/LH data.

Files:

- `gradient_history.csv`: diagnostics at requested optimization steps.
- `gradient_latest.csv`: one latest-step row per setting and variant.
- `summary.json`: exact command arguments and setting metadata.

Boundary anchors are a proxy: the top local-y-variance anchors according to
`boundary_quantile`.

```json
{
  "args": {
    "settings": "l2_q2_train1000,lh_q5_train1000",
    "variants": "baseline,pls_kl",
    "seed_start": 0,
    "num_exp": 1,
    "max_anchors": 80,
    "m1": 2,
    "m2": 40,
    "num_steps": 80,
    "lr": 0.01,
    "warmup_frac": 0.5,
    "diag_steps": "0,1,5,10,25,50,80",
    "boundary_quantile": 0.75,
    "out_dir": "experiments\\synthetic\\l2_lh_elbo_gradient_diagnostics_20260514",
    "settings_list": [
      "l2_q2_train1000",
      "lh_q5_train1000"
    ],
    "variants_list": [
      "baseline",
      "pls_kl"
    ],
    "diag_steps_list": [
      0,
      1,
      5,
      10,
      25,
      50,
      80
    ]
  },
  "device": "cuda",
  "settings": {
    "l2_q2_train1000": {
      "family": "L2_phantom",
      "N_train": 1000,
      "N_test": 200,
      "d": 2,
      "H": 20,
      "Q_true": 2,
      "Q": 2,
      "n": 25,
      "caseno": 6,
      "noise_std": 2.0,
      "noise_var": 4.0,
      "expansion": "rff",
      "lengthscale": 0.5,
      "kernel_var": 9.0
    },
    "lh_q5_train1000": {
      "family": "LH",
      "N_train": 1000,
      "N_test": 200,
      "d": 5,
      "H": 30,
      "Q_true": 5,
      "Q": 5,
      "n": 35,
      "caseno": 0,
      "noise_std": 2.0,
      "noise_var": 4.0,
      "expansion": "rff",
      "lengthscale": 0.5,
      "kernel_var": 9.0
    }
  },
  "outputs": {
    "gradient_history": "gradient_history.csv",
    "gradient_latest": "gradient_latest.csv",
    "readme": "README.md"
  }
}
```

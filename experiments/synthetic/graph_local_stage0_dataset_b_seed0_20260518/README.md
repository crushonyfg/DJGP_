# Graph-Local Stage-0 Diagnostics

This folder contains a generator-only pilot for
`docs/graph_local_djgp_experiment_plan.md`.

Synthetic provenance:

- Base low-dimensional generator: temporal AR latent state with SBM block means.
- Expansion: linear latent-to-observed features.
- Graph family: stochastic block model with explicit connected block/ring edges.
- Standardization: raw `dataset.npz` is not standardized; diagnostics standardize
  X only for feature-distance comparisons.
- Models: none trained.

Files:

- `dataset.npz`: generated arrays and graph metadata.
- `diagnostics.json`: full Stage-0 diagnostics.
- `diagnostics.csv`: one-row flat diagnostics table.
- `summary.json`: CLI settings and output manifest.

Runner summary:

```json
{
  "args": {
    "preset": "dataset_b",
    "seed": 0,
    "n_nodes": 300,
    "n_time": 20,
    "n_blocks": 8,
    "p_in": 0.15,
    "p_out": 0.01,
    "graph_radius": 1,
    "graph_lambda": 1.0,
    "noise_x": 0.5,
    "noise_y": 0.1,
    "soft_jump": false,
    "max_pairs": 10000,
    "knn_k": 10,
    "knn_anchors": 1000,
    "out_dir": "experiments/synthetic/graph_local_stage0_dataset_b_seed0_20260518"
  },
  "preset": {
    "description": "Graph-local hard-jump latent synthetic, D=30, q=3.",
    "D": 30,
    "q": 3,
    "jump": true,
    "misleading_features": false,
    "latent_type": "linear"
  },
  "metadata": {
    "generator_family": "graph_local_sbm_latent_surrogate",
    "base_low_dimensional_generator": "temporal AR latent state with SBM block means",
    "expansion_method": "linear_latent_to_observed",
    "latent_dimension": 3,
    "observed_dimension": 30,
    "n_nodes": 300,
    "n_time": 20,
    "train_test_sizes": "diagnostic masks only; see split diagnostics",
    "x_standardization": "not applied in raw saved data",
    "y_standardization": "not applied in raw saved data",
    "graph_type": "sbm",
    "n_blocks": 8,
    "graph_radius": 1,
    "jump": true,
    "soft_jump": false,
    "misleading_features": false,
    "noise_x": 0.5,
    "noise_y": 0.1,
    "seed": 0,
    "thresholds": [
      -0.28909109501443175,
      0.0030213592681747645,
      0.2970157785232205,
      0.22331390645469726,
      0.11612585217610903,
      0.13620481066443244,
      0.34285537258718873,
      -0.0352634669213385
    ],
    "jump_delta": [
      1.2105830416822514,
      2.399065498071243,
      1.6944875882559933,
      1.3760678685063834,
      1.6294257968421753,
      2.1808680986557323,
      1.2629473429244245,
      2.0008930665768676
    ]
  },
  "artifacts": [
    "dataset.npz",
    "diagnostics.json",
    "diagnostics.csv",
    "README.md",
    "summary.json"
  ]
}
```

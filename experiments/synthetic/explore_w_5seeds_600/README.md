# explore_w_5seeds_600

Six variants × 5 seeds (`--seed` 0..4), `--max_test 200`, `--num_steps 600`, `--eval_every 150`.

## Variants

| Name | Description |
|------|-------------|
| `D3_snfix0.3_exploreW_inner20_lrW1_decay` | Phase A: inner 20, `lr_W=1.0×` base lr, freeze `m_j`/gate/`σ_n`; B: decay `lr_W`→0.1, inner→8; C: freeze W |
| `..._lrW0.5_decay` | Same with `lr_W_a=0.5` |
| `..._lrW0.5_trust` | + `lambda_W_trust=1` on `μ_B` |
| `..._lrW0.5_acceptCEM` | Every 20 steps reject W step if work-scale `RMSE_cem` worsens |
| `D3_snfix0.3_warm120_blend0.5_slowW5x` | Two-timescale baseline |
| `D3_snfix0.3_warm120_blend_to0.5_freeze_W` | Freeze W baseline |

## Logs & outputs

- `seed{s}.log` / `seed{s}.err` — stdout/stderr per seed
- `metrics_seed{s}/metrics.csv` — all variant rows

## Re-run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments/synthetic/run_explore_w_5seeds.ps1
```

## Implementation

See `test_remedy_d_structured_vi.py`: `explore_w_profile`, `_explore_w_profile_schedules`, `_train_variant_explore_w_profile`, `rho_blend_alpha_cap` in `_compute_elbo`.

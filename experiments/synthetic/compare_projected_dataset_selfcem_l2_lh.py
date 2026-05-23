"""Run projected-dataset self-CEM experiments on paper L2/LH synthetic data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    # Smoke check for the transductive projected-dataset CEM loop:
    python experiments/synthetic/compare_projected_dataset_selfcem_l2_lh.py ^
        --settings lh_q5_train1000 --seed 0 ^
        --variants pd_pls_selfcem,pd_lowrank_transductive,pd_lowrank_transductive_shuffled,pd_lowrank_random ^
        --max_test 6 --n_neighbors 8 --epochs 1 --inner_steps 1 ^
        --self_cem_updates 1 --self_cem_hyper_steps 1 --self_cem_gate_steps 1 ^
        --out_dir experiments/synthetic/projected_dataset_transductive_selfcem_smoke_20260517

    # LH seed0 screen:
    python experiments/synthetic/compare_projected_dataset_selfcem_l2_lh.py ^
        --settings lh_q5_train1000 --seed 0 ^
        --variants pd_raw_selfcem,pd_pca_selfcem,pd_pls_selfcem,pd_static_transductive,pd_lowrank_transductive,pd_lowrank_transductive_shuffled,pd_lowrank_random ^
        --max_test 200 --epochs 4 --inner_steps 20 ^
        --self_cem_updates 2 --self_cem_hyper_steps 10 --self_cem_gate_steps 20 ^
        --out_dir experiments/synthetic/projected_dataset_transductive_selfcem_lh_seed0_20260517

This is a clearer entrypoint for the whole-dataset ``z_i = W(x_i)x_i`` line.
It intentionally delegates to ``compare_global_z_supervised_jgp_l2_lh.py`` for
implementation compatibility; use the ``pd_*`` variants for projected-dataset
transductive self-CEM experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.synthetic.compare_global_z_supervised_jgp_l2_lh import main


if __name__ == "__main__":
    main()

# Active Learning Experiments

Synthetic DJGP/JGP active-learning experiments.

- `run_synth_al4jgp.py`: synthetic AL4JGP comparison; `run_al4jgp()` needs the external ActiveJGP checkout.
- `run_synth_and_plot_acq.py`: acquisition plotting and AL loop.
- `run_synth_and_plot_acq_v2.py`: later 4D-boundary acquisition/AL variant.

Root-level `run_synth_*.py` files are compatibility entrypoints.
Set `ACTIVE_JGP_DIR` or add ActiveJGP to `PYTHONPATH` before calling `run_al4jgp()`.

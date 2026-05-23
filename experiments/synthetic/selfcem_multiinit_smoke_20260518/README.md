# Self-CEM Multi-Initialization LH Screen

Runner: `experiments/synthetic/compare_selfcem_multiinit_lh.py`

This fixed-W screen treats W initializations as candidate geometries.  It uses
a raw-X candidate pool, reranks within that pool under each W, runs one batched
self-CEM head over all `(init, anchor)` pairs, and forms a likelihood-soft
Gaussian mixture by moment matching.

See `docs/experiments_log.md` for the dated result entry.

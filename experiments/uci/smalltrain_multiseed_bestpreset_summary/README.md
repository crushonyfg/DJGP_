# UCI Small-Train Best-Preset Summary

This folder consolidates the 5-seed train=200, train=500, and train=1000 UCI small-train results into single CSV files.

Files:
- `aggregate_bestpresets.csv`: latest aggregate metrics by train size, dataset, and method.
- `metrics_bestpresets.csv`: latest per-seed metric rows used to compute the aggregates.
- `aggregate_train200_500_bestpresets.csv` and `metrics_train200_500_bestpresets.csv`: the earlier 200/500-only snapshot.
- `aggregate_train200_500_1000_bestpresets.csv` and `metrics_train200_500_1000_bestpresets.csv`: explicit 200/500/1000 snapshot.
- `summary.json`: source artifact pointers.

Preset policy:
- Wine: random-row split, `std_x`, `n=25`.
- Appliances: random-row split, `std_x,std_y`, `n=35`; metrics are on the original target scale.
- Parkinsons: subject-grouped split. XGBoost/DKL rows are retained from the raw grouped comparison; LMJGP-family rows use the newer grouped `std_x` runs because the grouped scaling probe showed `std_x` is better than the old raw preset for LMJGP.

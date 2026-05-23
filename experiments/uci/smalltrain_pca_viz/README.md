# Small-Train PCA Prediction Visualizations

Artifacts from `experiments/uci/visualize_smalltrain_pca_predictions.py`.

Run command:

```powershell
python experiments/uci/visualize_smalltrain_pca_predictions.py `
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" `
  --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 `
  --neighbors "25,35" --steps "200,300,500" --xgb_boots 30 `
  --scaling_preset uci_empirical `
  --out_dir experiments/uci/smalltrain_pca_viz
```

Main files:

- `metrics.csv` - XGBoost plus supervised-LMJGP neighborhood/step sweep.
- `predictions.csv` - test PC1, truth, mean, sigma, and 90% range for all plotted methods/configs.
- `summary.json` - split/scaling metadata, PCA explained variance, metrics, and figure paths.
- `figures/*_pc1_response.png` - PC1 vs response scatter.
- `figures/*_pc1_predictions.png` - PC1-sorted test predictions and 90% ranges.

Headline from the one-seed 200-train run:

| dataset | XGBoost RMSE | best LMJGP config | best LMJGP RMSE | XGBoost CRPS | best LMJGP CRPS |
|---|---:|---|---:|---:|---:|
| Wine Quality | 0.728 | n=25, steps=300 | 0.751 | 0.433 | 0.420 |
| Parkinsons Telemonitoring | 5.558 | n=25, steps=300 | 7.773 | 3.393 | 4.399 |
| Appliances Energy Prediction | 103.260 | n=35, steps=300 | 105.065 | 41.900 | 40.622 |

In this seed, 300 LMJGP steps is best among 200/300/500 for all three datasets
under the tested neighborhoods. A smaller neighborhood (`n=25`) helps Wine and
Parkinsons, while Appliances prefers `n=35`.

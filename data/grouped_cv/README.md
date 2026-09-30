# Grouped cross-validation (Supplementary Section S11)

These files test how the Random Forest downscaling step transfers to unseen ERA5-Land cells.

## Files

- `GroupedCV_Samples_<Ta|RH>_<region>_20250715.csv`: the Random Forest training samples for 15 July 2025, exported from Google Earth Engine with `gee/Part15_export_grouped_cv_samples.js`. Each row holds the predictors, the ERA5-Land target, the coordinates (`lon`, `lat`) and the parent ERA5-Land cell (`era5_cell`, 0.1 degree grid).
- `Table_S11_grouped_cv_results.csv`: the results reported in Supplementary Table S11.

## Reproduce Table S11

```
cd python
python grouped_cv.py ../data/grouped_cv/GroupedCV_Samples_*.csv
```

The script refits the Random Forest in scikit-learn with the Earth Engine settings (200 trees, minimum leaf size 5, bag fraction 0.7, seed 7). It compares three five-fold splits: random, whole parent cells held out, and 0.3 degree blocks held out. Skill is 1 - RMSE/SD.

"""Grouped cross-validation of the Random Forest downscaling models (Supplementary Section S11).

Input: GroupedCV_Samples_<Ta|RH>_<region>.csv, exported by gee/Part15_export_grouped_cv_samples.js.
Each row is one Random Forest training sample with its predictors, the ERA5-Land target value,
its coordinates (lon, lat) and its parent ERA5-Land cell (era5_cell, 0.1 degree grid).

The Random Forest settings match Google Earth Engine:
200 trees, minimum leaf size 5, bag fraction 0.7, seed 7.

Three 5-fold schemes are compared:
  random : ordinary random split (samples from one ERA5-Land cell can fall in both sets)
  cell   : whole ERA5-Land parent cells are held out together
  block  : 0.3 degree blocks (3 x 3 parent cells) are held out together
Skill = 1 - RMSE/SD, where SD is the standard deviation of the target (the RMSE of a constant field).

usage: python grouped_cv.py ../data/grouped_cv/GroupedCV_Samples_*.csv
output: grouped_cv_results.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold

META = {'system:index', '.geo', 'lon', 'lat', 'era5_cell'}
FOLDS, SEED = 5, 7


def rf():
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_samples=0.7,
                                 bootstrap=True, random_state=SEED, n_jobs=-1)


def cv_rmse(X, y, splitter, groups=None):
    pred = np.zeros_like(y)
    for tr, te in splitter.split(X, y, groups):
        pred[te] = rf().fit(X[tr], y[tr]).predict(X[te])
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def run(path):
    df = pd.read_csv(path)
    target = 'Ta' if 'Ta' in df.columns else 'RH'
    cols = [c for c in df.columns if c not in META and c != target]
    df = df.dropna(subset=cols + [target])
    X, y = df[cols].to_numpy(float), df[target].to_numpy(float)
    cell = df['era5_cell'].to_numpy()
    block = (np.floor(df['lon'] / 0.3) * 1000 + np.floor(df['lat'] / 0.3)).to_numpy()
    sd = float(np.std(y))
    r = cv_rmse(X, y, KFold(FOLDS, shuffle=True, random_state=SEED))
    c = cv_rmse(X, y, GroupKFold(FOLDS), cell)
    b = cv_rmse(X, y, GroupKFold(FOLDS), block)
    return {'file': path.split('/')[-1], 'variable': target, 'samples': len(y),
            'cells': int(len(np.unique(cell))), 'sd': round(sd, 2),
            'rmse_random': round(r, 2), 'rmse_cell': round(c, 2), 'rmse_block': round(b, 2),
            'skill_random': round(1 - r / sd, 2), 'skill_cell': round(1 - c / sd, 2),
            'skill_block': round(1 - b / sd, 2)}


if __name__ == '__main__':
    out = pd.DataFrame([run(p) for p in sys.argv[1:]])
    print(out.to_string(index=False))
    out.to_csv('grouped_cv_results.csv', index=False)

"""
validate_field.py
Model versus field observations (EnviWear transects), at three stages of the workflow.

Usage
  python validate_field.py                          field_data/observations.csv
  python validate_field.py <observations.csv>

Observations CSV (one row per observation; see field_data/observations_template.csv)
  city, site, datetime_local, lat, lon, Ta, RH, WS, MRT, UTCI
  city is a city or field-site name from config.CITIES. datetime_local is local time.
  WS is pedestrian height (about 1.5 m) and is compared with WS_1p5m.

Each row is matched to the run of the same city whose local hour is nearest, within
MAX_MINUTES. For every matched row the script samples:
  stage "era5"  ERA5-Land value of the parent cell (ERA5_<city>_<tag>_coarse.tif)
  stage "rf"    raw Random Forest output at 10 m (Ta_10m_raw, RH_10m_raw)
  stage "final" corrected 10 m field (Ta_10m, RH_10m), and WS_1p5m, MRT_10m, UTCI_10m
  stage "utci"  the adjusted 10 m wind passed to UTCI (WS_utci), paired with the observed 1.5 m wind

Writes UTCI_outcome/
  validation_pairs.csv     every matched observation with model values for each stage
  validation_metrics.csv   n, bias (model minus observed), MAE, RMSE, r per city, variable, stage
All values are uncorrected model output.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

try:
    from config import DRIVE_FOLDER, RUNS, run_cities
except ImportError:
    DRIVE_FOLDER, RUNS = "UTCI_downscaling", [("2025-07-15", 14, ""), ("2025-12-31", 14, "")]
    run_cities = None

HERE = Path(__file__).parent
OBS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "field_data" / "observations.csv"
COMP_DIR = HERE / DRIVE_FOLDER
MAP_DIR = HERE / "UTCI_maps"
OUT_DIR = HERE / "UTCI_outcome"
MAX_MINUTES = 60
NODATA = -9999
# observed column -> (stage, file kind, band)
STAGES = {
    "Ta": [("era5", "coarse", "ERA5_Ta_C"), ("rf", "comp", "Ta_10m_raw"), ("final", "comp", "Ta_10m")],
    "RH": [("era5", "coarse", "ERA5_RH_percent"), ("rf", "comp", "RH_10m_raw"), ("final", "comp", "RH_10m")],
    "WS": [("final", "comp", "WS_1p5m"), ("utci", "comp", "WS_utci")],   # utci: 10 m wind fed to UTCI
    "MRT": [("final", "comp", "MRT_10m")],
}


def sample(path, band_name, lon, lat):
    with rasterio.open(path) as src:
        xs, ys = transform("EPSG:4326", src.crs, list(lon), list(lat))
        idx = list(src.descriptions).index(band_name) + 1
        vals = np.array([v[0] for v in src.sample(zip(xs, ys), indexes=idx)], dtype="float64")
    vals[(vals == NODATA)] = np.nan
    return vals


def match_run(row):
    """Tag of the run of this city nearest in time, or None."""
    best, best_dt = None, None
    for r in RUNS:
        date, hour = r[0], r[1]
        if run_cities is not None and row.city not in run_cities(r):
            continue
        dt = abs((row.datetime_local - (pd.Timestamp(date) + pd.Timedelta(hours=hour))).total_seconds())
        if dt <= MAX_MINUTES * 60 and (best_dt is None or dt < best_dt):
            best, best_dt = f"{date.replace('-', '')}_{hour:02d}00", dt
    return best


def metrics(g):
    d = g.model - g.observed
    return pd.Series(dict(n=len(g), bias=d.mean(), mae=d.abs().mean(),
                          rmse=np.sqrt((d ** 2).mean()),
                          r=g.model.corr(g.observed) if len(g) > 2 else np.nan))


def main():
    if not OBS.exists():
        sys.exit(f"No {OBS}. Copy field_data/observations_template.csv and fill it.")
    obs = pd.read_csv(OBS, parse_dates=["datetime_local"])
    obs["run"] = obs.apply(match_run, axis=1)
    skipped = obs[obs.run.isna()]
    if len(skipped):
        print(f"  {len(skipped)} row(s) match no run in config.RUNS, skipped")
    obs = obs.dropna(subset=["run"])

    pairs = []
    for (city, run), g in obs.groupby(["city", "run"]):
        comp = COMP_DIR / f"UTCI_{city}_{run}_components.tif"
        coarse = COMP_DIR / f"ERA5_{city}_{run}_coarse.tif"
        utci = MAP_DIR / f"UTCI_10m_{city}_{run}.tif"
        if not comp.exists():
            print(f"  missing {comp.name}, skipped")
            continue
        for col, stages in STAGES.items():
            if col not in g or not g[col].notna().any():
                continue
            h = g[g[col].notna()]
            for stage, kind, band in stages:
                path = comp if kind == "comp" else coarse
                if not path.exists():
                    print(f"  missing {path.name}, stage {stage} skipped")
                    continue
                pairs.append(pd.DataFrame(dict(
                    city=city, site=h.site, run=run, variable=col, stage=stage,
                    datetime_local=h.datetime_local, lat=h.lat, lon=h.lon,
                    observed=h[col].values, model=sample(path, band, h.lon, h.lat))))
        if "UTCI" in g and g.UTCI.notna().any() and utci.exists():
            h = g[g.UTCI.notna()]
            pairs.append(pd.DataFrame(dict(city=city, site=h.site, run=run, variable="UTCI", stage="final",
                                           datetime_local=h.datetime_local, lat=h.lat, lon=h.lon,
                                           observed=h.UTCI.values,
                                           model=sample(utci, "UTCI_10m", h.lon, h.lat))))
    if not pairs:
        sys.exit("No observation matched a downloaded run.")
    p = pd.concat(pairs).dropna(subset=["model", "observed"])
    OUT_DIR.mkdir(exist_ok=True)
    p.to_csv(OUT_DIR / "validation_pairs.csv", index=False)
    m = (p.groupby(["city", "site", "run", "variable", "stage"])
         .apply(metrics, include_groups=False).reset_index())
    m.round(3).to_csv(OUT_DIR / "validation_metrics.csv", index=False)
    print(m.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

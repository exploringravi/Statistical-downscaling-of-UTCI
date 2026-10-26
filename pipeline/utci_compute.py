"""
utci_compute.py
Splits the exported components into map files and checks them.

Usage
  python utci_compute.py                       reads ./UTCI_downscaling
  python utci_compute.py <path to UTCI_downscaling>

Reads   UTCI_<City>_<YYYYMMDD>_<HHMM>_components.tif
Writes  UTCI_maps/ next to the input folder:
  UTCI_10m_<City>_<YYYYMMDD>_<HHMM>.tif     float32 degC, NaN nodata, wind set by UTCI_WIND (10 m)
  UTCI_ws1p5_<City>_<YYYYMMDD>_<HHMM>.tif   UTCI with 1.5 m wind, as the JS and v1.0.0
  UTCI_class_<City>_<YYYYMMDD>_<HHMM>.tif   uint8 class 1 (extreme cold) to 10 (extreme heat), 0 nodata
  Ta_10m_, RH_10m_, WS_10m_, WS_1p5m_, WS_utci_, MRT_10m_, dTmrt_10m_, Pa_10m_<City>_<...>.tif
                                            one file per variable, NaN nodata
  utci_qa.csv                               per city and run:
      coverage_pct          share of the city boundary with a UTCI value
      thermofeel_max_diff   largest |UTCI from Earth Engine - UTCI recomputed with thermofeel|
                            from the same Ta, MRT, wind and vapour pressure (should be ~0)
      *_at_clamp_pct        share of pixels sitting on a clamp limit of the JS method
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from thermofeel.thermofeel import calculate_utci_polynomial

try:
    from config import DRIVE_FOLDER
except ImportError:
    DRIVE_FOLDER = "UTCI_downscaling"

HERE = Path(__file__).parent
IN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / DRIVE_FOLDER
OUT_DIR = IN_DIR.parent / "UTCI_maps"
NODATA = -9999
PATTERN = re.compile(r"UTCI_(?P<city>[A-Za-z0-9]+)_(?P<tag>\d{8}_\d{4})_components\.tif$")

# Stress categories (Brode et al. 2012), lower bound inclusive, degC.
CLASSES = [
    (1, "extreme cold stress", -np.inf, -40),
    (2, "very strong cold stress", -40, -27),
    (3, "strong cold stress", -27, -13),
    (4, "moderate cold stress", -13, 0),
    (5, "slight cold stress", 0, 9),
    (6, "no thermal stress", 9, 26),
    (7, "moderate heat stress", 26, 32),
    (8, "strong heat stress", 32, 38),
    (9, "very strong heat stress", 38, 46),
    (10, "extreme heat stress", 46, np.inf),
]
try:
    from config import RUNS
    from utci_method import PROFILES
except ImportError:
    RUNS, PROFILES = [], {}


def clamps_for(tag):
    """Clamp limits of the profile used for this run (date and hour from the tag)."""
    prof = next((PROFILES.get(r[2]) for r in RUNS if f"{r[0].replace('-', '')}_{r[1]:02d}00" == tag), None)
    prof = prof or PROFILES.get("summer") or dict(ta_clamp=(20, 45), mrt_clamp=(15, 85))
    out = {"WS_utci": (0.5, 17), "UTCI_10m": (-50, 70)}
    for band, key in [("Ta_10m", "ta_clamp"), ("MRT_10m", "mrt_clamp")]:
        if isinstance(prof.get(key), tuple):
            out[band] = prof[key]
    return out


# Wind used for the UTCI_10m and UTCI_class files.
#   "10m"  : WS_local_10m, as UTCI is defined (Brode et al. 2012) and as the paper states (Eq. 11)
#   "1p5m" : WS_utci from Earth Engine = pedestrian wind, as main_workflow.js and the v1.0.0 maps
# UTCI_ws1p5_*.tif is always written too, for the comparison with v1.0.0.
UTCI_WIND = "10m"


def classify(u):
    out = np.zeros(u.shape, dtype="uint8")
    for code, _, lo, hi in CLASSES:
        out[(u >= lo) & (u < hi)] = code
    return out


# component band -> file name prefix (as main_workflow.js exportImage names)
SPLIT = {"Ta_10m": "Ta_10m", "RH_10m": "RH_10m", "WS_local_10m": "WS_10m", "WS_1p5m": "WS_1p5m",
         "WS_utci": "WS_utci", "MRT_10m": "MRT_10m", "dTmrt_10m": "dTmrt_10m", "Pa_10m": "Pa_10m"}


def read_bands(path):
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        data = {n: src.read(i + 1).astype("float64") for i, n in enumerate(names)}
        profile = src.profile
    for a in data.values():
        a[a == NODATA] = np.nan
    return data, profile


def write(path, arr, profile, dtype, nodata, name):
    p = profile.copy()
    p.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)
        dst.set_band_description(1, name)


def join_tiles():
    """Join UTCI_<City>_<tag>_components_t0..tN.tif (tiled exports) into one components file."""
    from collections import defaultdict
    import rasterio.transform
    groups = defaultdict(list)
    for f in IN_DIR.glob("*_components_t*.tif"):
        groups[f.name.split("_components_t")[0]].append(f)
    for stem, parts in groups.items():
        out = IN_DIR / f"{stem}_components.tif"
        if out.exists() and out.stat().st_mtime > max(p.stat().st_mtime for p in parts):
            continue
        # Manual mosaic: Earth Engine writes NaN (not -9999) on tile edges, and rasterio.merge
        # treats NaN as valid data, which left a one-row gap at a tile seam. Fill only where
        # the mosaic is still empty and the tile has a finite, non-nodata value.
        srcs = [rasterio.open(p) for p in sorted(parts)]
        res = srcs[0].res[0]
        left = min(s_.bounds.left for s_ in srcs); top = max(s_.bounds.top for s_ in srcs)
        right = max(s_.bounds.right for s_ in srcs); bottom = min(s_.bounds.bottom for s_ in srcs)
        h, w = int(round((top - bottom) / res)), int(round((right - left) / res))
        arr = np.full((srcs[0].count, h, w), np.nan, dtype="float32")
        for s_ in srcs:
            r0, c0 = int(round((top - s_.bounds.top) / res)), int(round((s_.bounds.left - left) / res))
            t = s_.read().astype("float32")
            t[t == NODATA] = np.nan
            win = arr[:, r0:r0 + s_.height, c0:c0 + s_.width]
            empty = np.isnan(win) & ~np.isnan(t)
            win[empty] = t[empty]
        arr[np.isnan(arr)] = NODATA
        transform = rasterio.transform.from_origin(left, top, res, res)
        prof = srcs[0].profile.copy()
        prof.update(height=h, width=w, transform=transform, dtype="float32", nodata=NODATA,
                    compress="deflate", BIGTIFF="IF_SAFER")
        names = srcs[0].descriptions
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(arr)
            for i, nm in enumerate(names, 1):
                dst.set_band_description(i, nm)
        for s_ in srcs:
            s_.close()
        print(f"  joined {len(parts)} tiles into {out.name}")


def main():
    join_tiles()
    files = sorted(f for f in IN_DIR.glob("*_components.tif") if PATTERN.search(f.name))
    if not files:
        sys.exit(f"No *_components.tif in {IN_DIR}. Run download_drive.py first.")
    OUT_DIR.mkdir(exist_ok=True)
    qa = []
    for f in files:
        m = PATTERN.search(f.name)
        city, tag = m["city"], m["tag"]
        b, profile = read_bands(f)
        u_ee = b["UTCI_10m"]                 # UTCI from Earth Engine, wind as in the JS (WS_utci)
        valid = ~np.isnan(u_ee)
        ta, rh = b["Ta_10m"][valid], b["RH_10m"][valid]
        pa = np.clip(0.1 * 6.112 * np.exp(17.62 * ta / (243.12 + ta)) * rh / 100, 0, 5)
        d = np.clip(b["MRT_10m"][valid] - ta, -30, 70)
        u_check = np.clip(calculate_utci_polynomial(ta, ta + d, b["WS_utci"][valid], pa), -50, 70)
        u10 = np.full(u_ee.shape, np.nan)       # UTCI with 10 m wind, as the UTCI definition and Eq. 11
        u10[valid] = np.clip(calculate_utci_polynomial(
            ta, ta + d, np.clip(b["WS_local_10m"][valid], 0.5, 17), pa), -50, 70)
        u = u10 if UTCI_WIND == "10m" else u_ee
        cls = np.where(valid, classify(np.nan_to_num(u, nan=0)), 0)
        write(OUT_DIR / f"UTCI_10m_{city}_{tag}.tif", u, profile, "float32", np.nan, "UTCI_10m")
        write(OUT_DIR / f"UTCI_class_{city}_{tag}.tif", cls, profile, "uint8", 0, "UTCI_class")
        write(OUT_DIR / f"UTCI_ws1p5_{city}_{tag}.tif", u_ee, profile, "float32", np.nan, "UTCI_ws1p5")
        # one file per variable, same names as the JS exports
        for band, short in SPLIT.items():
            if band in b:
                write(OUT_DIR / f"{short}_{city}_{tag}.tif", b[band], profile, "float32", np.nan, short)

        inside = b["in_city"] == 1
        coverage = 100 * float((valid & inside).sum()) / max(int(inside.sum()), 1)
        row = dict(city=city, run=tag, wind=UTCI_WIND, pixels=int(valid.sum()),
                   boundary_pixels=int(inside.sum()), coverage_pct=coverage,
                   thermofeel_max_diff=float(np.abs(u_check - u_ee[valid]).max()),
                   utci_mean=float(u[valid].mean()),
                   utci_ws1p5_mean=float(u_ee[valid].mean()),
                   utci_10m_minus_ws1p5_mean=float((u10[valid] - u_ee[valid]).mean()))
        for band, (lo, hi) in clamps_for(tag).items():
            v = b[band][valid]
            row[f"{band}_at_clamp_pct"] = 100 * float((np.isclose(v, lo, atol=1e-3) | np.isclose(v, hi, atol=1e-3)).mean())
        qa.append(row)
        msg = (f"  {city} {tag}: UTCI mean {row['utci_mean']:.2f} degC ({UTCI_WIND} wind; "
               f"1.5 m wind gives {row['utci_ws1p5_mean']:.2f}), coverage {coverage:.1f}%")
        if coverage < 99:
            msg += "  WARNING: map does not cover the whole boundary"
        if row.get("Ta_10m_at_clamp_pct", 0) > 1:
            msg += f"  WARNING: {row['Ta_10m_at_clamp_pct']:.1f}% of Ta on a clamp limit"
        print(msg)

    pd.DataFrame(qa).round(4).to_csv(OUT_DIR / "utci_qa.csv", index=False)
    print(f"Written to {OUT_DIR}")


if __name__ == "__main__":
    main()

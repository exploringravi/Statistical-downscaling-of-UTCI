"""
utci_outcome.py
Statistics, maps and a reproduction check against the published v1.0.0 rasters.

Usage
  python utci_outcome.py                 reads ./UTCI_maps and ./UTCI_downscaling
  python utci_outcome.py <UTCI_maps dir>

Writes UTCI_outcome/
  utci_stats.csv            per city and run: mean, SD, percentiles, class shares (%)
  reference_check.csv       new run minus published v1.0.0 raster, same city and run
  <City>_<run>_utci.png     continuous UTCI and stress classes
  <City>_<run>_ta.png       ERA5-Land Ta (coarse) vs downscaled Ta (10 m)
  <City>_rf_importance.png  top predictors for Ta and RH (first run found)
  <City>_<run>_<MRT_10m|WS_10m|Ta_10m|RH_10m>.png   one map per variable
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.plot import plotting_extent
from rasterio.warp import Resampling, reproject

from utci_compute import CLASSES

try:
    from config import DRIVE_FOLDER, MACRO_TAG
except ImportError:
    DRIVE_FOLDER = "UTCI_downscaling"
    MACRO_TAG = {"Limassol": "Cyprus", "Perugia": "Umbria", "Aachen": "Koln"}

HERE = Path(__file__).parent
MAP_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "UTCI_maps"
COMP_DIR = MAP_DIR.parent / DRIVE_FOLDER
REF_DIR = HERE / "reference" / "v1.0.0_rasters"
OUT_DIR = MAP_DIR.parent / "UTCI_outcome"
PATTERN = re.compile(r"UTCI_10m_(?P<city>[A-Za-z0-9]+)_(?P<tag>\d{8}_\d{4})\.tif$")
NODATA = -9999
TOP_N = 12

CLASS_COLORS = ["#08306b", "#2171b5", "#4292c6", "#9ecae1", "#deebf7",
                "#a1d99b", "#fdae6b", "#f16913", "#d7301f", "#7f0000"]


def read(path, band=1):
    with rasterio.open(path) as src:
        a = src.read(band).astype("float64")
        a[a == NODATA] = np.nan
        return a, src


def stats_row(city, tag, u):
    v = u[~np.isnan(u)]
    row = dict(city=city, run=tag, pixels=v.size, mean=v.mean(), sd=v.std(ddof=1),
               p5=np.percentile(v, 5), p50=np.median(v), p95=np.percentile(v, 95),
               min=v.min(), max=v.max())
    for code, label, lo, hi in CLASSES:
        row[f"pct_{label.replace(' ', '_')}"] = 100 * float(((v >= lo) & (v < hi)).mean())
    return row


def reference_check(city, tag, path):
    ref = REF_DIR / f"UTCI_10m_{city}_{MACRO_TAG[city]}_{tag}.tif"
    if not ref.exists():
        return None
    # same method as v1.0.0 (1.5 m wind) when available, so this tests the boundary fix only
    same = path.with_name(path.name.replace("UTCI_10m_", "UTCI_ws1p5_"))
    if same.exists():
        path = same
    with rasterio.open(path) as new, rasterio.open(ref) as old:
        a = new.read(1).astype("float64")
        b = np.full(a.shape, np.nan)
        reproject(old.read(1).astype("float64"), b, src_transform=old.transform, src_crs=old.crs,
                  dst_transform=new.transform, dst_crs=new.crs, resampling=Resampling.nearest,
                  src_nodata=np.nan, dst_nodata=np.nan)
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = a[ok] - b[ok]
    return dict(city=city, run=tag, reference=ref.name, overlap_pixels=int(ok.sum()),
                new_only_pixels=int((~np.isnan(a) & np.isnan(b)).sum()),
                v1_only_pixels=int((np.isnan(a) & ~np.isnan(b)).sum()),
                mean_diff=d.mean(), max_abs_diff=np.abs(d).max(), rmse=np.sqrt((d ** 2).mean()),
                r=np.corrcoef(a[ok], b[ok])[0, 1],
                reproduced=bool(np.sqrt((d ** 2).mean()) < 0.05))


def map_utci(city, tag, path):
    u, src = read(path)
    with rasterio.open(path) as s:
        ext = plotting_extent(s)
    cls_path = path.with_name(path.name.replace("UTCI_10m_", "UTCI_class_"))
    with rasterio.open(cls_path) as s:
        c = s.read(1).astype("float64")
    c[c == 0] = np.nan
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")
    im = ax[0].imshow(u, extent=ext, cmap="RdYlBu_r")
    fig.colorbar(im, ax=ax[0], shrink=0.8, label="UTCI (degC)")
    ax[0].set_title(f"{city} {tag}: UTCI 10 m")
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 11.5, 1), cmap.N)
    im = ax[1].imshow(c, extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax[1], shrink=0.8, ticks=range(1, 11))
    cb.ax.set_yticklabels([lbl for _, lbl, _, _ in CLASSES], fontsize=7)
    ax[1].set_title("Thermal stress class")
    for a in ax:
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{city}_{tag}_utci.png", dpi=300, facecolor="white")
    plt.close(fig)


def map_ta(city, tag):
    comp = COMP_DIR / f"UTCI_{city}_{tag}_components.tif"
    coarse = COMP_DIR / f"ERA5_{city}_{tag}_coarse.tif"
    if not (comp.exists() and coarse.exists()):
        return
    with rasterio.open(comp) as s:
        ext = plotting_extent(s)
        ta = s.read(list(s.descriptions).index("Ta_10m") + 1).astype("float64")
    with rasterio.open(coarse) as s:
        ext5 = plotting_extent(s)
        ta5 = s.read(1).astype("float64")
    for a in (ta, ta5):
        a[a == NODATA] = np.nan
    lo, hi = np.nanpercentile(np.r_[ta.ravel(), ta5.ravel()], [1, 99])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")
    for a, img, e, title in [(ax[0], ta5, ext5, "ERA5-Land Ta (9 km)"), (ax[1], ta, ext, "Downscaled Ta (10 m)")]:
        im = a.imshow(img, extent=e, cmap="RdYlBu_r", vmin=lo, vmax=hi, interpolation="nearest")
        a.set_xlim(ext[0], ext[1]), a.set_ylim(ext[2], ext[3])
        a.set_title(f"{city} {tag}: {title}")
        a.set_xticks([]), a.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="Ta (degC)")
    fig.savefig(OUT_DIR / f"{city}_{tag}_ta.png", dpi=300, facecolor="white")
    plt.close(fig)


VAR_MAPS = {  # file prefix in UTCI_maps: (title, unit, colormap)
    "MRT_10m": ("Mean radiant temperature", "degC", "magma"),
    "WS_10m": ("Wind speed at 10 m", "m/s", "viridis"),
    "Ta_10m": ("Air temperature", "degC", "RdYlBu_r"),
    "RH_10m": ("Relative humidity", "%", "BrBG"),
}


def map_vars(city, tag):
    """One PNG per variable: MRT, wind, Ta, RH, with mean, 5th to 95th percentile and max."""
    for prefix, (title, unit, cmap) in VAR_MAPS.items():
        f = MAP_DIR / f"{prefix}_{city}_{tag}.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as s:
            a = s.read(1).astype("float64")
            ext = plotting_extent(s)
        v = a[~np.isnan(a)]
        if not v.size:
            continue
        lo, hi = np.percentile(v, [1, 99])
        fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="white")
        im = ax.imshow(a, extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.85, label=f"{title} ({unit})")
        p5, p95 = np.percentile(v, [5, 95])
        ax.set_title(f"{city} {tag}: {title}\nmean {v.mean():.1f}, P5 to P95 {p5:.1f} to {p95:.1f}, "
                     f"max {v.max():.1f} {unit}", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{city}_{tag}_{prefix}.png", dpi=300, facecolor="white")
        plt.close(fig)


def importance(city):
    files = sorted(COMP_DIR.glob(f"UTCI_{city}_*_rf.csv"))
    if not files:
        return
    df = pd.read_csv(files[0])
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
    for a, t in zip(ax, ["Ta", "RH"]):
        d = df[df.target == t].nlargest(TOP_N, "importance").iloc[::-1]
        a.barh(d.predictor, d.importance, color="#4d4d4d")
        oob = d.oob_rmse.iloc[0] if len(d) else np.nan
        a.set_title(f"{city} {t}: impurity-based importance (OOB RMSE {oob:.2f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{city}_rf_importance.png", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    files = sorted(f for f in MAP_DIR.glob("UTCI_10m_*.tif") if PATTERN.search(f.name))
    if not files:
        sys.exit(f"No UTCI_10m_*.tif in {MAP_DIR}. Run utci_compute.py first.")
    OUT_DIR.mkdir(exist_ok=True)
    stats, checks, cities = [], [], set()
    for f in files:
        m = PATTERN.search(f.name)
        city, tag = m["city"], m["tag"]
        cities.add(city)
        u, _ = read(f)
        stats.append(dict(stats_row(city, tag, u), source="rerun"))
        chk = reference_check(city, tag, f)
        if chk:
            checks.append(chk)
            print(f"  {city} {tag}: vs v1.0.0 RMSE {chk['rmse']:.3f} degC, "
                  f"{'reproduced' if chk['reproduced'] else 'DIFFERS'}")
        map_utci(city, tag, f)
        map_ta(city, tag)
        map_vars(city, tag)
    # cities not rerun: statistics from the published v1.0.0 rasters
    ref_pat = re.compile(r"UTCI_10m_(?P<city>[A-Za-z0-9]+)_[A-Za-z]+_(?P<tag>\d{8}_\d{4})\.tif$")
    for f in sorted(REF_DIR.glob("UTCI_10m_*.tif")):
        m = ref_pat.search(f.name)
        if m and m["city"] not in cities:
            u, _ = read(f)
            stats.append(dict(stats_row(m["city"], m["tag"], u), source="v1.0.0"))
    for city in sorted(cities):
        importance(city)
    pd.DataFrame(stats).round(3).to_csv(OUT_DIR / "utci_stats.csv", index=False)
    if checks:
        pd.DataFrame(checks).round(4).to_csv(OUT_DIR / "reference_check.csv", index=False)
    print(f"Written to {OUT_DIR}")


if __name__ == "__main__":
    main()

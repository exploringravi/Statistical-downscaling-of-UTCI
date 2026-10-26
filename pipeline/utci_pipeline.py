"""
utci_pipeline.py
10 m UTCI for Limassol, Perugia and Aachen, for every run in config.RUNS.
Earth Engine Python API. The science is in utci_method.py (port of gee/main_workflow.js);
this file submits the exports, waits and resumes.

Per city and run, to Drive folder UTCI_downscaling:
  UTCI_<City>_<YYYYMMDD>_<HHMM>_components.tif   UTCI_10m, UTCI_class, Ta_10m, RH_10m,
      WS_local_10m, WS_1p5m, WS_utci, MRT_10m, dTmrt_10m, Pa_10m, Ta_10m_raw, RH_10m_raw,
      in_city (1 inside the boundary); nodata -9999
  ERA5_<City>_<YYYYMMDD>_<HHMM>_coarse.tif       ERA5-Land inputs at 9 km
  UTCI_<City>_<YYYYMMDD>_<HHMM>_rf.csv           RF importance and OOB RMSE
  GroupedCV_Samples_<Ta|RH>_<Region>_<YYYYMMDD>_<HHMM>.csv   for tools/grouped_cv.py
Then run download_drive.py and utci_compute.py.

Resumable: finished exports are skipped, running ones are waited on.

Run
  python utci_pipeline.py                  all cities
  python utci_pipeline.py Limassol         selected cities
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import ee

import config
import utci_method as M
from config import (BOUNDARIES, CITIES, CRS, DRIVE_FOLDER, MACRO_REGIONS, MACRO_TAG,
                    PROJECT, RUNS, TIMEZONE, USER, run_cities)

POLL_SECONDS = 60
NODATA = -9999
DONE_LOG = Path(__file__).with_name("utci_pipeline_done.json")


# ============================================================== task handling
def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}", flush=True)


def load_done():
    return set(json.loads(DONE_LOG.read_text())) if DONE_LOG.exists() else set()


def mark_done(desc):
    done = load_done()
    done.add(desc)
    DONE_LOG.write_text(json.dumps(sorted(done), indent=1))


def task_states(description):
    """States of all Earth Engine tasks with this description (most recent first)."""
    return [(t.get("state"), t["id"]) for t in ee.data.getTaskList()
            if t.get("description") == description]


def submit(export_fn, description, **kwargs):
    if description in load_done():
        log(f"skip, done: {description}")
        return None
    states = task_states(description)
    # finished while this script was not watching (laptop offline, script stopped)
    if any(st in ("COMPLETED", "SUCCEEDED") for st, _ in states):
        mark_done(description)
        log(f"skip, finished earlier: {description}")
        return None
    running = [tid for st, tid in states if st in ("READY", "RUNNING")]
    if running:
        log(f"already running, waiting: {description}")
        return description, running[0]
    task = export_fn(description=description, **kwargs)
    task.start()
    log(f"submitted: {description}")
    return description, task.id


def wait(tasks):
    pending = dict(t for t in tasks if t)
    failed = []
    while pending:
        for desc, tid in list(pending.items()):
            try:
                s = ee.data.getTaskStatus(tid)[0]
            except Exception as e:  # laptop offline or a network hiccup: tasks keep running in EE
                log(f"no connection ({type(e).__name__}), retrying in {POLL_SECONDS} s")
                break
            state = s.get("state")
            if state in ("COMPLETED", "SUCCEEDED"):
                log(f"finished: {desc}")
                mark_done(desc)
                pending.pop(desc)
            elif state in ("FAILED", "CANCELLED", "CANCEL_REQUESTED"):
                log(f"FAILED: {desc}: {s.get('error_message', state)}")
                failed.append(desc)
                pending.pop(desc)
        if pending:
            time.sleep(POLL_SECONDS)
    if failed:
        raise RuntimeError(f"{len(failed)} task(s) failed: {failed}. Fix and rerun; finished work is kept.")


# ============================================================== cities and runs
def utc_hour(city, date, hour_local):
    """Local date and hour to UTC, with the city's daylight saving rule."""
    local = datetime.fromisoformat(date).replace(hour=hour_local, tzinfo=ZoneInfo(TIMEZONE[city]))
    return local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S")


def run_tag(date, hour_local):
    return f"{date.replace('-', '')}_{hour_local:02d}00"


def macro_fc(city):
    spec = MACRO_REGIONS[city]
    if isinstance(spec, str):
        return ee.FeatureCollection(spec)
    fc_id, prop, value = spec
    return ee.FeatureCollection(fc_id).filter(ee.Filter.eq(prop, value))


def boundary_fc(name):
    """City boundary asset, or a lon/lat box ("bbox", lon0, lat0, lon1, lat1) for field sites."""
    spec = BOUNDARIES[name]
    if isinstance(spec, tuple) and spec[0] == "bbox":
        return ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle(list(spec[1:]), "EPSG:4326", False))])
    return ee.FeatureCollection(spec if spec.startswith(("users/", "projects/")) else USER + spec)


class City:
    def __init__(self, name):
        self.name = name
        self.fc = boundary_fc(name)
        self.macro = macro_fc(name)
        self.crs = CRS[name]
        self.gba = M.gba_tiles(self.macro.geometry())
        if not self.gba:
            raise RuntimeError(f"{name}: no Global Building Atlas tile found for the macro-region")


# ============================================================== run
def asset_exists(asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.EEException:
        return False


# Bump when the method changes: task names get this suffix so earlier finished tasks are not
# taken as done. File names in Drive stay the same.
RUN_VERSION = "v10"   # v10: v9 plus longwave reflected by ground, walls and trees (isothermal check exact) and a sunlit band. v9: MRT_SCHEME "cylinder" (cast shadows, two-hemisphere person); Ta, RH, wind as v8. Previous: v8   # v8: corrections calibrated (utci_method.CAL) and anomaly scaled by weather (ANOM_WEATHER). v6: v6: dist_to_water_m fixed, MODIS LST projection fixed, new account, field sites added


SAMPLES_VERSION = "v4"   # v4: dist_to_water_m in metres at any scale; MODIS LST at its native 1 km grid


# Cities whose 10 m map is exported as an n x n grid of tiles (joined again in utci_compute.py).
# Perugia (446 km2) stalled for 7+ hours as one export.
EXPORT_TILES = {"Perugia": 2}


def is_done(name):
    """Finished according to the local log or to Earth Engine's task list."""
    d = desc(name)
    if d in load_done():
        return True
    if any(st in ("COMPLETED", "SUCCEEDED") for st, _ in task_states(d)):
        mark_done(d)
        return True
    return False


# Tiles that are still too slow are split again into m x m sub-tiles (names _t<k>_<j>).
# Perugia tile 3 (north-east) ran 1.5 h+ while the other three took 30 to 46 min.
SPLIT_TILES = {"Perugia": {3: 2}}


def _grid(x0, y0, x1, y1, n):
    dx, dy = (x1 - x0) / n, (y1 - y0) / n
    return [(x0 + i * dx, y0 + j * dy, x0 + (i + 1) * dx, y0 + (j + 1) * dy)
            for j in range(n) for i in range(n)]


def tile_boxes(c, n):
    """{tile name suffix: (x0, y0, x1, y1)} covering the city bounds, in the city's UTM metres."""
    xy = c.fc.geometry().bounds(1, c.crs).coordinates().get(0).getInfo()
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    out = {}
    for k, box in enumerate(_grid(min(xs), min(ys), max(xs), max(ys), n)):
        m = SPLIT_TILES.get(c.name, {}).get(k)
        if m:
            for j, sub in enumerate(_grid(*box, m)):
                out[f"t{k}_{j}"] = sub
        else:
            out[f"t{k}"] = box
    return out


def rect(c, box, pad=20):
    x0, y0, x1, y1 = box
    return ee.Geometry.Rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], c.crs, False)


def components_done(c, tag):
    name = f"UTCI_{c.name}_{tag}_components"
    if is_done(name):
        return True
    n = EXPORT_TILES.get(c.name, 1)
    if n == 1:
        return False
    for k in range(n * n):
        m = SPLIT_TILES.get(c.name, {}).get(k)
        if is_done(f"{name}_t{k}"):
            continue
        if m and all(is_done(f"{name}_t{k}_{j}") for j in range(m * m)):
            continue
        return False
    return True


WRITE_ROOT = "projects/cityai-512212/assets/"   # the running account can write only here
_SID = {}


def samples_id(city, target, tag):
    """Existing samples are reused where they are; missing ones are written to WRITE_ROOT."""
    key = (city, target, tag)
    if key not in _SID:
        name = f"utci_samples_{SAMPLES_VERSION}_{city}_{target}_{tag}"
        root = getattr(config, "SAMPLE_ROOT", {}).get(city, USER)
        aid = f"{root}{name}"
        if not asset_exists(aid) and root != WRITE_ROOT:
            aid = f"{WRITE_ROOT}{name}"
        _SID[key] = aid
    return _SID[key]


def desc(name):
    return f"{name}_{RUN_VERSION}"


def run(cities):
    # Step 1: training samples to assets. Training on stored samples instead of sampling inside
    # the map export keeps each export within Earth Engine memory (Aachen ran out of memory).
    # Limassol 15 July / 31 Dec were made in one step and are skipped if already finished.
    M.SAMPLE_TILESCALE = 16
    jobs, tasks = [], []
    for c in cities:
        log(f"{c.name}: GBA tiles {[t.rsplit('/', 1)[1] for t in c.gba]}")
        for r in RUNS:
            date, hour, profile = r[:3]
            if c.name not in run_cities(r):
                continue
            tag = run_tag(date, hour)
            utc = utc_hour(c.name, date, hour)
            if components_done(c, tag) and is_done(f"UTCI_{c.name}_{tag}_rf"):
                log(f"skip, done: {c.name} {tag}")
                continue
            log(f"{c.name} {tag} ({profile}): local {hour:02d}:00 = {utc} UTC")
            try:
                _, _, s_ta, s_rh, _ = M.build(c.fc, c.macro, c.crs, date, utc, profile, c.gba)
            except NotImplementedError as e:
                log(f"SKIPPED {c.name} {tag}: {e}")
                continue
            for t, fc in [("Ta", s_ta), ("RH", s_rh)]:
                aid = samples_id(c.name, t, tag)
                if asset_exists(aid):
                    log(f"skip, exists: {aid}")
                else:
                    tasks.append(submit(ee.batch.Export.table.toAsset, desc(f"utci_samples_{c.name}_{t}_{tag}"),
                                        collection=fc, assetId=aid))
            jobs.append((c, date, hour, profile, tag, utc))
    wait(tasks)

    # Step 2: train on the stored samples, map, export
    tasks = []
    for c, date, hour, profile, tag, utc in jobs:
        train = {t: ee.FeatureCollection(samples_id(c.name, t, tag)) for t in ("Ta", "RH")}
        stack, era5, s_ta, s_rh, rf = M.build(c.fc, c.macro, c.crs, date, utc, profile, c.gba, train)
        region = c.fc.geometry()
        name = f"UTCI_{c.name}_{tag}_components"
        n = EXPORT_TILES.get(c.name, 1)
        if components_done(c, tag):
            log(f"skip, done: {name}")
        elif n == 1:
            tasks.append(submit(ee.batch.Export.image.toDrive, desc(name), image=stack, region=region,
                                crs=c.crs, scale=M.OUT_SCALE, folder=DRIVE_FOLDER, fileNamePrefix=name,
                                maxPixels=int(1e13), fileFormat="GeoTIFF",
                                formatOptions={"noData": NODATA}))
        else:
            # Same 10 m grid for every tile (crsTransform), 20 m overlap; joined in utci_compute.py
            for suffix, box in tile_boxes(c, n).items():
                tname = f"{name}_{suffix}"
                k = int(suffix[1:].split("_")[0])
                if "_" in suffix and is_done(f"{name}_t{k}"):
                    continue                     # the whole tile already finished before splitting
                tasks.append(submit(ee.batch.Export.image.toDrive, desc(tname), image=stack,
                                    region=rect(c, box), crs=c.crs,
                                    crsTransform=[M.OUT_SCALE, 0, 0, 0, -M.OUT_SCALE, 0],
                                    folder=DRIVE_FOLDER, fileNamePrefix=tname, maxPixels=int(1e13),
                                    fileFormat="GeoTIFF", formatOptions={"noData": NODATA}))
        name = f"ERA5_{c.name}_{tag}_coarse"
        tasks.append(submit(ee.batch.Export.image.toDrive, desc(name), image=era5, region=region,
                            crs=c.crs, scale=9000, folder=DRIVE_FOLDER, fileNamePrefix=name,
                            maxPixels=int(1e13)))
        name = f"UTCI_{c.name}_{tag}_rf"
        tasks.append(submit(ee.batch.Export.table.toDrive, desc(name), collection=rf,
                            folder=DRIVE_FOLDER, fileNamePrefix=name, fileFormat="CSV"))
        for t in ("Ta", "RH"):
            name = f"GroupedCV_Samples_{t}_{MACRO_TAG[c.name]}_{tag}"
            tasks.append(submit(ee.batch.Export.table.toDrive, desc(name), collection=train[t],
                                folder=DRIVE_FOLDER, fileNamePrefix=name, fileFormat="CSV"))
    wait(tasks)


def main():
    names = sys.argv[1:] or CITIES
    unknown = set(names) - set(CITIES)
    if unknown:
        sys.exit(f"Unknown city: {sorted(unknown)}. Options: {CITIES}")
    ee.Initialize(project=PROJECT)
    run([City(n) for n in names])
    log("All done. Run download_drive.py, then utci_compute.py.")


if __name__ == "__main__":
    main()

"""
check_setup.py
Run before the pipeline. Checks login, boundaries, macro-regions, Building Atlas tiles,
datasets, profiles, the UTC hour of each run, and ERA5-Land coverage of each city.
"""

import sys

import ee

import utci_method as M
from config import BOUNDARIES, CITIES, PROJECT, RUNS, USER, run_cities
from utci_pipeline import boundary_fc, macro_fc, utc_hour

problems = []


def ok(msg):
    print(f"  OK    {msg}")


def bad(msg):
    print(f"  FAIL  {msg}")
    problems.append(msg)


def warn(msg):
    print(f"  WARN  {msg}")


def main():
    print("Profiles")
    for r in RUNS:
        date, hour, profile = r[:3]
        if M.PROFILES.get(profile):
            ok(f"{date} {hour:02d}:00 uses '{profile}'")
        else:
            warn(f"{date} {hour:02d}:00 uses '{profile}', which has no values yet: this run is skipped")

    print("Earth Engine login and project")
    try:
        ee.Initialize(project=PROJECT)
        ok(f"initialised with project {PROJECT}")
    except Exception as e:
        bad(f"initialise failed: {e}. Run the 'Authenticate Earth Engine' task.")
        return

    print(f"Your table assets under {USER}")
    try:
        root = "projects/earthengine-legacy/assets/" + USER.rstrip("/")
        for a in ee.data.listAssets({"parent": root}).get("assets", []):
            if a.get("type") == "TABLE":
                aid = a["name"].replace("projects/earthengine-legacy/assets/", "")
                try:
                    km2 = ee.FeatureCollection(aid).geometry().area(1000).divide(1e6).getInfo()
                    print(f"  {aid:55s} {km2:>10,.0f} km2")
                except Exception:
                    print(f"  {aid}")
    except Exception as e:
        warn(f"could not list assets: {e}")

    print("City boundaries, macro-regions, Building Atlas tiles")
    for city in CITIES:
        aid = str(BOUNDARIES[city])
        try:
            fc = boundary_fc(city)
            area = fc.geometry().area(10).divide(1e6).getInfo()
            ok(f"{city} boundary {aid}: {area:,.1f} km2")
        except Exception as e:
            bad(f"{city} boundary {aid}: {e}")
            continue
        try:
            m = macro_fc(city)
            n = m.size().getInfo()
            marea = m.geometry().area(1000).divide(1e6).getInfo() if n else 0
            (ok if n else bad)(f"{city} macro-region: {n} feature(s), {marea:,.0f} km2")
            tiles = M.gba_tiles(m.geometry())
            (ok if tiles else bad)(f"{city} GBA tiles: {[t.rsplit('/', 1)[1] for t in tiles]}")
        except Exception as e:
            bad(f"{city} macro-region: {e}")

    print("ERA5-Land land cells over each city (sea cells are masked in ERA5-Land)")
    date0, hour0 = RUNS[0][:2]
    for city in CITIES:
        fc = boundary_fc(city)
        t0 = ee.Date(utc_hour(city, date0, hour0))
        era5 = ee.Image(ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                        .filterDate(t0, t0.advance(1, "hour")).first())
        frac = (era5.select("temperature_2m").mask().rename("m")
                .reduceRegion(ee.Reducer.mean(), fc.geometry(), 100, maxPixels=1e10).get("m").getInfo())
        msg = f"{city}: {100 * frac:.1f}% of the boundary lies in ERA5-Land land cells"
        if frac < 0.999:
            (ok if M.FILL_ERA5_SEA else bad)(
                msg + (" (filled, FILL_ERA5_SEA = True)" if M.FILL_ERA5_SEA else ": UTCI will be empty in the rest"))
        else:
            ok(msg)

    print("Runs (local hour -> UTC)")
    for city in CITIES:
        for r in RUNS:
            date, hour, profile = r[:3]
            if city in run_cities(r):
                print(f"  {city:16s} {date} {hour:02d}:00 local = {utc_hour(city, date, hour)} UTC  [{profile}]")

    active = [t["description"] for t in ee.data.getTaskList() if t.get("state") in ("READY", "RUNNING")]
    print(f"Running or queued tasks: {active or 'none'}")


if __name__ == "__main__":
    main()
    if problems:
        print(f"\n{len(problems)} problem(s). Fix before running the pipeline.")
        sys.exit(1)
    print("\nReady.")

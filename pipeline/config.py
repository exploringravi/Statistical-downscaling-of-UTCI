"""Edit these before running."""

# Account used to run (new account; the old one ran out of quota on 19 Oct 2026).
PROJECT = "cityai-512212"                    # Earth Engine Cloud project for the v10 run (CityAI account). Previous: "exploringravi"
USER = "projects/exploringravi/assets/"      # asset root of the new account (samples are written here)
# Boundary and macro-region tables stay in the old account; share them with the new account as Reader.
OLD_ASSETS = "users/ravipandey336/"
# All three are rerun so UTCI uses 10 m wind everywhere (v1.0.0 used 1.5 m wind).
CITIES = ["Limassol", "Perugia", "Aachen"]   # extended with the field sites below

# City boundary asset per city, under USER. Maps are clipped to this.
BOUNDARIES = {
    "Limassol": OLD_ASSETS + "Limassol",
    "Perugia": OLD_ASSETS + "Perugia",
    "Aachen": OLD_ASSETS + "AachenStadt",
}

# Macro-region the Random Forest is trained over (1 km sampling grid): an asset id, or
# (FeatureCollection id, property, value). Limassol uses the asset of main_workflow.js.
MACRO_REGIONS = {
    "Limassol": "users/ravipandey336/CYP_Country",
    "Perugia": "users/ravipandey336/UMB_B",                 # Umbria, 8,439 km2
    "Aachen": "users/ravipandey336/KolnRegierungsbezirk",   # 7,338 km2
}
MACRO_TAG = {"Limassol": "Cyprus", "Perugia": "Umbria", "Aachen": "Koln"}

# Output grid: UTM zone per city, same as the published rasters.
CRS = {"Limassol": "EPSG:32636", "Perugia": "EPSG:32633", "Aachen": "EPSG:32632"}
TIMEZONE = {"Limassol": "Asia/Nicosia", "Perugia": "Europe/Rome", "Aachen": "Europe/Berlin"}

# Runs: (date, LOCAL hour, profile). Profiles are in utci_method.py.
# "data": limits from the ERA5-Land data and the formulas, one rule for every city and date.
# "summer" is the exact main_workflow.js setting (Ta floor 20 degC), kept for reference only:
# it forced 43% of Aachen 15 July pixels to 20 degC.
MAIN_CITIES = ["Limassol", "Perugia", "Aachen"]

# Runs: (date, LOCAL hour, profile[, cities]). Without a cities tuple a run applies to MAIN_CITIES.
RUNS = [
    ("2025-07-15", 14, "data", tuple(MAIN_CITIES)),
    ("2025-12-31", 14, "data", tuple(MAIN_CITIES)),
]

# ------------------------------------------------------------------ field-campaign sites
# Small boxes around each EnviWear transect (lon0, lat0, lon1, lat1, WGS84). Each site uses the
# training region, CRS and time zone of its parent city, so its Random Forest is trained exactly
# like the city maps; only the 10 m export is limited to the box. Hours are the local hour nearest
# to the middle of each walk (Limassol device clock was UTC+2; times in observations.csv are EEST).
SITES = {
    "PerugiaMassiano":  dict(parent="Perugia",  bbox=(12.3560, 43.1010, 12.3740, 43.1150), run=("2021-06-15", 19)),
    "PerugiaUnistra":   dict(parent="Perugia",  bbox=(12.3880, 43.1070, 12.4000, 43.1150), run=("2022-07-19", 13)),
    "PerugiaCortonese": dict(parent="Perugia",  bbox=(12.3620, 43.0970, 12.3730, 43.1040), run=("2023-08-08", 13)),
    "LimassolOldtown":  dict(parent="Limassol", bbox=(33.0370, 34.6690, 33.0510, 34.6790), run=("2025-05-21", 20)),
    "Episkopi":         dict(parent="Limassol", bbox=(32.8840, 34.6610, 32.8930, 34.6680), run=("2025-05-21", 18)),
    # RE-ACT Schools stations, Agios Andreas Primary School, Nicosia (P1 courtyard 35.17411 N 33.34775 E,
    # P2 parking 35.17403 N 33.34821 E): 24 hourly runs on 22 May 2026, trained over Cyprus
    "Nicosia":          dict(parent="Limassol", bbox=(33.3440, 35.1710, 33.3520, 35.1770),
                             runs=[("2026-05-22", h) for h in range(24)]),
}
# Fixed TinyTag loggers (R1 validation). 2022: five shielded loggers at 2 m around UniStraPg
# (points 1 to 5 of Tarpani et al. 2024, georeferenced from their Fig. 2 against the EnviWear GPS track).
# 2021: four loggers at the Pian di Massiano to Cortonese walk stops (EnviWear GPS stop medians).
# Clear days at 04:00 (before sunrise, no shield error) and 14:00 local; 2021 also at 18:00 (evening walks).
SITES["PerugiaLoggers22"] = dict(parent="Perugia", bbox=(12.3880, 43.1090, 12.3955, 43.1158),
    runs=[(d, h) for d in ("2022-02-10", "2022-02-17", "2022-05-28", "2022-05-31", "2022-07-14", "2022-07-19")
          for h in (4, 14)])
SITES["PerugiaLoggers21"] = dict(parent="Perugia", bbox=(12.3595, 43.1035, 12.3700, 43.1122),
    runs=[(d, h) for d in ("2021-06-12", "2021-06-13", "2021-06-15") for h in (4, 14, 18)])
# EnviWear walks in the walled city of Nicosia, 26 May 2025, 10:36 to 17:24 EEST (device clock UTC+2,
# shifted by +1 h as for the 21 May Limassol and Episkopi walks). Not used in any calibration: an
# independent mobile test across the day, with pyranometer readings for the sun and shade check.
SITES["NicosiaOldtown"] = dict(parent="Limassol", bbox=(33.3540, 35.1660, 33.3680, 35.1760),
    runs=[("2025-05-26", h) for h in range(11, 18)])
# Samples of these runs are written to the CityAI project (no write access to the exploringravi assets).
SAMPLE_ROOT = {s_: "projects/cityai-512212/assets/" for s_ in ("PerugiaLoggers22", "PerugiaLoggers21", "NicosiaOldtown")}
RUNS += [(d, h, "data", (site,)) for site, v in SITES.items() for d, h in (v.get("runs") or [v["run"]])]


def run_cities(run):
    """Cities a RUNS entry applies to."""
    return tuple(run[3]) if len(run) > 3 else tuple(MAIN_CITIES)

DRIVE_FOLDER = "UTCI_downscaling_v10"        # Drive folder for outputs (v8 in UTCI_downscaling, v9 in UTCI_downscaling_v9)

# Extra years for Hotspot_project change detection. Each year adds a 15 July and a 31 Dec run
# per city with the same method. Building and canopy height stay fixed (GBA, ETH 2020);
# Sentinel-2, MODIS LST and ERA5-Land change with the year.
CHANGE_YEARS = []   # e.g. [2019, 2020, 2021, 2022, 2023, 2024]
from datetime import date as _date
RUNS = RUNS + [(f"{y}-07-15", 14, "data", tuple(MAIN_CITIES)) for y in CHANGE_YEARS if y != 2025 and _date(y, 7, 16) < _date.today()] \
            + [(f"{y}-12-31", 14, "data", tuple(MAIN_CITIES)) for y in CHANGE_YEARS if y != 2025 and _date(y + 1, 1, 1) < _date.today()]


# field sites inherit everything from their parent city
for _site, _v in SITES.items():
    _p = _v["parent"]
    BOUNDARIES[_site] = ("bbox",) + tuple(_v["bbox"])
    MACRO_REGIONS[_site] = MACRO_REGIONS[_p]
    MACRO_TAG[_site] = MACRO_TAG[_p] + _site
    CRS[_site] = CRS[_p]
    TIMEZONE[_site] = TIMEZONE[_p]
CITIES = CITIES + [s_ for s_ in SITES if s_ not in CITIES]

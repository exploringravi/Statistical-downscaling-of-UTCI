"""
utci_method.py
Python port of gee/main_workflow.js (the published Limassol 15 July 2025 script), made
general for Limassol, Perugia and Aachen. Section numbers follow the JS parts.

What is identical to the JS
  Every formula, weight, clamp, kernel, fill value and RF setting. The UTCI polynomial
  is copied term by term (utci_poly.py). With PROFILES["summer"] and FILL_ERA5_SEA = False
  this reproduces the JS for Limassol.

What was made general (the JS hard-codes Limassol)
  - Study area, macro-region, CRS: from config.py per city.
  - UTC offset: from the city's time zone and date (JS: fixed tzOffsetHours = 3, which is
    wrong for Cyprus in winter, UTC+2).
  - Sentinel-2 window: first day of the run month to the last day of the next month.
    For 15 July this is 2025-07-01 to 2025-08-31, the JS window.
  - Global Building Atlas tiles: picked from the macro-region bounds (JS: two Cyprus tiles).

What was added
  FILL_ERA5_SEA. ERA5-Land is land only, so cells classed as sea are masked. In Limassol the
  cells east of 33.05 E and south of 34.65 N are sea cells, which left the eastern city and
  the port area without wind, humidity and radiation, so UTCI was empty there. With
  FILL_ERA5_SEA = True the forcing (Td, U10, V10, SW, LW) in masked cells is filled from the
  neighbouring land cells. The RF training targets (Ta, RH) are left unfilled, so the model
  itself is unchanged.

Seasonal settings
  The JS clamps Ta to 20 to 45 degC, ground temperature to 20 to 70 degC and so on. These
  are summer values (Ta for Aachen on 31 Dec would be forced up to 20 degC). They sit in
  PROFILES. "summer" holds the JS values. "winter" takes its limits from the ERA5-Land data
  and from the bounds the formulas already imply; all weights are unchanged.
"""

import math
from datetime import date as _date, timedelta

import ee

from utci_poly import UTCI_POLY

# ============================================================== settings (PART 1.7)
SAMPLE_SCALE = 1000
SAMPLE_N = 7000
SAMPLE_TILESCALE = 4         # JS value; the pipeline uses 16 when it stores samples first
RF_TREES = 200
SEED = 7
OUT_SCALE = 10
# Weather scaling of the expert-set corrections (localHeatCorrection, localRHCorrection).
#   None    : full correction in any weather (v1 to v6)
#   "kt"    : f = clamp(kt / 0.75, 0, 1), kt = SW / S0 the clearness index; f = 1 with the sun
#             below 3 degrees, where kt is undefined
#   "cloud" : f = 1 - 0.75 N^3.4 (Kasten and Czeplak 1980), N the cloud fraction from ERA5-Land
#             downward longwave: N = (eps - eps_clear) / (1 - eps_clear), eps = LW / (sigma T^4),
#             eps_clear = 1.24 (e / T)^(1/7) (Brutsaert 1975); valid by day and night
# Microclimate contrasts shrink under overcast skies (Oke 1982), while the corrections were set
# for clear summer conditions.
CORR_WEATHER = None
# Calibration of the expert-set corrections and of the wind (all 1.0 = v1 to v6 behaviour).
#   a_warm : multiplies the warming terms of localHeatCorrection (LST anomaly, heat storage, NDBI)
#   a_cool : multiplies its cooling terms (NDVI, moisture, coastal exposure, ventilation, shade)
#   a_rh   : multiplies localRHCorrection
#   k_wind : multiplies the local 10 m wind (and so the 1.5 m wind derived from it)
CAL = dict(a_warm=0.5, a_cool=0.4, a_rh=0.15, k_wind=1.0)   # v8: fitted on 5 field campaigns + 10 DWD stations x 2 dates
# with ANOM_WEATHER = "kt" (ana/calibrate.py --shrink, leave one group out); v1 to v6 used all 1.0
# Weather scaling of the whole 10 m anomaly from ERA5-Land (Random Forest detail + corrections):
#   Ta_10m = Ta_ERA5 + f (Ta_10m - Ta_ERA5), the same for RH, with f = clamp(kt / 0.75, 0, 1) and f = 1
#   with the sun below 3 degrees. Under overcast skies local contrasts shrink (Oke 1982); at the DWD
#   stations on 31 Dec 2025 (overcast) the full anomaly made the field worse than ERA5-Land.
#   None: no scaling (v1 to v7).
ANOM_WEATHER = "kt"
# True: the 10 m stack also carries the parts needed to fit CAL (tools/calibration_points.py)
EXPORT_PARTS = False
# True: the 10 m stack also carries the Random Forest air-temperature predictors, prefixed p_ (tools/station_cell_check.py)
EXPORT_PREDICTORS = False
# MRT scheme. "projected" (v8): direct sun x (1 - shade proxy), surfaces weighted by proxy view factors.
# "cylinder" (v9): standing person as in Di Napoli et al. (2020), half upper and half lower hemisphere,
# with direct-beam shadows cast at the target hour from terrain + building + canopy heights
# (ee.Terrain.hillShadow) and 10% of the beam transmitted under canopy taller than 2 m.
MRT_SCHEME = "cylinder"
FILL_ERA5_SEA = True

PROFILES = {
    "summer": dict(
        ta_clamp=(20, 45),
        rh_clamp=(5, 100),
        t_ground_clamp=(20, 70),
        t_wall_clamp=(20, 65),
        t_tree_clamp=(15, 50),
        mrt_clamp=(15, 85),
        lst_fill=38,                 # safeBand fill for MODIS LST
        solar_elevation_factor=0.90,  # "Cyprus summer midday assumption"
        coast_direction_factor=0.65,
        ws10_clamp=(0.05, 5.5),
        ws1p5_clamp=(0.05, 3.2),
        utci_wind="WS_1p5m",         # JS PART 10 uses pedestrian wind, see README
    ),
    # Winter: limits taken from the data and from the method's own formulas, no fixed numbers.
    #   Ta: the RF cannot predict outside the ERA5-Land Ta range of the macro-region, and
    #       localHeatCorrection adds -2.5 to +4.0 degC, so Ta_10m is bounded by
    #       [ERA5 min - 2.5, ERA5 max + 4.0] at that hour ("data").
    #   Ground, wall, tree temperature: each formula is Ta plus bounded terms (ground -11.7
    #       to +11.7, wall -5.8 to +10.9, tree -3.2 to -0.2 degC), so no absolute clamp (None).
    #   MRT: no absolute clamp; dTmrt is still limited to -30 to +70 K before UTCI, as in the JS.
    #   MODIS gaps: macro-region mean LST of the window instead of 38 degC.
    #   Everything else as summer.
    "data": dict(
        ta_clamp="data",
        s2_gap_fill=True,
        rh_clamp=(5, 100),
        t_ground_clamp=None,
        t_wall_clamp=None,
        t_tree_clamp=None,
        mrt_clamp=None,
        lst_fill="mean",
        solar_elevation_factor=0.90,          # not used when mrt_sw = "projected"
        coast_direction_factor=0.65,
        ws10_clamp=(0.05, 17.0),              # UTCI wind range; JS capped 10 m wind at 5.5 m/s
        ws1p5_clamp=(0.05, 17.0),
        utci_wind="WS_1p5m",
        mrt_sw="projected",                   # see solar_geometry() and PART 9
    ),
}
PROFILES["winter"] = PROFILES["data"]


def _s2_window(date_str):
    d = _date.fromisoformat(date_str)
    start = d.replace(day=1)
    nxt2 = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    nxt2 = (nxt2.replace(day=28) + timedelta(days=4)).replace(day=1)
    return start.isoformat(), (nxt2 - timedelta(days=1)).isoformat()


def solar_geometry(utc_iso, lon, lat):
    """Solar elevation (deg), cos(zenith) and top-of-atmosphere horizontal irradiance (W m-2)
    for the middle of the ERA5-Land accumulation hour (time stamp minus 30 min).
    NOAA general solar position equations (Spencer 1971 series)."""
    import math
    from datetime import datetime as _dt
    t = _dt.fromisoformat(utc_iso) - timedelta(minutes=30)
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cosz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    elev = math.degrees(math.asin(max(-1.0, min(1.0, cosz))))
    s0 = 1361 * (1 + 0.033 * math.cos(2 * math.pi * doy / 365)) * max(cosz, 0.0)
    return elev, cosz, s0


def solar_azimuth(utc_iso, lon, lat):
    """Solar azimuth (degrees clockwise from north) for the middle of the accumulation hour."""
    import math
    from datetime import datetime as _dt
    t = _dt.fromisoformat(utc_iso) - timedelta(minutes=30)
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = math.radians((hour * 60 + eqtime + 4 * lon) / 4 - 180)
    la = math.radians(lat)
    cosz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    sinz = math.sqrt(max(1e-9, 1 - cosz ** 2))
    caz = (math.sin(decl) - math.sin(la) * cosz) / (math.cos(la) * sinz)
    az = math.degrees(math.acos(max(-1.0, min(1.0, caz))))
    return 360 - az if ha > 0 else az


def gba_tiles(macro_geom):
    """5 degree Global Building Atlas tiles covering the macro-region (north/east only)."""
    coords = macro_geom.bounds(1000).coordinates().get(0).getInfo()
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    out = []
    for lo in range(int(min(lons) // 5) * 5, int(max(lons) // 5) * 5 + 5, 5):
        for la in range(int(min(lats) // 5) * 5, int(max(lats) // 5) * 5 + 5, 5):
            tid = (f"projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS/"
                   f"e{lo:03d}_n{la + 5:02d}_e{lo + 5:03d}_n{la:02d}")
            try:
                ee.data.getAsset(tid)
                out.append(tid)
            except ee.EEException:
                pass
    return out


def fill_sea(img):
    """Fill ERA5-Land cells masked as sea from neighbouring land cells (up to 3 cells)."""
    p = img.projection()
    f = img.reproject(p).focalMean(radius=1, kernelType="square", units="pixels",
                                   iterations=3).reproject(p)
    return img.unmask(f, False)


def weather_factor(utc_iso, area, Ta_coarse, Td, SW, LW, mode="default"):
    """Scaling of the expert-set corrections for the weather at this hour (see CORR_WEATHER)."""
    mode = CORR_WEATHER if mode == "default" else mode
    if mode is None:
        return ee.Image.constant(1)
    if mode == "kt":
        elev, cosz, s0 = solar_geometry(utc_iso, *area.geometry().centroid(100).coordinates().getInfo())
        if elev < 3:
            return ee.Image.constant(1)
        return SW.divide(s0).divide(0.75).clamp(0, 1).resample("bilinear").rename("f_weather")
    if mode == "cloud":
        T = Ta_coarse.add(273.15)
        e = Td.expression("6.112 * exp(17.62 * Td / (243.12 + Td))", {"Td": Td})
        eps = LW.divide(T.pow(4).multiply(5.670374419e-8))
        eps_clear = e.divide(T).pow(1 / 7).multiply(1.24)
        N = eps.subtract(eps_clear).divide(ee.Image(1).subtract(eps_clear)).clamp(0, 1)
        return ee.Image(1).subtract(N.pow(3.4).multiply(0.75)).resample("bilinear").rename("f_weather")
    raise ValueError(f"CORR_WEATHER must be None, 'kt' or 'cloud', not {mode!r}")


def clampx(img, lim):
    """clamp() that accepts None (no clamp) or computed ee.Number limits."""
    if lim is None:
        return img
    lo, hi = lim
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return img.clamp(lo, hi)
    return img.max(ee.Image.constant(lo)).min(ee.Image.constant(hi))


def safe_band(img, name, fill):
    return img.rename(name).unmask(fill, False).toFloat()


def build(study_fc, macro_fc, out_crs, date_str, utc_iso, profile_name, gba_ids, train_fc=None):
    """train_fc: optional {"Ta": fc, "RH": fc} of stored samples to train on (saves memory)."""
    """Everything for one city and hour. Returns (stack_10m, era5_coarse, samples_ta, samples_rh)."""
    P = PROFILES[profile_name]
    if P is None:
        raise NotImplementedError(
            f"Profile '{profile_name}' has no values yet. Add the script used for the "
            "published winter rasters, or define PROFILES['winter'] in utci_method.py.")
    studyArea, macroRegion = study_fc, macro_fc

    # ---------------------------------------------------------- PART 2 ERA5-Land
    utc = ee.Date(utc_iso)
    era5 = ee.Image(ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                    .filterDate(utc, utc.advance(1, "hour")).first())
    Ta_coarse = era5.select("temperature_2m").subtract(273.15).rename("Ta")
    Td_coarse = era5.select("dewpoint_temperature_2m").subtract(273.15).rename("Td")
    RH_coarse = Ta_coarse.expression(
        "100 * (exp((17.625 * Td) / (243.04 + Td)) / exp((17.625 * Ta) / (243.04 + Ta)))",
        {"Ta": Ta_coarse, "Td": Td_coarse}).clamp(0, 100).rename("RH")
    U10 = era5.select("u_component_of_wind_10m").rename("U10")
    V10 = era5.select("v_component_of_wind_10m").rename("V10")
    WS_reference_10m = U10.pow(2).add(V10.pow(2)).sqrt().clamp(0.1, 17).rename("WS_reference_10m")
    SWdown = era5.select("surface_solar_radiation_downwards_hourly").divide(3600).clamp(0, 1200).rename("SWdown_Wm2")
    LWdown = era5.select("surface_thermal_radiation_downwards_hourly").divide(3600).clamp(150, 600).rename("LWdown_Wm2")

    era5_coarse = ee.Image.cat([Ta_coarse, RH_coarse, WS_reference_10m, SWdown, LWdown, Td_coarse]).rename(
        ["ERA5_Ta_C", "ERA5_RH_percent", "ERA5_WS10_ms", "ERA5_SWdown_Wm2", "ERA5_LWdown_Wm2", "ERA5_Td_C"]).toFloat()

    # forcing used at 10 m (filled), training targets stay raw
    if FILL_ERA5_SEA:
        Td_f, U10_f, V10_f = fill_sea(Td_coarse), fill_sea(U10), fill_sea(V10)
        SW_f, LW_f = fill_sea(SWdown), fill_sea(LWdown)
    else:
        Td_f, U10_f, V10_f, SW_f, LW_f = Td_coarse, U10, V10, SWdown, LWdown

    # ---------------------------------------------------------- PART 3 Sentinel-2
    def mask_s2(img):
        scl = img.select("SCL")
        m = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
             .And(scl.neq(10)).And(scl.neq(11)))
        return (img.updateMask(m).divide(10000).select(["B2", "B3", "B4", "B8", "B11", "B12"])
                .copyProperties(img, ["system:time_start"]))

    s2_start, s2_end = _s2_window(date_str)
    s2col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(macroRegion).filterDate(s2_start, s2_end)
    s2Main = s2col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).map(mask_s2)
    s2Fallback = s2col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)).map(mask_s2)
    s2 = s2Main.median().unmask(s2Fallback.median(), False)
    s2_gaps = 0.0
    if P.get("s2_gap_fill") and train_fc is not None:
        # Only build the fill when the city actually has gaps: the +-90 day median over a whole
        # city is expensive (Perugia 15 July export stalled for 6+ hours) and changes nothing
        # where every pixel already has a clear scene.
        s2_gaps = ee.Number(s2.select("B4").mask().Not().rename("gap").clip(studyArea)
                            .reduceRegion(reducer=ee.Reducer.mean(), geometry=studyArea.geometry(),
                                          scale=100, maxPixels=1e9, bestEffort=True)
                            .get("gap")).getInfo() or 0.0
        print(f"    Sentinel-2 gaps in the city for {date_str}: {100 * s2_gaps:.1f}%"
              + (", filling from +-90 days" if s2_gaps > 0 else ", no fill needed"))
    if P.get("s2_gap_fill") and train_fc is not None and s2_gaps > 0:
        # Map step only (training samples fill their own gaps with safe_band), and only inside the
        # city: pixels with no clear scene in the window (Aachen, Dec 2025: everything east of one
        # Sentinel-2 orbit edge) are filled from a +-90 day median. Over the whole macro-region this
        # exceeded Earth Engine memory.
        d1 = ee.Date(date_str)
        wide = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(studyArea)
                .filterDate(d1.advance(-90, "day"), d1.advance(91, "day"))
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)).map(mask_s2).median()
                .clip(studyArea))
        s2 = s2.unmask(wide, False)
    s2 = s2.clip(macroRegion)

    NDVI = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    NDBI = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")
    NDWI = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
    NDMI = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
    EVI = s2.expression("2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
                        {"NIR": s2.select("B8"), "RED": s2.select("B4"), "BLUE": s2.select("B2")}).rename("EVI")
    Albedo = s2.expression("(0.356 * B2 + 0.130 * B4 + 0.373 * B8 + 0.085 * B11 + 0.072 * B12 - 0.0018)",
                           {b: s2.select(b) for b in ["B2", "B4", "B8", "B11", "B12"]}
                           ).clamp(0.04, 0.45).rename("Albedo")

    # ---------------------------------------------------------- PART 3B MODIS LST
    d0 = ee.Date(date_str)
    modisCol = (ee.ImageCollection("MODIS/061/MOD11A1").filterBounds(macroRegion)
                .filterDate(d0.advance(-4, "day"), d0.advance(5, "day")).select("LST_Day_1km"))
    # A median composite has no native projection (Earth Engine default: 1 degree). Without
    # setDefaultProjection, resample("bilinear") below interpolated from that 1 degree grid, so the
    # LST predictor carried almost no 1 km detail (within-cell SD 0.01 to 0.12 degC in v1 to v5 samples).
    modisLST = (modisCol.median().setDefaultProjection(ee.Image(modisCol.first()).projection())
                .multiply(0.02).subtract(273.15).rename("MODIS_LST_C").clip(macroRegion))
    modisLST_10m = modisLST.resample("bilinear").rename("MODIS_LST_C")
    lst_stats = modisLST.reduceRegion(reducer=ee.Reducer.mean(), geometry=macroRegion.geometry(),
                                      scale=1000, maxPixels=1e13)
    lstMean = lst_stats.getNumber("MODIS_LST_C")
    LST_anomaly = modisLST_10m.subtract(lstMean).clamp(-8, 8).rename("LST_anomaly_C")
    if P["lst_fill"] == "mean":
        # MODIS gaps (cloud, common in winter): anomaly 0 = regional mean, so the correction terms
        # stay defined instead of masking Ta, RH, MRT and UTCI (Aachen 31 Dec: 76% masked).
        lstMean = ee.Number(ee.Algorithms.If(lst_stats.get("MODIS_LST_C"), lst_stats.get("MODIS_LST_C"), 0))
        LST_anomaly = LST_anomaly.unmask(0)

    # ---------------------------------------------------------- PART 4 DEM, GBA, canopy
    dem = (ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(macroRegion).select("DEM")
           .mosaic().clip(macroRegion).rename("DEM_m"))
    dem = dem.unmask(dem.focalMean(radius=3, units="pixels", iterations=2), False)
    terrain = ee.Terrain.products(dem.reproject(crs=out_crs, scale=30))
    slope = terrain.select("slope").rename("slope_deg")
    terrainExposure = slope.unitScale(0, 30).clamp(0, 1).rename("terrain_exposure")
    aspect = terrain.select("aspect").rename("aspect_deg")
    aspectRad = aspect.multiply(3.141592653589793 / 180)
    eastness = aspectRad.sin().rename("eastness")
    northness = aspectRad.cos().rename("northness")
    demMean250 = dem.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.circle(250, "meters"))
    TPI_250m = dem.subtract(demMean250).rename("TPI_250m")
    ridgeExposure = TPI_250m.unitScale(0, 80).clamp(0, 1).rename("ridge_exposure")
    valleyShelter = TPI_250m.multiply(-1).unitScale(0, 80).clamp(0, 1).rename("valley_shelter")
    terrainWindExposure = (terrainExposure.multiply(0.30).add(ridgeExposure.multiply(0.45))
                           .subtract(valleyShelter.multiply(0.35)).clamp(0, 1).rename("terrain_wind_exposure"))

    gba = ee.FeatureCollection(gba_ids[0])
    for gid in gba_ids[1:]:
        gba = gba.merge(ee.FeatureCollection(gid))
    gbaBuildings = gba.filterBounds(macroRegion).filter(ee.Filter.gt("height", 0))
    buildingHeight_GBA = (gbaBuildings.reduceToImage(properties=["height"], reducer=ee.Reducer.first())
                          .clip(macroRegion).unmask(0).clamp(0, 80).rename("building_height_GBA_m"))
    buildingMask_GBA = buildingHeight_GBA.gt(3.5).And(NDBI.gt(-0.02)).rename("building_mask_GBA")
    builtProxy = NDBI.gt(0.05).And(NDVI.lt(0.30)).And(NDWI.lt(0.00)).rename("built_proxy")
    combinedBuildingMask = buildingMask_GBA.max(
        builtProxy.And(NDBI.gt(0.12)).And(NDVI.lt(0.25))).rename("combined_building_mask")

    def density(r):
        return combinedBuildingMask.reduceNeighborhood(
            reducer=ee.Reducer.mean(), kernel=ee.Kernel.circle(r, "meters")).rename(f"building_density_{r}m")

    buildingDensity30m, buildingDensity50m = density(30), density(50)
    buildingDensity100m, buildingDensity250m = density(100), density(250)

    buildingHeightProxy = (NDBI.unitScale(0.0, 0.45).clamp(0, 1)
                           .multiply(buildingDensity50m.unitScale(0.1, 0.8).clamp(0, 1))
                           .multiply(18).rename("building_height_proxy_m"))
    buildingHeight = (buildingHeight_GBA.where(buildingHeight_GBA.lte(0), buildingHeightProxy)
                      .clamp(0, 50).rename("building_height_m"))

    ethCanopyHeight2020 = (ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1")
                           .rename("canopy_height_2020_m").clip(macroRegion))
    vegVitality = (NDVI.unitScale(0.20, 0.75).clamp(0, 1).multiply(0.60)
                   .add(EVI.unitScale(0.10, 0.70).clamp(0, 1).multiply(0.40))
                   .clamp(0.20, 1.15).rename("veg_vitality"))
    canopyHeight = (ethCanopyHeight2020.multiply(vegVitality).updateMask(NDVI.gt(0.18))
                    .updateMask(buildingMask_GBA.neq(1)).unmask(0).clamp(0, 35)
                    .rename("canopy_height_effective_m"))

    # ---------------------------------------------------------- PART 5 proxies
    one = ee.Image(1)
    buildingHeightEffect = buildingHeight.unitScale(0, 30).clamp(0, 1).rename("building_height_effect")
    treeHeightEffect = canopyHeight.unitScale(0, 20).clamp(0, 1).rename("tree_height_effect")
    roughnessProxy = (buildingDensity30m.multiply(0.25).add(buildingDensity50m.multiply(0.25))
                      .add(buildingDensity100m.multiply(0.20)).add(buildingHeightEffect.multiply(0.20))
                      .add(treeHeightEffect.multiply(0.10)).clamp(0, 1).rename("roughness_proxy"))
    svfProxy = (one.subtract(buildingDensity30m.multiply(0.25)).subtract(buildingDensity50m.multiply(0.25))
                .subtract(buildingDensity100m.multiply(0.15)).subtract(buildingHeightEffect.multiply(0.25))
                .subtract(treeHeightEffect.multiply(0.10)).clamp(0.15, 1).rename("svf_proxy"))
    waterMask = NDWI.gt(0.1).unmask(0).rename("water_mask")
    # v5 fix: pixel distance x pixel size, so metres are right at 10 m (maps) and 1 km (RF samples)
    distToWater = waterMask.fastDistanceTransform(1024).sqrt().multiply(ee.Image.pixelArea().sqrt()).rename("dist_to_water_m")
    coastalExposure = one.subtract(distToWater.unitScale(0, 800).clamp(0, 1)).clamp(0, 1).rename("coastal_exposure")
    heatStorageProxy = (NDBI.unitScale(-0.2, 0.5).clamp(0, 1).multiply(0.25)
                        .add(buildingDensity30m.multiply(0.20)).add(buildingDensity50m.multiply(0.15))
                        .add(buildingHeightEffect.multiply(0.15))
                        .add(LST_anomaly.unitScale(-4, 8).clamp(0, 1).multiply(0.30))
                        .subtract(Albedo.unitScale(0.05, 0.35).clamp(0, 1).multiply(0.15))
                        .clamp(0, 1).rename("heat_storage_proxy"))
    moistureProxy = (NDVI.unitScale(0.1, 0.8).clamp(0, 1).multiply(0.30)
                     .add(EVI.unitScale(0.0, 0.8).clamp(0, 1).multiply(0.25))
                     .add(NDWI.unitScale(-0.3, 0.4).clamp(0, 1).multiply(0.20))
                     .add(NDMI.unitScale(-0.3, 0.5).clamp(0, 1).multiply(0.25))
                     .clamp(0, 1).rename("moisture_proxy"))
    urbanObstruction = (buildingDensity30m.multiply(0.25).add(buildingDensity50m.multiply(0.25))
                        .add(buildingDensity100m.multiply(0.15)).add(buildingHeightEffect.multiply(0.25))
                        .add(treeHeightEffect.multiply(0.10)).clamp(0, 1).rename("urban_obstruction"))
    openness = (one.subtract(buildingDensity30m.multiply(0.30)).subtract(buildingDensity50m.multiply(0.25))
                .subtract(buildingDensity100m.multiply(0.15)).subtract(treeHeightEffect.multiply(0.10))
                .add(coastalExposure.multiply(0.15)).add(svfProxy.multiply(0.20))
                .clamp(0, 1).rename("openness"))
    ventilationPotential = (coastalExposure.multiply(0.30).add(openness.multiply(0.30))
                            .add(svfProxy.multiply(0.20)).add(terrainExposure.multiply(0.10))
                            .subtract(urbanObstruction.multiply(0.40)).clamp(0, 1)
                            .rename("ventilation_potential"))
    shadeProxy = (treeHeightEffect.multiply(0.35).add(buildingHeightEffect.multiply(0.25))
                  .add(buildingDensity30m.multiply(0.20)).add(one.subtract(svfProxy).multiply(0.25))
                  .clamp(0, 0.90).rename("shade_proxy"))

    # ---------------------------------------------------------- PART 7 predictor stacks
    lst_fill = ee.Image.constant(lstMean) if P["lst_fill"] == "mean" else P["lst_fill"]
    common_veg = [safe_band(NDVI, "NDVI", 0), safe_band(EVI, "EVI", 0),
                  safe_band(NDWI, "NDWI", 0), safe_band(NDMI, "NDMI", 0)]
    terrain_bands = [
        safe_band(dem, "DEM_m", 0), safe_band(slope, "slope_deg", 0),
        safe_band(terrainExposure, "terrain_exposure", 0), safe_band(aspect, "aspect_deg", 0),
        safe_band(eastness, "eastness", 0), safe_band(northness, "northness", 0),
        safe_band(TPI_250m, "TPI_250m", 0), safe_band(ridgeExposure, "ridge_exposure", 0),
        safe_band(valleyShelter, "valley_shelter", 0),
        safe_band(terrainWindExposure, "terrain_wind_exposure", 0)]
    predictorsTa = ee.Image.cat(common_veg + [
        safe_band(NDBI, "NDBI", 0), safe_band(Albedo, "Albedo", 0.15),
        safe_band(modisLST_10m, "MODIS_LST_C", lst_fill), safe_band(LST_anomaly, "LST_anomaly_C", 0),
        safe_band(buildingHeight, "building_height_m", 0),
        safe_band(buildingDensity30m, "building_density_30m", 0),
        safe_band(buildingDensity50m, "building_density_50m", 0),
        safe_band(buildingDensity100m, "building_density_100m", 0),
        safe_band(buildingDensity250m, "building_density_250m", 0),
        safe_band(canopyHeight, "canopy_height_m", 0), safe_band(vegVitality, "veg_vitality", 0.5),
        safe_band(svfProxy, "svf_proxy", 0.5), safe_band(roughnessProxy, "roughness_proxy", 0),
        safe_band(heatStorageProxy, "heat_storage_proxy", 0),
        safe_band(moistureProxy, "moisture_proxy", 0),
        safe_band(coastalExposure, "coastal_exposure", 0), safe_band(openness, "openness", 0.5),
        safe_band(ventilationPotential, "ventilation_potential", 0.5),
        safe_band(shadeProxy, "shade_proxy", 0)] + terrain_bands).clip(macroRegion)
    predictorsRH = ee.Image.cat(common_veg + [
        safe_band(modisLST_10m, "MODIS_LST_C", lst_fill), safe_band(LST_anomaly, "LST_anomaly_C", 0),
        safe_band(moistureProxy, "moisture_proxy", 0),
        safe_band(coastalExposure, "coastal_exposure", 0),
        safe_band(distToWater, "dist_to_water_m", 1000),
        safe_band(buildingHeight, "building_height_m", 0),
        safe_band(buildingDensity30m, "building_density_30m", 0),
        safe_band(buildingDensity50m, "building_density_50m", 0),
        safe_band(buildingDensity100m, "building_density_100m", 0),
        safe_band(canopyHeight, "canopy_height_m", 0), safe_band(vegVitality, "veg_vitality", 0.5),
        safe_band(svfProxy, "svf_proxy", 0.5), safe_band(roughnessProxy, "roughness_proxy", 0),
        safe_band(heatStorageProxy, "heat_storage_proxy", 0), safe_band(openness, "openness", 0.5),
        safe_band(ventilationPotential, "ventilation_potential", 0.5),
        safe_band(shadeProxy, "shade_proxy", 0)] + terrain_bands).clip(macroRegion)
    bandsTa, bandsRH = predictorsTa.bandNames(), predictorsRH.bandNames()

    # ---------------------------------------------------------- PART 6 / 8 RF
    def sample(target, name, stack, geometries=False):
        return stack.addBands(target.rename(name)).sample(
            region=macroRegion.geometry(), scale=SAMPLE_SCALE, numPixels=SAMPLE_N, seed=SEED,
            geometries=geometries, dropNulls=True, tileScale=SAMPLE_TILESCALE)

    def downscale(target, name, stack, bands):
        rf = (ee.Classifier.smileRandomForest(numberOfTrees=RF_TREES, seed=SEED, bagFraction=0.7,
                                              minLeafPopulation=5)
              .setOutputMode("REGRESSION")
              .train(features=(train_fc[name] if train_fc else sample(target, name, stack)),
                     classProperty=name, inputProperties=bands))
        pred = stack.select(bands).classify(rf).rename(name + "_10m_raw").clip(studyArea).toFloat()
        return pred, rf

    if P["ta_clamp"] == "data":
        mm = Ta_coarse.reduceRegion(reducer=ee.Reducer.minMax(), geometry=macroRegion.geometry(),
                                    scale=9000, maxPixels=1e13, bestEffort=True)
        ta_lim = (ee.Number(mm.get("Ta_min")).subtract(2.5), ee.Number(mm.get("Ta_max")).add(4.0))
    else:
        ta_lim = P["ta_clamp"]
    Ta_raw, rf_ta = downscale(Ta_coarse, "Ta", predictorsTa, bandsTa)
    Ta_10m_raw = clampx(Ta_raw, ta_lim).rename("Ta_10m_raw")
    RH_raw, rf_rh = downscale(RH_coarse, "RH", predictorsRH, bandsRH)
    RH_10m_raw = RH_raw.clamp(*P["rh_clamp"]).rename("RH_10m_raw")

    Ta_forc = fill_sea(Ta_coarse) if FILL_ERA5_SEA else Ta_coarse
    f_weather = weather_factor(utc_iso, studyArea, Ta_forc, Td_f, SW_f, LW_f).unmask(1)
    corrWarm = (LST_anomaly.multiply(0.35).add(heatStorageProxy.multiply(1.40))
                .add(NDBI.unitScale(-0.2, 0.5).clamp(0, 1).multiply(0.80))).rename("corr_warm")
    corrCool = (NDVI.unitScale(0.1, 0.8).clamp(0, 1).multiply(0.60)
                .add(moistureProxy.multiply(0.40)).add(coastalExposure.multiply(0.35))
                .add(ventilationPotential.multiply(0.35)).add(shadeProxy.multiply(0.50))).rename("corr_cool")
    localHeatCorrection = (corrWarm.multiply(CAL["a_warm"]).subtract(corrCool.multiply(CAL["a_cool"]))
                           .clamp(-2.5, 4.0).multiply(f_weather).rename("local_Ta_correction"))
    Ta_10m = clampx(Ta_10m_raw.add(localHeatCorrection), ta_lim).rename("Ta_10m")
    f_anom = (weather_factor(utc_iso, studyArea, Ta_forc, Td_f, SW_f, LW_f, ANOM_WEATHER).unmask(1)
              if ANOM_WEATHER else ee.Image.constant(1))
    Ta_ref = Ta_forc.resample("bilinear")
    if ANOM_WEATHER:
        Ta_10m = clampx(Ta_ref.add(Ta_10m.subtract(Ta_ref).multiply(f_anom)), ta_lim).rename("Ta_10m")
    RH_from_corrected_Ta = Ta_10m.expression(
        "100 * (exp((17.625 * Td) / (243.04 + Td)) / exp((17.625 * Ta) / (243.04 + Ta)))",
        {"Ta": Ta_10m, "Td": Td_f.resample("bilinear")}).clamp(0, 100).rename("RH_from_corrected_Ta")
    localRHCorrection = (moistureProxy.multiply(6.0).add(coastalExposure.multiply(4.0))
                         .add(shadeProxy.multiply(2.0)).subtract(heatStorageProxy.multiply(5.0))
                         .subtract(LST_anomaly.max(0).multiply(0.7))
                         .subtract(ventilationPotential.multiply(2.0))
                         .clamp(-12, 12).rename("corr_rh_expert"))
    corrRH = localRHCorrection
    localRHCorrection = localRHCorrection.multiply(CAL["a_rh"]).multiply(f_weather).rename("local_RH_correction")
    RH_10m = (RH_from_corrected_Ta.multiply(0.65).add(RH_10m_raw.multiply(0.35))
              .add(localRHCorrection).clamp(*P["rh_clamp"]).rename("RH_10m"))
    if ANOM_WEATHER:
        RH_ref = (fill_sea(RH_coarse) if FILL_ERA5_SEA else RH_coarse).resample("bilinear")
        RH_10m = RH_ref.add(RH_10m.subtract(RH_ref).multiply(f_anom)).clamp(*P["rh_clamp"]).rename("RH_10m")

    # ---------------------------------------------------------- PART 8C wind
    U10_ref = U10_f.resample("bilinear").rename("U10_ref")
    V10_ref = V10_f.resample("bilinear").rename("V10_ref")
    seaWindFactor = (coastalExposure.multiply(ee.Image.constant(P["coast_direction_factor"]))
                     .multiply(one.subtract(urbanObstruction.multiply(0.45))).clamp(0, 1))
    coastalWindPenetration = coastalExposure.multiply(seaWindFactor).multiply(0.20)
    terrainShelter = slope.unitScale(2, 25).clamp(0, 1).multiply(one.subtract(terrainExposure.multiply(0.3)))
    urbanDrag = (urbanObstruction.multiply(0.40).add(roughnessProxy.multiply(0.30))
                 .add(buildingDensity30m.multiply(0.20)).add(treeHeightEffect.multiply(0.10)).clamp(0, 1))
    corridorGain = (openness.multiply(0.35).add(ventilationPotential.multiply(0.25))
                    .add(svfProxy.multiply(0.15)).add(coastalWindPenetration.multiply(0.35)).clamp(0, 1))
    windFactor10m = (ee.Image(0.75).subtract(urbanDrag.multiply(0.50)).subtract(terrainShelter.multiply(0.12))
                     .add(corridorGain.multiply(0.45)).add(terrainWindExposure.multiply(0.35))
                     .clamp(0.18, 1.35))
    WS_local_10m = (U10_ref.multiply(windFactor10m).pow(2).add(V10_ref.multiply(windFactor10m).pow(2))
                    .sqrt().multiply(CAL["k_wind"]).clamp(*P["ws10_clamp"]).rename("WS_local_10m"))
    pedestrianFactor = (ee.Image(0.58).subtract(urbanDrag.multiply(0.35)).add(corridorGain.multiply(0.28))
                        .add(terrainWindExposure.multiply(0.22)).add(coastalWindPenetration.multiply(0.08))
                        .clamp(0.12, 0.95))
    WS_1p5m = WS_local_10m.multiply(pedestrianFactor).clamp(*P["ws1p5_clamp"]).rename("WS_1p5m")

    # ---------------------------------------------------------- PART 9 MRT
    sigma = ee.Number(5.670374419e-8)
    alphaHumanSW = ee.Number(0.70)
    skyViewFactor = svfProxy.clamp(0.15, 0.95)
    enclosureFactor = one.subtract(skyViewFactor)
    treeViewFactor = treeHeightEffect.multiply(0.30).add(shadeProxy.multiply(0.20)).clamp(0, 0.35)
    wallViewFactor = (enclosureFactor.multiply(0.65).add(buildingDensity30m.multiply(0.20))
                      .add(buildingHeightEffect.multiply(0.15)).subtract(treeViewFactor.multiply(0.30))
                      .clamp(0, 0.65))
    groundViewFactor = (one.subtract(skyViewFactor).subtract(wallViewFactor).subtract(treeViewFactor)
                        .clamp(0.10, 0.55))
    vfSum = skyViewFactor.add(wallViewFactor).add(groundViewFactor).add(treeViewFactor)
    VF_sky, VF_wall = skyViewFactor.divide(vfSum), wallViewFactor.divide(vfSum)
    VF_ground, VF_tree = groundViewFactor.divide(vfSum), treeViewFactor.divide(vfSum)

    T_ground_C = clampx(Ta_10m.add(LST_anomaly.multiply(0.85)).add(heatStorageProxy.multiply(5.0))
                        .subtract(moistureProxy.multiply(2.0)).subtract(shadeProxy.multiply(2.5))
                        .subtract(Albedo.multiply(1.5)), P["t_ground_clamp"])
    T_wall_C = clampx(Ta_10m.add(LST_anomaly.multiply(0.55)).add(heatStorageProxy.multiply(4.0))
                      .add(buildingDensity30m.multiply(1.5)).add(buildingHeightEffect.multiply(1.0))
                      .subtract(shadeProxy.multiply(1.5)), P["t_wall_clamp"])
    T_tree_C = clampx(Ta_10m.subtract(moistureProxy.multiply(2.0)).subtract(vegVitality.multiply(1.0)),
                      P["t_tree_clamp"])
    T_sky_K = LW_f.divide(sigma.multiply(0.95)).pow(0.25)

    SW_reflected_ground = SW_f.multiply(Albedo).multiply(VF_ground).multiply(alphaHumanSW)
    SW_reflected_wall = SW_f.multiply(Albedo).multiply(VF_wall).multiply(enclosureFactor).multiply(alphaHumanSW)
    if P.get("mrt_sw") == "projected":
        # Standard human-body shortwave (Fanger 1970 projected area factor, as in RayMan and
        # SOLWEIG). The JS applied all horizontal global radiation x 0.9 to the body, as if a
        # person were a flat roof: about 2.5 times too much direct sun at high solar elevation.
        #   direct normal I = (SW - D) / cos(zenith), D from the Erbs et al. (1982) diffuse
        #   fraction of the clearness index kt = SW / S0
        #   direct on body  = f_p x I x (1 - shade), f_p = 0.308 cos(h (0.998 - h^2 / 50000))
        #   diffuse on body = D x VF_sky (isotropic sky)
        #   reflected ground and wall terms unchanged
        elev, cosz, s0 = solar_geometry(utc_iso, *studyArea.geometry().centroid(100)
                                        .coordinates().getInfo())
        if cosz > 0.02:
            kt = SW_f.divide(s0).clamp(0, 1)
            kd = (kt.expression(
                "k <= 0.22 ? 1 - 0.09 * k : (k <= 0.80 ? 0.9511 - 0.1604 * k + 4.388 * k ** 2"
                " - 16.638 * k ** 3 + 12.336 * k ** 4 : 0.165)", {"k": kt}))
            D = SW_f.multiply(kd)
            I_dir = SW_f.subtract(D).divide(max(cosz, 0.05))
            fp = 0.308 * math.cos(math.radians(elev * (0.998 - elev ** 2 / 50000)))
        else:                              # sun below the horizon: all diffuse
            D, I_dir, fp = SW_f, ee.Image(0), 0.0
        SW_direct_human = I_dir.multiply(fp).multiply(one.subtract(shadeProxy)).multiply(alphaHumanSW)
        SW_diffuse_human = D.multiply(VF_sky).multiply(alphaHumanSW)
        SW_total_human = (SW_direct_human.add(SW_diffuse_human).add(SW_reflected_ground)
                          .add(SW_reflected_wall).clamp(0, 1200))
    else:                                  # main_workflow.js
        SW_direct_human = (SW_f.multiply(ee.Image.constant(P["solar_elevation_factor"]))
                           .multiply(one.subtract(shadeProxy)).multiply(alphaHumanSW))
        SW_total_human = SW_direct_human.add(SW_reflected_ground).add(SW_reflected_wall).clamp(0, 1200)

    LW_total = (T_sky_K.pow(4).multiply(sigma).multiply(0.95).multiply(VF_sky)
                .add(T_ground_C.add(273.15).pow(4).multiply(sigma).multiply(0.95).multiply(VF_ground))
                .add(T_wall_C.add(273.15).pow(4).multiply(sigma).multiply(0.94).multiply(VF_wall))
                .add(T_tree_C.add(273.15).pow(4).multiply(sigma).multiply(0.97).multiply(VF_tree)))
    if P.get("mrt_sw") == "projected":
        LW_total = LW_total.multiply(0.95)  # absorbed longwave = body emissivity x incident (JS: x 1)
    MRT_10m = clampx(SW_total_human.add(LW_total).divide(sigma.multiply(0.95)).pow(0.25)
                     .subtract(273.15), P["mrt_clamp"]).rename("MRT_10m")
    if MRT_SCHEME == "cylinder":
        lon_c, lat_c = studyArea.geometry().centroid(100).coordinates().getInfo()
        elev, cosz, s0 = solar_geometry(utc_iso, lon_c, lat_c)
        if cosz > 0.02:
            kt = SW_f.divide(s0).clamp(0, 1)
            kd = (kt.expression(
                "k <= 0.22 ? 1 - 0.09 * k : (k <= 0.80 ? 0.9511 - 0.1604 * k + 4.388 * k ** 2"
                " - 16.638 * k ** 3 + 12.336 * k ** 4 : 0.165)", {"k": kt}))
            D = SW_f.multiply(kd)
            I_dir = SW_f.subtract(D).divide(max(cosz, 0.05))
            fp = 0.308 * math.cos(math.radians(elev * (0.998 - elev ** 2 / 50000)))
            az = solar_azimuth(utc_iso, lon_c, lat_c)
            dsm = (dem.resample("bilinear").add(buildingHeight.multiply(combinedBuildingMask)).add(canopyHeight)
                   .reproject(ee.Projection("EPSG:3857").atScale(OUT_SCALE)))
            lit = (ee.Terrain.hillShadow(dsm, az, 90 - elev, 150, False).unmask(1)
                   .multiply(one.subtract(canopyHeight.gt(2).multiply(0.9))).rename("sunlit"))
            heat = kt.divide(0.75).clamp(0, 1).multiply(8.0)          # sunlit ground warms up to 8 K above shade
        else:
            D, I_dir, fp, lit, heat = SW_f, ee.Image(0), 0.0, ee.Image(0), ee.Image(0)
        svf = skyViewFactor
        f_tree = treeViewFactor.divide(treeViewFactor.add(wallViewFactor).max(0.001))
        T_obs4 = (T_tree_C.add(273.15).pow(4).multiply(f_tree)
                  .add(T_wall_C.add(273.15).pow(4).multiply(one.subtract(f_tree))))
        T_gnd = T_ground_C.add(lit.multiply(heat)).add(273.15)
        # v10: walls, trees and ground emit with emissivity 0.95 and reflect the other 5% of incoming longwave,
        # so an isothermal enclosure returns MRT equal to its temperature
        L_up = LW_f.multiply(svf).add(T_obs4.multiply(sigma).multiply(0.95).add(LW_f.multiply(0.05))
                                      .multiply(one.subtract(svf)))
        L_dn = T_gnd.pow(4).multiply(sigma).multiply(0.95).add(L_up.multiply(0.05))
        K_up = D.multiply(svf)
        K_dn = Albedo.multiply(I_dir.multiply(max(cosz, 0.0)).multiply(lit).add(D))
        R = (L_up.multiply(0.5).add(L_dn.multiply(0.5))
             .add(K_up.multiply(0.5).add(K_dn.multiply(0.5)).add(I_dir.multiply(fp).multiply(lit)).multiply(0.70 / 0.97)))
        MRT_10m = clampx(R.divide(sigma).pow(0.25).subtract(273.15), P["mrt_clamp"]).rename("MRT_10m")
    dTmrt_10m = MRT_10m.subtract(Ta_10m).clamp(-30, 70).rename("dTmrt_10m")

    # ---------------------------------------------------------- PART 10 / 11 UTCI
    Pa_10m = Ta_10m.expression("0.1 * (6.112 * exp((17.62 * Ta) / (243.12 + Ta))) * (RH / 100)",
                               {"Ta": Ta_10m, "RH": RH_10m}).clamp(0, 5).rename("Pa_10m")
    ws_src = WS_1p5m if P["utci_wind"] == "WS_1p5m" else WS_local_10m
    WS_utci = ws_src.clamp(0.5, 17).rename("WS_utci")
    dTmrt_utci = MRT_10m.subtract(Ta_10m).clamp(-30, 70)
    UTCI_10m = Ta_10m.expression(UTCI_POLY, {"Ta": Ta_10m, "va": WS_utci, "D_Tmrt": dTmrt_utci,
                                             "Pa_kPa": Pa_10m}).clamp(-50, 70).rename("UTCI_10m")
    UTCI_class = (ee.Image(0)
                  .where(UTCI_10m.lt(-40), 1)
                  .where(UTCI_10m.gte(-40).And(UTCI_10m.lt(-27)), 2)
                  .where(UTCI_10m.gte(-27).And(UTCI_10m.lt(-13)), 3)
                  .where(UTCI_10m.gte(-13).And(UTCI_10m.lt(0)), 4)
                  .where(UTCI_10m.gte(0).And(UTCI_10m.lt(9)), 5)
                  .where(UTCI_10m.gte(9).And(UTCI_10m.lt(26)), 6)
                  .where(UTCI_10m.gte(26).And(UTCI_10m.lt(32)), 7)
                  .where(UTCI_10m.gte(32).And(UTCI_10m.lt(38)), 8)
                  .where(UTCI_10m.gte(38).And(UTCI_10m.lt(46)), 9)
                  .where(UTCI_10m.gte(46), 10)
                  .updateMask(UTCI_10m.mask()).rename("UTCI_class"))

    stack = ee.Image.cat([
        UTCI_10m, UTCI_class, Ta_10m, RH_10m, WS_local_10m, WS_1p5m, WS_utci, MRT_10m, dTmrt_10m,
        Pa_10m, Ta_10m_raw, RH_10m_raw,
        ee.Image.constant(1).rename("in_city"),
    ]).toFloat().clip(studyArea)
    if MRT_SCHEME == "cylinder":   # v10: direct-beam sunlit fraction at the target hour (for the pyranometer check)
        stack = stack.addBands(ee.Image(lit).toFloat().rename("sunlit").clip(studyArea))
    if EXPORT_PREDICTORS:
        stack = stack.addBands(predictorsTa.regexpRename("^", "p_").toFloat().clip(studyArea))
    if EXPORT_PARTS:
        stack = stack.addBands(ee.Image.cat([
            corrWarm, corrCool, corrRH,
            # the individual weighted terms of localHeatCorrection (warm: +, cool: subtracted)
            LST_anomaly.multiply(0.35).rename("t_lst"), heatStorageProxy.multiply(1.40).rename("t_heat"),
            NDBI.unitScale(-0.2, 0.5).clamp(0, 1).multiply(0.80).rename("t_ndbi"),
            NDVI.unitScale(0.1, 0.8).clamp(0, 1).multiply(0.60).rename("t_ndvi"),
            moistureProxy.multiply(0.40).rename("t_moist"), coastalExposure.multiply(0.35).rename("t_coast"),
            ventilationPotential.multiply(0.35).rename("t_vent"), shadeProxy.multiply(0.50).rename("t_shade"),
            weather_factor(utc_iso, studyArea, Ta_forc, Td_f, SW_f, LW_f, "kt").unmask(1).rename("f_kt"),
            weather_factor(utc_iso, studyArea, Ta_forc, Td_f, SW_f, LW_f, "cloud").unmask(1).rename("f_cloud"),
            Td_f.resample("bilinear").rename("Td_10m"),
            ee.Image.constant(ta_lim[0]).rename("ta_lo"), ee.Image.constant(ta_lim[1]).rename("ta_hi"),
        ]).toFloat().clip(studyArea))

    # samples with coordinates and parent ERA5-Land cell, for tools/grouped_cv.py (Part 15)
    def tag(f):
        xy = f.geometry().coordinates()
        lon, lat = ee.Number(xy.get(0)), ee.Number(xy.get(1))
        cell = lon.divide(0.1).round().multiply(100000).add(lat.divide(0.1).round())
        return f.set({"lon": lon, "lat": lat, "era5_cell": cell})  # keep the point: asset tables need geometry

    samples_ta = sample(Ta_coarse, "Ta", predictorsTa.select(bandsTa), True).map(tag)
    samples_rh = sample(RH_coarse, "RH", predictorsRH.select(bandsRH), True).map(tag)

    def explain(rf, name):
        ex = ee.Dictionary(rf.explain())
        imp = ee.Dictionary(ex.get("importance"))
        oob = ex.get("outOfBagErrorEstimate")
        return ee.FeatureCollection(imp.keys().map(lambda k: ee.Feature(None, {
            "target": name, "predictor": k, "importance": imp.get(k), "oob_rmse": oob})))

    rf_table = explain(rf_ta, "Ta").merge(explain(rf_rh, "RH"))
    return stack, era5_coarse, samples_ta, samples_rh, rf_table

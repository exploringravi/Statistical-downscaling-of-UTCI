# Statistical Downscaling of UTCI

Neighbourhood-scale mapping of the Universal Thermal Climate Index (UTCI) at 10 m resolution, by statistical downscaling of ERA5-Land reanalysis using Earth observation and urban morphology predictors.

This repository contains the Google Earth Engine workflow and post-processing code supporting the manuscript:

> Pandey, R. K., Pisello, A. L., Cureau, R. J., Papadopoulos, P., Kyprianou, I., & Carlucci, S. *Neighbourhood-Scale UTCI Mapping by Statistical Downscaling of Reanalysis Using Earth Observation Across Three European Cities*. Submitted to GIScience & Remote Sensing.


## Overview

The workflow redistributes coarse ERA5-Land atmospheric forcing (approximately 9 km) according to high-resolution environmental controls, producing 10 m UTCI maps. It does not generate independent 10 m atmospheric observations.

Processing stages:

1. Acquisition of ERA5-Land meteorological inputs
2. Predictor generation from Sentinel-2, MODIS, building morphology, canopy height, and terrain
3. Training dataset generation on a 1 km sampling grid over a macro-regional domain
4. Random Forest downscaling of air temperature and relative humidity
5. Morphology-informed wind speed estimation
6. Mean radiant temperature estimation via a simplified radiation balance
7. UTCI computation using the Brode et al. (2012) polynomial
8. Thermal stress classification and analysis

Applied to three European cities: Limassol (Cyprus), Perugia (Italy), and Aachen (Germany).

## Repository structure

```
gee/                 Google Earth Engine scripts (JavaScript)
  [main_workflow.js]     End-to-end downscaling and UTCI computation
  [predictors.js]        Predictor and environmental proxy generation
python/              Post-processing and validation (Python)
  [validation.py]        Model versus observation statistics
  [figures.py]           Figure generation
data/                Derived validation tables
LICENSE
README.md
```

## Input datasets

All inputs are freely and globally available.

| Dataset | Variable | Resolution | Source |
|---|---|---|---|
| ERA5-Land | Air temperature, dew point, wind components, radiation | ~9 km | Copernicus Climate Data Store |
| Sentinel-2 L2A | NDVI, EVI, NDWI, NDMI, NDBI, albedo | 10 m | Copernicus |
| MODIS MOD11A1 | Land surface temperature | 1 km | NASA LP DAAC |
| Global Building Atlas | Building height and density | 10 m | Zhu et al. (2025) |
| ETH Global Canopy Height | Canopy height | 10 m | Lang et al. (2023) |
| Copernicus DEM | Elevation, slope, aspect, TPI | 30 m | ESA (2022) |

## Running the workflow

**Requirements:** a Google Earth Engine account, and Python 3.9 or later for post-processing.

1. Open the scripts in `gee/` in the Earth Engine Code Editor.
2. Set the region of interest, the target date, and the target hour at the top of the main script.
3. Run the workflow. Outputs export to Google Drive as 10 m GeoTIFFs.
4. Run the scripts in `python/` for validation statistics and figures.

Install Python dependencies with:

```
pip install -r requirements.txt
```

## Data availability

Derived model outputs and validation comparison tables are included in `data/`.

Field observation data are subject to third-party restrictions. The Perugia and Limassol mobile campaign records were collected with EnviWear systems by collaborators at the University of Perugia. The Nicosia fixed station records were collected under the RE-ACT Schools project. [Adjust this section to state exactly what you are releasing, and name the contact point for the restricted records.]

## Citation

If you use this code, please cite both the paper and the archived release.

```bibtex
@software{pandey_utci_downscaling,
  author  = {Pandey, Ravi Kumar},
  title   = {Statistical Downscaling of UTCI},
  year    = {2026},
  url     = {https://github.com/exploringravi/Statistical-downscaling-of-UTCI},
  doi     = {(https://doi.org/10.5281/zenodo.21486895)}
}
```

## License

[MIT or Apache 2.0] for code. Derived data in `data/` are released under CC BY 4.0.

## Funding

This work was carried out within the MuSIC Doctoral Network, funded by the European Union's Horizon Europe Marie Sklodowska-Curie Actions programme under Grant Agreement No. 101073357.

## Contact

Ravi Kumar Pandey, Energy, Environment and Water Research Center (EEWRC), The Cyprus Institute, Nicosia, Cyprus. r.pandey@cyi.ac.cy

"""
utci_status.py
Shows the latest state of every UTCI pipeline task in Earth Engine, per city and run.

  python utci_status.py
"""

import ee

from config import CITIES, MACRO_TAG, PROJECT, RUNS, run_cities
from utci_pipeline import RUN_VERSION, run_tag

ee.Initialize(project=PROJECT)
latest = {}
for t in ee.data.getTaskList():          # newest first; a finished run beats a later cancelled copy
    d = t.get("description")
    if d not in latest or (t.get("state") == "COMPLETED" and latest[d].get("state") != "COMPLETED"):
        latest[d] = t

for city in CITIES:
    for r in RUNS:
        date, hour = r[:2]
        if city not in run_cities(r):
            continue
        tag = run_tag(date, hour)
        print(f"\n{city} {tag}")
        names = [f"utci_samples_{city}_Ta_{tag}", f"utci_samples_{city}_RH_{tag}",
                 f"UTCI_{city}_{tag}_components", f"ERA5_{city}_{tag}_coarse", f"UTCI_{city}_{tag}_rf",
                 f"GroupedCV_Samples_Ta_{MACRO_TAG[city]}_{tag}", f"GroupedCV_Samples_RH_{MACRO_TAG[city]}_{tag}"]
        for n in names:
            n = f"{n}_{RUN_VERSION}"
            t = latest.get(n)
            if not t:
                print(f"  {'not submitted':12s} {n}")
                continue
            err = t.get("error_message", "")
            print(f"  {t.get('state', '?'):12s} {n}" + (f"  ({err})" if err else ""))
        # tiled map exports (Perugia): UTCI_<city>_<tag>_components_t*_<version>
        stem = f"UTCI_{city}_{tag}_components_t"
        for d in sorted(k for k in latest if k and k.startswith(stem) and k.endswith(f"_{RUN_VERSION}")):
            t = latest[d]
            err = t.get("error_message", "")
            print(f"  {t.get('state', '?'):12s} {d}" + (f"  ({err})" if err else ""))

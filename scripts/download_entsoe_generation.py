"""Download Belgian ENTSO-E data for one year into data/ (Phase 3 inputs).

Run from the repo root (paths are relative):

    python scripts/download_entsoe_generation.py 2022
    python scripts/download_entsoe_generation.py 2023

Each dataset goes to data/entsoe_be_{year}_{name}.csv. Existing files are skipped, never
overwritten (ENTSO-E revises data, so a re-download can change earlier results); delete a
file to re-download it. A dataset that fails is reported and the others still run.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2023

with Path("entsoe_tokens.json").open("r", encoding="utf-8") as f:
    client = EntsoePandasClient(api_key=json.load(f)["api_key"])

# Brussels-time calendar year, as for the 2023 generation file
start = pd.Timestamp(f"{YEAR}-01-01", tz="Europe/Brussels")
end = pd.Timestamp(f"{YEAR + 1}-01-01", tz="Europe/Brussels")

QUERIES = {
    # actual generation per production type
    "generation": lambda: client.query_generation("BE", start=start, end=end, psr_type=None),
    # actual total load
    "load": lambda: client.query_load("BE", start=start, end=end),
    # physical cross-border flows per neighbour, into and out of BE
    "imports": lambda: client.query_physical_crossborder_allborders(
        "BE", start=start, end=end, export=False),
    "exports": lambda: client.query_physical_crossborder_allborders(
        "BE", start=start, end=end, export=True),
    # actual generation per unit (ENTSO-E publishes units >= 100 MW)
    "generation_per_plant": lambda: client.query_generation_per_plant("BE", start=start, end=end),
    # installed capacity per unit
    "capacity_per_unit": lambda: client.query_installed_generation_capacity_per_unit(
        "BE", start=start, end=end),
}

Path("data").mkdir(parents=True, exist_ok=True)
failed = []
for name, query in QUERIES.items():
    out = Path(f"data/entsoe_be_{YEAR}_{name}.csv")
    if out.exists():
        print(f"skip {out} (exists; delete it to re-download)")
        continue
    print(f"downloading {name} {YEAR} ...", flush=True)
    try:
        data = query()
    except Exception as e:  # keep going: one failing query should not block the others
        print(f"FAILED {name}: {type(e).__name__}: {e}")
        failed.append(name)
        continue
    # write to a temp file, then rename: an interrupted write never leaves a partial CSV that
    # the next run would skip as done
    tmp = out.with_suffix(".csv.partial")
    data.to_csv(tmp)
    tmp.replace(out)
    print(f"saved {out} ({len(data)} rows)")

if failed:
    sys.exit(f"failed: {failed} (re-run to retry only these)")

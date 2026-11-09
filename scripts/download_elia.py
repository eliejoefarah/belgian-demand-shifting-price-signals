"""Download Elia Open Data (TSO measurements + forecasts) for Phase 4 into data/.

Run from the repo root (paths are relative):

    python scripts/download_elia.py 2021 2022 2023

No API key needed (public Elia Open Data, Explore API v2.1). One file per dataset and
calendar year (UTC): data/elia_{name}_{year}.csv, timestamps in UTC, 15-min resolution.
The files start with a UTF-8 byte-order mark: read them with encoding="utf-8-sig".
Existing files are skipped, never overwritten (Elia revises data; delete a file to
re-download it). Writes go to a temp file then a rename, and a dataset that fails is
reported while the others still run (same conventions as download_entsoe_generation.py).

Datasets (field lists checked 9 Nov 2026, temp_phase4_log.md 1):
- wind  = ods031 "Wind power production estimation and forecast on Belgian grid
  (Historical)", from 2015. Five rows per quarter-hour (Offshore/Federal/Elia and
  Onshore x Flanders/Wallonia x Elia/Dso); there is no Belgian total row, so sum them.
  Columns include measured, dayahead11hforecast (+ confidence10/90) = the day-ahead
  11AM forecast, usable before the 14:00 MPC re-plan; dayaheadforecast = 6PM (NOT
  usable: after 14:00); weekaheadforecast; mostrecentforecast; monitoredcapacity.
- solar = ods032 "Photovoltaic power production estimation and forecast on Belgian grid
  (Historical)", from Oct 2020. Only region = "Belgium" is kept (13 sub-regions dropped).
  Same forecast columns as wind (dayahead11hforecast usable, 6PM not).
- load  = ods001 "Measured and forecasted total load on the Belgian grid (Historical
  data)", from 2015. totalload; dayaheadforecast = 6PM (NOT usable before 14:00);
  weekaheadforecast; mostrecentforecast.
- net_position_dayahead = ods023 "Day-ahead implicit net position (Belgium's balance)",
  from 2019, 15-min, one row per quarter-hour: implicitnetposition = "Net sum of day-ahead
  nominations of the implicit capacity allocated for energy exchanges for Belgium" (the
  day-ahead market-coupling result). Candidate import forecast; publication time vs the
  14:00 cutoff TO CHECK (temp_phase4_log.md 6).
- generation_schedule_dayahead = ods034 "Day-ahead schedule of power generation -
  aggregated by fuel type", 2015 to 21/05/2024 (ends at Elia's ICAROS go-live), one row
  per fuel and quarter-hour: dayaheadgenerationschedule by fuelcode, from ARP nominations
  of CIPU units "submitted on Day -1". Candidate gas forecast; submission time vs 14:00
  TO CHECK.
"""

import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets"
DATASETS = {
    "wind": ("ods031", None),
    "solar": ("ods032", 'region = "Belgium"'),
    "load": ("ods001", None),
    "net_position_dayahead": ("ods023", None),
    "generation_schedule_dayahead": ("ods034", None),
}
YEARS = [int(a) for a in sys.argv[1:]] or [2021, 2022, 2023]


def export_url(dataset_id: str, year: int, extra_where: str | None) -> str:
    where = f'datetime >= "{year}-01-01T00:00:00Z" and datetime < "{year + 1}-01-01T00:00:00Z"'
    if extra_where:
        where += f" and {extra_where}"
    params = {"where": where, "order_by": "datetime", "timezone": "UTC", "delimiter": ","}
    return f"{BASE}/{dataset_id}/exports/csv?{urllib.parse.urlencode(params)}"


Path("data").mkdir(parents=True, exist_ok=True)
failed = []
for year in YEARS:
    for name, (dataset_id, extra_where) in DATASETS.items():
        out = Path(f"data/elia_{name}_{year}.csv")
        if out.exists():
            print(f"skip {out} (exists; delete it to re-download)")
            continue
        url = export_url(dataset_id, year, extra_where)
        print(f"downloading {name} ({dataset_id}) {year} ...", flush=True)
        tmp = out.with_suffix(".csv.partial")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "thesis-data-download"})
            with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            n_rows = sum(1 for _ in tmp.open("rb")) - 1  # minus header
            if n_rows <= 0:
                raise ValueError("empty export")
        except Exception as e:  # keep going: one failing dataset should not block the others
            print(f"FAILED {name} {year}: {type(e).__name__}: {e}")
            tmp.unlink(missing_ok=True)
            failed.append(f"{name} {year}")
            continue
        tmp.replace(out)
        print(f"saved {out} ({n_rows} rows)")

if failed:
    sys.exit(f"failed: {failed} (re-run to retry only these)")

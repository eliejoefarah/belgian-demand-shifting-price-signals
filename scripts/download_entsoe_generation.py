"""Download Belgian ENTSO-E data for one year into data/ (Phase 3 and 4 inputs).

Run from the repo root (paths are relative):

    python scripts/download_entsoe_generation.py 2022
    python scripts/download_entsoe_generation.py 2023

Datasets: generation per type, load, imports/exports, generation and capacity per unit,
unavailability of generation units (outages), day-ahead prices, installed capacity per
production type (all added 9 Nov 2026) and the day-ahead load forecast (Phase 4, 9 Nov 2026).
Each dataset goes to data/entsoe_be_{year}_{name}.csv. Existing files are skipped, never
overwritten (ENTSO-E revises data, so a re-download can change earlier results); delete a
file to re-download it. A dataset that fails is reported and the others still run.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient, EntsoeRawClient
from entsoe.mappings import PSRTYPE_MAPPINGS

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2023

with Path("entsoe_tokens.json").open("r", encoding="utf-8") as f:
    API_KEY = json.load(f)["api_key"]
client = EntsoePandasClient(api_key=API_KEY)

# Brussels-time calendar year, as for the 2023 generation file
start = pd.Timestamp(f"{YEAR}-01-01", tz="Europe/Brussels")
end = pd.Timestamp(f"{YEAR + 1}-01-01", tz="Europe/Brussels")
# The same instants in UTC. Per-plant data (A73) is limited to 1 day per request and
# entsoe-py cuts the days in the timezone of `start`: in Brussels time the autumn DST day
# lasts 25 h and ENTSO-E answers 400 Bad Request (found 8 Nov 2026 on 2022-10-30), which
# fails the whole year. UTC days are always 24 h.
start_utc, end_utc = start.tz_convert("UTC"), end.tz_convert("UTC")


def capacity_per_unit():
    """Installed capacity per unit (A71), one row per unit valid in YEAR.

    entsoe-py 0.8.1's parser crashes on this document ("'NoneType' object has no
    attribute 'text'", 8 Nov 2026) although the XML is fine, so it is parsed here.
    ENTSO-E returns one yearly Period per unit and may include the following year too;
    only Periods starting in YEAR (Brussels time) are kept.
    """
    xml = EntsoeRawClient(api_key=API_KEY).query_installed_generation_capacity_per_unit(
        "BE", start=start_utc, end=end_utc)
    root = ET.fromstring(xml)
    ns = {"n": root.tag.split("}")[0].strip("{")}
    rows = []
    for ts in root.findall("n:TimeSeries", ns):
        def text(path):
            el = ts.find(path, ns)
            return el.text if el is not None else None
        psr = text("n:MktPSRType/n:psrType")
        for period in ts.findall("n:Period", ns):
            p_start = pd.Timestamp(period.find("n:timeInterval/n:start", ns).text)
            if p_start.tz_convert("Europe/Brussels").year != YEAR:
                continue
            qty = period.find("n:Point/n:quantity", ns)
            rows.append({
                "eic": text("n:registeredResource.mRID"),
                "name": text("n:registeredResource.name"),
                "psr_type": psr,
                "production_type": PSRTYPE_MAPPINGS.get(psr, psr),
                "voltage_kv": text("n:MktPSRType/n:voltage_PowerSystemResources.highVoltageLimit"),
                "capacity_mw": float(qty.text) if qty is not None else None,
                "valid_from": p_start,
            })
    if not rows:
        raise ValueError(f"no capacity-per-unit periods for {YEAR} in the response")
    return pd.DataFrame(rows).sort_values(["production_type", "name"]).reset_index(drop=True)

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
    # actual generation per unit (ENTSO-E publishes units >= 100 MW); UTC bounds, see above.
    # The index comes back in Brussels time.
    "generation_per_plant": lambda: client.query_generation_per_plant(
        "BE", start=start_utc, end=end_utc),
    # installed capacity per unit (own parser, see capacity_per_unit)
    "capacity_per_unit": capacity_per_unit,
    # unavailability of generation units (15.1.A-D, doc type A80): one row per outage period
    # with production_resource_id (= unit EIC), businesstype (Planned maintenance / Unplanned
    # outage), start, end, avail_qty (MW still available), nominal_power, plus docstatus,
    # mrid and revision. Saved raw, ALL revisions and statuses: the dispatch model keeps the
    # latest revision per mrid and drops cancelled/withdrawn ones. entsoe-py pages through
    # the 200-documents-per-request limit and splits the year itself.
    "unavailability": lambda: client.query_unavailability_of_generation_units(
        "BE", start=start, end=end),
    # day-ahead price, EUR/MWh (validation: model marginal cost vs price). Hourly in 2022-23.
    "day_ahead_prices": lambda: client.query_day_ahead_prices("BE", start=start, end=end),
    # installed capacity per production type (14.1.A, one value per type and year), MW:
    # the whole-fleet x-axis of the Martin et al. Fig. 1a figure (temp_phase3_log.md 5.55)
    "installed_capacity_per_type": lambda: client.query_installed_generation_capacity(
        "BE", start=start, end=end, psr_type=None),
    # day-ahead total load forecast (6.1.B, documentType A65, processType A01 = day-ahead),
    # MW. Phase 4 leakage check: due "no later than two hours before the gate closure of
    # the day-ahead market" but updatable; compare with Elia's 6PM forecast (ods001) to see
    # which version ENTSO-E keeps (temp_phase4_log.md 1, 6).
    "load_forecast_dayahead": lambda: client.query_load_forecast(
        "BE", start=start, end=end, process_type="A01"),
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

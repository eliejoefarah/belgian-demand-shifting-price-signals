"""
Load an ENTSO-E Generation per Production Type CSV into the flat
generation-mix shape price_signals.carbon_intensity.compute_intensity expects:
index = tz-aware DatetimeIndex, columns = production-type names (matching
emission_factors.py keys), values = MW of ACTUAL GENERATION only.

entsoe-py's query_generation (scripts/download_entsoe_generation.py)
writes a TWO-ROW header: (production_type, "Actual Aggregated" | "Actual
Consumption"). "Actual Aggregated" is generation output. "Actual Consumption"
only appears for storage/auxiliary draw (e.g. Hydro Pumped Storage charging,
small Wind Offshore station auxiliary load), that is DEMAND, not generation.
It is dropped here since it is electricity drawn from other generators
that are already counted in their own "Actual Aggregated" columns.

"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

GENERATION_LABEL = "Actual Aggregated"
CONSUMPTION_LABEL = "Actual Consumption"


def load_generation_mix(path) -> pd.DataFrame:
    path = Path(path)

    raw = pd.read_csv(path, index_col=0, header=[0, 1])
    is_generation = raw.columns.get_level_values(1) == GENERATION_LABEL
    df = raw.loc[:, is_generation]
    df.columns = df.columns.get_level_values(0)

    # parse_dates=True silently fails to build a proper tz-aware DatetimeIndex
    # here: the timestamps carry mixed UTC offsets (+01:00 CET / +02:00 CEST)
    # across the DST transition, so pandas falls back to plain strings, which
    # then sort wrong (not monotonic)
    # reindex in fleet_harness.co2. Parse explicitly instead.
    df.index = pd.to_datetime(
        df.index, 
        utc=True
    ).tz_convert("Europe/Brussels")
    return df

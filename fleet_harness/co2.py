"""
CO2 metric for fleet_harness rungs.

Pairs a simulated run's per-timestep grid electricity OFFTAKE (pulled straight out
of microgrid.power_hist -- the same source belgian_dwellings.get_consumption sums
internally but doesn't expose) with the Belgian grid carbon-intensity signal built
in price_signals.carbon_intensity from ENTSO-E generation data + emission_factors.py.

Only offtake (import from the grid) counts against a rung -- injection (export) is
NOT credited as avoided emissions here, since crediting it requires a displacement
assumption (what generation it actually offsets) that is a separate methodological
choice, unsettled by this module.

"""
from __future__ import annotations

import pandas as pd

from price_signals.carbon_intensity import compute_intensity

PUBLIC_GRID = "PublicGrid_0"


def get_grid_offtake_kwh(microgrid, public_grid_name: str = PUBLIC_GRID) -> pd.Series:
    """Per-timestep grid electricity OFFTAKE (kWh drawn from the grid) for one
    simulated run. Power.electrical sign convention (matches get_consumption):
    positive = importing/offtake, negative = exporting/injection -- injection is
    clipped out here, not counted.
    """
    node0 = microgrid.power_hist[0]  # single-node house configs
    dts = pd.DatetimeIndex(node0["datetime"]).tz_convert("Europe/Brussels")
    kw = pd.Series([p.electrical for p in node0[public_grid_name]], index=dts)

    step_h = microgrid.time_step.total_seconds() / 3600
    offtake_kwh = kw.clip(lower=0) * step_h
    offtake_kwh.name = "grid_offtake_kwh"
    return offtake_kwh


def compute_rung_co2(microgrid, mix: pd.DataFrame, factor_set: str = "direct") -> dict:
    """Total CO2 attributable to one rung's grid offtake, plus the offtake-weighted
    average intensity actually paid (gCO2/kWh) -- the number that moves when a
    price/EMS signal shifts load toward cleaner hours without changing total kWh.

    mix: ENTSO-E generation-mix DataFrame, real or synthetic (see
    price_signals.carbon_intensity.compute_intensity). factor_set: "direct" or
    "lifecycle" -- pass explicitly, never hardcode the choice at this layer either.
    """
    intensity = compute_intensity(mix, factor_set=factor_set)  # gCO2/kWh
    offtake_kwh = get_grid_offtake_kwh(microgrid)

    matched = intensity.reindex(offtake_kwh.index, method="ffill")
    if matched.isna().any():
        missing = matched[matched.isna()]
        raise ValueError(
            f"{len(missing)} offtake timestamp(s) fall outside the intensity "
            f"signal's coverage, e.g. {missing.index[0]!r} -- check the mix "
            "data's date range against the simulated year"
        )

    co2_g = float((offtake_kwh * matched).sum())
    total_kwh = float(offtake_kwh.sum())
    return {
        "co2_kg": co2_g / 1000,
        "co2_g": co2_g,
        "grid_offtake_kwh": total_kwh,
        "avg_intensity_paid_gCO2_per_kWh": co2_g / total_kwh if total_kwh else float("nan"),
        "factor_set": factor_set,
    }

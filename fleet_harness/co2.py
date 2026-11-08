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

# How far past its last sample an hourly intensity may be forward-filled: its own hour,
# plus 1 h because the ENTSO-E year is a Brussels year (ends 2024-01-01 00:00 Brussels =
# 2023-12-31 23:00 UTC) while the simulated year is a UTC year (last step 23:45 UTC). So the
# last hour of a 2023 run reuses the 23:00 Brussels intensity. Anything further (a mix file
# for the wrong year, a gap) raises.
INTENSITY_MAX_AGE = pd.Timedelta("2h")


def ffill_onto(values: pd.Series, index: pd.DatetimeIndex, max_age: pd.Timedelta, what: str) -> pd.Series:
    """values forward-filled onto index; raises if any timestamp of index has no sample of
    values at most max_age before it (before the start, past the end, or in a gap).

    Plain reindex(method="ffill") only catches timestamps before the start: past the end
    it silently repeats the last value, e.g. a 2022 mix applied to a 2023 run.
    """
    matched = values.reindex(index, method="ffill", tolerance=max_age)
    if matched.isna().any():
        missing = matched.index[matched.isna()]
        raise ValueError(
            f"{len(missing)} timestamp(s) fall outside the {what} coverage "
            f"({values.index.min()} .. {values.index.max()}, max {max_age} past a sample), "
            f"e.g. {missing[0]!r}; check its date range against the simulated year"
        )
    return matched


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


def co2_from_grid_kw(grid_kw: pd.Series, time_step_h: float, intensity: pd.Series) -> dict:
    """Average-factor CO2 of one house-run from its grid power (saved fleet results).

    grid_kw: PublicGrid_0 kW, + offtake / - injection, tz-aware index at step start
    (the `grid_kw` column of a fleet power table). intensity: gCO2/kWh from
    compute_intensity, forward-filled onto the grid timestamps.

    Injection is reported two ways, kept separate (see the module docstring):
      uncredited  co2_kg = offtake only (the default metric, as compute_rung_co2)
      credited    co2_net_kg = offtake - injection x the grid intensity at that time
                  (assumes injected kWh displace the average mix; a sensitivity only)
    """
    matched = ffill_onto(
        intensity, grid_kw.index.tz_convert(intensity.index.tz), INTENSITY_MAX_AGE, "intensity"
    ).to_numpy()
    offtake_kwh = grid_kw.clip(lower=0).to_numpy() * time_step_h
    injection_kwh = (-grid_kw).clip(lower=0).to_numpy() * time_step_h
    co2_g = float((offtake_kwh * matched).sum())
    credit_g = float((injection_kwh * matched).sum())
    off = float(offtake_kwh.sum())
    return {
        "co2_kg": co2_g / 1000,
        "grid_offtake_kwh": off,
        "avg_intensity_paid_gCO2_per_kWh": co2_g / off if off else float("nan"),
        "grid_injection_kwh": float(injection_kwh.sum()),
        "injection_credit_kg": credit_g / 1000,
        "co2_net_kg": (co2_g - credit_g) / 1000,
    }


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

    matched = ffill_onto(intensity, offtake_kwh.index, INTENSITY_MAX_AGE, "intensity")

    co2_g = float((offtake_kwh * matched).sum())
    total_kwh = float(offtake_kwh.sum())
    return {
        "co2_kg": co2_g / 1000,
        "co2_g": co2_g,
        "grid_offtake_kwh": total_kwh,
        "avg_intensity_paid_gCO2_per_kWh": co2_g / total_kwh if total_kwh else float("nan"),
        "factor_set": factor_set,
    }

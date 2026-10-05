"""
Hourly carbon intensity of Belgian electricity generation:

    intensity[t] = sum_type  share[type, t] * factor[type]      (gCO2eq/kWh)

with shares from ENTSO-E "Actual Generation per Production Type" and factors
from emission_factors.py (sources listed there). Same method as Tranberg et
al. (2019) and Electricity Maps.

This is an AVERAGE (generation-weighted) intensity, not a marginal one, and
it is production-based (imports ignored). It describes the grid only;
fleet_harness.co2 multiplies it by a run's grid offtake to get kg CO2.
"""
from __future__ import annotations

import pandas as pd

from price_signals.emission_factors import FACTOR_SETS, check_coverage


def compute_intensity(mix: pd.DataFrame, factor_set: str = "direct") -> pd.Series:
    """Carbon intensity (gCO2eq/kWh) per timestep of `mix`.

    mix: columns = ENTSO-E production-type names, values = MW, tz-aware index.
    factor_set: "direct" or "lifecycle".

    Raises ValueError if a column has no factor (it would silently drop out)
    or if total generation is <= 0 at any timestep.
    """
    if factor_set not in FACTOR_SETS:
        raise ValueError(f"factor_set must be one of {list(FACTOR_SETS)}, got {factor_set!r}")
    factors = FACTOR_SETS[factor_set]

    uncovered, _unused = check_coverage(mix.columns, factors)
    if uncovered:
        raise ValueError(
            "generation-mix columns with no emission factor (would silently drop "
            f"out of the metric): {sorted(uncovered)}"
        )

    total = mix.sum(axis=1)
    bad = total[total <= 0]
    if len(bad):
        raise ValueError(
            f"zero or negative total generation at {len(bad)} timestamp(s), "
            f"e.g. {bad.index[0]!r} -- can't compute a share there"
        )

    share = mix.div(total, axis=0)
    factor_vec = pd.Series({c: factors[c] for c in mix.columns})
    intensity = share.mul(factor_vec, axis=1).sum(axis=1)
    intensity.name = f"carbon_intensity_{factor_set}_gCO2_per_kWh"
    return intensity


def generation_weighted_average(intensity: pd.Series, mix: pd.DataFrame) -> float:
    """Generation-weighted annual average (gCO2/kWh). Sanity check against
    Unnewehr et al. (2022): BE ~195-219 in 2018; 2023 should be lower."""
    total = mix.sum(axis=1)
    return float((intensity * total).sum() / total.sum())

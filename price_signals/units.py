"""
Belgian thermal units >= 100 MW for the Phase 3 merit-order dispatch model.

Inputs (hand-built, every cell sourced):
- be_units.csv: one row per ENTSO-E production unit (EIC) that is not nuclear,
  wind or pumped storage. `role` = "merit" (dispatchable, ranked by efficiency)
  or "fixed" (held at its observed output, never marginal). Source ids point
  to be_units_sources.csv, which holds the citation, URL and a verbatim quote.
- be_units_sources.csv.

Emission rate per merit unit, `method` of load_units (decided 9 Nov,
temp_phase3_log.md 5.28-5.30):
- "measured" (MAIN RUN): g/kWh = EU ETS verified emissions of the unit's
  installation [S20] / the unit's ENTSO-E generation [S1], same year. If that
  year's ETS value is blank, the unit's other year (2022 <-> 2023) is used;
  DIW only if both are missing. Column `ef_basis` says which was used.
- "diw" (comparison / sensitivity): from the efficiency below.
- "pglib" (comparison): PGLib-CO2 CCGT class, 362.1 g/kWh for every CCGT.

Derived here (not typed into the CSV, so they cannot drift from their source):
- efficiency eta = DIW Data Documentation 72 (Egerer et al. 2014) Table 21
  [S17], CCGT column, linearly interpolated between its decade rows (the rows
  are themselves linear: +6.67 points per decade); years after 2010, the last
  row, are held at 60.0 % (no extrapolation beyond the source).
- emission factor g/kWh_el = gas CO2 per MWh of fuel / eta, with IPCC 2006
  Vol. 2 Ch. 2 Table 2.2 [S18]: 56 100 kg CO2/TJ (net calorific basis)
  = 0.20196 t CO2/MWh_fuel.

Decisions behind this (PLAN.md Phase 3, temp_phase3_log.md 5.8, 5.21, 5.26, 5.28-5.30).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from price_signals.emission_factors import PGLIB_CO2_FACTORS

HERE = Path(__file__).parent
UNITS_CSV = HERE / "be_units.csv"
SOURCES_CSV = HERE / "be_units_sources.csv"

# [S17] DIW DD72 Table 21, CCGT column, efficiency in % by commissioning year.
DIW_TABLE21_CCGT = {1950: 20.0, 1960: 26.7, 1970: 33.3, 1980: 40.0,
                    1990: 46.7, 2000: 53.3, 2010: 60.0}
# Same table, OCGT column (for open-cycle units, e.g. a sensitivity block).
DIW_TABLE21_OCGT = {1950: 24.7, 1960: 27.3, 1970: 29.9, 1980: 32.5,
                    1990: 35.1, 2000: 37.7, 2010: 40.3}

# [S18] IPCC 2006 natural gas default, kg CO2 per TJ of fuel (net calorific).
GAS_KG_CO2_PER_TJ = 56_100.0
GAS_T_CO2_PER_MWH_FUEL = GAS_KG_CO2_PER_TJ * 3.6e-3 / 1000.0  # 1 MWh = 3.6 GJ = 3.6e-3 TJ

ROLES = {"merit", "fixed"}


def diw_efficiency(year: float, table: dict[int, float] = DIW_TABLE21_CCGT) -> float:
    """Efficiency (0-1) for a commissioning year from DIW DD72 Table 21.

    Linear between the table's decade rows; held at the first/last row
    outside 1950-2010 (no extrapolation beyond the source).
    """
    years = np.array(sorted(table))
    vals = np.array([table[y] for y in years])
    return float(np.interp(year, years, vals)) / 100.0


def emission_factor_g_per_kwh(eta: float) -> float:
    """Direct CO2 per kWh of electricity for a gas unit of efficiency eta (0-1)."""
    if not 0 < eta <= 1:
        raise ValueError(f"efficiency must be in (0, 1], got {eta}")
    return GAS_T_CO2_PER_MWH_FUEL / eta * 1000.0  # t/MWh_el -> g/kWh_el


def load_sources() -> pd.DataFrame:
    return pd.read_csv(SOURCES_CSV, dtype=str).set_index("id")


METHODS = {"measured", "diw", "pglib"}
YEARS = (2022, 2023)


def measured_g_per_kwh(units: pd.DataFrame, year: int) -> pd.Series:
    """ETS t CO2 / ENTSO-E MWh x 1000 for `year`; NaN where the ETS cell is blank."""
    return units[f"ets_t_{year}"] / units[f"gen_mwh_{year}"] * 1000.0


def load_units(year: int = 2023, method: str = "measured",
               eta_shift: float = 0.0) -> pd.DataFrame:
    """The unit list with derived columns.

    year: 2022 or 2023 (the run year: picks the measured rate).
    method: "measured" (main), "diw" or "pglib" (see module docstring).
    eta_shift: added to every merit unit's DIW efficiency (e.g. +-0.02 or
    0.03 for the sensitivity in PLAN.md); used by method="diw" and by the
    DIW fallback of "measured".

    Added columns: eta_diw, eta_diw_low / eta_diw_high (from year_low /
    year_high), eta (= eta_diw + eta_shift), ef_diw_g_per_kwh,
    ef_measured_2022 / ef_measured_2023, ef_g_per_kwh (the rate used),
    ef_basis (where it came from) and generation_columns (list). Fixed units
    get NaN rates: they are never marginal, so no rate is assumed for them.
    Raises if a role is unknown, a merit unit has no commissioning year or
    source, or a cited source id is missing from be_units_sources.csv.
    """
    if year not in YEARS:
        raise ValueError(f"year must be one of {YEARS}, got {year}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}, got {method!r}")
    units = pd.read_csv(UNITS_CSV, dtype={"eic": str})
    sources = load_sources()

    bad_roles = set(units["role"]) - ROLES
    if bad_roles:
        raise ValueError(f"unknown role(s) in {UNITS_CSV.name}: {bad_roles}")
    if units["eic"].duplicated().any():
        raise ValueError(f"duplicate EIC in {UNITS_CSV.name}")

    for col in ("year_sources", "capacity_source", "operator_efficiency_source", "ets_source"):
        for cell in units[col].dropna():
            missing = set(cell.split(";")) - set(sources.index)
            if missing:
                raise ValueError(f"{col} cites unknown source id(s) {missing}")

    merit = units["role"] == "merit"
    for col in ("commissioning_year", "year_low", "year_high", "year_sources"):
        if units.loc[merit, col].isna().any():
            raise ValueError(f"merit unit(s) without {col}: "
                             f"{units.loc[merit & units[col].isna(), 'unit'].tolist()}")

    units["generation_columns"] = units["entsoe_generation_columns"].str.split(";")
    for col, ycol in (("eta_diw", "commissioning_year"),
                      ("eta_diw_low", "year_low"), ("eta_diw_high", "year_high")):
        units[col] = [diw_efficiency(y) if m else np.nan
                      for y, m in zip(units[ycol], merit)]
    units["eta"] = units["eta_diw"] + eta_shift
    units["ef_diw_g_per_kwh"] = [emission_factor_g_per_kwh(e) if m else np.nan
                                 for e, m in zip(units["eta"], merit)]
    for y in YEARS:
        units[f"ef_measured_{y}"] = measured_g_per_kwh(units, y).where(merit)

    if method == "diw":
        units["ef_g_per_kwh"] = units["ef_diw_g_per_kwh"]
        units["ef_basis"] = np.where(merit, "diw", "fixed")
    elif method == "pglib":
        units["ef_g_per_kwh"] = np.where(merit, PGLIB_CO2_FACTORS["CCGT"], np.nan)
        units["ef_basis"] = np.where(merit, "pglib CCGT", "fixed")
    else:
        other = YEARS[1 - YEARS.index(year)]
        same, alt = units[f"ef_measured_{year}"], units[f"ef_measured_{other}"]
        units["ef_g_per_kwh"] = same.fillna(alt).fillna(units["ef_diw_g_per_kwh"])
        units["ef_basis"] = np.select(
            [~merit, same.notna(), alt.notna()],
            ["fixed", f"measured {year}", f"measured {other} (ETS {year} blank)"],
            default="diw (no ETS value)")
    return units

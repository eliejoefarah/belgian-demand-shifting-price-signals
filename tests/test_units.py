from pathlib import Path

import pandas as pd
import pytest

from price_signals.units import (
    GAS_T_CO2_PER_MWH_FUEL,
    diw_efficiency,
    emission_factor_g_per_kwh,
    load_sources,
    load_units,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def test_diw_table21_rows_and_interpolation():
    assert diw_efficiency(1990) == pytest.approx(0.467)
    assert diw_efficiency(2000) == pytest.approx(0.533)
    assert diw_efficiency(2010) == pytest.approx(0.600)
    assert diw_efficiency(1994) == pytest.approx(0.467 + 0.4 * 0.066)
    assert diw_efficiency(2012) == pytest.approx(0.600)  # held, not extrapolated


def test_ipcc_gas_factor():
    assert GAS_T_CO2_PER_MWH_FUEL == pytest.approx(0.20196)
    assert emission_factor_g_per_kwh(0.5) == pytest.approx(403.92)
    with pytest.raises(ValueError):
        emission_factor_g_per_kwh(0.0)


def test_unit_list_consistent():
    units = load_units(method="diw")
    assert len(units) == 14
    merit = units[units.role == "merit"]
    assert set(units[units.role == "fixed"].unit) == {
        "INESCO WKK", "Scheldelaan Exxonmobil", "Zelzate 2 Knippegroen", "RODENHUIZE 4"}
    assert len(merit) == 10
    assert (merit.year_low <= merit.commissioning_year).all()
    assert (merit.commissioning_year <= merit.year_high).all()
    # DIW range of Belgian CCGTs: ~337-409 g/kWh (temp_phase3_log.md 5.26)
    assert merit.ef_g_per_kwh.between(330, 415).all()
    assert units[units.role == "fixed"].eta.isna().all()


def test_eta_shift():
    base, up = load_units(method="diw"), load_units(method="diw", eta_shift=0.02)
    m = base.role == "merit"
    assert (up.eta[m] - base.eta[m]).round(9).eq(0.02).all()
    assert (up.ef_g_per_kwh[m] < base.ef_g_per_kwh[m]).all()


def test_every_source_has_url_and_quote():
    src = load_sources()
    assert src[["citation", "url", "accessed", "verbatim_quote"]].notna().all().all()


@pytest.mark.parametrize("year", [2022, 2023])
def test_matches_entsoe_files(year):
    cap_path = DATA / f"entsoe_be_{year}_capacity_per_unit.csv"
    gen_path = DATA / f"entsoe_be_{year}_generation_per_plant.csv"
    if not (cap_path.exists() and gen_path.exists()):
        pytest.skip("ENTSO-E per-unit files not downloaded")
    units = load_units()
    cap = pd.read_csv(cap_path)
    thermal = cap[~cap.production_type.isin(
        ["Nuclear", "Wind Offshore", "Wind Onshore", "Hydro Pumped Storage"])]
    # every non-nuclear/wind/storage ENTSO-E unit is in the list, and vice versa
    assert set(thermal.eic) == set(units.eic)
    merged = units.merge(thermal, on="eic")
    assert (merged[f"capacity_mw_{year}"] == merged["capacity_mw"]).all()
    gen_cols = set(pd.read_csv(gen_path, header=[0, 1, 2], nrows=0).columns.get_level_values(0))
    for cols in units.generation_columns:
        assert set(cols) <= gen_cols


def test_measured_rates_and_fallback():
    u23 = load_units(2023).set_index("unit")
    u22 = load_units(2022).set_index("unit")
    # T-Power 2022: 636 648 t / 1 791 253.6 MWh = 355.4 g/kWh
    assert u22.loc["T-power Beringen", "ef_g_per_kwh"] == pytest.approx(355.42, abs=0.01)
    assert u22.loc["T-power Beringen", "ef_basis"] == "measured 2022"
    # Marcinelle: ETS 2023 blank -> its 2022 rate, not DIW
    m = u23.loc["Marcinelle Energie (Carsid)"]
    assert m.ef_basis == "measured 2022 (ETS 2023 blank)"
    assert m.ef_g_per_kwh == pytest.approx(u22.loc["Marcinelle Energie (Carsid)", "ef_g_per_kwh"])
    merit = u23[u23.role == "merit"]
    assert merit.ef_g_per_kwh.between(330, 450).all()  # plausibility band (temp_phase3_log 5.27)
    assert not merit.ef_basis.str.startswith("diw").any()
    assert u23[u23.role == "fixed"].ef_g_per_kwh.isna().all()


def test_methods_and_bad_args():
    assert (load_units(method="pglib").query("role == 'merit'").ef_g_per_kwh == 362.1).all()
    with pytest.raises(ValueError):
        load_units(2021)
    with pytest.raises(ValueError):
        load_units(method="average")


@pytest.mark.parametrize("year", [2022, 2023])
def test_stored_generation_matches_entsoe(year):
    gen_path = DATA / f"entsoe_be_{year}_generation_per_plant.csv"
    if not gen_path.exists():
        pytest.skip("ENTSO-E per-unit files not downloaded")
    g = pd.read_csv(gen_path, header=[0, 1, 2], index_col=0)
    g.columns = g.columns.get_level_values(0)
    for _, u in load_units(year).iterrows():
        assert g[u.generation_columns].sum().sum() == pytest.approx(u[f"gen_mwh_{year}"], abs=0.01)

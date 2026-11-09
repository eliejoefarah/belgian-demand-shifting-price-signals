"""emission_factors.check_coverage and compute_intensity's guards."""
import pandas as pd
import pytest

from price_signals.carbon_intensity import compute_intensity, gas_share
from price_signals.emission_factors import (
    DIRECT_FACTORS,
    LIFECYCLE_FACTORS,
    PGLIB_CO2_FACTORS,
    check_coverage,
)


def test_check_coverage_reports_uncovered_and_unused():
    factors = {"Fossil Gas": 370.0, "Nuclear": 0.0, "Solar": 0.0}
    uncovered, unused = check_coverage(["Fossil Gas", "Nuclear", "Mystery"], factors)
    assert uncovered == {"Mystery"}
    assert unused == {"Solar"}


def test_check_coverage_full_match():
    assert check_coverage(DIRECT_FACTORS.keys(), DIRECT_FACTORS) == (set(), set())


def test_factor_sets_consistent():
    assert set(DIRECT_FACTORS) == set(LIFECYCLE_FACTORS)
    assert all(LIFECYCLE_FACTORS[k] >= DIRECT_FACTORS[k] for k in DIRECT_FACTORS)


def _mix(rows, columns=("Fossil Gas", "Nuclear")):
    idx = pd.date_range("2023-01-01", periods=len(rows), freq="h", tz="Europe/Brussels")
    return pd.DataFrame(rows, index=idx, columns=list(columns))


def test_compute_intensity_is_share_weighted():
    # 25 % gas, 75 % nuclear -> 0.25 * 370 = 92.5 direct.
    intensity = compute_intensity(_mix([[100.0, 300.0]]), factor_set="direct")
    assert intensity.iloc[0] == pytest.approx(0.25 * DIRECT_FACTORS["Fossil Gas"])
    lifecycle = compute_intensity(_mix([[100.0, 300.0]]), factor_set="lifecycle")
    assert lifecycle.iloc[0] == pytest.approx(
        0.25 * LIFECYCLE_FACTORS["Fossil Gas"] + 0.75 * LIFECYCLE_FACTORS["Nuclear"]
    )


def test_compute_intensity_rejects_uncovered_column():
    with pytest.raises(ValueError, match="no emission factor"):
        compute_intensity(_mix([[100.0, 300.0]], columns=("Fossil Gas", "Mystery")))


def test_compute_intensity_rejects_zero_generation():
    with pytest.raises(ValueError, match="zero or negative total generation"):
        compute_intensity(_mix([[100.0, 300.0], [0.0, 0.0]]))


def test_compute_intensity_rejects_unknown_factor_set():
    with pytest.raises(ValueError, match="factor_set"):
        compute_intensity(_mix([[100.0, 300.0]]), factor_set="marginal")


def test_gas_share_is_gas_over_total():
    # 25 % gas, then 0 % gas.
    share = gas_share(_mix([[100.0, 300.0], [0.0, 400.0]]))
    assert share.tolist() == pytest.approx([0.25, 0.0])
    assert share.name == "gas_share"


def test_gas_share_treats_empty_cells_as_zero():
    # An empty gas cell is no gas; an empty other cell drops out of the total.
    nan = float("nan")
    share = gas_share(_mix([[nan, 300.0], [100.0, nan]]))
    assert share.tolist() == pytest.approx([0.0, 1.0])


def test_gas_share_rejects_missing_gas_column():
    with pytest.raises(ValueError, match="Fossil Gas"):
        gas_share(_mix([[100.0, 300.0]], columns=("Nuclear", "Solar")))


def test_gas_share_rejects_zero_generation():
    with pytest.raises(ValueError, match="zero or negative total generation"):
        gas_share(_mix([[100.0, 300.0], [0.0, 0.0]]))


def test_pglib_co2_gas_classes_bracket_the_belgian_ets_value():
    # Values copied from PGLib-CO2 (arXiv:2506.14662), checked 8 Nov 2026.
    assert PGLIB_CO2_FACTORS["CCGT"] == pytest.approx(362.1)
    assert PGLIB_CO2_FACTORS["NG"] == pytest.approx(517.3)
    # Belgian gas fleet average (EU-ETS fit, 382.65) must fall between the two classes.
    assert PGLIB_CO2_FACTORS["CCGT"] < 382.65 < PGLIB_CO2_FACTORS["NG"]

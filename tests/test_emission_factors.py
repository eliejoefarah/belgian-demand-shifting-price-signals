"""emission_factors.check_coverage and compute_intensity's guards."""
import pandas as pd
import pytest

from price_signals.carbon_intensity import compute_intensity
from price_signals.emission_factors import (
    DIRECT_FACTORS,
    LIFECYCLE_FACTORS,
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

import numpy as np
import pandas as pd
import pytest

from price_signals.mef_validation import _shift_lists, deetjen_validation, rolling_curves


def test_shift_lists_match_their_construction():
    x = np.arange(10.0)
    low, high = _shift_lists(x, p=4)            # p/2 = 2 zeros at each end
    assert low.tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 0, 0]
    assert high.tolist() == [0, 0, 4, 5, 6, 7, 8, 9, 0, 0]


def test_linear_emissions_give_constant_slope():
    rng = np.random.default_rng(0)
    d = rng.uniform(1000, 5000, 8760)
    df = rolling_curves(d, 0.4 * d + 50.0)
    s = df["co2_slope"].dropna()
    # their method compares SMOOTHED emissions with RAW demand at i +- p/2, so even an
    # exactly linear system gives slopes scattered by a few % (documented in the module)
    assert len(s) > 6000
    assert s.mean() == pytest.approx(0.4, rel=0.01)
    assert np.allclose(s, 0.4, rtol=0.06)


def test_identical_series_zero_error_and_regression_recovers_slope():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2023-01-01", periods=8760, freq="1h", tz="UTC")
    d = pd.Series(3000 + 1500 * np.sin(np.arange(8760) / 24 * 2 * np.pi) + rng.normal(0, 200, 8760), index=idx)
    e = 0.38 * d
    v = deetjen_validation(d, e, e)
    assert v["errors"]["co2_slope_sim"] == pytest.approx(0.0, abs=1e-12)
    assert v["errors"]["co2_slope_regression"] < 0.03   # only the method's own smoothing noise
    assert np.allclose(v["regression"]["mef_t_per_mwh"], 0.38)

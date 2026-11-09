from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from price_signals.dispatch_model import (
    lost_capacity,
    merit_order_dispatch,
    run_year,
    to_quarter_hours,
)

DATA = Path(__file__).resolve().parents[1] / "data"
RATE = np.array([400.0, 350.0, 380.0])          # unit 1 cleanest, then 2, then 0
AVAIL = np.array([[100.0, 100.0, 100.0]])


def stack(demand, avail=AVAIL, rate=RATE):
    return merit_order_dispatch(np.atleast_1d(demand), avail, rate)


def test_cheapest_first_and_marginal():
    r = stack(150.0)
    assert r["dispatch"][0].tolist() == [0.0, 100.0, 50.0]
    assert r["marginal"][0] == 2 and r["headroom"][0] == 50.0
    assert r["next_unit"][0] == 0 and not r["shortage"][0]


def test_exact_boundary_next_mw_goes_to_next_unit():
    r = stack(100.0)                      # unit 1 exactly full
    assert r["marginal"][0] == 2 and r["headroom"][0] == 100.0


def test_zero_or_negative_demand_cheapest_is_marginal():
    for d in (0.0, -50.0):
        r = stack(d)
        assert r["marginal"][0] == 1 and r["dispatch"].sum() == 0.0


def test_shortage():
    r = stack(400.0)
    assert r["shortage"][0] and r["marginal"][0] == -1
    assert r["dispatch"].sum() == 300.0


def test_outage_skips_unit():
    r = stack(50.0, avail=np.array([[100.0, 0.0, 100.0]]))   # cleanest unit out
    assert r["marginal"][0] == 2 and r["dispatch"][0, 1] == 0.0
    assert r["next_unit"][0] == 0


def test_lost_capacity_partial_hour_and_overlap():
    idx = pd.date_range("2023-01-01", periods=3, freq="1h", tz="UTC")
    o = pd.DataFrame({
        "production_resource_id": ["A", "A"],
        "start": pd.to_datetime(["2023-01-01 00:30", "2023-01-01 00:00"], utc=True),
        "end": pd.to_datetime(["2023-01-01 02:00", "2023-01-01 01:00"], utc=True),
        "lost_mw": [100.0, 40.0],
    })
    lost = lost_capacity(o, ["A"], idx)["A"].tolist()
    assert lost == pytest.approx([50.0 + 40.0, 100.0, 0.0])


def test_to_quarter_hours_repeats():
    s = pd.Series([1.0, 2.0], index=pd.date_range("2023-01-01", periods=2, freq="1h", tz="UTC"))
    q = to_quarter_hours(s)
    assert len(q) == 8 and q.tolist() == [1.0] * 4 + [2.0] * 4


@pytest.mark.parametrize("year", [2022, 2023])
def test_real_year(year):
    if not (DATA / f"entsoe_be_{year}_unavailability.csv").exists():
        pytest.skip("ENTSO-E files not downloaded")
    res = run_year(year)
    h = res.hourly
    assert len(h) == 8760
    # the model produces exactly the observed merit-fleet energy each hour (no shortage)
    ok = ~h["shortage"]
    assert np.allclose(res.dispatch[ok].sum(axis=1), h.loc[ok, "demand_mw"])
    assert (res.dispatch <= res.available + 1e-6).all().all()
    m = h["mef_g_per_kwh"].dropna()
    assert m.between(330, 450).all()

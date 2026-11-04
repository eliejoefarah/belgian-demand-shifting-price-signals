"""fleet_harness.mpc_runner: control-signal separation and real-price billing.

Fast tests (synthetic, always run) check the billing formula.
Slow tests (marked `slow`, skipped by default) simulate house 0 for 3 days with the MPC.
They need the paper's Zenodo data, EnergyPlus and a Gurobi licence, and take ~1-2 min:

    python -m pytest tests -m slow
"""
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import fleet_harness.mpc_runner as mpc_runner
from fleet_harness.mpc_runner import (
    PAPER_ROOT,
    add_control_signal,
    calibration_year,
    get_calibration,
    house_config_path,
    load_house_price_and_tariffs,
    parsed_house,
    real_price_day_ahead_cost,
    run_isolated,
    shifted_2023_config,
)

DAYS = 3
HOUSE = 0


# ---------------------------------------------------------------------------
# Fast: billing formula on synthetic data
# ---------------------------------------------------------------------------

def grid(kw_values, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(kw_values), freq="15min", tz="UTC")
    return pd.Series(kw_values, index=idx)


def test_day_ahead_cost_offtake_injection_and_vat():
    g = grid([4.0, -2.0, 4.0])                      # kW -> 1, -0.5, 1 kWh per step
    price = pd.Series([0.10, 0.10, -0.05], index=g.index.tz_localize(None))
    cost = real_price_day_ahead_cost(g, 0.25, price, offtake_extra=0.02, injection_extra=-0.01)
    expected = -(
        1.0 * 0.12 * 1.06            # offtake, positive price -> VAT
        + -0.5 * (0.10 - 0.01)       # injection, no VAT
        + 1.0 * (-0.05 + 0.02)       # offtake, negative price -> no VAT
    )
    assert cost == pytest.approx(expected)


def test_day_ahead_cost_naive_price_is_utc_and_ffilled():
    g = grid([4.0] * 4)
    hourly_naive = pd.Series([0.2], index=pd.DatetimeIndex(["2023-01-01 00:00"]))
    assert real_price_day_ahead_cost(g, 0.25, hourly_naive, 0.0, 0.0) == pytest.approx(-4 * 0.2 * 1.06)


def test_day_ahead_cost_rejects_uncovered_timestamps():
    g = grid([1.0, 1.0])
    late_price = pd.Series([0.1], index=pd.DatetimeIndex(["2023-01-02 00:00"]))
    with pytest.raises(ValueError, match="cover"):
        real_price_day_ahead_cost(g, 0.25, late_price, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Slow: 3-day MPC runs of house 0
# ---------------------------------------------------------------------------

HOUSE_CONFIG = PAPER_ROOT / "data" / "houses_belgium_500" / f"house_{HOUSE}.json"
needs_paper_data = pytest.mark.skipif(
    not HOUSE_CONFIG.exists(), reason=f"paper data not found: {HOUSE_CONFIG}"
)


# ---------------------------------------------------------------------------
# Fast: atomic 2023 config (needs paper data, no simulation)
# ---------------------------------------------------------------------------

@needs_paper_data
def test_shifted_config_matches_paper_tmp_2023_config(tmp_path):
    from belgian_dwellings.simulation.tmp_2023_config import tmp_2023_config

    src_dir = tmp_path / "houses"
    src_dir.mkdir()
    src = src_dir / HOUSE_CONFIG.name
    shutil.copy(HOUSE_CONFIG, src)

    ours = json.loads(open(shifted_2023_config(src)).read())
    paper_path = tmp_2023_config(str(src))
    assert paper_path == str(tmp_path / "houses_tmp_2023" / HOUSE_CONFIG.name)
    assert ours == json.loads(open(paper_path).read())


# ---------------------------------------------------------------------------
# Fast: run year -> config and calibration year
# ---------------------------------------------------------------------------

def test_calibration_year_rule():
    assert calibration_year("realistic", 2023) == 2022  # Julian's rule
    assert calibration_year("perfect", 2023) == 2023  # Julian's rule
    assert calibration_year("realistic", 2022) == 2023  # never in-sample EV history
    assert calibration_year("perfect", 2022) == 2022
    with pytest.raises(ValueError, match="year"):
        calibration_year("realistic", "2022")
    with pytest.raises(ValueError, match="year"):
        calibration_year("realistic", 2021)
    with pytest.raises(ValueError, match="mode"):
        calibration_year("rbc", 2023)


@needs_paper_data
def test_house_config_path_per_year():
    cfg_2022 = json.loads(open(house_config_path(HOUSE, 2022)).read())["Microgrid"]
    cfg_2023 = json.loads(open(house_config_path(HOUSE, 2023)).read())["Microgrid"]
    assert cfg_2022["start_time"].startswith("2022/01/01")
    assert cfg_2023["start_time"].startswith("2023/01/01")


@needs_paper_data
@pytest.mark.parametrize(
    "mode, year, cal_year",
    [("realistic", 2023, 2022), ("realistic", 2022, 2023), ("perfect", 2022, 2022)],
)
def test_get_calibration_source_and_cache_name(tmp_path, monkeypatch, mode, year, cal_year):
    """get_calibration fits on the calibration year's config and caches under that year."""
    seen = []
    monkeypatch.setattr(mpc_runner, "CAL_DIR", tmp_path)
    monkeypatch.setattr(
        mpc_runner, "get_energyplus_calibration", lambda cfg: seen.append(cfg) or (["thermal"], ["ev"])
    )
    assert get_calibration(HOUSE, mode, year=year) == (["thermal"], ["ev"])
    assert seen == [house_config_path(HOUSE, cal_year)]
    assert [f.name for f in tmp_path.iterdir()] == [f"house_{HOUSE}_500_{mode}_cal{cal_year}.pkl"]
    get_calibration(HOUSE, mode, year=year)  # second call reads the cache
    assert len(seen) == 1


def _write_then_read(src):
    for _ in range(20):
        with open(shifted_2023_config(src)) as f:
            json.load(f)  # raises JSONDecodeError if a reader ever sees a partial file
    return True


@needs_paper_data
def test_shifted_config_safe_under_parallel_writes(tmp_path):
    src_dir = tmp_path / "houses"
    src_dir.mkdir()
    src = src_dir / HOUSE_CONFIG.name
    shutil.copy(HOUSE_CONFIG, src)
    spawn = multiprocessing.get_context("spawn")  # fork from a threaded pytest process can deadlock
    with ProcessPoolExecutor(max_workers=8, mp_context=spawn) as pool:
        assert all(pool.map(_write_then_read, [str(src)] * 16))
    assert not list((tmp_path / "houses_tmp_2023").glob(".partial_*"))


@pytest.fixture(scope="module")
def price_and_tariffs():
    return load_house_price_and_tariffs(HOUSE)


@pytest.fixture(scope="module")
def flat(price_and_tariffs):
    price, _ = price_and_tariffs
    return pd.Series(price.mean(), index=price.index)


@pytest.fixture(scope="module")
def run_julian():
    return run_isolated(HOUSE, days=DAYS, use_signal=False)


@pytest.fixture(scope="module")
def run_real_signal():
    return run_isolated(HOUSE, days=DAYS)


@pytest.fixture(scope="module")
def run_flat_signal(flat):
    return run_isolated(HOUSE, days=DAYS, signal_series=flat)


@pytest.mark.slow
@needs_paper_data
def test_control_signal_forecast_matches_series(flat):
    """6c: get_forecast on control_signal returns the injected values over a horizon."""
    with parsed_house(HOUSE) as mg:
        add_control_signal(mg, flat)
        horizon = 96
        values = mg.environments[0].env_values["control_signal"].get_forecast(
            mg.utc_datetime, mg.utc_datetime + horizon * mg.time_step
        )["values"]
    assert len(values) == horizon
    assert np.allclose(values, flat.iloc[0])


@pytest.mark.slow
@needs_paper_data
def test_add_control_signal_rejects_misaligned_index(flat):
    shifted = flat.copy()
    shifted.index = shifted.index + timedelta(hours=1)
    with parsed_house(HOUSE) as mg:
        with pytest.raises(ValueError, match="same timestamps"):
            add_control_signal(mg, shifted)
        brussels = flat.tz_localize("UTC").tz_convert("Europe/Brussels")
        with pytest.raises(ValueError, match="same timestamps"):
            add_control_signal(mg, brussels)


@pytest.mark.slow
@needs_paper_data
def test_signal_path_identical_to_julian(run_julian, run_real_signal):
    """6a: control_signal = real price reproduces Julian's MPC exactly."""
    assert run_real_signal["kpis"] == run_julian["kpis"]
    pd.testing.assert_series_equal(
        run_real_signal["power"]["PublicGrid_0"], run_julian["power"]["PublicGrid_0"]
    )


@pytest.mark.slow
@needs_paper_data
def test_mpc_follows_signal(run_real_signal, run_flat_signal):
    """6b part 1: a different signal changes the MPC's decisions."""
    real = run_real_signal["power"]["PublicGrid_0"].to_numpy()
    flat = run_flat_signal["power"]["PublicGrid_0"].to_numpy()
    assert np.abs(real - flat).max() > 0.1  # kW


@pytest.mark.slow
@needs_paper_data
def test_opex_billed_at_real_price(price_and_tariffs, flat, run_flat_signal):
    """6b part 2: the MPC planned on the flat signal, but DayAheadEngie billed the real price."""
    price, extras = price_and_tariffs
    r = run_flat_signal
    g, dt = r["power"]["PublicGrid_0"], r["time_step_h"]
    at_real = real_price_day_ahead_cost(g, dt, price, **extras)
    at_signal = real_price_day_ahead_cost(g, dt, flat, **extras)
    assert r["kpis"]["day_ahead"] == pytest.approx(at_real, abs=1e-6)
    assert abs(at_signal - at_real) > 1e-3  # the check can tell the two apart


@pytest.mark.slow
@needs_paper_data
def test_run_2022_starts_in_2022_and_bills_2022_price():
    """year=2022: simulates 2022 (original config) and DayAheadEngie bills the 2022 price."""
    r = run_isolated(HOUSE, days=1, year=2022)
    grid = r["power"]["PublicGrid_0"]
    assert grid.index[0] == pd.Timestamp("2022-01-01", tz="UTC")
    assert grid.index[-1] < pd.Timestamp("2022-01-02", tz="UTC")
    price, extras = load_house_price_and_tariffs(HOUSE, year=2022)
    assert r["kpis"]["day_ahead"] == pytest.approx(
        real_price_day_ahead_cost(grid, r["time_step_h"], price, **extras), abs=1e-6
    )

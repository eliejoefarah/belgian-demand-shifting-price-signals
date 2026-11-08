"""fleet_harness.fleet, the fleet scripts and co2_from_grid_kw. All fast: synthetic runs or
the JSON configs only (no EnergyPlus, no Gurobi solve)."""
import json

import numpy as np
import pandas as pd
import pytest

import fleet_harness.mpc_runner as mpc_runner
from experiments import fleet_baseline
from fleet_harness import fleet
from fleet_harness.co2 import co2_from_grid_kw
from fleet_harness.runner import PAPER_ROOT

HOUSE_CONFIG = PAPER_ROOT / "data" / "houses_belgium_500" / "house_0.json"
needs_paper_data = pytest.mark.skipif(not HOUSE_CONFIG.exists(), reason="paper data not found")


def synthetic_result(pv=True, ev=True, n=4):
    """A fake extract_results() dict that obeys simugrid's signs (columns sum to 0)."""
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    power = pd.DataFrame(
        {
            "Consumer_0": [-0.2, -0.4, -0.2, -0.2][:n],
            "EnergyPlus_0": [-1.0, 0.0, -2.0, 0.0][:n],
            "WaterHeater_0": [0.0, -2.0, 0.0, 0.0][:n],
        },
        index=idx,
    )
    if ev:
        power["Charger_0"] = [0.0, 0.0, -3.7, 0.0][:n]
    if pv:
        power["SolarPv_0"] = [0.0, 1.0, 1.0, 4.0][:n]
    power["PublicGrid_0"] = -power.sum(axis=1)
    solves = pd.DataFrame(
        {"status": [2, 2, 9, 2][:n], "runtime_s": [0.01, 0.02, 10.0, 0.01][:n]},
        index=pd.Index(idx, name="datetime"),
    )
    attributes = pd.DataFrame({"EnergyPlus_0.zn0_temp": [20.0] * n}, index=idx + idx.freq)
    kpis = {"opex": -1.0, "discomfort": 0.0}
    signal = pd.DataFrame(
        {mpc_runner.PRICE_COL: [0.10, 0.20, 0.30, 0.40][:n], mpc_runner.SIGNAL_COL: [0.25] * n},
        index=pd.Index(idx, name="datetime"),
    )
    return {"kpis": kpis, "power": power, "attributes": attributes, "solves": solves,
            "signal": signal, "time_step_h": 0.25}


# ---------------------------------------------------------------------------
# power table, totals, solve stats
# ---------------------------------------------------------------------------

def test_power_table_derived_columns_and_balance():
    t = fleet.power_table(synthetic_result()["power"])
    assert t["offtake_kw"].tolist() == [1.2, 1.4, 4.9, 0.0]
    assert t["injection_kw"].tolist() == [0.0, 0.0, 0.0, 3.8]
    assert t["appliance_kw"].tolist() == [0.2, 0.4, 0.2, 0.2]
    assert t["ev_kw"].tolist() == [0.0, 0.0, 3.7, 0.0]
    assert (t[["offtake_kw", "injection_kw", "consumption_kw", "flexible_kw"]] >= 0).all().all()
    balance = t["offtake_kw"] - t["injection_kw"] - t["consumption_kw"] + t["pv_kw"]
    assert np.allclose(balance, 0)


def test_power_table_missing_assets_are_zero():
    t = fleet.power_table(synthetic_result(pv=False, ev=False)["power"])
    assert (t["ev_kw"] == 0).all() and (t["pv_kw"] == 0).all()
    assert np.allclose(t["consumption_kw"], t["offtake_kw"])


def test_energy_totals_and_solve_stats():
    r = synthetic_result()
    e = fleet.energy_totals(fleet.power_table(r["power"]), 0.25)
    assert e["offtake_kwh"] == pytest.approx((1.2 + 1.4 + 4.9) * 0.25)
    assert e["ev_kwh"] == pytest.approx(3.7 * 0.25)
    s = fleet.solve_stats(r["solves"])
    assert (s["n_solves"], s["n_optimal"], s["n_time_limit"], s["n_other"]) == (4, 3, 1, 0)


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_and_done_marker(tmp_path):
    r = synthetic_result()
    fleet.house_paths(tmp_path, 7)["error"].write_text("old failure")
    assert not fleet.is_done(tmp_path, 7)
    fleet.save_house_result(tmp_path, 7, r, {"year": 2023, "mode": "realistic"})
    assert fleet.is_done(tmp_path, 7)
    assert not fleet.house_paths(tmp_path, 7)["error"].exists()
    assert not list(tmp_path.glob(".*partial"))

    table, meta = fleet.load_house(tmp_path, 7)
    pd.testing.assert_series_equal(
        table["PublicGrid_0"], r["power"]["PublicGrid_0"], check_freq=False, check_names=False
    )
    assert table.index.name == "datetime"
    assert str(table.index.tz) == "UTC"
    assert meta["n_steps"] == 4 and meta["solves"]["n_time_limit"] == 1
    assert meta["start"].startswith("2023-01-01 00:00")

    signal = fleet.load_signal(fleet.house_paths(tmp_path, 7)["signal"])
    pd.testing.assert_frame_equal(signal, r["signal"], check_freq=False)

    summary = fleet.load_run_summary(tmp_path)
    assert summary.index.tolist() == [7]
    assert summary.loc[7, "kpis.opex"] == -1.0
    assert summary.loc[7, "energy_kwh.ev_kwh"] == pytest.approx(0.925)


def test_run_and_save_success_and_failure(tmp_path, monkeypatch):
    calls = []

    def fake_run_isolated(house, **kw):
        calls.append((house, kw))
        if house == 2:
            raise RuntimeError("EnergyPlus died")
        return synthetic_result()

    monkeypatch.setattr(mpc_runner, "run_isolated", fake_run_isolated)
    settings = {"year": 2022, "mode": "realistic", "days": 7, "gurobi_threads": 1,
                "signal_name": "real_price"}
    ok = fleet.run_and_save(1, tmp_path, settings, {"git_commit": "abc"})
    bad = fleet.run_and_save(2, tmp_path, settings, {"git_commit": "abc"})

    assert ok["ok"] and not bad["ok"]
    assert calls[0] == (1, {"year": 2022, "mode": "realistic", "days": 7, "gurobi_threads": 1})
    meta = json.loads(fleet.house_paths(tmp_path, 1)["meta"].read_text())
    assert meta["signal"] == "real_price" and meta["calibration_year"] == 2023
    assert meta["git_commit"] == "abc"
    assert "EnergyPlus died" in fleet.house_paths(tmp_path, 2)["error"].read_text()
    assert not fleet.is_done(tmp_path, 2)


def fake_microgrid_with_signal(signal_values=None):
    """Only what signal_table reads: env values with dt_and_values, start and current time."""
    import datetime as dt
    from types import SimpleNamespace

    # the env values hold more than the simulated window (here: 1 step before, 2 after)
    dts = [dt.datetime(2022, 12, 31, 23, 45) + i * dt.timedelta(minutes=15) for i in range(7)]
    env = {"day_ahead_price": SimpleNamespace(dt_and_values=list(zip(dts, range(7))))}
    if signal_values is not None:
        env["control_signal"] = SimpleNamespace(dt_and_values=list(zip(dts, signal_values)))
    utc = dt.timezone.utc
    return SimpleNamespace(
        environments=[SimpleNamespace(env_values=env)],
        start_time=dt.datetime(2023, 1, 1, 1, 0, tzinfo=dt.timezone(dt.timedelta(hours=1))),
        utc_datetime=dt.datetime(2023, 1, 1, 1, 0, tzinfo=utc),
    )


def test_signal_table_window_and_columns():
    t = mpc_runner.signal_table(fake_microgrid_with_signal([5.0] * 7))
    # start 01:00 Brussels = 00:00 UTC; end (current time) 01:00 UTC, exclusive
    assert t.index.tolist() == list(pd.date_range("2023-01-01", periods=4, freq="15min", tz="UTC"))
    assert t.index.name == "datetime"
    assert t[mpc_runner.PRICE_COL].tolist() == [1, 2, 3, 4]
    assert (t[mpc_runner.SIGNAL_COL] == 5.0).all()


def test_signal_table_without_injected_signal_is_the_price():
    t = mpc_runner.signal_table(fake_microgrid_with_signal())
    assert t[mpc_runner.SIGNAL_COL].tolist() == t[mpc_runner.PRICE_COL].tolist()


# ---------------------------------------------------------------------------
# strata and groups
# ---------------------------------------------------------------------------

def toy_inventory():
    rows = ([(True, True, False)] * 40 + [(True, False, False)] * 20 + [(True, True, True)] * 15
            + [(False, True, False)] * 4 + [(False, False, True)] * 1)
    inv = pd.DataFrame(rows, columns=fleet.STRATUM_KEYS)
    inv.index.name = "house"
    inv["stratum"] = fleet.stratum_labels(inv)
    return inv


def test_allocate_proportional_with_every_stratum():
    counts = toy_inventory()["stratum"].value_counts()
    alloc = fleet._allocate(counts, 10)
    assert alloc.sum() == 10
    assert (alloc >= 1).all()  # 5 strata, 10 houses: everyone gets one
    assert (alloc <= counts[alloc.index]).all()
    assert alloc["heat+ev+nopv"] == alloc.max()
    with pytest.raises(ValueError):
        fleet._allocate(counts, 81)


def test_select_groups_disjoint_stratified_reproducible():
    inv = toy_inventory()
    g = fleet.select_groups(inv, n_train=10, n_val=10, seed=3)
    assert (g == "train").sum() == 10 and (g == "val").sum() == 10
    assert set(inv.loc[g == "train", "stratum"]) == set(inv["stratum"])
    pd.testing.assert_series_equal(g, fleet.select_groups(inv, 10, 10, seed=3))
    assert not g.equals(fleet.select_groups(inv, 10, 10, seed=4))


@needs_paper_data
def test_house_inventory_matches_known_counts():
    inv = fleet.house_inventory()
    assert len(inv) == 500
    assert int(inv.has_heating.sum()) == 447 and int(inv.has_dhw.sum()) == 447
    assert int(inv.has_ev.sum()) == 344 and int(inv.has_pv.sum()) == 129
    assert (inv.has_heating == inv.has_dhw).all()
    assert inv.loc[0, "stratum"] == "heat+ev+nopv"


# ---------------------------------------------------------------------------
# fleet_baseline CLI helpers
# ---------------------------------------------------------------------------

def test_parse_houses_and_heavy_first():
    inv = toy_inventory()
    assert fleet_baseline.parse_houses("0-2,5", inv) == [0, 1, 2, 5]
    assert len(fleet_baseline.parse_houses("all", inv)) == len(inv)
    smoke = fleet_baseline.parse_houses("smoke", inv)
    assert set(inv.loc[smoke, "stratum"]) == set(inv["stratum"])
    with pytest.raises(SystemExit):
        fleet_baseline.parse_houses("0,999", inv)
    order = fleet_baseline.heavy_first([79, 60, 0], inv)  # noheat+nopv, heat+noev, heat+ev
    assert order == [0, 60, 79]


@needs_paper_data
def test_shards_cover_all_houses_once(tmp_path, capsys):
    seen = []
    for k in range(3):
        fleet_baseline.main(["--year", "2023", "--houses", "0-9", "--shard", str(k),
                             "--n-shards", "3", "--dry-run", "--out-root", str(tmp_path)])
        seen += json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert sorted(seen) == list(range(10))


# ---------------------------------------------------------------------------
# co2_from_grid_kw
# ---------------------------------------------------------------------------

def test_co2_from_grid_kw_credited_and_uncredited():
    idx = pd.date_range("2023-06-01 00:00", periods=4, freq="15min", tz="UTC")
    grid = pd.Series([4.0, -4.0, 8.0, 0.0], index=idx)  # 1 kWh off, 1 kWh inj, 2 kWh off
    hours = pd.date_range("2023-06-01 02:00", periods=1, freq="h", tz="Europe/Brussels")
    intensity = pd.Series([200.0], index=hours)  # 02:00 Brussels = 00:00 UTC
    r = co2_from_grid_kw(grid, 0.25, intensity)
    assert r["grid_offtake_kwh"] == 3.0 and r["grid_injection_kwh"] == 1.0
    assert r["co2_kg"] == pytest.approx(0.6)
    assert r["injection_credit_kg"] == pytest.approx(0.2)
    assert r["co2_net_kg"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="outside"):
        co2_from_grid_kw(grid, 0.25, intensity.shift(1, freq="h"))


def test_co2_from_grid_kw_rejects_mix_that_ends_too_early():
    # A mix for the wrong year used to be forward-filled silently onto every later step.
    idx = pd.date_range("2023-06-01 00:00", periods=4, freq="15min", tz="UTC")
    grid = pd.Series([4.0] * 4, index=idx)
    old = pd.Series([200.0], index=pd.DatetimeIndex(["2022-12-31 23:00"]).tz_localize("Europe/Brussels"))
    with pytest.raises(ValueError, match="outside the intensity coverage"):
        co2_from_grid_kw(grid, 0.25, old)


# ---------------------------------------------------------------------------
# Resume safety, sharding, worker crashes (fleet_baseline)
# ---------------------------------------------------------------------------

SETTINGS = {"year": 2023, "mode": "realistic", "days": None, "gurobi_threads": 1,
            "disable_enforced": False, "tot_houses": 500, "signal_name": "real_price"}


def saved_meta(settings):
    meta = {k: v for k, v in settings.items() if k != "signal_name"}
    return {**meta, "signal": settings["signal_name"]}


def test_settings_mismatch_flags_a_short_run(tmp_path):
    fleet.save_house_result(tmp_path, 3, synthetic_result(), saved_meta({**SETTINGS, "days": 7}))
    assert fleet.settings_mismatch(tmp_path, 3, SETTINGS) == {"days": (7, None)}
    fleet.save_house_result(tmp_path, 4, synthetic_result(), saved_meta(SETTINGS))
    assert fleet.settings_mismatch(tmp_path, 4, SETTINGS) == {}


@needs_paper_data  # main() builds the house inventory from the paper's house configs
def test_main_refuses_done_houses_with_other_settings(tmp_path):
    out = tmp_path / "baseline_realistic_2023"
    fleet.save_house_result(out, 0, synthetic_result(), saved_meta({**SETTINGS, "days": 7}))
    with pytest.raises(SystemExit, match="other settings"):
        fleet_baseline.main(["--year", "2023", "--houses", "0-1", "--dry-run",
                             "--out-root", str(tmp_path)])


def test_default_name_includes_days():
    assert fleet_baseline.default_name("realistic", 2023, None) == "baseline_realistic_2023"
    assert fleet_baseline.default_name("realistic", 2023, 7) == "baseline_realistic_2023_7d"


def slurm_env(monkeypatch, task, lo, hi, count):
    for k, v in {"ID": task, "MIN": lo, "MAX": hi, "COUNT": count}.items():
        monkeypatch.setenv(f"SLURM_ARRAY_TASK_{k}", str(v))


def test_resolve_shard_full_array_uses_count(monkeypatch):
    slurm_env(monkeypatch, 3, 0, 9, 10)
    assert fleet_baseline.resolve_shard(None, None) == (3, 10)


def test_resolve_shard_partial_resubmission_needs_n_shards(monkeypatch):
    slurm_env(monkeypatch, 7, 3, 7, 2)  # sbatch --array=3,7
    with pytest.raises(SystemExit, match="--n-shards"):
        fleet_baseline.resolve_shard(None, None)
    assert fleet_baseline.resolve_shard(None, 10) == (7, 10)


def test_run_parallel_a_dead_process_only_fails_its_own_job():
    import os
    from concurrent.futures.process import BrokenProcessPool

    jobs = {0: (abs, (-1,)), 1: (os._exit, (1,)), 2: (abs, (-2,)), 3: (abs, (-3,))}
    out = {k: (r, e) for k, r, e in fleet_baseline.run_parallel(jobs, workers=4)}
    assert {k: out[k][0] for k in (0, 2, 3)} == {0: 1, 2: 2, 3: 3}
    assert all(out[k][1] is None for k in (0, 2, 3))
    assert isinstance(out[1][1], BrokenProcessPool)

"""Run one house with Julian's MPC, optionally planning on a separate control signal.

The MPC reads the price in exactly one place, `MPCManager.set_cost_constr`
(mpc_ems.py:501). The scorer `DayAheadEngie.calculate_kpi` (custom_classes.py:56)
reads the same `day_ahead_price` env value. To let the MPC plan on a signal while
opex is still billed at the real price, the signal is stored as a separate env value,
`control_signal`, and swapped in only while `set_cost_constr` builds the objective.
"""

import contextlib
import copy
import datetime
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from simugrid.simulation.config_parser import parse_config_file
from simugrid.simulation.environment import BaseListValue
from simugrid.assets.energyplus import EnergyPlus
from belgian_dwellings.simulation.calibrate_mpc import get_energyplus_calibration
from belgian_dwellings.simulation.custom_classes import DayAheadEngie
from belgian_dwellings.simulation.mpc_ems import MPCRealistic, PerfectMPC
from belgian_dwellings.utils.progress import simulation_bar

from fleet_harness.co2 import ffill_onto
from fleet_harness.runner import PAPER_ROOT, get_price_series

CAL_DIR = Path(__file__).resolve().parent.parent / "results" / "calibration"


def shifted_2023_config(config_path):
    """Write the 2023 version of a house config, atomically; return its path.

    Same output as the paper's tmp_2023_config (same path, same content, checked by a test),
    but that function truncates and rewrites the file on every call, so a parallel worker can
    read it empty (JSONDecodeError). Here the file is written to a unique temp name and moved
    into place with os.replace, so readers always see a complete file.
    """
    config_path = str(config_path)
    tmp_dir = os.path.dirname(config_path) + "_tmp_2023"
    os.makedirs(tmp_dir, exist_ok=True)
    with open(config_path) as f:
        config = json.load(f)
    config["Microgrid"]["start_time"] = config["Microgrid"]["start_time"].replace("2022", "2023")
    config["Microgrid"]["end_time"] = config["Microgrid"]["end_time"].replace("2023", "2024")
    for asset in config["Assets"].values():
        if asset["name"].startswith("EnergyPlus_"):
            asset["epw_weather"] = asset["epw_weather"].replace("2022", "2023")

    target = os.path.join(tmp_dir, os.path.basename(config_path))
    fd, partial = tempfile.mkstemp(dir=tmp_dir, prefix=".partial_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=4)
        os.replace(partial, target)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return target


def house_config_paths(house_num, tot_houses=500):
    """Return (original 2022 config, year-shifted 2023 config) as absolute paths."""
    original = str(PAPER_ROOT / "data" / f"houses_belgium_{tot_houses}" / f"house_{house_num}.json")
    return original, shifted_2023_config(original)


YEARS = (2022, 2023)


def _check_year(year):
    if year not in YEARS:
        raise ValueError(f"unknown year {year!r}, expected 2022 or 2023 (int)")


def house_config_path(house_num, year=2023, tot_houses=500):
    """Return the config that simulates `year`: the original file (2022) or the shifted one (2023)."""
    _check_year(year)
    original = str(PAPER_ROOT / "data" / f"houses_belgium_{tot_houses}" / f"house_{house_num}.json")
    return original if year == 2022 else shifted_2023_config(original)


def calibration_year(mode, year):
    """Year whose simulation the MPC is calibrated on, for a run of `year`.

    perfect:   the run year itself (Julian's rule: 2023 run -> 2023 calibration).
    realistic: the other year. Julian calibrates a 2023 run on 2022, so the EV predictor's
               history holds only sessions the house could have seen before. Calibrating a
               2022 run on 2022 would put every 2022 session, the one being predicted
               included, in the KNN training set (near-perfect EV forecasts). 2023 stands in
               as the "previous" year for 2022 runs instead.
    """
    _check_year(year)
    if mode == "perfect":
        return year
    if mode == "realistic":
        return 2023 if year == 2022 else 2022
    raise ValueError(f"unknown mode {mode!r}, expected 'perfect' or 'realistic'")


@contextlib.contextmanager
def parsed_house(house_num, tot_houses=500, year=2023):
    """Parse a house's config for `year` inside a temp dir, without simulating it.

    parse_config_file already starts EnergyPlus (which writes to ./out), so this runs in a
    temp working directory and stops the EnergyPlus threads on exit. Process-wide chdir:
    don't use from threads.
    """
    previous_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix=f"parse_house{house_num}_") as d:
        os.chdir(d)
        microgrid = None
        try:
            microgrid = parse_config_file(house_config_path(house_num, year, tot_houses))
            yield microgrid
        finally:
            if microgrid is not None:
                for asset in microgrid.assets:
                    if isinstance(asset, EnergyPlus):
                        asset.stop_energyplus_thread()
            os.chdir(previous_cwd)


def load_house_price_and_tariffs(house_num, tot_houses=500, year=2023):
    """Return (real day-ahead price Series, {offtake_extra, injection_extra}) for a house-year.

    The price index is timezone-naive UTC (the house configs use timezone UTC). The two
    extras are read at the first step; they are constant in the house configs.
    """
    with parsed_house(house_num, tot_houses, year) as microgrid:
        env_values = microgrid.environments[0].env_values
        price = get_price_series(microgrid)
        extras = {k: float(env_values[k].value) for k in ("offtake_extra", "injection_extra")}
    return price, extras


def real_price_day_ahead_cost(grid_kw, time_step_h, price, offtake_extra, injection_extra):
    """Recompute DayAheadEngie's `day_ahead` KPI from grid power and a price series.

    Same formula as custom_classes.DayAheadEngie.calculate_kpi:
      offtake   (E > 0): E * (price + offtake_extra), times 1.06 VAT when that price > 0
      injection (E <= 0): E * (price + injection_extra)
    and the KPI is minus the sum (negative = cost).

    grid_kw: PublicGrid_0 power in kW (positive = offtake), tz-aware UTC index, as in
        extract_results()["power"]. price: Series; a naive index is taken as UTC.
    Used to bill a run at the real price, independent of the signal the MPC planned on.
    """
    if price.index.tz is None:
        price = price.tz_localize("UTC")
    # The price file is 15-min (hourly also works); 1 h past a sample is the most allowed,
    # so a price for the wrong year or one sliced too short raises instead of repeating
    # its last value.
    p = ffill_onto(price, grid_kw.index, pd.Timedelta("1h"), "price").to_numpy()
    energy = grid_kw.to_numpy() * time_step_h
    offtake_price = p + offtake_extra
    offtake = energy * offtake_price * np.where(offtake_price > 0, 1.06, 1.0)
    injection = energy * (p + injection_extra)
    return -float(np.where(energy > 0, offtake, injection).sum())


def get_calibration(house_num, mode, tot_houses=500, year=2023):
    """Return (thermal calibration list, EV session history) for a run of `year`, cached on disk.

    Calibration is price-independent, so it is computed once per house, mode and
    calibration year. The calibration year follows calibration_year(): for 2023 runs this
    is Julian's rule in get_results.mpc_execution (realistic on 2022, perfect on 2023).
    The cache file is named after the calibration year, not the run year.

    A deep copy is returned every time: EasyForcaster appends new EV sessions to the
    history in place (forecaster.py:274-279), so reusing one object across runs would
    leak one run's sessions into the next run's EV predictor.
    """
    cal_year = calibration_year(mode, year)
    path = CAL_DIR / f"house_{house_num}_{tot_houses}_{mode}_cal{cal_year}.pkl"
    if path.exists():
        cal = pickle.loads(path.read_bytes())
    else:
        cal = get_energyplus_calibration(house_config_path(house_num, cal_year, tot_houses))
        CAL_DIR.mkdir(parents=True, exist_ok=True)
        # temp file + os.replace: a job reading the cache never sees a half-written pickle
        fd, partial = tempfile.mkstemp(dir=CAL_DIR, prefix=f".{path.name}.", suffix=".partial")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(pickle.dumps(cal))
            os.replace(partial, path)
        except BaseException:
            if os.path.exists(partial):
                os.remove(partial)
            raise
    return copy.deepcopy(cal)


def add_control_signal(microgrid, signal_series=None):
    """Add env value `control_signal`; defaults to a copy of the real day-ahead price.

    Always builds a new BaseListValue, even for the default: Environment.update() calls
    sample_next() once per dict entry, so one object under two keys would advance twice
    per step.
    """
    env_values = microgrid.environments[0].env_values
    baseline = get_price_series(microgrid)
    if signal_series is None:
        signal_series = baseline
    elif not signal_series.index.equals(baseline.index):
        raise ValueError("control_signal must have the same timestamps as day_ahead_price")

    value = BaseListValue.construct_from_two_lists(
        microgrid,
        list(signal_series.index.to_pydatetime()),
        signal_series.to_list(),
    )
    value.sample_date()
    env_values["control_signal"] = value


class ControlSignalMixin:
    """Make set_cost_constr read `control_signal` instead of `day_ahead_price`.

    set_cost_constr only calls .get_forecast() on the price, which does not advance the
    value, so swapping the key for the duration of the call is safe. The real price is
    restored before DayAheadEngie reads it later in the same step.
    """

    def set_cost_constr(self):
        env_values = self.microgrid.environments[0].env_values
        real_price = env_values["day_ahead_price"]
        env_values["day_ahead_price"] = env_values["control_signal"]
        try:
            super().set_cost_constr()
        finally:
            env_values["day_ahead_price"] = real_price


class SignalMPCRealistic(ControlSignalMixin, MPCRealistic):
    pass


class SignalPerfectMPC(ControlSignalMixin, PerfectMPC):
    pass


def record_solves(microgrid, ems):
    """Log the Gurobi outcome of every MPC step to `microgrid.solve_log` (observation only).

    Wraps the instance's update_control_points (called once per step by
    HouseManager.simulate_step). After it returns, the model holds the last solve of the
    step: its status (2 = optimal, 9 = time limit, 17 = memory limit, anything else =
    retries exhausted) and its runtime. Runtimes of earlier retries in the same step are
    not counted.
    """
    microgrid.solve_log = []
    original = ems.update_control_points

    def wrapped():
        start = microgrid.utc_datetime
        result = original()
        microgrid.solve_log.append((start, ems.model.Status, ems.model.Runtime))
        return result

    ems.update_control_points = wrapped


def _progress_logger(ems, house_num):
    """Opt-in diagnostic: with MPC_PROGRESS_EVERY=N in the environment, print one flushed
    line every N steps (simulated time, wall time, solves incl. comfort retries).

    Off (returns None) when the variable is unset, so normal runs are unchanged. Counts
    every `build_model` call on this ems instance: Julian's update_control_points calls it
    once per solve, also for each infeasibility retry (up to 41 per step), which
    record_solves cannot see (it keeps only the last solve of a step).
    """
    every = int(os.environ.get("MPC_PROGRESS_EVERY", "0") or 0)
    if every <= 0:
        return None

    state = {"builds": 0, "max_retry": 0, "steps": 0, "t0": time.perf_counter()}
    state["t_last"] = state["t0"]
    original_build = ems.build_model

    def counting_build(retry=0, *args, **kwargs):
        state["builds"] += 1
        state["max_retry"] = max(state["max_retry"], retry)
        return original_build(retry, *args, **kwargs)

    ems.build_model = counting_build

    def report(microgrid):
        state["steps"] += 1
        if state["steps"] % every:
            return
        now = time.perf_counter()
        solves = microgrid.solve_log[-every:]
        n_tl = sum(1 for _, status, _ in solves if status == 9)
        n_bad = sum(1 for _, status, _ in solves if status not in (2, 9, 17))
        print(
            f"[progress house {house_num}] sim {microgrid.utc_datetime:%Y-%m-%d %H:%M} UTC | "
            f"step {state['steps']} | wall {(now - state['t0']) / 60:.1f} min "
            f"(+{now - state['t_last']:.0f} s) | solves {state['builds']} for {every} steps, "
            f"max retry {state['max_retry']}, time-limit {n_tl}, non-optimal {n_bad}",
            flush=True,
        )
        state.update(builds=0, max_retry=0, t_last=now)

    return report


def run_one_house_mpc(
    house_num,
    mode="realistic",
    signal_series=None,
    disable_enforced=False,
    days=None,
    use_signal=True,
    tot_houses=500,
    gurobi_threads=1,
    year=2023,
):
    """Run one house-year (or the first `days` days) of `year` (2022 or 2023) with the MPC.

    use_signal=False runs Julian's unmodified MPC classes with the same calibration and
    window, as the baseline for the equivalence check.
    gurobi_threads overrides Julian's Threads = 4 per solve (None keeps his 4). Default 1,
    decided 3 Nov (experiments/thread_check.py): all solves optimal, opex within 0.003 %,
    +21 % fleet throughput (8 workers x 1 vs x 4). The thread count changes how Gurobi
    breaks ties, so every run that is compared (baselines included) must use the same value.
    The calibration year is the other year for realistic and the same year for perfect
    (see calibration_year).
    """
    cfg = house_config_path(house_num, year, tot_houses)
    thermal_cal, ev_hist = get_calibration(house_num, mode, tot_houses, year)

    microgrid = parse_config_file(cfg)
    microgrid.config_file = Path(cfg).name
    if use_signal:
        add_control_signal(microgrid, signal_series)
    microgrid.set_reward(DayAheadEngie())

    if mode == "perfect":
        cls = SignalPerfectMPC if use_signal else PerfectMPC
        ems = cls(microgrid, thermal_cal)
    elif mode == "realistic":
        cls = SignalMPCRealistic if use_signal else MPCRealistic
        ems = cls(microgrid, thermal_cal, ev_hist)
    else:
        raise ValueError(f"unknown mode {mode!r}, expected 'perfect' or 'realistic'")
    if disable_enforced:
        ems.do_forced_charging = False
    if gurobi_threads is not None:
        ems.model.Params.Threads = gurobi_threads
    record_solves(microgrid, ems)

    end_time = microgrid.end_time
    if days is not None:
        end_time = microgrid.start_time + datetime.timedelta(days=days)

    for asset, attr in [
        ("EnergyPlus_0", "zn0_temp"),
        ("EnergyPlus_0", "cur_t_low"),
        ("EnergyPlus_0", "cur_t_up"),
        ("Charger_0", "soc"),
        ("WaterHeater_0", "t_tank"),
    ]:
        microgrid.attribute_to_log(asset, attr)

    progress = _progress_logger(ems, house_num)
    bar = simulation_bar(microgrid, end_time, f"MPC {mode} house {house_num}")
    try:
        while microgrid.utc_datetime < end_time:
            microgrid.management_system.simulate_step()
            bar.update(1)
            if progress:
                progress(microgrid)
    finally:
        bar.close()
        ems.model.close()
        for asset in microgrid.assets:
            if isinstance(asset, EnergyPlus):
                asset.stop_energyplus_thread()
    return microgrid


PRICE_COL = "day_ahead_price_eur_per_kwh"
SIGNAL_COL = "control_signal"


def _env_series(microgrid, key):
    """An env value's full series (naive UTC index), like runner.get_price_series."""
    value = microgrid.environments[0].env_values[key]
    dts, vals = zip(*value.dt_and_values)
    return pd.Series(list(vals), index=pd.DatetimeIndex(list(dts)))


def signal_table(microgrid, start=None, end=None):
    """The real price and the signal the MPC planned on, over [start, end).

    Columns: day_ahead_price_eur_per_kwh (what the bill uses) and control_signal (what the
    MPC optimised; equal to the price when no signal was injected or use_signal=False).
    Index: tz-aware UTC, named datetime, like the fleet power table. The env values hold the
    whole input file (2022-2023), so the default window is the simulated part: from the
    microgrid start to its current time (the end of a finished run).
    """
    price = _env_series(microgrid, "day_ahead_price")
    env_values = microgrid.environments[0].env_values
    signal = _env_series(microgrid, "control_signal") if "control_signal" in env_values else price
    table = pd.DataFrame({PRICE_COL: price, SIGNAL_COL: signal})
    table.index = table.index.tz_localize("UTC")
    table.index.name = "datetime"
    start = pd.Timestamp(start if start is not None else microgrid.start_time).tz_convert("UTC")
    end = pd.Timestamp(end if end is not None else microgrid.utc_datetime).tz_convert("UTC")
    return table[(table.index >= start) & (table.index < end)]


def extract_results(microgrid):
    """Pull plain, picklable results out of a finished run.

    The microgrid itself holds EnergyPlus threads and a Gurobi model, so it cannot be
    returned from a worker process. Returns a dict with:
      kpis       DayAheadEngie KPIs at the end of the run (opex is billed at the real price)
      power      one row per step, one column per asset, Power.electrical in kW
                 (PublicGrid_0: positive = offtake, negative = injection, as in co2.py)
      attributes the attribute_to_log series (indoor temp, comfort bounds, EV soc, tank temp)
      solves     one row per MPC solve: Gurobi status and runtime (see record_solves)
      signal     real price and control signal over the simulated window (signal_table)
      time_step_h
    Timestamps are kept as recorded (tz-aware). power_hist stamps the start of each step,
    attributes_hist the end (datetime + time_step), so the two frames are offset by one step.
    """
    kpis = {k: float(v) for k, v in microgrid.tot_reward.KPIs.items()}

    node0 = microgrid.power_hist[0]  # single-node house configs
    power = pd.DataFrame(
        {name: [p.electrical for p in values] for name, values in node0.items() if name != "datetime"},
        index=pd.DatetimeIndex(node0["datetime"]),
    )

    attributes = {}
    for asset, att_dict in microgrid.attributes_hist.items():
        index = pd.DatetimeIndex(att_dict["datetime"])
        for att_name, values in att_dict.items():
            if att_name != "datetime":
                attributes[f"{asset.name}.{att_name}"] = pd.Series(values, index=index)
    attributes = pd.DataFrame(attributes)

    solves = pd.DataFrame(
        getattr(microgrid, "solve_log", []), columns=["datetime", "status", "runtime_s"]
    ).set_index("datetime")

    return {
        "kpis": kpis,
        "power": power,
        "attributes": attributes,
        "solves": solves,
        "signal": signal_table(microgrid),
        "time_step_h": microgrid.time_step.total_seconds() / 3600,
    }


def run_isolated(house_num, **kwargs):
    """Run one house inside its own temporary working directory and return extract_results.

    simugrid launches EnergyPlus with `-d out`, relative to the current directory, so runs
    sharing a directory overwrite each other's output. This also covers calibration, which
    runs EnergyPlus inside run_one_house_mpc when the cache is empty.

    os.chdir is process-wide: call this in separate PROCESSES (e.g. ProcessPoolExecutor),
    never threads. The previous working directory is restored afterwards. All paths used by
    run_one_house_mpc are absolute.
    """
    previous_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix=f"eplus_house{house_num}_") as run_dir:
        os.chdir(run_dir)
        try:
            microgrid = run_one_house_mpc(house_num, **kwargs)
            return extract_results(microgrid)
        finally:
            os.chdir(previous_cwd)

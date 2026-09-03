import os
from pathlib import Path

import pandas as pd

from simugrid.simulation.config_parser import parse_config_file
from simugrid.simulation.environment import BaseListValue
from simugrid.assets.energyplus import EnergyPlus
from treec.logger import TreeLogger

from belgian_dwellings.simulation.run_treec import (
    find_best_tree_for_each_house,
    execute_treec,
)
from belgian_dwellings.simulation.tmp_2023_config import tmp_2023_config
from belgian_dwellings.simulation.custom_classes import DayAheadEngie, TreeManager
from belgian_dwellings.training.input_functions import first_input_function
from belgian_dwellings.utils.progress import simulation_bar

PAPER_ROOT = Path(__file__).resolve().parent.parent / "should_belgians_install_advanced_ems_paper"


def _resolve_house_paths(house_num, tot_houses):
    """Look up a house's trained-tree model folder and its year-shifted config.

    Writes only into should_belgians_install_advanced_ems_paper/data/houses_belgium_*_tmp_2023/,
    the same disposable scratch copy get_results.py itself produces. Never touches the original
    house configs or the trained trees under treec_train_*/.
    """
    log_folder = str(PAPER_ROOT / f"treec_train_{tot_houses}")
    best_trees = find_best_tree_for_each_house(log_folder)

    run_house = f"house_{house_num}"
    model_path = best_trees[run_house]["model_path"]
    model_folder = os.path.dirname(model_path) + "/"

    config_file = str(PAPER_ROOT / "data" / f"houses_belgium_{tot_houses}" / f"{run_house}.json")
    tmp_config_file = tmp_2023_config(config_file)

    return tmp_config_file, model_folder


def run_one_house(house_num, tot_houses=500, disable_enforced=True):
    """Run a single trained TreeC house and return its microgrid.

    Faithful wrapper around get_results.py's actual execute_treec call (year-shifted config).
    No signal injection — this is the baseline (rung 0).
    """
    tmp_config_file, model_folder = _resolve_house_paths(house_num, tot_houses)
    return execute_treec(tmp_config_file, model_folder, disable_enforced=disable_enforced)


def get_price_series(microgrid):
    """Read a parsed microgrid's day-ahead price BaseListValue out as a pandas Series."""
    day_ahead = microgrid.environments[0].env_values["day_ahead_price"]
    dts, vals = zip(*day_ahead.dt_and_values)
    return pd.Series(list(vals), index=pd.DatetimeIndex(list(dts)))


def run_one_house_with_price(house_num, price_transform, tot_houses=500, disable_enforced=True):
    """Run a single trained TreeC house with its day-ahead price replaced in memory.

    `price_transform` receives the house's own baseline price series (a pandas Series, as
    returned by get_price_series) and must return the replacement series. The replacement is
    wrapped back into a BaseListValue and swapped into the microgrid's environment before the
    simulation loop starts — same tree, same house, same simulated year as run_one_house(), only
    the price signal the tree reads each step differs. Nothing is written to disk beyond the
    same disposable tmp_2023 config _resolve_house_paths already produces for the baseline.
    """
    tmp_config_file, model_folder = _resolve_house_paths(house_num, tot_houses)

    microgrid = parse_config_file(tmp_config_file)

    baseline_price = get_price_series(microgrid)
    new_price_series = price_transform(baseline_price)
    # BaseListValue.construct_from_pandas_series assumes numpy's datetime64[ns] .tolist()
    # returns integer nanoseconds; on current numpy/pandas it returns datetime objects instead,
    # which crashes that helper. construct_from_two_lists takes real datetimes directly.
    new_price_value = BaseListValue.construct_from_two_lists(
        microgrid,
        list(new_price_series.index.to_pydatetime()),
        new_price_series.to_list(),
    )
    microgrid.environments[0].env_values["day_ahead_price"] = new_price_value
    # The original value was already primed for microgrid.datetime by parse_config_file's
    # new_start_time() call, before we swapped it out. Ours hasn't been sampled yet, and the
    # very first reward calculation reads .value directly (not get_forecast), so without this
    # it's still None and calculate_kpi() crashes on the first step.
    new_price_value.sample_date()

    microgrid.set_reward(DayAheadEngie())

    trees = TreeLogger.get_best_trees(model_folder)
    ems = TreeManager(microgrid, trees, first_input_function)
    if disable_enforced:
        ems.do_forced_charging = False

    end_time = microgrid.end_time
    microgrid.attribute_to_log("EnergyPlus_0", "zn0_temp")
    microgrid.attribute_to_log("EnergyPlus_0", "cur_t_low")
    microgrid.attribute_to_log("EnergyPlus_0", "cur_t_up")
    microgrid.attribute_to_log("Charger_0", "soc")
    microgrid.attribute_to_log("WaterHeater_0", "t_tank")
    bar = simulation_bar(microgrid, end_time, "TreeC (custom price)")
    while microgrid.utc_datetime < end_time:
        microgrid.management_system.simulate_step()
        bar.update(1)
    bar.close()

    for asset in microgrid.assets:
        if isinstance(asset, EnergyPlus):
            asset.stop_energyplus_thread()

    return microgrid
"""Phase 1 'done when' run: house 0, full 2023, MPC realistic, real price vs flat signal.

Runs the two signals (real price, flat signal) in parallel, both on 2023 (one process each,
own EnergyPlus dir via run_isolated), saves the extracted results to results/mpc_runs/
(pickle + a _signal.csv with the real price and the control signal per step), and prints
timing and KPIs.

Run from the repo root:
    python -m experiments.full_year_house0 2>&1 | tee results/mpc_runs/full_year_house0.log
"""

import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from fleet_harness.mpc_runner import get_calibration, house_config_paths, run_isolated

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "mpc_runs"
HOUSE = 0


def flat_signal(house_num):
    """The house's own real day-ahead price, flattened to the mean of the WHOLE price file.

    get_price_series holds the whole file (2021-12-30 .. 2024-01-01), so this mean is
    0.1705 EUR/kWh, not the 2023 mean (0.0973). Kept as is so the 3 Nov run-4 numbers stay
    reproducible; new signals should be built from the simulated year only.
    Note: run 4 used 4 Gurobi threads; this script now runs with the 1-thread default.
    """
    import os
    import tempfile

    from simugrid.assets.energyplus import EnergyPlus
    from simugrid.simulation.config_parser import parse_config_file

    from fleet_harness.runner import get_price_series

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            mg = parse_config_file(house_config_paths(house_num)[1])
            price = get_price_series(mg)
            for asset in mg.assets:
                if isinstance(asset, EnergyPlus):
                    asset.stop_energyplus_thread()
        finally:
            os.chdir(cwd)
    return pd.Series(price.mean(), index=price.index)


def timed_run(name, kwargs):
    t = time.time()
    result = run_isolated(HOUSE, **kwargs)
    result["elapsed_s"] = time.time() - t
    result["name"] = name
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    get_calibration(HOUSE, "realistic")  # make sure the cache exists before the workers start

    jobs = {
        "real_price": {},
        "flat_signal": {"signal_series": flat_signal(HOUSE)},
    }

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(timed_run, name, kw) for name, kw in jobs.items()}
        results = {name: f.result() for name, f in futures.items()}
    wall = time.time() - t0

    for name, r in results.items():
        path = OUT_DIR / f"house{HOUSE}_2023_realistic_{name}.pkl"
        path.write_bytes(pickle.dumps(r))
        r["signal"].to_csv(path.with_name(path.stem + "_signal.csv"))  # price + signal, for plots
        grid = r["power"]["PublicGrid_0"]
        dt = r["time_step_h"]
        print(f"\n== {name}: {r['elapsed_s'] / 60:.1f} min, {len(grid)} steps, saved {path.name}")
        print(f"   offtake {grid.clip(lower=0).sum() * dt:.1f} kWh | injection {-grid.clip(upper=0).sum() * dt:.1f} kWh")
        for k, v in r["kpis"].items():
            print(f"   {k}: {v:.3f}")

    print(f"\nwall time, 2 runs in parallel: {wall / 60:.1f} min")


if __name__ == "__main__":
    main()

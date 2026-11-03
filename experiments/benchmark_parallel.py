"""Phase 1 benchmark: how MPC throughput scales with the number of parallel workers.

Each configuration runs `workers` copies of the same job at once (house 0, first DAYS days
of 2023, MPC realistic, real price), one process per job, each in its own EnergyPlus dir.
Using one house for every job keeps the workload identical, so only parallelism changes.

Measured per configuration: wall time, mean and max time per run, peak memory per worker,
and throughput in simulated house-days per minute, extrapolated to house-years per hour.
The extrapolation assumes the January week is representative; calibrate it against the
full-year run (32.5 min per house-year at 2 workers x 4 threads, run 4).

Run from the repo root (takes roughly 10-20 min; nothing else heavy should run meanwhile):
    python -u -m experiments.benchmark_parallel 2>&1 | tee results/mpc_runs/benchmark_parallel.log

Specific configurations, saved to a separate CSV so the full table is not overwritten:
    python -u -m experiments.benchmark_parallel --configs 12x1 14x1 --name benchmark_14workers
"""

import argparse
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from fleet_harness.mpc_runner import get_calibration, run_isolated

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "mpc_runs"
HOUSE = 0
DAYS = 7

# (workers, gurobi threads per solve). Threads = 4 is Julian's setting; the last row tests
# many single-threaded workers instead, which is often faster for small MILPs.
CONFIGS = [(1, 4), (2, 4), (4, 4), (6, 4), (8, 4), (12, 1)]


def job(threads):
    t = time.time()
    run_isolated(HOUSE, days=DAYS, gurobi_threads=threads)
    elapsed = time.time() - t
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB
    return elapsed, peak_mb


def power_state():
    """Laptop power state, printed with the results: on battery the CPU throttles."""
    import glob
    import subprocess

    ac = [open(f).read().strip() for f in glob.glob("/sys/class/power_supply/A*/online")]
    try:
        profile = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        profile = "unknown"
    plugged = "plugged in" if "1" in ac else ("ON BATTERY" if ac else "unknown")
    return f"power: {plugged} | profile: {profile or 'unknown'}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", metavar="WxT",
                        help="workers x threads, e.g. 12x1 14x1 (default: the full sweep)")
    parser.add_argument("--name", default="benchmark_parallel", help="output CSV name (no extension)")
    args = parser.parse_args()
    configs = CONFIGS
    if args.configs:
        configs = [tuple(int(n) for n in c.lower().split("x")) for c in args.configs]
    return configs, OUT_DIR / f"{args.name}.csv"


def main():
    configs, out = parse_args()
    out.parent.mkdir(parents=True, exist_ok=True)
    get_calibration(HOUSE, "realistic")  # cached before workers start
    print(f"cores: {os.cpu_count()} | house {HOUSE} | {DAYS} days per run | {power_state()}\n")

    rows = []
    for workers, threads in configs:
        t0 = time.time()
        # max_tasks_per_child=1: a fresh process per job, so peak memory is per run
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
            results = list(pool.map(job, [threads] * workers))
        wall = time.time() - t0

        run_times = [r[0] for r in results]
        house_days_per_min = workers * DAYS / (wall / 60)
        row = {
            "workers": workers,
            "threads_per_solve": threads,
            "wall_s": round(wall, 1),
            "mean_run_s": round(sum(run_times) / workers, 1),
            "max_run_s": round(max(run_times), 1),
            "peak_mem_per_worker_mb": round(max(r[1] for r in results)),
            "house_days_per_min": round(house_days_per_min, 2),
            "house_years_per_hour": round(house_days_per_min * 60 / 365, 2),
        }
        rows.append(row)
        print(
            f"{workers:>2} workers x {threads} threads: wall {wall:6.1f}s | "
            f"run mean {row['mean_run_s']:6.1f}s max {row['max_run_s']:6.1f}s | "
            f"{row['house_years_per_hour']:.2f} house-years/h | "
            f"{row['peak_mem_per_worker_mb']} MB/worker"
        )
        pd.DataFrame(rows).to_csv(out, index=False)  # saved after every config

    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

"""Phase 1 check: can the MPC run with 1 Gurobi thread instead of Julian's 4?

The benchmark showed 12 workers x 1 thread gives ~40 % more throughput than 8 x 4.
That is only usable if 1 thread does not change the results or hit the 10 s time limit
more often. This runs house 0 (first DAYS days of 2023, realistic, real price) with
4 and 1 threads side by side and compares:
  - KPIs (opex, discomfort, forced charging, ...) and grid power;
  - Gurobi outcome per step: time-limit hits, non-optimal steps, solve times.

Both runs execute at the same time, so they see the same machine load.

Run from the repo root (~1-2 min):
    python -u -m experiments.thread_check 2>&1 | tee results/mpc_runs/thread_check.log

How to read it:
  - identical KPIs and grid power      -> adopt 1 thread freely;
  - small differences, both all optimal -> Gurobi picked a different optimum among ties;
    adopt only if every run (baselines included) uses 1 thread;
  - more time-limit hits with 1 thread  -> keep 4 threads (or fewer workers).
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fleet_harness.mpc_runner import get_calibration, run_isolated

OUT = Path(__file__).resolve().parent.parent / "results" / "mpc_runs"
HOUSE = 0
DAYS = 7
THREADS = [4, 1]

OPTIMAL, TIME_LIMIT = 2, 9


def job(threads):
    return run_isolated(HOUSE, days=DAYS, gurobi_threads=threads)


def solve_summary(solves):
    s = solves["status"]
    t = solves["runtime_s"]
    return {
        "solves": len(s),
        "optimal": int((s == OPTIMAL).sum()),
        "time_limit": int((s == TIME_LIMIT).sum()),
        "other_status": int((~s.isin([OPTIMAL, TIME_LIMIT])).sum()),
        "runtime_mean_s": round(t.mean(), 3),
        "runtime_p99_s": round(t.quantile(0.99), 3),
        "runtime_max_s": round(t.max(), 3),
        "runtime_total_s": round(t.sum(), 1),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    get_calibration(HOUSE, "realistic")  # cached before workers start
    print(f"house {HOUSE} | {DAYS} days | threads {THREADS} run in parallel\n")

    with ProcessPoolExecutor(max_workers=len(THREADS), max_tasks_per_child=1) as pool:
        results = dict(zip(THREADS, pool.map(job, THREADS)))

    a, b = (results[t] for t in THREADS)

    kpis = pd.DataFrame({f"{t} threads": results[t]["kpis"] for t in THREADS})
    kpis["diff"] = kpis.iloc[:, 1] - kpis.iloc[:, 0]
    print("KPIs")
    print(kpis.to_string(float_format=lambda x: f"{x:.4f}"), "\n")

    solves = pd.DataFrame({f"{t} threads": solve_summary(results[t]["solves"]) for t in THREADS})
    print("Gurobi solves")
    print(solves.to_string(), "\n")

    ga, gb = a["power"]["PublicGrid_0"], b["power"]["PublicGrid_0"]
    diff = (gb - ga).abs()
    identical = a["kpis"] == b["kpis"] and ga.equals(gb)
    print("Grid power (kW)")
    print(f"  steps that differ: {(diff > 1e-6).sum()} of {len(diff)}")
    print(f"  max abs diff:      {diff.max():.4f}")
    print(f"  offtake kWh:       {np.clip(ga, 0, None).sum() * a['time_step_h']:.2f} (4) vs "
          f"{np.clip(gb, 0, None).sum() * b['time_step_h']:.2f} (1)\n")
    print("RESULT:", "identical" if identical else "different (see read-me in the docstring)")

    kpis.to_csv(OUT / "thread_check_kpis.csv")
    solves.to_csv(OUT / "thread_check_solves.csv")
    print(f"\nsaved {OUT / 'thread_check_kpis.csv'} and thread_check_solves.csv")


if __name__ == "__main__":
    main()

"""Phase 2 fleet baseline: run houses with the MPC on the real day-ahead price, one result
folder per (mode, year), one set of files per house (see fleet_harness/fleet.py).

Written for the CÉCI cluster (SLURM), also runs on a laptop. Resumable: houses whose
house_{n}.json exists are skipped, so rerunning the same command after a crash, timeout or
preemption only runs what is missing. Failed houses leave house_{n}.error.txt and are
retried on the next run.

    # all 500 houses of 2023, one process
    python -u -m experiments.fleet_baseline --year 2023 --workers 10

    # SLURM array: task k of N takes houses k, k+N, k+2N ... (reads SLURM_ARRAY_TASK_ID
    # and SLURM_CPUS_PER_TASK by itself). Always give --n-shards: SLURM_ARRAY_TASK_COUNT is
    # the number of tasks submitted, so resubmitting only --array=3,7 would split by 2.
    python -u -m experiments.fleet_baseline --year 2023 --n-shards 10

    # smoke test: 1 week of a few houses covering every stratum (the default folder name
    # includes the day count, and done houses saved with other settings stop the run)
    python -u -m experiments.fleet_baseline --year 2023 --houses smoke --days 7 --name smoke_2023

    # calibration only (EnergyPlus, ~15 s per house), e.g. as a first short job
    python -u -m experiments.fleet_baseline --year 2023 --calibrate-only
"""

import argparse
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from fleet_harness.fleet import (
    FLEET_DIR,
    house_inventory,
    house_paths,
    is_done,
    provenance,
    run_and_save,
    settings_mismatch,
)

# Heaviest heating/EV class first (PV ignored, ties by house number) so the long runs
# don't end up alone at the tail of the job.
COST_ORDER = {"heat+ev": 0, "heat+noev": 1, "noheat+ev": 2, "noheat+noev": 3}


def parse_houses(spec, inv):
    """'0-499', '0,3,10-12', 'smoke' (first two houses of each stratum) or 'all'."""
    if spec in (None, "all"):
        return list(inv.index)
    if spec == "smoke":
        return sorted(inv.groupby("stratum").head(2).index)
    houses = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            houses.extend(range(lo, hi + 1))
        else:
            houses.append(int(part))
    unknown = sorted(set(houses) - set(inv.index))
    if unknown:
        raise SystemExit(f"unknown houses: {unknown[:10]}")
    return sorted(set(houses))


def heavy_first(houses, inv):
    def key(h):
        s = inv.loc[h, "stratum"]
        return (COST_ORDER["+".join(s.split("+")[:2])], h)
    return sorted(houses, key=key)


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True, choices=[2022, 2023])
    p.add_argument("--mode", default="realistic", choices=["realistic", "perfect"])
    p.add_argument("--houses", default="all", help="'all', 'smoke', '0-499' or '0,3,10-12'")
    p.add_argument("--days", type=int, default=None, help="first N days only (default: full year)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel houses (default: SLURM_CPUS_PER_TASK, else 10)")
    p.add_argument("--gurobi-threads", type=int, default=1, help="keep 1 for every thesis run")
    p.add_argument("--shard", type=int, default=None, help="this task's index (default SLURM_ARRAY_TASK_ID)")
    p.add_argument("--n-shards", type=int, default=None,
                   help="number of shards; required with SLURM unless the array is exactly 0..N-1")
    p.add_argument("--name", default=None,
                   help="result folder name (default baseline_{mode}_{year}, + _{days}d with --days)")
    p.add_argument("--out-root", type=Path, default=FLEET_DIR)
    p.add_argument("--tot-houses", type=int, default=500)
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the houses this task would run")
    return p.parse_args(argv)


def calibrate(house, mode, year, tot_houses):
    """Build the calibration cache of one house (EnergyPlus writes to ./out, so use a temp cwd)."""
    from fleet_harness.mpc_runner import get_calibration

    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix=f"cal_house{house}_") as d:
        os.chdir(d)
        try:
            get_calibration(house, mode, tot_houses, year)
        finally:
            os.chdir(previous)
    return house


def main(argv=None):
    args = parse_args(argv)
    inv = house_inventory(args.tot_houses)
    houses = parse_houses(args.houses, inv)

    shard, n_shards = resolve_shard(args.shard, args.n_shards)
    if shard is not None:
        houses = houses[shard::n_shards]

    workers = args.workers or env_int("SLURM_CPUS_PER_TASK", 10)
    name = args.name or default_name(args.mode, args.year, args.days)
    out_dir = args.out_root / name

    settings = {
        "year": args.year,
        "mode": args.mode,
        "days": args.days,
        "gurobi_threads": args.gurobi_threads,
        "disable_enforced": False,
        "tot_houses": args.tot_houses,
        "signal_name": "real_price",
    }
    done = [h for h in houses if is_done(out_dir, h)]
    if not args.calibrate_only:
        stale = {h: d for h in done if (d := settings_mismatch(out_dir, h, settings))}
        if stale:
            h, diff = next(iter(stale.items()))
            raise SystemExit(
                f"{len(stale)} done house(s) in {out_dir} were run with other settings, e.g. "
                f"house {h}: {diff} (saved, wanted). Use another --name or remove them."
            )
    todo = heavy_first([h for h in houses if h not in done], inv)

    print(f"{name}: {len(houses)} houses in this task, {len(houses) - len(todo)} already done, "
          f"{len(todo)} to run, {workers} workers, out={out_dir}", flush=True)
    if args.dry_run or not todo:
        print(todo)
        return 0

    t0 = time.perf_counter()
    failed = []
    if args.calibrate_only:
        jobs = {h: (calibrate, (h, args.mode, args.year, args.tot_houses)) for h in todo}
        for i, (h, _, err) in enumerate(run_parallel(jobs, workers), 1):
            if err is None:
                print(f"[{i}/{len(todo)}] calibrated house {h}", flush=True)
            else:
                failed.append(h)
                print(f"[{i}/{len(todo)}] calibration of house {h} FAILED: {err!r}", flush=True)
        return 1 if failed else 0

    prov = provenance()
    if prov["git_dirty"]:
        print("WARNING: uncommitted changes in the repo; results record git_dirty=true", flush=True)

    jobs = {h: (run_and_save, (h, out_dir, settings, prov)) for h in todo}
    for i, (h, r, err) in enumerate(run_parallel(jobs, workers), 1):
        if err is not None:  # the process died, so run_and_save could not record it
            r = {"house": h, "ok": False, "elapsed_s": float("nan"),
                 "error": f"worker process died: {err!r}"}
            out_dir.mkdir(parents=True, exist_ok=True)
            house_paths(out_dir, h)["error"].write_text(r["error"] + "\n")
        elapsed_h = (time.perf_counter() - t0) / 3600
        eta_h = elapsed_h / i * (len(todo) - i)
        status = "ok" if r["ok"] else "FAILED"
        print(f"[{i}/{len(todo)}] house {h} {status} in {r['elapsed_s'] / 60:.1f} min "
              f"| elapsed {elapsed_h:.2f} h, ETA {eta_h:.2f} h", flush=True)
        if not r["ok"]:
            failed.append(h)
            print(r["error"].strip().splitlines()[-1], flush=True)

    print(f"done: {len(todo) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


def _in_own_process(fn, *args):
    """Run fn(*args) in a fresh spawned process of its own (one per house: EnergyPlus
    threads, Gurobi env, memory). If that process dies (out of memory, segfault) only
    this call raises BrokenProcessPool; the houses running beside it are not affected,
    unlike a shared pool, where every queued house would fail with it."""
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


def run_parallel(jobs, workers):
    """jobs: {key: (fn, args)}. Runs at most `workers` at once, each in its own process,
    and yields (key, result, error) as they finish; error is None or the exception (e.g.
    BrokenProcessPool when the process died). The threads only wait on their child."""
    with ThreadPoolExecutor(workers) as threads:
        futures = {threads.submit(_in_own_process, fn, *args): key for key, (fn, args) in jobs.items()}
        for fut in as_completed(futures):
            try:
                yield futures[fut], fut.result(), None
            except Exception as e:  # noqa: BLE001
                yield futures[fut], None, e


def default_name(mode, year, days):
    return f"baseline_{mode}_{year}" + (f"_{days}d" if days else "")


def resolve_shard(shard, n_shards):
    """(shard, n_shards), each from the arguments or SLURM; (None, None) = no sharding.

    n_shards falls back to SLURM_ARRAY_TASK_COUNT only when the array is exactly 0..N-1:
    for a partial resubmission (--array=3,7) the count is 2, which would split the houses
    differently from the original 10-task run.
    """
    if shard is None:
        shard = env_int("SLURM_ARRAY_TASK_ID", None)
    if n_shards is None and shard is not None:
        count = env_int("SLURM_ARRAY_TASK_COUNT", None)
        lo, hi = env_int("SLURM_ARRAY_TASK_MIN", None), env_int("SLURM_ARRAY_TASK_MAX", None)
        if count is not None and lo == 0 and hi == count - 1:
            n_shards = count
        elif count is not None:
            raise SystemExit(
                f"SLURM array is not 0..N-1 (min {lo}, max {hi}, count {count}): "
                "give --n-shards explicitly"
            )
    if (shard is None) != (n_shards is None):
        raise SystemExit("give both --shard and --n-shards (or neither)")
    if shard is not None and not 0 <= shard < n_shards:
        raise SystemExit(f"shard {shard} outside 0..{n_shards - 1}")
    return shard, n_shards


if __name__ == "__main__":
    sys.exit(main())

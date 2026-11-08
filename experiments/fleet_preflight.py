"""Check that a machine can run the fleet, without simulating anything (seconds).

Run this first on the cluster (login node is fine, except the Gurobi check, which needs the
licence the compute nodes will use):

    python -m experiments.fleet_preflight --years 2022 2023

Checks, for every house config:
  - every referenced input file exists (appliance load, hot water, EV sessions, PV, price,
    EnergyPlus model and weather of each year, EnergyPlus install);
  - each time-series CSV covers the simulated year (first and last timestamp);
and once: pyarrow can write Parquet, the results folder is writable, and Gurobi can solve a
model above the 2000-variable limit of the restricted pip licence.
Exit code 1 if anything fails.
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from fleet_harness.fleet import FLEET_DIR
from fleet_harness.runner import PAPER_ROOT


def first_last_timestamp(path):
    """First and last timestamp of a CSV whose first column is a timestamp (reads 2 lines)."""
    with open(path, "rb") as f:
        f.readline()  # header
        first = f.readline().split(b",")[0].decode()
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        last = f.read().strip().splitlines()[-1].split(b",")[0].decode()
    return pd.Timestamp(first), pd.Timestamp(last)


def year_window(year):
    return pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year}-12-31 23:45", tz="UTC")


def check_configs(years, tot_houses):
    folder = PAPER_ROOT / "data" / f"houses_belgium_{tot_houses}"
    problems = []
    csv_files = defaultdict(set)  # csv path -> houses
    for house in range(tot_houses):
        cfg_path = folder / f"house_{house}.json"
        try:
            cfg = json.loads(cfg_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            problems.append(f"house {house}: config unreadable: {e!r}")
            continue
        env = cfg["Environments"]["Environment_0"]
        refs = [v for v in env.values() if isinstance(v, str) and v.startswith("./")]
        refs += [k for k in env if k.startswith("./")]  # EV sessions are stored as keys
        for asset in cfg["Assets"].values():
            if asset["name"].startswith("EnergyPlus_"):
                refs.append(asset["idf_model"])
                refs.append(asset["eplus_dir"])
                for year in years:
                    refs.append(asset["epw_weather"].replace("2022", str(year)))
        for ref in refs:
            path = (folder / ref).resolve()
            if not path.exists():
                problems.append(f"house {house}: missing {path}")
            elif path.suffix == ".csv":
                csv_files[path].add(house)
    for path, houses in csv_files.items():
        try:
            first, last = first_last_timestamp(path)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{path.name}: cannot read timestamps: {e!r}")
            continue
        for year in years:
            lo, hi = year_window(year)
            if first.tz is None:
                first, last = first.tz_localize("UTC"), last.tz_localize("UTC")
            if first > lo or last < hi:
                problems.append(f"{path.name} (houses {sorted(houses)[:5]}...): covers "
                                f"{first} -> {last}, not all of {year}")
    return problems, len(csv_files)


def check_gurobi():
    import gurobipy as gp

    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
        x = m.addVars(2500, ub=1.0)  # above the restricted licence's 2000-variable limit
        m.setObjective(x.sum(), gp.GRB.MAXIMIZE)
        m.optimize()
        if m.Status != gp.GRB.OPTIMAL:
            return [f"Gurobi test model status {m.Status}"]
    return []


def check_output(out_root):
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out_root) as d:
            pd.DataFrame({"a": [1.0]}).to_parquet(Path(d) / "t.parquet")
    except Exception as e:  # noqa: BLE001
        return [f"cannot write Parquet under {out_root}: {e!r}"]
    return []


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="+", default=[2022, 2023])
    p.add_argument("--tot-houses", type=int, default=500)
    p.add_argument("--out-root", type=Path, default=FLEET_DIR)
    p.add_argument("--skip-gurobi", action="store_true")
    args = p.parse_args(argv)

    problems, n_csv = check_configs(args.years, args.tot_houses)
    print(f"configs: {args.tot_houses} houses, {n_csv} distinct CSV inputs checked for {args.years}")
    problems += check_output(args.out_root)
    if not args.skip_gurobi:
        try:
            problems += check_gurobi()
            print("gurobi: full licence OK (2500-variable model solved)")
        except Exception as e:  # noqa: BLE001
            problems.append(f"Gurobi: {e!r}")

    for line in problems:
        print("PROBLEM:", line)
    print("preflight OK" if not problems else f"preflight FAILED ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

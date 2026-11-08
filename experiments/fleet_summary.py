"""Summarise a finished (or partly finished) fleet run: per-house table, fleet totals per
stratum, sanity checks, and average-factor CO2 when the ENTSO-E mix covers the year.

    python -m experiments.fleet_summary results/fleet/baseline_realistic_2023
    python -m experiments.fleet_summary results/fleet/baseline_realistic_2023 \
        --mix data/entsoe_be_2023_generation.csv --factor-set direct

Writes summary.csv (one row per house) and strata.csv next to the results.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fleet_harness.co2 import co2_from_grid_kw
from fleet_harness.fleet import house_inventory, load_house, load_run_summary


def sanity_checks(summary, inv):
    """Problems that should not happen in a good baseline (list of strings)."""
    problems = []
    s = summary
    if "solves.n_time_limit" in s and s["solves.n_time_limit"].sum() > 0:
        bad = s.index[s["solves.n_time_limit"] > 0].tolist()
        problems.append(f"Gurobi time-limit hits in houses {bad[:10]}")
    if "solves.n_other" in s and s["solves.n_other"].sum() > 0:
        bad = s.index[s["solves.n_other"] > 0].tolist()
        problems.append(f"non-optimal solves (retries exhausted?) in houses {bad[:10]}")
    expected_steps = s["n_steps"].mode().iloc[0]
    short = s.index[s["n_steps"] != expected_steps].tolist()
    if short:
        problems.append(f"houses with a different number of steps than {expected_steps}: {short[:10]}")
    joined = s.join(inv[[c for c in inv if c not in s]])
    for col, flag in [("energy_kwh.heating_kwh", "has_heating"), ("energy_kwh.ev_kwh", "has_ev"),
                      ("energy_kwh.pv_kwh", "has_pv")]:
        wrong = joined.index[(joined[col] > 0) != joined[flag]].tolist()
        if wrong:
            problems.append(f"{col} > 0 does not match {flag} for houses {wrong[:10]}")
    balance = (s["energy_kwh.offtake_kwh"] - s["energy_kwh.injection_kwh"]
               - s["energy_kwh.consumption_kwh"] + s["energy_kwh.pv_kwh"]).abs()
    if (balance > 1e-3).any():
        problems.append(f"energy balance off by up to {balance.max():.3g} kWh")
    if "kpis.discomfort" in s:
        high = s["kpis.discomfort"].quantile(0.95)
        problems.append(f"(info) discomfort 95th percentile {high:.2f} Kh, max "
                        f"{s['kpis.discomfort'].max():.2f} Kh (house {s['kpis.discomfort'].idxmax()})")
    return problems


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--mix", type=Path, default=None, help="ENTSO-E generation CSV for the run year")
    p.add_argument("--factor-set", default="direct", choices=["direct", "lifecycle"])
    p.add_argument("--tot-houses", type=int, default=500)
    args = p.parse_args(argv)

    summary = load_run_summary(args.run_dir)
    if summary.empty:
        print(f"no finished houses in {args.run_dir}")
        return 1
    inv = house_inventory(args.tot_houses)
    errors = sorted(args.run_dir.glob("house_*.error.txt"))
    print(f"{len(summary)} houses finished, {len(errors)} with an error file")

    if args.mix is not None:
        from price_signals.carbon_intensity import compute_intensity
        from price_signals.load_generation_mix import load_generation_mix

        intensity = compute_intensity(load_generation_mix(args.mix), factor_set=args.factor_set)
        rows = {}
        for house in summary.index:
            table, meta = load_house(args.run_dir, house)
            rows[house] = co2_from_grid_kw(table["grid_kw"], meta["time_step_h"], intensity)
        co2 = pd.DataFrame(rows).T.add_prefix(f"co2_{args.factor_set}.")
        summary = summary.join(co2.drop(columns=[c for c in co2 if c.endswith("_kwh")]))

    summary = summary.join(inv[["stratum", "region", "has_heating", "has_ev", "has_pv"]])
    summary.to_csv(args.run_dir / "summary.csv")

    sums = [c for c in summary if c.startswith(("energy_kwh.", "kpis.opex", "kpis.day_ahead"))
            or c.endswith(("co2_kg", "co2_net_kg", "injection_credit_kg"))]
    strata = summary.groupby("stratum")[sums].agg(["count", "sum", "mean"])
    strata.to_csv(args.run_dir / "strata.csv")

    pd.set_option("display.width", 200)
    print("\nfleet totals (simulated houses):")
    print(summary[sums].sum().to_string(float_format=lambda v: f"{v:,.1f}"))
    print("\nmean per house by stratum:")
    print(summary.groupby("stratum")[sums].mean().round(1).T.to_string())
    if "elapsed_s" in summary:
        print(f"\nwall time per house: mean {summary['elapsed_s'].mean() / 60:.1f} min, "
              f"max {summary['elapsed_s'].max() / 60:.1f} min "
              f"(house {summary['elapsed_s'].idxmax()})")
    for line in sanity_checks(summary, inv):
        print("CHECK:", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Write the house inventory, strata and the stratified training / validation groups.

Reads only the JSON configs (no simulation, seconds):

    python -m experiments.fleet_groups --n-train 20 --n-val 20 --seed 0

Output (results/fleet/):
    inventory.csv                       one row per house: assets, sizes, region, stratum
    groups_train{T}_val{V}_seed{S}.csv  house -> train / val / rest
The group size is a Phase 5 decision (compute budget, PLAN.md Phase 1 Output); rerun with
other sizes as needed, the seed makes it reproducible.
"""

import argparse

from fleet_harness.fleet import FLEET_DIR, STRATUM_KEYS, house_inventory, select_groups


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tot-houses", type=int, default=500)
    args = p.parse_args(argv)

    inv = house_inventory(args.tot_houses)
    FLEET_DIR.mkdir(parents=True, exist_ok=True)
    inv.to_csv(FLEET_DIR / "inventory.csv")

    print("strata (has_heating, has_ev, has_pv):")
    print(inv["stratum"].value_counts().to_string())
    print("\nmarginals:", {k: int(inv[k].sum()) for k in STRATUM_KEYS})

    group = select_groups(inv, args.n_train, args.n_val, args.seed)
    path = FLEET_DIR / f"groups_train{args.n_train}_val{args.n_val}_seed{args.seed}.csv"
    inv.join(group)[["stratum", "group"]].to_csv(path)
    print(f"\ngroups -> {path}")
    print(inv.join(group).groupby(["stratum", "group"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

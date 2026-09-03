from belgian_dwellings.simulation.get_results import get_consumption

from fleet_harness.runner import run_one_house, run_one_house_with_price


def print_kpis(label, microgrid):
    k = microgrid.tot_reward.KPIs
    c = get_consumption(microgrid)
    print(
        f"{label}: opex={k['opex']:.4f}  discomfort={k['discomfort']:.4f}  "
        f"consumption={c['consumption_only']:.4f}  "
        f"grid_offtake={c['grid_offtake']:.4f}  grid_injection={c['grid_injection']:.4f}"
    )


# Rung 0: reproduce the baseline, no signal injection.
baseline = run_one_house(0, disable_enforced=False)
print_kpis("rung0 (baseline day-ahead price)", baseline)


# Rung 1: same house, same tree, same simulated year — but the price signal the tree reads
# each step is flattened to its own mean. If TreeC's charging/heating timing only reacts to
# price *variation* (not price *level*), this should wash out the load-shifting behavior.
def constant_price(series):
    flat = series.copy()
    flat[:] = series.mean()
    return flat


constant = run_one_house_with_price(0, constant_price, disable_enforced=False)
print_kpis("rung1 (constant price)", constant)

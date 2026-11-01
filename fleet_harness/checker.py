"""
Rung comparison for one TreeC house: rung 0 (baseline day-ahead price) vs
rung 1 (constant price), with KPIs and CO2 appended to
results/fleet_harness/rung_metrics.csv.

Run from the repo root:

    python -m fleet_harness.checker

Importing this module has no side effects: nothing is simulated, loaded or
written until main() runs.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path

from belgian_dwellings.simulation.get_results import get_consumption

from fleet_harness.runner import run_one_house, run_one_house_with_price
from fleet_harness.co2 import compute_rung_co2
from price_signals.load_generation_mix import load_generation_mix

REPO_ROOT = Path(__file__).resolve().parent.parent

# Persisted history of every checker run (stdout alone doesn't survive past the
# terminal scrollback). results/ is gitignored, same as data/.
RESULTS_DIR = REPO_ROOT / "results" / "fleet_harness"
METRICS_CSV = RESULTS_DIR / "rung_metrics.csv"

# Real ENTSO-E pull (scripts/download_entsoe_generation.py). load_generation_mix
# handles the two-row header and keeps generation-only columns.
GENERATION_MIX_CSV = REPO_ROOT / "data" / "entsoe_be_2023_generation.csv"


def _log_metrics(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not METRICS_CSV.exists()
    with open(METRICS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_kpis(label, microgrid, generation_mix, run_id):
    """Print one rung's KPIs and append them to METRICS_CSV.

    run_id is shared by every rung of one checker run, so rows from the same
    run can be grouped back together later.
    """
    k = microgrid.tot_reward.KPIs
    c = get_consumption(microgrid)
    co2 = compute_rung_co2(microgrid, generation_mix, factor_set="direct")
    row = {
        "run_id": run_id,
        "rung": label,
        "opex": k["opex"],
        "discomfort": k["discomfort"],
        "consumption_kwh": c["consumption_only"],
        "grid_offtake_kwh": c["grid_offtake"],
        "grid_injection_kwh": c["grid_injection"],
        "co2_kg": co2["co2_kg"],
        "avg_intensity_paid_gCO2_per_kWh": co2["avg_intensity_paid_gCO2_per_kWh"],
        "co2_factor_set": co2["factor_set"],
        "generation_mix_source": GENERATION_MIX_CSV.name,  # real ENTSO-E pull
    }
    _log_metrics(row)
    print(
        f"{label}: opex={row['opex']:.4f}  discomfort={row['discomfort']:.4f}  "
        f"consumption={row['consumption_kwh']:.4f}  "
        f"grid_offtake={row['grid_offtake_kwh']:.4f}  grid_injection={row['grid_injection_kwh']:.4f}  "
        f"co2_kg={row['co2_kg']:.3f}  "
        f"avg_intensity_paid_gCO2_per_kWh={row['avg_intensity_paid_gCO2_per_kWh']:.1f}"
    )
    print(f"  -> logged to {METRICS_CSV.relative_to(REPO_ROOT)}")


def constant_price(series):
    """Rung 1 price transform: flatten the price to its own mean.

    If TreeC's charging/heating timing only reacts to price *variation* (not
    price *level*), this should wash out the load-shifting behaviour.
    """
    flat = series.copy()
    flat[:] = series.mean()
    return flat


def main(house_num: int = 0) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generation_mix = load_generation_mix(GENERATION_MIX_CSV)

    # Rung 0: reproduce the baseline, no signal injection.
    baseline = run_one_house(house_num, disable_enforced=False)
    print_kpis("rung0 (baseline day-ahead price)", baseline, generation_mix, run_id)

    # Rung 1: same house, same tree, same simulated year, but the price the tree
    # reads each step is flattened to its own mean.
    constant = run_one_house_with_price(house_num, constant_price, disable_enforced=False)
    print_kpis("rung1 (constant price)", constant, generation_mix, run_id)


if __name__ == "__main__":
    main()

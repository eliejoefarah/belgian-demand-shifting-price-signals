import csv
from datetime import datetime, timezone
from pathlib import Path

from belgian_dwellings.simulation.get_results import get_consumption

from fleet_harness.runner import run_one_house, run_one_house_with_price
from fleet_harness.co2 import compute_rung_co2
from price_signals.load_generation_mix import load_generation_mix

REPO_ROOT = Path(__file__).resolve().parent.parent

# Persisted history of every checker.py run -- previously print_kpis only printed
# to stdout, so nothing survived past the terminal scrollback. results/ mirrors
# the existing results_rbc_test/results_treec_test convention and is already
# gitignored (see .gitignore), same as data/.
RESULTS_DIR = REPO_ROOT / "results" / "fleet_harness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_CSV = RESULTS_DIR / "rung_metrics.csv"
# Shared across every rung printed in this process invocation, so rows from the
# same checker.py run can be grouped back together later.
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _log_metrics(row: dict) -> None:
    write_header = not METRICS_CSV.exists()
    with open(METRICS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# Real ENTSO-E pull (scripts/download_entsoe_generation.py) -- load_generation_mix
# handles the two-row header (production_type x Actual Aggregated/Consumption)
# entsoe-py writes and keeps generation-only columns; see its docstring.
GENERATION_MIX = load_generation_mix(REPO_ROOT / "data" / "entsoe_be_2023_generation.csv")


def print_kpis(label, microgrid):
    k = microgrid.tot_reward.KPIs
    c = get_consumption(microgrid)
    co2 = compute_rung_co2(microgrid, GENERATION_MIX, factor_set="direct")
    row = {
        "run_id": RUN_ID,
        "rung": label,
        "opex": k["opex"],
        "discomfort": k["discomfort"],
        "consumption_kwh": c["consumption_only"],
        "grid_offtake_kwh": c["grid_offtake"],
        "grid_injection_kwh": c["grid_injection"],
        "co2_kg": co2["co2_kg"],
        "avg_intensity_paid_gCO2_per_kWh": co2["avg_intensity_paid_gCO2_per_kWh"],
        "co2_factor_set": co2["factor_set"],
        "generation_mix_source": "entsoe_be_2023_generation.csv",  # real ENTSO-E pull
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

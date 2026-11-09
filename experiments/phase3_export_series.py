"""Phase 3 outputs: hourly grid-signal series for 2022-2023 + comparison and sensitivity runs.

Run from the repo root:  python -m experiments.phase3_export_series

Writes, per year, results/phase3/grid_signals_{year}_hourly.csv (UTC index, 8760 rows):
- merit-order model (main run, measured unit rates, price_signals.dispatch_model):
  demand_mw (energy left for the 10 CCGTs), available_mw, n_available, marginal_unit,
  mef_g_per_kwh (MAIN), headroom_mw, next_unit, next_mef_g_per_kwh, shortage,
  co2_merit_model_t / co2_merit_observed_t (t CO2 in the hour, model vs observed dispatch,
  same measured rates) = the stack emissions E_t (PLAN.md).
- comparison runs (PLAN.md Phase 3 decisions; temp_phase3_log.md 5.8, 5.30):
  mef_diw (DIW DD72 Table 21 efficiencies), mef_diw_eta_m3 / mef_diw_eta_p3 (eta -/+ 3
  points), mef_pglib (PGLib-CO2 CCGT class, 362.1 for every unit -> flat).
- small-gas sensitivity (temp_phase3_log.md 5.21 item 2, beta from 5.31):
  MEF_beta = (1 - beta) * MEF_main + beta * EF_R, beta in {0.03, 0.10} (share of a load
  change met by small/decentralised gas, hourly-to-daily range), EF_R in {362.1, 517.3}
  g/kWh (PGLib CCGT and generic-gas classes, the bounds set in 5.21). Columns
  mef_beta03_362, mef_beta03_517, mef_beta10_362, mef_beta10_517.
- average-factor signals from the ENTSO-E generation mix (price_signals.carbon_intensity,
  production-based): gas_share (0-1), aef_direct_g_per_kwh, aef_lifecycle_g_per_kwh.
Hourly because the generation data are hourly; dispatch_model.to_quarter_hours repeats a
value over its four quarter-hours when the 15-min MPC needs it.

Also results/phase3/grid_signals_summary.csv: per year and MEF variant, mean / std / p5 /
p50 / p95, Spearman correlation with the main MEF, and the variant's annual merit CO2.
Results and interpretation: temp_phase3_log.md 5.51.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from price_signals.carbon_intensity import compute_intensity, gas_share
from price_signals.dispatch_model import run_year
from price_signals.emission_factors import PGLIB_CO2_FACTORS
from price_signals.load_generation_mix import load_generation_mix

OUT = Path("results/phase3")
BETAS = (0.03, 0.10)
EF_SMALL = (PGLIB_CO2_FACTORS["CCGT"], PGLIB_CO2_FACTORS["NG"])   # 362.1, 517.3 g/kWh


def co2_t(res, dispatch: pd.DataFrame) -> pd.Series:
    rate = res.units.set_index("unit")["ef_g_per_kwh"]
    return (dispatch * rate).sum(axis=1) / 1000.0     # MWh x kg/MWh -> kg -> t


def export_year(year: int) -> tuple[pd.DataFrame, list[dict]]:
    main = run_year(year, method="measured")
    df = main.hourly.copy()
    df["co2_merit_model_t"] = co2_t(main, main.dispatch)
    df["co2_merit_observed_t"] = co2_t(main, main.observed)

    runs = {"mef_g_per_kwh": main}
    runs["mef_diw"] = run_year(year, method="diw")
    runs["mef_diw_eta_m3"] = run_year(year, method="diw", eta_shift=-0.03)
    runs["mef_diw_eta_p3"] = run_year(year, method="diw", eta_shift=+0.03)
    runs["mef_pglib"] = run_year(year, method="pglib")
    for col, res in runs.items():
        if col != "mef_g_per_kwh":
            df[col] = res.hourly["mef_g_per_kwh"]
    for b in BETAS:
        for ef in EF_SMALL:
            df[f"mef_beta{round(b * 100):02d}_{ef:.0f}"] = (1 - b) * df["mef_g_per_kwh"] + b * ef

    mix = load_generation_mix(f"data/entsoe_be_{year}_generation.csv")
    mix.index = mix.index.tz_convert("UTC")
    df["gas_share"] = gas_share(mix).reindex(df.index)
    df["aef_direct_g_per_kwh"] = compute_intensity(mix, "direct").reindex(df.index)
    df["aef_lifecycle_g_per_kwh"] = compute_intensity(mix, "lifecycle").reindex(df.index)

    summary = []
    for col in [c for c in df.columns if c.startswith("mef_")]:
        m = df[col]
        res = runs.get(col)
        summary.append({
            "year": year, "variant": col, "mean": m.mean(), "std": m.std(),
            "p5": m.quantile(0.05), "p50": m.median(), "p95": m.quantile(0.95),
            "spearman_vs_main": m.corr(df["mef_g_per_kwh"], method="spearman"),
            "merit_co2_model_kt": co2_t(res, res.dispatch).sum() / 1000 if res is not None else np.nan,
        })
    for col in ("gas_share", "aef_direct_g_per_kwh", "aef_lifecycle_g_per_kwh"):
        m = df[col]
        summary.append({"year": year, "variant": col, "mean": m.mean(), "std": m.std(),
                        "p5": m.quantile(0.05), "p50": m.median(), "p95": m.quantile(0.95),
                        "spearman_vs_main": m.corr(df["mef_g_per_kwh"], method="spearman")})
    return df, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in (2022, 2023):
        df, summary = export_year(year)
        if df[["mef_g_per_kwh", "gas_share", "aef_direct_g_per_kwh"]].isna().any().any():
            raise ValueError(f"{year}: NaN in the main signal columns")
        df.to_csv(OUT / f"grid_signals_{year}_hourly.csv", index_label="datetime_utc")
        rows += summary
        print(f"saved grid_signals_{year}_hourly.csv ({len(df)} rows, {df.shape[1]} columns)")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "grid_signals_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

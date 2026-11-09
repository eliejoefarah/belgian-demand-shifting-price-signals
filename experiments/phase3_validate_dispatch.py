"""Phase 3: validate the dispatch-model MEF by replicating Deetjen & Azevedo (2019).

Run from the repo root:  python -m experiments.phase3_validate_dispatch

For 2022 and 2023 (main run, measured unit rates):
- Deetjen's three CO2 errors (price_signals.mef_validation), for (A) all points as in
  their calcError and (C) excluding hours with no merit unit running (d < 100 MW, where
  tied demands make their slope blow up) and the top decile;
- a falsification test: the same validation for two deliberately wrong dispatches
  (merit order reversed; every unit loaded pro rata to its available capacity);
- the tables -> results/phase3/deetjen_validation_errors.csv.
The Fig. 2-style figure is drawn by visualization_scripts.phase3_deetjen_validation.
Results and interpretation: temp_phase3_log.md 5.49.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from price_signals.dispatch_model import merit_order_dispatch, run_year
from price_signals.mef_validation import deetjen_validation

OUT = Path("results/phase3")


def errors(v: dict, label: str, year: int, dispatch: str) -> list[dict]:
    h, s = v["hist"], v["sim"]
    n = len(h)
    pos = np.arange(n)
    rows = []
    for variant, mask in (("A all points", np.ones(n, bool)),
                          ("C d>=100 MW, <=p90", (h["demand"] >= 100).to_numpy() & (pos < 0.9 * n))):
        ok = mask & h["co2_slope"].notna().to_numpy() & s["co2_slope"].notna().to_numpy()
        hs, ss, rg = h["co2_slope"][ok], s["co2_slope"][ok], h["regression_mef"][ok]
        tot = mask & h["co2_tot_rolling"].notna().to_numpy()
        rows.append({
            "year": year, "dispatch": dispatch, "variant": variant, "n_points": int(ok.sum()),
            "err_total": float(np.mean(np.abs(h["co2_tot"][tot] - s["co2_tot_rolling"][tot])) / h["co2_tot"].mean()),
            "err_mef_sim": float(np.mean(np.abs(hs - ss)) / hs.mean()),
            "err_mef_regression": float(np.mean(np.abs(hs - rg)) / hs.mean()),
            "mean_mef_hist_g": hs.mean() * 1000, "mean_mef_sim_g": ss.mean() * 1000,
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in (2022, 2023):
        res = run_year(year)
        rate = res.units.set_index("unit")["ef_g_per_kwh"]
        d = res.hourly["demand_mw"]
        av = res.available
        r = rate[av.columns].to_numpy()
        hist = (res.observed[av.columns] * r).sum(axis=1) / 1000          # t CO2 per hour
        reversed_ = merit_order_dispatch(d.to_numpy(), av.to_numpy(), -r)["dispatch"]
        prorata = av.mul((d / av.sum(axis=1).replace(0, np.nan)).clip(upper=1), axis=0).fillna(0).to_numpy()
        for name, disp in (("merit order (model)", res.dispatch[av.columns].to_numpy()),
                           ("reversed merit order", reversed_),
                           ("pro rata to availability", prorata)):
            sim = pd.Series((disp * r).sum(axis=1) / 1000, index=d.index)
            v = deetjen_validation(d, hist, sim)
            rows += errors(v, name, year, name)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "deetjen_validation_errors.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

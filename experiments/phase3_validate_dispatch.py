"""Phase 3: validate the dispatch-model MEF by replicating Deetjen & Azevedo (2019).

Run from the repo root:  python -m experiments.phase3_validate_dispatch

For 2022 and 2023 (main run, measured unit rates):
- Deetjen's three CO2 errors (price_signals.mef_validation), for (A) all points as in
  their calcError and (C) excluding hours with no merit unit running (d < 100 MW, where
  tied demands make their slope blow up) and the top decile;
- a falsification test: the same validation for two deliberately wrong dispatches
  (merit order reversed; every unit loaded pro rata to its available capacity);
- the Fig. 2-style figure -> results/phase3/deetjen_validation_2022_2023.png;
- the tables -> results/phase3/deetjen_validation_errors.csv.
Results and interpretation: temp_phase3_log.md 5.49.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from price_signals.dispatch_model import merit_order_dispatch, run_year
from price_signals.mef_validation import deetjen_validation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results/phase3")
SIM, HIST, REG = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (validated)
INK, MUTED, GRID, SURF = "#0b0b0b", "#5f5e5a", "#e6e5e0", "#fcfcfb"


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
    rows, curves = [], {}
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
            if name == "merit order (model)":
                curves[year] = v
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "deetjen_validation_errors.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string(index=False))
    plot(curves)


def plot(curves: dict):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.facecolor": SURF, "figure.facecolor": SURF})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, (year, v) in enumerate(curves.items()):
        h, s = v["hist"], v["sim"]
        n = len(h)
        pos = np.arange(n)
        lo, hi = h["demand"].quantile(0.05), h["demand"].quantile(0.95)
        ax = axes[0, col]
        m = h["co2_tot_rolling"].notna()
        ax.scatter(h["demand"], h["co2_tot"], s=2, color=HIST, alpha=0.15, lw=0, label="historical, hourly")
        ax.plot(h["demand"][m], h["co2_tot_rolling"][m], color=HIST, ls="--", lw=2, label="historical, rolling")
        ax.plot(s["demand"][m], s["co2_tot_rolling"][m], color=SIM, lw=2, label="simulated, rolling")
        ax.set_xlim(lo, hi)
        ax.set_title(f"{year}: total CO$_2$ of the 10 CCGTs", color=INK, loc="left")
        ax.set_xlabel("Net demand on the merit units d [MW]")
        ax.set_ylabel("t CO$_2$ per hour")
        ax = axes[1, col]
        keep = (h["demand"] >= 100) & (pos < 0.9 * n)
        ax.plot(h["demand"][keep], h["co2_slope"][keep] * 1000, color=HIST, ls="--", lw=2, label="historical (slope of curve)")
        ax.plot(s["demand"][keep], s["co2_slope"][keep] * 1000, color=SIM, lw=2, label="simulated (slope of curve)")
        ax.plot(h["demand"][keep], h["regression_mef"][keep] * 1000, color=REG, ls=":", lw=2, label="regression by demand decile")
        ax.axhspan(355, 430, color=GRID, zorder=0, label="range of unit rates (355–430)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(250, 650)
        ax.set_title(f"{year}: marginal emission factor vs demand", color=INK, loc="left")
        ax.set_xlabel("Net demand on the merit units d [MW]")
        ax.set_ylabel("MEF [g CO$_2$/kWh]")
        for a in axes[:, col]:
            a.grid(color=GRID, lw=0.6)
            a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[1, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Validation of the Belgian dispatch model, replicating Deetjen & Azevedo (2019) Fig. 2 / Eqs. 2–3",
                 color=INK, x=0.01, ha="left")
    fig.savefig(OUT / "deetjen_validation_2022_2023.png", dpi=200)


if __name__ == "__main__":
    main()

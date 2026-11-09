"""Dispatch-model validation figure in the layout of Deetjen & Azevedo (2019), Fig. 2.

Top: total CO2 of the 10 CCGTs vs net demand d (historical hourly points, historical and
simulated rolling curves). Bottom: MEF vs d = slope of those curves (their Eq. 2, exact
procedure in price_signals.mef_validation), plus our regression-by-decile analogue; hours
with d < 100 MW (no CCGT running) and the top decile are left out of the MEF panel, where
tied/sparse demands make the method's slope blow up (temp_phase3_log.md 5.49).
Numbers (errors, falsification test): experiments.phase3_validate_dispatch.

Run from the repo root:  python -m visualization_scripts.phase3_deetjen_validation
Output: results/phase3/deetjen_validation_2022_2023.png
"""
from pathlib import Path

import numpy as np

from price_signals.dispatch_model import run_year
from price_signals.mef_validation import deetjen_validation
from visualization_scripts.style import GRID, INK, SERIES, apply_style, plt, tidy

OUT = Path("results/phase3")
SIM, HIST, REG = SERIES[0], SERIES[1], SERIES[2]


def curves(year: int) -> dict:
    res = run_year(year)
    rate = res.units.set_index("unit")["ef_g_per_kwh"]
    d = res.hourly["demand_mw"]
    hist = (res.observed * rate).sum(axis=1) / 1000
    sim = (res.dispatch * rate).sum(axis=1) / 1000
    return deetjen_validation(d, hist, sim)


def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, year in enumerate((2022, 2023)):
        v = curves(year)
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
            tidy(a)
    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[1, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Validation of the Belgian dispatch model, replicating Deetjen & Azevedo (2019) Fig. 2 / Eqs. 2–3",
                 color=INK, x=0.01, ha="left")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "deetjen_validation_2022_2023.png", dpi=200)
    print(f"saved {OUT / 'deetjen_validation_2022_2023.png'}")


if __name__ == "__main__":
    main()

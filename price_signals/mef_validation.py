"""
Validation of the dispatch-model MEF, replicating Deetjen & Azevedo (2019) exactly.

Deetjen, T. A., & Azevedo, I. L. (2019). Reduced-Order Dispatch Model for Simulating
Marginal Emissions Factors for the United States Power Sector. Environ. Sci. Technol.
53(17), 10506-10513, doi:10.1021/acs.est.9b02500 (THESIS/literature/6.emission_factors/
marginal_intensity/). Procedure taken from their code, github.com/tdeetjen/simple_dispatch,
`mefs_from_simple_dispatch.py`, class `plotDispatch`, methods `rollingCalculations(p=438)`
and `calcError` (read 9 Nov 2026), which is what produced their published numbers:

1. Sort the hours by net demand d (validation scenario: d = the dispatchable fleet's
   observed generation, the same for the historical and the simulated data; ours is
   `DispatchResult.hourly.demand_mw`).
2. Total-emissions curve: centered rolling mean of hourly total CO2 over 2p sorted hours.
3. Slope at sorted position i: (E_roll[i + p/2] - E_roll[i - p/2]) / (d[i + p/2] - d[i - p/2]),
   then a centered rolling mean over p positions. This is the MEF as a function of d.
   NOTE: the paper's Eq. 2 says "438 data points above d to 438 data points below d";
   the code uses p/2 = 219 on each side. We follow the code.
4. Errors (their Eq. 3 = mean absolute error / historical mean, on the demand-sorted,
   position-aligned series, rows where the rolling values exist):
   - total:   |E_hist(hourly) - E_sim(rolling)| / mean(E_hist)
   - slope:   |slope_hist - slope_sim| / mean(slope_hist)
   - regression: |slope_hist - MEF_regression(d)| / mean(slope_hist)
   Their regression MEFs are the CEDM ones (Azevedo et al.), given per demand decile and
   interpolated linearly in d (`interp1d`, extended flat beyond the end points). Our
   analogue (no published Belgian regression MEFs exist, temp_phase3_log.md 5.20):
   per decile of d, the OLS slope of hourly dCO2_hist on hourly dd (Siler-Evans 2012
   style), placed at the decile's median d.

Property of their method (kept, since we replicate it): the slope compares the SMOOTHED
emissions curve with the RAW demand at i +- p/2, so even an exactly linear system gives
slopes scattered by a few % around the true value (test: true 0.40 t/MWh, mean 0.4007,
range 0.381-0.419 for uniform random demand). Errors below ~2-3 % are within that noise.

Differences from their setting, to state: our "historical" hourly CO2 is observed MW per
unit x that unit's measured ANNUAL rate (EU ETS / ENTSO-E), not hourly measured stack
emissions (CEMS), so part-load and start-up effects within a year are not in it; and d
has ties (hours with no merit unit running, d = 0), where d[i+p/2] - d[i-p/2] = 0 gives
NaN slopes (their code's behaviour, kept), so the lowest-demand region drops out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P_DEFAULT = 438  # Deetjen's p (5 % of 8760 h)


def _shift_lists(x: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Their list construction: low[i] = x[i - p/2], high[i] = x[i + p/2], 0 at the ends."""
    h = p // 2
    n = len(x)
    low = np.concatenate([np.zeros(h), x[: n - p], np.zeros(h)])
    high = np.concatenate([np.zeros(h), x[p:], np.zeros(h)])
    return low, high


def rolling_curves(demand: np.ndarray, co2: np.ndarray, p: int = P_DEFAULT) -> pd.DataFrame:
    """Demand-sorted table with co2_tot, co2_tot_rolling and co2_slope (t/MWh), as in
    `rollingCalculations`. Sorting is stable (mergesort) so tied demands keep one order."""
    order = np.argsort(np.asarray(demand), kind="mergesort")
    d = np.asarray(demand, dtype=float)[order]
    e = np.asarray(co2, dtype=float)[order]
    df = pd.DataFrame({"pos": order, "demand": d, "co2_tot": e})
    dlow, dhigh = _shift_lists(d, p)
    df["dd"] = dhigh - dlow
    df["co2_tot_rolling"] = df["co2_tot"].rolling(window=2 * p, min_periods=2 * p, center=True).mean()
    elow, ehigh = _shift_lists(df["co2_tot_rolling"].to_numpy(), p)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope0 = np.divide(ehigh - elow, df["dd"].to_numpy())
    slope0[~np.isfinite(slope0)] = np.nan
    df["co2_slope_0"] = slope0
    df["co2_slope"] = df["co2_slope_0"].rolling(window=p, min_periods=p, center=True).mean()
    return df


def regression_mef_by_decile(demand: pd.Series, co2: pd.Series) -> pd.DataFrame:
    """Our CEDM analogue: per decile of demand, OLS slope (with intercept) of hourly
    first differences dCO2 on dd. Returns decile, median demand, mef (t/MWh), n."""
    dd, de = demand.diff(), co2.diff()
    ok = dd.notna() & de.notna()
    dec = pd.qcut(demand[ok], 10, labels=False, duplicates="drop")
    rows = []
    for k, idx in dd[ok].groupby(dec).groups.items():
        x, y = dd[idx].to_numpy(), de[idx].to_numpy()
        slope = np.polyfit(x, y, 1)[0] if np.ptp(x) > 0 else np.nan
        rows.append({"decile": int(k), "demand_median": float(demand[idx].median()),
                     "mef_t_per_mwh": slope, "n": len(idx)})
    return pd.DataFrame(rows)


def interp_regression(reg: pd.DataFrame, demand: np.ndarray) -> np.ndarray:
    """Linear in demand between decile points, flat beyond the ends (their interp1d with
    the first/last value repeated at 0 and 1e6 MW)."""
    x = np.r_[0.0, reg["demand_median"].to_numpy(), 1e6]
    y = np.r_[reg["mef_t_per_mwh"].iloc[0], reg["mef_t_per_mwh"].to_numpy(), reg["mef_t_per_mwh"].iloc[-1]]
    return np.interp(demand, x, y)


def deetjen_validation(demand: pd.Series, co2_hist: pd.Series, co2_sim: pd.Series,
                       p: int = P_DEFAULT) -> dict:
    """Their three CO2 errors plus the curves (t CO2 per hour, MEF in t/MWh)."""
    hist = rolling_curves(demand.to_numpy(), co2_hist.to_numpy(), p)
    sim = rolling_curves(demand.to_numpy(), co2_sim.to_numpy(), p)
    reg = regression_mef_by_decile(demand, co2_hist)
    hist["regression_mef"] = interp_regression(reg, hist["demand"].to_numpy())

    h_tot = hist.loc[hist["co2_tot_rolling"].notna(), "co2_tot"].to_numpy()
    s_tot = sim.loc[sim["co2_tot_rolling"].notna(), "co2_tot_rolling"].to_numpy()
    hs = hist.loc[hist["co2_slope"].notna(), "co2_slope"]
    ss = sim.loc[sim["co2_slope"].notna(), "co2_slope"]
    n = min(len(hs), len(ss))  # their code pairs the two notna series by position
    hm = hs.mean()
    errors = {
        "co2_tot_hour_vs_rolling": float(np.mean(np.abs(h_tot - s_tot)) / hist["co2_tot"].mean()),
        "co2_slope_sim": float(np.mean(np.abs(hs.to_numpy()[:n] - ss.to_numpy()[:n])) / hm),
        "co2_slope_regression": float(np.mean(np.abs(hs.to_numpy() - hist.loc[hs.index, "regression_mef"].to_numpy())) / hm),
        "n_slope_points_hist": int(len(hs)), "n_slope_points_sim": int(len(ss)),
    }
    return {"errors": errors, "hist": hist, "sim": sim, "regression": reg}

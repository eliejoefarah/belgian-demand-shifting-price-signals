"""
Belgian single-node merit-order dispatch model -> hourly marginal emission factor (MEF).

Option (a) of PLAN.md Phase 3 (temp_phase3_log.md 5.21, 5.26-5.40):
- The 10 dispatchable CCGTs of be_units.csv (role "merit") form the merit order,
  ranked by their emission rate (g/kWh; for gas units this is also the fuel-cost
  order, since every unit burns the same fuel: fuel per MWh_el is proportional to
  g/kWh_el).
- Everything else is held at its observed output and never marginal: nuclear,
  wind, solar, net imports, pumped storage, the fixed industrial units
  (Knippegroen, INESCO, ExxonMobil), small/decentralised gas, other. So the energy
  left for the merit units in hour t is what they produced in reality:
      D_t = sum of the merit units' observed generation
  (= load - all held-fixed sources, up to the ENTSO-E balance residual).
  `extra_load_mw` shifts D_t (e.g. the base-load variant without the 500 houses'
  appliance load, or a fleet's load), and the stack is re-solved.
- Hourly available capacity per unit = capacity minus the MW lost in announced
  outages (ENTSO-E unavailability 15.1.A-D), weighted by the share of the hour each
  outage covers. Capacity = max(ENTSO-E capacity of the year, highest observed hourly
  output of the year): gas turbines exceed their rating in cold air, and Saint-
  Ghislain's 2022 capacity (350 MW) is below its output (385 MW) (5.38).
- Stack: available units in rate order; the marginal unit is the one that would
  serve the next MW of demand (the first unit with headroom left at D_t). Its rate
  is the MEF for a load increase. If D_t exceeds all available merit capacity,
  there is no Belgian merit unit at the margin: `shortage` is True and mef NaN.
- No minimum stable load, start-up costs or ramping (economic dispatch, not unit
  commitment): stated limitation, checked by comparing the dispatch with actuals.

Resolution: hourly (per-unit and per-type generation are hourly in 2022-23);
`to_quarter_hours` repeats each hourly value over its four quarter-hours.

Option (b) (PLAN.md) builds on this: it weights this Belgian MEF with the shares of
imports, pumped storage and small gas, so the per-hour outputs below are kept
complete (marginal unit, headroom, next unit, full dispatch).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from price_signals.units import load_units

DATA = Path(__file__).resolve().parents[1] / "data"


# --------------------------------------------------------------------------- core
def merit_order_dispatch(demand: np.ndarray, avail: np.ndarray, rate: np.ndarray) -> dict:
    """Pure merit-order stack.

    demand: (T,) MW for the merit units. avail: (T, N) available MW per unit.
    rate: (N,) g/kWh per unit (the merit order key; ties broken by column order).

    Returns dict of arrays: dispatch (T, N) MW; marginal (T,) column index of the
    unit serving the next MW (-1 if shortage); headroom (T,) MW left on the
    marginal unit; next_unit (T,) the unit after it (-1 if none); shortage (T,) bool.
    Negative demand is treated as 0 (the cheapest available unit is then marginal).
    """
    demand = np.clip(np.asarray(demand, dtype=float), 0.0, None)
    avail = np.clip(np.asarray(avail, dtype=float), 0.0, None)
    T, N = avail.shape
    order = np.argsort(rate, kind="stable")
    a = avail[:, order]
    cum = np.cumsum(a, axis=1)
    before = cum - a
    disp_sorted = np.clip(demand[:, None] - before, 0.0, a)
    # marginal: first unit (in order) with available MW whose cumulative capacity
    # exceeds demand, i.e. it still has headroom for the next MW
    has_room = (cum > demand[:, None]) & (a > 0)
    any_room = has_room.any(axis=1)
    k = np.where(any_room, has_room.argmax(axis=1), -1)
    headroom = np.where(any_room, cum[np.arange(T), np.maximum(k, 0)] - demand, 0.0)
    # next unit with available capacity after the marginal one
    nxt = np.full(T, -1)
    for t in np.flatnonzero(any_room):
        later = np.flatnonzero(a[t, k[t] + 1:] > 0)
        if later.size:
            nxt[t] = k[t] + 1 + later[0]
    dispatch = np.empty_like(disp_sorted)
    dispatch[:, order] = disp_sorted
    to_orig = lambda idx: np.where(idx >= 0, order[np.maximum(idx, 0)], -1)
    return {"dispatch": dispatch, "marginal": to_orig(k), "headroom": headroom,
            "next_unit": to_orig(nxt), "shortage": ~any_room}


# --------------------------------------------------------------------------- data
def _utc(index) -> pd.DatetimeIndex:
    return pd.to_datetime(index, utc=True)


def load_unit_generation(year: int, units: pd.DataFrame) -> pd.DataFrame:
    """Observed hourly MW per unit (sum of its GT/ST columns), UTC index."""
    g = pd.read_csv(DATA / f"entsoe_be_{year}_generation_per_plant.csv",
                    header=[0, 1, 2], index_col=0)
    g.columns = g.columns.get_level_values(0)
    g.index = _utc(g.index)
    return pd.DataFrame({u.unit: g[u.generation_columns].sum(axis=1)
                         for _, u in units.iterrows()})


def load_outages(year: int) -> pd.DataFrame:
    """ENTSO-E unavailability, cleaned (temp_phase3_log.md 5.38): exact duplicates
    dropped, Cancelled/Withdrawn dropped, latest revision per mrid kept.
    Adds lost_mw = nominal_power - avail_qty (>= 0)."""
    o = pd.read_csv(DATA / f"entsoe_be_{year}_unavailability.csv").drop_duplicates()
    o = o[~o["docstatus"].isin(["Cancelled", "Withdrawn"])]
    o = o[o["revision"] == o.groupby("mrid")["revision"].transform("max")].copy()
    o["start"], o["end"] = _utc(o["start"]), _utc(o["end"])
    o["lost_mw"] = (o["nominal_power"] - o["avail_qty"]).clip(lower=0)
    return o


def lost_capacity(outages: pd.DataFrame, eics: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly MW lost per unit EIC: each outage's lost_mw x the share of the hour
    [t, t+1h) it covers; overlapping outages (e.g. GT and ST of one CCGT) add up."""
    t0 = index.values.astype("datetime64[s]").astype(np.int64)
    t1 = t0 + 3600
    out = pd.DataFrame(0.0, index=index, columns=eics)
    for _, o in outages[outages["production_resource_id"].isin(eics)].iterrows():
        s = o["start"].value // 10**9
        e = o["end"].value // 10**9
        frac = np.clip(np.minimum(t1, e) - np.maximum(t0, s), 0, 3600) / 3600.0
        out[o["production_resource_id"]] += o["lost_mw"] * frac
    return out


def available_capacity(units: pd.DataFrame, gen: pd.DataFrame, outages: pd.DataFrame,
                       year: int) -> pd.DataFrame:
    """Hourly available MW per merit unit (columns = unit names)."""
    cap = np.maximum(units[f"capacity_mw_{year}"].to_numpy(), gen[units["unit"]].max().to_numpy())
    lost = lost_capacity(outages, list(units["eic"]), gen.index)
    avail = np.clip(cap[None, :] - lost.to_numpy(), 0.0, cap[None, :])
    return pd.DataFrame(avail, index=gen.index, columns=list(units["unit"]))


# --------------------------------------------------------------------------- run
@dataclass
class DispatchResult:
    year: int
    method: str
    hourly: pd.DataFrame        # demand, available, marginal unit, mef, headroom, next unit, shortage
    dispatch: pd.DataFrame      # model MW per merit unit
    observed: pd.DataFrame      # observed MW per merit unit
    available: pd.DataFrame     # available MW per merit unit
    units: pd.DataFrame         # merit units with their rates


def run_year(year: int, method: str = "measured", extra_load_mw: pd.Series | None = None,
             eta_shift: float = 0.0) -> DispatchResult:
    """Hourly merit-order dispatch and MEF for 2022 or 2023.

    method: unit rates from price_signals.units ("measured" main run, "diw", "pglib").
    extra_load_mw: optional MW added to the merit demand per hour (UTC index; aligned,
    missing hours count as 0). Negative values remove load.
    """
    units_all = load_units(year, method=method, eta_shift=eta_shift)
    merit = units_all[units_all["role"] == "merit"].reset_index(drop=True)
    gen = load_unit_generation(year, merit)
    avail = available_capacity(merit, gen, load_outages(year), year)
    demand = gen.sum(axis=1)
    if extra_load_mw is not None:
        demand = demand + extra_load_mw.reindex(demand.index).fillna(0.0)
    rate = merit["ef_g_per_kwh"].to_numpy()
    r = merit_order_dispatch(demand.to_numpy(), avail.to_numpy(), rate)
    names = np.array(list(merit["unit"]) + [None], dtype=object)  # index -1 -> None
    hourly = pd.DataFrame({
        "demand_mw": demand.to_numpy(),
        "available_mw": avail.sum(axis=1).to_numpy(),
        "n_available": (avail.to_numpy() > 0).sum(axis=1),
        "marginal_unit": names[r["marginal"]],
        "mef_g_per_kwh": np.where(r["marginal"] >= 0, rate[np.maximum(r["marginal"], 0)], np.nan),
        "headroom_mw": r["headroom"],
        "next_unit": names[r["next_unit"]],
        "next_mef_g_per_kwh": np.where(r["next_unit"] >= 0, rate[np.maximum(r["next_unit"], 0)], np.nan),
        "shortage": r["shortage"],
    }, index=demand.index)
    dispatch = pd.DataFrame(r["dispatch"], index=demand.index, columns=list(merit["unit"]))
    return DispatchResult(year, method, hourly, dispatch, gen, avail, merit)


def to_quarter_hours(hourly: pd.Series | pd.DataFrame):
    """Repeat each hourly value over its four quarter-hours (no interpolation)."""
    idx = pd.date_range(hourly.index[0], hourly.index[-1] + pd.Timedelta("45min"),
                        freq="15min", tz=hourly.index.tz)
    return hourly.reindex(idx, method="ffill")


# --------------------------------------------------------------------------- checks
def validate(res: DispatchResult) -> dict:
    """Model vs reality (temp_phase3_log.md 5.21 item 3, 5.35 item 3).

    - per_unit: annual GWh model vs observed, and hourly MAE.
    - emissions: merit-fleet CO2, model vs observed generation at the same rates.
    - price: Spearman of the hourly MEF (and of the marginal unit's merit rank) with
      the BE day-ahead price; computed within months as well, since gas and CO2
      prices (not in the model) move between months.
    - availability: hours where observed output exceeds modelled availability.
    """
    rate = res.units.set_index("unit")["ef_g_per_kwh"]
    per_unit = pd.DataFrame({
        "rate_g_per_kwh": rate,
        "model_GWh": res.dispatch.sum() / 1000,
        "observed_GWh": res.observed.sum() / 1000,
        "hourly_MAE_MW": (res.dispatch - res.observed).abs().mean(),
        "observed_above_available_h": (res.observed > res.available + 5).sum(),
    }).sort_values("rate_g_per_kwh")
    # hourly MW = MWh; MWh x g/kWh = MWh x kg/MWh = kg; / 1e6 -> kt
    em_model = (res.dispatch * rate).sum().sum() / 1e6
    em_obs = (res.observed * rate).sum().sum() / 1e6
    out = {"per_unit": per_unit,
           "emissions_kt": {"model": em_model, "observed": em_obs},
           "shortage_h": int(res.hourly["shortage"].sum())}
    price_path = DATA / f"entsoe_be_{res.year}_day_ahead_prices.csv"
    if price_path.exists():
        p = pd.read_csv(price_path, index_col=0).iloc[:, 0]
        p.index = _utc(p.index)
        p = p.groupby(p.index.floor("1h")).mean()
        h = res.hourly.join(p.rename("price"), how="inner").dropna(subset=["mef_g_per_kwh"])
        month = h.index.tz_convert("Europe/Brussels").month
        within = [g["mef_g_per_kwh"].corr(g["price"], method="spearman")
                  for _, g in h.groupby(month) if g["mef_g_per_kwh"].nunique() > 1]
        out["price_spearman"] = {
            "all_hours": h["mef_g_per_kwh"].corr(h["price"], method="spearman"),
            "within_month_mean": float(np.nanmean(within)),
            "demand_vs_price": h["demand_mw"].corr(h["price"], method="spearman"),
        }
    return out

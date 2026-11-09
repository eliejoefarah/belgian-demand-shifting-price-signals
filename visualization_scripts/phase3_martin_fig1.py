"""Belgian version of Martin, Powell & Rajagopal (2025), Fig. 1 a-b (faithful layout).

Martin et al. (2025), "Cascading marginal emissions signals for green charging with growing
electric vehicle adoption", Nature Communications 16:10150, doi 10.1038/s41467-025-64979-7,
Fig. 1: (a) the whole fleet in merit order for one week, x = generation capacity [GW],
coloured by technology, generation cost on top and CO2 emissions below; (b) hourly MEF vs
AEF [kg/MWh] over that week. (Panel c, added emissions vs number of EVs, belongs to the
signal evaluation, Phases 5-6.)

Belgian construction (temp_phase3_log.md 5.55-5.57):
(a) x: installed capacity per technology (ENTSO-E 14.1.A, data/entsoe_be_{year}_
    installed_capacity_per_type.csv); the 10 CCGTs individually (ENTSO-E per-unit capacity,
    price_signals.units); "gas held fixed" = Fossil Gas capacity minus the 10 CCGTs
    (industrial CHP incl. Knippegroen/INESCO/ExxonMobil, small units). Pumped storage is
    left out (storage, not generation).
    Order: what our dispatch model holds at observed output (not priced in the model; drawn
    at ~0 cost like Martin's renewables) - solar, wind, hydro, nuclear, biomass & waste,
    gas held fixed, oil & other - then the 10 CCGTs by cost.
    CCGT cost [EUR/MWh] = gas price / eta_i + EUA price x rate_i, with eta_i implied by the
    measured rate (eta = 0.20196 t/MWh_fuel / rate, IPCC factor), so cost_i = rate_i x
    (gas / 0.20196 + EUA); VOM ignored. Gas price: IMF "Global price of Natural gas, EU"
    (FRED PNGASEUUSDM, monthly, matches TTF) converted with FRED EXUSEU; EUA: mean EEX
    primary-auction clearing price in the week (data/prices/).
    Emission bars: CCGTs at their measured rates; other blocks at the direct factor of their
    ENTSO-E type (price_signals.emission_factors.DIRECT_FACTORS: renewables, nuclear,
    biomass, waste 0; oil 850.74; other 575; gas held fixed at the Fossil Gas factor 370).
(b) Hourly MEF (main run) and AEF (direct, whole Belgian mix) for the week, kg/MWh, from
    results/phase3/grid_signals_{year}_hourly.csv.

Run from the repo root:
    python -m visualization_scripts.phase3_martin_fig1                      # 16-22 Jan 2023
    python -m visualization_scripts.phase3_martin_fig1 --start 2022-12-12  # any week
Output: results/phase3/martin_fig1_be_{start}.png
"""
import argparse
from pathlib import Path

import pandas as pd

from price_signals.emission_factors import DIRECT_FACTORS
from price_signals.units import GAS_T_CO2_PER_MWH_FUEL, load_units
from visualization_scripts.style import INK, MUTED, SERIES, apply_style, plt, tidy

OUT = Path("results/phase3")
MMBTU_PER_MWH = 3.412
SHORT = {
    "Amercoeur 1 R TGV": "Amercoeur", "DROGENBOS TGV": "Drogenbos",
    "EDF Luminus Seraing TGV": "Seraing", "HERDERSBRUG STEG": "Herdersbrug",
    "Marcinelle Energie (Carsid)": "Marcinelle", "RINGVAART STEG": "Ringvaart",
    "SAINT-GHISLAIN STEG": "Saint-Ghislain", "T-power Beringen": "T-Power",
    "VILVOORDE TGV": "Vilvoorde", "Zandvliet Power": "Zandvliet",
}
# technology -> (colour slot, ENTSO-E types it aggregates); fixed order, never cycled
TECH = {
    "Solar": (0, ["Solar"]),
    "Wind": (1, ["Wind Onshore", "Wind Offshore"]),
    "Hydro": (2, ["Hydro Run-of-river and pondage"]),
    "Nuclear": (3, ["Nuclear"]),
    "Biomass & waste": (4, ["Biomass", "Waste"]),
    "Gas held fixed": (5, None),            # Fossil Gas minus the 10 CCGTs
    "Oil & other": (6, ["Fossil Oil", "Other"]),
    "Gas CCGT (merit order)": (7, None),
}


def gas_price_eur_per_mwh(month: pd.Timestamp) -> float:
    g = pd.read_csv("data/prices/imf_pngaseu_usd_per_mmbtu_monthly.csv", index_col=0, parse_dates=True).iloc[:, 0]
    fx = pd.read_csv("data/prices/fred_exuseu_usd_per_eur_monthly.csv", index_col=0, parse_dates=True).iloc[:, 0]
    key = pd.Timestamp(month.year, month.month, 1)
    return float(g[key] * MMBTU_PER_MWH / fx[key])


def eua_price(t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    e = pd.read_csv("data/prices/eex_eua_auction_daily_2022_2023.csv", index_col=0, parse_dates=True).iloc[:, 0]
    wk = e[(e.index >= t0.tz_localize(None).normalize()) & (e.index < t1.tz_localize(None).normalize())]
    return float(wk.mean()) if len(wk) else float(e[e.index.month == t0.month].mean())


def blocks(year: int, gas: float, eua: float) -> pd.DataFrame:
    cap = pd.read_csv(f"data/entsoe_be_{year}_installed_capacity_per_type.csv", index_col=0).iloc[0]
    units = load_units(year)
    merit = units[units["role"] == "merit"].sort_values("ef_g_per_kwh")
    rows = []
    for tech, (_, types) in TECH.items():
        if types is None:
            continue
        mw = sum(cap.get(t, 0.0) for t in types)
        ef = sum(cap.get(t, 0.0) * DIRECT_FACTORS[t] for t in types) / mw if mw else 0.0
        rows.append({"label": tech, "tech": tech, "mw": mw, "ef": ef, "cost": None})
        if tech == "Biomass & waste":
            fixed_gas = cap["Fossil Gas"] - merit[f"capacity_mw_{year}"].sum()
            rows.append({"label": "Gas held fixed", "tech": "Gas held fixed", "mw": fixed_gas,
                         "ef": DIRECT_FACTORS["Fossil Gas"], "cost": None})
    for _, u in merit.iterrows():
        rate_t = u["ef_g_per_kwh"] / 1000.0
        rows.append({"label": SHORT[u["unit"]], "tech": "Gas CCGT (merit order)", "mw": u[f"capacity_mw_{year}"],
                     "ef": u["ef_g_per_kwh"], "cost": rate_t * (gas / GAS_T_CO2_PER_MWH_FUEL + eua)})
    df = pd.DataFrame(rows)
    df["x0"] = df["mw"].cumsum().shift(fill_value=0.0) / 1000
    df["w"] = df["mw"] / 1000
    return df


def main(start: str):
    t0 = pd.Timestamp(start, tz="Europe/Brussels")
    t1 = t0 + pd.Timedelta(days=7)
    year = t0.year
    gas, eua = gas_price_eur_per_mwh(t0), eua_price(t0, t1)
    b = blocks(year, gas, eua)

    sig = pd.read_csv(OUT / f"grid_signals_{year}_hourly.csv", index_col=0, parse_dates=True)
    sig.index = sig.index.tz_convert("Europe/Brussels")
    wk = sig.loc[t0:t1 - pd.Timedelta(hours=1)]

    apply_style()
    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax_c, ax_e, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

    ccgt_cost_max = b["cost"].max()
    for _, r in b.iterrows():
        colour = SERIES[TECH[r["tech"]][0]]
        fixed = r["cost"] is None or pd.isna(r["cost"])
        h_cost = ccgt_cost_max * 0.02 if fixed else r["cost"]
        ax_c.bar(r["x0"], h_cost, width=r["w"], align="edge", color=colour, edgecolor="#fcfcfb", linewidth=1)
        ax_e.bar(r["x0"], max(r["ef"], 4), width=r["w"], align="edge", color=colour, edgecolor="#fcfcfb", linewidth=1)
        if fixed and r["w"] > 0.15:   # direct labels (identity never by colour alone)
            ax_c.text(r["x0"] + r["w"] / 2, h_cost + ccgt_cost_max * 0.03, r["label"],
                      rotation=90, ha="center", va="bottom", fontsize=7, color=INK)
            ax_e.text(r["x0"] + r["w"] / 2, max(r["ef"], 4) + 15, r["label"],
                      rotation=90, ha="center", va="bottom", fontsize=7, color=INK)
    x_fixed_end = b.loc[b["cost"].isna(), "mw"].sum() / 1000
    for ax in (ax_c, ax_e):
        ax.axvline(x_fixed_end, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(0, b["x0"].iloc[-1] + b["w"].iloc[-1])
        tidy(ax)
    ax_c.text(x_fixed_end / 2, ccgt_cost_max * 0.55, "held at observed output\n(not priced in the dispatch model)",
              ha="center", va="center", fontsize=7.5, color=MUTED)
    ccgt = b[b["cost"].notna()]
    x_ccgt = (ccgt["x0"].iloc[0] + ccgt["x0"].iloc[-1] + ccgt["w"].iloc[-1]) / 2
    ax_c.text(x_ccgt, ccgt_cost_max * 1.04, "10 CCGTs", ha="center", va="bottom", fontsize=7, color=INK)
    ax_e.text(x_ccgt, ccgt["ef"].max() + 15, "10 CCGTs", ha="center", va="bottom", fontsize=7, color=INK)
    ax_c.set_ylim(0, ccgt_cost_max * 1.3)
    ax_c.set_ylabel("Generation cost [€/MWh]")
    ax_c.set_title(f"a  Belgian generation capacity and merit order, {t0:%d}–{(t1 - pd.Timedelta(days=1)):%d %b %Y}",
                   loc="left", color=INK)
    ax_c.text(0.01, 0.97, f"CCGT cost = gas/η + EUA × rate;  gas {gas:.0f} €/MWh$_{{th}}$ ({t0:%b %Y}), EUA {eua:.0f} €/t (week mean)",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=7, color=MUTED)
    ax_e.set_ylim(0, 1150)
    ax_e.set_ylabel("CO$_2$ emissions [kg/MWh]")
    ax_e.set_xlabel("Generation capacity [GW]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[s]) for s, _ in TECH.values()]
    ax_e.legend(handles, list(TECH), frameon=False, fontsize=6.5, ncol=2, loc="upper left")

    ax_b.plot(wk.index, wk["mef_g_per_kwh"], color=SERIES[0], lw=1.6, label="MEF")
    ax_b.plot(wk.index, wk["aef_direct_g_per_kwh"], color=SERIES[1], lw=1.6, ls="--", label="AEF")
    ax_b.set_ylim(0, 500)
    ax_b.set_xlim(t0, t1)
    ax_b.set_ylabel("CO$_2$ Emissions [kg/MWh]")
    ax_b.set_xlabel("Date")
    ax_b.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b-%d", tz=t0.tz))
    ax_b.set_title("b  Hourly MEF and AEF", loc="left", color=INK)
    ax_b.legend(frameon=False, loc="lower right")
    tidy(ax_b)

    fig.suptitle("Belgian analogue of Martin et al. (2025), Fig. 1a–b", x=0.01, ha="left", color=INK)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"martin_fig1_be_{t0:%Y-%m-%d}.png"
    fig.savefig(path, dpi=200)
    print(f"saved {path} | gas {gas:.1f} EUR/MWh_th, EUA {eua:.1f} EUR/t, CCGT cost {ccgt['cost'].min():.0f}-"
          f"{ccgt['cost'].max():.0f} EUR/MWh, fixed {x_fixed_end:.1f} GW, total {b['mw'].sum() / 1000:.1f} GW")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-16", help="first day of the week (Brussels)")
    main(p.parse_args().start)

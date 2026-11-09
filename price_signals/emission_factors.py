"""
Emission factors (gCO2eq/kWh_el) for Belgian electricity generation, keyed to
ENTSO-E production-type names as entsoe-py returns them (e.g. "Fossil Gas").

Two factor sets:
  DIRECT_FACTORS     combustion CO2 at the stack only; non-combusting = 0.
  LIFECYCLE_FACTORS  full life cycle (construction, fuel chain, operation,
                     decommissioning). Always >= direct (asserted below).

How values are chosen
  1. IPCC AR5 Annex III [1] wherever it covers the technology.
  2. Otherwise the Electricity Maps convention [2]: its ENTSO-E mapping [2c]
     (Waste -> biomass; Lignite/Peat/Oil shale -> coal; Coal-derived gas ->
     gas; Other -> unknown) with its default factors [2a], or its Belgian
     2023 value fitted to EU-ETS verified emissions [2b] where one exists.

Convention-dependent entries (CONTESTED) -- report with a sensitivity:
  Biomass, Waste  biogenic: 0 direct / 230 lifecycle [1][2a][7]. Waste is
                  only partly fossil; upper bound = 91,700 kg/TJ [4] x fossil
                  share x 0.0036 / eta (eta still to be sourced).
  Other           unknown conventional thermal: 575 / 700 [2a]. Upper bound
                  919 (cross-country "other fossil" fallback) [3].
  Knippegroen     the steelworks-gas plant on the ArcelorMittal Gent site
                  (42 % efficiency [5]) is reported by ENTSO-E as Fossil Gas,
                  so it gets the gas factor, as in [2c]. Full attribution of
                  the steel gases would give ~1,560-2,230 g/kWh [4][5].

Belgium 2023 shares (ENTSO-E, 79.8 TWh): nuclear 39 %, gas 24 %, wind 18 %,
solar 9 %, biomass 3 %, waste 2.6 %, other 2.1 %, oil 0.7 %. Gas is the
factor that matters; types with 0 GWh are kept so other years/zones work.

Sources
[1] IPCC (2014). Climate Change 2014: Mitigation of Climate Change, WGIII
    AR5. Annex III, Table A.III.2 (direct + lifecycle medians); Ch. 7
    (oil lifecycle range 510-1170, plausibility check).
[2] Electricity Maps, electricitymaps-contrib, GitHub (cite file + commit):
    [2a] config/defaults.yaml, commit d9a920f (2026-04-28).
    [2b] config/zones/BE.yaml, commit 0115d70 (2026-10-02): oil 850.74 /
         1094.74 (used); gas 382.65 (cross-check of AR5's 370).
    [2c] electricitymap/contrib/parsers/ENTSOE.py, ENTSOE_PARAMETER_GROUPS.
[3] Unnewehr, J.F. et al. (2022). Open-data based carbon emission intensity
    signals for electricity generation in European countries. Cleaner
    Energy Systems 3, 100018. doi:10.1016/j.cles.2022.100018. Also the
    sanity benchmark: BE ~195-219 gCO2/kWh in 2018.
[4] IPCC (2006). Guidelines for National GHG Inventories, Vol. 2: Table 2.2
    (non-biomass MSW 91,700 kg/TJ; biomass CO2 reported as memo item),
    Table 1.4 (blast-furnace gas 260,000; converter gas 182,000 kg/TJ).
[5] ENGIE Belgium, "Knippegroen power plant", corporate.engie.be.
[6] Tranberg, B. et al. (2019). Real-time carbon accounting method for the
    European electricity markets. Energy Strategy Reviews 26, 100367.
    doi:10.1016/j.esr.2019.100367 (method precedent).
[7] Attia, S. (2026). Belgian Electricity Grid GHG Emission Factors
    1950-2050. Tech. report v1.1, Univ. de Liege. ISBN 978-2-930909-30-1
    (Belgian precedent: "biomass and waste" 230 lifecycle).
[8] Cho, Y., Ko, M.-S., Zhu, H. (2025). PGLib-CO2: A Power Grid Library for
    Real-Time Computation and Optimization of Carbon Emissions (title of v4;
    v1 "...for Computing and Optimizing Carbon Emissions"). arXiv:2506.14662
    [eess.SY], v1 17 Jun 2025, v4 18 Mar 2026, CC BY 4.0, no journal yet.
    https://arxiv.org/abs/2506.14662. Code: github.com/jacobcho0103/PGLib-CO2
    (no licence file: cite the values, don't copy the code), lookup table in
    Python/carbon_casefile.py. One factor per fuel class in tCO2/MWh_el (the
    code calls it "GCI"), sourced from IPCC 1996 (Houghton et al.; SAR; Revised
    1996 Guidelines) and [9]; no heat rates, no OCGT/CHP/biomass class, no
    must-run or start-up emissions. Marginal metric: LMCE = e_G^T dP_G/dP_D
    from a DC-OPF; on one uncongested node this is the factor of the marginal
    unit, i.e. the merit-order MEF. Values in PGLIB_CO2_FACTORS below. Caveat:
    the repo README uses keys "GAS"/"BIT", which the lookup does not have
    (silently 0); the valid keys are the ones below. Checked 8 Nov 2026.
[9] Durvasulu, V., Hansen, T.M. (2021). IET Smart Grid 4(4), 429-444 (source
    cited by [8] for its factors).
"""

# ---------------------------------------------------------------------------
# DIRECT / OPERATIONAL -- combustion CO2 only
# ---------------------------------------------------------------------------
DIRECT_FACTORS = {
    # --- fossil ---
    "Fossil Gas":                     370.0,  # B04  [1] 370 [350-490]. BE ETS check 383 [2b].
                                              #      Includes Knippegroen. Sweep this one.
    "Fossil Hard coal":               760.0,  # B05  [1] 760 [670-870].
    "Fossil Brown coal/Lignite":      760.0,  # B02  [2c] -> coal [1].
    "Fossil Oil":                     850.74, # B06  [2b] BE 2023, ETS-fitted. Not in AR5.
    "Fossil Coal-derived gas":        370.0,  # B03  [2c] -> gas [1].
    "Fossil Peat":                    760.0,  # B08  [2c] -> coal [1].
    "Fossil Oil shale":               760.0,  # B07  [2c] -> coal [1].

    # --- convention-dependent (see CONTESTED) ---
    "Biomass":                          0.0,  # B01  [2a]; biogenic-neutral [4].
    "Waste":                            0.0,  # B17  [2c] -> biomass.
    "Other":                          575.0,  # B20  [2a] unknown thermal. Upper bound 919 [3].

    # --- zero direct emissions [1] ---
    "Nuclear":                          0.0,  # B14
    "Solar":                            0.0,  # B16
    "Wind Onshore":                     0.0,  # B19
    "Wind Offshore":                    0.0,  # B18
    "Hydro Run-of-river and pondage":   0.0,  # B11
    "Hydro Water Reservoir":            0.0,  # B12
    "Hydro Pumped Storage":             0.0,  # B10  storage: emissions belong to the charging mix.
    "Geothermal":                       0.0,  # B09
    "Marine":                           0.0,  # B13
    "Other renewable":                  0.0,  # B15
}

# ---------------------------------------------------------------------------
# LIFECYCLE -- IPCC AR5 Annex III lifecycle medians [1] unless noted
# ---------------------------------------------------------------------------
LIFECYCLE_FACTORS = {
    # --- fossil ---
    "Fossil Gas":                     490.0,  # B04  [1] 490 [410-650].
    "Fossil Hard coal":               820.0,  # B05  [1] 820 [740-910].
    "Fossil Brown coal/Lignite":      820.0,  # B02  [2c] -> coal [1].
    "Fossil Oil":                    1094.74, # B06  [2b] ETS-fitted direct + IPCC 2014 upstream.
    "Fossil Coal-derived gas":        490.0,  # B03  [2c] -> gas [1].
    "Fossil Peat":                    820.0,  # B08  [2c] -> coal [1].
    "Fossil Oil shale":               820.0,  # B07  [2c] -> coal [1].

    # --- convention-dependent ---
    "Biomass":                        230.0,  # B01  [1] 230 [130-420].
    "Waste":                          230.0,  # B17  [2c] -> biomass [1]; same in [7].
    "Other":                          700.0,  # B20  [2a] unknown thermal.

    # --- low-carbon ---
    "Nuclear":                         12.0,  # B14  [1] 12 [3.7-110].
    "Solar":                           48.0,  # B16  [1] utility PV 48 [18-180].
    "Wind Onshore":                    11.0,  # B19  [1] 11 [7-56].
    "Wind Offshore":                   12.0,  # B18  [1] 12 [8-35].
    "Hydro Run-of-river and pondage":  24.0,  # B11  [1] 24 [1.0-2200].
    "Hydro Water Reservoir":           24.0,  # B12  [1] 24 [1.0-2200].
    "Hydro Pumped Storage":            24.0,  # B10  [1] hydropower (infrastructure only).
    "Geothermal":                      38.0,  # B09  [1] 38 [6.0-79].
    "Marine":                          17.0,  # B13  [1] ocean 17 [5.6-28].
    "Other renewable":                700.0,  # B15  [2c] -> unknown [2a].
}

# ---------------------------------------------------------------------------
# PGLib-CO2 [8]: emission factor per generator FUEL CLASS (not ENTSO-E types),
# gCO2/kWh_el (= its t/MWh x 1000), for the Phase 3 merit-order dispatch model
# (marginal emission factor = factor of the marginal unit). Values copied from
# Python/carbon_casefile.py, 8 Nov 2026. CO2 only; its CO2e (SAR GWPs CH4 21,
# N2O 310) is at most 0.5 % higher.
# Implied efficiency (our calculation, IPCC gas default 56.1 kgCO2/GJ_fuel):
# CCGT ~56 %, NG ~39 % (generic / peaker-like gas). Belgian gas in the dispatch
# should average ~383 g/kWh (BE ETS fit [2b]), between CCGT and NG.
# Plan (PLAN.md Phase 3): CCGT units -> "CCGT", other gas (OCGT, old steam, CHP
# not run as must-run) -> "NG"; oil uses the BE ETS-fitted 850.74 above, not
# PEL; biomass/waste follow the CONTESTED convention (0 direct).
# ---------------------------------------------------------------------------
PGLIB_CO2_FACTORS = {
    "ANT":     909.5,  # anthracite coal
    "COW":     820.4,  # bituminous coal
    "PEL":     700.1,  # oil (paper table: distillate; code comment: heavy oil)
    "NG":      517.3,  # natural gas, generic (~39 % implied efficiency)
    "CCGT":    362.1,  # natural-gas combined cycle (~56 % implied efficiency)
    "ICE":     603.0,  # internal combustion engine
    "Thermal": 687.4,  # general thermal (in the code, not in the paper)
    "NUC":       0.0,  # nuclear
    "RE":        0.0,  # renewables
    "HYD":       0.0,  # hydro
    "SYNC":      0.0,  # synchronous condenser
}

# Values that are a convention, not a measurement: report with a sensitivity.
CONTESTED = {"Biomass", "Waste", "Other"}

# Values not read directly from IPCC AR5 [1]. Only Waste, Other and Fossil Oil
# have nonzero Belgian generation in 2023.
NON_IPCC = {
    "Fossil Oil", "Fossil Brown coal/Lignite", "Fossil Coal-derived gas",
    "Fossil Peat", "Fossil Oil shale", "Waste", "Other", "Other renewable",
}

FACTOR_SETS = {"direct": DIRECT_FACTORS, "lifecycle": LIFECYCLE_FACTORS}

assert set(DIRECT_FACTORS) == set(LIFECYCLE_FACTORS)
assert all(LIFECYCLE_FACTORS[k] >= DIRECT_FACTORS[k] for k in DIRECT_FACTORS), \
    "lifecycle factor below direct factor"


def check_coverage(mix_columns, factors):
    """Return (uncovered, unused). `uncovered` = generation columns with no
    factor, which would otherwise drop out of the intensity unnoticed."""
    cols, keys = set(mix_columns), set(factors)
    return cols - keys, keys - cols

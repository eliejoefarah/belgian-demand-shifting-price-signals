"""Phase 4 smoke test: Chronos-2 day-ahead forecast of Belgian gas generation, 2022.

Run from the repo root:  python -m experiments.phase4_chronos_test

Target: hourly ENTSO-E "Fossil Gas" generation (2022 file). For 5 delivery days D
(Tuesdays 18 Jan - 15 Feb 2022) the context ends at 14:00 Brussels on D-1 (the MPC
re-plan time) and the forecast covers 14:00 D-1 -> 24:00 D; only D's 24 hours are scored.
Covariates known before 14:00 D-1 (temp_phase4_log.md 1, 8): Elia day-ahead 11AM wind and
solar forecasts and the Elia week-ahead load forecast (2021-2022 files), given as past and
known-future covariates. Compared with Chronos-2 without covariates and a naive "same hour
2 days earlier". Results: temp_phase4_log.md 9. Not the backtest (that is the next step).
"""
import time

import pandas as pd
from chronos import Chronos2Pipeline
from price_signals.load_generation_mix import load_generation_mix

def elia(name, years, col):
    fr = []
    for y in years:
        d = pd.read_csv(f"data/elia_{name}_{y}.csv", encoding="utf-8-sig")
        d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
        fr.append(d.groupby("datetime")[col].sum(min_count=1))
    return pd.concat(fr).sort_index().resample("1h").mean()

mix = pd.concat([load_generation_mix(f"data/entsoe_be_{y}_generation.csv") for y in (2022,)])
mix.index = mix.index.tz_convert("UTC")
gas = mix["Fossil Gas"].rename("target")
cov = pd.DataFrame({
    "wind_11h": elia("wind", (2021, 2022), "dayahead11hforecast"),
    "solar_11h": elia("solar", (2021, 2022), "dayahead11hforecast"),
    "load_week": elia("load", (2021, 2022), "weekaheadforecast"),
})
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
rows = []
for day in pd.date_range("2022-01-18", periods=5, freq="7D", tz="Europe/Brussels"):
    cutoff = (day - pd.Timedelta(hours=10)).tz_convert("UTC")      # 14:00 Brussels on D-1
    end = (day + pd.Timedelta(days=1)).tz_convert("UTC")
    hist = pd.concat([gas, cov], axis=1).sort_index().loc[cutoff - pd.Timedelta(days=60):cutoff - pd.Timedelta(hours=1)].dropna()
    fut_idx = pd.date_range(cutoff, end - pd.Timedelta(hours=1), freq="1h")
    df = hist.reset_index().rename(columns={"index": "timestamp"}); df.columns = ["timestamp"] + list(df.columns[1:]); df["item_id"] = "BE_gas"
    fdf = cov.reindex(fut_idx).reset_index(); fdf.columns = ["timestamp"] + list(cov.columns); fdf["item_id"] = "BE_gas"
    for k in (df, fdf): k["timestamp"] = k["timestamp"].dt.tz_localize(None)
    t = time.time()
    pred = pipe.predict_df(df, future_df=fdf, prediction_length=len(fut_idx), quantile_levels=[0.1, 0.5, 0.9])
    dt = time.time() - t
    pred["timestamp"] = pd.to_datetime(pred["timestamp"]).dt.tz_localize("UTC"); pred = pred.set_index("timestamp")
    p = pred.loc[day.tz_convert("UTC"):end - pd.Timedelta(hours=1)]
    act = gas.reindex(p.index)
    naive = gas.shift(48).reindex(p.index)          # same hour 2 days earlier (known at 14:00 D-1)
    nocov = pipe.predict_df(df[["timestamp","item_id","target"]], prediction_length=len(fut_idx), quantile_levels=[0.5]).set_index("timestamp")
    nocov.index = pd.to_datetime(nocov.index).tz_localize("UTC"); nc = nocov.reindex(p.index)
    rows.append({"day": day.date(), "mean_gas_MW": act.mean(), "MAE_chronos_cov": (p["0.5"] - act).abs().mean(),
                 "MAE_chronos_nocov": (nc["0.5"] - act).abs().mean(), "MAE_naive_2d": (naive - act).abs().mean(),
                 "cover_p10_p90": ((act >= p["0.1"]) & (act <= p["0.9"])).mean(), "seconds": dt})
print(pred.columns.tolist())
r = pd.DataFrame(rows); print(r.round(2).to_string(index=False)); print(r.mean(numeric_only=True).round(2).to_dict())

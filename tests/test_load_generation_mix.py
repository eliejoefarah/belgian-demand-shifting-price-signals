"""load_generation_mix on a tiny CSV in the exact format entsoe-py writes
(copied from data/entsoe_be_2023_generation.csv): a two-row header, empty
cells, and the CET -> CEST clock change on 2023-03-26."""
import pandas as pd

from price_signals.load_generation_mix import load_generation_mix

CSV = """\
,Fossil Gas,Nuclear,Hydro Pumped Storage,Hydro Pumped Storage
,Actual Aggregated,Actual Aggregated,Actual Aggregated,Actual Consumption
2023-03-26 00:00:00+01:00,1000.0,4000.0,,9.5
2023-03-26 01:00:00+01:00,1100.0,4000.0,50.0,
2023-03-26 03:00:00+02:00,1200.0,4000.0,,540.0
2023-03-26 04:00:00+02:00,1300.0,4000.0,,
"""


def _load(tmp_path):
    path = tmp_path / "mix.csv"
    path.write_text(CSV)
    return load_generation_mix(path)


def test_keeps_only_generation_columns(tmp_path):
    mix = _load(tmp_path)
    # "Actual Consumption" (pumped-storage charging) is demand, not generation.
    assert list(mix.columns) == ["Fossil Gas", "Nuclear", "Hydro Pumped Storage"]
    assert mix["Hydro Pumped Storage"].iloc[1] == 50.0  # the generation value, not 540


def test_index_is_tz_aware_monotonic_across_dst(tmp_path):
    mix = _load(tmp_path)
    assert isinstance(mix.index, pd.DatetimeIndex)
    assert str(mix.index.tz) == "Europe/Brussels"
    assert mix.index.is_monotonic_increasing
    # 01:00 CET -> 03:00 CEST is one real hour apart, not two.
    assert (mix.index[2] - mix.index[1]) == pd.Timedelta(hours=1)
    assert mix.index[0] == pd.Timestamp("2023-03-25 23:00", tz="UTC")


def test_values_preserved_and_gaps_stay_nan(tmp_path):
    mix = _load(tmp_path)
    assert mix["Fossil Gas"].tolist() == [1000.0, 1100.0, 1200.0, 1300.0]
    assert mix["Hydro Pumped Storage"].isna().sum() == 3

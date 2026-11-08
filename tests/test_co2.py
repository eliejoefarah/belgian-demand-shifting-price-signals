"""fleet_harness.co2 on a tiny synthetic run. The fake microgrid exposes only
what co2.py reads: power_hist[0] (datetimes + PublicGrid_0 Power objects with
.electrical in kW) and time_step."""
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from fleet_harness.co2 import compute_rung_co2, get_grid_offtake_kwh
from price_signals.emission_factors import DIRECT_FACTORS

GAS = DIRECT_FACTORS["Fossil Gas"]


def fake_microgrid(start, kw_values, step=timedelta(minutes=15)):
    dts = pd.date_range(start, periods=len(kw_values), freq=step, tz="UTC")
    node0 = {
        "datetime": list(dts.to_pydatetime()),
        "PublicGrid_0": [SimpleNamespace(electrical=v) for v in kw_values],
    }
    return SimpleNamespace(power_hist=[node0], time_step=step)


def hourly_mix(start, rows):
    """rows: list of (gas MW, nuclear MW), hourly from `start` (Brussels time)."""
    idx = pd.date_range(start, periods=len(rows), freq="h", tz="Europe/Brussels")
    return pd.DataFrame(rows, index=idx, columns=["Fossil Gas", "Nuclear"])


def test_offtake_clips_injection_and_converts_to_kwh():
    # 4 kW for 15 min = 1 kWh; negative (injection) counts as 0.
    mg = fake_microgrid("2023-01-01 00:00", [4.0, -2.0, 8.0, 0.0])
    offtake = get_grid_offtake_kwh(mg)
    assert offtake.tolist() == [1.0, 0.0, 2.0, 0.0]
    assert str(offtake.index.tz) == "Europe/Brussels"


def test_co2_uses_ffill_alignment_and_weighted_intensity():
    # Brussels hours 01:00 (100 % gas) and 02:00 (100 % nuclear) = 00:00, 01:00 UTC.
    mix = hourly_mix("2023-01-01 01:00", [(500.0, 0.0), (0.0, 500.0)])
    # 15-min steps: first hour 4 kW (1 kWh/step), second hour 8 kW (2 kWh/step).
    mg = fake_microgrid("2023-01-01 00:00", [4.0] * 4 + [8.0] * 4)

    result = compute_rung_co2(mg, mix, factor_set="direct")

    # Each 15-min step inherits the intensity of its hour (ffill):
    # 4 kWh at GAS g/kWh + 8 kWh at 0 g/kWh.
    assert result["grid_offtake_kwh"] == pytest.approx(12.0)
    assert result["co2_g"] == pytest.approx(4 * GAS)
    assert result["co2_kg"] == pytest.approx(4 * GAS / 1000)
    assert result["avg_intensity_paid_gCO2_per_kWh"] == pytest.approx(4 * GAS / 12)
    assert result["factor_set"] == "direct"


def test_co2_raises_when_offtake_starts_before_mix_coverage():
    mix = hourly_mix("2023-01-01 02:00", [(500.0, 0.0), (0.0, 500.0)])
    mg = fake_microgrid("2023-01-01 00:00", [4.0] * 8)  # starts 01:00 Brussels
    with pytest.raises(ValueError, match="outside the intensity"):
        compute_rung_co2(mg, mix)


def test_zero_offtake_gives_nan_average_not_crash():
    mix = hourly_mix("2023-01-01 01:00", [(500.0, 0.0)])
    mg = fake_microgrid("2023-01-01 00:00", [-3.0] * 4)
    result = compute_rung_co2(mg, mix)
    assert result["co2_kg"] == 0.0
    assert pd.isna(result["avg_intensity_paid_gCO2_per_kWh"])


def test_co2_raises_when_mix_ends_before_offtake():
    # reindex(ffill) alone would repeat the last hour onto every later step.
    mix = hourly_mix("2023-01-01 01:00", [(500.0, 0.0)])
    mg = fake_microgrid("2023-01-01 00:00", [4.0] * 16)  # 4 h of steps, mix covers 1 h
    with pytest.raises(ValueError, match="outside the intensity"):
        compute_rung_co2(mg, mix)


def test_co2_accepts_the_utc_year_end_hour():
    # ENTSO-E year ends 23:00 Brussels (22:00 UTC) + 1 h; the UTC run goes to 23:45 UTC.
    mix = hourly_mix("2023-12-31 22:00", [(500.0, 0.0), (0.0, 500.0)])
    mg = fake_microgrid("2023-12-31 21:00", [4.0] * 12)  # 21:00..23:45 UTC
    result = compute_rung_co2(mg, mix)
    assert result["co2_g"] == pytest.approx(4 * GAS)  # 21:xx UTC = 22:xx Brussels = gas

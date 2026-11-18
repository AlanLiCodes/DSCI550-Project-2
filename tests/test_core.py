import numpy as np
import pandas as pd
import pytest

from vesselnoise import config
from vesselnoise.data import BAND_COLS, standardise
from vesselnoise.energy import db_to_lin, energy_mean_db, lin_to_db, share_from_parts
from vesselnoise.schema import match_band_columns, parse_frequency
from vesselnoise.windows import build_windows, windows_from_long


def test_band_centres():
    assert config.N_BANDS == 25
    assert config.BAND_CENTRES_HZ[0] == pytest.approx(31.62, rel=1e-3)
    assert config.BAND_CENTRES_HZ[-1] == pytest.approx(7943, rel=1e-3)


def test_db_lin_roundtrip_and_energy_mean():
    x = np.array([60.0, 70.0, np.nan])
    assert np.allclose(lin_to_db(db_to_lin(x))[:2], x[:2])
    assert np.isnan(lin_to_db(db_to_lin(x))[2])
    # energy mean of 60 and 70 dB is ~67.4 dB, not 65
    assert energy_mean_db(x) == pytest.approx(67.40, abs=0.01)


def test_share_ignores_missing_cells():
    added = np.ones((2, 2))
    mix = np.array([[2.0, 2.0], [np.nan, 2.0]])
    assert share_from_parts(added, mix, axis=None) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "name,freq",
    [("spl_31.6Hz", 31.6), ("band_1000", 1000.0), ("1.25k", 1250.0), ("b03_31.6", 31.6), ("L_7943", 7943.0)],
)
def test_parse_frequency(name, freq):
    assert parse_frequency(name) == pytest.approx(freq)


def test_match_band_columns_orders_by_frequency():
    names = [f"{c:.0f}Hz" for c in config.BAND_CENTRES_HZ][::-1] + ["site"]
    cols = match_band_columns(names)
    assert cols[0] == "32Hz" and cols[-1] == "7943Hz"


def _records(minutes, start="2025-01-01 00:00", station="A"):
    t = pd.date_range(start, periods=minutes, freq="min")
    df = pd.DataFrame(80.0, index=range(minutes), columns=[f"f{c:.1f}" for c in config.BAND_CENTRES_HZ])
    df.insert(0, "timestamp", t)
    df.insert(0, "station", station)
    return df


def test_windows_do_not_bridge_gaps_or_days():
    a = _records(40, "2025-01-01 00:00")  # 2 windows + 10 leftover
    b = _records(20, "2025-01-01 01:00")  # gap before -> 1 window
    c = _records(30, "2025-01-01 23:50")  # crosses midnight: 10 before, 20 after -> 1 window
    std, _ = standardise(pd.concat([a, b, c], ignore_index=True))
    X, meta = build_windows(std)
    assert X.shape == (4, 15, 25)
    assert meta["start"].astype(str).tolist() == [
        "2025-01-01 00:00:00",
        "2025-01-01 00:15:00",
        "2025-01-01 01:00:00",
        "2025-01-02 00:00:00",
    ]


def test_quality_flag_masks_rows():
    df = _records(15)
    df["quality_flag"] = 0
    df.loc[3, "quality_flag"] = 2
    std, sch = standardise(df)
    assert sch.quality == "quality_flag"
    assert std.loc[3, BAND_COLS].isna().all() and not std.loc[4, BAND_COLS].isna().any()


def test_linear_units_converted_to_db():
    df = _records(15)
    bands = [c for c in df.columns if c.startswith("f")]
    df[bands] = 1e8  # 80 dB in linear energy
    std, sch = standardise(df)
    assert sch.units == "linear"
    assert np.allclose(std[BAND_COLS].to_numpy(), 80.0)


def test_windows_from_long_orders_by_minute():
    rows = []
    for wid in ["w2", "w1"]:
        d = _records(15).drop(columns="timestamp")
        d.insert(0, "minute", np.arange(15)[::-1])
        d.insert(0, "ID", wid)
        d.iloc[:, 3] = d["minute"].to_numpy() + 60.0
        rows.append(d)
    std, _ = standardise(pd.concat(rows, ignore_index=True))
    X, meta = windows_from_long(std)
    assert X.shape == (2, 15, 25)
    assert np.allclose(X[0, :, 0], np.arange(15) + 60.0)


def test_short_runs_padded_with_masked_minutes():
    df = _records(13)  # duty-cycled 13-minute run
    std, _ = standardise(df)
    X, meta = build_windows(std, min_run=13)
    assert X.shape == (1, 15, 25)
    assert np.isnan(X[0, 13:]).all() and not np.isnan(X[0, :13]).any()
    assert meta["observed_minutes"].iloc[0] == 13
    assert build_windows(std, min_run=15)[0].shape[0] == 0


def test_sentinels_and_short_duration_masked():
    df = _records(15)
    df["dur"] = 60.0
    df.loc[2, "dur"] = 20.0
    band = [c for c in df.columns if c.startswith("f")][0]
    df.loc[5, band] = -150.0
    std, _ = standardise(df)
    assert std.loc[2, BAND_COLS].isna().all()
    assert np.isnan(std.loc[5, BAND_COLS[0]]) and not np.isnan(std.loc[5, BAND_COLS[1]])


def test_wide_test_format_parsed():
    from vesselnoise.data import windows_from_wide

    cols = [f"m{m:02d}_{b}" for m in range(1, 16) for b in config.BAND_COLUMN_NAMES]
    vals = np.arange(2 * 375, dtype=float).reshape(2, 375) / 10 + 50
    df = pd.DataFrame(vals, columns=cols)
    df.insert(0, "ID", ["a", "b"])
    X, meta = windows_from_wide(df)
    assert X.shape == (2, 15, 25)
    assert X[0, 0, 1] == vals[0, 1] and X[0, 1, 0] == vals[0, 25]
    assert meta["window_id"].tolist() == ["a", "b"]

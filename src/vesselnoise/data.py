"""Loading and standardising the competition files.

Standard internal layout of training records (one row per recorder-minute):
    station, recorder, timestamp, b00 .. b24   (band levels in dB; NaN = missing)

Released files
--------------
* ``South_African_Real_Acoustics.zip`` -> ``acoustics/<station>.parquet``: columns
  ``site, recorder, t, dur, TOL_<f> ...`` (many more bands than the 25 used).
* ``Test.csv``: one row per window, ``ID`` + 375 columns ``mMM_TOL_<f>``
  (15 minute positions x 25 bands, low -> high frequency within each minute).
  No station or timestamp is provided.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .energy import lin_to_db
from .schema import Schema, match_band_columns

BAND_COLS = [f"b{i:02d}" for i in range(config.N_BANDS)]


def _is_good_quality(series: pd.Series) -> np.ndarray:
    good = config.GOOD_QUALITY_VALUES
    s = series
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
    return s.isin(good).to_numpy()


def mask_invalid_levels(vals: np.ndarray, valid_range=config.VALID_DB_RANGE) -> np.ndarray:
    """Set non-finite and out-of-range (sentinel) dB values to NaN, in place."""
    lo, hi = valid_range
    with np.errstate(invalid="ignore"):
        vals[~np.isfinite(vals) | (vals < lo) | (vals > hi)] = np.nan
    return vals


def standardise(df: pd.DataFrame, schema: Schema | None = None, min_duration_s: float | None = config.MIN_DURATION_S):
    """Rename to the standard layout, convert to dB, and mask bad cells.

    * Rows whose quality flag is not in ``config.GOOD_QUALITY_VALUES`` are masked.
    * Rows observed for less than ``min_duration_s`` seconds (``dur``) are masked.
    * Sentinel levels outside ``config.VALID_DB_RANGE`` are masked.
    * Timestamps are floored to the minute; duplicate recorder-minutes keep the first row.

    Masked rows are kept (all bands NaN) so that windows are not bridged across them.
    """
    schema = schema or Schema.detect(df)
    out = pd.DataFrame(index=df.index)
    out["station"] = df[schema.station].astype(str) if schema.station else "S0"
    if schema.recorder:
        out["recorder"] = df[schema.recorder].astype(str)
    if schema.window_id:
        out["window_id"] = df[schema.window_id]
    if schema.minute:
        out["minute"] = pd.to_numeric(df[schema.minute], errors="coerce")
    if schema.time:
        out["timestamp"] = pd.to_datetime(df[schema.time], errors="coerce").dt.floor("min")

    vals = df[schema.band_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    if schema.units == "linear":
        vals = lin_to_db(vals)
    mask_invalid_levels(vals)

    if schema.quality:
        bad = ~_is_good_quality(df[schema.quality])
        vals[bad] = np.nan
        out["quality_ok"] = ~bad
    if schema.duration and min_duration_s is not None:
        dur = pd.to_numeric(df[schema.duration], errors="coerce").to_numpy()
        vals[~(dur >= min_duration_s)] = np.nan

    out[BAND_COLS] = vals.astype(np.float32)
    if "timestamp" in out and "window_id" not in out:
        keys = [c for c in ["station", "recorder", "timestamp"] if c in out]
        out = out.drop_duplicates(keys, keep="first")
    return out, schema


def load_csv(path, nrows=None, min_duration_s=config.MIN_DURATION_S):
    raw = pd.read_csv(path, nrows=nrows)
    return standardise(raw, min_duration_s=min_duration_s)


def load_station_parquets(path, stations=None, min_duration_s=config.MIN_DURATION_S):
    """Load ``acoustics/*.parquet`` (one file per station) from the training archive.

    Only the identifier columns, ``dur`` and the 25 common bands are read.
    """
    path = Path(path)
    files = sorted((path / "acoustics").glob("*.parquet")) or sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    cols = ["site", "recorder", "t", "dur", *config.BAND_COLUMN_NAMES]
    frames, schema = [], None
    for f in files:
        if stations and not any(f.stem.startswith(s) for s in stations):
            continue
        std, schema = standardise(pd.read_parquet(f, columns=cols), min_duration_s=min_duration_s)
        frames.append(std)
    return pd.concat(frames, ignore_index=True), schema


def load_records(path, nrows=None):
    """Training records from a parquet directory/zip extract or a CSV file."""
    p = Path(path)
    if p.is_dir():
        return load_station_parquets(p)
    if p.suffix == ".parquet":
        return standardise(pd.read_parquet(p))
    return load_csv(p, nrows=nrows)


_WIDE_RE = re.compile(r"^m(\d{1,2})_(.+)$")


def is_wide_window_file(columns) -> bool:
    return sum(bool(_WIDE_RE.match(str(c))) for c in columns) == config.WINDOW_MINUTES * config.N_BANDS


def windows_from_wide(df: pd.DataFrame, id_col: str = "ID"):
    """Parse a Test.csv-style file: one row per window, columns ``mMM_<band>``.

    Returns X (n, 15, 25) dB with NaN for missing cells, and meta with window_id.
    Station and hour are not available for these windows.
    """
    parsed = {}
    for c in df.columns:
        m = _WIDE_RE.match(str(c))
        if m:
            parsed[c] = (int(m.group(1)), m.group(2))
    minutes = sorted({v[0] for v in parsed.values()})
    band_names = match_band_columns(sorted({v[1] for v in parsed.values()}))
    order = [f"m{mi:02d}_{b}" for mi in minutes for b in band_names]
    missing = [c for c in order if c not in df.columns]
    if missing:
        raise ValueError(f"unexpected wide column naming, e.g. {missing[:3]}")
    vals = df[order].to_numpy(dtype=float, copy=True)
    mask_invalid_levels(vals)
    X = vals.reshape(len(df), len(minutes), len(band_names))
    meta = pd.DataFrame({"window_id": df[id_col].astype(str).to_numpy()})
    meta["missing_frac"] = np.isnan(X).mean(axis=(1, 2))
    return X, meta


def load_test(path):
    """Load the competition test file -> X, meta, id column name."""
    df = pd.read_csv(path, dtype={"ID": str})
    if is_wide_window_file(df.columns):
        id_col = next((c for c in df.columns if str(c).lower() == "id"), df.columns[0])
        X, meta = windows_from_wide(df, id_col)
        return X, meta, id_col
    from .windows import windows_from_long

    std, sch = standardise(df)
    X, meta = windows_from_long(std)
    return X, meta, sch.window_id or "ID"

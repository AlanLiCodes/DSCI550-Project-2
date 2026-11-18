"""Loading and standardising the competition files.

Standard internal layout (one row per recorder-minute):
    station, timestamp, b00 .. b24   (band levels in dB; NaN = missing)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .energy import lin_to_db
from .schema import Schema

BAND_COLS = [f"b{i:02d}" for i in range(config.N_BANDS)]


def _is_good_quality(series: pd.Series) -> np.ndarray:
    good = config.GOOD_QUALITY_VALUES
    s = series
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
    return s.isin(good).to_numpy()


def standardise(df: pd.DataFrame, schema: Schema | None = None, min_duration_frac: float | None = None):
    """Rename to the standard layout, convert to dB, and mask bad cells.

    * Rows whose quality flag is not in ``config.GOOD_QUALITY_VALUES`` have all
      band values set to NaN (kept as masked minutes, so windows are not bridged).
    * If ``min_duration_frac`` is given and a duration column exists (seconds per
      one-minute record), records observed for less than that fraction are masked.
    """
    schema = schema or Schema.detect(df)
    out = pd.DataFrame(index=df.index)
    if schema.station:
        out["station"] = df[schema.station].astype(str)
    else:
        out["station"] = "S0"
    if schema.window_id:
        out["window_id"] = df[schema.window_id]
    if schema.minute:
        out["minute"] = pd.to_numeric(df[schema.minute], errors="coerce")
    if schema.time:
        out["timestamp"] = pd.to_datetime(df[schema.time], errors="coerce", utc=False)

    vals = df[schema.band_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    if schema.units == "linear":
        vals = lin_to_db(vals)
    vals[~np.isfinite(vals)] = np.nan

    if schema.quality:
        bad = ~_is_good_quality(df[schema.quality])
        vals[bad] = np.nan
        out["quality_ok"] = ~bad
    if schema.duration and min_duration_frac is not None:
        dur = pd.to_numeric(df[schema.duration], errors="coerce").to_numpy()
        short = dur < min_duration_frac * 60.0
        vals[short] = np.nan

    out[BAND_COLS] = vals
    return out, schema


def load_csv(path, nrows=None, min_duration_frac=None):
    raw = pd.read_csv(path, nrows=nrows)
    return standardise(raw, min_duration_frac=min_duration_frac)

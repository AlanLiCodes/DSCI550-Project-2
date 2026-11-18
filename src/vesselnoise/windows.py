"""Forming 15-minute windows.

A window lives inside one recorder and one calendar day and never bridges a
recording gap. Missing cells inside a window stay NaN (masked), they are not
filled at this stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import BAND_COLS


def build_windows(
    df: pd.DataFrame,
    minutes: int = config.WINDOW_MINUTES,
    max_step_minutes: float = config.MAX_STEP_MINUTES,
    max_missing_frac: float = 0.5,
):
    """Cut standardised records into non-overlapping windows.

    Returns
    -------
    X : ndarray (n_windows, minutes, 25), dB with NaN for missing cells
    meta : DataFrame with station, start, day, hour, segment, missing_frac
    """
    d = df.dropna(subset=["timestamp"]).sort_values(["station", "timestamp"]).reset_index(drop=True)
    ts = d["timestamp"]
    day = ts.dt.floor("D")
    step = ts.diff().dt.total_seconds().div(60.0)
    new_seg = (
        (d["station"] != d["station"].shift())
        | (day != day.shift())
        | (step > max_step_minutes)
        | step.isna()
    )
    seg = new_seg.cumsum().to_numpy()
    pos = d.groupby(seg).cumcount().to_numpy()
    win_key = seg.astype(np.int64) * 100_000 + pos // minutes
    sizes = pd.Series(win_key).map(pd.Series(win_key).value_counts()).to_numpy()
    keep = sizes == minutes
    d = d.loc[keep].reset_index(drop=True)
    win_key = win_key[keep]

    X = d[BAND_COLS].to_numpy(dtype=float).reshape(-1, minutes, config.N_BANDS)
    first = d.iloc[::minutes]
    meta = pd.DataFrame(
        {
            "station": first["station"].to_numpy(),
            "start": first["timestamp"].to_numpy(),
            "segment": seg[keep][::minutes],
        }
    )
    meta["day"] = pd.to_datetime(meta["start"]).dt.floor("D")
    meta["hour"] = pd.to_datetime(meta["start"]).dt.hour
    meta["missing_frac"] = np.isnan(X).mean(axis=(1, 2))

    ok = meta["missing_frac"].to_numpy() <= max_missing_frac
    X, meta = X[ok], meta.loc[ok].reset_index(drop=True)
    meta["window_id"] = [f"{s}_{pd.Timestamp(t):%Y%m%d%H%M}" for s, t in zip(meta["station"], meta["start"])]
    return X, meta


def windows_from_long(df: pd.DataFrame, minutes: int = config.WINDOW_MINUTES):
    """Rebuild windows from a file that already has one row per (window_id, minute).

    Used for Test.csv. Rows are ordered by ``minute`` if present, else by
    ``timestamp``, else by file order.
    """
    if "window_id" not in df.columns:
        raise ValueError(
            "No window ID column detected in the test file. Run scripts/inspect_data.py "
            "and add the ID column name to config.ID_CANDIDATES."
        )
    order = [c for c in ["minute", "timestamp"] if c in df.columns]
    d = df.copy()
    d["_row"] = np.arange(len(d))
    d = d.sort_values(["window_id", *order, "_row"], kind="stable")
    counts = d.groupby("window_id", sort=False).size()
    bad = counts[counts != minutes]
    if len(bad):
        raise ValueError(f"{len(bad)} test windows do not have {minutes} rows (e.g. {bad.index[:3].tolist()})")
    ids = counts.index.to_numpy()
    X = d[BAND_COLS].to_numpy(dtype=float).reshape(-1, minutes, config.N_BANDS)
    first = d.groupby("window_id", sort=False).head(1).reset_index(drop=True)
    meta = pd.DataFrame({"window_id": ids, "station": first["station"].to_numpy()})
    if "timestamp" in first.columns:
        meta["start"] = first["timestamp"].to_numpy()
        meta["hour"] = pd.to_datetime(meta["start"]).dt.hour
        meta["day"] = pd.to_datetime(meta["start"]).dt.floor("D")
    else:
        meta["hour"] = np.nan
    meta["missing_frac"] = np.isnan(X).mean(axis=(1, 2))
    return X, meta

"""Forming 15-minute windows.

A window lives inside one station, one recorder and one calendar day and never bridges a
recording gap. Missing cells inside a window stay NaN (masked), they are not
filled at this stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import BAND_COLS
from .energy import db_to_lin, lin_to_db


def run_start_transients(df: pd.DataFrame, threshold_db: float = 6.0, max_pos: int = 3, min_runs: int = 1) -> pd.DataFrame:
    """Detect recorder-specific start-up transients after recording gaps.

    For each recorder, a run is a stretch of consecutive one-minute records. The
    broadband level at run position k (k < ``max_pos``) is compared with the run's
    own median over positions 3-12; positions whose median deviation exceeds
    ``threshold_db`` are flagged. In the released archive most recorders read
    18-36 dB high in the first minute after a gap (two minutes for the 6509 units
    at Bird Island / St Croix); St Francis and Fish Hoek show no transient.

    Returns a table: station, recorder, runs, dev_pos0.., masked_positions.
    """
    keys = ["station", "recorder"] if "recorder" in df.columns else ["station"]
    rows = []
    for key, d in df.dropna(subset=["timestamp"]).sort_values([*keys, "timestamp"]).groupby(keys, sort=False):
        lin = db_to_lin(d[BAND_COLS].to_numpy(dtype=float))
        with np.errstate(invalid="ignore"):
            bb = lin_to_db(np.nanmean(lin, axis=1))
        step = d["timestamp"].diff().dt.total_seconds().div(60.0).to_numpy()
        run = np.cumsum(~(step == 1.0))
        pos = pd.Series(run).groupby(run).cumcount().to_numpy()
        ref = pd.Series(np.where((pos >= 3) & (pos < 13), bb, np.nan)).groupby(run).transform("median").to_numpy()
        dev = bb - ref
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row["runs"] = int(run.max())
        flagged = []
        for k in range(max_pos):
            sel = (pos == k) & np.isfinite(dev)
            med = float(np.median(dev[sel])) if sel.sum() else np.nan
            row[f"dev_pos{k}"] = med
            if row["runs"] >= min_runs and np.isfinite(med) and abs(med) > threshold_db:
                flagged.append(k)
        row["masked_positions"] = tuple(flagged)
        rows.append(row)
    return pd.DataFrame(rows)


def mask_run_start_transients(df: pd.DataFrame, table: pd.DataFrame | None = None, **kwargs):
    """Mask (set to NaN) the flagged run-start minutes of each recorder. Returns (df, table)."""
    table = run_start_transients(df, **kwargs) if table is None else table
    keys = ["station", "recorder"] if "recorder" in df.columns else ["station"]
    out = df.sort_values([*keys, "timestamp"]).copy()
    step = out.groupby(keys, sort=False)["timestamp"].diff().dt.total_seconds().div(60.0)
    run = (~(step == 1.0)).cumsum()
    pos = out.groupby(run).cumcount()
    flagged = {tuple(r[k] for k in keys): set(r["masked_positions"]) for _, r in table.iterrows()}
    rec_key = list(zip(*[out[k] for k in keys]))
    mask = np.array([p in flagged.get(k, ()) for k, p in zip(rec_key, pos.to_numpy())])
    out.loc[mask, BAND_COLS] = np.nan
    return out.sort_index(), table


def build_windows(
    df: pd.DataFrame,
    minutes: int = config.WINDOW_MINUTES,
    max_step_minutes: float = config.MAX_STEP_MINUTES,
    max_missing_frac: float = 0.5,
    min_run: int = config.MIN_RUN_MINUTES,
):
    """Cut standardised records into non-overlapping windows.

    Each contiguous segment (same station, recorder and day, no gap) is cut into
    consecutive 15-minute pieces. A piece shorter than 15 minutes (segment end, or
    a duty-cycled recorder such as the 13-minute runs at Cape Recife / Riy Banks)
    is kept when it has at least ``min_run`` minutes; its trailing minute
    positions are masked (NaN), never filled from outside the segment.

    Returns
    -------
    X : ndarray (n_windows, minutes, 25) float32, dB with NaN for missing cells
    meta : DataFrame with station, recorder, start, day, hour, segment, missing_frac
    """
    keys = ["station", "recorder"] if "recorder" in df.columns else ["station"]
    d = df.dropna(subset=["timestamp"]).sort_values([*keys, "timestamp"]).reset_index(drop=True)
    ts = d["timestamp"]
    day = ts.dt.floor("D")
    step = ts.diff().dt.total_seconds().div(60.0)
    new_seg = (
        (d["station"] != d["station"].shift())
        | (d[keys[-1]] != d[keys[-1]].shift())
        | (day != day.shift())
        | (step > max_step_minutes)
        | step.isna()
    )
    seg = new_seg.cumsum().to_numpy()
    pos = d.groupby(seg).cumcount().to_numpy()
    win_key = seg.astype(np.int64) * 100_000 + pos // minutes
    counts = pd.Series(win_key).value_counts()
    sizes = pd.Series(win_key).map(counts).to_numpy()
    keep = sizes >= min(min_run, minutes)
    d = d.loc[keep].reset_index(drop=True)
    win_key, pos, seg = win_key[keep], pos[keep] % minutes, seg[keep]

    uniq, w_idx = np.unique(win_key, return_inverse=True)
    X = np.full((len(uniq), minutes, config.N_BANDS), np.nan, dtype=np.float32)
    X[w_idx, pos] = d[BAND_COLS].to_numpy(dtype=np.float32)
    first_row = np.flatnonzero(pos == 0)
    first = d.iloc[first_row]
    order = w_idx[first_row]
    meta = pd.DataFrame(index=order)
    meta["station"] = first["station"].to_numpy()
    meta["start"] = first["timestamp"].to_numpy()
    meta["segment"] = seg[first_row]
    if "recorder" in first.columns:
        meta["recorder"] = first["recorder"].to_numpy()
    meta = meta.sort_index().reset_index(drop=True)
    meta["day"] = pd.to_datetime(meta["start"]).dt.floor("D")
    meta["hour"] = pd.to_datetime(meta["start"]).dt.hour
    meta["missing_frac"] = np.isnan(X).mean(axis=(1, 2))
    meta["observed_minutes"] = (~np.isnan(X).all(axis=2)).sum(axis=1)

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

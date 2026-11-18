"""Evaluation splits that respect time, site and source-vessel structure (Roberts et al., 2017)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def chronological_split(meta: pd.DataFrame, fracs=(0.6, 0.2, 0.2), buffer_days: int = 1) -> np.ndarray:
    """Per station, assign whole days to train / val / test in time order.

    The first ``buffer_days`` days of the val and test blocks are labelled
    'buffer' and excluded, so neighbouring days never straddle a split.
    """
    labels = np.full(len(meta), "buffer", dtype=object)
    days = pd.to_datetime(meta["day"]).dt.normalize()
    for s in meta["station"].unique():
        m = (meta["station"] == s).to_numpy()
        ud = np.sort(days[m].unique())
        n = len(ud)
        c1 = int(round(fracs[0] * n))
        c2 = int(round((fracs[0] + fracs[1]) * n))
        # Short records: make sure val and test each keep at least one day after their buffer.
        need = buffer_days + 1
        if n >= 1 + 2 * need:
            c1 = min(c1, n - 2 * need)
            c2 = min(max(c2, c1 + need), n - need)
        day_label = {}
        for i, d in enumerate(ud):
            if i < c1:
                day_label[d] = "train"
            elif i < c2:
                day_label[d] = "buffer" if i < c1 + buffer_days else "val"
            else:
                day_label[d] = "buffer" if i < c2 + buffer_days else "test"
        labels[m] = days[m].map(day_label).to_numpy()
    return labels


def assign_reference_splits(refs, fracs=(0.6, 0.2, 0.2), seed: int = config.SEED) -> dict:
    """Each source vessel (and every clip derived from it) belongs to exactly one split.

    Assignment is stratified by reference source.
    """
    rng = np.random.default_rng(seed)
    out = {}
    by_source: dict[str, list[str]] = {}
    for r in refs:
        by_source.setdefault(r.source, []).append(r.vessel_id)
    for ids in by_source.values():
        ids = list(rng.permutation(sorted(set(ids))))
        n = len(ids)
        c1, c2 = int(round(fracs[0] * n)), int(round((fracs[0] + fracs[1]) * n))
        for i, v in enumerate(ids):
            out[v] = "train" if i < c1 else ("val" if i < c2 else "test")
    return out


def leave_one_site_out(stations):
    for s in sorted(pd.unique(np.asarray(stations))):
        yield s


def daily_residual_autocorr(residuals, meta: pd.DataFrame, max_lag: int = 5) -> pd.Series:
    """Mean (over stations) autocorrelation of daily-mean residuals at lags 1..max_lag days.

    Used to check that a one-day buffer is enough.
    """
    df = pd.DataFrame({"r": residuals, "station": meta["station"].to_numpy(), "day": pd.to_datetime(meta["day"]).to_numpy()})
    acs = {lag: [] for lag in range(1, max_lag + 1)}
    for _, g in df.groupby("station"):
        daily = g.groupby("day")["r"].mean().asfreq("D")
        for lag in acs:
            if daily.notna().sum() < lag + 3:
                continue
            v = daily.autocorr(lag)
            if np.isfinite(v):
                acs[lag].append(v)
    return pd.Series({lag: float(np.mean(v)) if v else np.nan for lag, v in acs.items()}, name="autocorr")

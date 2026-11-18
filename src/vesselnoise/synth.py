"""Synthetic stand-in for the competition files.

Only for development and tests while the real data are not in the repo. The
generator mimics the structure described in the proposal: ten stations, one-
minute band levels in dB, recording gaps, missing cells, quality flags,
station-specific spectra, wind-driven variability and a diurnal biological
chorus. Column names are intentionally *not* the internal ones so that the
schema-detection path is exercised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config
from .mixtures import make_mixtures
from .reference import synthetic_reference_set

RAW_BAND_NAMES = [f"spl_{c:.1f}Hz" for c in config.BAND_CENTRES_HZ]


def synthetic_records(n_stations: int = 10, n_days: int = 20, seed: int = config.SEED, gap_rate: float = 0.002):
    rng = np.random.default_rng(seed)
    logf = np.log10(config.BAND_CENTRES_HZ)
    wind_w = np.clip((logf - np.log10(300)) / (np.log10(2000) - np.log10(300)), 0, 1)
    chorus_w = np.exp(-0.5 * ((logf - np.log10(rng.uniform(600, 1500))) / 0.12) ** 2)
    frames = []
    for s in range(n_stations):
        base = 95 - 12 * (logf - logf[0]) + rng.normal(0, 3) + rng.normal(0, 1.5, config.N_BANDS)
        n = n_days * 1440
        t = pd.date_range("2025-03-01", periods=n, freq="min") + pd.Timedelta(days=int(rng.integers(0, 60)))
        wind = lfilter([1.0], [1.0, -0.999], rng.normal(0, 0.05, n))  # AR(1) sea state
        hour = t.hour.to_numpy() + t.minute.to_numpy() / 60
        night = 0.5 * (1 + np.cos(2 * np.pi * (hour - 1) / 24))
        lvl = (
            base[None, :]
            + 6 * wind[:, None] * wind_w[None, :]
            + rng.uniform(4, 12) * night[:, None] * chorus_w[None, :]
            + rng.normal(0, 1.5, (n, config.N_BANDS))
        )
        df = pd.DataFrame(lvl, columns=RAW_BAND_NAMES)
        df.insert(0, "timestamp", t)
        df.insert(0, "site", f"ZA{s + 1:02d}")
        qf = np.zeros(n, dtype=int)
        qf[rng.random(n) < 0.003] = 2
        df["quality_flag"] = qf
        # Missing cells.
        miss = rng.random(df[RAW_BAND_NAMES].shape) < 0.01
        df[RAW_BAND_NAMES] = df[RAW_BAND_NAMES].mask(miss)
        # Recording gaps: drop runs of rows.
        drop = np.zeros(n, dtype=bool)
        for g in np.flatnonzero(rng.random(n) < gap_rate):
            drop[g : g + int(rng.integers(5, 240))] = True
        frames.append(df.loc[~drop])
    return pd.concat(frames, ignore_index=True)


def synthetic_test(records_std: pd.DataFrame, n: int = 300, seed: int = config.SEED + 1):
    """Synthetic Test.csv in long format (ID, minute, bands) plus hidden labels."""
    from .windows import build_windows

    X_bg, meta = build_windows(records_std)
    refs = synthetic_reference_set(30, 20, seed=seed)
    X, y, present, info = make_mixtures(X_bg, meta, refs, n, seed=seed)
    ids = [f"ID_{i:05d}" for i in range(n)]
    rows = []
    for i in range(n):
        d = pd.DataFrame(X[i], columns=RAW_BAND_NAMES)
        d.insert(0, "minute", np.arange(X.shape[1]))
        d.insert(0, "timestamp", pd.date_range(info["start"].iloc[i], periods=X.shape[1], freq="min"))
        d.insert(0, "site", info["station"].iloc[i])
        d.insert(0, "ID", ids[i])
        rows.append(d)
    labels = pd.DataFrame({"ID": ids, config.TARGET_SHARE: y, config.TARGET_SCORE: present})
    return pd.concat(rows, ignore_index=True), labels


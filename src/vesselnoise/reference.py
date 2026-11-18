"""Vessel-passage references.

A reference is a background-subtracted vessel passage expressed as linear
band energy per minute in the 25 competition bands, together with the
(approximate) range track at which it was recorded. Real references come from
HearMyShip (small vessels) and Ocean Networks Canada (large ships) audio via
:func:`audio_to_minute_bands`; until those are converted, :func:`synthetic_reference`
produces stand-ins so the whole pipeline can be developed and tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .energy import db_to_lin
from .propagation import passage_ranges, transmission_loss_db


@dataclass
class Reference:
    vessel_id: str
    source: str  # e.g. "hearmyship", "onc", "synthetic_small"
    lin: np.ndarray  # (T, 25) background-subtracted received energy per minute
    range_m: np.ndarray  # (T,) approximate range at which each minute was recorded
    spreading: float = 20.0

    @property
    def n_minutes(self) -> int:
        return self.lin.shape[0]


# ---------------------------------------------------------------- audio -----

def audio_to_minute_bands(x: np.ndarray, fs: float, sensitivity_db: float = 0.0, minute_s: float = 60.0):
    """Convert a pressure time series to minute-averaged decidecade band energy.

    * 1-second Hann-windowed FFT frames (1 Hz resolution), PSD in units^2/Hz.
    * PSD integrated over each decidecade band [lower, upper) edge.
    * Seconds averaged in linear energy into minute bins (partial final minute dropped).

    ``sensitivity_db`` (dB re V/uPa) converts volts to uPa; because references are
    rescaled to a target addition-to-background ratio, absolute calibration only
    matters for diagnostics, not for the labels.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x / (10.0 ** (sensitivity_db / 20.0))
    n = int(round(fs))
    n_frames = len(x) // n
    if n_frames == 0:
        raise ValueError("recording shorter than one second")
    frames = x[: n_frames * n].reshape(n_frames, n)
    win = np.hanning(n)
    spec = np.fft.rfft(frames * win, axis=1)
    psd = (np.abs(spec) ** 2) / (fs * np.sum(win**2))
    psd[:, 1:-1] *= 2.0  # one-sided
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    df = freqs[1] - freqs[0]
    bands = np.zeros((n_frames, config.N_BANDS))
    for k, (lo, hi) in enumerate(zip(config.BAND_LOWER_HZ, config.BAND_UPPER_HZ)):
        sel = (freqs >= lo) & (freqs < hi)
        bands[:, k] = psd[:, sel].sum(axis=1) * df if sel.any() else np.nan
    per_min = int(round(minute_s))
    n_min = n_frames // per_min
    if n_min == 0:
        raise ValueError("recording shorter than one minute")
    return bands[: n_min * per_min].reshape(n_min, per_min, -1).mean(axis=1)


def subtract_own_background(lin: np.ndarray, quantile: float = 0.1):
    """Remove the recording's own background: per band, subtract a low quantile over time."""
    bg = np.nanquantile(lin, quantile, axis=0, keepdims=True)
    return np.clip(lin - bg, 0.0, None), bg[0]


def reference_from_bands(vessel_id, source, lin, cpa_m=100.0, speed_ms=5.0, bg_quantile=0.1):
    """Build a Reference from minute band energy of a full passage recording.

    The range track is approximated from the recording's CPA (if known from AIS
    or metadata) and speed, centred on the minute of peak broadband energy.
    """
    lin_bs, _ = subtract_own_background(np.asarray(lin, dtype=float), bg_quantile)
    t_peak = float(np.nanargmax(np.nansum(lin_bs, axis=1))) + 0.5
    rng_track = passage_ranges(len(lin_bs), cpa_m, speed_ms, t_peak).mean(axis=1)
    return Reference(str(vessel_id), source, lin_bs, rng_track)


def load_reference_dir(path, source: str, default_cpa_m=100.0, default_speed_ms=5.0):
    """Load converted references: one CSV per passage with 25 band columns (linear energy).

    Optional sidecar ``metadata.csv`` with columns vessel_id, cpa_m, speed_ms.
    """
    path = Path(path)
    meta = {}
    mfile = path / "metadata.csv"
    if mfile.exists():
        m = pd.read_csv(mfile)
        meta = {str(r.vessel_id): r for r in m.itertuples()}
    refs = []
    for f in sorted(path.glob("*.csv")):
        if f.name == "metadata.csv":
            continue
        lin = pd.read_csv(f).iloc[:, -config.N_BANDS :].to_numpy(dtype=float)
        r = meta.get(f.stem)
        refs.append(
            reference_from_bands(
                f.stem,
                source,
                lin,
                cpa_m=getattr(r, "cpa_m", default_cpa_m),
                speed_ms=getattr(r, "speed_ms", default_speed_ms),
            )
        )
    return refs


# ------------------------------------------------------------ synthetic -----

def synthetic_source_level(rng: np.random.Generator, kind: str = "small"):
    """Plausible band source levels (dB re 1 uPa^2 m^2) for a small boat or large ship."""
    logf = np.log10(config.BAND_CENTRES_HZ)
    if kind == "small":
        peak_hz, peak_db = rng.uniform(250, 1500), rng.uniform(140, 160)
        lo_slope, hi_slope = rng.uniform(8, 15), rng.uniform(10, 20)
    else:
        peak_hz, peak_db = rng.uniform(40, 120), rng.uniform(165, 185)
        lo_slope, hi_slope = rng.uniform(5, 10), rng.uniform(15, 25)
    d = logf - np.log10(peak_hz)
    sl = np.where(d < 0, peak_db + lo_slope * d * 2.0, peak_db - hi_slope * d)
    # A few tonal / machinery bumps.
    for k in rng.choice(config.N_BANDS, size=rng.integers(1, 4), replace=False):
        sl[k] += rng.uniform(2, 8)
    return sl + rng.normal(0, 1.0, size=config.N_BANDS)


def synthetic_reference(rng: np.random.Generator, vessel_id: str, kind: str = "small", n_minutes: int = 30):
    """Simulate a recorded passage (as if from HearMyShip/ONC) at a recording CPA."""
    sl = synthetic_source_level(rng, kind)
    speed = rng.uniform(3, 15) if kind == "small" else rng.uniform(5, 10)
    cpa_rec = rng.uniform(30, 300)
    t_cpa = n_minutes / 2 + rng.uniform(-3, 3)
    r = passage_ranges(n_minutes, cpa_rec, speed, t_cpa)  # (T, sub)
    tl = transmission_loss_db(r.ravel(), spreading=20.0).reshape(n_minutes, r.shape[1], -1)
    lin = db_to_lin(sl[None, None, :] - tl).mean(axis=1)
    lin *= db_to_lin(rng.normal(0, 1.0, size=lin.shape))  # aspect / Lloyd-mirror jitter
    return Reference(vessel_id, f"synthetic_{kind}", lin, r.mean(axis=1))


def synthetic_reference_set(n_small: int = 60, n_large: int = 40, seed: int = config.SEED):
    rng = np.random.default_rng(seed)
    refs = [synthetic_reference(rng, f"small_{i:03d}", "small") for i in range(n_small)]
    refs += [synthetic_reference(rng, f"large_{i:03d}", "large") for i in range(n_large)]
    return refs

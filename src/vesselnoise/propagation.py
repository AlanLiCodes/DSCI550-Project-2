"""Simple range- and frequency-dependent propagation loss.

TL(r, f) = k * log10(r) + alpha(f) * r / 1000      [dB]

with k between 15 (practical) and 20 (spherical) spreading and alpha from
Thorp's absorption formula. This is deliberately simple: it only has to give
references a plausible range-dependent spectral tilt and passage shape
(Hjelmervik et al., 2024) instead of a single broadband gain.
"""

from __future__ import annotations

import numpy as np

from . import config


def thorp_absorption_db_per_km(f_hz):
    f = np.asarray(f_hz, dtype=float) / 1000.0  # kHz
    f2 = f * f
    return 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003


def transmission_loss_db(range_m, f_hz=config.BAND_CENTRES_HZ, spreading: float = 20.0):
    """TL for every (range, band) pair -> array (len(range_m), len(f_hz))."""
    r = np.maximum(np.atleast_1d(np.asarray(range_m, dtype=float)), 1.0)[:, None]
    alpha = thorp_absorption_db_per_km(f_hz)[None, :]
    return spreading * np.log10(r) + alpha * r / 1000.0


def passage_ranges(n_minutes: int, cpa_m: float, speed_ms: float, t_cpa_min: float, substeps: int = 6):
    """Range (m) to a vessel on a straight track, sampled ``substeps`` times per minute.

    Returns array (n_minutes, substeps).
    """
    t = (np.arange(n_minutes)[:, None] + (np.arange(substeps)[None, :] + 0.5) / substeps) - t_cpa_min
    along = speed_ms * 60.0 * t
    return np.sqrt(cpa_m**2 + along**2)

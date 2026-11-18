"""Level <-> energy conversions and masked energy arithmetic.

All averaging and summation is done in linear energy, never in dB.
Missing cells are NaN and stay masked (they never contribute zero energy).
"""

from __future__ import annotations

import numpy as np


def db_to_lin(db):
    """Convert levels in dB to linear energy (power ratio)."""
    return np.power(10.0, np.asarray(db, dtype=float) / 10.0)


def lin_to_db(lin, floor: float = 1e-30):
    """Convert linear energy to dB; non-positive values map to -inf-ish floor."""
    lin = np.asarray(lin, dtype=float)
    out = 10.0 * np.log10(np.maximum(lin, floor))
    return np.where(np.isnan(lin), np.nan, out)


def energy_mean_db(db, axis=None):
    """Energy (linear) mean of dB levels, returned in dB, ignoring NaN."""
    lin = db_to_lin(db)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(lin, axis=axis)
    return lin_to_db(m)


def masked_sum(lin, mask=None, axis=None):
    """Sum of linear energy over cells where ``mask`` is True (and not NaN)."""
    lin = np.asarray(lin, dtype=float)
    valid = ~np.isnan(lin)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return np.where(valid, lin, 0.0).sum(axis=axis)


def share_from_parts(added_lin, mixture_lin, observed=None, axis=(-2, -1)):
    """Label = summed added energy / summed mixture energy over observed cells."""
    if observed is None:
        observed = ~np.isnan(mixture_lin)
    num = masked_sum(added_lin, observed, axis=axis)
    den = masked_sum(mixture_lin, observed, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(den > 0, num / den, 0.0)
    return np.clip(share, 0.0, 1.0)

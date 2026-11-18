"""Controlled mixtures: South African background windows + placed vessel references.

For each training example:
  1. draw a background window (dB, NaN = missing) from the pool;
  2. with probability ``p_zero`` add nothing (label 0);
  3. otherwise crop a reference to the window so the passage's rise and fall stay
     inside it, re-place it at a sampled CPA distance with range-dependent
     propagation loss, and scale it to a sampled addition-to-background ratio;
  4. add in linear energy; label = summed added / summed mixture energy over
     the same observed (non-missing) cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .energy import db_to_lin, lin_to_db, masked_sum, share_from_parts
from .propagation import transmission_loss_db
from .reference import Reference


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def place_reference(
    ref: Reference,
    rng: np.random.Generator,
    minutes: int = config.WINDOW_MINUTES,
    cpa_range_m=config.CPA_RANGE_M,
    spreading_range=config.SPREADING_RANGE,
    peak_margin: int = 3,
):
    """Crop ``ref`` to ``minutes`` around its peak and move it to a new CPA distance.

    Returns (added_lin (minutes, 25), info dict).
    """
    broadband = np.nansum(ref.lin, axis=1)
    peak = int(np.nanargmax(broadband))
    peak_pos = int(rng.integers(peak_margin, minutes - peak_margin))
    start = peak - peak_pos

    seg = np.zeros((minutes, config.N_BANDS))
    r_ref = np.full(minutes, np.nan)
    src = np.arange(start, start + minutes)
    ok = (src >= 0) & (src < ref.n_minutes)
    seg[ok] = ref.lin[src[ok]]
    r_ref[ok] = ref.range_m[src[ok]]

    cpa_ref = float(np.nanmin(ref.range_m))
    cpa_new = _loguniform(rng, *cpa_range_m)
    k_new = float(rng.uniform(*spreading_range))
    along2 = np.clip(np.nan_to_num(r_ref, nan=cpa_ref) ** 2 - cpa_ref**2, 0.0, None)
    r_new = np.sqrt(cpa_new**2 + along2)
    r_ref_f = np.nan_to_num(r_ref, nan=cpa_ref)
    gain_db = transmission_loss_db(r_ref_f, spreading=ref.spreading) - transmission_loss_db(r_new, spreading=k_new)
    added = seg * db_to_lin(gain_db)
    info = {"cpa_m": cpa_new, "spreading": k_new, "peak_minute": peak_pos}
    return added, info


def mix_one(bg_db: np.ndarray, added_lin: np.ndarray, atb_db: float):
    """Scale ``added_lin`` to the target ATB (over observed cells) and add to background.

    Returns (mixture_db, scaled_added_lin, share).
    """
    observed = ~np.isnan(bg_db)
    bg_lin = db_to_lin(bg_db)
    a_sum = masked_sum(added_lin, observed)
    b_sum = masked_sum(bg_lin, observed)
    if a_sum <= 0 or b_sum <= 0:
        return bg_db.copy(), np.zeros_like(added_lin), 0.0
    scale = (b_sum * 10.0 ** (atb_db / 10.0)) / a_sum
    add = added_lin * scale
    mix_lin = np.where(observed, bg_lin + add, np.nan)
    share = float(share_from_parts(add, mix_lin, observed, axis=None))
    return lin_to_db(mix_lin), add, share


def make_mixtures(
    X_bg: np.ndarray,
    meta_bg: pd.DataFrame,
    refs: list[Reference],
    n: int,
    seed: int = config.SEED,
    atb_range_db=config.ATB_RANGE_DB,
    p_zero: float = config.P_ZERO_ADDITION,
    atb_sampler=None,
    **place_kwargs,
):
    """Build ``n`` labelled mixtures.

    ``atb_sampler(rng) -> float`` overrides the default uniform ATB draw, e.g. to
    match a target label distribution.

    Returns X_mix (n, 15, 25) dB, y_share (n,), y_present (n,), info DataFrame
    (carries station/day/hour of the background so splits stay valid).
    """
    rng = np.random.default_rng(seed)
    minutes = X_bg.shape[1]
    X = np.empty((n, minutes, X_bg.shape[2]))
    y = np.zeros(n)
    present = np.zeros(n, dtype=int)
    rows = []
    bg_idx = rng.integers(0, len(X_bg), size=n)
    for i, b in enumerate(bg_idx):
        bg = X_bg[b]
        row = {"bg_index": int(b), "vessel_id": None, "source": None, "atb_db": np.nan, "cpa_m": np.nan}
        if not refs or rng.random() < p_zero:
            X[i] = bg
        else:
            ref = refs[int(rng.integers(len(refs)))]
            added, info = place_reference(ref, rng, minutes=minutes, **place_kwargs)
            atb = float(atb_sampler(rng)) if atb_sampler else float(rng.uniform(*atb_range_db))
            X[i], _, y[i] = mix_one(bg, added, atb)
            present[i] = int(y[i] > 0)
            row.update(vessel_id=ref.vessel_id, source=ref.source, atb_db=atb, cpa_m=info["cpa_m"])
        rows.append(row)
    info = pd.DataFrame(rows)
    keep_cols = [c for c in ["station", "day", "hour", "start", "window_id"] if c in meta_bg.columns]
    info = pd.concat([meta_bg.iloc[bg_idx][keep_cols].reset_index(drop=True), info], axis=1)
    info["missing_frac"] = np.isnan(X).mean(axis=(1, 2))
    return X, y, present, info


def share_from_atb(atb_db):
    """Label implied by an ATB ratio when every cell is observed: r / (1 + r)."""
    r = 10.0 ** (np.asarray(atb_db, dtype=float) / 10.0)
    return r / (1.0 + r)


def atb_from_share(share):
    s = np.clip(np.asarray(share, dtype=float), 1e-6, 1 - 1e-6)
    return 10.0 * np.log10(s / (1.0 - s))


def share_matching_sampler(shares: np.ndarray):
    """ATB sampler that reproduces an empirical distribution of non-zero labels."""
    shares = np.asarray(shares, dtype=float)
    shares = shares[(shares > 0) & (shares < 1)]

    def _draw(rng):
        return float(atb_from_share(rng.choice(shares)))

    return _draw

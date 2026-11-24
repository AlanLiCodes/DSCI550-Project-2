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


def stretch_reference(lin: np.ndarray, range_m: np.ndarray, factor: float):
    """Time-stretch a minute-level passage by ``factor`` (>= 1) with linear interpolation.

    The organisers state that they varied the time trajectories of the VTUAD
    passages; strict VTUAD clips are short (median 3 min), so stretching lets a
    passage occupy more of a 15-minute window.
    """
    T = lin.shape[0]
    n = max(int(round(T * factor)), 1)
    if n == T:
        return lin, range_m
    src = (np.arange(n) + 0.5) / n * T - 0.5
    src = np.clip(src, 0, T - 1)
    out = np.stack([np.interp(src, np.arange(T), lin[:, k]) for k in range(lin.shape[1])], axis=1)
    rng_out = np.interp(src, np.arange(T), range_m)
    return out, rng_out


def place_reference(
    ref: Reference,
    rng: np.random.Generator,
    minutes: int = config.WINDOW_MINUTES,
    cpa_range_m=config.CPA_RANGE_M,
    spreading_range=config.SPREADING_RANGE,
    peak_margin: int = 3,
    stretch_range=(1.0, 1.0),
):
    """Optionally time-stretch ``ref``, crop it to ``minutes`` around its peak and move it to a new CPA distance.

    ``stretch_range=(lo, hi)``: stretch factor drawn log-uniformly, capped so the
    stretched passage is at most ``minutes`` long unless it already was longer.
    Returns (added_lin (minutes, 25), info dict).
    """
    lo, hi = stretch_range
    factor = 1.0
    if hi > 1.0:
        cap = max(minutes / max(ref.n_minutes, 1), 1.0)
        hi_eff = min(hi, cap)
        if hi_eff > lo:
            factor = _loguniform(rng, max(lo, 1.0), hi_eff)
    if factor != 1.0:
        lin_s, rng_s = stretch_reference(ref.lin, ref.range_m, factor)
        ref = Reference(ref.vessel_id, ref.source, lin_s, rng_s, ref.spreading, ref.group)
    broadband = np.nansum(ref.lin, axis=1)
    peak = int(np.nanargmax(broadband))
    # With a negative margin the peak may fall outside the window; redraw until some
    # passage energy is inside, so a placed passage never silently becomes a zero label.
    for _ in range(20):
        peak_pos = int(rng.integers(peak_margin, minutes - peak_margin))
        start = peak - peak_pos
        src = np.arange(start, start + minutes)
        ok = (src >= 0) & (src < ref.n_minutes)
        if ok.any() and broadband[src[ok]].sum() > 0:
            break
    else:  # fall back to a centred peak
        peak_pos = minutes // 2
        start = peak - peak_pos
        src = np.arange(start, start + minutes)
        ok = (src >= 0) & (src < ref.n_minutes)

    seg = np.zeros((minutes, config.N_BANDS))
    r_ref = np.full(minutes, np.nan)
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
    info = {"cpa_m": cpa_new, "spreading": k_new, "peak_minute": peak_pos, "stretch": factor}
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


def drop_minutes(bg_db: np.ndarray, rng: np.random.Generator, p: float, max_minutes: int) -> np.ndarray:
    """With probability ``p``, mask 1..max_minutes whole minutes (all bands) at random positions."""
    if p <= 0 or rng.random() >= p:
        return bg_db
    out = bg_db.copy()
    k = int(rng.integers(1, max_minutes + 1))
    out[rng.choice(out.shape[0], size=k, replace=False)] = np.nan
    return out


_LOG_F_REL = np.log10(config.BAND_CENTRES_HZ / 1000.0)  # decades relative to 1 kHz


def gain_tilt_db(rng: np.random.Generator, gain_db: float, tilt_db_per_decade: float, jitter_db: float = 0.0):
    """Random per-band dB offset: uniform global gain + uniform spectral tilt (pivot 1 kHz) + jitter."""
    g = rng.uniform(-gain_db, gain_db) if gain_db > 0 else 0.0
    t = rng.uniform(-tilt_db_per_decade, tilt_db_per_decade) if tilt_db_per_decade > 0 else 0.0
    off = g + t * _LOG_F_REL
    if jitter_db > 0:
        off = off + rng.normal(0.0, jitter_db, size=off.shape)
    return off


def add_busy_vessel(bg_db: np.ndarray, refs, rng, atb_range_db, **place_kwargs):
    """Add a vessel passage to a background *as background* (it is not part of the label).

    Real test backgrounds can already contain vessels; this teaches the model that a
    ship being present does not by itself mean energy was added.
    Returns (new background dB, atb used).
    """
    ref = refs[int(rng.integers(len(refs)))]
    added, _ = place_reference(ref, rng, minutes=bg_db.shape[0], **place_kwargs)
    atb = float(rng.uniform(*atb_range_db))
    busy_db, _, _ = mix_one(bg_db, added, atb)
    return busy_db, atb


def make_mixtures(
    X_bg: np.ndarray,
    meta_bg: pd.DataFrame,
    refs: list[Reference],
    n: int,
    seed: int = config.SEED,
    atb_range_db=config.ATB_RANGE_DB,
    p_zero: float = config.P_ZERO_ADDITION,
    atb_sampler=None,
    minute_dropout=(config.MINUTE_DROPOUT_P, config.MINUTE_DROPOUT_MAX),
    station_balance: float = 0.5,
    gain_db: float = 0.0,
    bg_tilt_db: float = 0.0,
    ref_tilt_db: float = 0.0,
    ref_jitter_db: float = 0.0,
    busy_p: float = 0.0,
    busy_atb_range_db=(-15.0, 5.0),
    busy_refs=None,
    bg_weights=None,
    return_cells: bool = False,
    edge_dropout=(0.0, 0.0),
    **place_kwargs,
):
    """Build ``n`` labelled mixtures.

    ``atb_sampler(rng) -> float`` overrides the default uniform ATB draw, e.g. to
    match a target label distribution.

    ``minute_dropout=(p, k)`` masks up to ``k`` whole minutes in a fraction ``p`` of
    backgrounds *before* mixing, so labels are computed over the same observed cells.

    ``station_balance`` in [0, 1] tempers station imbalance when drawing backgrounds:
    window weight = count(station) ** -station_balance (0 = uniform over windows,
    1 = uniform over stations). Richards Bay holds ~65% of released windows.

    Loudness augmentation (all default off):
    * ``gain_db`` / ``bg_tilt_db``: random global gain (+-dB) and spectral tilt
      (+-dB per decade, pivot 1 kHz) applied to the *background* before mixing, so
      absolute level and spectral balance stop being a proxy for added vessel.
      Because the addition is scaled relative to the background, labels stay exact.
    * ``ref_tilt_db`` / ``ref_jitter_db``: random channel tilt and per-band jitter
      applied to the placed passage ("relative acoustic channel ... varied").

    Busy-background augmentation: with probability ``busy_p`` a second passage
    (from ``busy_refs``, default ``refs``) is added to the background at an ATB
    drawn from ``busy_atb_range_db`` and counted as *background* (label 0 part).

    Returns X_mix (n, 15, 25) dB, y_share (n,), y_present (n,), info DataFrame
    (carries station/day/hour of the background so splits stay valid).

    ``bg_weights`` (one per background window) multiplies the sampling weight of each
    background, e.g. to over-sample hard negatives (see pipeline.hard_negative_weights).

    ``edge_dropout=(p_first, p_last)`` masks the first / last minute of the window
    with those probabilities (in addition to ``minute_dropout``). Released training
    windows and Test.csv windows only ever miss edge minutes, never middle ones.

    ``peak_margin`` (passed to place_reference) sets where a passage may peak: minutes
    [margin, 15 - margin). A negative margin lets the peak fall outside the window, so
    only the rise or the tail of a passage is inside it.

    ``return_cells=True`` appends a fifth output: the per-cell added-energy fraction
    (n, 15, 25), added / mixture in each observed cell (NaN where unobserved). The
    window label equals the mixture-energy-weighted mean of these fractions.
    """
    busy_refs = busy_refs if busy_refs is not None else refs
    rng = np.random.default_rng(seed)
    minutes = X_bg.shape[1]
    X = np.empty((n, minutes, X_bg.shape[2]))
    y = np.zeros(n)
    present = np.zeros(n, dtype=int)
    cells = np.zeros((n, minutes, X_bg.shape[2]), dtype=np.float32) if return_cells else None
    rows = []
    w = np.ones(len(X_bg))
    if station_balance and "station" in meta_bg:
        counts = meta_bg["station"].map(meta_bg["station"].value_counts()).to_numpy(dtype=float)
        w = w * counts ** (-float(station_balance))
    if bg_weights is not None:
        w = w * np.asarray(bg_weights, dtype=float)
    if np.allclose(w, w[0]):
        bg_idx = rng.integers(0, len(X_bg), size=n)
    else:
        bg_idx = rng.choice(len(X_bg), size=n, p=w / w.sum())
    for i, b in enumerate(bg_idx):
        bg = drop_minutes(np.asarray(X_bg[b], dtype=float), rng, *minute_dropout)
        if edge_dropout[0] > 0 and rng.random() < edge_dropout[0]:
            bg[0] = np.nan
        if edge_dropout[1] > 0 and rng.random() < edge_dropout[1]:
            bg[-1] = np.nan
        row = {"bg_index": int(b), "vessel_id": None, "source": None, "atb_db": np.nan, "cpa_m": np.nan, "busy": 0}
        if gain_db > 0 or bg_tilt_db > 0:
            bg = bg + gain_tilt_db(rng, gain_db, bg_tilt_db)[None, :]
        if busy_refs and busy_p > 0 and rng.random() < busy_p:
            bg, _ = add_busy_vessel(bg, busy_refs, rng, busy_atb_range_db, **place_kwargs)
            row["busy"] = 1
        if not refs or rng.random() < p_zero:
            X[i] = bg
        else:
            ref = refs[int(rng.integers(len(refs)))]
            added, info = place_reference(ref, rng, minutes=minutes, **place_kwargs)
            if ref_tilt_db > 0 or ref_jitter_db > 0:
                added = added * db_to_lin(gain_tilt_db(rng, 0.0, ref_tilt_db, ref_jitter_db))[None, :]
            atb = float(atb_sampler(rng)) if atb_sampler else float(rng.uniform(*atb_range_db))
            X[i], add_i, y[i] = mix_one(bg, added, atb)
            if return_cells:
                with np.errstate(invalid="ignore", divide="ignore"):
                    cells[i] = add_i / db_to_lin(X[i])
            present[i] = int(y[i] > 0)
            row.update(vessel_id=ref.vessel_id, source=ref.source, atb_db=atb, cpa_m=info["cpa_m"])
        rows.append(row)
    info = pd.DataFrame(rows)
    keep_cols = [c for c in ["station", "day", "hour", "start", "window_id"] if c in meta_bg.columns]
    info = pd.concat([meta_bg.iloc[bg_idx][keep_cols].reset_index(drop=True), info], axis=1)
    info["missing_frac"] = np.isnan(X).mean(axis=(1, 2))
    if return_cells:
        cells[np.isnan(X)] = np.nan
        return X, y, present, info, cells
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

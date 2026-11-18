"""Window-level features, grouped into families for ablations.

Families
--------
level          energy-mean band levels and broadband level
variability    per-band and broadband minute-to-minute spread
station_norm   band levels relative to the station's training median
spectral       wind proxy (0.63-2 kHz), low/high-frequency balance, spectral slope
temporal       passage shape: peak minus median, rise, fall, peak position, max step
missing        missingness of the window
time           hour of day (cyclic), since biological choruses follow daily cycles
excess         excess-energy baseline estimate

Missing values are left as NaN here; median imputation + missingness
indicators are fitted on training data inside the model pipelines.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .baseline import ExcessEnergyBaseline
from .energy import db_to_lin, lin_to_db

FAMILIES = ["level", "variability", "station_norm", "spectral", "temporal", "missing", "time", "excess"]

_F = config.BAND_CENTRES_HZ
_WIND = (_F >= config.WIND_BAND_HZ[0] * 0.95) & (_F <= config.WIND_BAND_HZ[1] * 1.05)
_LOW = _F <= config.LOW_BAND_MAX_HZ
_HIGH = _F >= config.HIGH_BAND_MIN_HZ


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _group_series_db(lin, mask):
    """Per-minute energy-mean level over a band group -> (N, T) dB."""
    return lin_to_db(_nanmean(lin[:, :, mask], axis=2))


def _shape_features(series_db, prefix):
    """Passage-shape descriptors of a per-minute level series (N, T)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak = np.nanmax(series_db, axis=1)
        med = np.nanmedian(series_db, axis=1)
        T = series_db.shape[1]
        filled = pd.DataFrame(series_db.T).ffill().bfill().T.to_numpy()
        first, last = filled[:, 0], filled[:, -1]
        pos = np.argmax(np.nan_to_num(series_db, nan=-np.inf), axis=1) / (T - 1)
        step = np.nanmax(np.abs(np.diff(series_db, axis=1)), axis=1)
        q10 = np.nanpercentile(series_db, 10, axis=1)
    return {
        f"{prefix}_peak_minus_median": peak - med,
        f"{prefix}_peak_minus_q10": peak - q10,
        f"{prefix}_rise": peak - first,
        f"{prefix}_fall": peak - last,
        f"{prefix}_peak_pos": pos,
        f"{prefix}_peak_central": 1.0 - 2.0 * np.abs(pos - 0.5),
        f"{prefix}_max_step": step,
    }


class FeatureBuilder:
    """Fit station statistics and the excess baseline on training backgrounds only."""

    def __init__(self, baseline: ExcessEnergyBaseline | None = None):
        self.baseline = baseline or ExcessEnergyBaseline()

    def fit(self, X_bg: np.ndarray, meta_bg: pd.DataFrame, X_tune=None, y_tune=None, meta_tune=None):
        lin = db_to_lin(X_bg)
        self.global_median_db_ = lin_to_db(np.nanmedian(lin.reshape(-1, lin.shape[2]), axis=0))
        self.station_median_db_ = {}
        for s in meta_bg["station"].unique():
            m = (meta_bg["station"] == s).to_numpy()
            self.station_median_db_[s] = lin_to_db(np.nanmedian(lin[m].reshape(-1, lin.shape[2]), axis=0))
        self.baseline.fit(X_bg, meta_bg)
        if X_tune is not None:
            self.baseline.tune(X_tune, y_tune, meta_tune)
        return self

    def transform(self, X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        lin = db_to_lin(X)
        N, T, B = X.shape
        feats: dict[str, np.ndarray] = {}
        fam: dict[str, str] = {}

        def add(family, d):
            for k, v in d.items():
                feats[k] = np.asarray(v, dtype=float)
                fam[k] = family

        mean_db = lin_to_db(_nanmean(lin, axis=1))  # (N, B)
        bb_series = lin_to_db(_nanmean(lin, axis=2))  # (N, T)
        bb_db = lin_to_db(_nanmean(lin, axis=(1, 2)))

        add("level", {f"lvl_b{k:02d}": mean_db[:, k] for k in range(B)} | {"lvl_broadband": bb_db})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            band_std = np.nanstd(X, axis=1)
            add(
                "variability",
                {f"std_b{k:02d}": band_std[:, k] for k in range(B)}
                | {"std_broadband": np.nanstd(bb_series, axis=1), "std_band_mean": np.nanmean(band_std, axis=1)},
            )

        ref = np.stack([self.station_median_db_.get(s, self.global_median_db_) for s in meta["station"]])
        rel = mean_db - ref
        add(
            "station_norm",
            {f"rel_b{k:02d}": rel[:, k] for k in range(B)}
            | {"rel_broadband": _nanmean(rel, axis=1), "rel_low": _nanmean(rel[:, _LOW], axis=1)},
        )

        low_db = lin_to_db(_nanmean(lin[:, :, _LOW], axis=(1, 2)))
        high_db = lin_to_db(_nanmean(lin[:, :, _HIGH], axis=(1, 2)))
        wind_db = lin_to_db(_nanmean(lin[:, :, _WIND], axis=(1, 2)))
        logf = np.log10(_F)
        xc = logf - logf.mean()
        filled = np.where(np.isnan(mean_db), _nanmean(mean_db, axis=1)[:, None], mean_db)
        slope = (filled - filled.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
        add(
            "spectral",
            {"wind_proxy": wind_db, "low_db": low_db, "high_db": high_db, "low_minus_high": low_db - high_db, "spectral_slope": slope},
        )

        add("temporal", _shape_features(bb_series, "bb"))
        add("temporal", _shape_features(_group_series_db(lin, _LOW), "low"))
        add("temporal", _shape_features(_group_series_db(lin, _HIGH), "high"))

        minute_missing = np.isnan(X).all(axis=2)
        add("missing", {"missing_frac": np.isnan(X).mean(axis=(1, 2)), "missing_minutes": minute_missing.sum(axis=1)})

        hour = meta["hour"].to_numpy(dtype=float) if "hour" in meta else np.full(N, np.nan)
        add("time", {"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24)})

        add("excess", {"excess_share": self.baseline.predict(X, meta)})

        df = pd.DataFrame(feats)
        df.attrs["families"] = fam
        return df


def columns_for(df: pd.DataFrame, drop_families=()) -> list[str]:
    fam = df.attrs.get("families", {})
    return [c for c in df.columns if fam.get(c) not in set(drop_families)]

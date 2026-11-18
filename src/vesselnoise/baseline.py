"""Label-free excess-energy baseline (after Merchant et al., 2012).

Per band, a background level is estimated from
  (a) the window's own quietest minutes, and
  (b) station x hour-of-day percentiles of the *training* background windows,
and the energy above it (optionally above a margin) is reported as a share of
the window's total observed energy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .energy import db_to_lin, masked_sum


def quiet_minutes_background(lin: np.ndarray, k: int = 3):
    """Mean of the ``k`` quietest observed minutes per band. lin: (N, T, B) -> (N, B)."""
    s = np.sort(np.where(np.isnan(lin), np.inf, lin), axis=1)[:, :k, :]
    s = np.where(np.isinf(s), np.nan, s)
    with np.errstate(invalid="ignore"):
        return np.nanmean(s, axis=1)


class ExcessEnergyBaseline:
    def __init__(self, k_quiet: int = 3, prior_quantile: float = 0.5, threshold_db: float = 0.0, combine: str = "min"):
        self.k_quiet = k_quiet
        self.prior_quantile = prior_quantile
        self.threshold_db = threshold_db
        self.combine = combine
        self.prior_: dict = {}

    # -- priors from training backgrounds ---------------------------------
    def fit(self, X_bg: np.ndarray, meta: pd.DataFrame):
        lin = db_to_lin(X_bg)  # (N, T, B)
        st = meta["station"].to_numpy()
        hr = meta["hour"].to_numpy() if "hour" in meta else np.zeros(len(meta))
        q = self.prior_quantile
        self.global_ = np.nanquantile(lin.reshape(-1, lin.shape[2]), q, axis=0)
        self.station_ = {}
        self.station_hour_ = {}
        for s in np.unique(st):
            m = st == s
            self.station_[s] = np.nanquantile(lin[m].reshape(-1, lin.shape[2]), q, axis=0)
            for h in np.unique(hr[m]):
                mh = m & (hr == h)
                if mh.sum() >= 4:
                    self.station_hour_[(s, h)] = np.nanquantile(lin[mh].reshape(-1, lin.shape[2]), q, axis=0)
        return self

    def _prior(self, meta: pd.DataFrame):
        out = np.empty((len(meta), len(self.global_)))
        st = meta["station"].to_numpy()
        hr = meta["hour"].to_numpy() if "hour" in meta else np.full(len(meta), -1)
        for i, (s, h) in enumerate(zip(st, hr)):
            out[i] = self.station_hour_.get((s, h), self.station_.get(s, self.global_))
        return out

    def background(self, X: np.ndarray, meta: pd.DataFrame | None = None):
        lin = db_to_lin(X)
        quiet = quiet_minutes_background(lin, self.k_quiet)
        if meta is None or not hasattr(self, "global_") or self.combine == "window":
            return quiet
        prior = self._prior(meta)
        if self.combine == "min":
            bg = np.fmin(quiet, prior)
        elif self.combine == "prior":
            bg = prior
        else:  # geometric mean
            bg = np.sqrt(quiet * prior)
        return np.where(np.isnan(bg), prior, bg)

    def predict(self, X: np.ndarray, meta: pd.DataFrame | None = None):
        lin = db_to_lin(X)
        bg = self.background(X, meta)[:, None, :]
        thr = 10.0 ** (self.threshold_db / 10.0)
        excess = np.where(lin > bg * thr, lin - bg, 0.0)
        num = masked_sum(excess, axis=(1, 2))
        den = masked_sum(lin, axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(den > 0, num / den, 0.0)
        return np.clip(share, 0.0, 1.0)

    def tune(self, X, y, meta=None, thresholds=(0, 1, 2, 3, 4, 6, 8), k_values=(1, 2, 3, 5)):
        """Pick threshold / k on validation data by RMSE. Returns self."""
        best = (np.inf, self.threshold_db, self.k_quiet)
        for k in k_values:
            for t in thresholds:
                self.k_quiet, self.threshold_db = k, t
                rmse = float(np.sqrt(np.mean((self.predict(X, meta) - y) ** 2)))
                if rmse < best[0]:
                    best = (rmse, t, k)
        _, self.threshold_db, self.k_quiet = best
        return self

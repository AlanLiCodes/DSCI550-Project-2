import numpy as np
import pandas as pd
import pytest

from vesselnoise.mixtures import make_mixtures
from vesselnoise.reference import synthetic_reference


def _pool(n=120, seed=0, noisy_station=None):
    rng = np.random.default_rng(seed)
    X = 80 + rng.normal(0, 1.0, (n, 15, 25))
    st = np.where(np.arange(n) % 4 == 0, "B", "A")
    if noisy_station:
        # station B: large natural minute-to-minute swings in low bands
        X[st == "B", :, :8] += rng.normal(0, 8, ((st == "B").sum(), 15, 1))
    days = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(n) % 10, unit="D")
    return X, pd.DataFrame({"station": st, "day": days, "hour": 0})


def _refs():
    return [synthetic_reference(np.random.default_rng(i), f"v{i}", "large") for i in range(4)]


def test_cells_reconstruct_label():
    X, meta = _pool()
    X[:10, 0] = np.nan
    Xm, y, _, _, cells = make_mixtures(X, meta, _refs(), 150, seed=1, return_cells=True, busy_p=0.5, gain_db=5)
    lin = np.where(np.isnan(Xm), 0, 10 ** (Xm / 10))
    share = (np.nan_to_num(cells) * lin).sum(axis=(1, 2)) / lin.sum(axis=(1, 2))
    assert np.allclose(share, y, atol=1e-6)
    assert (np.isnan(cells) == np.isnan(Xm)).all()


def test_bg_weights_shift_sampling():
    X, meta = _pool()
    w = np.where(meta["station"] == "B", 20.0, 1.0)
    _, _, _, info = make_mixtures(X, meta, _refs(), 400, seed=2, bg_weights=w)
    assert (info["station"] == "B").mean() > 0.7  # B is 25% of windows but weighted 20x


def test_hard_negative_weights_upweight_noisy_station():
    from vesselnoise.pipeline import hard_negative_weights

    X, meta = _pool(200, noisy_station=True)
    w, oof = hard_negative_weights(X, meta, _refs(), {"p_zero": 0.3}, strength=10, n=1500, folds=2)
    assert w.shape == (len(X),) and (w >= 1).all()
    assert oof[meta["station"] == "B"].mean() > oof[meta["station"] == "A"].mean()


def test_window_channels_and_energy_weights():
    from vesselnoise.cnn import energy_weights, window_channels

    X = 80 + np.random.default_rng(0).normal(0, 2, (4, 15, 25))
    X[0, 0] = np.nan
    ch = window_channels(X)
    assert ch.shape == (4, 5, 15, 25) and np.isfinite(ch).all()
    assert (ch[0, 3, 0] == 0).all() and (ch[0, :, 0] == 0).all()
    w = energy_weights(X)
    assert np.allclose(w.sum(axis=(1, 2)), 1.0) and (w[0, 0] == 0).all()


def test_cnn_learns_simple_mixtures():
    pytest.importorskip("torch")
    from vesselnoise.cnn import CellCNN

    X, meta = _pool(200)
    Xm, y, _, _, cells = make_mixtures(X, meta, _refs(), 1200, seed=3, return_cells=True, atb_range_db=(-20, 5))
    m = CellCNN(epochs=6, width=16, verbose=False).fit(Xm[:1000], y[:1000], cells[:1000])
    p, pc = m.predict(Xm[1000:], return_cells=True)
    assert pc.shape == (200, 15, 25) and ((p >= 0) & (p <= 1)).all()
    assert np.sqrt(np.mean((p - y[1000:]) ** 2)) < 0.5 * y[1000:].std()

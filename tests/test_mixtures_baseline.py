import numpy as np
import pandas as pd
import pytest

from vesselnoise import config
from vesselnoise.baseline import ExcessEnergyBaseline
from vesselnoise.energy import db_to_lin
from vesselnoise.mixtures import make_mixtures, mix_one, place_reference, share_from_atb
from vesselnoise.propagation import thorp_absorption_db_per_km, transmission_loss_db
from vesselnoise.reference import audio_to_minute_bands, reference_from_bands, synthetic_reference


def test_thorp_increases_with_frequency():
    a = thorp_absorption_db_per_km(config.BAND_CENTRES_HZ)
    assert np.all(np.diff(a) > 0)
    tl = transmission_loss_db([100.0, 1000.0])
    assert np.all(tl[1] > tl[0])


def test_audio_tone_lands_in_its_band():
    fs = 24000
    t = np.arange(fs * 120) / fs
    x = np.sin(2 * np.pi * 1000.0 * t)
    bands = audio_to_minute_bands(x, fs)
    assert bands.shape == (2, 25)
    k = int(np.argmin(np.abs(config.BAND_CENTRES_HZ - 1000)))
    assert np.nanargmax(bands[0]) == k
    # Parseval: band energy ~ mean square of the sine (0.5)
    assert np.nansum(bands[0]) == pytest.approx(0.5, rel=0.05)


def test_audio_bands_above_nyquist_are_nan():
    fs = 16000
    bands = audio_to_minute_bands(np.random.default_rng(0).normal(size=fs * 60), fs)
    assert np.isnan(bands[0, -1]) and np.isfinite(bands[0, :-1]).all()


def test_mix_one_hits_target_atb_label():
    rng = np.random.default_rng(0)
    bg = 80 + rng.normal(0, 2, (15, 25))
    added = db_to_lin(70 + rng.normal(0, 2, (15, 25)))
    for atb in [-20, -5, 0, 7, 20]:
        _, _, share = mix_one(bg, added, atb)
        assert share == pytest.approx(share_from_atb(atb), abs=1e-9)


def test_label_uses_only_observed_cells():
    bg = np.full((15, 25), 80.0)
    bg[:5] = np.nan
    added = np.zeros((15, 25))
    added[:5] = 1e12  # all added energy sits in missing minutes
    mix_db, _, share = mix_one(bg, added, 0.0)
    assert share == 0.0
    assert np.isnan(mix_db[:5]).all()


def test_place_reference_keeps_peak_inside_window():
    rng = np.random.default_rng(1)
    ref = synthetic_reference(rng, "v", "large", n_minutes=40)
    for _ in range(20):
        added, info = place_reference(ref, rng)
        peak = int(np.argmax(added.sum(axis=1)))
        assert 3 <= peak <= 11
        assert added.shape == (15, 25) and np.all(added >= 0)


def test_far_passage_loses_high_frequencies():
    rng = np.random.default_rng(2)
    ref = synthetic_reference(rng, "v", "small")
    near, _ = place_reference(ref, np.random.default_rng(3), cpa_range_m=(50, 50.0001))
    far, _ = place_reference(ref, np.random.default_rng(3), cpa_range_m=(5000, 5000.0001))
    tilt = lambda a: np.log10(a[:, -5:].sum() / a[:, :5].sum())
    assert tilt(far) < tilt(near)


def test_reference_from_bands_subtracts_background():
    lin = np.ones((20, 25))
    lin[8:12] += 100.0
    ref = reference_from_bands("x", "test", lin)
    assert ref.lin[0].sum() == pytest.approx(0.0)
    assert ref.range_m.argmin() in (8, 9, 10, 11)


def _bg_pool(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = 80 + rng.normal(0, 1.0, (n, 15, 25))
    meta = pd.DataFrame({"station": rng.choice(["A", "B"], n), "hour": rng.integers(0, 24, n)})
    meta["day"] = pd.Timestamp("2025-01-01")
    return X, meta


def test_make_mixtures_shapes_and_zero_fraction():
    X, meta = _bg_pool()
    refs = [synthetic_reference(np.random.default_rng(i), f"v{i}", "small") for i in range(5)]
    Xm, y, present, info = make_mixtures(X, meta, refs, 500, seed=0, p_zero=0.3)
    assert Xm.shape == (500, 15, 25)
    assert 0.2 < (y == 0).mean() < 0.4
    assert set(info.columns) >= {"station", "hour", "vessel_id", "atb_db", "cpa_m"}
    assert np.allclose(y[present == 1], share_from_atb(info["atb_db"][present == 1]), atol=1e-6)


def test_excess_baseline_ranks_additions():
    X, meta = _bg_pool()
    refs = [synthetic_reference(np.random.default_rng(i), f"v{i}", "large") for i in range(5)]
    Xm, y, _, info = make_mixtures(X, meta, refs, 400, seed=1)
    bl = ExcessEnergyBaseline().fit(X, meta).tune(Xm, y, info)
    pred = bl.predict(Xm, info)
    assert np.corrcoef(pred, y)[0, 1] > 0.6
    assert pred[y == 0].mean() < 0.1


def test_stretch_reference_preserves_shape_and_peak():
    from vesselnoise.mixtures import stretch_reference

    lin = np.zeros((3, 25))
    lin[1] = 10.0
    out, r = stretch_reference(lin, np.array([300.0, 100.0, 300.0]), 3.0)
    assert out.shape == (9, 25) and r.shape == (9,)
    assert np.argmax(out[:, 0]) == 4 and out.max() == pytest.approx(10.0)


def test_place_reference_stretch_spreads_short_passage():
    from vesselnoise.reference import Reference

    rng = np.random.default_rng(0)
    lin = np.zeros((3, 25))
    lin[1] = 1.0
    ref = Reference("v", "vtuad_tug", lin, np.array([500.0, 100.0, 500.0]))
    widths = []
    for _ in range(30):
        added, info = place_reference(ref, rng, stretch_range=(1.0, 5.0), cpa_range_m=(100, 100.0001))
        widths.append((added.sum(axis=1) > 0).sum())
        assert 1.0 <= info["stretch"] <= 5.0
    assert max(widths) > 3

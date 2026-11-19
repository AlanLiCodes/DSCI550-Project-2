import numpy as np
import pandas as pd
import pytest

from vesselnoise.data import standardise
from vesselnoise.features import FAMILIES, FeatureBuilder, columns_for
from vesselnoise.metrics import block_bootstrap_ci, by_true_fraction, leaderboard_score, regression_report
from vesselnoise.pipeline import build_experiment, evaluate_models
from vesselnoise.reference import synthetic_reference_set
from vesselnoise.splits import assign_reference_splits, chronological_split
from vesselnoise.synth import synthetic_records
from vesselnoise.windows import build_windows


@pytest.fixture(scope="module")
def pool():
    std, _ = standardise(synthetic_records(n_stations=3, n_days=10, seed=3))
    return build_windows(std)


def test_leaderboard_metric():
    y = np.array([0.0, 0.5, 1.0])
    assert leaderboard_score(y, y) == 1.0
    assert leaderboard_score(y, y + 0.1) == pytest.approx(0.9)
    r = regression_report(y, y + 0.1)
    assert r["bias"] == pytest.approx(0.1) and r["mae"] == pytest.approx(0.1)
    assert by_true_fraction(y, y)["n"].sum() == 3
    lo, hi = block_bootstrap_ci(y, y + 0.1, ["a", "b", "c"])
    assert lo == pytest.approx(0.9) and hi == pytest.approx(0.9)


def test_chronological_split_is_ordered_with_buffer(pool):
    X, meta = pool
    lab = chronological_split(meta, buffer_days=1)
    for s, g in meta.assign(lab=lab).groupby("station"):
        days = {k: set(pd.to_datetime(v["day"])) for k, v in g.groupby("lab")}
        assert max(days["train"]) < min(days["val"]) < min(days["test"])
        # every day belongs to exactly one split label
        assert sum(len(v) for v in days.values()) == g["day"].nunique()
        assert "buffer" in days


def test_reference_vessels_in_one_split_only():
    refs = synthetic_reference_set(10, 10, seed=0)
    sp = assign_reference_splits(refs)
    assert set(sp) == {r.vessel_id for r in refs}
    assert set(sp.values()) == {"train", "val", "test"}


def test_features_families_and_no_leakage(pool):
    X, meta = pool
    fb = FeatureBuilder().fit(X[:200], meta.iloc[:200].reset_index(drop=True))
    F = fb.transform(X[200:260], meta.iloc[200:260].reset_index(drop=True))
    fams = set(F.attrs["families"].values())
    assert fams == set(FAMILIES)
    assert len(columns_for(F, ["temporal"])) < F.shape[1]
    assert np.isfinite(F["lvl_broadband"]).all()


def test_experiment_end_to_end_beats_constant(pool):
    X, meta = pool
    refs = synthetic_reference_set(15, 10, seed=1)
    exp = build_experiment(X, meta, refs, {"train": 1500, "val": 400, "test": 400})
    # backgrounds and vessels never shared across splits
    for a in ["train", "val"]:
        for b in ["val", "test"]:
            if a != b:
                va = set(exp.splits[a].info["vessel_id"].dropna())
                vb = set(exp.splits[b].info["vessel_id"].dropna())
                assert not va & vb
    res, _, _ = evaluate_models(exp, fast=True, with_ci=False)
    rmse = res[res["split"] == "test"].set_index("model")["rmse"]
    # at least halve the constant model's error (independent of the label prior's spread)
    assert rmse["hist_gb"] < 0.5 * rmse["constant"]
    assert test["excess_baseline"] > test["constant"]

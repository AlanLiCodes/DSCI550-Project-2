#!/usr/bin/env python
"""How distinguishable are training mixtures from Test.csv windows, for several simulation designs?

Adversarial validation on Test.csv *inputs* only (no labels): a classifier tries to tell
mixtures from test windows; lower AUC = the simulation looks more like the test set.
Also reports the label mean (must stay near 0.197 to keep the benchmark calibration),
missing-minute pattern and passage peak positions. Paths are local; edit them.

python scripts/compare_simulation_designs.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from vesselnoise.data import load_records, load_test  # noqa: E402
from vesselnoise.features import FeatureBuilder  # noqa: E402
from vesselnoise.mixtures import make_mixtures  # noqa: E402
from vesselnoise.pipeline import hard_negative_weights  # noqa: E402
from vesselnoise.splits import chronological_split  # noqa: E402
from vesselnoise.vtuad import load_vtuad_references  # noqa: E402
from vesselnoise.windows import build_windows, mask_run_start_transients  # noqa: E402

RECORDS, TEST, REFS = "data/South_African_Real_Acoustics", "data/Test.csv", "data/references/vtuad"

V6 = dict(station_balance=0, atb_range_db=(-20, 5), gain_db=10, bg_tilt_db=6, ref_tilt_db=6, ref_jitter_db=1,
          busy_p=0.4, busy_atb_range_db=(-15, 5))
EDGE = dict(minute_dropout=(0.0, 2), edge_dropout=(0.4, 0.07))
DESIGNS = {
    "v6": V6,
    "A: edge-only missing minutes": {**V6, **EDGE},
    "A + peak anywhere in window": {**V6, **EDGE, "peak_margin": 0},
    "v7: A + peak anywhere + no background gain/tilt": {**V6, **EDGE, "peak_margin": 0, "gain_db": 0, "bg_tilt_db": 0},
}


def adversarial_auc(Fm, Ft, drop=()):
    A = pd.concat([Fm.drop(columns=list(drop)), Ft.drop(columns=list(drop))]).reset_index(drop=True)
    lab = np.r_[np.zeros(len(Fm)), np.ones(len(Ft))]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=150, random_state=0), A, lab, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(lab, p)


def main():
    rec, _ = load_records(RECORDS)
    rec, _ = mask_run_start_transients(rec)
    X, m = build_windows(rec)
    keep = chronological_split(m) != "buffer"
    X, m = X[keep], m[keep].reset_index(drop=True)
    Xt, mt, _ = load_test(TEST)
    refs = load_vtuad_references(REFS, background="none")
    hnw, _ = hard_negative_weights(X, m, refs, V6, strength=30, n=12000)
    fb = FeatureBuilder(station_aware=False).fit(X, m)
    Ft = fb.transform(Xt, mt)
    miss_cols = [c for c in Ft.columns if "missing" in c]
    rows = []
    for name, kw in DESIGNS.items():
        Xm, y, _, info = make_mixtures(X, m, refs, 6000, seed=5, bg_weights=hnw, **kw)
        Fm = fb.transform(Xm, info)
        rows.append({
            "design": name,
            "auc_all": round(adversarial_auc(Fm, Ft), 3),
            "auc_no_missing": round(adversarial_auc(Fm, Ft, miss_cols), 3),
            "label_mean": round(float(y.mean()), 3),
            "minute1_missing": round(float(np.isnan(Xm).all(axis=2)[:, 0].mean()), 3),
            "middle_minutes_missing": round(float(np.isnan(Xm).all(axis=2)[:, 1:-1].mean()), 3),
        })
        print(rows[-1], flush=True)
    print(f"test: minute1_missing {np.isnan(Xt).all(axis=2)[:, 0].mean():.3f}, middle {np.isnan(Xt).all(axis=2)[:, 1:-1].mean():.3f}")
    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("outputs/simulation_designs.csv", index=False)


if __name__ == "__main__":
    main()

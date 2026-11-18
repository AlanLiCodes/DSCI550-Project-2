#!/usr/bin/env python
"""Data-dictionary check: confirm column detection, units, resolution, gaps and missingness.

python scripts/inspect_data.py --records data/South_African_Real_Acoustics --test data/Test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vesselnoise import config  # noqa: E402
from vesselnoise.data import BAND_COLS, load_records, load_test  # noqa: E402
from vesselnoise.windows import build_windows  # noqa: E402


def report_records(path, nrows):
    std, sch = load_records(path, nrows=nrows)
    print(f"== {path}: {len(std):,} recorder-minutes after de-duplication")
    print("\n-- detected schema\n" + sch.describe())
    print("\n-- band centres (Hz) vs columns")
    for c, f in zip(sch.band_cols, config.BAND_CENTRES_HZ):
        print(f"  {f:8.1f}  {c}")
    lv = std[BAND_COLS]
    print(f"\n-- level range: {np.nanmin(lv.values):.1f} .. {np.nanmax(lv.values):.1f} dB, median {np.nanmedian(lv.values):.1f}")
    print(f"-- missing cells: {lv.isna().mean().mean():.3%}; rows with all bands missing: {lv.isna().all(axis=1).mean():.3%}")
    if sch.duration:
        print(f"-- rows with {sch.duration} < {config.MIN_DURATION_S:.0f} s and out-of-range sentinels are masked (counted above)")
    if "timestamp" in std:
        keys = [c for c in ["station", "recorder"] if c in std]
        st = std.sort_values([*keys, "timestamp"])
        step = st.groupby(keys)["timestamp"].diff().dt.total_seconds().div(60)
        print("\n-- time step (minutes) distribution:\n" + step.value_counts().head(10).to_string())
        per = st.groupby("station").agg(rows=("timestamp", "size"), first=("timestamp", "min"), last=("timestamp", "max"))
        per["gaps_gt_1.5min"] = step.gt(config.MAX_STEP_MINUTES).groupby(st["station"]).sum()
        per["missing_cell_frac"] = st[BAND_COLS].isna().mean(axis=1).groupby(st["station"]).mean()
        print("\n-- per station\n" + per.to_string())
        X, meta = build_windows(std)
        per_w = meta.groupby("station").agg(windows=("station", "size"), days=("day", "nunique"), mean_observed_minutes=("observed_minutes", "mean"))
        print(f"\n-- {len(X):,} 15-min windows (runs >= {config.MIN_RUN_MINUTES} min); per station:\n" + per_w.round(2).to_string())
        return X


def report_test(path, X_bg=None):
    X, meta, id_col = load_test(path)
    M = np.isnan(X)
    print(f"\n== {path}: {len(X)} windows (ID column '{id_col}'), station column: {'station' in meta}")
    print(f"-- missing cells {M.mean():.2%}; windows with any missing {M.any(axis=(1, 2)).mean():.1%}")
    print(f"-- whole-minute missing share of missing cells: {M.all(axis=2).sum() * X.shape[2] / max(M.sum(), 1):.1%}")
    print("-- missing share by minute position: " + " ".join(f"{v:.3f}" for v in M.all(axis=2).mean(axis=0)))
    print("-- band median levels (dB): " + " ".join(f"{v:.1f}" for v in np.nanmedian(X, axis=(0, 1))))
    if X_bg is not None:
        print("-- train  median levels (dB): " + " ".join(f"{v:.1f}" for v in np.nanmedian(X_bg, axis=(0, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records")
    ap.add_argument("--test")
    ap.add_argument("--nrows", type=int, default=None)
    a = ap.parse_args()
    X_bg = report_records(a.records, a.nrows) if a.records else None
    if a.test:
        report_test(a.test, X_bg)


if __name__ == "__main__":
    main()

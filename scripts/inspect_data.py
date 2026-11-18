#!/usr/bin/env python
"""Data-dictionary check: confirm column detection, units, resolution, gaps and missingness.

python scripts/inspect_data.py --records data/Train.csv --test data/Test.csv [--nrows 200000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vesselnoise import config  # noqa: E402
from vesselnoise.data import BAND_COLS, standardise  # noqa: E402
from vesselnoise.windows import build_windows, windows_from_long  # noqa: E402


def report_records(path, nrows):
    raw = pd.read_csv(path, nrows=nrows)
    print(f"== {path}: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    print(raw.dtypes.to_string())
    std, sch = standardise(raw)
    print("\n-- detected schema\n" + sch.describe())
    print("\n-- band centres (Hz) vs columns")
    for c, f in zip(sch.band_cols, config.BAND_CENTRES_HZ):
        print(f"  {f:8.1f}  {c}")
    lv = std[BAND_COLS]
    print(f"\n-- level range: {np.nanmin(lv.values):.1f} .. {np.nanmax(lv.values):.1f} dB, median {np.nanmedian(lv.values):.1f}")
    print(f"-- missing cells: {lv.isna().mean().mean():.3%}; rows with all bands missing: {lv.isna().all(axis=1).mean():.3%}")
    if sch.quality:
        print("-- quality flag values:\n" + raw[sch.quality].value_counts(dropna=False).head(20).to_string())
    if sch.duration:
        print("-- observed duration:\n" + raw[sch.duration].describe().to_string())
    if "timestamp" in std:
        st = std.sort_values(["station", "timestamp"])
        step = st.groupby("station")["timestamp"].diff().dt.total_seconds().div(60)
        print("\n-- time step (minutes) distribution:\n" + step.value_counts().head(10).to_string())
        per = st.groupby("station").agg(rows=("timestamp", "size"), first=("timestamp", "min"), last=("timestamp", "max"))
        per["gaps_gt_1.5min"] = step.gt(config.MAX_STEP_MINUTES).groupby(st["station"]).sum()
        per["missing_cell_frac"] = st[BAND_COLS].isna().mean(axis=1).groupby(st["station"]).mean()
        print("\n-- per station\n" + per.to_string())
        X, meta = build_windows(std)
        print(f"\n-- {len(X):,} complete 15-min windows; per station:\n" + meta["station"].value_counts().sort_index().to_string())


def report_test(path):
    raw = pd.read_csv(path)
    print(f"\n== {path}: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    print(raw.head().to_string())
    std, sch = standardise(raw)
    print("\n-- detected schema\n" + sch.describe())
    try:
        X, meta = windows_from_long(std)
        print(f"-- {len(X)} windows, missing cell fraction {np.isnan(X).mean():.3%}")
        print("-- stations:\n" + meta["station"].value_counts().to_string())
    except ValueError as e:
        print(f"!! could not build test windows: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records")
    ap.add_argument("--test")
    ap.add_argument("--nrows", type=int, default=None)
    a = ap.parse_args()
    if a.records:
        report_records(a.records, a.nrows)
    if a.test:
        report_test(a.test)


if __name__ == "__main__":
    main()

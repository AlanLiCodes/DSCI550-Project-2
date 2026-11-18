#!/usr/bin/env python
"""Write synthetic stand-in Train/Test CSVs (with hidden test labels) to exercise the file-based path.

python scripts/make_synthetic_data.py --out data/synthetic
python scripts/run_pipeline.py --records data/synthetic/Train.csv --test data/synthetic/Test.csv --fast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vesselnoise.data import standardise  # noqa: E402
from vesselnoise.synth import synthetic_records, synthetic_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=500)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = synthetic_records(n_days=a.days)
    raw.to_csv(out / "Train.csv", index=False)
    std, _ = standardise(raw)
    test, labels = synthetic_test(std, n=a.n_test)
    test.to_csv(out / "Test.csv", index=False)
    labels.to_csv(out / "test_labels_hidden.csv", index=False)
    print(f"wrote {len(raw):,} records, {a.n_test} test windows -> {out}")


if __name__ == "__main__":
    main()

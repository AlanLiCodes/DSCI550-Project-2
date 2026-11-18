#!/usr/bin/env python
"""Convert vessel-passage WAV recordings (VTUAD/ONC, HearMyShip) to minute x 25-band linear energy CSVs.

python scripts/convert_reference_audio.py --in raw/vtuad --out data/references/onc
python scripts/convert_reference_audio.py --in raw/hearmyship --out data/references/hearmyship --sensitivity-db -170

Each WAV becomes <stem>.csv with 25 band columns (linear energy, uPa^2 when the
sensitivity is right). Add a metadata.csv (vessel_id, cpa_m, speed_ms) next to
the outputs when CPA / speed are known from AIS or the dataset metadata;
otherwise the defaults in vesselnoise.reference.load_reference_dir are used.
Record dataset versions and licences in data/references/SOURCES.md and declare
them on the competition forum before using them for submissions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vesselnoise import config  # noqa: E402
from vesselnoise.reference import audio_to_minute_bands  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensitivity-db", type=float, default=0.0, help="hydrophone sensitivity, dB re V/uPa")
    ap.add_argument("--min-minutes", type=int, default=config.WINDOW_MINUTES)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = [f"{c:.1f}Hz" for c in config.BAND_CENTRES_HZ]
    for f in sorted(Path(a.inp).rglob("*.wav")):
        fs, x = wavfile.read(f)
        if x.dtype.kind == "i":
            x = x / float(2 ** (8 * x.dtype.itemsize - 1))
        if fs < 2 * config.BAND_UPPER_HZ[-1]:
            print(f"note {f.name}: fs={fs} -> bands above Nyquist will be NaN")
        bands = audio_to_minute_bands(x, fs, a.sensitivity_db)
        if len(bands) < a.min_minutes:
            print(f"skip {f.name}: only {len(bands)} minutes")
            continue
        pd.DataFrame(bands, columns=cols).to_csv(out / f"{f.stem}.csv", index=False)
        print(f"{f.name}: {len(bands)} minutes")


if __name__ == "__main__":
    main()

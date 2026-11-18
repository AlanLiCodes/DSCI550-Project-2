#!/usr/bin/env python
"""Convert VTUAD v1.3.2 strict-interval WAVs to minute x 25-band references.

python scripts/convert_vtuad.py --vtuad-dir ~/dsci550/vtuad --out data/references/vtuad

Writes <out>/vessel/<clip>.csv, <out>/background/<clip>.csv and <out>/clips.csv
(registry + status + minutes). Re-running skips clips already converted.
Then: python scripts/run_pipeline.py ... --refs-vtuad data/references/vtuad
Attribution: VTUAD (Yuan et al., v1.3.2, doi:10.5281/zenodo.23037895), Ocean Networks
Canada Oceans 3.0, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vesselnoise.vtuad import convert  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtuad-dir", required=True, help="folder with meta/metadata.csv and wav/strict/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="strict", choices=["strict", "containers"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    t0 = time.time()
    clips = convert(a.vtuad_dir, a.out, view=a.view, limit=a.limit)
    print(clips["status"].value_counts().to_string())
    ok = clips[clips["status"] == "ok"]
    print(ok.groupby("label")["minutes"].agg(["count", "sum", "median"]).to_string())
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()

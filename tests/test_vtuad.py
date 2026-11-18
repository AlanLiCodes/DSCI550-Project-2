import numpy as np
import pandas as pd
from scipy.io import wavfile

from vesselnoise.reference import Reference
from vesselnoise.splits import assign_reference_splits
from vesselnoise.vtuad import convert, load_vtuad_references


def test_hull_grouping_keeps_vessel_in_one_split():
    refs = [Reference(f"clip{i}", "vtuad_tug", np.ones((5, 25)), np.ones(5), group=f"mmsi_{i % 4}") for i in range(40)]
    sp = assign_reference_splits(refs)
    for g in {r.group for r in refs}:
        assert len({sp[r.vessel_id] for r in refs if r.group == g}) == 1


def test_convert_and_load_roundtrip(tmp_path):
    root = tmp_path / "vtuad"
    (root / "meta").mkdir(parents=True)
    fs = 32000
    rng = np.random.default_rng(0)
    rows = []
    for name, label, mmsi, amp in [("a", "tug", 1, 1.0), ("b", "tug", 1, 1.0), ("bg", "background", np.nan, 0.05)]:
        d = root / "wav" / "strict" / ("background" if label == "background" else "vessel")
        d.mkdir(parents=True, exist_ok=True)
        t = np.arange(fs * 150) / fs
        x = 0.05 * rng.normal(size=t.size) + amp * np.sin(2 * np.pi * 100 * t) * np.exp(-(((t - 75) / 30) ** 2))
        wavfile.write(d / f"{name}.wav", fs, (x / 4 * 32767).astype(np.int16))
        rows.append({"label": label, "MMSI": mmsi, "path": f"{label}/{name}.wav", "utc_begin": "", "utc_end": ""})
    pd.DataFrame(rows).to_csv(root / "meta" / "metadata.csv", index=False)
    clips = convert(root, tmp_path / "conv", log=lambda *_: None)
    assert (clips["status"] == "ok").all() and (clips["minutes"] == 2).all()
    refs = load_vtuad_references(tmp_path / "conv")
    assert len(refs) == 2 and {r.group for r in refs} == {"mmsi_1"}
    assert refs[0].lin.shape == (2, 25) and (refs[0].lin >= 0).all()

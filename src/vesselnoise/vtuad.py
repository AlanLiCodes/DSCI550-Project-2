"""VTUAD v1.3.2 (Ocean Networks Canada, CC BY 4.0) -> minute band-level references.

The challenge's vessel additions are derived from VTUAD, so these are the primary
real references. Conversion follows the organisers' starter notebook
(:func:`vesselnoise.reference.audio_to_minute_bands`).

Layout produced by ``scripts/setup_vtuad.sh`` / expected here::

    vtuad/meta/metadata.csv            clip registry (label, MMSI, path, utc_begin, ...)
    vtuad/wav/strict/vessel/**.wav     clips cut to the exact AIS-qualified interval
    vtuad/wav/strict/background/**.wav

Strict-interval clips are used by default: 167 archive-aligned containers end by
re-reading audio from *before* the labelled interval (wrap-around splice), which
would corrupt a passage's time profile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .reference import Reference, audio_to_minute_bands, passage_ranges

# Defaults when no AIS track is available (only 3 days of raw AIS survive in the
# release). All VTUAD vessels were inside the 4 km inclusion radius.
DEFAULT_CPA_M = 1500.0
DEFAULT_SPEED_MS = 5.0


def _find_col(df: pd.DataFrame, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def read_registry(meta_dir) -> pd.DataFrame:
    """Load metadata.csv and standardise the columns this project uses."""
    meta_dir = Path(meta_dir)
    reg = pd.read_csv(meta_dir / "metadata.csv")
    out = pd.DataFrame(
        {
            "label": reg[_find_col(reg, "label", "class")].astype(str).str.lower(),
            "path": reg[_find_col(reg, "path", "relative_path", "wav_path")].astype(str),
        }
    )
    for std, cands in {
        "mmsi": ("mmsi", "hull_id"),
        "clip_uid": ("clip_uid", "uid"),
        "utc_begin": ("utc_begin",),
        "utc_end": ("utc_end",),
        "duration_sec": ("duration_sec", "duration"),
        "length_m": ("length", "length_m"),
        "beam_m": ("beam", "beam_m"),
    }.items():
        c = _find_col(reg, *cands)
        out[std] = reg[c] if c else np.nan
    # Registry paths look like "./vessel/12.wav" or "./background/3.wav"; stems repeat
    # across the two folders, so the key is "<folder>/<stem>".
    out["folder"] = out["path"].map(lambda p: Path(p).parent.name)
    out["stem"] = out["path"].map(lambda p: Path(p).stem)
    out["key"] = out["folder"] + "/" + out["stem"]
    return out


def index_wavs(root) -> dict:
    """Map "<parent folder>/<stem>" -> path for every .wav under ``root``.

    Strict clips unzip to .../strict_interval_wav/inclusion_4000_exclusion_6000/{vessel,background}/N.wav.
    """
    return {f"{p.parent.name}/{p.stem}": p for p in Path(root).rglob("*.wav")}


def wav_minute_bands(path, minute_s: float = 60.0) -> np.ndarray:
    """Minute band energy for one WAV, processed a minute at a time (memory-safe)."""
    from scipy.io import wavfile

    fs, x = wavfile.read(path, mmap=True)
    if x.ndim > 1:
        x = x[:, 0]
    scale = float(2 ** (8 * x.dtype.itemsize - 1)) if x.dtype.kind == "i" else 1.0
    per_min = int(round(fs * minute_s))
    n_min = len(x) // per_min
    rows = []
    for m in range(n_min):
        seg = np.asarray(x[m * per_min : (m + 1) * per_min], dtype=np.float64) / scale
        rows.append(audio_to_minute_bands(seg, fs)[0])
    return np.array(rows).reshape(-1, config.N_BANDS), fs


def convert(vtuad_dir, out_dir, view: str = "strict", limit: int | None = None, log=print):
    """Convert every registry clip to ``out_dir/<group>/<stem>.csv`` + ``out_dir/clips.csv``.

    group is ``vessel`` or ``background``. Band values are linear mean-square in the
    WAV's digital units (uncalibrated; one hydrophone, so consistent across clips).
    """
    vtuad_dir, out_dir = Path(vtuad_dir), Path(out_dir)
    reg = read_registry(vtuad_dir / "meta")
    wavs = index_wavs(vtuad_dir / "wav" / view)
    cols = [f"{c:.1f}Hz" for c in config.BAND_CENTRES_HZ]
    rows = []
    for i, r in enumerate(reg.itertuples()):
        if limit is not None and i >= limit:
            break
        p = wavs.get(r.key)
        # Dedicated background recordings live in background/. 22 clips under vessel/
        # also carry label "background"; they are kept apart and not used as references.
        if r.folder == "background":
            group = "background"
        elif r.label == "background":
            group = "vessel_labelled_background"
        else:
            group = "vessel"
        if p is None:
            rows.append({**r._asdict(), "group": group, "status": "wav not found"})
            continue
        target = out_dir / group / f"{r.folder}_{r.stem}.csv"
        if not target.exists():
            try:
                bands, fs = wav_minute_bands(p)
            except Exception as e:  # keep going; report at the end
                rows.append({**r._asdict(), "group": group, "status": f"error: {e}"})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(bands, columns=cols).to_csv(target, index=False)
        n = sum(1 for _ in open(target)) - 1
        rows.append({**r._asdict(), "group": group, "status": "ok", "minutes": n, "csv": str(target.relative_to(out_dir))})
        if (i + 1) % 25 == 0:
            log(f"{i + 1}/{len(reg)} clips")
    clips = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
    clips.to_csv(out_dir / "clips.csv", index=False)
    return clips


def onc_background_spectrum(converted_dir) -> np.ndarray:
    """Per-band median ambient level of the VTUAD background class (same hydrophone)."""
    files = sorted((Path(converted_dir) / "background").glob("*.csv"))
    if not files:
        return None
    lin = np.vstack([pd.read_csv(f).to_numpy(dtype=float) for f in files])
    return np.nanmedian(lin, axis=0)


def load_vtuad_references(converted_dir, min_minutes: int = 2, background="onc", own_quantile: float = 0.1):
    """Build References from converted VTUAD clips.

    * ``group`` (split unit) is the hull MMSI, so one vessel never spans two splits.
    * Ambient handling: ``background="onc"`` subtracts the ONC background-class median per
      band; ``"own"`` subtracts the clip's own low quantile; ``"none"`` keeps the clip as
      recorded. The organisers note that "passage references may contain ambient sound",
      i.e. their additions appear to be raw clips, so ``"none"`` matches the challenge.
    """
    converted_dir = Path(converted_dir)
    clips = pd.read_csv(converted_dir / "clips.csv")
    clips = clips[(clips["status"] == "ok") & (clips["group"] == "vessel") & (clips["minutes"] >= min_minutes)]
    bg = onc_background_spectrum(converted_dir) if background == "onc" else None
    refs = []
    for r in clips.itertuples():
        lin = pd.read_csv(converted_dir / r.csv).to_numpy(dtype=float)
        if background == "none":
            sub = lin.copy()
        elif bg is not None:
            sub = np.clip(lin - bg[None, :], 0.0, None)
        else:
            sub = np.clip(lin - np.nanquantile(lin, own_quantile, axis=0, keepdims=True), 0.0, None)
        sub = np.nan_to_num(sub, nan=0.0)
        if sub.sum() <= 0:
            continue
        t_peak = float(np.argmax(sub.sum(axis=1))) + 0.5
        track = passage_ranges(len(sub), DEFAULT_CPA_M, DEFAULT_SPEED_MS, t_peak).mean(axis=1)
        if pd.isna(r.mmsi):
            mmsi = str(r.key)
        else:
            try:
                mmsi = str(int(float(r.mmsi)))
            except ValueError:
                mmsi = str(r.mmsi)
        refs.append(Reference(vessel_id=str(r.key), source=f"vtuad_{r.label}", lin=sub, range_m=track, group=f"mmsi_{mmsi}"))
    return refs

"""Column detection for the competition files.

The released files' exact column names were not available when this was
written, so detection is heuristic and reported by ``scripts/inspect_data.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

_FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(k)?(?:hz)?", re.IGNORECASE)


def parse_frequency(name: str) -> float | None:
    """Extract a frequency in Hz from a column name like 'band_31.6', '1k', 'f_1000Hz', 'L_7943'."""
    best = None
    for m in _FREQ_RE.finditer(str(name)):
        val = float(m.group(1))
        if m.group(2):
            val *= 1000.0
        best = val  # last numeric token wins (e.g. 'b03_31.6' -> 31.6)
    return best


def match_band_columns(columns, tol: float = 0.06) -> list[str]:
    """Return columns that map one-to-one onto the 25 decidecade centres, ordered by frequency.

    A column matches a centre if its parsed frequency is within ``tol`` (relative)
    of the centre, or equals the centre rounded to an integer (31.6 -> 31 or 32).
    Falls back to columns named by band index (e.g. 'band_0'..'band_24').
    """
    hits: dict[int, str] = {}
    for col in columns:
        f = parse_frequency(col)
        if f is None or f <= 0:
            continue
        rel = np.abs(config.BAND_CENTRES_HZ - f) / config.BAND_CENTRES_HZ
        k = int(np.argmin(rel))
        if rel[k] <= tol and k not in hits:
            hits[k] = col
    if len(hits) == config.N_BANDS:
        return [hits[k] for k in range(config.N_BANDS)]

    # Fallback: index-named bands.
    idx_cols = {}
    for col in columns:
        m = re.fullmatch(r"(?:band|b|bnd)[_\- ]?(\d{1,2})", str(col), re.IGNORECASE)
        if m:
            idx_cols[int(m.group(1))] = col
    if len(idx_cols) == config.N_BANDS:
        return [idx_cols[k] for k in sorted(idx_cols)]
    raise ValueError(
        f"Could not identify {config.N_BANDS} band columns (found {len(hits)} by frequency). "
        "Edit config.py or rename columns."
    )


def find_column(columns, candidates) -> str | None:
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def infer_units(values: np.ndarray) -> str:
    """'db' if values look like sound-pressure levels, else 'linear'."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return "db"
    med = float(np.median(v))
    if 0.0 < med < 250.0 and np.percentile(v, 1) > -50:
        return "db"
    return "linear"


@dataclass
class Schema:
    band_cols: list[str]
    station: str | None = None
    time: str | None = None
    window_id: str | None = None
    minute: str | None = None
    quality: str | None = None
    duration: str | None = None
    units: str = "db"
    extra: list[str] = field(default_factory=list)

    @classmethod
    def detect(cls, df: pd.DataFrame) -> "Schema":
        cols = list(df.columns)
        bands = match_band_columns(cols)
        sch = cls(
            band_cols=bands,
            station=find_column(cols, config.STATION_CANDIDATES),
            time=find_column(cols, config.TIME_CANDIDATES),
            window_id=find_column(cols, config.ID_CANDIDATES),
            minute=find_column(cols, config.MINUTE_CANDIDATES),
            quality=find_column(cols, config.QUALITY_CANDIDATES),
            duration=find_column(cols, config.DURATION_CANDIDATES),
        )
        sample = df[bands].to_numpy(dtype=float, na_value=np.nan)[:50_000]
        sch.units = infer_units(sample)
        used = set(bands) | {c for c in [sch.station, sch.time, sch.window_id, sch.minute, sch.quality, sch.duration] if c}
        sch.extra = [c for c in cols if c not in used]
        return sch

    def describe(self) -> str:
        lines = [
            f"band columns ({len(self.band_cols)}): {self.band_cols[0]} .. {self.band_cols[-1]}",
            f"units: {self.units}",
        ]
        for name in ["station", "time", "window_id", "minute", "quality", "duration"]:
            lines.append(f"{name}: {getattr(self, name)}")
        lines.append(f"other columns: {self.extra}")
        return "\n".join(lines)

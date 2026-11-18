"""Project-wide constants and schema assumptions.

Everything that depends on the exact layout of the Zindi files lives here so
it can be corrected in one place once ``scripts/inspect_data.py`` has been run
on the real data.
"""

from __future__ import annotations

import numpy as np

SEED = 550

# --- Frequency bands -------------------------------------------------------
# 25 base-10 decidecade bands, centres 10**(n/10) Hz for n = 15..39
# (31.6 Hz ... 7.94 kHz). Band edges are centre * 10**(+-1/20).
BAND_INDEX = np.arange(15, 40)
BAND_CENTRES_HZ = 10.0 ** (BAND_INDEX / 10.0)
BAND_LOWER_HZ = BAND_CENTRES_HZ * 10.0 ** (-1.0 / 20.0)
BAND_UPPER_HZ = BAND_CENTRES_HZ * 10.0 ** (1.0 / 20.0)
N_BANDS = len(BAND_CENTRES_HZ)

# Band groups used by features.
WIND_BAND_HZ = (630.0, 2000.0)  # wind / sea-state proxy (0.63-2 kHz)
LOW_BAND_MAX_HZ = 200.0  # shipping-dominated low frequencies
HIGH_BAND_MIN_HZ = 2000.0

# --- Windows ---------------------------------------------------------------
WINDOW_MINUTES = 15
# Consecutive records further apart than this are a recording gap; windows
# never bridge a gap.
MAX_STEP_MINUTES = 1.5

# --- Column-name candidates (case-insensitive, first match wins) ----------
STATION_CANDIDATES = ["station", "site", "station_id", "site_id", "recorder", "location", "deployment"]
TIME_CANDIDATES = ["timestamp", "datetime", "time", "date_time", "start_time", "utc", "date"]
ID_CANDIDATES = ["id", "window_id", "sample_id", "ID", "example_id"]
MINUTE_CANDIDATES = ["minute", "minute_index", "t", "step", "offset_min"]
QUALITY_CANDIDATES = ["quality", "quality_flag", "qc", "qc_flag", "flag"]
DURATION_CANDIDATES = ["duration", "observed_duration", "duration_s", "seconds_observed"]

# Quality-flag values treated as good when a quality column is present.
GOOD_QUALITY_VALUES = {0, 1, "good", "ok", "pass", True}

# --- Targets / submission -------------------------------------------------
TARGET_SHARE = "share_added_vessel"
TARGET_SCORE = "score_added_vessel"

# --- Mixture construction -------------------------------------------------
ATB_RANGE_DB = (-20.0, 20.0)  # addition-to-background ratio
P_ZERO_ADDITION = 0.2
CPA_RANGE_M = (50.0, 5000.0)  # closest point of approach, log-uniform
SPREADING_RANGE = (15.0, 20.0)  # practical .. spherical spreading coefficient

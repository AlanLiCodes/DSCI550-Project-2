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
# A window may be built from a run of at least this many consecutive minutes; the
# remaining minute positions are masked (Test.csv windows miss at most 2 minutes).
MIN_RUN_MINUTES = 13

# --- Column-name candidates (case-insensitive, first match wins) ----------
STATION_CANDIDATES = ["station", "site", "station_id", "site_id", "recorder", "location", "deployment"]
TIME_CANDIDATES = ["timestamp", "t", "datetime", "time", "date_time", "start_time", "utc", "date"]
ID_CANDIDATES = ["id", "window_id", "sample_id", "ID", "example_id"]
MINUTE_CANDIDATES = ["minute", "minute_index", "step", "offset_min"]
QUALITY_CANDIDATES = ["quality", "quality_flag", "qc", "qc_flag", "flag"]
DURATION_CANDIDATES = ["dur", "duration", "observed_duration", "duration_s", "seconds_observed"]
RECORDER_CANDIDATES = ["recorder", "deployment_id", "instrument"]

# Real export names of the 25 common bands (South_African_Real_Acoustics, Test.csv).
BAND_COLUMN_NAMES = [f"TOL_{c}" for c in ["31.6", "39.8", "50.1", "63.1", "79.4", "100", "125.9", "158.5", "199.5",
    "251.2", "316.2", "398.1", "501.2", "631", "794.3", "1000", "1258.9", "1584.9", "1995.3", "2511.9", "3162.3",
    "3981.1", "5011.9", "6309.6", "7943.3"]]

# Records observed for less than this many seconds are masked (README: check `dur`).
MIN_DURATION_S = 55.0
# Band levels outside this range are export sentinels / glitches and are masked
# (653 Richards Bay rows hold negative levels down to -156 dB; Test.csv minimum is 29 dB).
VALID_DB_RANGE = (20.0, 200.0)

# Whole-minute dropout applied to mixture backgrounds. Test.csv windows are missing
# whole minutes (all 25 bands at once) while the released training records have no
# missing band cells, so models must learn to handle masked minutes.
MINUTE_DROPOUT_P = 0.5  # probability a window gets any dropped minutes
MINUTE_DROPOUT_MAX = 2  # at most this many minutes dropped

# Quality-flag values treated as good when a quality column is present.
GOOD_QUALITY_VALUES = {0, 1, "good", "ok", "pass", True}

# --- Targets / submission -------------------------------------------------
TARGET_SHARE = "share_added_vessel"
TARGET_SCORE = "score_added_vessel"

# --- Mixture construction -------------------------------------------------
# Addition-to-background ratio range. (-20, 5) with 20% zero windows is the smallest change
# to the proposal's (-20, 20) whose labels are consistent with the organisers' published
# benchmark: a constant 0.1 scores 0.7539, i.e. E[(y - 0.1)^2] = 0.0606 (ours: 0.0632).
ATB_RANGE_DB = (-20.0, 5.0)
P_ZERO_ADDITION = 0.2
CPA_RANGE_M = (50.0, 5000.0)  # closest point of approach, log-uniform
SPREADING_RANGE = (15.0, 20.0)  # practical .. spherical spreading coefficient

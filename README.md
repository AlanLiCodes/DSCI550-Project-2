# Estimating Added Vessel Noise in Underwater Acoustic Data

DSCI 550 Project 2. The project enters the Zindi **Underwater Vessel Noise Challenge**
(competition period Oct 16 – Dec 13, 2026).

Team: Alan Li, Yi Sheng Chao, Jeremy Han, Hanqi Ni

The goal is to estimate `share_added_vessel`, the fraction of acoustic energy
in a 15-minute x 25-band window that comes from an added vessel passage. A
presence score, `score_added_vessel`, is produced alongside it. The leaderboard
metric is **1 − RMSE** on `share_added_vessel`.

## Setup

```bash
pip install -e ".[dev]"     # also: pip install pyarrow (parquet)
pytest                      # 38 tests, ~7 s
```

Data layout (nothing under `data/` or `outputs/` is committed):

```
data/
  South_African_Real_Acoustics/   # unzipped: acoustics/*.parquet, stations.csv, recorders.csv, DATA_DICTIONARY.md
  Test.csv                         # 1,500 rows: ID + 375 cols m01_TOL_31.6 .. m15_TOL_7943.3
  SampleSubmission.csv
```

## Workflow

```bash
# 1. Data check: schema, cadence per recorder, windows per station, test missingness and levels
python scripts/inspect_data.py --records data/South_African_Real_Acoustics --test data/Test.csv

# 2. Full run (about 5 min, 2.4 GB peak on 2 CPUs)
python scripts/run_pipeline.py --records data/South_African_Real_Acoustics --test data/Test.csv \
    --sample-submission data/SampleSubmission.csv --loso --ablations --source-shift --out outputs/run1

# 3. VTUAD references: download the record's audio + metadata from Zenodo, then
bash scripts/setup_vtuad.sh          # copy next to where the data should live; verifies MD5s, unzips
python scripts/convert_vtuad.py --vtuad-dir <vtuad dir> --out data/references/vtuad
python scripts/run_pipeline.py ... --refs-vtuad data/references/vtuad [--stretch-max 4]
```

`--synthetic` runs everything on stand-in data. Useful flags:

- `--drop-families level,spectral`
- `--minute-dropout`, `--station-balance`, `--p-zero`
- `--station-aware`: experiments only; Test.csv has no station or time

Outputs go to `--out`:

- `results.csv`, breakdowns, `ablations.csv`, `loso*.csv`, `source_shift.csv`, `residual_autocorr.csv`
- `submission.csv`, written in the exact `ID,share_added_vessel,score_added_vessel` format and checked to be finite and in [0, 1]
- `run_summary.json`

## What the real files look like (and what the code does about it)

| Finding | Handling |
|---|---|
| Training data: 10 stations, 17 recorders, one parquet per station, 70 TOL bands; the 25 common bands are `TOL_31.6` … `TOL_7943.3`, timestamp `t`, duration `dur` | Exact names are tried first (`config.BAND_COLUMN_NAMES`) |
| Richards Bay holds 1.0M of 1.7M rows (65% of windows) | Background sampling weight is count(station)^−0.5 (`--station-balance`) |
| Duty-cycled recorders: Cape Recife and Riy Banks record 13-min runs, St Francis and Fish Hoek mostly 5-min runs | Windows come from runs of ≥ 13 minutes, with the remaining positions masked → 103,227 windows from all 10 stations |
| `dur` is not always 60 s; 2-second timestamp offsets create near-duplicate minutes | Rows with `dur` < 55 s are masked; timestamps are floored to the minute and duplicates dropped |
| 653 Richards Bay rows hold sentinel levels down to −156 dB | Levels outside 20–200 dB are masked (Test.csv minimum is 29 dB) |
| Windows must not span a recorder change | Segments also break on `recorder` |
| **Test.csv has no station or timestamp** | Submissions use station-agnostic features only (no station_norm, hour or station × hour baseline priors); leave-one-site-out is the main model-selection criterion |
| Test missingness is whole minutes only (3.3% of cells, 48% of windows), mostly minute 1 (41.6%) and minute 15 (6.7%); the training bands have no missing cells | Generic whole-minute dropout on mixture backgrounds (50% of windows, 1–2 random minutes). The test's edge-heavy pattern is deliberately **not** copied, because that would be tuning to test inputs. Team decision. |
| Test band levels are 4–12 dB above the training-background medians, with the gap growing toward 8 kHz | Consistent with substantial added energy and/or withheld test stations |
| Leaderboard: the starter's constant 0.1 scores 0.7539 → E[(y − 0.1)²] = 0.061, so the test label mean is ≤ 0.35 | Our mixtures have label mean 0.40 and a wider spread; the ATB / `p_zero` prior should be revisited |

### References

The starter notebook states that the test additions are derived from **VTUAD v1.3.2**
(Ocean Networks Canada archives, CC BY 4.0, doi:10.5281/zenodo.23037895). The organisers
converted that audio to minute band energy and varied the time trajectory, relative channel
and energy. VTUAD is therefore the reference source that matches the test, ahead of
HearMyShip. `reference.audio_to_minute_bands` follows the starter's conversion: 2-second
Hann periodogram segments, 30 per minute, averaged in linear units, with bands above
Nyquist set to NaN.

## Code map (`src/vesselnoise/`)

| Module | What it does |
|---|---|
| `config.py` | 25 decidecade bands (31.6 Hz to 7.94 kHz), window length, the candidate column names, the mixture ranges. **Fix schema assumptions here.** |
| `schema.py`, `data.py` | Column detection (station, time, ID, minute, quality, duration, band columns), dB vs linear units, masking of quality-flagged rows |
| `windows.py` | 15-minute windows within one recorder and day that never bridge gaps. Also rebuilds test windows from long format. |
| `energy.py` | dB/linear conversions and masked energy sums. All averaging is in linear energy. |
| `reference.py` | Audio → 2 s Hann periodograms → band energy → minute means (the organisers' conversion); own-background subtraction; synthetic stand-in passages |
| `propagation.py` | Spreading (k = 15–20) plus Thorp absorption, and straight-track range profiles |
| `mixtures.py` | Crop a passage so its rise and fall stay inside the window, re-place it at a sampled CPA, scale it to the target ATB, add it to the background. Labels are computed over observed cells only. |
| `baseline.py` | Label-free excess-energy baseline (Merchant et al., 2012) from quiet-minute and station × hour background priors, with a margin tuned on training mixtures |
| `features.py` | Feature families for the ablations: level, variability, station_norm, spectral, temporal, missing, time, excess |
| `models.py` | Constant, Ridge, random forest and histogram GB, clipped to [0, 1]; logistic regression for `score_added_vessel`. Median imputation and missingness indicators are fitted on train only. |
| `splits.py` | Per-station chronological 60/20/20 day blocks with one-day buffers; each vessel in exactly one split; LOSO; residual autocorrelation |
| `metrics.py` | 1 − RMSE, breakdowns, day-block bootstrap, ROC/PR AUC, the AIS contrast helper |
| `pipeline.py` | Orchestration: build experiment → evaluate → ablations / LOSO / source shift → final fit |

### Leakage rules enforced in code

- Backgrounds are split by station-day before any mixture is built, and vessels are split before cropping. No background day or source vessel appears in two splits (this is tested).
- Station medians, baseline priors, imputation and scaling are fitted on training data only.
- `Test.csv` is only passed to `transform` / `predict`.
- The final model is chosen by blocked CV: the LOSO ranking (average of mean and worst site) when `--loso` is run, otherwise the validation score. It is never tuned against the public leaderboard.

## Results on the real South African backgrounds

Scores are 1 − RMSE on held-out **mixtures** (held-out days and vessels), not leaderboard
scores. Every run uses 20k train / 5k val / 5k local-test mixtures and station-agnostic features.

### VTUAD references (current)

`scripts/setup_vtuad.sh` → `scripts/convert_vtuad.py` gives 482 usable vessel passages
from 172 hulls (strict-interval clips, ambient removed using the 16 dedicated ONC background
recordings). Splits are by hull MMSI.

| Run | Model | Val | Local test | LOSO mean / worst site |
|---|---|---|---|---|
| `v2a_vtuad` (no stretch) | histogram GB | **0.866** | **0.863** (0.849–0.877) | **0.860 / 0.738** (St Francis) |
| | random forest | 0.859 | 0.853 | – |
| | Ridge | 0.815 | 0.824 | 0.808 / 0.712 |
| | excess-energy baseline | 0.639 | 0.643 | 0.626 / 0.317 |
| | constant | 0.606 | 0.608 | 0.607 / 0.604 |
| `v2b_vtuad_stretch` (stretch ≤ 4×) | histogram GB | 0.843 | 0.843 | 0.838 / 0.693 |
| `v2c` (stretch ≤ 4×, no level / spectral) | histogram GB | 0.843 | 0.841 | 0.832 / 0.719 |

**Does VTUAD help? A like-for-like check.** Both models are tested on the same mixtures
built from VTUAD hulls neither has seen (10k train / 4k test, fast models, saved to
`outputs/reference_realism_check.csv`):

| Trained on | histogram GB | random forest | Ridge |
|---|---|---|---|
| synthetic references | 0.757 | 0.763 | 0.692 |
| VTUAD references (other hulls) | **0.850** | 0.842 | 0.819 |

Notes:

- **Stretch scores are not comparable to the no-stretch run.** With stretching, the local test mixtures are stretched too, which is a harder task. Whether the organisers' "varied time trajectories" means stretching can only be settled on the leaderboard. Submit `v2a`, then `v2b`, and compare.
- With VTUAD references, dropping absolute level/spectral features no longer helps (0.841 vs 0.843).
- The VTUAD vessel signal is often weak: the median clip is only 3–9 dB above the ONC ambient, and 20% of clips are within 1 dB of it. Strict clips are short (median 3 min, max 23).
- St Francis is consistently the hardest site, with Fish Hoek next. Both are 5-minute duty-cycled recorders.

### Leaderboard results and what they revealed

| Submission | Public score | Score ROC AUC |
|---|---|---|
| starter constant 0.1 (benchmark) | 0.7539 | – |
| `v2a_vtuad` (ONC ambient subtracted, no stretch) | 0.7388 | 0.537 |
| `v2b_vtuad_stretch` (stretch ≤ 4×) | 0.7161 | 0.541 |

Local scores of about 0.86 did **not** transfer, so the mixtures did not match how the test windows were built.
Diagnostics on the test *inputs* (no labels; `scripts/diagnose_test_shift.py`):

- **The mixtures and test windows were easy to tell apart** (classifier AUC 0.96). Test windows have more high-frequency activity and less extreme short peaks: at the 90th percentile, peak minus median is 14 dB in the test against 23 dB in the mixtures.
- **Test backgrounds look like Richards Bay** (90% of windows by spectral and temporal shape), while station balancing had down-weighted it. Balancing is now off by default.
- **The starter notes that "passage references may contain ambient sound".** Adding raw VTUAD clips, rather than ambient-subtracted ones, lowers the classifier AUC (0.94 → 0.90). A quieter ATB range (−20 to +10 dB) lowers it further (0.88). Stretching raises it (0.92), which agrees with v2b scoring worse. `--vtuad-ambient none` is now the default.
- **Calibration matters as much as the model.** Combining the 0.1 benchmark with our scores: if the test label mean is about 0.10, the v2a predictions correlate about 0.8 with the labels but sit too high (their mean is 0.31).

Next candidates: `v3a_raw_atb-20to10` (raw clips, natural station mix, ATB −20 to +10 dB) and
`v3b_raw_atb-20to20`. Local histogram GB scores: 0.894 / 0.896 on held-out days and hulls.

### Loudness and busy-background augmentation (v4)

Why v3a still over-predicts: on clean held-out backgrounds it predicts 0.02, but on
Test.csv windows it averages 0.33, with 58% of windows above 0.2. Test windows are much louder and busier than
the released backgrounds (median 2–8 kHz level 84.8 vs ~74 dB). The model reads that
variability as added vessel, though natural ship traffic counts as background (label 0).

Two augmentations in `make_mixtures` (CLI flags, all off by default):

- **Loudness:** `--gain-aug 10 --tilt-aug 6` (random ±10 dB background gain, ±6 dB/decade
  tilt) and `--ref-tilt-aug 6 --ref-jitter 1` (random channel on the added passage).
  Labels stay exact because the addition is scaled against the augmented background.
- **Busy backgrounds:** `--busy-p 0.4 --busy-atb=-15,5` adds a second VTUAD passage to
  40% of backgrounds *as background*, so a ship being present is not enough to predict added energy.

Comparison (`scripts/compare_augmentations.py`; same held-out mixtures for all; test
inputs only, no labels):

| Augmentation | Plain held-out | Augmented held-out | Clean-bg prediction | Test prediction mean / median | Test > 0.2 |
|---|---|---|---|---|---|
| none (v3a) | 0.897 | 0.801 | 0.024 | 0.331 / 0.278 | 58% |
| loudness only | 0.892 | 0.802 | 0.030 | 0.328 / 0.273 | 58% |
| busy only | 0.859 | 0.820 | 0.025 | 0.248 / 0.193 | 49% |
| both | 0.853 | 0.824 | 0.030 | 0.247 / 0.191 | 48% |

Busy backgrounds account for the change; loudness augmentation on its own does little.
`v4_loud_busy` (both, ATB −20 to +10 dB, histogram GB) scores 0.823 on its own augmented held-out split. Its
submission averages 0.244 (median 0.194), against 0.320 for v3a (correlation 0.94).
St Francis is still a false-positive site (about 0.2 on clean background).

### Synthetic references (`real_v1`, for reference)

| Model | Val | Local test (95% day-block CI) | LOSO mean / worst site |
|---|---|---|---|
| constant | 0.605 | 0.608 (0.605–0.611) | 0.607 / 0.604 |
| excess-energy baseline | 0.556 | 0.561 (0.538–0.583) | 0.535 / 0.308 |
| Ridge | 0.762 | 0.764 (0.755–0.772) | 0.754 / 0.708 |
| random forest | 0.828 | 0.814 (0.802–0.825) | – |
| histogram GB | 0.839 | 0.828 (0.817–0.837) | 0.816 / 0.720 |

Ablations (histogram GB, synthetic references): dropping temporal shape costs 0.042;
dropping absolute level (+0.008) or spectral (+0.008) features slightly improved the score.
Daily residual autocorrelation is small at lags 1–2 (0.02, −0.03), which supports the one-day buffer.

## Status against the proposal timeline

| Item | Owner | Status |
|---|---|---|
| Data dictionary check (`inspect_data.py`) | A | Done on the real archive and Test.csv (findings above) |
| Cleaning, windowing, excess-energy baseline | A | Done; the baseline is weak on real data (see results) |
| External audio access and licences; conversion | B | VTUAD v1.3.2 downloaded, verified and converted (482 passages, 172 hulls). Still to do: declare it on the forum; HearMyShip optional |
| Controlled mixtures, feature extraction | B | Implemented (synthetic references until real ones are converted) |
| Constant, Ridge, RF, GB, evidence classifier | C | Implemented |
| Limited tuning | C | To do (Nov 19 – Dec 2): feature-family drops, ATB / p_zero prior, GB depth |
| LOSO, source shift, error analysis, leaderboard comparison | D | LOSO and source shift implemented; AIS check needs AIS data (`metrics.ais_contrast` is ready) |
| Leakage and missing-data review | A, B | Rules above; still needs a review on the real data |

## References

Hjelmervik et al. (2024), *Proc. IOA* 46(1) · Merchant et al. (2012), *Mar. Pollut. Bull.* 64(7) ·
Roberts et al. (2017), *Ecography* 40(8) · Schoeman, Erbe & Plön (2022), *JMSE* 10(6) ·
Shipton et al. (2025), *Sci. Data* 12, 289

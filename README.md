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
pytest                      # 31 tests, ~5 s
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

# 3. Once the vessel references are converted (see "References" below)
python scripts/convert_reference_audio.py --in raw/vtuad --out data/references/onc
python scripts/run_pipeline.py ... --refs-onc data/references/onc [--refs-hearmyship data/references/hearmyship]
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

## Results on the real South African backgrounds (synthetic references)

Run `outputs/real_v1`: 20k train / 5k val / 5k local-test mixtures, station-agnostic
features, synthetic vessel references. Scores are 1 − RMSE on held-out
**mixtures**, not leaderboard scores.

| Model | Val | Local test (95% day-block CI) | LOSO mean / worst site |
|---|---|---|---|
| constant | 0.605 | 0.608 (0.605–0.611) | 0.607 / 0.604 |
| excess-energy baseline | 0.556 | 0.561 (0.538–0.583) | 0.535 / 0.308 |
| Ridge | 0.762 | 0.764 (0.755–0.772) | 0.754 / 0.708 |
| random forest | 0.828 | 0.814 (0.802–0.825) | – |
| histogram GB | **0.839** | **0.828** (0.817–0.837) | **0.816 / 0.720** (St Francis) |

Presence score (logistic regression): ROC AUC 0.889 and PR AUC 0.973 on the local test split.

Takeaways:

1. **The label-free excess baseline does not work on real backgrounds.** It scores below the constant model, and 0.31 at Cape Recife and Riy Banks. Real ambient noise varies too much within 15 minutes for the quiet-minute estimate to work. It stays in as a feature, but its ablation shows it adds nothing (0.8282 vs 0.8281).
2. **Temporal shape is the most useful feature family.** Dropping it costs 0.042 on local test. Dropping absolute `level` (+0.008) or `spectral` (+0.008) **improves** the score, so the model is fitting station loudness. Try `--drop-families level,spectral` next.
3. **Reference realism matters more than anything else.** Trained on one synthetic reference type and tested on the other, every model falls below the constant (0.47–0.57). Getting VTUAD passages in is the top priority.
4. Daily residual autocorrelation is small at lags 1–2 (0.02, −0.03), which supports the one-day buffer.
5. `outputs/real_v1/submission.csv` (histogram GB, chosen by LOSO): mean predicted share 0.28, which is consistent with the leaderboard bound above. Its leaderboard score has not been recorded yet.

## Status against the proposal timeline

| Item | Owner | Status |
|---|---|---|
| Data dictionary check (`inspect_data.py`) | A | Done on the real archive and Test.csv (findings above) |
| Cleaning, windowing, excess-energy baseline | A | Done; the baseline is weak on real data (see results) |
| External audio access and licences; conversion | B | Converter ready. **Download VTUAD v1.3.2 (CC BY 4.0) first**, then HearMyShip; declare both on the forum |
| Controlled mixtures, feature extraction | B | Implemented (synthetic references until real ones are converted) |
| Constant, Ridge, RF, GB, evidence classifier | C | Implemented |
| Limited tuning | C | To do (Nov 19 – Dec 2): feature-family drops, ATB / p_zero prior, GB depth |
| LOSO, source shift, error analysis, leaderboard comparison | D | LOSO and source shift implemented; AIS check needs AIS data (`metrics.ais_contrast` is ready) |
| Leakage and missing-data review | A, B | Rules above; still needs a review on the real data |

## References

Hjelmervik et al. (2024), *Proc. IOA* 46(1) · Merchant et al. (2012), *Mar. Pollut. Bull.* 64(7) ·
Roberts et al. (2017), *Ecography* 40(8) · Schoeman, Erbe & Plön (2022), *JMSE* 10(6) ·
Shipton et al. (2025), *Sci. Data* 12, 289

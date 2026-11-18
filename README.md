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
pip install -e ".[dev]"
pytest                      # 27 tests, ~5 s
```

Put the competition CSVs in `data/` (see `data/README.md`). Nothing under
`data/` or `outputs/` is committed.

## Workflow

```bash
# 1. Data-dictionary check: confirm detected columns, units, time step, gaps, missingness
python scripts/inspect_data.py --records data/Train.csv --test data/Test.csv --nrows 300000

# 2. Convert external vessel passages (after checking access and licences)
python scripts/convert_reference_audio.py --in raw/hearmyship --out data/references/hearmyship --sensitivity-db <S>
python scripts/convert_reference_audio.py --in raw/onc        --out data/references/onc        --sensitivity-db <S>

# 3. Experiment + submission
python scripts/run_pipeline.py --records data/Train.csv --test data/Test.csv \
    --refs-hearmyship data/references/hearmyship --refs-onc data/references/onc \
    --loso --source-shift --ablations --out outputs/run1
```

Without the real files, `--synthetic` runs everything on stand-in data, and
`scripts/make_synthetic_data.py` writes stand-in CSVs so the file path can be
tested too. If no reference directories are given, synthetic references are used
and the run prints a warning.

Outputs in `--out`:

- `results.csv`: score, RMSE, MAE, bias, day-block bootstrap CI, ROC/PR AUC, fit time, prediction time and model size, per model and split
- breakdowns by true fraction, by site and by reference source
- `ablations.csv`, `loso.csv` / `loso_summary.csv`, `source_shift.csv`
- `residual_autocorr.csv`
- `submission.csv`
- `run_summary.json`: arguments, split sizes, chosen model, peak memory and wall time

## Code map (`src/vesselnoise/`)

| Module | What it does |
|---|---|
| `config.py` | 25 decidecade bands (31.6 Hz to 7.94 kHz), window length, the candidate column names, the mixture ranges. **Fix schema assumptions here.** |
| `schema.py`, `data.py` | Column detection (station, time, ID, minute, quality, duration, band columns), dB vs linear units, masking of quality-flagged rows |
| `windows.py` | 15-minute windows within one recorder and day that never bridge gaps. Also rebuilds test windows from long format. |
| `energy.py` | dB/linear conversions and masked energy sums. All averaging is in linear energy. |
| `reference.py` | Audio → 1 Hz FFT → band energy → minute bins; own-background subtraction; synthetic stand-in passages |
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

## Development results (synthetic stand-in data only)

These numbers come from **synthetic** backgrounds and references. They only show
that the pipeline works; they say nothing about the competition data.

Command: `python scripts/run_pipeline.py --synthetic --fast --n-train 6000 --n-val 2000 --n-test 2000 --loso --source-shift --ablations`

| Model | Val score | Local test score (95% day-block CI) | LOSO mean / worst site |
|---|---|---|---|
| constant | 0.607 | 0.609 (0.604–0.614) | 0.606 / 0.601 |
| excess-energy baseline | 0.929 | 0.922 (0.904–0.934) | 0.883 / 0.731 |
| Ridge | 0.942 | 0.927 (0.917–0.934) | **0.899 / 0.800** |
| random forest | 0.971 | 0.960 (0.950–0.967) | – |
| histogram GB | **0.975** | **0.964** (0.954–0.972) | 0.846 / 0.619 |

Things to check once the real data are in:

1. **Unseen stations.** The tree models win on held-out days at known sites but drop sharply in leave-one-site-out. On the worst site, histogram GB falls to 0.62. They seem to lean on absolute station levels. Ridge and the excess baseline transfer better. Try dropping the absolute `level` family, or keeping only station-normalised and shape features.
2. **Reference-source shift.** Histogram GB trained on large-ship passages scores 0.61 on small-boat passages. The excess baseline holds at 0.92. Train on both sources, and keep the baseline as a feature.
3. **Label distribution.** On our mixtures the constant model scores 0.61, but the leaderboard's constant benchmark is 0.754, which implies a test-label spread of about 0.246 RMSE. Uniform −20 to +20 dB ATB plus 20% zeros spreads labels more than the test set does. Compare against the real distribution, and adjust `--p-zero` or pass `share_matching_sampler` as `atb_sampler`.
4. **Ablations (histogram GB).** On validation, dropping station_norm (−0.010) or the excess feature (−0.011) hurts much more than dropping temporal shape (−0.001). On local test all three cost between 0.003 and 0.005.
5. **Residual autocorrelation.** The synthetic local-test blocks are only about 3 days per station, too short to estimate lag autocorrelation. With the real record lengths, check `residual_autocorr.csv` before keeping the one-day buffer.

## Status against the proposal timeline

| Item | Owner | Status |
|---|---|---|
| Data dictionary check (`inspect_data.py`) | A | Script ready; run on real `Train.csv` / `Test.csv` |
| Cleaning, windowing, excess-energy baseline | A | Implemented and tested |
| External audio access and licences; conversion | B | Converter ready; HearMyShip / ONC downloads and the forum declaration are still to do |
| Controlled mixtures, feature extraction | B | Implemented (synthetic references until real ones are converted) |
| Constant, Ridge, RF, GB, evidence classifier | C | Implemented |
| Limited tuning | C | To do (Nov 19 – Dec 2) |
| LOSO, source shift, error analysis, leaderboard comparison | D | LOSO and source shift implemented; AIS check needs AIS data (`metrics.ais_contrast` is ready) |
| Leakage and missing-data review | A, B | Rules above; still needs a review on the real data |

## References

Hjelmervik et al. (2024), *Proc. IOA* 46(1) · Merchant et al. (2012), *Mar. Pollut. Bull.* 64(7) ·
Roberts et al. (2017), *Ecography* 40(8) · Schoeman, Erbe & Plön (2022), *JMSE* 10(6) ·
Shipton et al. (2025), *Sci. Data* 12, 289

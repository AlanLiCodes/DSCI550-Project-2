# Data directory (not committed)

Place the competition files here after downloading them from Zindi:

```
data/
  Train.csv          # 1,715,779 unlabelled one-minute band-level records (10 sites)
  Test.csv           # 1,500 windows x 15 minutes x 25 bands -- inference only
  SampleSubmission.csv
  references/        # converted vessel passages (see scripts/convert_reference_audio.py)
    hearmyship/*.csv
    onc/*.csv
```

Exact file names may differ; pass paths explicitly to the scripts. Run
`python scripts/inspect_data.py --records data/Train.csv --test data/Test.csv`
first to confirm that column detection (station, timestamp, band columns,
quality flags, window ID) matches the real schema, and adjust
`src/vesselnoise/config.py` if it does not.

**Test.csv is never used for fitting preprocessing, training or calibration.**

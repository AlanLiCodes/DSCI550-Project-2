# Data directory (not committed)

Place the competition files here after downloading them from Zindi:

```
data/
  South_African_Real_Acoustics/   # unzipped archive: 1,715,779 unlabelled records, 10 stations
    acoustics/*.parquet           # one file per station: site, recorder, t, dur, TOL_<f> ...
    stations.csv, recorders.csv, DATA_DICTIONARY.md
  Test.csv                        # 1,500 rows: ID + 375 cols mMM_TOL_<f> -- inference only
  SampleSubmission.csv
  references/                     # converted vessel passages (scripts/convert_reference_audio.py)
    onc/*.csv                     # VTUAD v1.3.2 (ONC), the source the test additions derive from
    hearmyship/*.csv
```

Run `python scripts/inspect_data.py --records data/South_African_Real_Acoustics --test data/Test.csv`
to reproduce the data checks summarised in the top-level README. The South African data are
under the challenge data-use terms and must not be redistributed.

**Test.csv is never used for fitting preprocessing, training or calibration.**

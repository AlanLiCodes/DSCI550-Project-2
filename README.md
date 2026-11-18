# Estimating Added Vessel Noise in Underwater Acoustic Data

DSCI 550 Project 2. The project enters the Zindi **Underwater Vessel Noise Challenge**.

Team: Alan Li, Yi Sheng Chao, Jeremy Han, Hanqi Ni

The goal is to estimate `share_added_vessel`, the fraction of acoustic energy
in a 15-minute x 25-band window that comes from an added vessel passage.
`score_added_vessel` is produced alongside it as a presence score.

## Setup

```bash
pip install -e ".[dev]"
pytest
```

Put the competition CSVs in `data/` (see `data/README.md`).

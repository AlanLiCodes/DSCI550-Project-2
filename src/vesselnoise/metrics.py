"""Leaderboard metric and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FRACTION_BINS = [-1e-9, 1e-9, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 + 1e-9]


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def leaderboard_score(y, p) -> float:
    """Official metric: 1 - RMSE on share_added_vessel (higher is better)."""
    return 1.0 - rmse(y, p)


def regression_report(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "score": leaderboard_score(y, p),
        "rmse": rmse(y, p),
        "mae": float(np.mean(np.abs(y - p))),
        "bias": float(np.mean(p - y)),
    }


def breakdown(y, p, groups, name="group") -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p, name: groups})
    rows = []
    for g, d in df.groupby(name, observed=True):
        rows.append({name: g, "n": len(d), **regression_report(d["y"], d["p"])})
    return pd.DataFrame(rows)


def by_true_fraction(y, p) -> pd.DataFrame:
    labels = ["0", "(0,.1]", "(.1,.3]", "(.3,.5]", "(.5,.7]", "(.7,.9]", "(.9,1]"]
    bins = pd.cut(np.asarray(y), FRACTION_BINS, labels=labels)
    return breakdown(y, p, bins, "true_fraction")


def block_bootstrap_ci(y, p, blocks, n_boot: int = 500, alpha: float = 0.05, seed: int = 0):
    """CI for the leaderboard score, resampling whole blocks (e.g. station-days)."""
    y, p, blocks = np.asarray(y, float), np.asarray(p, float), np.asarray(blocks)
    se = (y - p) ** 2
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=se)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ub), size=(n_boot, len(ub)))
    scores = 1.0 - np.sqrt(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def detection_report(present, score) -> dict:
    present = np.asarray(present).astype(int)
    if present.min() == present.max():
        return {"roc_auc": np.nan, "pr_auc": np.nan}
    return {"roc_auc": float(roc_auc_score(present, score)), "pr_auc": float(average_precision_score(present, score))}


def ais_contrast(pred_near, pred_far) -> dict:
    """One-sided AIS check: windows with an AIS vessel nearby (positive) vs none within 20 km for an hour."""
    pred_near, pred_far = np.asarray(pred_near, float), np.asarray(pred_far, float)
    y = np.r_[np.ones(len(pred_near)), np.zeros(len(pred_far))]
    return detection_report(y, np.r_[pred_near, pred_far])

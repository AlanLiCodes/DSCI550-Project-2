"""Model zoo. All regressors predict share_added_vessel; outputs are clipped to [0, 1]."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def _imputer():
    # Median imputation fitted on training data + missingness indicators.
    return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)


def regressors(seed: int = config.SEED, fast: bool = False) -> dict:
    n_trees = 100 if fast else 300
    return {
        "constant": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(_imputer(), StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": make_pipeline(
            _imputer(),
            RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=5, max_features=0.4, n_jobs=-1, random_state=seed),
        ),
        "hist_gb": HistGradientBoostingRegressor(
            max_iter=200 if fast else 500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def presence_classifier(seed: int = config.SEED):
    """Logistic regression producing score_added_vessel."""
    return make_pipeline(_imputer(), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=seed))


def predict_clipped(model, X) -> np.ndarray:
    return np.clip(model.predict(X), 0.0, 1.0)

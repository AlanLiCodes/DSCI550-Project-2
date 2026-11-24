"""End-to-end experiment orchestration (used by scripts/run_pipeline.py)."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .features import FeatureBuilder, columns_for
from .metrics import block_bootstrap_ci, breakdown, by_true_fraction, detection_report, regression_report
from .mixtures import make_mixtures
from .models import predict_clipped, presence_classifier, regressors
from .splits import assign_reference_splits, chronological_split

SPLITS = ("train", "val", "test")


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    present: np.ndarray
    info: pd.DataFrame
    F: pd.DataFrame | None = None
    cells: np.ndarray | None = None

    @property
    def blocks(self):
        return (self.info["station"].astype(str) + "_" + pd.to_datetime(self.info["day"]).dt.strftime("%Y%m%d")).to_numpy()


@dataclass
class Experiment:
    splits: dict
    features: FeatureBuilder
    bg_split: np.ndarray
    ref_split: dict
    log: dict = field(default_factory=dict)


def build_experiment(
    X_bg,
    meta_bg,
    refs,
    n_mix=None,
    seed=config.SEED,
    fracs=(0.6, 0.2, 0.2),
    buffer_days=1,
    atb_sampler=None,
    p_zero=config.P_ZERO_ADDITION,
    ref_filter=None,
    station_aware=False,
    mix_kwargs=None,
    hard_negatives: float = 0.0,
    with_cells: bool = False,
):
    """Split backgrounds chronologically and references by vessel, *then* build mixtures.

    ``ref_filter`` maps split name -> predicate on Reference (used for the
    reference-source-shift experiment). ``station_aware=False`` (default) matches
    the competition test set, which carries no station or timestamp.
    ``hard_negatives`` > 0 over-samples training backgrounds that a first-pass model
    wrongly flags (training split only). ``with_cells`` keeps per-cell targets for the CNN.
    """
    mix_kwargs = mix_kwargs or {}
    n_mix = n_mix or {"train": 20000, "val": 5000, "test": 5000}
    bg_split = chronological_split(meta_bg, fracs, buffer_days)
    ref_split = assign_reference_splits(refs, fracs, seed)
    splits = {}
    for i, s in enumerate(SPLITS):
        m = bg_split == s
        if not m.any():
            raise ValueError(f"no background windows in the '{s}' split; records span too few days per station")
        rs = [r for r in refs if ref_split[r.vessel_id] == s]
        if ref_filter and s in ref_filter:
            # Source-shift runs ignore the per-vessel split for the filtered side and
            # use every vessel of the requested source instead.
            rs = [r for r in refs if ref_filter[s](r)]
        kw = dict(mix_kwargs)
        meta_s = meta_bg.loc[m].reset_index(drop=True)
        if s == "train" and hard_negatives > 0:
            kw["bg_weights"], _ = hard_negative_weights(X_bg[m], meta_s, rs, mix_kwargs, strength=hard_negatives, seed=seed)
        out = make_mixtures(
            X_bg[m], meta_s, rs, n_mix[s],
            seed=seed + 101 * i, atb_sampler=atb_sampler, p_zero=p_zero, return_cells=with_cells and s == "train", **kw,
        )
        splits[s] = SplitData(*out[:4], cells=out[4] if len(out) == 5 else None)
    tr = bg_split == "train"
    fb = FeatureBuilder(station_aware=station_aware).fit(
        X_bg[tr], meta_bg.loc[tr].reset_index(drop=True), X_tune=splits["train"].X, y_tune=splits["train"].y, meta_tune=splits["train"].info
    )
    for s in SPLITS:
        splits[s].F = fb.transform(splits[s].X, splits[s].info)
    log = {
        "background_windows": {s: int((bg_split == s).sum()) for s in [*SPLITS, "buffer"]},
        "reference_vessels": {s: sum(v == s for v in ref_split.values()) for s in SPLITS},
        "baseline": {"threshold_db": fb.baseline.threshold_db, "k_quiet": fb.baseline.k_quiet, "combine": fb.baseline.combine},
        "station_aware": station_aware,
        "mixture_label_mean": {s: float(splits[s].y.mean()) for s in SPLITS},
    }
    return Experiment(splits, fb, bg_split, ref_split, log)


def _model_size_kb(model) -> float:
    try:
        return len(pickle.dumps(model)) / 1024.0
    except Exception:  # pragma: no cover
        return float("nan")


def evaluate_models(exp: Experiment, models=None, eval_splits=("val", "test"), drop_families=(), seed=config.SEED, fast=False, with_ci=True, cnn_kwargs=None):
    """Fit each model on train mixtures, score on val/test. Returns (results, predictions, fitted)."""
    models = models or regressors(seed, fast=fast)
    tr = exp.splits["train"]
    cols = columns_for(tr.F, drop_families)
    rows, preds, fitted = [], {}, {}

    def record(name, split, p, fit_s=np.nan, pred_s=np.nan, size_kb=np.nan):
        d = exp.splits[split]
        r = {"model": name, "split": split, "n": len(d.y), **regression_report(d.y, p), **detection_report(d.present, p)}
        if with_ci:
            r["ci_lo"], r["ci_hi"] = block_bootstrap_ci(d.y, p, d.blocks, seed=seed)
        ms_per_1k = 1e6 * pred_s / max(len(d.y), 1)  # NaN propagates for label-free baselines
        r.update(fit_s=fit_s, predict_ms_per_1k=ms_per_1k, size_kb=size_kb)
        rows.append(r)
        preds[(name, split)] = p

    if "excess" not in drop_families:
        for s in eval_splits:
            record("excess_baseline", s, exp.splits[s].F["excess_share"].to_numpy())

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(tr.F[cols], tr.y)
        fit_s = time.perf_counter() - t0
        fitted[name] = model
        for s in eval_splits:
            t1 = time.perf_counter()
            p = predict_clipped(model, exp.splits[s].F[cols])
            record(name, s, p, fit_s, time.perf_counter() - t1, _model_size_kb(model))

    if cnn_kwargs is not None:
        if tr.cells is None:
            raise ValueError("CNN evaluation needs build_experiment(..., with_cells=True)")
        from .cnn import CellCNN

        t0 = time.perf_counter()
        va = exp.splits["val"]
        cnn = CellCNN(seed=seed, **cnn_kwargs).fit(tr.X, tr.y, tr.cells, X_val=va.X, y_val=va.y)
        fit_s = time.perf_counter() - t0
        fitted["cnn"] = cnn
        for s in eval_splits:
            t1 = time.perf_counter()
            p = cnn.predict(exp.splits[s].X)
            record("cnn", s, p, fit_s, time.perf_counter() - t1)
            if ("hist_gb", s) in preds:
                record("ensemble_gb_cnn", s, 0.5 * (p + preds[("hist_gb", s)]))

    clf = presence_classifier(seed)
    clf.fit(tr.F[cols], tr.present)
    fitted["presence_logreg"] = clf
    for s in eval_splits:
        d = exp.splits[s]
        sc = clf.predict_proba(d.F[cols])[:, 1]
        preds[("presence_logreg", s)] = sc
        rows.append({"model": "presence_logreg", "split": s, "n": len(d.y), **detection_report(d.present, sc)})
    return pd.DataFrame(rows), preds, fitted


def diagnostics(exp: Experiment, preds: dict, model: str, split: str = "test"):
    d = exp.splits[split]
    p = preds[(model, split)]
    return {
        "by_fraction": by_true_fraction(d.y, p).assign(model=model, split=split),
        "by_site": breakdown(d.y, p, d.info["station"].to_numpy(), "station").assign(model=model, split=split),
        "by_source": breakdown(d.y, p, d.info["source"].fillna("none").to_numpy(), "source").assign(model=model, split=split),
    }


def run_ablations(exp: Experiment, families=("level", "temporal", "spectral", "excess", "station_norm"), model_name="hist_gb", seed=config.SEED, fast=False):
    out = []
    present = set(exp.splits["train"].F.attrs.get("families", {}).values())
    for fam in [None, *[f for f in families if f in present]]:
        drop = () if fam is None else (fam,)
        m = {model_name: regressors(seed, fast=fast)[model_name]}
        res, _, _ = evaluate_models(exp, m, drop_families=drop, seed=seed, with_ci=False)
        res = res[res["model"] == model_name].assign(dropped=fam or "none")
        out.append(res)
    return pd.concat(out, ignore_index=True)


def run_loso(
    X_bg, meta_bg, refs, n_train=10000, n_test=3000, seed=config.SEED, model_names=("ridge", "hist_gb"),
    fast=True, station_aware=False, drop_families=(), mix_kwargs=None,
):
    """Leave-one-site-out: train on mixtures from nine sites, test on the held-out site.

    Station normalisation for the held-out site falls back to the global median,
    as it would for an unseen station. Reference vessels stay split train/test.
    """
    mix_kwargs = mix_kwargs or {}
    ref_split = assign_reference_splits(refs, (0.75, 0.0, 0.25), seed)
    refs_tr = [r for r in refs if ref_split[r.vessel_id] == "train"]
    refs_te = [r for r in refs if ref_split[r.vessel_id] == "test"]
    rows = []
    for j, site in enumerate(sorted(meta_bg["station"].unique())):
        m_te = (meta_bg["station"] == site).to_numpy()
        mtr, mte = meta_bg.loc[~m_te].reset_index(drop=True), meta_bg.loc[m_te].reset_index(drop=True)
        Xtr, ytr, ptr, itr = make_mixtures(X_bg[~m_te], mtr, refs_tr, n_train, seed=seed + j, **mix_kwargs)
        Xte, yte, pte, ite = make_mixtures(X_bg[m_te], mte, refs_te, n_test, seed=seed + 1000 + j, **mix_kwargs)
        fb = FeatureBuilder(station_aware=station_aware).fit(X_bg[~m_te], mtr, X_tune=Xtr, y_tune=ytr, meta_tune=itr)
        Ftr, Fte = fb.transform(Xtr, itr), fb.transform(Xte, ite)
        cols = columns_for(Ftr, drop_families)
        Ftr, Fte = Ftr[cols], Fte[cols]
        rows.append({"site": site, "model": "constant", **regression_report(yte, np.full_like(yte, ytr.mean()))})
        if "excess_share" in Fte:
            rows.append({"site": site, "model": "excess_baseline", **regression_report(yte, Fte["excess_share"])})
        zoo = regressors(seed, fast=fast)
        for name in model_names:
            mdl = zoo[name].fit(Ftr, ytr)
            rows.append({"site": site, "model": name, **regression_report(yte, predict_clipped(mdl, Fte))})
    res = pd.DataFrame(rows)
    summary = res.groupby("model")["score"].agg(mean="mean", worst="min", best="max").reset_index()
    return res, summary


def hard_negative_weights(X_bg, meta_bg, refs, mix_kwargs=None, strength=10.0, n=12000, folds=2, seed=config.SEED):
    """Sampling weights that over-sample backgrounds a first-pass model wrongly flags.

    Station-days are split into ``folds``; for each fold a quick histogram-GB model is
    trained on mixtures built from the other folds and predicts the *clean* windows
    of the held-out fold (true share 0). weight = 1 + strength * prediction, so a
    clean window predicted at 0.3 is drawn ~4x as often. Uses training data only.

    Returns (weights, out-of-fold predictions on clean backgrounds).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    mix_kwargs = dict(mix_kwargs or {})
    mix_kwargs.pop("bg_weights", None)
    keys = meta_bg["station"].astype(str) + "_" + pd.to_datetime(meta_bg["day"]).dt.strftime("%Y%m%d")
    uk = np.unique(keys)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uk, rng.integers(0, folds, size=len(uk))))
    fold = keys.map(fold_of).to_numpy()
    pred = np.zeros(len(X_bg))
    for f in range(folds):
        tr, te = fold != f, fold == f
        mtr = meta_bg.loc[tr].reset_index(drop=True)
        Xm, y, _, info = make_mixtures(X_bg[tr], mtr, refs, n, seed=seed + 31 * f, **mix_kwargs)
        fb = FeatureBuilder(station_aware=False).fit(X_bg[tr], mtr)
        mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=seed).fit(fb.transform(Xm, info), y)
        pred[te] = np.clip(mdl.predict(fb.transform(X_bg[te], meta_bg.loc[te].reset_index(drop=True))), 0, 1)
    return 1.0 + strength * pred, pred


def fit_final_and_predict(
    X_bg, meta_bg, refs, X_test, meta_test, model_name="hist_gb", n_train=25000, seed=config.SEED,
    atb_sampler=None, fast=False, drop_families=(), mix_kwargs=None, hard_negatives=0.0, cnn_kwargs=None,
):
    """Train on mixtures from all non-test background windows and predict the competition test set.

    ``model_name``: a regressor from ``models.regressors``, "excess_baseline", "cnn",
    or "ensemble_gb_cnn" (mean of histogram GB and CNN). Test windows are used only
    for inference. Features are station-agnostic unless the test metadata carries a
    station column. Returns (share, score); the evidence score is the predicted
    share itself (the separate presence classifier was at chance on the leaderboard).
    """
    station_aware = "station" in meta_test.columns
    mix_kwargs = dict(mix_kwargs or {})
    if hard_negatives > 0:
        mix_kwargs["bg_weights"], _ = hard_negative_weights(X_bg, meta_bg, refs, mix_kwargs, strength=hard_negatives, seed=seed)
    need_cnn = model_name in ("cnn", "ensemble_gb_cnn")
    out = make_mixtures(X_bg, meta_bg, refs, n_train, seed=seed + 7, atb_sampler=atb_sampler, return_cells=need_cnn, **mix_kwargs)
    X, y, pres, info = out[:4]
    preds = []
    if model_name != "cnn":
        fb = FeatureBuilder(station_aware=station_aware).fit(X_bg, meta_bg, X_tune=X, y_tune=y, meta_tune=info)
        F = fb.transform(X, info)
        cols = columns_for(F, drop_families)
        F, Ft = F[cols], fb.transform(X_test, meta_test)[cols]
        if model_name == "excess_baseline":
            preds.append(Ft["excess_share"].to_numpy())
        else:
            tree = "hist_gb" if model_name == "ensemble_gb_cnn" else model_name
            preds.append(predict_clipped(regressors(seed, fast=fast)[tree].fit(F, y), Ft))
    if need_cnn:
        from .cnn import CellCNN

        preds.append(CellCNN(seed=seed, **(cnn_kwargs or {})).fit(X, y, out[4]).predict(X_test))
    share = np.clip(np.mean(preds, axis=0), 0.0, 1.0)
    return share, share.copy()

#!/usr/bin/env python
"""Run the full experiment: mixtures -> features -> models -> evaluation (-> submission).

Examples
--------
# Development run on synthetic stand-in data (no downloads needed)
python scripts/run_pipeline.py --synthetic --fast --out outputs/synthetic

# Real data (unzipped South_African_Real_Acoustics), synthetic references until ONC/VTUAD are converted
python scripts/run_pipeline.py --records data/South_African_Real_Acoustics --test data/Test.csv \
    --sample-submission data/SampleSubmission.csv --loso --out outputs/run1

# Real data with converted references, all evaluations
python scripts/run_pipeline.py --records data/South_African_Real_Acoustics --test data/Test.csv \
    --refs-hearmyship data/references/hearmyship --refs-onc data/references/onc \
    --loso --source-shift --ablations --out outputs/run2
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vesselnoise import config  # noqa: E402
from vesselnoise.data import load_records, load_test, standardise  # noqa: E402
from vesselnoise.pipeline import (  # noqa: E402
    build_experiment,
    diagnostics,
    evaluate_models,
    fit_final_and_predict,
    run_ablations,
    run_loso,
)
from vesselnoise.reference import load_reference_dir, synthetic_reference_set  # noqa: E402
from vesselnoise.vtuad import load_vtuad_references  # noqa: E402
from vesselnoise.splits import daily_residual_autocorr  # noqa: E402
from vesselnoise.synth import synthetic_records, synthetic_test  # noqa: E402
from vesselnoise.windows import build_windows, windows_from_long  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", help="training records: unzipped archive dir (acoustics/*.parquet), a parquet or a CSV")
    ap.add_argument("--test", help="competition test CSV (inference only)")
    ap.add_argument("--sample-submission", help="SampleSubmission.csv to copy column order / IDs from")
    ap.add_argument("--refs-hearmyship", help="dir of converted HearMyShip passages")
    ap.add_argument("--refs-onc", help="dir of converted ONC passages (generic per-clip CSVs)")
    ap.add_argument("--refs-vtuad", help="output dir of scripts/convert_vtuad.py (clips.csv + vessel/ + background/)")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic stand-in records/test/references")
    ap.add_argument("--nrows", type=int, help="read only the first N records (debugging)")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--p-zero", type=float, default=config.P_ZERO_ADDITION)
    ap.add_argument("--minute-dropout", type=float, default=config.MINUTE_DROPOUT_P, help="share of backgrounds with masked whole minutes")
    ap.add_argument("--stretch-max", type=float, default=1.0, help="max time-stretch of references (1 = off)")
    ap.add_argument("--station-balance", type=float, default=0.5, help="0 = sample windows uniformly, 1 = stations uniformly")
    ap.add_argument("--station-aware", action="store_true", help="use station/hour features in experiments (not usable for Test.csv)")
    ap.add_argument("--drop-families", default="", help="comma-separated feature families to drop everywhere, e.g. level")
    ap.add_argument("--loso", action="store_true", help="leave-one-site-out cross-validation")
    ap.add_argument("--source-shift", action="store_true", help="train on one reference source, test on the other")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--final-model", default="auto", help="model for the submission (auto = best by blocked CV)")
    ap.add_argument("--fast", action="store_true", help="smaller models / fewer trees")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default="outputs/run")
    return ap.parse_args()


def load_refs(args):
    refs = []
    if args.refs_hearmyship:
        refs += load_reference_dir(args.refs_hearmyship, "hearmyship")
    if args.refs_onc:
        refs += load_reference_dir(args.refs_onc, "onc")
    if args.refs_vtuad:
        refs += load_vtuad_references(args.refs_vtuad)
    if not refs:
        print("[refs] no converted references given -> using SYNTHETIC references (development only)")
        refs = synthetic_reference_set(seed=args.seed)
    print(f"[refs] {len(refs)} passages: " + ", ".join(f"{s}={n}" for s, n in pd.Series([r.source for r in refs]).value_counts().items()))
    return refs


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    np.random.seed(args.seed)

    # ---- data ----------------------------------------------------------------
    hidden_labels = None
    X_test = meta_test = None
    id_name = "ID"
    if args.synthetic:
        raw = synthetic_records(seed=args.seed)
        records, schema = standardise(raw)
        test_raw, hidden_labels = synthetic_test(records, n=500, seed=args.seed + 1)
        test_std, _ = standardise(test_raw)
        X_test, meta_test = windows_from_long(test_std)
    else:
        if not args.records:
            sys.exit("--records is required unless --synthetic")
        records, schema = load_records(args.records, nrows=args.nrows)
        if args.test:
            X_test, meta_test, id_name = load_test(args.test)
            print(f"[test] {len(X_test)} windows, missing cells {np.isnan(X_test).mean():.2%}, station column: {'station' in meta_test}")
    print("[schema]\n" + schema.describe())
    drop = tuple(f for f in args.drop_families.split(",") if f)
    mix_kwargs = {
        "minute_dropout": (args.minute_dropout, config.MINUTE_DROPOUT_MAX),
        "station_balance": args.station_balance,
        "stretch_range": (1.0, args.stretch_max),
    }

    X_bg, meta_bg = build_windows(records)
    print(f"[windows] {len(X_bg)} background windows from {meta_bg['station'].nunique()} stations")
    refs = load_refs(args)

    # ---- main blocked evaluation ----------------------------------------------
    n_mix = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    exp = build_experiment(
        X_bg, meta_bg, refs, n_mix, seed=args.seed, p_zero=args.p_zero, station_aware=args.station_aware, mix_kwargs=mix_kwargs
    )
    print("[split]", json.dumps(exp.log))
    results, preds, fitted = evaluate_models(exp, seed=args.seed, fast=args.fast, drop_families=drop)
    results.to_csv(out / "results.csv", index=False)
    show = ["model", "split", "score", "rmse", "mae", "bias", "ci_lo", "ci_hi", "roc_auc", "pr_auc", "fit_s"]
    print("\n[results]\n" + results[[c for c in show if c in results]].round(4).to_string(index=False))

    reg_val = results[(results["split"] == "val") & results["score"].notna()]
    best_val = reg_val.sort_values("score", ascending=False)["model"].iloc[0]
    diag = diagnostics(exp, preds, best_val, "test")
    for k, v in diag.items():
        v.to_csv(out / f"breakdown_{k}.csv", index=False)
    print(f"\n[by true fraction, {best_val}, local test]\n" + diag["by_fraction"].round(4).to_string(index=False))

    d = exp.splits["test"]
    ac = daily_residual_autocorr(preds[(best_val, "test")] - d.y, d.info)
    ac.to_csv(out / "residual_autocorr.csv")
    print("\n[daily residual autocorrelation by lag]\n" + ac.round(3).to_string())

    summary = {"best_by_val": best_val}

    # ---- optional evaluations --------------------------------------------------
    if args.ablations:
        abl = run_ablations(exp, seed=args.seed, fast=args.fast)
        abl.to_csv(out / "ablations.csv", index=False)
        print("\n[ablations]\n" + abl[["dropped", "split", "score", "rmse"]].round(4).to_string(index=False))

    if args.loso:
        loso, loso_sum = run_loso(
            X_bg, meta_bg, refs, n_train=max(args.n_train // 2, 2000), n_test=max(args.n_test // 2, 1000), seed=args.seed,
            fast=True, station_aware=args.station_aware, drop_families=drop, mix_kwargs=mix_kwargs,
        )
        loso.to_csv(out / "loso.csv", index=False)
        loso_sum.to_csv(out / "loso_summary.csv", index=False)
        print("\n[leave-one-site-out]\n" + loso_sum.round(4).to_string(index=False))
        loso_sum["rank_key"] = (loso_sum["mean"] + loso_sum["worst"]) / 2
        summary["best_by_loso"] = loso_sum.sort_values("rank_key", ascending=False)["model"].iloc[0]

    if args.source_shift:
        sources = sorted({r.source for r in refs})
        if len(sources) >= 2:
            a, b = sources[0], sources[1]
            rows = []
            for tr_src, te_src in [(a, b), (b, a)]:
                ex = build_experiment(
                    X_bg, meta_bg, refs, n_mix, seed=args.seed,
                    ref_filter={"train": lambda r, s=tr_src: r.source == s, "val": lambda r, s=tr_src: r.source == s, "test": lambda r, s=te_src: r.source == s},
                    station_aware=args.station_aware, mix_kwargs=mix_kwargs,
                )
                res, _, _ = evaluate_models(ex, seed=args.seed, fast=True, eval_splits=("test",), with_ci=False)
                rows.append(res.assign(train_source=tr_src, test_source=te_src))
            shift = pd.concat(rows, ignore_index=True)
            shift.to_csv(out / "source_shift.csv", index=False)
            print("\n[reference-source shift]\n" + shift[["train_source", "test_source", "model", "score", "rmse"]].dropna().round(4).to_string(index=False))
        else:
            print("[source-shift] needs two reference sources; skipped")

    # ---- submission ------------------------------------------------------------
    if X_test is not None:
        final = args.final_model if args.final_model != "auto" else summary.get("best_by_loso", best_val)
        if final in ("constant", "presence_logreg"):
            final = "hist_gb"
        keep = exp.bg_split != "buffer"
        share, score = fit_final_and_predict(
            X_bg[keep], meta_bg.loc[keep].reset_index(drop=True), refs, X_test, meta_test,
            model_name=final, n_train=args.n_train + args.n_val, seed=args.seed, fast=args.fast,
            drop_families=drop, mix_kwargs=mix_kwargs,
        )
        sub = pd.DataFrame({id_name: meta_test["window_id"].to_numpy(), config.TARGET_SHARE: share, config.TARGET_SCORE: score})
        if args.sample_submission:
            ss = pd.read_csv(args.sample_submission)
            sub = ss[[ss.columns[0]]].merge(sub.rename(columns={id_name: ss.columns[0]}), on=ss.columns[0], how="left")[ss.columns]
        assert sub[id_name].is_unique and len(sub) == len(X_test)
        assert np.isfinite(sub[[config.TARGET_SHARE, config.TARGET_SCORE]].to_numpy()).all()
        assert sub[[config.TARGET_SHARE, config.TARGET_SCORE]].stack().between(0, 1).all()
        sub.to_csv(out / "submission.csv", index=False)
        summary["final_model"] = final
        print(f"\n[submission] {len(sub)} rows with {final} -> {out / 'submission.csv'}")
        if hidden_labels is not None:
            m = hidden_labels.merge(sub, on="ID", suffixes=("", "_pred"))
            s = 1 - np.sqrt(np.mean((m[config.TARGET_SHARE] - m[config.TARGET_SHARE + "_pred"]) ** 2))
            summary["synthetic_test_score"] = float(s)
            print(f"[synthetic hidden-label score] {s:.4f}")

    summary["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary["wall_s"] = time.time() - t_start
    summary["args"] = vars(args)
    summary["split_log"] = exp.log
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[done] {summary['wall_s']:.0f}s, peak RSS {summary['peak_rss_mb']:.0f} MB -> {out}")


if __name__ == "__main__":
    main()

"""End-to-end reliability analysis of the UCI Maternal Health Risk data.

Usage:
    python scripts/run_all.py            # full run (about a few minutes on a laptop CPU)
    python scripts/run_all.py --quick    # fewer repeats and trees, used in CI

Outputs go to results/: results.json, summary.md and figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mhr import conformal, data, metrics, splits  # noqa: E402

N_SPLITS = 5
ALPHA = 0.10
TARGET_SENSITIVITY = 0.90
AGE_BANDS = [(0, 19, "under 20"), (20, 34, "20 to 34"), (35, 200, "35 and over")]


def make_models(n_trees: int):
    return {
        "logistic_regression": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": lambda s: RandomForestClassifier(n_estimators=n_trees, random_state=s, n_jobs=-1),
        "gradient_boosting": lambda s: HistGradientBoostingClassifier(random_state=s),
    }


def oof_proba(model_fn, X, y, folds, seed):
    proba = np.zeros((len(y), 3))
    for tr, te in folds:
        m = model_fn(seed).fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])
    return proba


def summarize(values):
    v = np.asarray(values, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"mean": float(v.mean()), "p2_5": float(lo), "p97_5": float(hi), "n_repeats": int(len(v))}


def inner_group_splits(y_tr, g_tr, seed, n_splits=4):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y_tr)), y_tr, g_tr))


def leakage_audit(X, y, n_seeds):
    fr = []
    for s in range(n_seeds):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
        fr.append(splits.test_rows_seen_in_train(X, tr, te))
    return summarize(fr)


def scheme_comparison(models, X, y, groups, X_u, y_u, g_u, repeats):
    out = {}
    for name, fn in models.items():
        out[name] = {}
        for scheme in ["row_level", "grouped", "deduplicated_grouped"]:
            accs, f1s = [], []
            for r in range(repeats):
                if scheme == "row_level":
                    folds = splits.row_level_folds(y, N_SPLITS, r)
                    p = oof_proba(fn, X, y, folds, r)
                    yy = y
                elif scheme == "grouped":
                    folds = splits.grouped_folds(y, groups, N_SPLITS, r)
                    p = oof_proba(fn, X, y, folds, r)
                    yy = y
                else:
                    folds = splits.grouped_folds(y_u, g_u, N_SPLITS, r)
                    p = oof_proba(fn, X_u, y_u, folds, r)
                    yy = y_u
                accs.append(metrics.accuracy(yy, p))
                f1s.append(metrics.macro_f1(yy, p))
            out[name][scheme] = {"accuracy": summarize(accs), "macro_f1": summarize(f1s)}
    return out


def calibration_study(models, X, y, groups, repeats):
    out, keep = {}, {}
    for name, fn in models.items():
        raw_m, cal_m = {"ece": [], "brier": [], "nll": []}, {"ece": [], "brier": [], "nll": []}
        for r in range(repeats):
            folds = splits.grouped_folds(y, groups, N_SPLITS, r)
            p_raw = np.zeros((len(y), 3))
            p_cal = np.zeros((len(y), 3))
            for tr, te in folds:
                p_raw[te] = fn(r).fit(X[tr], y[tr]).predict_proba(X[te])
                inner = inner_group_splits(y[tr], groups[tr], r)
                cal = CalibratedClassifierCV(fn(r), method="sigmoid", cv=inner).fit(X[tr], y[tr])
                p_cal[te] = cal.predict_proba(X[te])
            for d, p in ((raw_m, p_raw), (cal_m, p_cal)):
                d["ece"].append(metrics.top_label_ece(y, p))
                d["brier"].append(metrics.brier(y, p))
                d["nll"].append(metrics.nll(y, p))
            if r == 0:
                keep[name] = (p_raw, p_cal)
        out[name] = {
            "uncalibrated": {k: summarize(v) for k, v in raw_m.items()},
            "sigmoid_calibrated": {k: summarize(v) for k, v in cal_m.items()},
        }
    return out, keep


def screening_study(model_fn, X, y, groups, repeats):
    """Flag 'high risk' when P(high) >= t, with t chosen on inner out-of-fold scores of the training fold."""
    sens, spec, flag_rate = [], [], []
    for r in range(repeats):
        folds = splits.grouped_folds(y, groups, N_SPLITS, r)
        flags = np.zeros(len(y), dtype=bool)
        for tr, te in folds:
            inner = inner_group_splits(y[tr], groups[tr], r)
            p_inner = oof_proba(model_fn, X[tr], y[tr], inner, r)[:, 2]
            pos = p_inner[y[tr] == 2]
            t = float(np.quantile(pos, 1 - TARGET_SENSITIVITY, method="lower"))
            p_te = model_fn(r).fit(X[tr], y[tr]).predict_proba(X[te])[:, 2]
            flags[te] = p_te >= t
        se, sp = metrics.recall_specificity(y == 2, flags)
        sens.append(se)
        spec.append(sp)
        flag_rate.append(float(flags.mean()))
    return {"target_sensitivity_on_training_data": TARGET_SENSITIVITY,
            "sensitivity": summarize(sens), "specificity": summarize(spec), "flag_rate": summarize(flag_rate)}


def conformal_study(model_fn, X, y, groups, repeats):
    cov, size, single = [], [], []
    per_class = {k: [] for k in range(3)}
    for r in range(repeats):
        folds = splits.grouped_folds(y, groups, N_SPLITS, r)
        sets = np.zeros((len(y), 3), dtype=bool)
        rng = np.random.default_rng(r)
        for tr, te in folds:
            g_tr = groups[tr]
            uniq = np.unique(g_tr)
            cal_groups = set(rng.choice(uniq, size=max(1, len(uniq) // 4), replace=False).tolist())
            is_cal = np.array([g in cal_groups for g in g_tr])
            fit_idx, cal_idx = tr[~is_cal], tr[is_cal]
            m = model_fn(r).fit(X[fit_idx], y[fit_idx])
            qhat = conformal.lac_threshold(m.predict_proba(X[cal_idx]), y[cal_idx], ALPHA)
            sets[te] = conformal.lac_sets(m.predict_proba(X[te]), qhat)
        c, s = conformal.coverage_and_size(sets, y)
        cov.append(c)
        size.append(s)
        single.append(float((sets.sum(axis=1) == 1).mean()))
        for k in range(3):
            per_class[k].append(float(sets[y == k, k].mean()))
    return {"alpha": ALPHA, "coverage": summarize(cov), "mean_set_size": summarize(size),
            "singleton_rate": summarize(single),
            "coverage_by_true_class": {data.CLASSES[k]: summarize(v) for k, v in per_class.items()}}


def age_band_table(df, y, groups, p_oof):
    pred = p_oof.argmax(axis=1)
    rows = []
    for lo, hi, label in AGE_BANDS:
        m = ((df["Age"] >= lo) & (df["Age"] <= hi)).to_numpy()
        n_high = int(np.sum(m & (y == 2)))
        if n_high == 0:
            rows.append({"band": label, "n": int(m.sum()), "n_high_risk": 0})
            continue
        idx_band = np.flatnonzero(m)

        def recall_fn(ix, idx_band=idx_band):
            sel = ix[np.isin(ix, idx_band)]
            yh = y[sel] == 2
            return float((pred[sel][yh] == 2).mean()) if yh.any() else np.nan

        rec = float((pred[m & (y == 2)] == 2).mean())
        lo_ci, hi_ci = metrics.bootstrap_ci(recall_fn, len(y), n_boot=1000, seed=0, groups=groups)
        rows.append({"band": label, "n": int(m.sum()), "n_high_risk": n_high,
                     "high_risk_recall": rec, "ci95_group_bootstrap": [lo_ci, hi_ci]})
    return rows


def reliability_figure(y, p_raw, p_cal, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    edges = np.linspace(0, 1, 11)
    for p, lab in ((p_raw, "uncalibrated"), (p_cal, "sigmoid calibrated")):
        conf = p.max(axis=1)
        corr = (p.argmax(axis=1) == y).astype(float)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, 9)
        xs, ys = [], []
        for b in range(10):
            mm = idx == b
            if mm.sum() >= 5:
                xs.append(conf[mm].mean())
                ys.append(corr[mm].mean())
        ax.plot(xs, ys, marker="o", label=lab)
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
    ax.set_xlabel("top-label confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Random forest, duplicate-aware CV")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(quick: bool):
    t0 = time.time()
    repeats = 3 if quick else 10
    n_trees = 100 if quick else 300
    df, X, y = data.load()
    groups = data.duplicate_groups(X)
    uniq_df = df.drop_duplicates().reset_index(drop=True)
    X_u = uniq_df[data.FEATURES].to_numpy(dtype=float)
    y_u = uniq_df[data.TARGET].map({c: i for i, c in enumerate(data.CLASSES)}).to_numpy()
    g_u = data.duplicate_groups(X_u)
    models = make_models(n_trees)

    results = {
        "data_sha256": data.file_sha256(),
        "audit": data.audit(df),
        "leakage_random_80_20_split": leakage_audit(X, y, 50 if not quick else 10),
    }
    results["accuracy_by_split_scheme"] = scheme_comparison(models, X, y, groups, X_u, y_u, g_u, repeats)
    results["calibration_grouped_cv"], keep = calibration_study(models, X, y, groups, repeats)
    rf = models["random_forest"]
    results["screening_high_risk_random_forest"] = screening_study(rf, X, y, groups, repeats)
    results["conformal_lac_random_forest"] = conformal_study(rf, X, y, groups, repeats)
    p_raw, p_cal = keep["random_forest"]
    results["age_bands_random_forest_grouped_cv_seed0"] = age_band_table(df, y, groups, p_raw)
    results["settings"] = {"quick": quick, "repeats": repeats, "n_trees": n_trees, "n_splits": N_SPLITS,
                           "runtime_seconds": round(time.time() - t0, 1)}

    out = ROOT / "results"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    reliability_figure(y, p_raw, p_cal, out / "figures" / "reliability_random_forest.png")
    write_summary(results, out / "summary.md")
    print(f"done in {results['settings']['runtime_seconds']} s")


def fmt(s, pct=True):
    if pct:
        return f"{100 * s['mean']:.1f}% ({100 * s['p2_5']:.1f} to {100 * s['p97_5']:.1f})"
    return f"{s['mean']:.3f} ({s['p2_5']:.3f} to {s['p97_5']:.3f})"


def write_summary(r, path):
    a = r["audit"]
    lines = ["# Results summary", "",
             f"Data file SHA256: `{r['data_sha256']}`", "",
             "## Data audit", "",
             f"- Rows: {a['rows']}; unique rows: {a['unique_rows']}; redundant exact copies: {a['redundant_full_row_copies']}.",
             f"- Feature vectors shared by two or more rows: rows involved {a['rows_sharing_a_feature_vector']}; largest group {a['largest_group_size']}.",
             f"- Feature vectors that carry conflicting labels: {a['feature_vectors_with_conflicting_labels']}.",
             f"- Heart rate below 40: {a['heart_rate_below_40']} rows; age below 15: {a['age_below_15']}; age above 50: {a['age_above_50']}.",
             "",
             "## Leakage under a random 80/20 split", "",
             "Share of test rows whose exact feature vector also appears in the training split "
             f"(across seeds): {fmt(r['leakage_random_80_20_split'])}.", "",
             "## Accuracy and macro F1 by split scheme", "",
             "Mean across repeated 5-fold cross-validation, with the 2.5 to 97.5 percentile range across repeats.", "",
             "| Model | Scheme | Accuracy | Macro F1 |", "|---|---|---|---|"]
    for m, d in r["accuracy_by_split_scheme"].items():
        for s, v in d.items():
            lines.append(f"| {m} | {s} | {fmt(v['accuracy'])} | {fmt(v['macro_f1'])} |")
    lines += ["", "## Calibration under duplicate-aware cross-validation", "",
              "| Model | Version | Top-label ECE | Brier | NLL |", "|---|---|---|---|---|"]
    for m, d in r["calibration_grouped_cv"].items():
        for ver, v in d.items():
            lines.append(f"| {m} | {ver} | {fmt(v['ece'], False)} | {fmt(v['brier'], False)} | {fmt(v['nll'], False)} |")
    s = r["screening_high_risk_random_forest"]
    lines += ["", "## Screening operating point (random forest)", "",
              f"Threshold on P(high risk) chosen inside each training fold to reach {int(100 * s['target_sensitivity_on_training_data'])}% "
              "sensitivity on inner out-of-fold scores, then applied to the held-out fold.", "",
              f"- Held-out sensitivity: {fmt(s['sensitivity'])}",
              f"- Held-out specificity: {fmt(s['specificity'])}",
              f"- Share of all mothers flagged: {fmt(s['flag_rate'])}"]
    c = r["conformal_lac_random_forest"]
    lines += ["", f"## Conformal prediction sets (random forest, target coverage {int(100 * (1 - c['alpha']))}%)", "",
              f"- Coverage: {fmt(c['coverage'])}",
              f"- Mean set size: {fmt(c['mean_set_size'], False)}",
              f"- Single-label sets: {fmt(c['singleton_rate'])}", "",
              "| True class | Coverage |", "|---|---|"]
    for k, v in c["coverage_by_true_class"].items():
        lines.append(f"| {k} | {fmt(v)} |")
    lines += ["", "## Age bands (random forest, duplicate-aware CV, seed 0)", "",
              "| Age band | Rows | High-risk rows | High-risk recall | 95% group bootstrap interval |", "|---|---|---|---|---|"]
    for row in r["age_bands_random_forest_grouped_cv_seed0"]:
        if row.get("n_high_risk", 0) == 0:
            lines.append(f"| {row['band']} | {row['n']} | 0 | not defined | not defined |")
        else:
            lo, hi = row["ci95_group_bootstrap"]
            lines.append(f"| {row['band']} | {row['n']} | {row['n_high_risk']} | {100 * row['high_risk_recall']:.1f}% | {100 * lo:.1f}% to {100 * hi:.1f}% |")
    st = r["settings"]
    lines += ["", f"Settings: repeats {st['repeats']}, trees {st['n_trees']}, folds {st['n_splits']}, runtime {st['runtime_seconds']} s."]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)

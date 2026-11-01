"""Train and compare Logistic Regression, SVM and Random Forest for SQLi detection.

Usage:
    python train.py --data-dir path/to/sqli_by_type --per-class 20000
"""
import argparse
import json
import random
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from augment import extract_fragment, make_benign, make_malicious_templates
from common import BASE_DIR, normalize
from tests.holdout import BENIGN as HO_BENIGN, MALICIOUS as HO_MALICIOUS

ATTACK_FILES = ["boolean_based", "error_based", "meta_based", "stackqueries_based", "time_based", "union_based"]
BENIGN_FILE = "none_type_benign"
CHUNK = 500_000


def sample_csv(path: Path, n: int, seed: int) -> pd.DataFrame:
    """Memory-safe random sample of n rows from a (possibly huge) CSV."""
    total = sum(1 for _ in open(path, "rb")) - 1  # fast row estimate
    frac = min(1.0, n * 1.05 / max(total, 1))
    parts = []
    for i, chunk in enumerate(pd.read_csv(path, usecols=["sql_query"], chunksize=CHUNK)):
        parts.append(chunk.sample(frac=frac, random_state=seed + i))
    df = pd.concat(parts).dropna()
    return df.sample(n=min(n, len(df)), random_state=seed)


def build_dataset(data_dir: Path, per_class: int, seed: int) -> pd.DataFrame:
    rows = []
    for name in ATTACK_FILES:
        t = time.time()
        df = sample_csv(data_dir / f"{name}.csv", per_class, seed)
        df["label"], df["source"], df["attack"] = 1, "dataset_query", name.replace("_based", "")
        rows.append(df)
        # payload fragments cut from the same statements (what an API actually receives)
        frag = df["sql_query"].map(extract_fragment).dropna()
        frag = frag.sample(n=min(len(frag), per_class // 2), random_state=seed)
        rows.append(pd.DataFrame({"sql_query": frag, "label": 1, "source": "fragment", "attack": name.replace("_based", "")}))
        print(f"  {name:<20} {len(df):>7} rows  ({time.time() - t:.0f}s)")
    n_mal = sum(len(r) for r in rows)
    # synthetic API-style malicious payloads
    rows.append(pd.DataFrame({"sql_query": make_malicious_templates(per_class, seed), "label": 1,
                              "source": "template", "attack": "mixed"}))
    n_mal += per_class

    # benign: full statements (50%) + realistic API inputs (50%)
    n_ben = n_mal
    t = time.time()
    ben = sample_csv(data_dir / f"{BENIGN_FILE}.csv", n_ben // 2, seed)
    ben["label"], ben["source"], ben["attack"] = 0, "dataset_query", "benign"
    rows.append(ben)
    print(f"  {BENIGN_FILE:<20} {len(ben):>7} rows  ({time.time() - t:.0f}s)")
    rows.append(pd.DataFrame({"sql_query": make_benign(n_ben - len(ben), seed), "label": 0,
                              "source": "api_input", "attack": "benign"}))

    full = pd.concat(rows, ignore_index=True)
    full["text"] = full["sql_query"].astype(str).map(normalize)
    full = full.drop_duplicates("text").reset_index(drop=True)
    return full


def metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def get_models(seed: int, n_jobs: int):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5.0, solver="liblinear", random_state=seed),
        "SVM (LinearSVC)": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_leaf=2,
                                                n_jobs=n_jobs, random_state=seed),
    }


def plot_confusions(results, out):
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.8))
    for ax, (name, m) in zip(np.atleast_1d(axes), results.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Benign", "Malicious"]); ax.set_yticks([0, 1], ["Benign", "Malicious"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--per-class", type=int, default=20000, help="rows sampled from EACH attack CSV")
    ap.add_argument("--ngram-max", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    print("1/5 Sampling and building dataset ...")
    data = build_dataset(args.data_dir, args.per_class, args.seed)
    print(f"  total {len(data):,} unique rows | malicious {data.label.sum():,} | benign {(data.label == 0).sum():,}")

    print("2/5 Splitting 80/20 (stratified) ...")
    train, test = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=args.seed)

    print("3/5 Training models (TF-IDF char n-grams) ...")
    results, fitted, extra = {}, {}, {}
    ho_X = [normalize(t) for t in HO_BENIGN + HO_MALICIOUS]
    ho_y = [0] * len(HO_BENIGN) + [1] * len(HO_MALICIOUS)
    for name, clf in get_models(args.seed, args.n_jobs).items():
        t0 = time.time()
        pipe = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, args.ngram_max), lowercase=True,
                            max_features=args.max_features, sublinear_tf=True, dtype=np.float32),
            clf)
        pipe.fit(train["text"], train["label"])
        fit_s = time.time() - t0
        t1 = time.time()
        pred = pipe.predict(test["text"])
        lat_ms = (time.time() - t1) / len(test) * 1000
        m = metrics(test["label"], pred)
        m.update(train_seconds=round(fit_s, 1), predict_ms_per_row=round(lat_ms, 4))
        # secondary evaluations
        ho = metrics(ho_y, pipe.predict(ho_X))
        m["holdout_api_inputs"] = {k: ho[k] for k in ("accuracy", "precision", "recall", "f1", "false_positive_rate")}
        sub = test.assign(pred=pred)
        m["recall_by_attack"] = {a: float((g.pred == 1).mean()) for a, g in sub[sub.label == 1].groupby("attack")}
        m["recall_by_source"] = {s: float((g.pred == 1).mean()) for s, g in sub[sub.label == 1].groupby("source")}
        m["fpr_by_source"] = {s: float((g.pred == 1).mean()) for s, g in sub[sub.label == 0].groupby("source")}
        results[name], fitted[name] = m, pipe
        print(f"  {name:<20} acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
              f"f1={m['f1']:.4f} fpr={m['false_positive_rate']:.4f} | holdout f1={ho['f1']:.3f} "
              f"fpr={ho['false_positive_rate']:.3f} | {fit_s:.0f}s")

    print("4/5 Selecting best model ...")
    # Selection uses ONLY the 20% test split (F1, tie-break lower FPR). The hand-written hold-out set is
    # reported afterwards and never influences the choice. For a rigorous procedure use experiments.py (grouped CV).
    def score(n):
        r = results[n]
        return (round(r["f1"], 6), -r["false_positive_rate"])
    best = max(results, key=score)
    print(f"  -> best: {best}")

    print("5/5 Saving artefacts ...")
    out = BASE_DIR / "models"; out.mkdir(exist_ok=True)
    joblib.dump(fitted[best], out / "best_model.joblib", compress=3)
    for n, p in fitted.items():
        joblib.dump(p, out / f"{n.split()[0].lower()}.joblib", compress=3) if False else None
    summary = {"best_model": best, "n_train": len(train), "n_test": len(test), "per_class": args.per_class,
               "results": results}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_confusions(results, out / "confusion_matrices.png")
    print(f"  saved models/best_model.joblib, metrics.json, confusion_matrices.png")


if __name__ == "__main__":
    main()

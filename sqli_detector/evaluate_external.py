"""Score ANY csv with the trained detector (data it was never trained or tested on).

Labelled data (prints accuracy / precision / recall / F1 / FPR / FNR + confusion matrix):
    python evaluate_external.py --csv my_data.csv --text-col payload --label-col label --positive 1

Unlabelled data (just scores every row and writes predictions):
    python evaluate_external.py --csv my_data.csv --text-col payload

Options:
    --positive   label value(s) that mean "malicious", comma separated (default: 1,malicious,sqli,attack,anomalous,true)
    --model      path to a .joblib pipeline   (default: models/best_model.joblib)
    --threshold  P(malicious) cut-off          (default: 0.5, same as the API)
    --out        where to write per-row predictions (default: external_predictions.csv)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import MODEL_PATH, load_model, predict_malicious_proba

DEFAULT_POSITIVE = "1,malicious,sqli,sql injection,attack,anomalous,true,yes"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--text-col", required=True, help="column holding the input / query / payload")
    ap.add_argument("--label-col", default=None, help="column holding the ground truth (omit for unlabelled data)")
    ap.add_argument("--positive", default=DEFAULT_POSITIVE)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("external_predictions.csv"))
    ap.add_argument("--encoding", default="utf-8")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, encoding=a.encoding, encoding_errors="replace", low_memory=False)
    if a.text_col not in df.columns:
        raise SystemExit(f"Column '{a.text_col}' not found. Available columns: {list(df.columns)}")
    df = df[df[a.text_col].notna()].copy()
    texts = df[a.text_col].astype(str).tolist()

    model = load_model(a.model)
    # score in batches so big files do not blow up memory
    scores = np.concatenate([predict_malicious_proba(model, texts[i:i + 20000]) for i in range(0, len(texts), 20000)])
    df["score"] = scores
    df["predicted"] = (scores >= a.threshold).astype(int)
    print(f"Scored {len(df):,} rows from {a.csv.name} | flagged malicious: {int(df.predicted.sum()):,} "
          f"({df.predicted.mean():.1%})")

    if a.label_col:
        if a.label_col not in df.columns:
            raise SystemExit(f"Label column '{a.label_col}' not found. Available: {list(df.columns)}")
        pos = {p.strip().lower() for p in a.positive.split(",")}
        y = df[a.label_col].astype(str).str.strip().str.lower().isin(pos).astype(int).values
        print(f"Ground truth: {int(y.sum()):,} positive (labels counted as malicious: {sorted(pos)}), "
              f"{int((y == 0).sum()):,} negative. Check these counts look right!")
        print("Distinct label values:", df[a.label_col].astype(str).value_counts().head(8).to_dict())
        tn, fp, fn, tp = confusion_matrix(y, df.predicted.values, labels=[0, 1]).ravel()
        d = lambda x, z: x / z if z else float("nan")
        prec, rec = d(tp, tp + fp), d(tp, tp + fn)
        print(f"\nAccuracy  {d(tp + tn, len(y)):.4f}\nPrecision {prec:.4f}\nRecall    {rec:.4f}\n"
              f"F1        {d(2 * prec * rec, prec + rec):.4f}\nFPR       {d(fp, fp + tn):.4f}\nFNR       {d(fn, fn + tp):.4f}")
        print(f"\nConfusion matrix   (rows = actual, cols = predicted)\n            benign  malicious\n"
              f"benign     {tn:7d} {fp:9d}\nmalicious  {fn:7d} {tp:9d}")
        df["actual"] = y

    df.to_csv(a.out, index=False)
    print(f"\nPer-row predictions written to {a.out}")
    print("Worst false positives / most confident detections are easy to inspect: sort that file by 'score'.")


if __name__ == "__main__":
    main()

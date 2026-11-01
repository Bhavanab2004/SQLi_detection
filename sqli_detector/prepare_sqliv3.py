"""Convert the raw SQLiV3.csv (no header, stray extra columns) into a clean UTF-8 CSV with columns: text,label.
    python prepare_sqliv3.py SQLiV3.csv external_data/sqliv3_clean.csv
"""
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, header=None, names=["text", "label", "x", "y"], encoding="utf-8", encoding_errors="replace",
                engine="python", on_bad_lines="skip", dtype=str)
d = d[d["label"].isin(["0", "1"])].copy()
d["label"] = d["label"].astype(int)
d["text"] = d["text"].astype(str)
before = len(d)
conflict = d.groupby("text")["label"].nunique()
d = d[~d["text"].isin(set(conflict[conflict > 1].index))].drop_duplicates("text")   # drop label-conflicts + exact duplicates
d[["text", "label"]].to_csv(dst, index=False)
print(f"{before:,} rows -> {len(d):,} after removing exact duplicates and label conflicts | "
      f"malicious {int(d.label.sum()):,}, benign {int((d.label == 0).sum()):,}")

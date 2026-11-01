"""Streamlit security-monitoring dashboard.   Run:  streamlit run dashboard.py"""
import json
import re

import pandas as pd
import streamlit as st

import db
from common import BASE_DIR, THRESHOLD, load_model, predict_malicious_proba

st.set_page_config(page_title="SQLi Guard Dashboard", page_icon="🛡️", layout="wide")
db.init_db()

# Heuristic label used ONLY for display (the blocking decision is made by the ML model).
HINTS = [
    ("Time-based", r"sleep\s*\(|waitfor\s+delay|pg_sleep|benchmark\s*\("),
    ("Union-based", r"union\s+(all\s+)?select"),
    ("Stacked queries", r";\s*(drop|insert|update|delete|exec|select|shutdown)"),
    ("Error-based", r"extractvalue|updatexml|convert\s*\(|cast\s*\(|having\s+1\s*=\s*1"),
    ("Meta / schema", r"information_schema|sqlite_master|@@version|version\s*\(\)|load_file"),
    ("Boolean-based", r"\bor\b.+=|\band\b.+=|'\s*or\s*'|\bor\s+true"),
]


def hint(p: str) -> str:
    for name, rx in HINTS:
        if re.search(rx, str(p), re.I):
            return name
    return "Other / unclassified"


st.title("🛡️ ML SQL Injection Detection — Security Monitor")
tab1, tab2, tab3 = st.tabs(["Live monitoring", "Model evaluation", "Try a payload"])

# ------------------------------------------------------------------ live monitoring
with tab1:
    if st.button("🔄 Refresh"):
        st.rerun()
    df = pd.DataFrame(db.fetch_all(50000))
    if df.empty:
        st.info("No traffic logged yet. Start the API and send some requests (see README step 6/7).")
    else:
        df["ts"] = pd.to_datetime(df["ts"])
        blocked = df[df.action == "BLOCKED"].copy()
        blocked["attack_hint"] = blocked["payload"].map(hint)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests inspected", f"{len(df):,}")
        c2.metric("Blocked", f"{len(blocked):,}")
        c3.metric("Allowed", f"{len(df) - len(blocked):,}")
        c4.metric("Block rate", f"{len(blocked) / len(df):.1%}")

        left, right = st.columns(2)
        with left:
            st.subheader("Requests over time")
            ts = df.set_index("ts").groupby([pd.Grouper(freq="1min"), "action"]).size().unstack(fill_value=0)
            st.line_chart(ts)
        with right:
            st.subheader("Blocked by attack pattern (heuristic)")
            if len(blocked):
                st.bar_chart(blocked["attack_hint"].value_counts())
        l2, r2 = st.columns(2)
        with l2:
            st.subheader("Top attacking IPs")
            if len(blocked):
                st.bar_chart(blocked["client_ip"].value_counts().head(10))
        with r2:
            st.subheader("Most targeted endpoints")
            if len(blocked):
                st.bar_chart(blocked["path"].value_counts().head(10))
        st.subheader("Recent blocked incidents")
        st.dataframe(blocked[["ts", "client_ip", "method", "path", "field", "payload", "score", "attack_hint"]].head(200),
                     width="stretch", hide_index=True)

# ------------------------------------------------------------------ model evaluation
with tab2:
    mp = BASE_DIR / "models" / "metrics.json"
    if not mp.exists():
        st.warning("Run train.py first.")
    else:
        m = json.loads(mp.read_text())
        st.success(f"Model deployed in the API: **{m['best_model']}**  (trained on {m['n_train']:,} rows, tested on {m['n_test']:,})")
        rows = []
        for name, r in m["results"].items():
            rows.append({"Model": name, "Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"],
                         "F1": r["f1"], "False-positive rate": r["false_positive_rate"],
                         "Hold-out F1": r["holdout_api_inputs"]["f1"], "Hold-out FPR": r["holdout_api_inputs"]["false_positive_rate"],
                         "Train s": r["train_seconds"]})
        tbl = pd.DataFrame(rows).set_index("Model")
        fmt = {c: "{:.4f}" for c in tbl.columns if c != "Train s"}
        fmt["Train s"] = "{:.0f}"
        st.dataframe(tbl.style.format(fmt), width="stretch")
        st.caption("'Hold-out' = 100 hand-written API-style inputs never used in training. It is the more realistic number.")
        png = BASE_DIR / "models" / "confusion_matrices.png"
        if png.exists():
            st.image(str(png), caption="Confusion matrices on the 20% test split")
        best = m["results"][m["best_model"]]
        st.subheader(f"Recall by attack type — {m['best_model']}")
        st.bar_chart(pd.Series(best["recall_by_attack"]))

# ------------------------------------------------------------------ try a payload
with tab3:
    @st.cache_resource
    def _model():
        return load_model()

    txt = st.text_input("Enter a string an API client might send", "' OR '1'='1' --")
    if txt:
        s = float(predict_malicious_proba(_model(), [txt])[0])
        (st.error if s >= THRESHOLD else st.success)(
            f"{'🚫 MALICIOUS' if s >= THRESHOLD else '✅ BENIGN'} — P(malicious) = {s:.3f} (threshold {THRESHOLD})")

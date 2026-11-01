"""Reproducible experiments for the paper.  Everything printed in results/paper_values.md comes from here.

    python experiments.py --data-dir <sqli_by_type> --per-class 10000                 # all stages
    python experiments.py --data-dir ... --stages cv,report                            # resume / subset
    python experiments.py ... --external-csv ext.csv --external-text-col Query --external-label-col Label \
                              --external-name "Name of dataset"

Stages: build -> cv -> final -> ablate -> latency -> report   (each stage caches its output in results/)
NOT covered (needs tools/data that are not in this repo): ModSecurity CRS baseline, external dataset (unless you
pass --external-csv), dataset citation/licence, repository URL.
"""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote_plus

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from augment import extract_fragment, make_benign_with_ids, make_malicious_templates
from common import BASE_DIR, normalize
from tests.holdout import BENIGN as HO_BENIGN, MALICIOUS as HO_MALICIOUS
from train import ATTACK_FILES, BENIGN_FILE, sample_csv

OUT = BASE_DIR / "results"
SEED = 42
TFIDF = dict(ngram=(1, 5), max_features=100_000, sublinear=True, analyzer="char")
ML_MODELS = ["Logistic Regression", "LinearSVC", "Random Forest", "XGBoost"]
BASELINES = ["libinjection"]  # ModSecurity CRS: see ../results/NOT_COMPUTED in report


# ------------------------------------------------------------------------------------------ helpers
def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=float))


def load(name):
    return json.loads((OUT / name).read_text())


def six(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    d = lambda a, b: a / b if b else 0.0
    prec, rec = d(tp, tp + fp), d(tp, tp + fn)
    return dict(accuracy=d(tp + tn, tp + tn + fp + fn), precision=prec, recall=rec,
                f1=d(2 * prec * rec, prec + rec), fpr=d(fp, fp + tn), fnr=d(fn, fn + tp))


def make_model(name, seed=SEED):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, C=5.0, solver="liblinear", random_state=seed)
    if name == "LinearSVC":
        return CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), cv=3)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_leaf=2, max_features="sqrt",
                                      n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                             random_state=seed, n_jobs=1, eval_metric="logloss")
    raise ValueError(name)


def make_pipe(name, analyzer="char", ngram=(1, 5), max_features=100_000, sublinear=True):
    kw = dict(analyzer=analyzer, ngram_range=ngram, lowercase=True, max_features=max_features,
              sublinear_tf=sublinear, dtype=np.float32)
    if analyzer == "word":
        kw["token_pattern"] = r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]"
    return make_pipeline(TfidfVectorizer(**kw), make_model(name))


def proba(pipe, texts):
    return pipe.predict_proba(list(texts))[:, 1]


def hardware():
    cpu = next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if "model name" in l), platform.processor())
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    import sklearn, xgboost
    try:
        import libinjection  # noqa
        lib = "installed"
    except ImportError:
        lib = "missing"
    return dict(cpu=cpu, logical_cores=os.cpu_count(), ram_gb=round(ram, 1), os=platform.platform(),
                python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                libinjection_python=lib)


# ------------------------------------------------------------------------------------------ stage: build
def stage_build(a):
    parts, counts = [], {}
    n_mal = 0
    for name in ATTACK_FILES:
        df = sample_csv(a.data_dir / f"{name}.csv", a.per_class, SEED)
        grp = [f"{name}:{i}" for i in df.index]            # original CSV row id
        parts.append(pd.DataFrame({"raw": df.sql_query.values, "label": 1, "source": "dataset_query", "group": grp}))
        fr = df.sql_query.map(extract_fragment).dropna()
        fr = fr.sample(n=min(len(fr), a.per_class // 2), random_state=SEED)
        parts.append(pd.DataFrame({"raw": fr.values, "label": 1, "source": "fragment",
                                   "group": [f"{name}:{i}" for i in fr.index]}))   # fragment shares its parent row's group
        n_mal += len(df) + len(fr)
    tpl = make_malicious_templates(a.per_class, SEED, with_ids=True)
    parts.append(pd.DataFrame({"raw": [t for t, _ in tpl], "label": 1, "source": "template",
                               "group": [f"tpl:{k}" for _, k in tpl]}))
    n_mal += len(tpl)
    ben = sample_csv(a.data_dir / f"{BENIGN_FILE}.csv", n_mal // 2, SEED)
    parts.append(pd.DataFrame({"raw": ben.sql_query.values, "label": 0, "source": "dataset_query",
                               "group": [f"benign:{i}" for i in ben.index]}))
    api = make_benign_with_ids(n_mal - len(ben), SEED)
    parts.append(pd.DataFrame({"raw": [t for t, _ in api], "label": 0, "source": "api_input",
                               "group": [f"benign_tpl:{k}" for _, k in api]}))
    full = pd.concat(parts, ignore_index=True)
    full["text"] = full["raw"].astype(str).map(normalize)

    before = full.groupby(["source", "label"]).size().rename("n").reset_index()
    # texts that occur with BOTH labels would be label noise -> drop all copies
    both = full.groupby("text")["label"].nunique()
    conflict = set(both[both > 1].index)
    full = full[~full.text.isin(conflict)]
    full = full.drop_duplicates("text").reset_index(drop=True)
    after = full.groupby(["source", "label"]).size().rename("n").reset_index()
    counts = dict(before=before.to_dict("records"), after=after.to_dict("records"),
                  total_before=int(before.n.sum()), total_after=int(len(full)),
                  conflicting_texts_dropped=len(conflict), n_groups=int(full.group.nunique()))
    full.to_pickle(OUT / "dataset.pkl")
    save("dataset_counts.json", counts)
    print(f"[build] {counts['total_before']:,} rows before dedup -> {counts['total_after']:,} after "
          f"({counts['conflicting_texts_dropped']} conflicting texts dropped); {counts['n_groups']} groups")


# ------------------------------------------------------------------------------------------ stage: cv
def lib_predict(texts):
    import libinjection
    return np.array([int(bool(libinjection.is_sql_injection(t)["is_sqli"])) for t in texts])


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def stage_cv(a):
    d = pd.read_pickle(OUT / "dataset.pkl")
    y = d.label.values
    oof = {m: np.zeros(len(d)) for m in ML_MODELS}
    oof_lib = np.zeros(len(d), dtype=int)
    folds = {m: [] for m in ML_MODELS + BASELINES}
    sizes = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=a.folds).split(d, y, d.group)):
        sizes.append(dict(fold=k, train=len(tr), test=len(te), test_malicious=int(y[te].sum())))
        for m in ML_MODELS:
            t = time.time()
            pipe = make_pipe(m, **{k_: v for k_, v in dict(analyzer=TFIDF["analyzer"], ngram=TFIDF["ngram"],
                                                          max_features=TFIDF["max_features"], sublinear=TFIDF["sublinear"]).items()})
            pipe.fit(d.text.iloc[tr], y[tr])
            p = proba(pipe, d.text.iloc[te])
            oof[m][te] = p
            folds[m].append(six(y[te], (p >= 0.5).astype(int)))
            print(f"[cv] fold {k} {m:<20} f1={folds[m][-1]['f1']:.4f} fpr={folds[m][-1]['fpr']:.4f} ({time.time() - t:.0f}s)", flush=True)
        oof_lib[te] = lib_predict(d.text.iloc[te])
        folds["libinjection"].append(six(y[te], oof_lib[te]))

    summary = {m: {k_: [float(np.mean([f[k_] for f in folds[m]])), float(np.std([f[k_] for f in folds[m]], ddof=1))]
                   for k_ in folds[m][0]} for m in folds}
    # selection on validation folds ONLY (hold-out and obfuscated sets are never consulted)
    best = min(ML_MODELS, key=lambda m: (-round(summary[m]["f1"][0], 6), summary[m]["fpr"][0]))

    # by-source behaviour (pooled out-of-fold)
    by_src = {}
    preds = {m: (oof[m] >= 0.5).astype(int) for m in ML_MODELS}
    preds["libinjection"] = oof_lib
    for m, pr in preds.items():
        by_src[m] = {f"{s}|{'mal' if l else 'ben'}": float((pr[(d.source == s) & (d.label == l)] == 1).mean())
                     for s in d.source.unique() for l in (0, 1) if ((d.source == s) & (d.label == l)).any()}

    # RF vs SVM significance
    f1_rf = [f["f1"] for f in folds["Random Forest"]]
    f1_sv = [f["f1"] for f in folds["LinearSVC"]]
    diff = np.array(f1_rf) - np.array(f1_sv)
    t_res = stats.ttest_rel(f1_rf, f1_sv) if np.any(diff != 0) else None
    try:
        w_res = stats.wilcoxon(f1_rf, f1_sv) if np.any(diff != 0) else None
    except ValueError:
        w_res = None
    rf_ok, sv_ok = preds["Random Forest"] == y, preds["LinearSVC"] == y
    b, c = int((rf_ok & ~sv_ok).sum()), int((~rf_ok & sv_ok).sum())
    mc_p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    sig = dict(per_fold_f1_rf=f1_rf, per_fold_f1_svc=f1_sv,
               paired_t_p=None if t_res is None else float(t_res.pvalue),
               wilcoxon_p=None if w_res is None else float(w_res.pvalue),
               mcnemar_rf_right_svc_wrong=b, mcnemar_svc_right_rf_wrong=c, mcnemar_exact_p=mc_p)

    calib = {m: dict(brier=float(np.mean((oof[m] - y) ** 2)), ece=ece(y, oof[m])) for m in ML_MODELS}
    save("cv.json", dict(folds=folds, summary=summary, best_model=best, fold_sizes=sizes, by_source=by_src,
                         significance=sig, calibration=calib))
    print(f"[cv] selected by mean validation-fold F1: {best}")


# ------------------------------------------------------------------------------------------ obfuscation
KW = r"\b(union|select|from|sleep|drop|and|or|order|where|having|insert|update|exec|waitfor|benchmark|group)\b"
import re  # noqa: E402


def _versioned(s, r):
    return re.sub(KW, lambda m: f"/*!50000{m.group(1)}*/", s, flags=re.I)


def _hexchar(s, r, i):
    def rep(m):
        v = m.group(1)
        return "0x" + v.encode().hex() if i % 2 == 0 else "CHAR(" + ",".join(str(ord(c)) for c in v) + ")"
    return re.sub(r"'([^'\s][^']*)'", rep, s)


def _altws(s, r):
    return re.sub(" ", lambda m: r.choice(["%09", "%0a", "%0d", "%0b", "%0c", "/**/", "%a0", "+"]), s)


def _case(s, r):
    return "".join(c.upper() if r.random() < 0.5 else c.lower() for c in s)


def _enc(s, n):
    for _ in range(n):
        s = quote(s, safe="")
    return s


def make_obfuscated(seed=SEED):
    """technique -> list of strings AS THE APP SEES THEM, i.e. after the web framework's single URL-decode.
    (FastAPI/Starlette decode the query string once; normalize() then decodes up to twice more.)"""
    r = random.Random(seed)
    T = {"MySQL versioned comments": [], "Hex / CHAR() literals": [], "Alternate whitespace": [],
         "Case mixing": [], "Triple URL encoding": [], "Quadruple URL encoding (extra)": []}
    for i, s in enumerate(HO_MALICIOUS):
        v = _versioned(s, r)
        if v != s: T["MySQL versioned comments"].append(unquote_plus(v))
        h = _hexchar(s, r, i)
        if h != s: T["Hex / CHAR() literals"].append(unquote_plus(h))
        if " " in s:
            for _ in range(3): T["Alternate whitespace"].append(unquote_plus(_altws(s, r)))
        for _ in range(3): T["Case mixing"].append(_case(s, r))
        T["Triple URL encoding"].append(unquote_plus(_enc(s, 3)))
        T["Quadruple URL encoding (extra)"].append(unquote_plus(_enc(s, 4)))
    return T


# ------------------------------------------------------------------------------------------ stage: final
def stage_final(a):
    d = pd.read_pickle(OUT / "dataset.pkl")
    (OUT / "final_models").mkdir(exist_ok=True)
    obf = make_obfuscated()
    ho_x = HO_BENIGN + HO_MALICIOUS
    ho_y = [0] * len(HO_BENIGN) + [1] * len(HO_MALICIOUS)
    res = {"obfuscated_counts": {k: len(v) for k, v in obf.items()}, "models": {}}
    for m in ML_MODELS:
        t = time.time()
        pipe = make_pipe(m).fit(d.text, d.label.values)
        path = OUT / "final_models" / f"{m.replace(' ', '_')}.joblib"
        joblib.dump(pipe, path)
        size_mb = path.stat().st_size / 2**20
        p_ho = proba(pipe, [normalize(x) for x in ho_x])
        r = dict(fit_seconds=round(time.time() - t, 1), size_mb_compressed=round(size_mb, 2),
                 holdout=six(ho_y, (p_ho >= 0.5).astype(int)))
        r["obfuscated_recall"] = {k: float((proba(pipe, [normalize(x) for x in v]) >= 0.5).mean()) for k, v in obf.items() if v}
        r["base_payload_recall"] = float((proba(pipe, [normalize(x) for x in HO_MALICIOUS]) >= 0.5).mean())
        res["models"][m] = r
        print(f"[final] {m:<20} fit {r['fit_seconds']}s  holdout f1={r['holdout']['f1']:.3f}  obf={r['obfuscated_recall']}", flush=True)
    save("final.json", res)



# ------------------------------------------------------------------------------------------ stage: external (CSIC 2010)
SQLISH = re.compile(r"(?i)(\bunion\b[\s\S]{0,40}\bselect\b|\bselect\b[\s\S]{0,60}\bfrom\b|\bdrop\s+table|\binsert\s+into|"
                    r"\bdelete\s+from|\bupdate\b[\s\S]{0,40}\bset\b|\bor\b\s*['\"(]?\s*\w+\s*['\"]?\s*=\s*['\"]?\s*\w+|"
                    r"\band\b\s*['\"(]?\s*\w+\s*['\"]?\s*=\s*['\"]?\s*\w+|['\"]\s*(or|and)\b|\bsleep\s*\(|\bwaitfor\b|"
                    r";\s*(select|drop|insert|update|delete|shutdown|exec)\b|\binformation_schema\b|\bxp_cmdshell\b)")


def csic_requests(path):
    """Mimic the guard: only query-string and form-body VALUES are inspected (not path, headers or cookies).
    parse_qsl performs the framework's single URL-decode; normalize() then decodes up to twice more."""
    from urllib.parse import parse_qsl
    d = pd.read_csv(path, low_memory=False).fillna("")
    y = (d["Class"] == "Anomalous").astype(int).values
    reqs = []
    for q, body in zip(d["GET-Query"], d["POST-Data"]):
        vals = []
        for part in (q, body):
            if part:
                vals += [v for _, v in parse_qsl(part, keep_blank_values=False) if v.strip()]
        reqs.append([v[:2000] for v in vals])
    return d, y, reqs


def stage_external(a):
    d, y, reqs = csic_requests(a.external_csv)
    uniq = sorted({v for r in reqs for v in r})
    print(f"[external] {len(d):,} requests ({int((y == 0).sum()):,} valid / {int(y.sum()):,} anomalous); "
          f"{sum(map(len, reqs)):,} values, {len(uniq):,} unique", flush=True)
    n_fields = np.array([len(r) for r in reqs])
    no_fields = n_fields == 0
    wants = [v for v in uniq]
    out = {"dataset": a.external_name, "n_requests": int(len(d)), "n_valid": int((y == 0).sum()),
           "n_anomalous": int(y.sum()), "unique_values": len(uniq),
           "anomalous_without_scannable_fields": float(no_fields[y == 1].mean()),
           "valid_without_scannable_fields": float(no_fields[y == 0].mean()),
           "models": {}, "source": a.external_source}
    has_sqlish = np.array([any(SQLISH.search(unquote_plus(unquote_plus(v))) for v in r) for r in reqs])
    out["anomalous_with_sqlish_value_share"] = float(has_sqlish[y == 1].mean())
    out["valid_with_sqlish_value_share"] = float(has_sqlish[y == 0].mean())

    def evaluate(name, score_values):
        sc = dict(zip(wants, score_values))
        flagged = np.array([any(sc[v] for v in r) for r in reqs], dtype=int)  # any value flagged -> request blocked
        m = six(y, flagged)
        sub = (y == 1) & has_sqlish & ~no_fields
        m["recall_sqlish_anomalous"] = float(flagged[sub].mean()) if sub.any() else None
        m["n_sqlish_anomalous"] = int(sub.sum())
        m["recall_anomalous_with_fields"] = float(flagged[(y == 1) & ~no_fields].mean())
        m["recall_anomalous_without_fields"] = 0.0  # nothing to inspect -> guard allows
        m["fpr_valid_with_fields"] = float(flagged[(y == 0) & ~no_fields].mean())
        fp = np.where((y == 0) & (flagged == 1))[0][:8]
        fn = np.where((y == 1) & (flagged == 0) & ~no_fields)[0][:8]
        m["examples_false_positive"] = [[v[:90] for v in reqs[i]][:3] for i in fp]
        m["examples_false_negative"] = [[v[:90] for v in reqs[i]][:3] for i in fn]
        out["models"][name] = m
        print(f"[external] {name:<20} acc={m['accuracy']:.4f} f1={m['f1']:.4f} fpr={m['fpr']:.4f} "
              f"recall={m['recall']:.4f} (sqli-like recall={m['recall_sqlish_anomalous']})", flush=True)

    for m in ML_MODELS:
        t = time.time()
        pipe = joblib.load(OUT / "final_models" / f"{m.replace(' ', '_')}.joblib")
        evaluate(m, proba(pipe, [normalize(v) for v in wants]) >= 0.5)
        print(f"[external]   scored in {time.time() - t:.0f}s", flush=True)
    evaluate("libinjection", lib_predict([normalize(v) for v in wants]).astype(bool))
    save("external.json", out)


# ------------------------------------------------------------------------------------------ stage: ablate
def stage_ablate(a):
    d = pd.read_pickle(OUT / "dataset.pkl")
    best = load("cv.json")["best_model"]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(d, d.label, d.group))
    obf = make_obfuscated()
    ho_x = HO_BENIGN + HO_MALICIOUS
    ho_y = [0] * len(HO_BENIGN) + [1] * len(HO_MALICIOUS)
    obf_all = [x for v in obf.values() for x in v]
    variants = {
        "Baseline (char 1-5 TF-IDF, normalised, sublinear)": dict(),
        "Word TF-IDF instead of char n-grams": dict(analyzer="word", ngram=(1, 2)),
        "N-gram range reduced to 1-3": dict(ngram=(1, 3)),
        "Normalisation disabled": dict(norm=False),
        "Sublinear TF disabled": dict(sublinear=False),
        "Fragments + templates excluded from training": dict(only_dataset=True),
    }
    out = {}
    for name, v in variants.items():
        t = time.time()
        norm = v.get("norm", True)
        col = "text" if norm else "raw"
        trn = d.iloc[tr]
        if v.get("only_dataset"):
            trn = trn[trn.source == "dataset_query"]
        pipe = make_pipe(best, **{k: x for k, x in v.items() if k in ("analyzer", "ngram", "sublinear")})
        pipe.fit(trn[col].astype(str), trn.label.values)
        f = (lambda s: normalize(s)) if norm else (lambda s: s)
        r = dict(split_test=six(d.label.values[te], (proba(pipe, d.iloc[te][col].astype(str)) >= 0.5).astype(int)),
                 holdout=six(ho_y, (proba(pipe, [f(x) for x in ho_x]) >= 0.5).astype(int)),
                 obfuscated_recall=float((proba(pipe, [f(x) for x in obf_all]) >= 0.5).mean()))
        out[name] = r
        print(f"[ablate] {name:<52} split f1={r['split_test']['f1']:.4f} holdout f1={r['holdout']['f1']:.4f} "
              f"obf recall={r['obfuscated_recall']:.3f} ({time.time() - t:.0f}s)", flush=True)
    save("ablation.json", dict(model=best, n_train=int(len(tr)), n_test=int(len(te)), variants=out))


# ------------------------------------------------------------------------------------------ stage: latency
def request_mix(n, seed=SEED):
    """Mirrors send_traffic.py: 1/3 GET /api/users (1 field), 1/3 search (1 field), 1/3 login (2 fields)."""
    r = random.Random(seed)
    reqs = []
    for _ in range(n):
        pool = HO_MALICIOUS if r.random() < 0.5 else HO_BENIGN
        k = 2 if r.random() < 1 / 3 else 1
        reqs.append([r.choice(pool) for _ in range(k)] if k == 2 else [r.choice(pool)])
    return reqs


def pct(x):
    return {f"p{q}": float(np.percentile(x, q)) for q in (50, 95, 99)}


def stage_latency(a):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    reqs = request_mix(2000)
    fields = float(np.mean([len(r) for r in reqs]))
    res = {"fields_per_request": fields, "in_process_ms": {}, "hardware": hardware()}
    for m in ML_MODELS:
        pipe = joblib.load(OUT / "final_models" / f"{m.replace(' ', '_')}.joblib")
        for r in reqs[:50]:
            pipe.predict_proba([normalize(x) for x in r])
        lat = []
        for r in reqs[:1000]:
            t = time.perf_counter()
            pipe.predict_proba([normalize(x) for x in r])
            lat.append((time.perf_counter() - t) * 1000)
        res["in_process_ms"][m] = dict(**pct(lat), mean=float(np.mean(lat)))
        print(f"[latency] {m:<20} {res['in_process_ms'][m]}", flush=True)

    # end-to-end HTTP against the real API (deployed model = models/best_model.joblib)
    port = 8765
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--port", str(port), "--log-level", "warning"],
                            cwd=BASE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.5)
        res["http"] = {}
        for conc in a.concurrency:
            total = 300 * conc if conc > 1 else 300

            def work(n):
                s, lat = requests.Session(), []
                rr = random.Random(n)
                for _ in range(total // conc):
                    p = rr.choice(HO_MALICIOUS + HO_BENIGN)
                    kind = rr.choice(["u", "s", "l"])
                    t = time.perf_counter()
                    if kind == "u": s.get(f"http://127.0.0.1:{port}/api/users", params={"username": p})
                    elif kind == "s": s.get(f"http://127.0.0.1:{port}/api/products/search", params={"q": p})
                    else: s.post(f"http://127.0.0.1:{port}/api/login", json={"username": p, "password": "pw"})
                    lat.append((time.perf_counter() - t) * 1000)
                return lat

            t0 = time.perf_counter()
            with ThreadPoolExecutor(conc) as ex:
                lats = [x for chunk in ex.map(work, range(conc)) for x in chunk]
            wall = time.perf_counter() - t0
            res["http"][str(conc)] = dict(requests=len(lats), req_per_sec=len(lats) / wall, **pct(lats))
            print(f"[latency] HTTP concurrency={conc}: {res['http'][str(conc)]}", flush=True)
    finally:
        proc.terminate()
        for f in (BASE_DIR / "data").glob("*.db"):
            f.unlink(missing_ok=True)
    save("latency.json", res)


# ------------------------------------------------------------------------------------------ stage: report
def ms(v, pct_=True):
    m, s = v
    return f"{m * 100:.2f} $\\pm$ {s * 100:.2f}" if pct_ else f"{m:.4f} $\\pm$ {s:.4f}"


def stage_report(a):
    cv, fin = load("cv.json"), load("final.json")
    cnt = load("dataset_counts.json")
    abl = load("ablation.json") if (OUT / "ablation.json").exists() else None
    lat = load("latency.json") if (OUT / "latency.json").exists() else None
    hw = hardware()
    L = ["# Values for the paper (auto-generated by experiments.py)\n",
         f"Run config: per-class={a.per_class}, folds={a.folds}, seed={SEED}. Hardware of THIS run: {json.dumps(hw)}\n"]
    tex = []

    L.append("## 1/2 Abstract & architecture\n- ModSecurity CRS version / paranoia level: **NOT COMPUTED** (see bottom)\n"
             f"- URL-decode iterations: **{2} inside `normalize()`** (+1 by the web framework when the value arrives in the "
             "query string = 3 total on the API path)\n"
             "- Inline comments: only the literal empty comment `/**/` is replaced by a space (`text.replace('/**/', ' ')`); "
             "MySQL versioned comments `/*!50000UNION*/` are NOT stripped (see Table III)\n"
             f"- p95 end-to-end latency / hardware: see Table VII + hardware line above\n")

    L.append("## 3 Dataset / Table I\n")
    L.append(f"Rows before dedup: **{cnt['total_before']:,}**, after: **{cnt['total_after']:,}** "
             f"(removed {cnt['total_before'] - cnt['total_after']:,}; {cnt['conflicting_texts_dropped']} texts with conflicting "
             "labels dropped). The count is NOT unchanged: dedup is on the *normalised* text, and the synthetic generators "
             "have limited variety so they produce many exact duplicates.\n")
    ba = {(r["source"], r["label"]): r["n"] for r in cnt["before"]}
    af = {(r["source"], r["label"]): r["n"] for r in cnt["after"]}
    L.append("| Source | Benign (before->after) | Malicious (before->after) |\n|---|---|---|")
    names = {"dataset_query": "Source dataset (full statements)", "fragment": "Extracted fragments",
             "template": "Template-generated", "api_input": "Synthetic API inputs"}
    rows = []
    for s, nm in names.items():
        b0, b1, m0, m1 = ba.get((s, 0), 0), af.get((s, 0), 0), ba.get((s, 1), 0), af.get((s, 1), 0)
        L.append(f"| {nm} | {b0:,} -> {b1:,} | {m0:,} -> {m1:,} |")
        rows.append(f"{nm} & {b1:,} & {m1:,} & {b1 + m1:,} \\\\")
    tb, tm = sum(v for (s, l), v in af.items() if l == 0), sum(v for (s, l), v in af.items() if l == 1)
    rows.append(f"\\textbf{{Total}} & {tb:,} & {tm:,} & {tb + tm:,} \\\\")
    tex += ["% Table I (after dedup)"] + rows
    fs = cv["fold_sizes"]
    L.append(f"\nGroup definition for GroupKFold ({cnt['n_groups']} groups): dataset rows -> `<csv file>:<original row id>`; "
             "extracted fragments inherit the group of the statement they were cut from; template-generated payloads -> "
             "attack-template id (21 templates); synthetic benign inputs -> generator-template id (12 templates). "
             "Whole templates are therefore held out together.\n"
             f"Fold sizes (train/test): {[(f['train'], f['test']) for f in fs]}; mean train {np.mean([f['train'] for f in fs]):,.0f}, "
             f"mean test {np.mean([f['test'] for f in fs]):,.0f}\n")

    L.append("## 4 Methodology parameters\n- LR: C=5.0, liblinear, seed 42\n- LinearSVC: C=1.0, seed 42, wrapped in "
             "CalibratedClassifierCV(cv=3, sigmoid)\n- RF: n_estimators=150, max_depth=None, min_samples_leaf=2, "
             "max_features='sqrt', seed 42\n- XGBoost: n_estimators=200, max_depth=6, learning_rate=0.1, tree_method=hist, seed 42\n"
             f"- TF-IDF: analyzer=char, ngram_range={TFIDF['ngram']}, max_features={TFIDF['max_features']:,}, sublinear_tf=True, "
             "lowercase=True\n")

    L.append(f"## 5 Table II (GroupKFold k={a.folds}, mean ± std over folds, %)\n")
    L.append("| Model | Acc | Prec | Rec | F1 | FPR | FNR |\n|---|---|---|---|---|---|---|")
    tex.append("% Table II")
    for m in ML_MODELS + BASELINES:
        s = cv["summary"][m]
        cells = [ms(s[k]) for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr")]
        L.append(f"| {m} | " + " | ".join(c.replace("$\\pm$", "±") for c in cells) + " |")
        tex.append(f"{m} & " + " & ".join(cells) + " \\\\")
    L.append(f"\nSelected model (highest mean validation-fold F1, tie-break lowest FPR): **{cv['best_model']}**. "
             "The hand-written hold-out, obfuscated set and ablations were NOT used for selection.\n")
    L.append("Pooled out-of-fold detection rate by source (malicious = recall, benign = FPR):\n")
    for m, d_ in cv["by_source"].items():
        L.append(f"- {m}: " + ", ".join(f"{k}={v:.3f}" for k, v in d_.items()))

    L.append("\nCaveat for the libinjection row: libinjection is designed to score individual input values, but 'Source dataset "
             "(full statements)' rows are whole SQL statements, so its pooled FPR is dominated by benign full statements "
             "(see `dataset_query|ben` above). Compare it on the value-level sources (`fragment`, `template`, `api_input`) "
             "for a fairer picture, and say so in the paper.\n")
    L.append("Hand-written hold-out (100 inputs, final models fitted on all data):\n\n| Model | Acc | Prec | Rec | F1 | FPR |\n|---|---|---|---|---|---|")
    for m in ML_MODELS:
        h_ = fin["models"][m]["holdout"]
        L.append(f"| {m} | {h_['accuracy']*100:.1f} | {h_['precision']*100:.1f} | {h_['recall']*100:.1f} | {h_['f1']*100:.1f} | {h_['fpr']*100:.1f} |")

    sg = cv["significance"]
    L.append("\n## 6 RF vs LinearSVC\n"
             f"- per-fold F1 RF: {[round(x, 5) for x in sg['per_fold_f1_rf']]}\n- per-fold F1 SVC: {[round(x, 5) for x in sg['per_fold_f1_svc']]}\n"
             f"- paired t-test p = {sg['paired_t_p']}; Wilcoxon p = {sg['wilcoxon_p']} (n=5 folds: the smallest attainable "
             "two-sided Wilcoxon p is 0.0625, so it can never reach 0.05)\n"
             f"- McNemar exact on pooled out-of-fold predictions: RF right/SVC wrong = {sg['mcnemar_rf_right_svc_wrong']}, "
             f"SVC right/RF wrong = {sg['mcnemar_svc_right_rf_wrong']}, p = {sg['mcnemar_exact_p']:.4g}\n"
             "- Caveat: CV folds share training data, so fold-level tests understate variance (Nadeau & Bengio 2003).\n")

    L.append(f"## 7 Table III — obfuscated set (recall %; bases = the {len(HO_MALICIOUS)} hand-written malicious payloads)\n")
    oc = fin["obfuscated_counts"]
    L.append(f"Total obfuscated samples: **{sum(oc.values())}** (per technique: {oc}); un-obfuscated base recall shown for reference.\n")
    L.append("| Technique (n) | " + " | ".join(ML_MODELS) + " |\n|---|" + "---|" * len(ML_MODELS))
    tex.append("% Table III")
    for t in oc:
        if not oc[t]: continue
        vals = [fin["models"][m]["obfuscated_recall"].get(t, float("nan")) * 100 for m in ML_MODELS]
        L.append(f"| {t} ({oc[t]}) | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
        tex.append(f"{t} ({oc[t]}) & " + " & ".join(f"{v:.1f}" for v in vals) + " \\\\")
    L.append("| (reference) un-obfuscated bases | " + " | ".join(f"{fin['models'][m]['base_payload_recall'] * 100:.1f}" for m in ML_MODELS) + " |")

    L.append("\n## 8 External dataset (Table IV)\n")
    if (OUT / "external.json").exists():
        ex = load("external.json")
        nm = ex["dataset"]
        L.append(f"**Dataset: {nm} HTTP dataset** (Giménez, Villegas & Álvarez Marañón, CSIC 2010). Copy used: {ex['source']}. "
                 f"{ex['n_requests']:,} requests = {ex['n_valid']:,} valid + {ex['n_anomalous']:,} anomalous. "
                 "Confirm the dataset's own terms of use on the CSIC page before citing a licence (the GitHub mirror's GPL-3.0 "
                 "covers the converter, not necessarily the data).\n")
        L.append("Protocol (run once, no tuning on this data; models = the final fits on the full training set): each request is "
                 "reduced to what the API guard would inspect (query-string and form-body VALUES; not the URL path, headers or cookies), "
                 "values are decoded once (as the framework does) and then passed through `normalize()`; a request is flagged if ANY value "
                 "scores >= 0.5. Positive class = CSIC label `Anomalous`.\n")
        L.append(f"**Table IV — {nm} (request level, %)**\n\n| Model | Accuracy | F1 | FPR | Precision | Recall |\n|---|---|---|---|---|---|")
        tex.append(f"% Table IV — {nm}")
        for m in ["Random Forest", "XGBoost"]:
            r = ex["models"][m]
            L.append(f"| {m} | {r['accuracy']*100:.2f} | {r['f1']*100:.2f} | {r['fpr']*100:.2f} | {r['precision']*100:.2f} | {r['recall']*100:.2f} |")
            tex.append(f"{m} & {r['accuracy']*100:.2f} & {r['f1']*100:.2f} & {r['fpr']*100:.2f} \\\\")
        L.append("\nAll models (same protocol):\n\n| Model | Acc | F1 | FPR (all valid) | FPR (valid w/ fields) | Recall (all anomalous) | Recall (anomalous w/ fields) | Recall (SQLi-like subset, n=%d) |\n|---|---|---|---|---|---|---|---|" % ex["models"]["Random Forest"]["n_sqlish_anomalous"])
        for m, r in ex["models"].items():
            L.append(f"| {m} | {r['accuracy']*100:.2f} | {r['f1']*100:.2f} | {r['fpr']*100:.2f} | {r['fpr_valid_with_fields']*100:.2f} | "
                     f"{r['recall']*100:.2f} | {r['recall_anomalous_with_fields']*100:.2f} | {r['recall_sqlish_anomalous']*100:.1f} |")
        L.append("\n**How to read these numbers (put this in the paper, not just the table):**\n"
                 f"- CSIC's `Anomalous` class is not SQL injection: it also contains XSS, buffer overflow, information gathering, "
                 f"file disclosure, CRLF injection, and valid requests with typos. {ex['anomalous_without_scannable_fields']*100:.1f}% of anomalous "
                 "requests have no query/body value at all (attack is in the URL path, e.g. `.bak` probing), so a value-level SQLi detector "
                 "cannot see them. Low recall/F1/accuracy against this label therefore does NOT mean SQLi is being missed.\n"
                 f"- CSIC has no per-attack labels. As a rough proxy I flagged anomalous requests whose values match SQL syntax by a fixed regex "
                 f"(`SQLISH` in experiments.py, written before the models were scored and not tuned): {ex['anomalous_with_sqlish_value_share']*100:.1f}% of "
                 f"anomalous and {ex['valid_with_sqlish_value_share']*100:.1f}% of valid requests match. Recall on that weak subset is the last column. "
                 "It is a weak label, not ground truth, and the regex overlaps with what the model learned.\n"
                 f"- {ex['valid_without_scannable_fields']*100:.1f}% of valid requests carry no values (static pages) and are trivially negative; "
                 "the 'FPR (valid w/ fields)' column removes them.\n"
                 "- The few false positives on valid traffic are random-looking generated passwords/names (RF e.g. `2ef4n4o7ar`, `lin?6`; "
                 "XGBoost e.g. `F*ino`, `eXc971+Bl5`). XGBoost and LinearSVC are clearly worse than Random Forest on valid traffic here.\n"
                 "- Domain shift: CSIC is Spanish-language e-commerce traffic with Latin-1 percent-encoding; the training data is English/synthetic.\n")
    else:
        L.append("**NOT COMPUTED** — run `python experiments.py --stages external --external-csv <csic_final.csv>`.\n")

    if abl:
        L.append(f"## 9 Table V — ablations (model = {abl['model']}; one grouped 80/20 split, train {abl['n_train']:,} / test {abl['n_test']:,})\n")
        L.append("ΔF1 below is **absolute percentage points** vs the baseline row (relative change also given). "
                 "'Split' = grouped test split; 'Hold-out' = 100 hand-written inputs.\n")
        base = abl["variants"][next(iter(abl["variants"]))]
        L.append("| Variant | Split F1 % | Δ pp | Hold-out F1 % | Δ pp | Δ rel % (hold-out) | Obfuscated recall % |\n|---|---|---|---|---|---|---|")
        tex.append("% Table V")
        for n, v in abl["variants"].items():
            sf, hf = v["split_test"]["f1"] * 100, v["holdout"]["f1"] * 100
            bs, bh = base["split_test"]["f1"] * 100, base["holdout"]["f1"] * 100
            L.append(f"| {n} | {sf:.2f} | {sf - bs:+.2f} | {hf:.2f} | {hf - bh:+.2f} | {(hf - bh) / bh * 100:+.2f} | {v['obfuscated_recall'] * 100:.1f} |")
            tex.append(f"{n} & {sf:.2f} & {sf - bs:+.2f} & {hf:.2f} & {hf - bh:+.2f} & {v['obfuscated_recall'] * 100:.1f} \\\\")

    L.append("\n## 10 Table VI — calibration (pooled out-of-fold, 10 equal-width bins on P(malicious))\n\n| Model | Brier | ECE |\n|---|---|---|")
    tex.append("% Table VI")
    for m in ML_MODELS:
        c = cv["calibration"][m]
        L.append(f"| {m} | {c['brier']:.5f} | {c['ece']:.5f} |")
        tex.append(f"{m} & {c['brier']:.5f} & {c['ece']:.5f} \\\\")

    if lat:
        h = lat["hardware"]
        L.append(f"\n## 11 Table VII — latency\nHardware of the latency run: {h['cpu']}, {h['logical_cores']} logical core(s), {h['ram_gb']} GB RAM. "
                 f"Average fields per request: **{lat['fields_per_request']:.2f}** (mix mirrors send_traffic.py).\n")
        L.append("In-process = normalize() + predict_proba() per request (1000 requests, after warm-up):\n\n| Model | p50 ms | p95 ms | p99 ms | Size MB (joblib, compress=3) |\n|---|---|---|---|---|")
        tex.append("% Table VII")
        for m in ML_MODELS:
            q = lat["in_process_ms"][m]
            L.append(f"| {m} | {q['p50']:.2f} | {q['p95']:.2f} | {q['p99']:.2f} | {fin['models'][m]['size_mb_compressed']} |")
            tex.append(f"{m} & {q['p50']:.2f} & {q['p95']:.2f} & {q['p99']:.2f} & {fin['models'][m]['size_mb_compressed']} \\\\")
        L.append("\nEnd-to-end HTTP through the real FastAPI/Uvicorn app (single worker; model = models/best_model.joblib):\n\n| Concurrency | Requests | Req/s | p50 ms | p95 ms | p99 ms |\n|---|---|---|---|---|---|")
        for c, q in lat["http"].items():
            L.append(f"| {c} | {q['requests']} | {q['req_per_sec']:.1f} | {q['p50']:.1f} | {q['p95']:.1f} | {q['p99']:.1f} |")

    L.append("\n## 12 Model-selection statement (safe to paste)\n"
             f"\"The final model was selected solely on mean F1 (tie-break: lower FPR) across the {a.folds} grouped "
             "validation folds. The hand-written hold-out set, the obfuscated set and the ablations were evaluated after "
             "selection and played no role in it.\"  — true for experiments.py. NOTE: the model file shipped earlier in "
             "sqli_detector.zip was chosen by an older rule in train.py that DID include the hold-out F1; that rule has now "
             "been removed. Do not describe the shipped file with the statement above unless you retrain with the new train.py.\n")

    L.append("## NOT COMPUTED — you must supply these\n"
             "- ModSecurity CRS version, paranoia level and its row in Table II: no ModSecurity engine/Docker in the sandbox. "
             "Run OWASP CRS (e.g. the `owasp/modsecurity-crs` container) in front of a stub app and record which values it blocks.\n"
             "- Source dataset name, link, licence, and Reference #9: I do not know where your `sqli_by_type` zip came from.\n"
             "- Repository URL.\n"
             ""
             "- Hardware for the abstract: use the machine you actually ran on (the figures above are for this run).\n")
    (OUT / "paper_values.md").write_text("\n".join(L))
    (OUT / "tables.tex").write_text("\n".join(tex))
    print(f"[report] wrote {OUT / 'paper_values.md'} and tables.tex")


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=False)
    ap.add_argument("--per-class", type=int, default=10000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stages", default="build,cv,final,external,ablate,latency,report")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--external-csv", type=Path)
    ap.add_argument("--external-text-col", default="Query")
    ap.add_argument("--external-label-col", default="Label")
    ap.add_argument("--external-name", default="CSIC 2010")
    ap.add_argument("--external-source", default="https://github.com/msudol/Web-Application-Attack-Datasets (CSVData/csic_final.csv)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    random.seed(SEED); np.random.seed(SEED)
    stages = dict(build=stage_build, cv=stage_cv, final=stage_final, external=stage_external, ablate=stage_ablate, latency=stage_latency, report=stage_report)
    for s in a.stages.split(","):
        t = time.time()
        print(f"=== stage {s} ===", flush=True)
        stages[s](a)
        print(f"=== {s} done in {time.time() - t:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()

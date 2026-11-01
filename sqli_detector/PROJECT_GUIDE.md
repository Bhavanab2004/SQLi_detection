# Project guide: ML-based SQL injection detection for APIs

## 1. The problem in plain words
An API receives values from users (a username, a search term, a JSON field). If the backend pastes that value into a SQL
query, an attacker can send `' OR '1'='1` and change what the query does. That is **SQL injection (SQLi)**.
Rules/signatures ("block the word UNION") are easy to dodge (`UN/**/ION`, `uNiOn`, URL-encoding). This project
trains a model to *learn what attack text looks like* and puts it in front of an API as a guard.

## 2. What the system does, end to end
```
client request ──► FastAPI app ──► guard: pull out every user value ──► normalise ──► ML model ──► P(malicious)
                                                                                         │
                          P >= 0.5: block (HTTP 403) + log to SQLite      P < 0.5: let the request through
                                                                                         ▼
                                                                  Streamlit dashboard reads the SQLite log
```
| Part | File | Job |
|---|---|---|
| Training | `train.py` | builds the dataset, trains 3 models, saves the best |
| Helpers | `common.py`, `augment.py` | text clean-up; extra realistic examples |
| API guard | `api.py`, `db.py` | inspect requests, block/allow, log |
| Dashboard | `dashboard.py` | charts of blocked attacks |
| Tests | `tests/`, `postman/`, `send_traffic.py` | prove it works |
| Paper experiments | `experiments.py` | cross-validation, ablations, latency |
| Other-data testing | `evaluate_external.py`, `prepare_sqliv3.py` | score any CSV |

## 3. The data you trained on
Your `sqli_by_type.zip` is the **RbSQLi** dataset (Rule-based SQL Injection Dataset, Mendeley Data,
https://data.mendeley.com/datasets/xz4d5zj5yw — take authors/year/DOI from that page for your citation).
* 10,190,450 rows in 7 CSVs (verified by scanning them): six attack types (boolean, error, meta, stacked-queries, time,
  union = about 2.7M malicious) plus `none_type_benign` (about 7.5M benign).
* Column used: `sql_query`. Label comes from the file it sits in (attack file = 1, benign file = 0).
* Weaknesses you must know about (they affect results): the dataset is **synthetic/rule-generated** (its authors say part
  of it came from the public payloadbox payload list and part was produced with ChatGPT); the benign rows are
  **malformed pseudo-SQL** such as `UPDATE id, email FROM login ...`, and none contain `--`, `#` or `/*`; and rows are
  **full statements**, while an API guard sees individual **values**.

## 4. How training works, from scratch
1. **Sample.** The files are huge (1.3 GB), so `train.py` reads them in chunks and takes `--per-class` random rows from each
   attack file, then as many benign rows as malicious ones (balanced).
2. **Add realistic examples (`augment.py`).** (a) payload *fragments* cut from the malicious statements (the part after
   `WHERE id =`), (b) hand-made attack templates, (c) believable benign inputs (names, emails, product searches,
   phrases containing words like "select" or "union"). Reason: see the weaknesses above.
3. **Normalise (`common.py`).** URL-decode up to twice, replace the empty comment `/**/` with a space, collapse whitespace.
4. **Remove duplicates, split 80/20** (stratified, so both sets keep the same benign/malicious ratio).
5. **Turn text into numbers: TF-IDF on character n-grams.** The text is cut into every 1–5 character piece
   (`' O`, `OR '`, `1=1`, `--`…). Each piece gets a weight: common-everywhere pieces count little, telling pieces count a lot.
   Character pieces survive spelling tricks that word-based features miss.
6. **Train three classifiers** on those numbers: Logistic Regression (a weighted vote of features), SVM (draws the best
   separating boundary), Random Forest (many decision trees voting).
7. **Evaluate** on the 20% the model never saw: accuracy, precision (of what I blocked, how much was truly an attack),
   recall (of all attacks, how many I caught), F1 (balance of the two), false-positive rate (good requests wrongly
   blocked), confusion matrix.
8. **Save** the best pipeline with Joblib (`models/best_model.joblib`); the API loads it at start-up.

## 5. Run it, step by step
```bash
# 0. open a terminal in the project folder
python -m venv venv
venv\Scripts\activate                 # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# 1. (optional) retrain; a trained model is already included
python train.py --data-dir "C:\path\to\sqli_by_type" --per-class 10000

# 2. start the guarded API            (terminal 1)
uvicorn api:app --port 8000
# 3. start the dashboard              (terminal 2)
streamlit run dashboard.py
# 4. try it                           (terminal 3)
curl "http://127.0.0.1:8000/api/users?username=alice"                              # allowed (200)
curl -G http://127.0.0.1:8000/api/users --data-urlencode "username=' OR '1'='1"    # blocked (403)
python send_traffic.py --n 100        # mixed traffic -> refresh the dashboard
pytest -v                             # 16 automated tests
```
Postman: **Import** `postman/sqli_guard.postman_collection.json`, then **Run collection** (choose HTTP, not WebSocket, if you
create requests by hand).

## 6. Test with a different dataset (not your training/test data)
**Dataset: SQLiV3** (Syed Saqlain Hussain, Kaggle "SQL Injection Dataset"; mirrored in the GitHub repo
`nidnogg/sqliv5-dataset`, MIT licence). 30,845 usable rows after removing duplicates and conflicting labels:
11,340 malicious, 19,505 benign. It is independent of RbSQLi — I scanned all 10.19M RbSQLi rows and found **0** of its
texts verbatim (near-duplicates were not checked). Malicious rows look like real scanner payloads
(`1' ) ) or sleep ( 5 ) ...`); benign rows are a mix of names/numbers/addresses **and** well-formed SQL statements.

Ready to use: `external_data/sqliv3_clean.csv` (columns `text,label`).
```bash
python evaluate_external.py --csv external_data/sqliv3_clean.csv --text-col text --label-col label --positive 1
```
To rebuild it from the original file: download
https://raw.githubusercontent.com/nidnogg/sqliv5-dataset/main/SQLiV3.csv then
`python prepare_sqliv3.py SQLiV3.csv external_data/sqliv3_clean.csv`.
Output: accuracy, precision, recall, F1, FPR, FNR, confusion matrix, and `external_predictions.csv` (sort by `score`).

### Result I got (shipped model = Random Forest, run once, nothing tuned on this data)
| View | Rows | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| All SQLiV3 rows | 30,845 | 72.0% | 57.4% | 91.9% | 70.7% | 39.6% |
| Excluding benign full SQL statements (post-hoc) | 19,417 | 94.7% | 98.8% | 91.9% | 95.3% | 1.5% |

Why the gap: of the 19,505 benign rows, 11,428 are complete SQL statements (`SELECT AVG ( Price ) FROM who`). The model
flags 66.5% of them because it learned "SQL keywords arriving as input = suspicious". On plain values (names, numbers,
addresses) only 1.5% are flagged. For an API guard, blocking a whole SELECT statement inside a form field is arguably
*correct*, but SQLiV3 labels it benign, so it counts as an error. The "excluding" row was chosen **after** looking at the errors,
so report it as an analysis, never as the headline.
Other models on all rows (3,000-per-class fits): XGBoost 80.8% acc / 25.9% FPR, Logistic Regression 70.4% / 38.1%,
LinearSVC 63.6% / 53.4%.

## 7. Honest limits
* Near-100% scores on the 20% split are expected on rule-generated data; they do not predict real-world accuracy.
* The model is an extra layer. Parameterised queries are the real fix for SQLi.
* The guard inspects query/body values only, not URL paths, headers or cookies.

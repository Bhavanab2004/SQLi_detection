# ML-Based SQL Injection Detection for APIs

Pipeline: dataset → preprocessing → TF-IDF char n-grams → Logistic Regression / SVM / Random Forest →
best model saved with Joblib → FastAPI + Uvicorn guard → SQLite logs → Streamlit dashboard.

## Files
| File | Purpose |
|---|---|
| `train.py` | samples the CSVs, trains and compares 3 models, saves the best one |
| `augment.py` | builds API-style benign inputs and payload fragments (see "Honest notes") |
| `common.py` | input normalisation (URL-decoding etc.) and model loading |
| `api.py` | FastAPI app. Every request is inspected; malicious ones get HTTP 403 and are logged |
| `db.py` | SQLite logging (`data/detections.db`) |
| `dashboard.py` | Streamlit security monitor |
| `send_traffic.py` | fires benign + attack requests at the running API |
| `tests/` | pytest suite + hand-written hold-out payloads |
| `postman/` | Postman collection with expected-status checks |
| `models/` | trained `best_model.joblib`, `metrics.json`, `confusion_matrices.png` |

## Step-by-step

### 1. Setup (Python 3.10+)
```bash
cd sqli_detector
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. (Optional) Retrain the model on your zip
Unzip `sqli_by_type.zip` somewhere, then:
```bash
python train.py --data-dir "C:\path\to\sqli_by_type" --per-class 10000
```
`--per-class` = rows sampled from EACH of the 6 attack CSVs (benign is sampled to match).
The benign CSV is ~1 GB, so it is read in chunks (about 1 minute). Random Forest is the slow part;
use `--per-class 5000` on a weak laptop or `30000` if you have RAM and time.
A model trained with 10000 is already included, so you can skip this step.

### 3. Start the API (terminal 1)
```bash
uvicorn api:app --port 8000
```
Open http://127.0.0.1:8000/docs for interactive Swagger docs.

### 4. Start the dashboard (terminal 2)
```bash
streamlit run dashboard.py
```
Opens at http://localhost:8501.

### 5. Test with curl
```bash
curl "http://127.0.0.1:8000/api/users?username=alice"                          # -> 200 allowed
curl -G http://127.0.0.1:8000/api/users --data-urlencode "username=' OR '1'='1" # -> 403 blocked
curl -X POST http://127.0.0.1:8000/api/login -H "Content-Type: application/json" \
     -d "{\"username\":\"admin'--\",\"password\":\"x\"}"                        # -> 403 blocked
curl -X POST http://127.0.0.1:8000/detect -H "Content-Type: application/json" \
     -d "{\"text\":\"1 UNION SELECT NULL,NULL\"}"                               # -> malicious: true
```

### 6. Test with Postman
Import `postman/sqli_guard.postman_collection.json`, press **Run collection**.
Every request has an expected status (200/401 for benign, 403 for attacks) and a pass/fail check.

### 7. Generate traffic and watch the dashboard
```bash
python send_traffic.py --n 100
```
Click **Refresh** on the dashboard.

### 8. Run the automated tests
```bash
pytest -v
```
16 tests: model recall and false positives on hand-written payloads, benign requests allowed,
attacks blocked in query string / JSON body / URL-encoded form, and logging verified.
These use a throw-away database, so your real logs are not touched.

## How blocking works
For each request, `api.py` extracts every user-controlled value (query string, path parameters, JSON body
including nested fields and keys, form body), scores each with the model, and blocks the request with 403 if any
value has P(malicious) >= 0.5 (`THRESHOLD` in `common.py`). Lower it to catch more, raise it to cut false alarms.

## Honest notes
* The dataset is synthetic. Its benign rows are malformed pseudo-SQL (e.g. `UPDATE id, email FROM login WHERE ...`) and
  never contain `--`, `#` or `/*`. A model trained on it alone would learn shortcuts, so `augment.py` adds realistic
  benign inputs (names, emails, product searches, phrases containing words like "select" or "union") and payload
  fragments cut from the malicious statements. Real API clients send values, not full statements.
* Scores of ~100% on the 20% test split are expected on data this regular; do not quote them as real-world accuracy.
  The 100 hand-written hold-out inputs (`tests/holdout.py`) are the more honest number, but that is still a small set.
* This is an additional defence layer. The demo endpoints also use parameterised queries, which is the real fix for SQLi.

## Paper experiments
`python experiments.py --data-dir <sqli_by_type> --per-class 10000` runs grouped CV, obfuscation tests, ablations, calibration and latency
and writes `results/paper_values.md` + `results/tables.tex`. For the external CSIC 2010 evaluation download `CSVData/csic_final.csv` from
https://github.com/msudol/Web-Application-Attack-Datasets and add:
`--stages external,report --external-csv csic_final.csv --external-name "CSIC 2010"`.

## Testing on your own / external data
```bash
# labelled CSV -> metrics + confusion matrix
python evaluate_external.py --csv other_data.csv --text-col payload --label-col label --positive 1
# unlabelled CSV -> just scores + predictions
python evaluate_external.py --csv other_data.csv --text-col payload
```
Writes `external_predictions.csv` (score + predicted label per row). Use data the model was NOT trained on, and for full HTTP requests
(e.g. CSIC 2010) use `experiments.py --stages external` instead, which extracts query/body values the way the API does.

## Beginner guide + other-dataset test
See `PROJECT_GUIDE.md`. Quick external test: `python evaluate_external.py --csv external_data/sqliv3_clean.csv --text-col text --label-col label --positive 1`

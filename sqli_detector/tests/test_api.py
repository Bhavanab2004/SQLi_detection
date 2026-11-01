"""Automated tests.  Run from the project root:  pytest -v"""
import urllib.parse

import pytest
from fastapi.testclient import TestClient

import api
import db
from common import load_model, predict_malicious_proba
from tests.holdout import BENIGN, MALICIOUS


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # use a throw-away log DB so tests never pollute real logs
    import common
    tmp = tmp_path_factory.mktemp("db") / "test.db"
    common.DB_PATH = tmp
    db.DB_PATH = tmp
    api.DEMO_DB = tmp.parent / "shop.db"
    with TestClient(api.app) as c:
        yield c


# ---------------------------------------------------------------- model-level tests
@pytest.fixture(scope="module")
def model():
    return load_model()


def test_model_holdout_recall(model):
    scores = predict_malicious_proba(model, MALICIOUS)
    recall = (scores >= 0.5).mean()
    missed = [m for m, s in zip(MALICIOUS, scores) if s < 0.5]
    assert recall >= 0.90, f"recall {recall:.2%}, missed: {missed}"


def test_model_holdout_false_positive_rate(model):
    scores = predict_malicious_proba(model, BENIGN)
    fpr = (scores >= 0.5).mean()
    flagged = [b for b, s in zip(BENIGN, scores) if s >= 0.5]
    assert fpr <= 0.10, f"FPR {fpr:.2%}, flagged: {flagged}"


# ---------------------------------------------------------------- API-level tests
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_benign_get_is_allowed(client):
    r = client.get("/api/users", params={"username": "alice"})
    assert r.status_code == 200
    assert r.json()["results"][0]["username"] == "alice"


def test_benign_search_with_apostrophe_is_allowed(client):
    r = client.get("/api/products/search", params={"q": "wireless mouse"})
    assert r.status_code == 200


@pytest.mark.parametrize("payload", ["' OR '1'='1", "1; DROP TABLE users", "' UNION SELECT username, password FROM users --",
                                     "admin'--", "1' AND SLEEP(5) --"])
def test_malicious_query_param_is_blocked(client, payload):
    r = client.get("/api/users", params={"username": payload})
    assert r.status_code == 403
    assert "SQL injection" in r.json()["detail"]["error"]


def test_url_encoded_payload_is_blocked(client):
    enc = urllib.parse.quote("' OR '1'='1' --")
    r = client.get(f"/api/users?username={urllib.parse.quote(enc)}")  # double-encoded
    assert r.status_code == 403


def test_malicious_json_body_is_blocked(client):
    r = client.post("/api/login", json={"username": "admin' --", "password": "x"})
    assert r.status_code == 403


def test_malicious_nested_json_is_blocked(client):
    r = client.post("/detect", json={"text": "hi"})  # /detect is unguarded
    assert r.status_code == 200
    r = client.post("/api/login", json={"username": "bob", "password": "' OR '1'='1"})
    assert r.status_code == 403


def test_benign_login_reaches_endpoint(client):
    r = client.post("/api/login", json={"username": "alice", "password": "wrong"})
    assert r.status_code == 401  # got past the guard, rejected by business logic


def test_detect_endpoint_verdicts(client):
    bad = client.post("/detect", json={"text": "1 UNION SELECT NULL,NULL,NULL"}).json()
    good = client.post("/detect", json={"text": "red running shoes"}).json()
    assert bad["malicious"] is True and good["malicious"] is False


def test_blocked_requests_are_logged(client):
    before = client.get("/stats").json()["blocked"]
    client.get("/api/users", params={"username": "' OR 1=1 --"})
    after = client.get("/stats").json()["blocked"]
    assert after == before + 1
    last = client.get("/logs", params={"limit": 1}).json()[0]
    assert last["action"] == "BLOCKED" and "1=1" in last["payload"]

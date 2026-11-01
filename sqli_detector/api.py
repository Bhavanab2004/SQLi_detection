"""FastAPI application protected by the ML SQL-injection detector.

Every request is inspected by a global dependency (`sqli_guard`):
  1. user-controlled values are extracted (query string, path params, JSON/form body)
  2. each value is scored by the trained model
  3. malicious -> HTTP 403 + logged;  benign -> request continues to the endpoint
Run:  uvicorn api:app --reload --port 8000
"""
import json
import sqlite3
from contextlib import asynccontextmanager
from typing import Any, Iterator, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

import db
from common import THRESHOLD, load_model, predict_malicious_proba

MODEL = None
UNGUARDED = {"/", "/health", "/docs", "/openapi.json", "/redoc", "/stats", "/logs", "/detect"}
MAX_VALUE_LEN = 2000  # very long values are truncated before scoring


def flatten(obj: Any, prefix: str = "") -> Iterator[Tuple[str, str]]:
    """Yield (field_name, string_value) for every leaf of a JSON structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
            yield (f"{prefix}.{k}[key]" if prefix else f"{k}[key]"), str(k)  # keys can carry payloads too
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    elif obj is not None and not isinstance(obj, bool):
        yield prefix or "value", str(obj)


async def extract_inputs(request: Request):
    items = [(f"query.{k}", v) for k, v in request.query_params.multi_items()]
    items += [(f"path.{k}", str(v)) for k, v in request.path_params.items()]
    ctype = request.headers.get("content-type", "")
    body = await request.body()
    if body:
        if "json" in ctype:
            try:
                items += [(f"body.{k}", v) for k, v in flatten(json.loads(body))]
            except ValueError:
                items.append(("body.raw", body.decode("utf-8", "ignore")))
        elif "x-www-form-urlencoded" in ctype:
            from urllib.parse import parse_qsl
            items += [(f"form.{k}", v) for k, v in parse_qsl(body.decode("utf-8", "ignore"))]
        else:
            items.append(("body.raw", body.decode("utf-8", "ignore")))
    return [(f, v[:MAX_VALUE_LEN]) for f, v in items if v and v.strip()]


async def sqli_guard(request: Request):
    if request.url.path in UNGUARDED:
        return
    inputs = await extract_inputs(request)
    if not inputs:
        return
    scores = predict_malicious_proba(MODEL, [v for _, v in inputs])
    ip = request.client.host if request.client else "unknown"
    worst = max(range(len(inputs)), key=lambda i: scores[i])
    if scores[worst] >= THRESHOLD:
        field, value = inputs[worst]
        db.log_event(ip, request.method, request.url.path, field, value, scores[worst], "BLOCKED")
        raise HTTPException(status_code=403, detail={
            "error": "Request blocked: possible SQL injection",
            "field": field, "score": round(float(scores[worst]), 4)})
    # log the highest-scoring benign value so the dashboard can show traffic volume
    field, value = inputs[worst]
    db.log_event(ip, request.method, request.url.path, field, value, scores[worst], "ALLOWED")


@asynccontextmanager
async def lifespan(_app):
    global MODEL
    db.init_db()
    MODEL = load_model()
    _seed_demo_db()
    yield


app = FastAPI(title="ML SQL Injection Guard", version="1.0", dependencies=[Depends(sqli_guard)], lifespan=lifespan)


# ----------------------------------------------------------------------------- demo data store
DEMO_DB = db.DB_PATH.parent / "shop.db"


def _seed_demo_db():
    con = sqlite3.connect(DEMO_DB)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, email TEXT, password TEXT);
        CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL);
    """)
    if not con.execute("SELECT 1 FROM users").fetchone():
        con.executemany("INSERT INTO users (username,email,password) VALUES (?,?,?)", [
            ("alice", "alice@example.com", "x"), ("bob", "bob@example.com", "x"), ("john", "john@example.com", "x")])
        con.executemany("INSERT INTO products (name,price) VALUES (?,?)", [
            ("laptop", 899.0), ("wireless mouse", 19.5), ("usb-c cable", 9.0), ("mechanical keyboard", 79.0)])
    con.commit(); con.close()


def query(sql, params=()):
    con = sqlite3.connect(DEMO_DB); con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(sql, params)]
    con.close()
    return rows


# ----------------------------------------------------------------------------- endpoints
class Login(BaseModel):
    username: str
    password: str


class Probe(BaseModel):
    text: str


@app.get("/")
def root():
    return {"service": "ML SQLi Guard", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/api/users")
def get_user(username: str):
    """Demo endpoint (uses parameterised SQL; the ML guard is an extra layer)."""
    return {"results": query("SELECT id, username, email FROM users WHERE username = ?", (username,))}


@app.get("/api/products/search")
def search_products(q: str):
    return {"results": query("SELECT id, name, price FROM products WHERE name LIKE ?", (f"%{q}%",))}


@app.post("/api/login")
def login(body: Login):
    rows = query("SELECT id, username FROM users WHERE username = ? AND password = ?", (body.username, body.password))
    if not rows:
        raise HTTPException(401, "Invalid credentials")
    return {"message": "login ok", "user": rows[0]}


@app.post("/detect")
def detect(body: Probe, request: Request):
    """Classify any string and get the verdict back (nothing is executed). Logged like normal traffic."""
    score = float(predict_malicious_proba(MODEL, [body.text])[0])
    malicious = score >= THRESHOLD
    ip = request.client.host if request.client else "unknown"
    db.log_event(ip, "POST", "/detect", "body.text", body.text, score, "BLOCKED" if malicious else "ALLOWED")
    return {"text": body.text, "malicious": malicious, "score": round(score, 4), "threshold": THRESHOLD}


@app.get("/stats")
def stats():
    rows = db.fetch_all(100000)
    blocked = sum(r["action"] == "BLOCKED" for r in rows)
    return {"total_requests_inspected": len(rows), "blocked": blocked, "allowed": len(rows) - blocked}


@app.get("/logs")
def logs(limit: int = 50, action: str = "BLOCKED"):
    return [r for r in db.fetch_all(5000) if r["action"] == action][:limit]

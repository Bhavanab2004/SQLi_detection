"""SQLite helpers for the detection log."""
import sqlite3
from contextlib import closing

from common import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','localtime')),
    client_ip TEXT, method TEXT, path TEXT, field TEXT,
    payload TEXT, score REAL, action TEXT   -- action: BLOCKED | ALLOWED
);
CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(ts);
"""


def connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with closing(connect()) as con:
        con.executescript(SCHEMA)
        con.commit()


def log_event(client_ip, method, path, field, payload, score, action):
    with closing(connect()) as con:
        con.execute(
            "INSERT INTO detections (client_ip, method, path, field, payload, score, action) VALUES (?,?,?,?,?,?,?)",
            (client_ip, method, path, field, payload[:500], float(score), action))
        con.commit()


def fetch_all(limit=5000):
    with closing(connect()) as con:
        return [dict(r) for r in con.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,))]

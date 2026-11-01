"""Send a mix of benign and malicious requests to a RUNNING API (fills the dashboard).
    python send_traffic.py --n 60
"""
import argparse
import random

import requests

from tests.holdout import BENIGN, MALICIOUS

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()

stats = {"allowed": 0, "blocked": 0, "other": 0}
for _ in range(a.n):
    bad = random.random() < 0.5
    payload = random.choice(MALICIOUS if bad else BENIGN)
    kind = random.choice(["get_user", "search", "login"])
    if kind == "get_user":
        r = requests.get(f"{a.url}/api/users", params={"username": payload})
    elif kind == "search":
        r = requests.get(f"{a.url}/api/products/search", params={"q": payload})
    else:
        r = requests.post(f"{a.url}/api/login", json={"username": payload, "password": "pw"})
    key = "blocked" if r.status_code == 403 else "allowed" if r.status_code < 500 else "other"
    stats[key] += 1
    print(f"{'ATTACK' if bad else 'benign':7} {r.status_code}  {payload[:60]}")
print(stats)

"""Shared helpers: input normalisation, model loading, scoring."""
import re
from pathlib import Path
from urllib.parse import unquote_plus

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
DB_PATH = BASE_DIR / "data" / "detections.db"
THRESHOLD = 0.5  # probability above which an input is blocked

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Decode URL-encoding (twice, to catch double-encoding), strip comment
    obfuscation such as UN/**/ION, and collapse whitespace."""
    text = str(text)
    for _ in range(2):
        text = unquote_plus(text)
    text = text.replace("/**/", " ")
    return _WS.sub(" ", text).strip()


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict_malicious_proba(model, texts):
    """Return P(malicious) for a list of raw strings."""
    cleaned = [normalize(t) for t in texts]
    return model.predict_proba(cleaned)[:, 1]

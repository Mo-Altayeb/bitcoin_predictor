# models/artifact_fetcher.py
import os
import requests
from pathlib import Path

GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "Mo-Altayeb")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "bitcoin_predictor")
BRANCH = os.environ.get("MODEL_BRANCH", "model-artifacts")
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{BRANCH}/models/saved_models"

LOCAL_DIR = Path("models/saved_models")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

def _get(url, token=None, timeout=60):
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    r = requests.get(url, headers=headers, timeout=timeout)
    return r

def fetch_artifact(filename: str, token: str | None = None) -> bool:
    url = f"{RAW_BASE}/{filename}"
    r = _get(url, token=token)
    if r.status_code == 200:
        (LOCAL_DIR / filename).write_bytes(r.content)
        print(f"Fetched artifact: {filename}")
        return True
    else:
        print(f"Failed fetching {filename}: {r.status_code} {url}")
        return False

def fetch_latest_model(token: str | None = None) -> bool:
    ok1 = fetch_artifact("bitcoin_model.pkl", token=token)
    ok2 = fetch_artifact("feature_info.json", token=token)
    # optional: wikipedia_edits.csv
    fetch_artifact("wikipedia_edits.csv", token=token)
    return ok1 and ok2

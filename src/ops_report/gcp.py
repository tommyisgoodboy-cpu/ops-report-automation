"""Minimal Google Cloud helpers using only the standard library (no API keys in code).

Auth, in order:
1. On Cloud Run (or any GCP runtime): the metadata server gives a short-lived access token
   for the service account the service runs as.
2. On a developer machine: `gcloud auth print-access-token` for the logged-in user.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

_METADATA = "http://metadata.google.internal/computeMetadata/v1"
_cache: dict = {"token": None, "expires": 0.0}


def project_id() -> str:
    """GOOGLE_CLOUD_PROJECT if set, else the metadata server, else gcloud config."""
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return os.environ["GOOGLE_CLOUD_PROJECT"]
    try:
        return _metadata("project/project-id")
    except OSError:
        out = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    raise RuntimeError("Cannot determine the Google Cloud project. Set GOOGLE_CLOUD_PROJECT.")


def _metadata(path: str) -> str:
    request = urllib.request.Request(f"{_METADATA}/{path}", headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.read().decode("utf-8")


def access_token() -> str:
    """Cached access token (refreshed a minute before it expires)."""
    if _cache["token"] and time.time() < _cache["expires"] - 60:
        return _cache["token"]
    try:
        data = json.loads(_metadata("instance/service-accounts/default/token"))
        _cache.update(token=data["access_token"], expires=time.time() + float(data["expires_in"]))
    except OSError:
        out = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
        if out.returncode != 0 or not out.stdout.strip():
            raise RuntimeError("No Google credentials: run on Cloud Run, or `gcloud auth login` locally.")
        _cache.update(token=out.stdout.strip(), expires=time.time() + 1800)
    return _cache["token"]


def vertex_model_url(project: str, location: str, model: str, method: str) -> str:
    """URL of a Google publisher model on Vertex AI.

    The `global` location uses the bare host `aiplatform.googleapis.com`; every regional
    location uses `<location>-aiplatform.googleapis.com`.
    """
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:{method}"


def post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {access_token()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"Google API error {exc.code}: {detail}") from exc

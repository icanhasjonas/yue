"""RunPod over plain HTTPS -- no SDK on the local side.

Two hosts, both with `Authorization: Bearer $RUNPOD_API_KEY`:

  rest.runpod.io/v1   management: templates, endpoints, network volumes, registry auth
                      (request bodies from https://rest.runpod.io/v1/openapi.json)
  api.runpod.ai/v2    jobs: /{endpoint}/run, /stream/{job}, /status/{job},
                      /cancel/{job}, /health (the same paths runpod.endpoint.runner uses)

Local state lives in ~/.config/yue/runpod.json (ids only, never the key).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

REST = "https://rest.runpod.io/v1"
JOBS = "https://api.runpod.ai/v2"
GRAPHQL = "https://api.runpod.io/graphql"
CONFIG = Path(os.environ.get("YUE_RUNPOD_CONFIG", Path.home() / ".config" / "yue" / "runpod.json"))


class RunPodError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"RunPod {status}: {message}")
        self.status = status


def job_key() -> str:
    """Jobs prefer the endpoint-restricted key; management never uses it."""
    return os.environ.get("YUE_RUNPOD_JOB_KEY") or api_key()


def api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RunPodError(0, "no API key: set RUNPOD_API_KEY or pass --api-key "
                             "(https://www.console.runpod.io/user/settings -> API Keys)")
    return key


def request(method: str, url: str, key: str, body=None, timeout: float = 30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": "yue-cli"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RunPodError(exc.code, detail or exc.reason) from exc
    except urllib.error.URLError as exc:
        raise RunPodError(0, f"network: {exc.reason}") from exc
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode(errors="replace")


# --- management --------------------------------------------------------------------
def rest(method: str, path: str, key: str, body=None):
    return request(method, f"{REST}{path}", key, body)


def graphql(query: str, key: str):
    out = request("POST", GRAPHQL, key, {"query": query})
    if isinstance(out, dict) and out.get("errors"):
        raise RunPodError(400, "; ".join(e.get("message", "") for e in out["errors"]))
    return out["data"]


def gpu_stock(key: str) -> dict[str, dict[str, str]]:
    """data center -> {gpu type id -> stock status}. Pod stock: a proxy for serverless supply."""
    data = graphql("query { dataCenters { id listed gpuAvailability { gpuTypeId stockStatus } } }", key)
    return {dc["id"]: {g["gpuTypeId"]: g["stockStatus"] for g in (dc["gpuAvailability"] or []) if g["stockStatus"]}
            for dc in data["dataCenters"] if dc.get("listed")}


# --- jobs ------------------------------------------------------------------------------
def run(endpoint: str, key: str, payload: dict) -> str:
    out = request("POST", f"{JOBS}/{endpoint}/run", key, {"input": payload}, timeout=120)
    return out["id"]


def stream(endpoint: str, job: str, key: str) -> dict:
    return request("GET", f"{JOBS}/{endpoint}/stream/{job}", key, timeout=60)


def status(endpoint: str, job: str, key: str) -> dict:
    return request("GET", f"{JOBS}/{endpoint}/status/{job}", key)


def cancel(endpoint: str, job: str, key: str) -> dict:
    return request("POST", f"{JOBS}/{endpoint}/cancel/{job}", key)


def health(endpoint: str, key: str) -> dict:
    return request("GET", f"{JOBS}/{endpoint}/health", key)


# --- local config ------------------------------------------------------------------------
def load_config() -> dict:
    return json.loads(CONFIG.read_text()) if CONFIG.is_file() else {}


def save_config(config: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n")
    tmp.replace(CONFIG)

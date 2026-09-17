"""RunPod serverless worker: runs a `yue` verb inside the container and streams it back.

Job input (built by remote/client.py):

    {"argv":  ["generate", "--prompt", "...", ...],   # the verb and its switches, minus -w/--remote
     "files": {"style.txt": "<b64>", "2-tokens/semantic.npy": "<b64>", ...},  # the workspace to start from
     "fetch": "all" | "audio"}                        # which results to send back

Every `yield` becomes one entry of RunPod's `/stream/{job}` feed:

    {"k": "event", "e": <run-events v1 object>}       # the verb's own --output-format stream-json lines
    {"k": "file", "path": "4-audio/audio.flac", "i": 0, "n": 17, "data": "<b64 chunk>"}
    {"k": "done", "exit_code": 0}

Files are chunked under 1 MiB of base64 per yield: a single RunPod return body is
capped at 20 MB (runpod.serverless.modules.rp_tips.check_return_size), and one
FLAC master of a 4-minute song is ~40 MB.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHUNK = 768 * 1024  # raw bytes per chunk -> ~1 MiB of base64
RESULT_DIRS = ("1-plan", "2-tokens", "3-latents", "4-audio", "refine", "transcription")
RESULT_FILES = ("job.json", "style.txt", "lyrics.txt", "score.abc", "score.rejected.abc")
NEVER_SEND = {".history", "init"}


def _safe(rel: str) -> str:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"refusing workspace path {rel!r}")
    return rel


def prime(job_input: dict):
    """Download the weights onto the network volume (HF_HOME) so later cold starts only read them."""
    from huggingface_hub import snapshot_download
    repos = job_input.get("repos") or ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"]
    for index, repo in enumerate(repos, 1):
        yield {"k": "event", "e": {"type": "log", "level": "info", "message": f"downloading {repo} ({index}/{len(repos)})"}}
        path = snapshot_download(repo)
        yield {"k": "event", "e": {"type": "log", "level": "info", "message": f"{repo} ready at {path}"}}
    total = shutil.disk_usage(os.environ.get("HF_HOME", "/"))
    yield {"k": "done", "exit_code": 0, "free_gb": round(total.free / 2**30, 1)}


def run_job(job_input: dict):
    argv = list(job_input.get("argv") or [])
    if not argv:
        yield {"k": "done", "exit_code": 1, "error": "no argv"}
        return
    if argv[0] == "__prime__":
        yield from prime(job_input)
        return
    root = Path(tempfile.mkdtemp(prefix="yue-job-"))
    ws = root / "ws"
    ws.mkdir()
    try:
        for rel, blob in (job_input.get("files") or {}).items():
            target = ws / _safe(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(blob))
        before = {p: p.stat().st_mtime_ns for p in ws.rglob("*") if p.is_file()}
        cmd = [sys.executable, "-m", "yuecli.cli", *argv, "-w", str(ws), "--output-format", "stream-json"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1,
                                cwd=root, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            _relativize(event, str(ws))
            yield {"k": "event", "e": event}
        code = proc.wait()
        for rel, path in _results(ws, before, job_input.get("fetch", "all")):
            data = path.read_bytes()
            n = max(1, -(-len(data) // CHUNK))
            for i in range(n):
                yield {"k": "file", "path": rel, "i": i, "n": n,
                       "data": base64.b64encode(data[i * CHUNK:(i + 1) * CHUNK]).decode("ascii")}
        yield {"k": "done", "exit_code": code}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _results(ws: Path, before: dict, fetch: str):
    for path in sorted(ws.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ws)
        if rel.parts[0] in NEVER_SEND or rel.parts[0].endswith(".partial"):
            continue
        changed = before.get(path) != path.stat().st_mtime_ns
        if not changed:
            continue
        if fetch == "audio" and not (rel.parts[0] == "4-audio" or rel.name.startswith("song.") or
                                     str(rel) in RESULT_FILES or rel.name == "stage.json"):
            continue
        if rel.parts[0] in RESULT_DIRS or str(rel) in RESULT_FILES or rel.name.startswith("song."):
            yield str(rel), path


def _relativize(value, prefix: str):
    """Workspace paths inside events become `ws://relative` so the client can map them."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and item.startswith(prefix):
                value[key] = "ws://" + item[len(prefix):].lstrip("/")
            else:
                _relativize(item, prefix)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, str) and item.startswith(prefix):
                value[index] = "ws://" + item[len(prefix):].lstrip("/")
            else:
                _relativize(item, prefix)


def handler(job):
    yield from run_job(job.get("input") or {})


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})

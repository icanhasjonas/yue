"""SheetSage2 runs in its own uv environment (envs/sheetsage2): its pins
(torch 2.8, transformers 4.45, numpy 1.24) cannot share an interpreter with
yue2-infer's (torch 2.10, transformers 4.57, numpy 2.2). This module never
imports it -- it runs `worker.py` there as a subprocess and reads JSON back.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV = REPO / "envs" / "sheetsage2"
WORKER = Path(__file__).with_name("sheetsage2_worker.py")


def ensure_env(log) -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("`uv` is not on PATH; it manages the SheetSage2 environment")
    if not (ENV / "pyproject.toml").is_file():
        raise RuntimeError(f"{ENV} is missing; this checkout has no SheetSage2 environment")
    if not (ENV / ".venv").is_dir():
        log("creating the SheetSage2 environment (one-time, ~2 GB)")
        subprocess.run([uv, "sync", "--project", str(ENV), "--python", "3.11"], check=True)
    return uv


def transcribe(audio: Path, output: Path, *, melody_only: bool, device: str | None, dtype: str | None,
               preset: str | None, max_seconds: float | None, render_score: bool, log) -> dict:
    uv = ensure_env(log)
    output.mkdir(parents=True, exist_ok=True)
    cmd = [uv, "run", "--project", str(ENV), "python", str(WORKER),
           "--input", str(audio), "--output", str(output)]
    if melody_only:
        cmd.append("--melody-only")
    if render_score:
        cmd.append("--render-score")
    for flag, value in (("--device", device), ("--dtype", dtype), ("--preset", preset), ("--max-seconds", max_seconds)):
        if value is not None:
            cmd += [flag, str(value)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("{"):
            try:
                result = json.loads(line)
                continue
            except json.JSONDecodeError:
                pass
        if line:
            log(f"sheetsage2| {line}")
    stderr = proc.stderr.read()
    if proc.wait() or result is None:
        tail = " | ".join(stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"SheetSage2 failed (exit {proc.returncode}): {tail}")
    return result

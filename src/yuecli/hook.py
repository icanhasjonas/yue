"""Hand the score to an external tool (Claude Code, a script, your editor) and take it back.

The contract is files and environment variables, nothing else, so any tool works:

    cwd              the workspace
    YUE_WORKSPACE    the workspace
    YUE_ABC          <ws>/score.abc        edit IN PLACE; this is what gets performed
    YUE_ABC_ORIGINAL <ws>/1-plan/score.abc read-only: what the planner (or last render) produced
    YUE_STYLE_FILE   <ws>/style.txt        may be edited; read back
    YUE_LYRICS_FILE  <ws>/lyrics.txt       may be edited; read back
    YUE_BRIEF        upstream's score-editing brief (markdown)
    YUE_JOB          <ws>/job.json         every setting of this song

After the command exits 0 the score is parsed with the upstream dialect checker
(fail closed: a malformed score would condition a 20-minute render on garbage)
and a before/after comparison is written to <ws>/refine/<n>.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from . import abc_tools
from .workspace import Workspace, write_json

BRIEF = Path(__file__).with_name("edit_brief.md")


class HookFailed(RuntimeError):
    pass


def run_hook(ws: Workspace, command: str, *, timeout: float | None, validate: bool, log) -> dict:
    abc_path = ws.root / "score.abc"
    if not abc_path.is_file():
        raise HookFailed(f"{abc_path} does not exist; plan or transcribe first")
    before = {name: ws.read_text(name) for name in ("score.abc", "style.txt", "lyrics.txt")}
    env = {
        **os.environ,
        "YUE_WORKSPACE": str(ws.root),
        "YUE_ABC": str(abc_path),
        "YUE_ABC_ORIGINAL": str(ws.stage_dir("plan") / "score.abc"),
        "YUE_STYLE_FILE": str(ws.root / "style.txt"),
        "YUE_LYRICS_FILE": str(ws.root / "lyrics.txt"),
        "YUE_BRIEF": str(BRIEF),
        "YUE_JOB": str(ws.job_path),
    }
    log(f"hook: {command}")
    start = time.perf_counter()
    try:
        proc = subprocess.run(command, shell=True, cwd=ws.root, env=env, timeout=timeout,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        raise HookFailed(f"hook timed out after {timeout:g}s") from exc
    seconds = time.perf_counter() - start
    for line in (proc.stdout or "").splitlines()[-40:]:
        log(f"hook| {line}")
    if proc.returncode:
        tail = (proc.stderr or "").strip().splitlines()[-10:]
        raise HookFailed(f"hook exited {proc.returncode}" + (": " + " | ".join(tail) if tail else ""))
    after = {name: ws.read_text(name) for name in before}
    changed = [name for name in before if before[name] != after[name]]
    report: dict = {"command": command, "seconds": seconds, "changed": changed}
    if validate:
        try:
            new = abc_tools.parse_abc(after["score.abc"] or "")
        except abc_tools.AbcError as exc:
            abc_path.write_text(before["score.abc"], encoding="utf-8")
            (ws.root / "score.rejected.abc").write_text(after["score.abc"] or "", encoding="utf-8")
            raise HookFailed(f"hook left a score the native dialect rejects ({exc}); "
                             "restored the previous score, kept the attempt as score.rejected.abc") from exc
        try:
            old = abc_tools.parse_abc(before["score.abc"])
            report["compare"] = abc_tools.compare(old, new, allow_tempo_change=True)
        except abc_tools.AbcError:
            report["compare"] = None
        report["score"] = {k: v for k, v in abc_tools.report(new).items() if k != "voices"}
    n = len(list((ws.root / "refine").glob("*.json"))) + 1 if (ws.root / "refine").is_dir() else 1
    write_json(ws.root / "refine" / f"{n:03d}.json", json.loads(json.dumps(report, default=str)))
    return report

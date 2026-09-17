"""Run a yue verb on RunPod and land the result in the LOCAL workspace, as if it ran here.

    local workspace ──files (b64)──▶ /run ──▶ worker: yue <verb> -w /tmp/ws --output-format stream-json
    local reporter  ◀──events────── /stream/{job} ◀── yield per event / per file chunk
    local workspace ◀──stage dirs── reassembled chunks, old stage dirs archived to .history/

The remote run's own run-events are forwarded (renumbered, paths mapped back), so
`--output-format stream-json` looks the same locally or remote. Its `result` is
held back until the files have landed, so `result` stays the last line.
"""
from __future__ import annotations

import base64
import json
import shutil
import time
from pathlib import Path

from ..args import fail
from ..workspace import DIRS, Workspace
from . import runpod_api as rp

SEND_DIRS = ("1-plan", "2-tokens", "3-latents")  # 4-audio is an output, never an input
SEND_FILES = ("job.json", "style.txt", "lyrics.txt", "score.abc")
LOCAL_ONLY = {"workspace", "output", "remote", "remote_fetch", "output_format", "quiet", "debug"}
MAX_INPUT_BYTES = 9 * 1024 * 1024  # RunPod caps a /run body at 10 MB
FINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def remote_argv(verb, values: dict, given: set[str], ws: Workspace, uploads: dict[str, bytes]) -> list[str]:
    """Rebuild argv from what the user actually set, with local files turned into uploads."""
    argv = [verb.name]
    for f in verb.fields:
        if f.name in LOCAL_ONLY or f.name not in given or values.get(f.name) is None:
            continue
        value = values[f.name]
        if f.kind == "bool":
            argv.append(f.switch if value else f"--no-{f.switch[2:]}")
            continue
        if f.kind == "path":
            path = Path(value).expanduser()
            if path.is_dir() and (path / "job.json").is_file():
                # a workspace given as an input (remix -i ws, render --source ws): send its inputs
                for rel, blob in workspace_files(Workspace(path)).items():
                    uploads[f"{f.name}/{rel}"] = blob
                value = f"inputs/{f.name}"
            elif path.is_file():
                rel = f"{f.name}{path.suffix}"
                uploads[rel] = path.read_bytes()
                value = f"inputs/{rel}"  # the worker runs with cwd = the job root; inputs/ sits beside ws/
            else:
                fail(f"`{f.switch} {value}`: no such file or yue workspace")
        argv += [f.switch, str(value)]
    return argv


def workspace_files(ws: Workspace) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for name in SEND_FILES:
        path = ws.root / name
        if path.is_file():
            files[name] = path.read_bytes()
    for d in SEND_DIRS:
        base = ws.root / d
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file():
                    files[str(path.relative_to(ws.root))] = path.read_bytes()
    return files


def encode_payload(ws: Workspace, uploads: dict[str, bytes]) -> tuple[dict[str, str], dict[str, str]]:
    files = workspace_files(ws)
    size = (sum(len(v) for v in files.values()) + sum(len(v) for v in uploads.values())) * 4 // 3
    if size > MAX_INPUT_BYTES:
        fail(f"the job inputs are {size / 2**20:.1f} MB encoded; RunPod accepts 10 MB per job",
             ["Compress input audio to mp3/opus first, or run that stage locally."])
    enc = lambda d: {k: base64.b64encode(v).decode("ascii") for k, v in d.items()}  # noqa: E731
    return enc(files), enc(uploads)


def collect_files(ws: Workspace, uploads: dict[str, bytes]) -> dict[str, str]:
    files, inputs = encode_payload(ws, uploads)
    return {**files, **{f"inputs/{k}": v for k, v in inputs.items()}}


def run_remote_with_hooks(verb, values: dict, given: set[str], ws: Workspace, reporter, *, fetch: str = "all") -> int:
    """A score hook runs HERE (Claude Code, your editor), the GPU stages run there.

    generate: remote `--until plan` -> local hook -> remote `--resume` for the rest
    plan:     remote plan -> local hook
    render:   local hook on the workspace's score -> remote render
    """
    hook = values.get("abc_hook")
    if not hook or verb.name not in ("generate", "plan", "render", "remix"):
        return run_remote(verb, values, given, ws, reporter, fetch=fetch)
    from ..hook import run_hook
    from ..verbs import VERBS
    if verb.name == "remix":
        fail("--abc-hook with a remote remix is not supported",
             ["Remix first, then `yue refine -w <new workspace> --with '...'` and `yue render --remote runpod`."])
    base_given = given - {"abc_hook", "hook_timeout"}
    log = lambda m: reporter.log(m, task_id=None) if reporter.mode != "text" else reporter.human("hook| " + m)  # noqa: E731
    if verb.name == "render":
        run_hook(ws, hook, timeout=values.get("hook_timeout"), validate=True, log=log)
        return run_remote(verb, values, base_given, ws, reporter, fetch=fetch)
    first_verb = VERBS["generate"] if verb.name == "generate" else verb
    first = dict(values)
    first_given = set(base_given)
    if verb.name == "generate":
        first["until"] = "plan"
        first_given.add("until")
    code = run_remote(first_verb, first, first_given, ws, reporter, fetch="all", final=verb.name == "plan")
    if code:
        return code
    report = run_hook(ws, hook, timeout=values.get("hook_timeout"), validate=True, log=log)
    if report["changed"]:
        job = ws.job()
        job["score_mode"] = "provided"
        ws.save_job(job)
    if verb.name == "plan":
        return 0
    rest = dict(values)
    rest["resume"] = True
    rest_given = (base_given | {"resume"}) - {"prompt", "lyrics", "abc"}  # the texts now live in the workspace
    return run_remote(verb, rest, rest_given, ws, reporter, fetch=fetch)


def run_remote(verb, values: dict, given: set[str], ws: Workspace, reporter, *, fetch: str = "all",
               final: bool = True) -> int:
    config = rp.load_config()
    endpoint = config.get("endpoint_id")
    if not endpoint:
        fail("no RunPod endpoint configured", ["Run `yue runpod setup` first."])
    key = rp.job_key()
    ws.root.mkdir(parents=True, exist_ok=True)
    uploads: dict[str, bytes] = {}
    argv = remote_argv(verb, values, given, ws, uploads)
    files, inputs = encode_payload(ws, uploads)
    payload = {"argv": argv, "files": files, "inputs": inputs, "fetch": fetch}
    job = rp.run(endpoint, key, payload)
    bridge = Bridge(reporter, ws)
    bridge.local_log(f"RunPod job {job} submitted to endpoint {endpoint}")
    chunks: dict[str, list] = {}
    exit_code = None
    last_state = None
    last_heartbeat = time.monotonic()
    try:
        while True:
            out = rp.stream(endpoint, job, key)
            state = out.get("status")
            for item in out.get("stream") or []:
                part = item.get("output") or {}
                kind = part.get("k")
                if kind == "event":
                    bridge.forward(part["e"])
                elif kind == "file":
                    slot = chunks.setdefault(part["path"], [None] * part["n"])
                    slot[part["i"]] = base64.b64decode(part["data"])
                elif kind == "done":
                    exit_code = int(part["exit_code"])
                elif kind == "worker":
                    bridge.worker = part
                    bridge.local_log(f"worker: {part.get('gpu')} ({part.get('data_center') or 'unknown DC'})")
            if state != last_state:
                if state == "IN_QUEUE":
                    bridge.local_log("queued: waiting for a worker (a cold start loads the image and the model)")
                elif state == "IN_PROGRESS" and last_state in (None, "IN_QUEUE"):
                    bridge.local_log("a worker picked the job up")
                last_state = state
            if state in FINAL and not out.get("stream"):
                break
            if state not in FINAL and time.monotonic() - last_heartbeat > 20:
                bridge.heartbeat(state)
                last_heartbeat = time.monotonic()
            time.sleep(1)
    except KeyboardInterrupt:
        try:
            rp.cancel(endpoint, job, key)
            bridge.local_log(f"cancelled RunPod job {job}; billing stops when the worker stops")
        except rp.RunPodError as exc:
            bridge.local_log(f"could not cancel RunPod job {job}: {exc}", level="warn")
        raise
    if state != "COMPLETED" and exit_code is None:
        detail = rp.status(endpoint, job, key)
        fail(f"RunPod job {job} ended {state}: {str(detail.get('error') or detail)[:400]}")
    landed = land(ws, chunks)
    if values.get("output") and (ws.root / "4-audio" / "stage.json").is_file():
        meta = ws.stage_meta("decode") or {}
        song = ws.root / Path(meta.get("export", "")).name if meta.get("export") else None
        if song and song.is_file():
            target = Path(values["output"]).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(song, target)
    if final or (exit_code or 0) != 0:
        bridge.finish(exit_code if exit_code is not None else 1, landed, job)
    return exit_code if exit_code is not None else 1


def land(ws: Workspace, chunks: dict[str, list]) -> list[str]:
    """Write received files; a received stage directory REPLACES the local one (old -> .history)."""
    incomplete = [p for p, parts in chunks.items() if any(c is None for c in parts)]
    if incomplete:
        fail(f"RunPod stream lost chunks of: {', '.join(incomplete)}")
    stage_dirs = {DIRS[s] for s in DIRS}
    replaced = set()
    for rel in sorted(chunks):
        top = Path(rel).parts[0]
        if top in stage_dirs and top not in replaced:
            if (ws.root / top).exists():
                ws.archive(ws.root / top)
            replaced.add(top)
    for rel, parts in chunks.items():
        target = ws.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = b"".join(parts)
        if target.suffix == ".json":
            # the worker sends JSON with its workspace as ws://; point it at this one.
            # (json.dumps escaping: a path with quotes/backslashes would need more, ours have none)
            text = data.decode("utf-8")
            data = text.replace('"ws://"', json.dumps(str(ws.root))).replace("ws://", str(ws.root) + "/").encode("utf-8")
        target.write_bytes(data)
    return sorted(chunks)


class Bridge:
    """Forwards remote run-events into the local reporter."""

    def __init__(self, reporter, ws: Workspace):
        self.r, self.ws = reporter, ws
        self.result: dict | None = None
        self.worker: dict | None = None

    def _local(self, value):
        if isinstance(value, str) and "ws://" in value:
            if value == "ws://":
                return str(self.ws.root)
            return value.replace("ws://", str(self.ws.root) + "/")
        if isinstance(value, dict):
            return {k: self._local(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._local(v) for v in value]
        return value

    def forward(self, event: dict) -> None:
        event = self._local(event)
        kind = event.get("type")
        fields = {k: v for k, v in event.items() if k not in ("v", "seq", "ts", "type")}
        if kind == "result":
            self.result = fields
            return
        # A hook splits one command into several remote runs; the local stream stays ONE
        # run: a single run:start, each task declared and closed once.
        if kind == "run:start" and self.r.started:
            return
        if kind == "task:declare":
            fields["tasks"] = [t for t in fields.get("tasks", []) if t["id"] not in self.r.state]
            if not fields["tasks"]:
                return
        elif kind.startswith("task:") and self.r.state.get(fields.get("id")) in ("succeeded", "failed", "cancelled", "skipped") \
                and kind != "task:declare":
            return
        if kind == "run:start":
            data = dict(fields.get("data") or {})
            data["remote"] = "runpod"
            fields["data"] = data
            self.r.started = True
        if self.r.mode == "stream-json":
            self.r.emit(kind, **fields)
            if kind == "task:declare":
                for t in fields.get("tasks", []):
                    self.r.state[t["id"]] = "pending"
            elif kind == "task:start":
                self.r.state[fields["id"]] = "running"
            elif kind in ("task:end", "task:skip"):
                self.r.state[fields["id"]] = fields.get("status", "skipped")
            elif kind == "artifact" and fields.get("role") == "final":
                self.r.final_artifacts.append({k: v for k, v in fields.items() if k not in ("reused", "data")})
            return
        if self.r.mode != "text":
            return
        if kind == "task:start" and fields.get("message"):
            self.r.human(f"▶ {fields['message']}  [runpod]")
        elif kind in ("task:end", "task:skip") and fields.get("message"):
            ok = fields.get("status", "succeeded") == "succeeded" or kind == "task:skip"
            self.r.human(f"{'✓' if ok else '✗'} {fields['message']}")
        elif kind == "task:progress" and fields.get("completed") is not None:
            self.r.state.setdefault(fields["id"], "running")
            self.r.task_t0.setdefault(fields["id"], time.monotonic())
            self.r.progress(fields["id"], fields["completed"], fields.get("total"), fields.get("unit"),
                            fields.get("message"), force=True)
        elif kind == "log" and fields.get("level") in ("warn", "error"):
            self.r.human(("warning: " if fields["level"] == "warn" else "error: ") + fields["message"])

    def local_log(self, message: str, level: str = "info") -> None:
        if self.r.mode == "stream-json":
            if not self.r.started:
                self.r.run_start("remote", data={"remote": "runpod", "workspace": str(self.ws.root)})
            self.r.log(message, level=level, code="yue.remote")
        elif self.r.mode == "text":
            self.r.human(("warning: " if level == "warn" else "☁ ") + message)

    def heartbeat(self, state: str | None) -> None:
        self.local_log(f"RunPod job still {str(state or 'pending').lower().replace('_', ' ')}")

    def finish(self, exit_code: int, landed: list[str], job: str) -> None:
        result = self.result or {}
        data = dict(result.get("data") or {})
        data.update({"remote": "runpod", "job_id": job, "landed_files": len(landed)})
        if self.worker:
            data["worker"] = {k: v for k, v in self.worker.items() if k != "k" and v}
        if self.r.mode in ("stream-json", "json"):
            self.r.result(result.get("status", "succeeded" if exit_code == 0 else "failed"), exit_code,
                          data=data, error=result.get("error"))
        elif self.r.mode == "text":
            if exit_code == 0:
                audio = data.get("audio") or data.get("score") or str(self.ws.root)
                print(audio)
            else:
                err = (result.get("error") or {}).get("message", "remote run failed")
                self.r.human(f"yue: error: {err}")

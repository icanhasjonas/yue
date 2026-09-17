"""Where a run reports what it is doing: `--output-format text | json | stream-json`.

text         human progress on stderr, the finished path on stdout
json         only the final `result` object, once, at the end
stream-json  run-events v1 (docs/EVENT-STREAM.md): JSON Lines on stdout, one
             event per line, flushed per line, and NOTHING else on stdout

The reporter owns the task state machine of that spec: every declared task gets
exactly one terminal event before `result`, `result` is always the last line,
and a failure or a Ctrl-C closes whatever is still open (`failed` for the task
that raised, `cancelled` + `upstream_failed`/`interrupted` for the rest).

Upstream yue2 reports progress through `pipe._status(label, total, unit)`;
`attach()` installs the reporter in its place so upstream's own loops feed the
same `task:progress` events as ours.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any

SCHEMA_VERSION = 1
TERMINAL = {"succeeded", "failed", "cancelled", "skipped"}


class Reporter:
    def __init__(self, mode: str = "text", tool: str = "yue", stdout=None, stderr=None, version: str | None = None):
        self.mode = mode
        self.tool = tool
        self.version = version
        self.err = stderr or sys.stderr
        self.out = stdout or sys.stdout
        if mode == "stream-json" and stdout is None and self.out is sys.__stdout__:
            # Spec 2.1.4: keep a private handle on the real stdout for events and
            # point fd 1 at stderr, so a library `print` can never corrupt the stream.
            self.out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
            os.dup2(2, 1)
        self.seq = 0
        self.run_id = uuid.uuid4().hex[:8]
        self.t0 = time.monotonic()
        self.started = False
        self.state: dict[str, str] = {}  # task id -> pending|running|terminal state
        self.task_t0: dict[str, float] = {}
        self.current: str | None = None
        self.final_artifacts: list[dict] = []
        self._last_progress: dict[str, float] = {}
        self._line_open = False

    # -- envelope -----------------------------------------------------------------
    def emit(self, type_: str, **fields: Any) -> None:
        if self.mode != "stream-json":
            return
        self.seq += 1
        now = time.time()
        event = {"v": SCHEMA_VERSION, "seq": self.seq,
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int(now % 1 * 1000):03d}Z",
                 "type": type_}
        event.update({k: v for k, v in fields.items() if v is not None})
        self.out.write(json.dumps(event, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        self.out.flush()

    def human(self, text: str) -> None:
        if self.mode == "text":
            self._close_line()
            self.err.write(text + "\n")
            self.err.flush()

    # -- run ----------------------------------------------------------------------
    def run_start(self, command: str, message: str | None = None, **data) -> None:
        if self.started:
            return
        self.started = True
        self.emit("run:start", tool=self.tool, version=self.version, command=command, run_id=self.run_id,
                  message=message, data=data or None)

    def declare(self, tasks: list[dict]) -> None:
        fresh = [t for t in tasks if t["id"] not in self.state]
        for t in fresh:
            self.state[t["id"]] = "pending"
        if fresh:
            self.emit("task:declare", tasks=fresh)

    def start(self, task_id: str, message: str | None = None) -> None:
        self.state[task_id] = "running"
        self.task_t0[task_id] = time.monotonic()
        self.current = task_id
        self.emit("task:start", id=task_id, message=message)
        if message:
            self.human(f"▶ {message}")

    def skip(self, task_id: str, reason: str, message: str | None = None, reused: list[dict] | None = None) -> None:
        for record in reused or []:
            self.artifact(task_id, reused=True, **record)
        self.state[task_id] = "skipped"
        self.emit("task:skip", id=task_id, reason=reason, message=message)
        self.human(f"✓ {message or task_id}")

    def end(self, task_id: str, status: str = "succeeded", *, data: dict | None = None, message: str | None = None,
            error: dict | None = None, reason: str | None = None) -> None:
        if self.state.get(task_id) in TERMINAL:
            return
        started = self.task_t0.get(task_id)
        duration = None if started is None else int((time.monotonic() - started) * 1000)
        self.state[task_id] = status
        self.emit("task:end", id=task_id, status=status, duration_ms=duration, reason=reason, error=error,
                  data=data, message=message)
        if message:
            self.human(f"{'✓' if status == 'succeeded' else '✗'} {message}")
        if self.current == task_id:
            self.current = None

    def progress(self, task_id: str | None, completed: float | None, total: float | None, unit: str | None,
                 message: str | None = None, force: bool = False, estimate: bool | None = None) -> None:
        task_id = task_id or self.current
        if task_id is None or self.state.get(task_id) != "running":
            return  # spec: progress belongs to a running task
        now = time.monotonic()
        final = total is not None and completed is not None and completed >= total
        if not (force or final) and now - self._last_progress.get(task_id, 0) < 0.25:
            return
        self._last_progress[task_id] = now
        elapsed = now - self.task_t0.get(task_id, now)
        rate = (completed / elapsed) if completed and elapsed > 0.5 else None
        eta = int((total - completed) / rate * 1000) if rate and total is not None and completed is not None else None
        self.emit("task:progress", id=task_id, completed=completed, total=total, unit=unit, estimate=estimate,
                  rate=None if rate is None else round(rate, 2), eta_ms=eta, elapsed_ms=int(elapsed * 1000),
                  message=message)
        if self.mode == "text" and completed is not None:
            pct = f" ({100 * completed / total:.0f}%)" if total else ""
            of = f"/{total:g}" if total is not None else ""
            self.err.write(f"\r  {message or task_id}: {completed:g}{of} {unit or ''}{pct}   ")
            self.err.flush()
            self._line_open = True

    def artifact(self, task_id: str | None, *, path: str, kind: str, role: str = "intermediate",
                 reused: bool | None = None, mime: str | None = None, data: dict | None = None) -> dict:
        size = os.path.getsize(path) if os.path.isfile(path) else None
        record = {"id": task_id, "path": path, "kind": kind, "mime": mime, "bytes": size, "role": role}
        self.emit("artifact", reused=reused, data=data, **record)
        if role == "final":
            self.final_artifacts.append({k: v for k, v in record.items() if v is not None})
        return record

    def log(self, message: str, level: str = "info", task_id: str | None = None, code: str | None = None,
            stream: str | None = None) -> None:
        self.emit("log", level=level, message=message, id=task_id, code=code, stream=stream)
        if self.mode == "text" and level != "debug":
            self.human(("warning: " if level == "warn" else "") + message)

    # -- terminal -----------------------------------------------------------------
    def result(self, status: str, exit_code: int, *, data: dict | None = None, error: dict | None = None,
               usage: dict | None = None) -> None:
        self._close_line()
        summary: dict[str, int] = {}
        for state in self.state.values():
            summary[state] = summary.get(state, 0) + 1
        payload = {"status": status, "exit_code": exit_code, "duration_ms": int((time.monotonic() - self.t0) * 1000),
                   "error": error, "artifacts": self.final_artifacts or None, "usage": usage,
                   "summary": summary or None, "data": data}
        if self.mode == "stream-json":
            self.emit("result", **payload)
        elif self.mode == "json":
            self.out.write(json.dumps({k: v for k, v in payload.items() if v is not None}, ensure_ascii=False,
                                      indent=2, allow_nan=False, default=str) + "\n")
            self.out.flush()

    def fail(self, code: str, message: str, exit_code: int, hints: list[str] | None = None,
             retryable: bool | None = None, cancelled: bool = False, signal_exit: bool = False) -> None:
        """Close every open task, then emit the failed/cancelled result (spec 2.5.5, 2.8)."""
        error = {"code": code, "message": message}
        if hints:
            error["hints"] = hints
        if retryable is not None:
            error["retryable"] = retryable
        if self.mode == "stream-json":
            if not self.started:
                self.run_start("unknown")
            for task_id, state in list(self.state.items()):
                if state in TERMINAL:
                    continue
                if cancelled:
                    self.end(task_id, "cancelled", reason="interrupted")
                elif state == "running":
                    self.end(task_id, "failed", error=error)
                else:
                    self.end(task_id, "cancelled", reason="upstream_failed")
            self.result("cancelled" if cancelled else "failed", exit_code, error=error)
        elif self.mode == "json":
            self.result("cancelled" if cancelled else "failed", exit_code, error=error)
        else:
            self._close_line()
            self.err.write(f"{self.tool}: {'cancelled' if cancelled else 'error'}: {message}\n")
            for hint in hints or []:
                self.err.write(f"  {hint}\n")

    def _close_line(self):
        if self._line_open:
            self.err.write("\n")
            self._line_open = False

    # -- upstream bridge ------------------------------------------------------------
    def attach(self, pipe) -> None:
        reporter = self

        @contextmanager
        def status(label, *, total=None, unit=None):
            stage = _Stage(reporter, label, total, unit)
            yield stage
            stage.flush()

        pipe._status = status


class _Stage:
    """Duck-types upstream's progress stage (update/advance/set_total/token/finish)."""

    def __init__(self, reporter: Reporter, label: str, total, unit):
        self.r, self.label, self.total, self.unit = reporter, label, total, unit
        self.completed = 0
        self.r.progress(None, None, None, None, message=label, force=True)  # heartbeat: the sub-step began

    def update(self, completed, total=None):
        self.completed = completed
        if total is not None:
            self.total = total
        self.r.progress(None, self.completed, self.total, self.unit, self.label)

    def set_total(self, total):
        self.total = total

    def advance(self, count=1):
        self.update(self.completed + count)

    def token(self, phase, token):
        self.advance()

    def finish(self, status="completed"):
        if status != "completed":
            self.r.log(f"{self.label}: {status}", level="warn", task_id=self.r.current, code=f"yue.{status}")

    def flush(self):
        if self.completed:
            self.r.progress(None, self.completed, self.total, self.unit, self.label, force=True)

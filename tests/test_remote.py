"""The RunPod path without RunPod: handler, argv rebuild, file landing, event forwarding."""
import base64
import json
from pathlib import Path

import pytest

from yuecli import cli
from yuecli.args import UsageError, parse
from yuecli.events import Reporter
from yuecli.remote import client, handler
from yuecli.remote import runpod_api as rp
from yuecli.verbs import VERBS
from yuecli.workspace import Workspace


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def test_handler_runs_the_verb_and_streams_its_events(tmp_path):
    out = list(handler.run_job({"argv": ["status"], "files": {"job.json": b64(json.dumps({"seed": 3}))}}))
    kinds = [o["k"] for o in out]
    assert kinds[-1] == "done" and out[-1]["exit_code"] == 0
    events = [o["e"] for o in out if o["k"] == "event"]
    assert events[-1]["type"] == "result"
    # workspace paths are relativized so the client can map them home
    assert events[-1]["data"]["workspace"] == "ws://"


def test_paths_inside_messages_are_mapped_both_ways(tmp_path):
    # Regression (first remote run): "20.0s -> /tmp/yue-job-x/ws/song.mp3" kept the worker path.
    event = {"message": "Audio: 20.0s -> /tmp/yue-job-x/ws/song.mp3 in 6.8s", "data": {"workspace": "/tmp/yue-job-x/ws"}}
    handler._relativize(event, "/tmp/yue-job-x/ws")
    assert event == {"message": "Audio: 20.0s -> ws://song.mp3 in 6.8s", "data": {"workspace": "ws://"}}
    bridge = client.Bridge(Reporter("json"), Workspace(tmp_path / "ws"))
    local = bridge._local(event)
    assert local["message"] == f"Audio: 20.0s -> {tmp_path / 'ws' / 'song.mp3'} in 6.8s"
    assert local["data"]["workspace"] == str((tmp_path / "ws").resolve())


def test_handler_refuses_path_traversal():
    with pytest.raises(ValueError, match="refusing workspace path"):
        list(handler.run_job({"argv": ["status"], "files": {"../evil": b64("x")}}))
    with pytest.raises(ValueError):
        handler._safe("../evil")
    with pytest.raises(ValueError):
        handler._safe("/etc/passwd")


def test_handler_chunks_large_files(tmp_path, monkeypatch):
    monkeypatch.setattr(handler, "CHUNK", 10)
    ws = tmp_path / "ws"
    (ws / "4-audio").mkdir(parents=True)
    path = ws / "4-audio" / "audio.flac"
    path.write_bytes(b"x" * 25)
    results = list(handler._results(ws, {}, "all"))
    assert results == [("4-audio/audio.flac", path)]


def test_remote_argv_inlines_text_uploads_paths_and_drops_local_only(tmp_path):
    abc = tmp_path / "my.abc"
    abc.write_text("X:1")
    lyrics = tmp_path / "l.txt"
    lyrics.write_text("[verse]\nhi")
    verb = VERBS["render"]
    parsed = parse(verb, ["-w", "x", "--abc", str(abc), "--lyrics", f"@{lyrics}", "--offload-ar", "--remote", "runpod",
                          "-o", "out.flac", "--steps", "8"])
    uploads = {}
    argv = client.remote_argv(verb, parsed.values, parsed.from_switches, Workspace(tmp_path / "ws"), uploads)
    assert argv[0] == "render"
    assert "--remote" not in argv and "-w" not in argv and "--output" not in argv
    assert argv[argv.index("--lyrics") + 1] == "[verse]\nhi"
    assert argv[argv.index("--abc") + 1] == "ws/inputs/abc.abc" and uploads["inputs/abc.abc"] == b"X:1"
    assert "--offload-ar" in argv and argv[argv.index("--steps") + 1] == "8"


def test_collect_files_sends_inputs_not_audio(tmp_path):
    ws = Workspace(tmp_path / "ws")
    for rel in ("job.json", "style.txt", "2-tokens/semantic.npy", "4-audio/audio.flac", ".history/x/y"):
        (ws.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws.root / rel).write_text("1")
    files = client.collect_files(ws, {})
    assert set(files) == {"job.json", "style.txt", "2-tokens/semantic.npy"}


def test_collect_files_refuses_oversized_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "MAX_INPUT_BYTES", 10)
    ws = Workspace(tmp_path / "ws")
    ws.write_text("style.txt", "x" * 100)
    with pytest.raises(UsageError, match="RunPod accepts 10 MB"):
        client.collect_files(ws, {})


def test_land_replaces_received_stage_dirs_and_archives_the_old_ones(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "2-tokens").mkdir(parents=True)
    (ws.root / "2-tokens" / "semantic.npy").write_text("old")
    (ws.root / "3-latents").mkdir()
    (ws.root / "3-latents" / "latent.npy").write_text("keep")
    meta = json.dumps({"export": "ws://song.flac", "edit": {"source": "ws://"}})
    landed = client.land(ws, {"2-tokens/semantic.npy": [b"ne", b"w"], "4-audio/stage.json": [meta.encode()],
                              "song.flac": [b"audio"]})
    assert (ws.root / "2-tokens" / "semantic.npy").read_text() == "new"
    assert (ws.root / "3-latents" / "latent.npy").read_text() == "keep"
    assert list((ws.root / ".history").rglob("semantic.npy"))
    landed_meta = json.loads((ws.root / "4-audio" / "stage.json").read_text())
    assert landed_meta["export"] == str(ws.root / "song.flac")
    assert landed_meta["edit"]["source"] == str(ws.root)
    assert "song.flac" in landed


def test_land_refuses_missing_chunks(tmp_path):
    with pytest.raises(UsageError, match="lost chunks"):
        client.land(Workspace(tmp_path / "ws"), {"song.flac": [b"a", None]})


def test_remote_run_end_to_end_with_a_scripted_runpod(tmp_path, monkeypatch, capsys):
    """generate --remote runpod: events renumbered locally, files landed, result last."""
    monkeypatch.setattr(rp, "load_config", lambda: {"endpoint_id": "ep1"})
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    sent = {}

    def fake_run(endpoint, key, payload):
        sent.update(payload)
        return "job1"

    feed = [
        {"status": "IN_QUEUE", "stream": []},
        {"status": "IN_PROGRESS", "stream": [
            {"output": {"k": "event", "e": {"v": 1, "seq": 1, "type": "run:start", "tool": "yue", "command": "generate",
                                            "data": {"workspace": "ws://"}}}},
            {"output": {"k": "event", "e": {"v": 1, "seq": 2, "type": "task:declare", "tasks": [{"id": "decode"}]}}},
            {"output": {"k": "event", "e": {"v": 1, "seq": 3, "type": "task:start", "id": "decode"}}},
            {"output": {"k": "event", "e": {"v": 1, "seq": 4, "type": "artifact", "id": "decode", "path": "ws://song.flac",
                                            "kind": "audio", "role": "final"}}},
            {"output": {"k": "event", "e": {"v": 1, "seq": 5, "type": "task:end", "id": "decode", "status": "succeeded"}}},
            {"output": {"k": "event", "e": {"v": 1, "seq": 6, "type": "result", "status": "succeeded", "exit_code": 0,
                                            "data": {"audio": "ws://song.flac"}}}},
        ]},
        {"status": "IN_PROGRESS", "stream": [
            {"output": {"k": "file", "path": "song.flac", "i": 0, "n": 1, "data": base64.b64encode(b"FLAC").decode()}},
            {"output": {"k": "done", "exit_code": 0}},
        ]},
        {"status": "COMPLETED", "stream": []},
    ]
    monkeypatch.setattr(rp, "run", fake_run)
    monkeypatch.setattr(rp, "stream", lambda e, j, k: feed.pop(0))
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    ws = tmp_path / "ws"
    code = cli.main(["generate", "-w", str(ws), "--prompt", "x", "--lyrics", "y", "--remote", "runpod",
                     "--output-format", "stream-json"])
    assert code == 0
    assert sent["argv"][:3] == ["generate", "--prompt", "x"]
    assert (ws / "song.flac").read_bytes() == b"FLAC"
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[-1]["type"] == "result" and events[-1]["data"]["job_id"] == "job1"
    assert events[-1]["data"]["audio"] == str(ws / "song.flac")
    assert any(e["type"] == "artifact" and e["path"] == str(ws / "song.flac") for e in events)


def test_remote_without_setup_says_how(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rp, "load_config", lambda: {})
    assert cli.main(["generate", "-w", str(tmp_path / "ws"), "--prompt", "x", "--lyrics", "y", "--remote", "runpod"]) == 1
    assert "yue runpod setup" in capsys.readouterr().err

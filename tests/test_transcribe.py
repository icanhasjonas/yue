"""The SheetSage2 subprocess environment.

Found live on RunPod 2026-09-17: the image sets HF_HUB_ENABLE_HF_TRANSFER=1 for
yue2's own downloads, the SheetSage2 environment has no hf_transfer package, and
huggingface_hub refuses to download at all in that combination -- so the first
remote transcribe failed before SheetSage2 loaded.
"""

import io
from pathlib import Path

from yuecli import transcribe


class FakeProc:
    def __init__(self, cmd, **kw):
        FakeProc.env = kw.get("env")
        self.stdout = io.StringIO('{"abc": "score.abc"}\n')
        self.stderr = io.StringIO("")
        self.returncode = 0

    def wait(self):
        return 0


def test_sheetsage2_never_inherits_hf_transfer(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setattr(transcribe, "ensure_env", lambda log: "uv")
    monkeypatch.setattr(transcribe.subprocess, "Popen", FakeProc)
    result = transcribe.transcribe(tmp_path / "a.mp3", tmp_path / "out", melody_only=False, device=None, dtype=None,
                                   preset=None, max_seconds=None, render_score=False, log=lambda m: None)
    assert result == {"abc": "score.abc"}
    assert FakeProc.env is not None and "HF_HUB_ENABLE_HF_TRANSFER" not in FakeProc.env

"""A workspace is one song's directory. Every artifact is a file you can edit or resume from.

    <ws>/
      job.json        every setting the last run used (the base layer for the next one)
      style.txt       the style prompt           -- editable, read back by render/refine
      lyrics.txt      the lyrics                 -- editable, read back by render/refine
      score.abc       the working score          -- editable, what `render` performs
      1-plan/         SymbolicPlan.save() output + stage.json
      2-tokens/       semantic.npy (25 Hz codec ids) + stage.json
      3-latents/      latent.npy [frames, 64] + stage.json
      4-audio/        audio.flac (48 kHz stereo, 24-bit) + stage.json
      song.<fmt>      the exported result
      .history/       every stage directory a re-run replaced, timestamped

A stage is FRESH when its stage.json `key` equals the key computed from the
current settings and the upstream stage's own key. Keys chain, so changing a
knob invalidates exactly that stage and everything after it -- which is what
`--resume` uses to skip work.

A stage is written into `<name>.partial/` and renamed into place only when it
finished, so an interrupted run leaves no half-stage that looks complete.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

STAGES = ("plan", "tokens", "synth", "decode")
DIRS = {"plan": "1-plan", "tokens": "2-tokens", "synth": "3-latents", "decode": "4-audio"}


def key_of(value) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def text_hash(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    # --- files -----------------------------------------------------------------
    @property
    def job_path(self) -> Path:
        return self.root / "job.json"

    def stage_dir(self, stage: str) -> Path:
        return self.root / DIRS[stage]

    def exists(self) -> bool:
        return self.root.is_dir() and any(self.root.iterdir())

    def job(self) -> dict:
        return read_json(self.job_path) if self.job_path.is_file() else {}

    def save_job(self, settings: dict) -> None:
        write_json(self.job_path, settings)

    def read_text(self, name: str) -> str | None:
        path = self.root / name
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def write_text(self, name: str, text: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_text(text, encoding="utf-8")

    # --- stages ----------------------------------------------------------------
    def stage_meta(self, stage: str) -> dict | None:
        path = self.stage_dir(stage) / "stage.json"
        return read_json(path) if path.is_file() else None

    def stage_key(self, stage: str) -> str | None:
        meta = self.stage_meta(stage)
        return meta["key"] if meta else None

    def begin(self, stage: str) -> Path:
        partial = self.root / (DIRS[stage] + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        return partial

    def commit(self, stage: str, partial: Path, meta: dict) -> Path:
        write_json(partial / "stage.json", meta)
        final = self.stage_dir(stage)
        if final.exists():
            self.archive(final)
        partial.replace(final)
        return final

    def archive(self, path: Path) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.root / ".history" / stamp / path.name
        n = 1
        while target.exists():
            target = self.root / ".history" / f"{stamp}-{n}" / path.name
            n += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        return target

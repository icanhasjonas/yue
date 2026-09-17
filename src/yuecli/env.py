"""Load keys from the REPO's .env files, whatever the working directory (same rule as img/src/env.ts).

Precedence: a real environment variable > <repo>/.claude/.env > <repo>/.env.
Both files are gitignored; .env.example documents every variable.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV_FILES = (REPO / ".claude" / ".env", REPO / ".env")


def parse(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load() -> None:
    for path in ENV_FILES:
        if path.is_file():
            for key, value in parse(path.read_text(encoding="utf-8")).items():
                os.environ.setdefault(key, value)

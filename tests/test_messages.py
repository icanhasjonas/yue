"""Every `--switch` a message names must be one the CLI accepts.

Found 2026-09-17: after `--prompt` became `--style`, `yue runpod setup` still ended
with `ready: yue generate --prompt ...`, and the missing-style error told you to pass
`--prompt` -- advice that aborts with "unknown switch". Scans string literals (not
comments) under src/yuecli, so a rename cannot leave a message behind.
"""

import ast
import re
from pathlib import Path

from yuecli.cli import spec

SRC = Path(__file__).resolve().parents[1] / "src" / "yuecli"
# Vendored upstream code, and flags that belong to other programs we invoke.
SKIP_FILES = {"abc_tools.py"}
FOREIGN = {"--python", "--project", "--frozen", "--extra", "--no-dev", "--version", "--help",
           "--upgrade-package",   # uv
           "--query-gpu",         # nvidia-smi
           "--dtype",             # sheetsage2_worker.py's own argv
           "--allowedTools",      # claude, in a --with example
           "--switch", "--switches", "--kebab-case"}  # prose about switches
# A whole switch: not a prefix being assembled (`--no-`, `--plan-`).
SWITCH = re.compile(r"(?<![\w-])--[a-zA-Z][a-zA-Z0-9-]*(?<!-)(?![\w-])")


def known() -> set[str]:
    out = set(FOREIGN)
    for verb in spec()["verbs"].values():
        for f in verb["fields"]:
            out.add(f["switch"])
            if f["kind"] == "flag":
                out.add("--no-" + f["switch"][2:])
    out.update({"--args", "--output-format"})
    return out


def test_every_switch_named_in_a_string_exists():
    accepted = known()
    bad = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name in SKIP_FILES:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for m in SWITCH.findall(node.value):
                    if m not in accepted:
                        bad.append(f"{path.relative_to(SRC)}:{node.lineno} {m}")
    assert not bad, "messages name switches the CLI refuses:\n" + "\n".join(bad)

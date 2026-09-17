"""The argument grammar, ported from img/vid/snd (`img/src/args/`).

    yue <verb> [--switches]

There are NO positional arguments. The verb is a command name; everything
after it is a named switch, and a stray bare word is a usage error. On the TS
side positions carrying meaning produced a whole class of silent, billed bugs.
Here nothing is billed, but a 20-minute render of the wrong request costs the
same kind of thing, so the rule is kept verbatim:

- an unknown switch, a value switch with no value (end of argv, or the next
  token is itself a `--switch`), a scalar switch given twice, or a bare word
  ABORTS with exit 1 and a did-you-mean;
- `--args @file.json | @- | '{json}'` supplies any subset of the verb's fields;
- precedence, lowest to highest: defaults < workspace job.json < --args < switches;
- a field is spelled `snake_case` in Python and JSON and `--kebab-case` on
  the command line (`tokens_top_p` -> `--tokens-top-p`);
- a TEXT field accepts `@path` to read the value from a file (`--lyrics @l.txt`),
  the same spelling `suno generate --lyrics @l.txt` uses.
"""
from __future__ import annotations

import difflib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NoReturn

TOOL = "yue"


class UsageError(Exception):
    def __init__(self, message: str, hints: list[str] | None = None):
        super().__init__(message)
        self.hints = hints or []


def fail(message: str, hints: list[str] | None = None) -> NoReturn:
    raise UsageError(message, hints)


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # str | text | path | int | float | bool | enum | json | list
    help: str
    choices: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()  # "-m", "--out", ... spelled as typed
    group: str = "Options"
    minimum: float | None = None
    maximum: float | None = None
    stage: str | None = None  # which pipeline stage key this field feeds

    @property
    def switch(self) -> str:
        return "--" + self.name.replace("_", "-")

    def spellings(self) -> list[str]:
        return [self.switch, *self.aliases]


@dataclass(frozen=True)
class Verb:
    name: str
    summary: str
    fields: tuple[Field, ...]
    examples: tuple[str, ...] = ()
    notes: str = ""

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)


@dataclass
class Parsed:
    values: dict[str, Any]
    from_switches: set[str] = field(default_factory=set)
    from_args: set[str] = field(default_factory=set)
    meta: str | None = None  # "help"


def _near(raw: str, known: list[str]) -> list[str]:
    bare = raw.lstrip("-")
    if len(bare) < 2:
        return [k for k in known if k == raw]
    return difflib.get_close_matches(raw, known, n=3, cutoff=0.75)


def _read_text(value: str) -> str:
    if value.startswith("@") and len(value) > 1:
        path = Path(value[1:]).expanduser()
        if not path.is_file():
            fail(f"`{value}` names no readable file")
        return path.read_text(encoding="utf-8")
    return value


def coerce(f: Field, raw: Any, source: str) -> Any:
    """Coerce a raw switch string OR a JSON value into the field's type."""
    where = f"`{f.switch}`" if source == "switch" else f"--args field `{f.name}`"
    try:
        if f.kind in ("str", "path"):
            if not isinstance(raw, str):
                fail(f"{where} needs a string, got {type(raw).__name__}")
            return raw
        if f.kind == "text":
            if not isinstance(raw, str):
                fail(f"{where} needs a string, got {type(raw).__name__}")
            return _read_text(raw)
        if f.kind == "bool":
            if isinstance(raw, bool):
                return raw
            fail(f"{where} is a flag and takes no value")
        if f.kind == "int":
            if isinstance(raw, bool) or (not isinstance(raw, (int, str))):
                fail(f"{where} needs a whole number")
            value = int(raw, 10) if isinstance(raw, str) else raw
        elif f.kind == "float":
            if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                fail(f"{where} needs a number")
            value = float(raw)
        elif f.kind == "enum":
            if raw not in f.choices:
                near = _near(str(raw), list(f.choices))
                fail(f"{where} must be one of: {', '.join(f.choices)} (got `{raw}`)",
                     [f"Did you mean `{near[0]}`?"] if near else [])
            return raw
        elif f.kind == "json":
            return json.loads(raw) if isinstance(raw, str) else raw
        elif f.kind == "list":
            return raw if isinstance(raw, list) else [raw]
        else:  # pragma: no cover - a spec bug, not a user error
            raise AssertionError(f"unknown field kind {f.kind}")
    except ValueError as exc:
        if isinstance(exc, UsageError):
            raise
        fail(f"{where}: {exc}" if f.kind == "json" else f"{where} needs a {'whole number' if f.kind == 'int' else 'number'}, got `{raw}`")
    if f.minimum is not None and value < f.minimum:
        fail(f"{where} must be >= {f.minimum:g}, got {value:g}")
    if f.maximum is not None and value > f.maximum:
        fail(f"{where} must be <= {f.maximum:g}, got {value:g}")
    return value


def load_args(raw: str) -> dict:
    if raw == "@-":
        text = sys.stdin.read()
    elif raw.startswith("@"):
        path = Path(raw[1:]).expanduser()
        if not path.is_file():
            fail(f"`--args {raw}` names no readable file")
        text = path.read_text(encoding="utf-8")
    else:
        text = raw
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        fail(f"`--args` needs valid JSON ({exc.msg} at line {exc.lineno})")
    if not isinstance(data, dict):
        fail("`--args` must be a JSON object")
    return data


def parse(verb: Verb, argv: list[str], base: dict | None = None) -> Parsed:
    """Parse argv for one verb. `base` is the workspace layer (job.json)."""
    index: dict[str, Field] = {}
    negations: dict[str, Field] = {}
    for f in verb.fields:
        for spelling in f.spellings():
            index[spelling] = f
        if f.kind == "bool":
            negations["--no-" + f.switch[2:]] = f
    known = sorted({*index, *negations, "--args", "--help"})

    values: dict[str, Any] = {}
    from_switches: set[str] = set()
    args_blob: dict | None = None
    meta = None
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("--help", "-h"):
            meta = "help"
            i += 1
            continue
        if not token.startswith("-") or token == "-":
            fail(f"unexpected bare word `{token}` for `{TOOL} {verb.name}`",
                 ["There are no positional arguments: every value is named, e.g. `--prompt \"...\"`.",
                  f"Run `{TOOL} {verb.name} --help` for the switches this verb takes."])
        name, inline = (token.split("=", 1) + [None])[:2] if token.startswith("--") and "=" in token else (token, None)
        if name in negations:
            f = negations[name]
            if inline is not None:
                fail(f"`{name}` is a flag and takes no value")
            if f.name in from_switches:
                fail(f"`{f.switch}` given twice")
            values[f.name] = False
            from_switches.add(f.name)
            i += 1
            continue
        if name == "--args":
            value, i = _value(name, inline, argv, i)
            if args_blob is not None:
                fail("`--args` given twice")
            args_blob = load_args(value)
            continue
        f = index.get(name)
        if f is None:
            near = _near(name, known)
            fail(f"unknown switch `{name}` for `{TOOL} {verb.name}`",
                 ([f"Did you mean {' or '.join(f'`{n}`' for n in near)}?"] if near else [])
                 + [f"Run `{TOOL} {verb.name} --help` for the switches this verb takes."])
        if f.kind == "bool":
            if inline is not None:
                fail(f"`{name}` is a flag and takes no value")
            if f.name in from_switches:
                fail(f"`{f.switch}` given twice")
            values[f.name] = True
            from_switches.add(f.name)
            i += 1
            continue
        raw, i = _value(name, inline, argv, i)
        if f.kind == "list":
            # Repeating a list switch collects; a switch list REPLACES one from --args.
            values[f.name] = [*values.get(f.name, []), raw] if f.name in from_switches else [raw]
        else:
            if f.name in from_switches:
                fail(f"`{f.switch}` given twice", ["A scalar switch takes one value; drop one of them."])
            values[f.name] = coerce(f, raw, "switch")
        from_switches.add(f.name)

    merged: dict[str, Any] = {}
    from_args: set[str] = set()
    fields = {f.name: f for f in verb.fields}
    for key, value in (base or {}).items():
        if key in fields and value is not None:
            merged[key] = value
    if args_blob is not None:
        for key, value in args_blob.items():
            if key not in fields:
                near = difflib.get_close_matches(key, list(fields), n=1, cutoff=0.6)
                fail(f"--args field `{key}` is not a field of `{TOOL} {verb.name}`",
                     [f"Did you mean `{near[0]}`?"] if near else [])
            merged[key] = coerce(fields[key], value, "args")
            from_args.add(key)
    merged.update(values)
    return Parsed(merged, from_switches, from_args, meta)


def _value(name: str, inline: str | None, argv: list[str], i: int) -> tuple[str, int]:
    if inline is not None:
        if inline == "":
            fail(f"switch `{name}` needs a value")
        return inline, i + 1
    if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
        fail(f"switch `{name}` needs a value")
    return argv[i + 1], i + 2


def render_help(verb: Verb) -> str:
    lines = [f"Usage: {TOOL} {verb.name} [switches]", "", verb.summary, ""]
    if verb.notes:
        lines += [verb.notes.rstrip(), ""]
    groups: dict[str, list[Field]] = {}
    for f in verb.fields:
        groups.setdefault(f.group, []).append(f)
    for group, fields in groups.items():
        lines.append(f"{group}:")
        rows = []
        for f in fields:
            names = ", ".join(f.spellings())
            if f.kind == "bool":
                names += f", --no-{f.switch[2:]}"
            elif f.kind in ("int", "float"):
                names += " <n>"
            elif f.kind == "text":
                names += " <text|@file>"
            elif f.kind == "enum":
                names += " <" + "|".join(f.choices) + ">"
            else:
                names += " <value>"
            desc = f.help
            if f.minimum is not None or f.maximum is not None:
                desc += f" [{'' if f.minimum is None else f'{f.minimum:g}'}..{'' if f.maximum is None else f'{f.maximum:g}'}]"
            rows.append((names, desc))
        width = min(max(len(n) for n, _ in rows) + 2, 44)
        for names, desc in rows:
            if len(names) + 2 > width:
                lines.append(f"  {names}")
                lines.append(f"  {'':{width}}{desc}")
            else:
                lines.append(f"  {names:{width}}{desc}")
        lines.append("")
    lines += [
        "Input:",
        "  --args @file.json   read switches from a JSON object (snake_case keys)",
        "  --args @-           ...from stdin",
        "  --args '{\"k\":1}'    ...inline",
        "",
        "  Precedence: defaults < workspace job.json < --args < switches.",
        "  There are no positional arguments: every value is named.",
    ]
    if verb.examples:
        lines += ["", "Examples:", *[f"  {e}" for e in verb.examples]]
    return "\n".join(lines) + "\n"


def report_usage_error(err: UsageError, write: Callable[[str], Any] | None = None) -> int:
    # Resolve stderr at CALL time: a default bound at import keeps writing to
    # whatever stderr was when the module loaded, not the one in effect now.
    write = write or sys.stderr.write
    write(f"{TOOL}: error: {err}\n")
    for hint in err.hints:
        write(f"  {hint}\n")
    write(f"\nRun `{TOOL} help` for usage.\n")
    return 1

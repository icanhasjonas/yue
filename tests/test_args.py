"""The grammar rules ported from img/vid/snd, each one a test."""
import json

import pytest

from yuecli.args import UsageError, parse, render_help
from yuecli.cli import main
from yuecli.verbs import VERBS

GEN = VERBS["generate"]


def test_bare_word_is_refused():
    with pytest.raises(UsageError, match="bare word `song.txt`"):
        parse(GEN, ["--style", "x", "song.txt"])


def test_unknown_switch_suggests_the_nearest():
    with pytest.raises(UsageError) as err:
        parse(GEN, ["--stlye", "x"])
    assert "unknown switch `--stlye`" in str(err.value)
    assert any("`--style`" in h for h in err.value.hints)


def test_value_switch_without_value_aborts_at_end_and_before_a_switch():
    with pytest.raises(UsageError, match="`--seed` needs a value"):
        parse(GEN, ["--seed"])
    with pytest.raises(UsageError, match="`--style` needs a value"):
        parse(GEN, ["--style", "--lyrics", "x"])


def test_scalar_twice_is_refused():
    with pytest.raises(UsageError, match="given twice"):
        parse(GEN, ["--seed", "1", "--seed", "2"])


def test_negative_number_is_a_value_not_a_switch():
    # only `--` starts a switch, so `-1` is a value (validation then rejects it)
    with pytest.raises(UsageError, match=">= 0"):
        parse(GEN, ["--seed", "-1"])


def test_types_ranges_and_enums():
    assert parse(GEN, ["--cfg", "1.5"]).values["cfg"] == 1.5
    with pytest.raises(UsageError, match="needs a whole number"):
        parse(GEN, ["--steps", "3.5"])
    with pytest.raises(UsageError, match="<= 20"):
        parse(GEN, ["--cfg", "21"])
    with pytest.raises(UsageError, match="one of: full, melody, off"):
        parse(GEN, ["--cot", "fulll"])


def test_bool_flags_and_negation():
    assert parse(GEN, ["--offload-ar"]).values["offload_ar"] is True
    assert parse(GEN, ["--no-verify-hashes"]).values["verify_hashes"] is False
    with pytest.raises(UsageError, match="takes no value"):
        parse(GEN, ["--offload-ar=yes"])


def test_short_letters_and_inline_values():
    v = parse(GEN, ["--workspace", "ws", "-s", "dark folk", "-f", "tokens", "--cfg=2", "-o", "x.flac", "-m", "m"]).values
    assert v == {"workspace": "ws", "style": "dark folk", "from": "tokens", "cfg": 2.0, "output": "x.flac", "model": "m"}


def test_vocabulary_matches_img_one_long_name_and_shared_letters():
    """Same rules as img tests/args/switch-vocabulary.test.ts (2026-09-17)."""
    from yuecli.args import SHORT_SWITCHES
    for verb in VERBS.values():
        for f in verb.fields:
            longs = [s for s in f.spellings() if s.startswith("--")]
            assert longs == [f.switch], (verb.name, f.name, longs)
            for short in (s for s in f.spellings() if not s.startswith("--")):
                assert f.name in SHORT_SWITCHES[short[1:]], (verb.name, f.name, short)
    # the letters snd reserves for other concepts must not reach yue fields of another meaning
    for retired in (["-w", "ws"], ["-q"], ["-y"], ["--prompt", "x"], ["--cfg-scale", "2"], ["--out", "x"]):
        with pytest.raises(UsageError):
            parse(GEN, retired)


def test_every_help_example_parses(tmp_path, monkeypatch):
    """A help example is documentation; it must survive the real parser (img: tests/docs-commands.test.ts)."""
    import shlex
    monkeypatch.chdir(tmp_path)
    for verb in VERBS.values():
        for example in verb.examples:
            words = shlex.split(example.split("  #", 1)[0])
            for word in words:  # an @file reference only has to exist, not to be a real song
                if word.startswith("@") and len(word) > 1:
                    (tmp_path / word[1:]).write_text("x")
            assert words[0] == "yue", example
            name = " ".join(words[1:3]) if words[1] == "runpod" else words[1]
            rest = words[3:] if words[1] == "runpod" else words[2:]
            parse(VERBS[name], rest)


def test_text_at_file(tmp_path):
    lyrics = tmp_path / "l.txt"
    lyrics.write_text("[verse]\nhello\n")
    assert parse(GEN, ["--lyrics", f"@{lyrics}"]).values["lyrics"] == "[verse]\nhello\n"
    with pytest.raises(UsageError, match="no readable file"):
        parse(GEN, ["--lyrics", "@/nope/missing.txt"])


def test_args_precedence_defaults_job_args_switches(tmp_path):
    blob = tmp_path / "a.json"
    blob.write_text(json.dumps({"seed": 5, "steps": 10, "cot": "melody"}))
    base = {"seed": 1, "steps": 99, "solver": "euler"}
    v = parse(GEN, ["--args", f"@{blob}", "--steps", "7"], base=base).values
    assert v["solver"] == "euler"  # job layer
    assert v["seed"] == 5  # --args beats job
    assert v["steps"] == 7  # switch beats --args
    assert v["cot"] == "melody"


def test_args_rejects_unknown_fields_and_bad_types():
    with pytest.raises(UsageError, match="`stepz` is not a field"):
        parse(GEN, ["--args", '{"stepz": 1}'])
    with pytest.raises(UsageError, match="whole number"):
        parse(GEN, ["--args", '{"steps": "many"}'])
    with pytest.raises(UsageError, match="valid JSON"):
        parse(GEN, ["--args", "{nope"])


def test_every_verb_renders_help_and_names_every_field():
    for verb in VERBS.values():
        text = render_help(verb)
        for f in verb.fields:
            assert f.switch in text, (verb.name, f.name)


def test_no_duplicate_spellings_within_a_verb():
    for verb in VERBS.values():
        seen = {}
        for f in verb.fields:
            for spelling in f.spellings():
                assert spelling not in seen, f"{verb.name}: {spelling} on {seen.get(spelling)} and {f.name}"
                seen[spelling] = f.name


def test_cli_exit_codes(capsys):
    assert main(["generate", "--promt", "x"]) == 1
    assert main(["nope"]) == 1
    assert main(["help"]) == 0
    assert main(["generate", "--help"]) == 0
    assert "Usage: yue generate" in capsys.readouterr().out


def test_usage_error_in_stream_json_is_a_stream(capsys):
    # run-events 2.8: once stream mode is recognised, even a usage error is run:start + result
    assert main(["status", "--output-format", "stream-json", "--wat"]) == 1
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e["type"] for e in events] == ["run:start", "result"]
    assert events[-1]["error"]["code"] == "usage" and events[-1]["exit_code"] == 1

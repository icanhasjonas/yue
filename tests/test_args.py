"""The grammar rules ported from img/vid/snd, each one a test."""
import json

import pytest

from yuecli.args import UsageError, parse, render_help
from yuecli.cli import main
from yuecli.verbs import VERBS

GEN = VERBS["generate"]


def test_bare_word_is_refused():
    with pytest.raises(UsageError, match="bare word `song.txt`"):
        parse(GEN, ["--prompt", "x", "song.txt"])


def test_unknown_switch_suggests_the_nearest():
    with pytest.raises(UsageError) as err:
        parse(GEN, ["--promt", "x"])
    assert "unknown switch `--promt`" in str(err.value)
    assert any("`--prompt`" in h for h in err.value.hints)


def test_value_switch_without_value_aborts_at_end_and_before_a_switch():
    with pytest.raises(UsageError, match="`--seed` needs a value"):
        parse(GEN, ["--seed"])
    with pytest.raises(UsageError, match="`--prompt` needs a value"):
        parse(GEN, ["--prompt", "--lyrics", "x"])


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


def test_aliases_and_inline_values():
    v = parse(GEN, ["-w", "ws", "--style=dark folk", "--from", "tokens", "--cfg-scale", "2"]).values
    assert v == {"workspace": "ws", "prompt": "dark folk", "from_stage": "tokens", "cfg": 2.0}


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

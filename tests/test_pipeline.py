"""Workspace, keys, --resume, single stages, edits and hooks -- against a fake engine.

The fake produces real artifact FILES (a real SymbolicPlan, .npy arrays, a FLAC),
so everything the CLI does with them is exercised; only the model math is stubbed.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from yuecli import cli, engine as engine_mod

FIX = Path(__file__).parent / "fixtures"
SMOKE = (FIX / "smoke.abc").read_text()


class FakeEngine:
    calls: list = []

    def __init__(self, rt, on_load=None):
        self.rt = rt
        self.device = "cpu"

    def plan(self, *, style, lyrics, cot, seed, abc, sampling):
        from yue2 import SymbolicPlan
        from yue2.protocol import SongRequest
        FakeEngine.calls.append(("plan", seed, abc is not None))
        request = SongRequest(style=style, lyrics=lyrics, cot=cot, seed=seed, abc=abc)
        text = abc if abc is not None else (None if cot == "off" else SMOKE)
        ids = [] if text is None else list(range(1, 50))
        return SymbolicPlan(request, text, ids, [1, 2, 3, *ids]), 0.01

    def tokens(self, plan, *, seed, cfg, sampling, keep=None, tail=None, exact=None, extend_frames=None):
        FakeEngine.calls.append(("tokens", seed, len(keep or []), len(tail or []), exact, extend_frames))
        n = exact if exact is not None else (2220 if not keep else (extend_frames or 100))
        new = [(seed + i) % 32768 for i in range(n)]
        return list(keep or []) + new + list(tail or []), {"sampled": n}, False

    def synth(self, plan, tokens, *, seed, steps, solver, strength, init, anchor, attention, query_chunk):
        FakeEngine.calls.append(("synth", seed, steps, solver, strength, None if anchor is None else int(anchor.sum())))
        return np.full((len(tokens), 64), seed % 7, dtype=np.float32), {"seconds": 0}

    def decode(self, latents, *, vae, revision, mode, tile_frames, halo_frames):
        FakeEngine.calls.append(("decode", vae, mode, tile_frames))
        return np.zeros((len(latents) * 4, 2), dtype=np.float32), {"seconds": 0}

    def encode(self, audio, *, vae, revision):
        return np.zeros((len(audio) // 1920, 64), dtype=np.float32)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    FakeEngine.calls = []
    monkeypatch.setattr(engine_mod, "Engine", FakeEngine)
    return FakeEngine


def run(*argv):
    return cli.main(list(argv))


def stages_called():
    return [c[0] for c in FakeEngine.calls]


def gen(ws, *extra):
    return run("generate", "-w", str(ws), "--prompt", "synthwave", "--lyrics", "[verse]\nhi", "--seed", "42", *extra)


def test_generate_writes_every_stage_and_the_song(tmp_path):
    ws = tmp_path / "ws"
    assert gen(ws) == 0
    assert stages_called() == ["plan", "tokens", "synth", "decode"]
    for d in ("1-plan/plan.json", "2-tokens/semantic.npy", "3-latents/latent.npy", "4-audio/audio.flac", "song.flac",
              "score.abc", "style.txt", "lyrics.txt", "job.json"):
        assert (ws / d).is_file(), d
    job = json.loads((ws / "job.json").read_text())
    assert job["seed"] == 42 and job["score_mode"] == "generated"


def test_generate_refuses_a_used_workspace_without_resume(tmp_path, capsys):
    ws = tmp_path / "ws"
    gen(ws)
    assert gen(ws) == 1
    assert "already holds a song" in capsys.readouterr().err


def test_resume_with_nothing_changed_runs_nothing(tmp_path):
    ws = tmp_path / "ws"
    gen(ws)
    FakeEngine.calls = []
    assert run("generate", "-w", str(ws), "--resume") == 0
    assert stages_called() == []


def test_changing_a_tokens_knob_redoes_tokens_and_everything_after(tmp_path):
    ws = tmp_path / "ws"
    gen(ws)
    FakeEngine.calls = []
    assert run("generate", "-w", str(ws), "--resume", "--tokens-temperature", "0.8") == 0
    assert stages_called() == ["tokens", "synth", "decode"]
    assert (ws / ".history").is_dir()  # the replaced stages were archived, not deleted


def test_single_stage_decode_with_another_vae_only_decodes(tmp_path):
    ws = tmp_path / "ws"
    gen(ws)
    FakeEngine.calls = []
    assert run("decode", "-w", str(ws), "--vae", "legacy", "--decode", "full", "-o", str(tmp_path / "x.wav"),
               "--format", "wav") == 0
    assert FakeEngine.calls == [("decode", "legacy", "full", 1024)]
    assert (tmp_path / "x.wav").is_file()
    # the decode knob persisted without erasing the tokens settings
    job = json.loads((ws / "job.json").read_text())
    assert job["vae"] == "legacy" and job["seed"] == 42


def test_stage_needs_its_upstream(tmp_path, capsys):
    ws = tmp_path / "ws"
    run("plan", "-w", str(ws), "--prompt", "x", "--lyrics", "y", "--seed", "1")
    assert run("synth", "-w", str(ws)) == 1
    assert "needs `tokens` first" in capsys.readouterr().err


def test_stage_seeds_override_the_song_seed(tmp_path):
    ws = tmp_path / "ws"
    gen(ws, "--synth-seed", "7")
    seeds = {c[0]: c[1] for c in FakeEngine.calls}
    assert seeds == {"plan": 42, "tokens": 42, "synth": 7, "decode": "standard"}


def test_budget_12_picks_small_decode_tiles(tmp_path):
    gen(tmp_path / "ws", "--budget", "12")
    assert FakeEngine.calls[-1] == ("decode", "standard", "tiled", 512)


def test_duration_forces_an_exact_token_count(tmp_path):
    gen(tmp_path / "ws", "--duration", "10")
    tokens = [c for c in FakeEngine.calls if c[0] == "tokens"][0]
    assert tokens[4] == 250


def test_a_cap_shorter_than_the_score_warns_before_the_tokens_run(tmp_path, capsys):
    # Regression (live run 2026-09-17): --max-duration 45 on a 96 s score only
    # warned AFTER 133 s of token sampling, as a generic "truncated".
    gen(tmp_path / "ws", "--max-duration", "45", "--output-format", "stream-json")
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    warn = [i for i, e in enumerate(events) if e.get("code") == "yue.score_exceeds_duration"]
    tokens_start = next(i for i, e in enumerate(events) if e["type"] == "task:start" and e["id"] == "tokens")
    assert warn and warn[0] < tokens_start
    assert "89s" in events[warn[0]]["message"] and "45s" in events[warn[0]]["message"]


def test_status_shows_the_effective_cot_when_it_was_never_set(tmp_path, capsys):
    run("generate", "-w", str(tmp_path / "ws"), "--prompt", "x", "--lyrics", "y", "--seed", "1")
    capsys.readouterr()
    run("status", "-w", str(tmp_path / "ws"))
    assert "cot full" in capsys.readouterr().out


def test_render_performs_an_edited_score_as_provided(tmp_path):
    ws = tmp_path / "ws"
    gen(ws)
    (ws / "score.abc").write_text(SMOKE.replace("Q:1/4=100", "Q:1/4=120"))
    FakeEngine.calls = []
    assert run("render", "-w", str(ws)) == 0
    assert stages_called() == ["plan", "tokens", "synth", "decode"]
    assert FakeEngine.calls[0] == ("plan", 42, True)
    assert json.loads((ws / "job.json").read_text())["score_mode"] == "provided"
    # and a resume afterwards keeps performing the provided score instead of re-planning
    FakeEngine.calls = []
    assert run("generate", "-w", str(ws), "--resume") == 0
    assert stages_called() == []


def test_render_bars_keeps_head_and_tail_and_anchors(tmp_path):
    ws = tmp_path / "ws"
    gen(ws)
    FakeEngine.calls = []
    assert run("render", "-w", str(ws), "--bars", "13-28", "--anchor-margin", "10") == 0
    tokens = [c for c in FakeEngine.calls if c[0] == "tokens"][0]
    # bars 13-28 at 100 BPM 4/4 = frames [720, 1680): keep 720, sample exactly 960, tail 540
    assert tokens[2:5] == (720, 540, 960)
    synth = [c for c in FakeEngine.calls if c[0] == "synth"][0]
    assert synth[5] == (720 - 10) + (2220 - 1680 - 10)
    meta = json.loads((ws / "2-tokens" / "stage.json").read_text())
    assert meta["edit"]["bars"] == "13-28" and meta["frames"] == 2220


def test_render_extend_keeps_everything_and_forbids_ending_early(tmp_path):
    ws = tmp_path / "ws"
    gen(ws)
    FakeEngine.calls = []
    assert run("render", "-w", str(ws), "--extend", "20") == 0
    tokens = [c for c in FakeEngine.calls if c[0] == "tokens"][0]
    assert tokens[2] == 2220 and tokens[5] == 500


def test_hook_edit_replans_from_the_refined_score(tmp_path):
    ws = tmp_path / "ws"
    hook = "python3 -c \"import os,pathlib;p=pathlib.Path(os.environ['YUE_ABC']);p.write_text(p.read_text().replace('Q:1/4=100','Q:1/4=90'))\""
    assert gen(ws, "--abc-hook", hook) == 0
    assert stages_called() == ["plan", "plan", "tokens", "synth", "decode"]
    assert FakeEngine.calls[1] == ("plan", 42, True)
    assert "Q:1/4=90" in (ws / "1-plan" / "score.abc").read_text()
    assert list((ws / "refine").glob("*.json"))


def test_hook_that_breaks_the_score_is_refused_and_rolled_back(tmp_path, capsys):
    ws = tmp_path / "ws"
    hook = "python3 -c \"import os,pathlib;pathlib.Path(os.environ['YUE_ABC']).write_text('garbage')\""
    assert gen(ws, "--abc-hook", hook) == 1
    assert (ws / "score.abc").read_text() == SMOKE
    assert (ws / "score.rejected.abc").read_text() == "garbage"
    assert "native dialect rejects" in capsys.readouterr().err


def test_remix_from_a_workspace_keeps_the_score_and_changes_the_style(tmp_path):
    src = tmp_path / "src"
    gen(src)
    FakeEngine.calls = []
    dst = tmp_path / "dst"
    assert run("remix", "-i", str(src), "-w", str(dst), "--prompt", "jazz trio") == 0
    assert stages_called() == ["plan", "tokens", "synth", "decode"]
    assert FakeEngine.calls[0][2] is True  # provided score, not a new plan
    assert (dst / "style.txt").read_text() == "jazz trio"
    assert (src / "style.txt").read_text() == "synthwave"


def test_remix_from_synth_reuses_the_performance(tmp_path):
    src = tmp_path / "src"
    gen(src)
    FakeEngine.calls = []
    assert run("remix", "-i", str(src), "-w", str(tmp_path / "d"), "--from", "synth", "--synth-seed", "3",
               "--strength", "0.5") == 0
    assert stages_called() == ["synth", "decode"]
    assert FakeEngine.calls[0][4] == 0.5


def test_stream_json_events_are_ordered_and_result_is_last(tmp_path, capsys):
    ws = tmp_path / "ws"
    gen(ws)
    capsys.readouterr()
    assert run("generate", "-w", str(ws), "--resume", "--vae", "legacy", "--output-format", "stream-json") == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    types = [e["type"] for e in lines]
    assert types[0] == "run:start" and types[1] == "task:declare"
    assert types[-1] == "result" and lines[-1]["status"] == "succeeded"
    assert [e["id"] for e in lines if e["type"] == "task:skip"] == ["plan", "tokens", "synth"]
    assert [e["id"] for e in lines if e["type"] == "task:end"] == ["decode"]
    assert [e["seq"] for e in lines] == list(range(1, len(lines) + 1))
    assert all(e["v"] == 1 and "ts" in e for e in lines)


def test_json_output_is_one_object(tmp_path, capsys):
    ws = tmp_path / "ws"
    gen(ws)
    capsys.readouterr()
    assert run("status", "-w", str(ws), "--output-format", "json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "succeeded" and data["data"]["stages"]["decode"]["state"] == "ok"


def test_failure_closes_open_tasks_before_the_result(tmp_path, capsys, monkeypatch):
    ws = tmp_path / "ws"

    def boom(self, *a, **k):
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(FakeEngine, "synth", boom)
    assert gen(ws, "--output-format", "stream-json") == 1
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    ends = {e["id"]: e for e in events if e["type"] == "task:end"}
    assert ends["synth"]["status"] == "failed" and ends["synth"]["error"]["code"] == "resource"
    assert ends["decode"]["status"] == "cancelled" and ends["decode"]["reason"] == "upstream_failed"
    assert events[-1]["type"] == "result" and events[-1]["status"] == "failed"


def test_interrupt_is_cancelled_130(tmp_path, capsys, monkeypatch):
    def interrupt(self, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(FakeEngine, "tokens", interrupt)
    assert gen(tmp_path / "ws", "--output-format", "stream-json") == 130
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    status = {e["id"]: e["status"] for e in events if e["type"] == "task:end"}
    assert status == {"plan": "succeeded", "tokens": "cancelled", "synth": "cancelled", "decode": "cancelled"}
    assert events[-1]["status"] == "cancelled" and events[-1]["exit_code"] == 130


def test_every_declared_task_gets_exactly_one_terminal_event(tmp_path, capsys):
    ws = tmp_path / "ws"
    gen(ws, "--output-format", "stream-json")
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    declared = [t["id"] for e in events if e["type"] == "task:declare" for t in e["tasks"]]
    terminal = [e["id"] for e in events if e["type"] in ("task:end", "task:skip")]
    assert sorted(declared) == sorted(terminal)
    finals = events[-1]["artifacts"]
    assert finals and finals[0]["role"] == "final" and finals[0]["path"].endswith("song.flac")

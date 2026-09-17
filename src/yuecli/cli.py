"""yue -- YuE2 songs from the command line. `yue help` for the verbs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import abc_tools, abcmap
from .args import TOOL, UsageError, fail, parse, render_help, report_usage_error
from .events import Reporter
from .verbs import NOT_PERSISTED, VERBS
from .workspace import DIRS, STAGES, Workspace, key_of, read_json, text_hash, write_json

DEFAULTS = {
    "format": "flac", "output_format": "text", "cot": "full", "solver": "midpoint", "steps": 32,
    "strength": 1.0, "attention": "sdpa", "decode": "tiled", "vae": "standard", "backend": "torch",
    "quantization": "none", "budget": 24.0, "device": "auto", "model": "m-a-p/YuE2-3B",
    "anchor_margin": 12, "validate": True, "verify_hashes": True, "keep_voice": "both", "voices": "both",
}
FPS = 25
STAGE_TEXT = {"plan": "Score (ABC plan)", "tokens": "Performance (semantic tokens)",
              "synth": "Acoustic latents (flow matching)", "decode": "Audio (VAE decode)"}


class Settings(dict):
    def __getattr__(self, name):
        value = self.get(name)
        return DEFAULTS.get(name) if value is None else value


# =============================================================================== entry
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("help", "-h", "--help"):
        if len(argv) > 1 and argv[1] in VERBS:
            sys.stdout.write(render_help(VERBS[argv[1]]))
        else:
            sys.stdout.write(overview())
        return 0
    if argv[0] in ("--version", "-V", "version"):
        print(f"{TOOL} {importlib.metadata.version('yue')} (yue2-infer {_upstream_version()})")
        return 0
    name = argv[0]
    if name == "runpod":  # a namespace: `yue runpod setup|status|teardown`
        sub = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else None
        if sub is None or f"runpod {sub}" not in VERBS:
            return report_usage_error(UsageError("`yue runpod` needs a sub-command: setup, status or teardown"))
        name = f"runpod {sub}"
        argv = [name, *argv[2:]]
    verb = VERBS.get(name)
    if verb is None:
        import difflib
        near = difflib.get_close_matches(name, list(VERBS), n=2, cutoff=0.5)
        return report_usage_error(UsageError(f"unknown command `{name}`",
                                             [f"Did you mean {' or '.join(f'`{n}`' for n in near)}?"] if near else []))
    debug = "--debug" in argv or "-d" in argv
    # Spec 2.8: once stream mode is recognised, even a usage error is a stream.
    mode = next((argv[i + 1] for i, t in enumerate(argv[:-1]) if t == "--output-format"
                 and argv[i + 1] in ("json", "stream-json")), "text")
    reporter = Reporter(mode, version=_version())
    _install_sigterm()
    try:
        first = parse(verb, argv[1:])
        if first.meta == "help":
            sys.stdout.write(render_help(verb))
            return 0
        base = {}
        if first.values.get("workspace") and verb.name not in ("transcribe", "remix"):
            base = Workspace(Path(first.values["workspace"])).job()
        parsed = parse(verb, argv[1:], base=base)
        # job.json keys this verb does not declare still describe the song
        # (e.g. `yue decode` computing keys that chain from tokens settings).
        settings = Settings({**{k: v for k, v in base.items() if k not in parsed.values}, **parsed.values})
        settings["_switches"] = parsed.from_switches | parsed.from_args
        if settings.get("remote") == "runpod":
            from .remote.client import run_remote
            ws = _workspace(settings, settings.get("prompt"), "song")
            return run_remote(verb, settings, settings["_switches"], ws, reporter,
                              fetch=settings.get("remote_fetch") or "all")
        handler = HANDLERS[verb.name]
        return handler(settings, reporter)
    except UsageError as err:
        if reporter.mode == "text":
            return report_usage_error(err)
        reporter.run_start(name)
        reporter.fail("usage", str(err), 1, err.hints)
        return 1
    except KeyboardInterrupt as exc:
        code = 143 if getattr(exc, "sigterm", False) else 130
        reporter.fail("interrupted", "interrupted by signal", code, cancelled=True)
        return code
    except Exception as exc:  # noqa: BLE001 - the CLI boundary
        if debug:
            traceback.print_exc()
        reporter.fail(_error_code(exc), str(exc), 1, [] if debug else ["Re-run with --debug for the traceback."])
        return 1


def overview() -> str:
    width = max(len(v) for v in VERBS) + 2
    rows = "\n".join(f"  {name:{width}}{verb.summary.split('. ')[0].rstrip('.')}" for name, verb in VERBS.items())
    return f"""Usage: {TOOL} <command> [switches]

YuE2 song generation: style + lyrics -> score (ABC) -> semantic tokens -> acoustic latents -> audio.
Every stage writes its artifacts into a workspace (-w); every stage can be run, resumed or redone alone.

Commands:
{rows}

  {TOOL} <command> --help    every switch of one command
  {TOOL} --version

Pipeline:  plan -> tokens -> synth -> decode      (yue generate runs all four)
Weights:   CC BY-NC 4.0 -- generated songs are for non-commercial use.
"""


# =============================================================================== pipeline
class Pipeline:
    """One invocation's view of a workspace: settings, keys, stages."""

    def __init__(self, s: Settings, ws: Workspace, reporter: Reporter):
        from .engine import Engine, Runtime
        self.s, self.ws, self.r = s, ws, reporter
        rt = Runtime(model=s.model, revision=s.get("revision"), device=s.device, backend=s.backend,
                     quantization=s.quantization, budget=float(s.budget), offload_ar=bool(s.get("offload_ar")),
                     offline=bool(s.get("offline")), verify_hashes=bool(s.verify_hashes),
                     quiet=bool(s.get("quiet")) and reporter.mode == "text")
        self.engine = Engine(rt, on_load=(reporter.attach if reporter.mode != "text" else None))
        self.edit: dict | None = None  # render --bars / --extend state, loaded before anything is overwritten
        self.init_latents: np.ndarray | None = None
        self.init_hash: str | None = None
        self.score_mode = s.get("score_mode") or ws.job().get("score_mode") or "generated"

    # ---- settings ---------------------------------------------------------------
    def seed(self, stage: str) -> int:
        return int(self.s.get(f"{stage}_seed") if self.s.get(f"{stage}_seed") is not None else self.s["seed"])

    def sampling(self, stage: str) -> dict:
        keys = ("temperature", "top_p", "top_k", "repetition_penalty", "penalty_window", "min_tokens", "max_tokens")
        return {k: self.s[f"{stage}_{k}"] for k in keys if self.s.get(f"{stage}_{k}") is not None}

    def style(self) -> str:
        text = self.ws.read_text("style.txt")
        if text is None:
            fail("no style: pass --prompt (it is saved to <workspace>/style.txt)")
        return text

    def lyrics(self) -> str:
        text = self.ws.read_text("lyrics.txt")
        if text is None:
            fail("no lyrics: pass --lyrics (it is saved to <workspace>/lyrics.txt)")
        return text

    def provided_abc(self) -> str | None:
        if self.score_mode != "provided":
            return None
        text = self.ws.read_text("score.abc")
        if text is None:
            fail("score mode is `provided` but <workspace>/score.abc is missing")
        return text

    def persist(self) -> None:
        # Merge: a single-stage verb only declares its own knobs, and must not
        # erase the others (`yue decode` would otherwise drop every sampling knob).
        current = {k: v for k, v in self.s.items() if not k.startswith("_") and k not in NOT_PERSISTED and v is not None}
        job = {**self.ws.job(), **current, "score_mode": self.score_mode}
        self.ws.save_job(job)

    # ---- keys ---------------------------------------------------------------------
    def key(self, stage: str, upstream: str | None) -> tuple[str, dict]:
        s = self.s
        if stage == "plan":
            inputs = {"model": s.model, "revision": s.get("revision"), "style": text_hash(self.style()),
                      "lyrics": text_hash(self.lyrics()), "cot": s.cot, "seed": self.seed("plan"),
                      "abc": text_hash(self.provided_abc()), "sampling": self.sampling("plan")}
        elif stage == "tokens":
            inputs = {"plan": upstream, "seed": self.seed("tokens"), "cfg": s.get("cfg"),
                      "sampling": self.sampling("tokens"), "duration": s.get("duration"),
                      "max_duration": s.get("max_duration"), "init": self.init_hash,
                      "edit": {k: v for k, v in (self.edit or {}).items() if k not in ("tokens", "latents")} or None}
        elif stage == "synth":
            inputs = {"tokens": upstream, "seed": self.seed("synth"), "steps": s.steps, "solver": s.solver,
                      "strength": s.strength if self.init_hash else 1.0, "init": self.init_hash}
        else:
            inputs = {"synth": upstream, "vae": s.vae, "vae_revision": s.get("vae_revision"), "decode": s.decode,
                      "tile_frames": self.tile_frames(), "halo_frames": s.get("halo_frames")}
        return key_of(inputs), inputs

    def tile_frames(self) -> int:
        return int(self.s.get("tile_frames") or (512 if float(self.s.budget) <= 12 else 1024))

    def stage_artifacts(self, stage: str) -> list[dict]:
        """The files a stage leaves behind, as run-events artifact records."""
        d = self.ws.stage_dir(stage)
        if stage == "plan":
            score = self.ws.root / "score.abc"
            return [{"path": str(score), "kind": "text", "mime": "text/vnd.abc"}] if score.is_file() else []
        if stage == "tokens":
            return [{"path": str(d / "semantic.npy"), "kind": "data"}]
        if stage == "synth":
            return [{"path": str(d / "latent.npy"), "kind": "data"}]
        meta = self.ws.stage_meta("decode") or {}
        out = [{"path": str(d / "audio.flac"), "kind": "audio", "mime": "audio/flac"}]
        export = meta.get("export")
        if export and Path(export).is_file() and export != str(d / "audio.flac"):
            fmt = Path(export).suffix.lstrip(".")
            out.append({"path": export, "kind": "audio", "role": "final",
                        "mime": {"mp3": "audio/mpeg", "wav": "audio/wav"}.get(fmt, "audio/flac")})
        return out

    def artifacts_ok(self, stage: str) -> bool:
        d = self.ws.stage_dir(stage)
        need = {"plan": "plan.json", "tokens": "semantic.npy", "synth": "latent.npy", "decode": "audio.flac"}[stage]
        return (d / need).is_file()

    # ---- run ----------------------------------------------------------------------
    def run(self, stages: list[str], *, force: bool, command: str) -> dict:
        r = self.r
        r.run_start(command, workspace=str(self.ws.root))
        r.declare([{"id": st, "kind": "stage", "description": STAGE_TEXT[st]} for st in stages])
        first = STAGES.index(stages[0])
        upstream = None
        if first > 0:
            prev = STAGES[first - 1]
            upstream = self.ws.stage_key(prev)
            if upstream is None or not self.artifacts_ok(prev):
                fail(f"stage `{stages[0]}` needs `{prev}` first",
                     [f"Run `{TOOL} {prev} -w {self.ws.root}` or `{TOOL} generate -w {self.ws.root} --resume`."])
        summary = {"workspace": str(self.ws.root), "stages": {}}
        index = 0
        while index < len(stages):
            stage = stages[index]
            desired, inputs = self.key(stage, upstream)
            if not force and self.ws.stage_key(stage) == desired and self.artifacts_ok(stage):
                r.skip(stage, "fresh", f"{STAGE_TEXT[stage]}: unchanged, reused", reused=self.stage_artifacts(stage))
                summary["stages"][stage] = "reused"
            else:
                r.start(stage, STAGE_TEXT[stage])
                started = time.perf_counter()
                meta = getattr(self, f"stage_{stage}")(inputs)
                seconds = time.perf_counter() - started
                meta.update({"key": desired, "inputs": inputs, "seconds": seconds,
                             "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "yue2_infer": _upstream_version()})
                partial = meta.pop("_partial")
                meta.pop("_artifacts", None)
                self.ws.commit(stage, partial, meta)
                self.persist()
                for record in self.stage_artifacts(stage):
                    r.artifact(stage, **record)
                r.end(stage, "succeeded", data=_public(meta),
                      message=f"{STAGE_TEXT[stage]}: {_describe(stage, meta)} in {seconds:.1f}s")
                summary["stages"][stage] = "ran"
                if stage == "plan" and self.s.get("abc_hook") and self.s.cot != "off" and not self.s.get("_hooked"):
                    self.s["_hooked"] = True
                    if self.hook():
                        continue  # re-plan with the refined score (fast: provided ABC) before tokens
            upstream = desired
            index += 1
        last = stages[-1]
        later = STAGES[STAGES.index(last) + 1:]
        stale = [st for st in later if self.ws.stage_meta(st) and
                 self.ws.stage_meta(st)["inputs"].get(STAGES[STAGES.index(st) - 1]) != self.ws.stage_key(STAGES[STAGES.index(st) - 1])]
        if stale:
            r.log(f"now stale (their input changed): {', '.join(stale)}; run `{TOOL} generate -w {self.ws.root} --resume`",
                  level="warn")
        return summary

    def hook(self) -> bool:
        from .hook import run_hook
        report = run_hook(self.ws, self.s["abc_hook"], timeout=self.s.get("hook_timeout"),
                          validate=bool(self.s.validate), log=lambda m: self.r.log(m, task_id="plan"))
        if not report["changed"]:
            self.r.log("hook changed nothing; performing the planned score")
            return False
        self.r.log(f"hook changed {', '.join(report['changed'])}; re-planning from the refined score")
        self.score_mode = "provided"
        self.persist()
        return True

    # ---- stages -------------------------------------------------------------------
    def stage_plan(self, inputs: dict) -> dict:
        s = self.s
        abc = self.provided_abc()
        if abc is not None and s.cot == "off":
            fail("a provided score needs --cot full or --cot melody (off has no score)")
        if abc is not None and s.cot == "melody" and abc_tools.parse_abc(abc).voices["Vocal"].chords:
            fail("--cot melody with a score that still has chord symbols",
                 [f"Strip them: `{TOOL} abc-strip -i score.abc -o melody.abc`, or use --cot full."])
        plan, seconds = self.engine.plan(style=self.style(), lyrics=self.lyrics(), cot=s.cot, seed=self.seed("plan"),
                                         abc=abc, sampling=self.sampling("plan") or None)
        partial = self.ws.begin("plan")
        plan.save(partial)
        meta = {"truncated": plan.truncated, "abc_tokens": len(plan.abc_ids), "prefix_tokens": len(plan.prefix),
                "provided": abc is not None, "timing": plan.timing}
        artifacts = []
        if plan.abc is not None:
            if abc is None:
                self.ws.write_text("score.abc", plan.abc)
            meta["score"] = _grid_meta(plan.abc)
            artifacts.append({"path": str(self.ws.root / "score.abc"), "kind": "score/abc"})
        if plan.truncated:
            self.r.log("the score hit --plan-max-tokens before its end token (truncated)", level="warn", task_id="plan")
        cap = s.get("duration") or s.get("max_duration")
        score_seconds = (meta.get("score") or {}).get("seconds")
        if cap and score_seconds and score_seconds > float(cap) + 1:
            flag = "--duration" if s.get("duration") else "--max-duration"
            self.r.log(f"the score runs {score_seconds:.0f}s but {flag} {float(cap):g} will cut the song at "
                       f"{float(cap):g}s, mid-score", level="warn", task_id="plan", code="yue.score_exceeds_duration")
        return {**meta, "_partial": partial, "_artifacts": artifacts}

    def stage_tokens(self, inputs: dict) -> dict:
        from yue2 import SymbolicPlan
        s = self.s
        plan = SymbolicPlan.load(self.ws.stage_dir("plan"))
        kwargs: dict = {}
        if self.edit:
            e = self.edit
            kwargs["keep"] = e["tokens"][:e["keep"]]
            if e.get("end") is not None:
                kwargs["tail"] = e["tokens"][e["end"]:]
                kwargs["exact"] = e["end"] - e["keep"]
            elif e.get("extend"):
                kwargs["extend_frames"] = round(e["extend"] * FPS)
        if self.init_latents is not None:
            if self.edit:
                fail("--init-audio cannot be combined with --bars/--extend")
            kwargs["exact"] = len(self.init_latents)
        elif s.get("duration") is not None:
            kwargs["exact"] = max(1, round(float(s["duration"]) * FPS))
        sampling = self.sampling("tokens")
        if s.get("max_duration") is not None and "exact" not in kwargs:
            sampling["max_tokens"] = max(1, round(float(s["max_duration"]) * FPS))
        tokens, timing, truncated = self.engine.tokens(plan, seed=self.seed("tokens"), cfg=s.get("cfg"),
                                                       sampling=sampling or None, **kwargs)
        partial = self.ws.begin("tokens")
        np.save(partial / "semantic.npy", np.asarray(tokens, dtype=np.int32))
        meta = {"frames": len(tokens), "seconds_of_audio": len(tokens) / FPS, "truncated": truncated, "timing": timing}
        if plan.abc:
            grid = abcmap.grid(plan.abc) if _parses(plan.abc) else None
            if grid:
                meta["score_frames"] = grid.frames
        if self.edit:
            meta["edit"] = {k: v for k, v in self.edit.items() if k not in ("tokens", "latents")}
            if self.edit.get("latents") is not None:
                np.save(partial / "source_latent.npy", self.edit["latents"])
        if self.init_latents is not None:
            np.save(partial / "init_latent.npy", self.init_latents)
            meta["init"] = self.init_hash
        if truncated:
            self.r.log("the performance hit its token budget before the end token (truncated)", level="warn", task_id="tokens")
        return {**meta, "_partial": partial}

    def stage_synth(self, inputs: dict) -> dict:
        from yue2 import SymbolicPlan
        from .engine import anchor_mask
        s = self.s
        plan = SymbolicPlan.load(self.ws.stage_dir("plan"))
        tdir = self.ws.stage_dir("tokens")
        tokens = np.load(tdir / "semantic.npy").tolist()
        tmeta = self.ws.stage_meta("tokens") or {}
        init = anchor = None
        strength = 1.0
        edit = tmeta.get("edit")
        if edit and (tdir / "source_latent.npy").is_file():
            source = np.load(tdir / "source_latent.npy")
            n = len(tokens)
            init = np.zeros((n, 64), dtype=np.float32)
            keep = edit["keep"]
            init[:min(keep, len(source))] = source[:keep]
            end_new = None
            if edit.get("end") is not None:
                end_new = keep + (edit["end"] - keep)
                tail = source[edit["end"]:]
                init[end_new:end_new + len(tail)] = tail[:max(0, n - end_new)]
            anchor = anchor_mask(n, keep if not edit.get("extend") else len(source), end_new, int(s.anchor_margin))
            inputs["anchor"] = {"keep": keep, "end": end_new, "margin": int(s.anchor_margin), "frames": int(anchor.sum())}
        elif edit:
            self.r.log("the source has no latents; the edit is re-synthesized without anchoring", level="warn", task_id="synth")
        if (tdir / "init_latent.npy").is_file() and self.s.strength < 1:
            init = np.load(tdir / "init_latent.npy")
            strength = float(self.s.strength)
        latents, timing = self.engine.synth(plan, tokens, seed=self.seed("synth"), steps=int(s.steps), solver=s.solver,
                                            strength=strength, init=init, anchor=anchor, attention=s.attention,
                                            query_chunk=s.get("query_chunk"))
        partial = self.ws.begin("synth")
        np.save(partial / "latent.npy", latents.astype(np.float32))
        return {"frames": len(latents), "strength": strength, "timing": timing, "_partial": partial}

    def stage_decode(self, inputs: dict) -> dict:
        from .engine import export_audio
        s = self.s
        latents = np.load(self.ws.stage_dir("synth") / "latent.npy")
        audio, timing = self.engine.decode(latents, vae=s.vae, revision=s.get("vae_revision"), mode=s.decode,
                                           tile_frames=self.tile_frames(), halo_frames=s.get("halo_frames"))
        partial = self.ws.begin("decode")
        master = export_audio(audio, partial / "audio.flac", "flac")
        target = Path(s["output"]).expanduser() if s.get("output") else self.ws.root / f"song.{s.format}"
        export_audio(audio, target, s.format, master=master)
        seconds = len(audio) / 48000
        return {"audio_seconds": seconds, "sample_rate": 48000, "export": str(target), "timing": timing,
                "_partial": partial,
                "_artifacts": [{"path": str(target), "kind": f"audio/{s.format}", "duration_s": round(seconds, 3)}]}

    # ---- edits --------------------------------------------------------------------
    def load_edit(self) -> None:
        s = self.s
        if not s.get("bars") and not s.get("extend"):
            return
        if s.get("bars") and s.get("extend"):
            fail("--bars and --extend are separate edits; pick one")
        src = Workspace(Path(s["source"])) if s.get("source") else self.ws
        tokens_path = src.stage_dir("tokens") / "semantic.npy"
        if not tokens_path.is_file():
            fail(f"{src.root} has no tokens to edit", [f"Render it first: `{TOOL} generate -w {src.root}`."])
        tokens = np.load(tokens_path).tolist()
        latent_path = src.stage_dir("synth") / "latent.npy"
        latents = np.load(latent_path) if latent_path.is_file() else None
        if latents is not None and len(latents) != len(tokens):
            latents = None
        edit = {"source": str(src.root), "source_tokens": hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()[:16],
                "tokens": tokens, "latents": latents}
        if s.get("extend"):
            edit.update(keep=len(tokens), end=None, extend=float(s["extend"]))
        else:
            score = (src.stage_dir("plan") / "score.abc")
            if not score.is_file():
                fail("--bars needs the score the source was performed from (1-plan/score.abc)")
            try:
                start, end = abcmap.bar_frames(score.read_text(encoding="utf-8"), s["bars"])
            except (ValueError, abc_tools.AbcError) as exc:
                fail(f"--bars {s['bars']}: {exc}")
            if start >= len(tokens):
                fail(f"--bars {s['bars']} starts at frame {start}, past the performance's {len(tokens)} frames")
            if end is not None and end >= len(tokens):
                end = None
            edit.update(bars=s["bars"], keep=start, end=end)
        self.edit = edit

    def load_init(self) -> None:
        path = self.s.get("init_audio")
        if not path:
            return
        from .engine import load_audio
        path = Path(path).expanduser()
        if not path.is_file():
            fail(f"--init-audio {path}: no such file")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        cache = self.ws.root / "init" / f"{digest}-{self.s.vae}.npy"
        if cache.is_file():
            self.init_latents = np.load(cache)
        else:
            self.r.log(f"encoding {path.name} with the {self.s.vae} VAE")
            self.init_latents = self.engine.encode(load_audio(path), vae=self.s.vae, revision=self.s.get("vae_revision"))
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.init_latents)
        self.init_hash = f"{digest}:{self.s.vae}:{self.s.strength}"


# =============================================================================== handlers
def _workspace(s: Settings, slug_from: str | None, fresh_default: str) -> Workspace:
    if s.get("workspace"):
        return Workspace(Path(s["workspace"]))
    slug = re.sub(r"[^a-z0-9]+", "-", (slug_from or fresh_default).lower()).strip("-")[:40] or fresh_default
    return Workspace(Path.cwd() / "yue" / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}")


def _seed(s: Settings, ws: Workspace) -> None:
    if s.get("seed") is None:
        s["seed"] = ws.job().get("seed", random.randrange(2**31))


def _write_texts(s: Settings, ws: Workspace) -> None:
    ws.root.mkdir(parents=True, exist_ok=True)
    if s.get("prompt") is not None:
        ws.write_text("style.txt", s["prompt"])
    if s.get("lyrics") is not None:
        ws.write_text("lyrics.txt", s["lyrics"])


def _stage_range(s: Settings, default_from="plan", default_until="decode") -> list[str]:
    a = STAGES.index(s.get("from_stage") or default_from)
    b = STAGES.index(s.get("until") or default_until)
    if b < a:
        fail(f"--until {STAGES[b]} comes before --from {STAGES[a]}")
    return list(STAGES[a:b + 1])


def _finish(p: Pipeline, summary: dict, command: str) -> int:
    ws = p.ws
    data = {**summary, "seed": p.s["seed"]}
    meta = ws.stage_meta("decode")
    if meta and summary["stages"].get("decode"):
        data["audio"] = meta.get("export")
        data["audio_seconds"] = meta.get("audio_seconds")
    if ws.read_text("score.abc"):
        data["score"] = str(ws.root / "score.abc")
    p.engine.close()
    p.r.result("succeeded", 0, data=data)
    if p.r.mode == "text":
        print(data.get("audio") or data.get("score") or str(ws.root))
    return 0


def cmd_generate(s: Settings, r: Reporter) -> int:
    ws = _workspace(s, s.get("prompt"), "song")
    stages = _stage_range(s)
    if ws.exists() and ws.job() and not (s.get("resume") or s.get("force")) and "workspace" in s["_switches"]:
        if any(ws.stage_meta(st) for st in STAGES):
            fail(f"{ws.root} already holds a song",
                 ["--resume reuses what is still valid and redoes the rest; --force redoes it all."])
    _write_texts(s, ws)
    _seed(s, ws)
    p = Pipeline(s, ws, r)
    if s.get("abc"):
        ws.write_text("score.abc", Path(s["abc"]).expanduser().read_text(encoding="utf-8"))
        p.score_mode = "provided"
    elif "prompt" in s["_switches"] or "lyrics" in s["_switches"] or "cot" in s["_switches"]:
        if p.score_mode == "provided" and not s.get("resume"):
            p.score_mode = "generated"
    if p.score_mode == "provided" and s.get("resume"):
        r.log("performing the workspace's provided score.abc (use `--from plan --force` without --abc to replan)")
    edited = ws.read_text("score.abc")
    plan_abc = (ws.stage_dir("plan") / "score.abc")
    if p.score_mode == "generated" and edited and plan_abc.is_file() and plan_abc.read_text(encoding="utf-8") != edited:
        r.log("score.abc was edited by hand but generate replans; use `yue render` to perform the edit", level="warn")
    p.load_init()
    p.persist()
    summary = p.run(stages, force=bool(s.get("force")), command="generate")
    return _finish(p, summary, "generate")


def cmd_stage(stage: str):
    def handler(s: Settings, r: Reporter) -> int:
        if not s.get("workspace"):
            if stage != "plan":
                fail(f"`{TOOL} {stage}` works on a workspace: pass -w <dir>")
        ws = _workspace(s, s.get("prompt"), "song")
        _write_texts(s, ws)
        _seed(s, ws)
        p = Pipeline(s, ws, r)
        if stage == "plan" and s.get("abc"):
            ws.write_text("score.abc", Path(s["abc"]).expanduser().read_text(encoding="utf-8"))
            p.score_mode = "provided"
        elif stage == "plan" and ("prompt" in s["_switches"] or "lyrics" in s["_switches"]):
            p.score_mode = "generated"
        p.persist()
        summary = p.run([stage], force=bool(s.get("force")), command=stage)
        return _finish(p, summary, stage)
    return handler


def cmd_render(s: Settings, r: Reporter) -> int:
    if not s.get("workspace"):
        fail(f"`{TOOL} render` works on a workspace: pass -w <dir> (its score.abc is performed)")
    ws = Workspace(Path(s["workspace"]))
    _seed(s, ws)
    p = Pipeline(s, ws, r)
    p.load_edit()  # BEFORE anything below can archive the source stages
    _write_texts(s, ws)
    if s.get("abc"):
        ws.write_text("score.abc", Path(s["abc"]).expanduser().read_text(encoding="utf-8"))
    if ws.read_text("score.abc") is None:
        fail(f"{ws.root}/score.abc does not exist", ["Plan one (`yue plan`), transcribe one, or pass --abc."])
    p.score_mode = "provided"
    p.load_init()
    p.persist()
    force = bool(s.get("force")) or bool(p.edit)
    stages = ["plan", "tokens", "synth", "decode"]
    if p.edit:
        # the plan stage may be fresh; tokens onward must re-run for an edit
        summary = p.run(["plan"], force=False, command="render")
        summary2 = p.run(["tokens", "synth", "decode"], force=True, command="render")
        summary["stages"].update(summary2["stages"])
    else:
        summary = p.run(stages, force=force, command="render")
    return _finish(p, summary, "render")


def cmd_remix(s: Settings, r: Reporter) -> int:
    source = s.get("input")
    if not source:
        fail("`yue remix` needs --input: an audio file or a yue workspace")
    source = Path(source).expanduser()
    if source.is_dir() and (source / "job.json").is_file():
        return _remix_workspace(s, r, Workspace(source))
    if not source.is_file():
        fail(f"--input {source}: not an audio file or a yue workspace")
    return _remix_audio(s, r, source)


def _remix_workspace(s: Settings, r: Reporter, src: Workspace) -> int:
    start = s.get("from_stage") or "tokens"
    ws = Workspace(Path(s["workspace"])) if s.get("workspace") else _next_free(src.root.parent / f"{src.root.name}-remix")
    if ws.exists():
        fail(f"{ws.root} is not empty; remix writes a NEW workspace")
    ws.root.mkdir(parents=True)
    job = src.job()
    for name in ("style.txt", "lyrics.txt", "score.abc"):
        if (src.root / name).is_file():
            shutil.copy2(src.root / name, ws.root / name)
    # A new style has to reach the token prefix, so remixing from tokens re-runs
    # the plan stage too -- as a PROVIDED score, which costs no sampling.
    run_from = "plan" if start in ("plan", "tokens") else start
    for stage in STAGES[:STAGES.index(run_from)]:
        if (src.stage_dir(stage)).is_dir():
            shutil.copytree(src.stage_dir(stage), ws.stage_dir(stage))
    merged = Settings({**job, **{k: v for k, v in s.items() if v is not None}})
    merged["_switches"] = s["_switches"]
    if start in ("synth", "decode") and ("prompt" in s["_switches"] or "lyrics" in s["_switches"]):
        r.log(f"--from {start} keeps the performance, so --prompt/--lyrics have no effect", level="warn")
    else:
        _write_texts(merged, ws)
    p = Pipeline(merged, ws, r)
    p.score_mode = "generated" if start == "plan" else "provided"
    if start == "tokens" and (src.stage_dir("plan") / "score.abc").is_file():
        shutil.copy2(src.stage_dir("plan") / "score.abc", ws.root / "score.abc")
    if start in ("synth",) and merged.get("strength") is not None and float(merged["strength"]) < 1:
        latent = src.stage_dir("synth") / "latent.npy"
        if not latent.is_file():
            fail(f"{src.root} has no latents to start from")
        # remix-from-synth with strength: seed the ODE with the source latents
        tokens_dir = ws.stage_dir("tokens")
        np.save(tokens_dir / "init_latent.npy", np.load(latent))
        p.init_hash = f"{src.root.name}:latent:{merged['strength']}"
        meta = read_json(tokens_dir / "stage.json")
        meta["init"] = p.init_hash
        write_json(tokens_dir / "stage.json", meta)
    p.persist()
    summary = p.run(list(STAGES[STAGES.index(run_from):STAGES.index(merged.get("until") or "decode") + 1]),
                    force=True, command="remix")
    summary["source"] = str(src.root)
    return _finish(p, summary, "remix")


def _remix_audio(s: Settings, r: Reporter, audio: Path) -> int:
    from .transcribe import transcribe
    if not s.get("lyrics"):
        fail("remixing audio needs --lyrics: the words cannot be recovered from a recording yet")
    if not s.get("prompt"):
        fail("remixing audio needs --prompt: the target style")
    ws = _workspace(s, f"{audio.stem}-{s['prompt']}", "remix")
    if ws.exists():
        fail(f"{ws.root} is not empty; remix writes a NEW workspace")
    _write_texts(s, ws)
    _seed(s, ws)
    cot = s.cot
    r.run_start("remix", workspace=str(ws.root), input=str(audio))
    r.declare([{"id": "transcribe", "kind": "transcribe", "description": "Score from audio (SheetSage2)"},
               *[{"id": st, "kind": "stage", "description": STAGE_TEXT[st]} for st in STAGES]])
    r.start("transcribe", f"Transcribing {audio.name} with SheetSage2")
    started = time.perf_counter()
    result = transcribe(audio, ws.root / "transcription", melody_only=bool(s.get("melody_only")) or cot == "melody",
                        device=s.get("transcribe_device"), dtype=s.get("transcribe_dtype"), preset=s.get("preset"),
                        max_seconds=s.get("max_seconds"), render_score=bool(s.get("render_score")),
                        log=lambda m: r.log(m, level="debug", task_id="transcribe"))
    abc = result["abc"]
    if cot == "melody" and abc_tools.parse_abc(abc).voices["Vocal"].chords:
        abc = abc_tools.strip_chords(abc)
    ws.write_text("score.abc", abc)
    r.artifact("transcribe", path=str(ws.root / "score.abc"), kind="text", mime="text/vnd.abc")
    r.end("transcribe", "succeeded", data={"seconds": time.perf_counter() - started},
          message=f"Score from audio: {_describe('plan', {'score': _grid_meta(abc)})}")
    if s.get("strength") is not None and float(s["strength"]) < 1:
        s["init_audio"] = str(audio)
    p = Pipeline(s, ws, r)
    p.score_mode = "provided"
    p.load_init()
    p.persist()
    summary = p.run(list(STAGES), force=True, command="remix")
    summary["source"] = str(audio)
    return _finish(p, summary, "remix")


def cmd_refine(s: Settings, r: Reporter) -> int:
    from .hook import run_hook
    if not s.get("workspace") or not s.get("command"):
        fail("`yue refine` needs -w <workspace> and --with '<command>'")
    ws = Workspace(Path(s["workspace"]))
    r.run_start("refine", workspace=str(ws.root))
    r.declare([{"id": "refine", "kind": "hook", "description": "External score edit"}])
    r.start("refine", f"Running: {s['command']}")
    report = run_hook(ws, s["command"], timeout=s.get("hook_timeout"), validate=bool(s.validate),
                      log=lambda m: r.log(m, task_id="refine"))
    r.artifact("refine", path=str(ws.root / "score.abc"), kind="text", mime="text/vnd.abc")
    r.end("refine", "succeeded", data=report, message=f"Refined: changed {', '.join(report['changed']) or 'nothing'}")
    if s.get("render") and report["changed"]:
        _seed(s, ws)
        p = Pipeline(Settings({**ws.job(), **{k: v for k, v in s.items() if v is not None}}), ws, r)
        p.s["_switches"] = s["_switches"]
        p.score_mode = "provided"
        p.persist()
        summary = p.run(list(STAGES), force=False, command="refine")
        return _finish(p, summary, "refine")
    r.result("succeeded", 0, data={"workspace": str(ws.root), "refine": report})
    if r.mode == "text":
        print(ws.root / "score.abc")
    return 0


def cmd_transcribe(s: Settings, r: Reporter) -> int:
    from .transcribe import transcribe
    audio = Path(s.get("input") or fail("`yue transcribe` needs --input <audio>")).expanduser()
    if not audio.is_file():
        fail(f"--input {audio}: no such file")
    ws = _workspace(s, audio.stem, "transcription")
    r.run_start("transcribe", workspace=str(ws.root), input=str(audio))
    r.declare([{"id": "transcribe", "kind": "transcribe", "description": "Score from audio (SheetSage2)"}])
    r.start("transcribe", f"Transcribing {audio.name} with SheetSage2")
    started = time.perf_counter()
    result = transcribe(audio, ws.root / "transcription", melody_only=bool(s.get("melody_only")),
                        device=s.get("transcribe_device"), dtype=s.get("transcribe_dtype"), preset=s.get("preset"),
                        max_seconds=s.get("max_seconds"), render_score=bool(s.get("render_score")),
                        log=lambda m: r.log(m, level="debug", task_id="transcribe"))
    ws.write_text("score.abc", result["abc"])
    job = ws.job()
    job.update({"score_mode": "provided", "cot": "melody" if s.get("melody_only") else "full"})
    ws.save_job(job)
    seconds = time.perf_counter() - started
    r.artifact("transcribe", path=str(ws.root / "score.abc"), kind="text", mime="text/vnd.abc", role="final")
    r.end("transcribe", "succeeded", data={"device": result["device"]},
          message=f"Score: {_describe('plan', {'score': _grid_meta(result['abc'])})} in {seconds:.1f}s")
    r.result("succeeded", 0, data={"workspace": str(ws.root), "score": str(ws.root / "score.abc")})
    if r.mode == "text":
        print(ws.root / "score.abc")
    return 0


def cmd_import(s: Settings, r: Reporter) -> int:
    """Rebuild a workspace from upstream artifacts, computing the keys this CLI
    would have written, so `yue synth/decode/render -w` can continue from it."""
    src = Path(s.get("input") or fail("`yue import` needs --input <upstream artifact dir>")).expanduser()
    for name in ("request.json", "plan.json", "plan_manifest.json", "semantic.npy"):
        if not (src / name).is_file():
            fail(f"{src} has no {name}; not an upstream SongResult directory")
    request = read_json(src / "request.json")
    config = read_json(src / "config.json") if (src / "config.json").is_file() else {}
    ws = _workspace(s, f"import-{src.name}", "import")
    if ws.exists():
        fail(f"{ws.root} is not empty")
    ws.root.mkdir(parents=True)
    ws.write_text("style.txt", request["style"])
    ws.write_text("lyrics.txt", request["lyrics"])
    gen = config.get("generation", {})
    settings = Settings({"seed": request["seed"], "cot": request["cot"], "_switches": set(),
                         "cfg": request.get("cfg_scale"), "steps": gen.get("ode_steps")})
    p = Pipeline(settings, ws, r)
    p.score_mode = "provided" if request.get("abc") else "generated"
    if request.get("abc"):
        ws.write_text("score.abc", request["abc"])
    plan_dir = ws.begin("plan")
    for name in ("plan.json", "plan_manifest.json", "abc_tokens.npy", "prefix.npy", "score.abc"):
        if (src / name).is_file():
            shutil.copy2(src / name, plan_dir / name)
    if (src / "score.abc").is_file() and not request.get("abc"):
        shutil.copy2(src / "score.abc", ws.root / "score.abc")
    upstream, inputs = p.key("plan", None)
    ws.commit("plan", plan_dir, {"key": upstream, "inputs": inputs, "imported_from": str(src)})
    for stage, files in (("tokens", ("semantic.npy",)), ("synth", ("latent.npy",)), ("decode", ("audio.flac",))):
        if not all((src / f).is_file() for f in files):
            break
        key, inputs = p.key(stage, upstream)
        d = ws.begin(stage)
        for f in files:
            shutil.copy2(src / f, d / f)
        meta = {"key": key, "inputs": inputs, "imported_from": str(src)}
        if stage == "tokens":
            meta["frames"] = int(np.load(src / f).shape[0])
        ws.commit(stage, d, meta)
        upstream = key
    p.persist()
    return _emit_data(r, {"workspace": str(ws.root), "stages": [st for st in STAGES if ws.stage_meta(st)]}) if r.mode != "text" \
        else (print(ws.root) or 0)


def cmd_status(s: Settings, r: Reporter) -> int:
    if not s.get("workspace"):
        fail("`yue status` needs -w <workspace>")
    ws = Workspace(Path(s["workspace"]))
    if not ws.root.is_dir():
        fail(f"{ws.root}: no such workspace")
    stages = {}
    previous = None
    for stage in STAGES:
        meta = ws.stage_meta(stage)
        if meta is None:
            stages[stage] = {"state": "missing"}
        else:
            upstream_ok = previous is None or meta["inputs"].get(STAGES[STAGES.index(stage) - 1]) == previous
            stages[stage] = {"state": "ok" if upstream_ok else "stale", "key": meta["key"],
                             "created": meta.get("created"), "seconds": round(meta.get("seconds", 0), 1),
                             **{k: meta[k] for k in ("truncated", "frames", "audio_seconds", "export", "score",
                                                     "score_frames", "edit", "provided") if k in meta}}
        previous = meta["key"] if meta else None
    job = ws.job()
    data = {"workspace": str(ws.root), "seed": job.get("seed"), "score_mode": job.get("score_mode"),
            "cot": job.get("cot") or DEFAULTS["cot"], "stages": stages}
    edited = ws.read_text("score.abc")
    planned = ws.stage_dir("plan") / "score.abc"
    if edited and planned.is_file() and planned.read_text(encoding="utf-8") != edited:
        data["score_edited"] = True
    if r.mode == "text":
        print(f"{ws.root}  (seed {data['seed']}, cot {data['cot']}, score {data['score_mode']})")
        for stage, info in stages.items():
            extra = " ".join(f"{k}={v}" for k, v in info.items() if k not in ("state", "key", "edit", "score"))
            print(f"  {DIRS[stage]:10} {info['state']:8} {extra}")
        if data.get("score_edited"):
            print("  score.abc differs from 1-plan/score.abc: `yue render` performs the edit")
        return 0
    r.result("succeeded", 0, data=data)
    return 0


def cmd_abc_inspect(s: Settings, r: Reporter) -> int:
    path = Path(s.get("input") or fail("--input <score.abc> is required")).expanduser()
    data = abc_tools.report(abc_tools.parse_abc(path.read_text(encoding="utf-8")))
    data["grid"] = _grid_meta(path.read_text(encoding="utf-8"))
    return _emit_data(r, data)


def cmd_abc_strip(s: Settings, r: Reporter) -> int:
    src = Path(s.get("input") or fail("--input <score.abc> is required")).expanduser()
    dst = Path(s.get("output") or fail("--output <new.abc> is required")).expanduser()
    if dst.exists():
        fail(f"{dst} exists; write a new file")
    dst.write_text(abc_tools.strip_chords(src.read_text(encoding="utf-8"), s.keep_voice), encoding="utf-8")
    return _emit_data(r, {"output": str(dst), "kept_voice": s.keep_voice})


def cmd_abc_compare(s: Settings, r: Reporter) -> int:
    before = Path(s.get("before") or fail("--before is required")).expanduser()
    after = Path(s.get("after") or fail("--after is required")).expanduser()
    names = abc_tools.VOICES if s.voices == "both" else (s.voices,)
    data = abc_tools.compare(abc_tools.parse_abc(before.read_text(encoding="utf-8")),
                             abc_tools.parse_abc(after.read_text(encoding="utf-8")), names, bool(s.get("allow_tempo_change")))
    _emit_data(r, data)
    return 0 if data["match"] else 1


def cmd_brief(s: Settings, r: Reporter) -> int:
    from .hook import BRIEF
    sys.stdout.write(BRIEF.read_text(encoding="utf-8"))
    return 0


def cmd_doctor(s: Settings, r: Reporter) -> int:
    import platform
    import torch
    data: dict = {"yue": importlib.metadata.version("yue"), "yue2_infer": _upstream_version(),
                  "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                  "cuda": {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda, "devices": []},
                  "mps": torch.backends.mps.is_available()}
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        data["cuda"]["devices"].append({"id": i, "name": prop.name, "memory_gib": round(prop.total_memory / 2**30, 1),
                                        "capability": f"{prop.major}.{prop.minor}",
                                        "fp8": (prop.major, prop.minor) >= (8, 9), "bf16": torch.cuda.is_bf16_supported()})
    for pkg in ("vllm", "triton", "transformers"):
        try:
            data[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            data[pkg] = None
    from huggingface_hub import scan_cache_dir
    try:
        cached = {repo.repo_id: round(repo.size_on_disk / 2**30, 2) for repo in scan_cache_dir().repos
                  if repo.repo_id.startswith("m-a-p/")}
    except Exception:  # noqa: BLE001
        cached = {}
    data["weights_gib"] = cached
    from .transcribe import ENV
    data["sheetsage2_env"] = (ENV / ".venv").is_dir()
    data["ffmpeg"] = shutil.which("ffmpeg")
    return _emit_data(r, data)


def _emit_data(r: Reporter, data: dict) -> int:
    if r.mode == "text":
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    else:
        r.result("succeeded", 0, data=data)
    return 0


HANDLERS = {
    "generate": cmd_generate, "plan": cmd_stage("plan"), "tokens": cmd_stage("tokens"),
    "synth": cmd_stage("synth"), "decode": cmd_stage("decode"), "render": cmd_render, "remix": cmd_remix,
    "refine": cmd_refine, "transcribe": cmd_transcribe, "import": cmd_import, "status": cmd_status, "abc-inspect": cmd_abc_inspect,
    "abc-strip": cmd_abc_strip, "abc-compare": cmd_abc_compare, "brief": cmd_brief, "doctor": cmd_doctor,
    "runpod setup": lambda s, r: _runpod("setup", s, r), "runpod status": lambda s, r: _runpod("status", s, r),
    "runpod teardown": lambda s, r: _runpod("teardown", s, r),
}


def _runpod(action: str, s: Settings, r: Reporter) -> int:
    from .remote import setup as rp_setup
    return getattr(rp_setup, action)(s, r)


# =============================================================================== helpers
def _version() -> str:
    try:
        return importlib.metadata.version("yue")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def _install_sigterm() -> None:
    """SIGTERM gets the same orderly close as Ctrl-C, exiting 143 (spec 2.8)."""
    import signal

    def handler(signum, frame):
        exc = KeyboardInterrupt()
        exc.sigterm = True
        raise exc

    try:
        signal.signal(signal.SIGTERM, handler)
    except ValueError:  # not the main thread (tests, embedding)
        pass


def _error_code(exc: BaseException) -> str:
    from .hook import HookFailed
    if isinstance(exc, HookFailed):
        return "yue.hook_failed"
    if isinstance(exc, abc_tools.AbcError):
        return "yue.invalid_score"
    if isinstance(exc, FileNotFoundError):
        return "not_found"
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
        return "resource"
    if isinstance(exc, subprocess_timeout()):
        return "timeout"
    if isinstance(exc, RuntimeError) and ("needs CUDA" in str(exc) or "not installed" in str(exc)):
        return "config"
    return "internal"


def subprocess_timeout():
    import subprocess
    return subprocess.TimeoutExpired


def _upstream_version() -> str:
    try:
        return importlib.metadata.version("yue2-infer")
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _parses(abc: str) -> bool:
    try:
        abc_tools.parse_abc(abc)
        return True
    except abc_tools.AbcError:
        return False


def _grid_meta(abc: str) -> dict | None:
    try:
        g = abcmap.grid(abc)
    except abc_tools.AbcError as exc:
        return {"parse_error": str(exc)}
    return {"bpm": g.bpm, "bars": g.bars, "frames": g.frames, "seconds": g.frames / FPS}


def _describe(stage: str, meta: dict) -> str:
    if stage == "plan":
        score = meta.get("score") or {}
        if not score:
            return "no score (cot off)"
        if "parse_error" in score:
            return f"score does not parse ({score['parse_error']})"
        return f"{score['bars']} bars at {score['bpm']} BPM (~{score['seconds']:.1f}s)"
    if stage == "tokens":
        return f"{meta['frames']} frames = {meta['seconds_of_audio']:.1f}s"
    if stage == "synth":
        return f"{meta['frames']} latent frames"
    return f"{meta['audio_seconds']:.1f}s -> {meta['export']}"


def _public(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if not k.startswith("_") and k not in ("inputs",)}


def _next_free(base: Path) -> Workspace:
    n = 1
    while (base.parent / f"{base.name}-{n}").exists():
        n += 1
    return Workspace(base.parent / f"{base.name}-{n}")


if __name__ == "__main__":
    raise SystemExit(main())

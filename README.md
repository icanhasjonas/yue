# yue

A command-line tool for [YuE2](https://map-yue2.github.io/). Style and lyrics go
in, a finished song comes out, and every step in between writes a file you can
keep, edit, resume or redo on its own.

```
style + lyrics ─▶ plan ─▶ tokens ─▶ synth ─▶ decode ─▶ song.flac
                  score.abc  semantic.npy  latent.npy  audio.flac
```

| Stage | What it makes | Upstream call | M1 Max, 89 s song |
|---|---|---|---|
| `plan` | the score: melody + chords as ABC text | `pipe.plan()` | 66 s |
| `tokens` | the performance: 25 codec tokens per second | `_generate()` | 194 s |
| `synth` | 64-dim acoustic latents, 25 per second (flow matching) | `CachedNAR.velocity` | 182 s |
| `decode` | 48 kHz stereo audio (VAE) | `YuE2VAE.decode*` | 45 s |

The weights are **CC BY-NC 4.0**: what you generate is for non-commercial use.

## Install

```sh
./install.sh          # uv sync + a `yue` launcher in ~/bin (or ~/.local/bin)
./install.sh --cuda   # Linux + NVIDIA: adds vllm, triton, nvml
yue doctor            # devices, versions, cached weights
```

The first run downloads `m-a-p/YuE2-3B` (7.3 GB) and `m-a-p/YuE2-Vae` into the
Hugging Face cache.

## Use

```sh
yue generate --style "dark folk, female vocal, cello" --lyrics @song.txt
yue generate --workspace runs/neon --resume --tokens-temperature 0.9    # redo tokens, synth, decode
yue plan --workspace runs/neon --style "..." --lyrics @song.txt       # just the score
yue render --workspace runs/neon                                         # perform an edited score.abc
yue render --workspace runs/neon --bars 17-24                            # repaint bars 17-24, keep the rest
yue render --workspace runs/neon --extend 30 --lyrics @longer.txt        # continue the song
yue synth  --workspace runs/neon --synth-seed 3 --steps 48               # same performance, new sound
yue decode --workspace runs/neon --vae legacy --decode full -o legacy.flac
yue remix  -i runs/neon --style "acoustic jazz trio"           # same score, new style
yue remix  -i song.mp3 --lyrics @words.txt --style "metal" --cot melody
yue transcribe -i song.mp3                                      # audio -> score.abc (SheetSage2)
yue status --workspace runs/neon
yue <verb> --help                                               # every switch, generated
```

`yue help` lists all the verbs.

### The grammar

It's the same as `img` / `vid` / `snd`:

- There are no positional arguments. Every value has a name.
- An unknown switch, a missing value, or a scalar switch given twice aborts with
  exit 1 and a suggestion.
- `--args @job.json | @- | '{"steps": 48}'` supplies any subset of the switches
  (snake_case keys).
- Precedence runs `defaults < workspace job.json < --args < switches`.
- Any text switch takes `@file`.
- One long name per switch. Single letters are shared with img/vid/snd and mean
  the same thing everywhere: `-s` style, `-i` input, `-o` output, `-m` model,
  `-d` debug, `-f` from. `--workspace` and `--quiet` get no letter because
  `-w` / `-q` are `--wait` / `--quality` in snd.
- `yue spec` prints all of it as JSON. `snd yue` is generated from that output.

## Workspaces

A song is a directory:

```
job.json      every knob of the last run: the base layer for the next one
style.txt     lyrics.txt     score.abc      editable; render and hooks read them back
1-plan/  2-tokens/  3-latents/  4-audio/    each with stage.json
song.<fmt>    the export
.history/     every stage a re-run replaced
init/         VAE-encoded --init-audio, cached
refine/       one report per hook run
```

Each `stage.json` records a **key**: a hash of that stage's inputs plus the key
of the stage before it. Keys chain, so changing a knob invalidates exactly that
stage and everything after it. That's all `--resume` needs. A stage is written
to `<dir>.partial/` and renamed into place when it finishes, so an interrupted
run never looks complete.

`score_mode` in `job.json` says whether the score was **generated** or
**provided** (by `--abc`, `render`, `transcribe` or a hook). Once it's provided,
`generate --resume` keeps performing it instead of silently replanning over your
edit.

## Every knob

`yue <verb> --help` is the reference. An unset knob falls through to upstream's
release default, so an upgrade that retunes a default is inherited.

| Group | Switches | Upstream default |
|---|---|---|
| Song | `--style` (`-s`), `--lyrics`, `--cot full\|melody\|off`, `--abc`, `--seed`, `--duration` (exact), `--max-duration` | `full`, random seed (recorded) |
| Score sampling | `--plan-seed --plan-temperature --plan-top-p --plan-top-k --plan-repetition-penalty --plan-penalty-window --plan-min-tokens --plan-max-tokens` | .7 / .9 / 30 / 1.005 / 100 / 32 / 4096 |
| Performance sampling | `--tokens-*` (same eight), `--cfg` | 1.0 / .95 / 100 / 1.2 / 50 / 200 / 9000, cfg 1.0 (1.01 off) |
| Synthesis | `--steps --solver midpoint\|euler\|heun --synth-seed --strength --init-audio --attention sdpa\|math\|flash --query-chunk` | 32, midpoint |
| Decoder | `--vae standard\|legacy\|<repo\|path> --vae-revision --decode tiled\|full --tile-frames --halo-frames` | standard, tiled 1024 |
| Edit | `--bars A-B\|A-end --extend <s> --source <ws> --anchor-margin <frames>` | margin 12 |
| Hook | `--abc-hook '<cmd>' --hook-timeout` | |
| Runtime | `--model --revision --device --backend torch\|torch-eager\|vllm --quantization none\|fp8 --budget --offload-ar --offline --no-verify-hashes` | auto, torch, 24 GiB |
| Output | `-w --output --format flac\|wav\|mp3 --resume --force --from --until --output-format text\|json\|stream-json` | flac, text |

### The synth stage is our own loop, and it matches upstream exactly

Upstream's `synthesize()` hard-codes a 32-step midpoint solve from pure noise.
`yue` runs its own loop over upstream's `CachedNAR.velocity`, so it can also:

- start from existing audio (`--strength`),
- swap the solver,
- re-impose a source's latents outside an edited region after every step. This
  is RePaint for a flow ODE, and it's what makes `render --bars` seamless.

With the defaults it **reproduces upstream bit for bit**. That was checked on a
real render: same tokens and seed, `max|diff| = 0.0` across 2220 × 64 latents.

### Bars are frames

The tokens and latents both run at 25 frames per second, so with `Q:1/4=<bpm>`
bar N starts at `quarters_before_N × 60 / bpm × 25`. The first local render was
100 BPM 4/4 with 37 bars = 2220 frames, and the model emitted exactly that.
That's one song, though. The model performs the score, it doesn't sequence it,
so `yue status` prints both numbers.

## Hand the score to another tool

`--abc-hook` (on `generate`, `plan`, `render`, `remix`) and `yue refine` run a
shell command after the score exists and before anything is performed:

```sh
yue generate --style "city pop" --lyrics @l.txt \
  --abc-hook 'claude -p "Read $YUE_BRIEF. Reharmonize $YUE_ABC with modern jazz
              voicings, keep every melody note. Edit the file in place." \
              --allowedTools Read,Edit'
```

The command gets `YUE_ABC` (edit in place), `YUE_ABC_ORIGINAL`, `YUE_STYLE_FILE`,
`YUE_LYRICS_FILE`, `YUE_BRIEF` (the upstream editing brief), `YUE_JOB` and
`YUE_WORKSPACE`. Afterwards the score is parsed with the upstream dialect
checker. If it doesn't pass, the previous score is restored, the attempt is kept
as `score.rejected.abc`, and the run stops rather than conditioning a long
render on a broken score. A before/after note comparison lands in `refine/`.

## Machine-readable output

`--output-format stream-json` prints **run-events v1** on stdout: `run:start`,
`task:declare`, `task:start`, `task:progress`, `artifact`, `task:skip`,
`task:end`, `log`, and `result` as the last line. The spec, its prior art and a
JSON Schema are in [docs/EVENT-STREAM.md](docs/EVENT-STREAM.md). It's written to
be reused by `img` / `vid` / `snd`. `--output-format json` prints only the final
result object.

## RunPod (remote GPU, per second)

```sh
yue runpod setup                 # guided, resumable; asks before anything that costs money
yue generate --style "..." --lyrics @song.txt --remote runpod
yue render --workspace runs/neon --bars 9-16 --remote runpod
yue runpod status                # workers, queue, volume
yue runpod teardown [--volume]   # endpoint + template (+ weights volume)
```

`--remote runpod` works on `generate`, `plan`, `tokens`, `synth`, `decode` and
`render`. It uploads the workspace inputs and runs the same verb on a serverless
worker. The worker streams the same run-events back, and the resulting stage
directories land in your **local** workspace. The replaced ones move to
`.history/`, just like a local run.

| Piece | What |
|---|---|
| Image | `ghcr.io/<you>/yue:<sha>`, built by `.github/workflows/image.yml` on every push to `main` (about 23 min cold, faster with cache) |
| Weights | on a network volume (`HF_HOME=/runpod-volume/hf`), downloaded once by the setup's prime job |
| Endpoint | 24 GB GPUs in preference order (A5000, 3090, A4500, 4090, L4), 0 workers when idle, FlashBoot, 10 min job cap (`--timeout-minutes`) |
| Keys | `RUNPOD_API_KEY` for setup, status and teardown. `YUE_RUNPOD_JOB_KEY` (a Restricted key with Read/Write on the yue endpoint only) for `--remote` runs. Both go in `.env` |
| State | `~/.config/yue/runpod.json` holds ids, never keys |

Measured on 2026-09-17 with a warm worker in EU-RO-1, for a 20 s song:

| Stage | Time |
|---|---|
| Score | 36 s |
| Tokens | 4 s (~131 tok/s) |
| Synth | 2 s |
| Decode | 7 s |
| Total, including upload and download | 65 s |

The first job on a **new image** pulls about 6 GB and took 8 min. A `render --bars 3-5` on RunPod
kept every token and latent outside bars 3-5 bit-identical to the source.

`transcribe` and `remix` run remotely too: the image carries SheetSage2 in its own
Python 3.11 environment. Neither has been verified on RunPod yet.

A job's uploads (input audio plus workspace files) are capped at about 6.7 MB:
RunPod accepts a 10 MB request and base64 adds a third. A song as mp3 fits; as WAV
it usually does not, and the client refuses it before submitting.

## CUDA

This was built on Apple Silicon (MPS), where upstream runs but isn't officially
supported: the test song ran about 5.6x slower than realtime. On Linux with a
24 GB NVIDIA GPU:

- `./install.sh --cuda`
- the default `torch` backend uses CUDA graphs
- `--backend vllm` and `--quantization fp8` (compute capability 8.9 or higher)
  become available
- `--attention flash` is accepted

`yue doctor` shows what the machine supports. The CUDA-only options are refused
in plain words anywhere else. **The CUDA path has not been run yet.**

## Upgrading YuE2

```sh
uv lock --upgrade-package yue2-infer && uv sync && uv run pytest
```

`src/yuecli/engine.py` is the only module that touches upstream internals.
`tests/test_upstream_contract.py` pins every one of them, so an upgrade that
moves an internal fails the suite before it renders anything.
`src/yuecli/abc_tools.py` is vendored verbatim from the upstream skill
(Apache-2.0); re-copy it on upgrade.

## Transcription (SheetSage2)

SheetSage2's dependency pins (torch 2.8, transformers 4.45, numpy 1.24) conflict
with YuE2's, so it lives in its own uv environment, `envs/sheetsage2`, created on
first use. It runs as a subprocess and hands back JSON. On the M1 Max it
transcribed an 89 s song in 49 s.

"""Every verb's surface, declared once. `--help`, parsing and `--args` all derive from these.

Names follow img/vid/snd where a concept is shared (`--style`, `--lyrics`,
`--seed`, `--duration`, `--input`, `--output`, `--format`, `--model`, `--from`,
`--debug`), and name the pipeline stage where it is not: a sampling knob is
`--plan-*` (the score) or `--tokens-*` (the performance), an ODE knob belongs to
synth, a decoder knob to decode. Defaults are upstream's release defaults
(yue2.protocol.GenerationConfig) and are NOT repeated here -- an unset knob is
left to upstream, so an upgrade that retunes a default is inherited, not masked.
"""
from __future__ import annotations

from .args import Field, Verb

F = Field
STAGE_CHOICES = ("plan", "tokens", "synth", "decode")

# --- workspace / output ---------------------------------------------------------
WORKSPACE = F("workspace", "path", "Song workspace directory (created if missing; default ./yue/<time>-<slug>)",
              group="Workspace")
OUTPUT = F("output", "path", "Also export the finished song here (default <workspace>/song.<format>)",
           group="Workspace")
FORMAT = F("format", "enum", "Export format; the 24-bit FLAC master is always kept in 4-audio/",
           choices=("flac", "wav", "mp3"), group="Workspace")
RESUME = F("resume", "bool", "Reuse every stage whose inputs did not change; redo the stale ones", group="Workspace")
FORCE = F("force", "bool", "Redo the stages in range even when fresh (old output moves to .history/)", group="Workspace")
FROM = F("from", "enum", "First stage to run", choices=STAGE_CHOICES, group="Workspace")
UNTIL = F("until", "enum", "Last stage to run (e.g. --until plan to stop at the score)",
          choices=STAGE_CHOICES, group="Workspace")
OUTPUT_FORMAT = F("output_format", "enum", "text: human progress on stderr; json: one result object; "
                  "stream-json: JSONL events on stdout (docs/EVENT-STREAM.md)",
                  choices=("text", "json", "stream-json"), group="Workspace")
QUIET = F("quiet", "bool", "No progress output (text mode)", group="Workspace")
DEBUG = F("debug", "bool", "Print tracebacks and the resolved settings", group="Workspace")

# --- request --------------------------------------------------------------------
# `--style`, not `--prompt`: across img/vid/snd a --prompt is a free description and
# --style is style tags (`snd suno --style "dark pop"`), which is exactly what YuE2 takes.
PROMPT = F("style", "text", "Style tags: genre, instruments, vocal, language, tempo", group="Song")
LYRICS = F("lyrics", "text", "Lyrics with section tags ([verse], [chorus], ...)", group="Song")
COT = F("cot", "enum", "Symbolic planning: full = melody+chords, melody = melody only (covers), off = no score",
        choices=("full", "melody", "off"), group="Song")
ABC = F("abc", "path", "Perform this score instead of planning one (needs --cot full|melody)", group="Song")
SEED = F("seed", "int", "Seed for every stage (a stage-specific seed overrides it); default random, recorded",
         minimum=0, maximum=2**63 - 1, group="Song")
DURATION = F("duration", "float", "EXACT song length in seconds: the end token is suppressed until then",
             minimum=0.04, maximum=24576 / 25, group="Song")
MAX_DURATION = F("max_duration", "float", "Longest the song may run, in seconds (upstream cap: 360)",
                 minimum=0.04, maximum=24576 / 25, group="Song")


def sampling(stage: str, label: str) -> tuple[Field, ...]:
    group = f"{label} sampling ({stage} stage)"
    return (
        F(f"{stage}_seed", "int", f"Seed for the {stage} stage only", minimum=0, maximum=2**63 - 1, group=group),
        F(f"{stage}_temperature", "float", "Sampling temperature (0 = greedy)", minimum=0, maximum=5, group=group),
        F(f"{stage}_top_p", "float", "Nucleus sampling mass", minimum=0.0001, maximum=1, group=group),
        F(f"{stage}_top_k", "int", "Top-k candidates", minimum=1, group=group),
        F(f"{stage}_repetition_penalty", "float", "Penalty on recently used tokens", minimum=0.0001, group=group),
        F(f"{stage}_penalty_window", "int", "How many recent tokens the penalty sees", minimum=1, maximum=100, group=group),
        F(f"{stage}_min_tokens", "int", "Suppress the end token before this many tokens", minimum=0, group=group),
        F(f"{stage}_max_tokens", "int", "Hard token budget", minimum=1, maximum=24576, group=group),
    )


PLAN_SAMPLING = sampling("plan", "Score")  # upstream: .7 / .9 / 30 / 1.005 / 100 / 32 / 4096
TOKENS_SAMPLING = sampling("tokens", "Performance") + (
    F("cfg", "float", "Text guidance scale; 1 = off (default 1.0, 1.01 with --cot off). Doubles token cost",
      minimum=0, maximum=20, group="Performance sampling (tokens stage)"),
)  # upstream: 1.0 / .95 / 100 / 1.2 / 50 / 200 / 9000

SYNTH = (
    F("steps", "int", "ODE steps (default 32)", minimum=1, maximum=1000, group="Acoustic synthesis (synth stage)"),
    F("solver", "enum", "ODE solver; midpoint is upstream's (2 model calls/step), euler is half the cost",
      choices=("midpoint", "euler", "heun"), group="Acoustic synthesis (synth stage)"),
    F("synth_seed", "int", "Noise seed for the synth stage only", minimum=0, maximum=2**63 - 1,
      group="Acoustic synthesis (synth stage)"),
    F("strength", "float", "With --init-audio or a source: 1 = ignore the source, lower keeps more of it",
      minimum=0.01, maximum=1, group="Acoustic synthesis (synth stage)"),
    F("init_audio", "path", "Audio to start the ODE from (VAE-encoded; forces the song to its length). EXPERIMENTAL",
      group="Acoustic synthesis (synth stage)"),
    F("attention", "enum", "Attention kernel; flash needs CUDA. Does not change the result",
      choices=("sdpa", "math", "flash"), group="Acoustic synthesis (synth stage)"),
    F("query_chunk", "int", "Attention query rows per block (memory bound, not the result)", minimum=1,
      group="Acoustic synthesis (synth stage)"),
)

DECODE = (
    F("vae", "str", "Decoder: standard (listening), legacy (paper benchmarks), an HF repo or a local path",
      group="Decoder (decode stage)"),
    F("vae_revision", "str", "Decoder repo revision", group="Decoder (decode stage)"),
    F("decode", "enum", "tiled: bounded memory, exact crops; full: one pass, more memory",
      choices=("tiled", "full"), group="Decoder (decode stage)"),
    F("tile_frames", "int", "Tiled decode core size in frames (default 1024; 512 when --budget <= 12)",
      minimum=1, group="Decoder (decode stage)"),
    F("halo_frames", "int", "Tiled decode context each side (default: the decoder's required minimum)",
      minimum=0, group="Decoder (decode stage)"),
)

RUNTIME = (
    F("model", "str", "YuE2 checkpoint: HF repo or local path (default m-a-p/YuE2-3B)", group="Runtime"),
    F("revision", "str", "Checkpoint revision", group="Runtime"),
    F("device", "str", "auto | cuda | cuda:N | mps | cpu", group="Runtime"),
    F("backend", "enum", "torch (CUDA graphs on CUDA), torch-eager, vllm (CUDA + --extra cuda)",
      choices=("torch", "torch-eager", "vllm"), group="Runtime"),
    F("quantization", "enum", "fp8 for the token stages (CUDA, compute capability >= 8.9)",
      choices=("none", "fp8"), group="Runtime"),
    F("budget", "float", "Memory budget in GiB (default 24)", minimum=4, group="Runtime"),
    F("offload_ar", "bool", "Move token-stage weights to CPU during synthesis (less VRAM)", group="Runtime"),
    F("offline", "bool", "Never touch the network; use cached weights only", group="Runtime"),
    F("verify_hashes", "bool", "Hash-check weights on load (default on)", group="Runtime"),
)

RENDER = (
    F("bars", "str", "Repaint only these bars (1-based, inclusive): 9-16, 9-end. EXPERIMENTAL", group="Edit"),
    F("extend", "float", "Keep the whole source and continue it by at least this many seconds. EXPERIMENTAL",
      minimum=0.04, group="Edit"),
    F("source", "path", "Workspace the edit reads tokens/latents from (default: the target workspace)", group="Edit"),
    F("anchor_margin", "int", "Frames each side of an edit re-solved for a seamless join (default 12 = 0.48 s)",
      minimum=0, maximum=500, group="Edit"),
)

HOOK = (
    F("abc_hook", "str", "Shell command run after the score exists, before it is performed; it may edit "
      "$YUE_ABC, $YUE_STYLE_FILE, $YUE_LYRICS_FILE (see `yue refine --help`)", group="Score hook"),
    F("hook_timeout", "float", "Seconds before the hook is killed (default: none)", minimum=1, group="Score hook"),
)

TRANSCRIBE = (
    F("melody_only", "bool", "Keep both melodies, drop chord symbols (recommended for covers)", group="Transcription"),
    F("transcribe_device", "str", "SheetSage2 device (default: auto)", group="Transcription"),
    F("transcribe_dtype", "enum", "SheetSage2 precision", choices=("bf16", "fp32"), group="Transcription"),
    F("preset", "enum", "SheetSage2 preset", choices=("default", "paper"), group="Transcription"),
    F("max_seconds", "float", "Only transcribe the first N seconds", minimum=1, group="Transcription"),
    F("render_score", "bool", "Also render the score to PDF/SVG/PNG", group="Transcription"),
)

REMOTE = (
    F("remote", "enum", "Run on a RunPod serverless GPU instead of this machine (`yue runpod setup` first)",
      choices=("local", "runpod"), group="Runtime"),
    F("remote_fetch", "enum", "What a remote run sends back: all stage artifacts, or just the audio",
      choices=("all", "audio"), group="Runtime"),
    F("dry_run", "bool", "With --remote runpod: show the job that WOULD be submitted (argv, files, sizes) and stop. "
      "A submitted job is billed and cannot be taken back", group="Runtime"),
)
COMMON_TAIL = (*REMOTE, OUTPUT_FORMAT, QUIET, DEBUG)
LOCAL_TAIL = (OUTPUT_FORMAT, QUIET, DEBUG)
PIPELINE = (WORKSPACE, OUTPUT, FORMAT, RESUME, FORCE, FROM, UNTIL,
            PROMPT, LYRICS, COT, ABC, SEED, DURATION, MAX_DURATION,
            *PLAN_SAMPLING, *TOKENS_SAMPLING, *SYNTH, *DECODE, *HOOK, *RUNTIME, *COMMON_TAIL)


def _only(*fields: Field) -> tuple[Field, ...]:
    return tuple(fields)


VERBS: dict[str, Verb] = {
    "generate": Verb("generate", "Style + lyrics -> score -> tokens -> latents -> song. Runs every stage; "
                     "--from/--until narrow it, --resume skips the fresh ones.", PIPELINE, examples=(
        'yue generate --style "synthwave pop, female vocal" --lyrics @song.txt',
        "yue generate --workspace runs/neon --resume --tokens-temperature 0.9   # redo tokens, synth, decode",
        'yue generate --style "..." --lyrics @l.txt --until plan --abc-hook \'claude -p "$(yue brief)"\'',
    )),
    "plan": Verb("plan", "Stage 1 only: write the score (score.abc) for a style and lyrics.",
                 _only(WORKSPACE, PROMPT, LYRICS, COT, ABC, SEED, FORCE, *PLAN_SAMPLING, *HOOK, *RUNTIME, *COMMON_TAIL),
                 examples=('yue plan --workspace runs/neon --style "dark folk" --lyrics @l.txt',)),
    "tokens": Verb("tokens", "Stage 2 only: perform the workspace's plan as semantic tokens.",
                   _only(WORKSPACE, SEED, DURATION, MAX_DURATION, FORCE, *TOKENS_SAMPLING, *RUNTIME, *COMMON_TAIL)),
    "synth": Verb("synth", "Stage 3 only: flow-match acoustic latents from the workspace's tokens.",
                  _only(WORKSPACE, SEED, FORCE, *SYNTH, *RUNTIME, *COMMON_TAIL),
                  examples=("yue synth --workspace runs/neon --synth-seed 7 --steps 48   # same performance, new detail",)),
    "decode": Verb("decode", "Stage 4 only: decode the workspace's latents to audio.",
                   _only(WORKSPACE, OUTPUT, FORMAT, FORCE, *DECODE, *RUNTIME, *COMMON_TAIL),
                   examples=("yue decode --workspace runs/neon --vae legacy --decode full -o neon-legacy.flac",)),
    "render": Verb("render", "Perform a score: the workspace's score.abc (or --abc). With --bars, repaint only "
                   "those bars and keep the rest; with --extend, continue the song.",
                   (WORKSPACE, OUTPUT, FORMAT, FORCE, ABC, PROMPT, LYRICS, COT, SEED, DURATION, MAX_DURATION,
                    *RENDER, *TOKENS_SAMPLING, *SYNTH, *DECODE, *HOOK, *RUNTIME, *COMMON_TAIL),
                   examples=(
                       "yue render --workspace runs/neon                      # after editing runs/neon/score.abc",
                       "yue render --workspace runs/neon --bars 17-24         # repaint the second chorus only",
                       "yue render --workspace runs/neon --extend 30 --lyrics @longer.txt",
                   )),
    "remix": Verb("remix", "One shot: take a song (an audio file or a workspace), change the prompt, re-enter "
                  "the pipeline at --from. Audio is transcribed with SheetSage2 first.",
                  (F("input", "path", "Audio file, or a yue workspace", group="Song"),
                   WORKSPACE, OUTPUT, FORMAT, FORCE, FROM, PROMPT, LYRICS, COT, SEED, DURATION, MAX_DURATION,
                   *TOKENS_SAMPLING, *SYNTH, *DECODE, *HOOK, *TRANSCRIBE, *RUNTIME, *COMMON_TAIL),
                  notes="--from for a workspace input: plan = new score, tokens (default) = same score new "
                        "performance, synth = same performance new sound (--strength), decode = new decoder.\n"
                        "Audio input: transcribe -> score -> perform; --strength also starts the ODE from the "
                        "audio itself (EXPERIMENTAL).",
                  examples=(
                      'yue remix -i runs/neon --style "acoustic jazz trio, male vocal"',
                      'yue remix -i song.mp3 --lyrics @words.txt --style "heavy metal" --cot melody',
                      'yue remix -i runs/neon --from synth --strength 0.6 --synth-seed 3',
                  )),
    "refine": Verb("refine", "Run an external tool on a workspace's score, validate it, then (optionally) render.",
                   (WORKSPACE, F("with", "str", "Shell command; it edits $YUE_ABC in place",
                                 group="Score hook"),
                    F("render", "bool", "Render the refined score afterwards", group="Score hook"),
                    F("validate", "bool", "Refuse a score the native ABC dialect rejects (default on)", group="Score hook"),
                    HOOK[1], *RUNTIME, *LOCAL_TAIL),
                   notes="The command runs in the workspace with:\n"
                         "  YUE_WORKSPACE  YUE_ABC (edit in place)  YUE_ABC_ORIGINAL (read-only)\n"
                         "  YUE_STYLE_FILE  YUE_LYRICS_FILE  YUE_BRIEF (the upstream edit brief)  YUE_JOB",
                   examples=(
                       "yue refine --workspace runs/neon --with 'claude -p \"Read $YUE_BRIEF. Reharmonize $YUE_ABC "
                       "as modern jazz. Edit it in place.\" --allowedTools Read,Edit' --render",
                   )),
    "transcribe": Verb("transcribe", "Audio -> score.abc + MIDI + annotations (SheetSage2, its own environment).",
                       (F("input", "path", "Audio file", group="Transcription"),
                        WORKSPACE, *TRANSCRIBE, *COMMON_TAIL)),
    "runpod setup": Verb("runpod setup", "Deploy yue to RunPod serverless from an API key, step by step (resumable).",
                         (F("api_key", "str", "RunPod API key (default: $RUNPOD_API_KEY)", group="RunPod"),
                          F("image", "str", "Worker image (default ghcr.io/<gh user>/yue:<HEAD sha>)", group="RunPod"),
                          F("public_image", "bool", "The image is public: skip the registry credential", group="RunPod"),
                          F("registry_user", "str", "GHCR user for a private image (default: gh user)", group="RunPod"),
                          F("registry_token", "str", "GHCR token with read:packages (default: gh auth token)", group="RunPod"),
                          F("gpu", "list", "GPU type id, repeatable, in preference order", group="RunPod"),
                          F("data_center", "str", "Data center id (default: best current stock for --gpu)", group="RunPod"),
                          F("volume_gb", "int", "Network volume size (default 20)", minimum=10, group="RunPod"),
                          F("no_volume", "bool", "No network volume: weights download on every cold start", group="RunPod"),
                          F("no_prime", "bool", "Do not pre-download the weights onto the volume", group="RunPod"),
                          F("max_workers", "int", "Upper bound on parallel GPUs (default 1)", minimum=1, group="RunPod"),
                          F("idle_timeout", "int", "Seconds a worker idles before scaling to zero (default 5)",
                            minimum=1, group="RunPod"),
                          F("timeout_minutes", "float", "Hard cap per job, in minutes (default 10)", minimum=1,
                            maximum=1440, group="RunPod"),
                          F("wait_image", "bool", "Wait for the GitHub Actions build to push the image", group="RunPod"),
                          F("yes", "bool", "Do not ask before the steps that cost money", group="RunPod"),
                          OUTPUT_FORMAT, DEBUG),
                         notes="Steps: key -> image -> registry credential -> network volume (paid) -> template -> "
                               "endpoint (scale to zero) -> prime weights (paid job). State: ~/.config/yue/runpod.json"),
    "runpod status": Verb("runpod status", "Show the configured endpoint, its workers and queue, and the volume.",
                          (F("api_key", "str", "RunPod API key (default: $RUNPOD_API_KEY)"), OUTPUT_FORMAT, DEBUG)),
    "runpod teardown": Verb("runpod teardown", "Delete the endpoint and template (and with --volume, the weights).",
                            (F("api_key", "str", "RunPod API key (default: $RUNPOD_API_KEY)"),
                             F("volume", "bool", "Also delete the network volume and registry credential"),
                             F("yes", "bool", "Do not ask"), OUTPUT_FORMAT, DEBUG)),
    "import": Verb("import", "Turn an upstream run (`yue2 generate` / SongResult.save_artifacts) into a workspace.",
                   (F("input", "path", "Upstream artifact directory (has request.json, plan.json, semantic.npy)"),
                    WORKSPACE, OUTPUT_FORMAT, DEBUG)),
    "status": Verb("status", "Show a workspace's stages, what is fresh, and the score/token timing.",
                   (WORKSPACE, OUTPUT_FORMAT, DEBUG)),
    "abc-inspect": Verb("abc-inspect", "Validate a score in the native dialect and print its events.",
                        (F("input", "path", "Score file"), OUTPUT_FORMAT, DEBUG)),
    "abc-strip": Verb("abc-strip", "Remove chord symbols (for --cot melody), proving the melody is unchanged.",
                      (F("input", "path", "Score file"), F("output", "path", "New score file"),
                       F("keep_voice", "enum", "Keep both melodies or silence one", choices=("both", "Vocal", "Ins")),
                       OUTPUT_FORMAT, DEBUG)),
    "abc-compare": Verb("abc-compare", "Compare two scores' sounding notes and meter (exit 1 on a difference).",
                        (F("before", "path", "Original score"), F("after", "path", "Edited score"),
                         F("voices", "enum", "Which voices", choices=("both", "Vocal", "Ins")),
                         F("allow_tempo_change", "bool", "Do not count a tempo change"), OUTPUT_FORMAT, DEBUG)),
    "spec": Verb("spec", "Print every verb and switch as JSON: the single source other front-ends (snd yue) generate from.",
                 (F("output_format", "enum", "json (default) or stream-json", choices=("json", "stream-json")),)),
    "brief": Verb("brief", "Print the upstream score-editing brief (for an agent doing --abc-hook work).", ()),
    "doctor": Verb("doctor", "Report devices, backends, installed versions and cached weights.",
                   (F("device", "str", "auto | cuda | cuda:N | mps | cpu"), OUTPUT_FORMAT, DEBUG)),
}

STAGE_VERBS = {"plan": "plan", "tokens": "tokens", "synth": "synth", "decode": "decode"}

# Fields persisted in job.json -- the knobs, not the one-off actions or the texts
# (style/lyrics/score live as files in the workspace).
NOT_PERSISTED = {"workspace", "output", "resume", "force", "from", "until", "output_format", "quiet",
                 "debug", "style", "lyrics", "abc", "input", "with", "render", "validate", "bars", "dry_run",
                 "extend", "source", "abc_hook", "hook_timeout", "init_audio", "strength", "remote", "remote_fetch"}

# run-events v1 -- a JSON Lines event stream for long-running CLIs

A CLI invoked with `--output-format stream-json` writes one JSON object per line
to stdout, describing what the run is doing: which tasks exist, which started,
how far along they are, what they logged, which files they produced, and how the
run ended. A wrapper (a UI, an agent, a dashboard, CI) can follow a run without
parsing human text.

The spec is tool-agnostic. `yue` implements it first (a local 4-stage pipeline);
`img`, `vid` and `snd` (remote queue jobs) are meant to reuse it unchanged.

Researched 2026-09-17 against primary docs and source. Every claim in Section 1
links to where it was read.

---

## 1. Prior art

### 1.1 Survey

| Tool | Flag | Discriminator | Notable ideas | Source |
|---|---|---|---|---|
| Claude Code | `-p --output-format stream-json` (also `json`, and `--input-format stream-json` for input) | `type` + `subtype` (`system`/`init`, `assistant`, `user`, `result`/`success`, `result`/`error_max_turns`, ...) | First event is `system/init` with an open `capabilities` array ("ignore values you don't recognize"). "The last line of the stream is a `result` message" with `duration_ms`, `total_cost_usd`, `is_error`, `num_turns`, `session_id`, `usage`, `errors[]`. Every message carries `uuid` + `session_id`. Nesting via `parent_tool_use_id`. `task_started` / `task_progress` / `task_updated` (status `pending\|running\|completed\|failed\|killed`, merge a `patch` into a map keyed by `task_id`). `tool_progress` heartbeat every 30s so a slow tool is distinguishable from a stall. `timestamp` is "display only, don't order messages by it". Invalid flag -> stderr before the run starts; a failure inside the run -> a result on stdout. SIGTERM exits 143 with no result. Retry events carry a machine `error` category with "handle a value you don't recognize the way you handle `unknown`". | https://code.claude.com/docs/en/headless, https://code.claude.com/docs/en/agent-sdk/typescript, https://code.claude.com/docs/en/cli-reference |
| OpenAI Codex CLI | `codex exec --json` | `type`, dotted `noun.verb`: `thread.started`, `turn.started`, `turn.completed`, `turn.failed`, `item.started`, `item.updated`, `item.completed`, `error` | Progress goes to stderr, stdout is the JSONL stream. Items have a stable `id` and a `status` (`in_progress`, `completed`, `failed`, `declined`) repeated across started/updated/completed. `turn.completed` carries token `usage`. Fatal error is a plain `{message}`, no code. | https://learn.chatgpt.com/docs/non-interactive-mode, https://github.com/openai/codex/blob/main/codex-rs/exec/src/exec_events.rs |
| Go | `go test -json` (test2json), `go build -json` | `Action`: `start run pause cont pass bench fail output skip`; builds add `build-output`, `build-fail` | "Every JSON stream begins with a `start` event." Concatenating all `Output` fields reproduces the exact text output. Test and build events interleave on one stream and are told apart by `Action`. `go help buildjson`: "there may also be non-JSON error text on standard error ... Typically, this indicates an early, serious error. Consumers should be robust to this." | https://pkg.go.dev/cmd/test2json, https://pkg.go.dev/cmd/go (`go help buildjson`) |
| Cargo | `--message-format json` | `reason`: `compiler-message`, `compiler-artifact`, `build-script-executed`, `build-finished` | `compiler-artifact` lists `filenames` and a `fresh` boolean (up-to-date, nothing rebuilt). `build-finished` carries `success`. "Tools should ignore unknown fields and message types they don't recognize." Cargo also tells consumers to only parse lines starting with `{`, because other tools write to the same stdout. | https://doc.rust-lang.org/cargo/reference/external-tools.html |
| Terraform | `-json` | `type` (`version`, `apply_start`, `apply_progress`, `apply_complete`, `apply_errored`, `change_summary`, `diagnostic`, ...) | Common fields on every line: `@level`, `@message`, `@module`, `@timestamp`. First message is `type: version` with a `ui` schema version; minor bump = backward compatible, major = breaking. Consumers must ignore unrecognised properties and, for an unknown `type`, "present at least the `@message`". `apply_progress` carries `elapsed_seconds`, i.e. indeterminate progress. | https://developer.hashicorp.com/terraform/internals/machine-readable-ui |
| Bazel | Build Event Protocol (`--build_event_json_file`) | `id` (a typed `BuildEventId`) | `children` announces events before they are posted: "all events, except for the first event, are announced by a previous event", and on completion "all announced events will have been posted". An `Aborted` payload (reasons incl. `USER_INTERRUPTED`, `TIME_OUT`, `SKIPPED`) replaces an announced event that will never come. `last_message: true` marks the final event. `BuildFinished` carries the exit code. `cached_locally` on test results. | https://bazel.build/remote/bep, https://bazel.build/remote/bep-glossary, https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto |
| Docker BuildKit | `docker buildx build --progress=rawjson`, `BUILDKIT_PROGRESS=rawjson` | none; each line is a `SolveStatus` batch of `vertexes`, `statuses`, `logs`, `warnings` | Vertex = task: `digest` (id), `inputs` (parent/dependency ids), `started`, `completed`, `cached`, `error`. `VertexStatus` = progress: absolute `current` + optional `total`, `timestamp`. Logs are per-vertex with a `stream` number. | https://docs.docker.com/reference/cli/docker/buildx/build/, https://docs.docker.com/build/building/variables/, https://github.com/moby/buildkit/blob/master/client/graph.go |
| ffmpeg | `-progress pipe:1`, `-stats_period` | none; `key=value` lines | Blocks of absolute values written periodically (default every 0.5s); the last key of each block is `progress=continue` or `progress=end`, an explicit terminal marker. | https://ffmpeg.org/ffmpeg.html |
| TAP 14 | n/a (text protocol) | line shape (`ok`, `not ok`, `1..N`, `Bail out!`) | `TAP version 14` first line. A plan `1..N` announces the count, before or after the tests. `# SKIP` and `# TODO` directives. Subtests by 4-space indent. Unknown lines are ignored by harnesses. | https://testanything.org/tap-version-14-specification.html |
| pytest-reportlog | `--report-log=FILE` | `$report_type`: `SessionStart`, `CollectReport`, `TestReport`, `SessionFinish` | JSONL, flushed per object. `SessionFinish` carries `exitstatus`, tying the stream to the process exit code. | https://pypi.org/project/pytest-reportlog/ |
| Node test runner | `--test-reporter`, `TestsStream` events | `type`: `test:enqueue`, `test:dequeue`, `test:start`, `test:pass`, `test:fail`, `test:plan`, `test:diagnostic`, `test:complete`, `test:summary`, `test:interrupted`, ... | `noun:verb` naming. Enqueue (declared) vs dequeue (execution begins). `testId` stable across events, `parentId` added "to track lineage when concurrent siblings at the same nesting level interleave". On SIGINT: `test:interrupted`, then "the buffered spine never flushes, so neither the finale ... nor those tests' own results are emitted" (a counter-example). | https://nodejs.org/api/test.html |
| OpenTelemetry traces | n/a (model) | n/a | Span = `name`, `span_id`, `parent_id` ("empty for root spans"), start/end time, `attributes`, point-in-time span `events`, `status` `Unset\|Error\|Ok`. A tree encoded as a flat set of records with parent pointers. | https://opentelemetry.io/docs/concepts/signals/traces/ |
| GitHub Actions | workflow commands | `::name params::message` in stdout | `::error file=,line=,title=::msg`, `::group::`/`::endgroup::`, `::stop-commands::`. In-band commands mixed into human stdout need an escape hatch (`stop-commands`) exactly because the channel is shared. | https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands |
| Turborepo | `turbo run --summarize`, `--output-logs` | n/a | Run summary JSON in `.turbo/runs` with executed tasks, timings and hashes; `--output-logs new-only` shows only cache misses. Cache hits are a reported task outcome. | https://turborepo.dev/docs/reference/run |
| JSON Lines | n/a | n/a | UTF-8 without BOM, one JSON value per line, `\n` terminator (`\r\n` tolerated), blank lines invalid, `.jsonl`. | https://jsonlines.org/ |
| bash | n/a | n/a | "When a command terminates on a fatal signal whose number is N, bash uses the value of 128+N as the exit status" (SIGINT -> 130, SIGTERM -> 143). | `man bash`, https://www.gnu.org/software/bash/manual/bash.html#Exit-Status |

### 1.2 Lessons

| # | Lesson | Where it comes from |
|---|---|---|
| L1 | **One discriminator field on every line, one flat vocabulary, a version on every line.** A consumer switches on a single key. The version belongs on each line because consumers attach late, `tail -f`, and grep; a version sent once in a header is lost to all of them. | `type` in Claude/Codex/Terraform/Node, `reason` in Cargo, `Action` in Go. Terraform sends `ui` only on the first line, which a late attacher never sees. |
| L2 | **Evolve additively; consumers ignore what they do not know.** Unknown fields, unknown event types and unknown enum values are skipped, and an unknown event with a human `message` can still be shown. | Cargo ("ignore unknown fields and message types"), Terraform (ignore unknown properties, show `@message` for unknown types), Claude (`capabilities` is an open set; unknown error categories map to `unknown`). |
| L3 | **Fixed first line and fixed last line.** The first event identifies the run; the last event is the result, carries the exit code, and is self-contained so a consumer that reads only the last line gets everything. | Go (`start` first), Terraform (`version` first), Claude (`system/init` first, `result` last with cost/duration/session), BEP (`last_message`, `BuildFinished` exit code), pytest-reportlog (`SessionFinish.exitstatus`), ffmpeg (`progress=end`). |
| L4 | **Declare before start.** Announcing the task list up front lets a UI draw every row immediately and lets a consumer detect a task that never finished. Late declarations must still precede use. | BEP children announcement and its "all announced events will have been posted" guarantee, TAP plan `1..N`, Node `test:enqueue` before `test:dequeue`. |
| L5 | **Nesting by parent id in a flat stream.** Hierarchy is a `parent` pointer on the task; depth is derived. Pointers survive interleaving of concurrent siblings; indentation or ordering does not. | OTel `parent_id`, Node `parentId` (added for interleaving), Claude `parent_tool_use_id`, BuildKit `inputs`. TAP's indentation-based subtests are the fragile alternative. |
| L6 | **Progress is absolute, with an optional total; liveness is a heartbeat.** `completed` + `total` (determinate) or no total (indeterminate). Absolute values make throttling and dropped lines harmless. A heartbeat with elapsed time separates "slow" from "stuck". | BuildKit `current`/`total`, ffmpeg absolute key=value blocks, Terraform `apply_progress.elapsed_seconds`, Claude `tool_progress` heartbeat every 30s. |
| L7 | **Reused work is a reported outcome, with its outputs.** "Skipped because fresh/cached" is a terminal state with a reason, and its artifacts are still announced, because the consumer needs the paths either way. Silently omitting cached work breaks declare-before-start. | Cargo `fresh` on `compiler-artifact` (files still listed), BuildKit `cached`, BEP `cached_locally`; BEP's `Fetch` event that "does not appear" when cached is the omission to avoid. Turborepo reports hits vs misses. |
| L8 | **Interruption still ends the stream properly.** On SIGINT every open task gets a terminal event and a result is written, then the process exits 128+2. Buffering events until the end loses them exactly when they matter. | BEP `Aborted` with `USER_INTERRUPTED` replaces announced events. Node is the counter-example: on SIGINT its buffered events never flush. Claude: SIGINT ends the turn, SIGTERM exits 143 "and records no result". bash 128+N. |
| L9 | **stdout is the protocol, stderr is for humans and early crashes.** In stream mode nothing but events reaches stdout. Consumers never parse stderr but must tolerate text on it. A missing result at EOF means the process died. | Codex (progress on stderr, JSONL on stdout), Go ("non-JSON error text on standard error ... consumers should be robust"), Claude (invalid flag -> stderr, in-run failure -> result on stdout). Cargo's "only parse lines starting with `{`" and GitHub's `::stop-commands::` are what a shared channel costs. |
| L10 | **Errors carry a stable machine code beside the message; order by sequence, display by timestamp.** A code is an open enum. Wall-clock timestamps are not ordering keys; a monotonic `seq` is, and a gap in it reveals a lost line. | Claude error categories and "don't order messages by `timestamp`", BEP `AbortReason` enum, Codex (message only, no code: the gap to avoid), Terraform `@level` + `@timestamp`. |

---

## 2. The spec: run-events v1

The key words MUST, SHOULD and MAY are used in their RFC 2119 sense.

### 2.1 Transport

1. Enabled by `--output-format stream-json`. Tools MAY also offer `json` (only the
   final result object, see 2.9) and `text` (human output, the default).
2. Output is JSON Lines on **stdout**: UTF-8 without BOM, exactly one JSON object
   per line, terminated by `\n`, no pretty-printing, no blank lines.
3. The producer MUST flush after every line.
4. Nothing else is written to stdout while in stream mode. Human progress and
   spinners are suppressed; warnings, tracebacks and third-party chatter go to
   stderr. Implementation note: output from libraries you do not control
   (`print`, `console.log`) will land on stdout unless you redirect it. In Python,
   keep a private handle to the real fd 1 (`os.dup(1)`) for events, then
   `os.dup2(2, 1)` so everything else goes to stderr. In Bun/Node, route
   `console.log` to stderr and write events through a dedicated writer.
5. Numbers MUST be finite. `NaN` and `Infinity` are not JSON (Python: `json.dumps(..., allow_nan=False)`).
6. Lines SHOULD stay under 64 KiB. Large payloads (audio, images, logs) go to
   files and are referenced by an `artifact` event, never inlined as base64.
7. Consumers MUST NOT parse stderr, and MUST tolerate non-JSON text there
   (early fatal errors can occur before the stream starts).

### 2.2 Envelope

Every event carries:

| Field | Type | Req | Meaning |
|---|---|---|---|
| `v` | integer | yes | Major schema version. `1` for this spec. Bumped only for breaking changes. |
| `seq` | integer | yes | 1 for the first line, then +1 per line, no gaps, no repeats. The ordering key. A gap means a lost line. |
| `ts` | string | yes | RFC 3339 UTC wall-clock time with milliseconds, e.g. `2026-09-17T14:02:00.012Z`. For display and correlation only; order by `seq`. |
| `type` | string | yes | Event type, `noun:verb` or a bare noun. See 2.3. |
| `message` | string | no | One human-readable line summarising the event. Any event MAY carry it; a consumer that does not know a `type` SHOULD display `message` if present. |

Durations are integers in milliseconds, measured with a monotonic clock, in
fields named `*_ms`.

### 2.3 Event vocabulary

| `type` | Purpose | Required fields (beyond envelope) | Optional fields |
|---|---|---|---|
| `run:start` | First line. Identifies tool and invocation. | `tool`, `command` | `version`, `run_id`, `capabilities`, `data` |
| `task:declare` | Announces one or more tasks. | `tasks[]` (each: `id`) | per task: `description`, `kind`, `parent`, `depends_on` |
| `task:queued` | Task is waiting for capacity (remote queue, slot limiter). Repeatable. | `id` | `position`, `remote_id` |
| `task:start` | Task began executing. | `id` | `remote_id` |
| `task:progress` | Progress or heartbeat of a running task. | `id` | `completed`, `total`, `unit`, `estimate`, `rate`, `eta_ms`, `elapsed_ms` |
| `log` | A log line, optionally tied to a task. | `level`, `message` | `id`, `code`, `stream`, `data` |
| `artifact` | A file or URL is complete and usable. | `path` or `url` | `id`, `kind`, `mime`, `bytes`, `sha256`, `role`, `reused`, `data` |
| `task:skip` | Terminal: the task will not run. | `id`, `reason` | `data` |
| `task:end` | Terminal: the task ran (or was cancelled). | `id`, `status` | `duration_ms`, `reason`, `error`, `remote_id`, `detached`, `data` |
| `result` | Last line. Outcome of the whole run. | `status`, `exit_code` | `duration_ms`, `error`, `artifacts`, `cost`, `usage`, `summary`, `data` |

`data` is always an object of tool-specific fields. Standard fields never move
into `data`, and tool-specific fields never appear at the top level of an event,
so a future version of this spec can add top-level fields without collisions.

Examples of each:

```json
{"v":1,"seq":1,"ts":"2026-09-17T14:02:00.012Z","type":"run:start","tool":"yue","version":"0.1.0","command":"render","run_id":"9b2e4c1a","message":"yue render /Users/me/songs/lighthouse"}
```

```json
{"v":1,"seq":2,"ts":"2026-09-17T14:02:00.020Z","type":"task:declare","tasks":[{"id":"synth","kind":"stage","description":"Acoustic synthesis"},{"id":"synth.chunk-1","kind":"chunk","description":"Chunk 1 of 3","parent":"synth"}]}
```

```json
{"v":1,"seq":5,"ts":"2026-09-17T14:10:03.400Z","type":"task:queued","id":"render","position":3,"remote_id":"a1b2c3d4-5e6f-4a1b-9c2d-3e4f5a6b7c8d"}
```

```json
{"v":1,"seq":6,"ts":"2026-09-17T14:02:00.140Z","type":"task:start","id":"tokens","message":"Generating performance tokens"}
```

```json
{"v":1,"seq":8,"ts":"2026-09-17T14:02:51.450Z","type":"task:progress","id":"tokens","completed":900,"total":2250,"unit":"tokens","estimate":true,"rate":45.1,"eta_ms":29900}
```

```json
{"v":1,"seq":13,"ts":"2026-09-17T14:03:11.900Z","type":"log","level":"warn","id":"synth","code":"yue.attention_fallback","message":"flash attention unavailable on mps, using sdpa"}
```

```json
{"v":1,"seq":22,"ts":"2026-09-17T14:06:11.900Z","type":"artifact","id":"decode","path":"/Users/me/songs/lighthouse/song.flac","kind":"audio","mime":"audio/flac","bytes":31744012,"role":"final"}
```

```json
{"v":1,"seq":4,"ts":"2026-09-17T14:02:00.110Z","type":"task:skip","id":"plan","reason":"fresh","message":"plan is fresh, reusing 1-plan/"}
```

```json
{"v":1,"seq":11,"ts":"2026-09-17T14:03:11.540Z","type":"task:end","id":"tokens","status":"succeeded","duration_ms":71400}
```

```json
{"v":1,"seq":24,"ts":"2026-09-17T14:06:12.050Z","type":"result","status":"succeeded","exit_code":0,"duration_ms":252038,"artifacts":[{"id":"decode","path":"/Users/me/songs/lighthouse/song.flac","kind":"audio","mime":"audio/flac","bytes":31744012,"role":"final"}],"summary":{"succeeded":3,"skipped":1}}
```

### 2.4 Field reference

**`run:start`**

| Field | Type | Meaning |
|---|---|---|
| `tool` | string | Program name (`yue`, `vid`). |
| `command` | string | The verb (`render`, `generate`). |
| `version` | string | Tool version. |
| `run_id` | string | Unique per invocation. Useful when several runs are multiplexed into one log. |
| `capabilities` | string[] | Open set of optional behaviours this producer implements (e.g. `"heartbeat"`, `"cost"`). Consumers ignore values they do not know. |
| `data` | object | Tool-specific context (workspace path, model). MUST NOT contain secrets or full argv (argv routinely holds tokens and prompts). |

**Task declaration** (items of `task:declare.tasks`)

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique within the run, stable for its lifetime. `[A-Za-z0-9][A-Za-z0-9._:/-]*`, max 128 chars. Deterministic ids (`tokens`, `synth.chunk-3`) are preferred over random ones. |
| `description` | string | Short human label for the row ("Acoustic synthesis"). |
| `kind` | string | Open category for icons and grouping: `stage`, `chunk`, `submit`, `render`, `download`, ... |
| `parent` | string | Id of an already-declared task (or one earlier in the same `tasks` array). Absent = top level. |
| `depends_on` | string[] | Ids this task waits for. Informational; lets a UI draw a DAG. |

**`task:progress`**

| Field | Type | Meaning |
|---|---|---|
| `completed` | number >= 0 | Absolute amount done, in `unit`. Never a delta. |
| `total` | number >= 0 | Absolute amount expected. Absent = indeterminate. |
| `unit` | string | Open set: `tokens`, `steps`, `chunks`, `bytes`, `frames`, `items`, `percent`. With `percent`, `total` is 100. |
| `estimate` | boolean | `true` when `total` is a guess or an upper bound and may be revised. |
| `rate` | number | `unit` per second, producer-computed. |
| `eta_ms` | integer | Producer's estimate of time remaining. |
| `elapsed_ms` | integer | Time since `task:start`. |

A `task:progress` with neither `completed` nor `total` is a **heartbeat**.

**`log`**

| Field | Type | Meaning |
|---|---|---|
| `level` | `debug` \| `info` \| `warn` \| `error` | Severity. An `error` log does not change any task state; only `task:end` and `result` do. |
| `id` | string | Task this line belongs to. Absent = run-level. |
| `code` | string | Machine code for a notable condition (see 2.8 for naming). |
| `stream` | `stdout` \| `stderr` | Set when the line is captured output of a child process. |

**`artifact`**

| Field | Type | Meaning |
|---|---|---|
| `path` | string | Absolute local path. |
| `url` | string | Remote location (may accompany `path` when a download was saved). |
| `id` | string | Task that produced (or reused) it. Absent = run-level. |
| `kind` | string | Open set: `audio`, `image`, `video`, `text`, `data`, `model`, `other`. |
| `mime` | string | Media type if known. |
| `bytes` | integer | Size. |
| `sha256` | string | Hex digest, when cheap to compute. |
| `role` | `final` \| `intermediate` | `final` = what the user asked for. Default `intermediate`. |
| `reused` | boolean | `true` when the file came from a previous run (fresh/cached). |

**`task:end`**

| Field | Type | Meaning |
|---|---|---|
| `status` | `succeeded` \| `failed` \| `cancelled` | Terminal status. |
| `duration_ms` | integer | From `task:start` (absent if it never started). |
| `reason` | string | For `cancelled`: `interrupted`, `upstream_failed`, `timeout`, or a tool value. |
| `error` | error object | Required when `status` is `failed`. |
| `remote_id` | string | Handle on remote work (job id, task UUID). |
| `detached` | boolean | `true` when work continues outside this process after cancellation (a submitted remote render that cannot be cancelled, and is still billed). |

**`task:skip`**

`reason` is an open set with these defined values:

| `reason` | Meaning |
|---|---|
| `fresh` | Output from a previous run is up to date for the current inputs. |
| `cached` | Output was taken from a cache (local or remote). |
| `not_selected` | Outside the requested range (`--from`, `--until`, filters). |
| `dry_run` | The run was a dry run. |

Consumers treat an unknown `reason` as a generic skip.

**Error object** (in `task:end.error` and `result.error`)

| Field | Type | Req | Meaning |
|---|---|---|---|
| `code` | string | yes | Stable machine code (2.7). |
| `message` | string | yes | Human explanation. |
| `retryable` | boolean | no | `true` if re-running the same command may succeed. |
| `hints` | string[] | no | Suggested fixes, one per entry. |
| `data` | object | no | Details (HTTP status, upstream body excerpt, offending field). |

**`result`**

| Field | Type | Meaning |
|---|---|---|
| `status` | `succeeded` \| `failed` \| `cancelled` | Outcome of the run. |
| `exit_code` | integer | MUST equal the process exit status. |
| `duration_ms` | integer | Whole run. |
| `error` | error object | Required when `status` is `failed` or `cancelled`. |
| `artifacts` | artifact[] | Every `final` artifact again, without envelope fields, so the result stands alone. |
| `cost` | object | `{amount, currency, basis}`; `basis` is `billed` (read from the provider) or `quoted` (computed from a rate card). Omit when unknown; never report 0 for unknown. |
| `usage` | object | Tool-specific counters (tokens, seconds rendered). |
| `summary` | object | Count of tasks per terminal state, e.g. `{"succeeded":3,"skipped":1}`. |
| `data` | object | Tool-specific outcome (seed used, workspace path). |

### 2.5 Ordering rules

1. `run:start` is the first line of every stream.
2. `result` is the last line of every stream. Nothing follows it.
3. Every task id used by any event MUST have been declared by an earlier
   `task:declare`. `task:declare` MAY appear more than once (tasks discovered at
   runtime, e.g. a chunk count known only after planning), always before result.
4. A task's `parent` MUST be declared before or in the same `task:declare`.
5. Every declared task reaches exactly one terminal event (`task:skip` or
   `task:end`) before `result`. A consumer can therefore list unfinished tasks at
   any moment, and a producer bug shows up as a task with no terminal event.
6. A child SHOULD reach its terminal event before its parent does.
7. An `artifact` referencing a task MUST precede that task's terminal event. A
   reused output is announced with `reused: true` immediately before `task:skip`.
8. `seq` is strictly increasing by 1.

### 2.6 Task states

```
 pending --task:queued--> queued --task:start--> running --task:end--> succeeded | failed | cancelled
    |                      ^  |                     |
    |                      |  +--task:end--> failed | cancelled
    |                      +------task:queued-------+
    +--task:start--> running
    +--task:skip---> skipped
    +--task:end----> cancelled
```

| From | Event | To |
|---|---|---|
| (none) | `task:declare` | `pending` |
| `pending`, `queued`, `running` | `task:queued` | `queued` |
| `pending`, `queued` | `task:start` | `running` |
| `running` | `task:progress` | `running` |
| `pending` | `task:skip` | `skipped` |
| `running` | `task:end` `succeeded` | `succeeded` |
| `queued`, `running` | `task:end` `failed` | `failed` |
| `pending`, `queued`, `running` | `task:end` `cancelled` | `cancelled` |

Terminal states: `succeeded`, `failed`, `cancelled`, `skipped`. Any other
transition is a producer bug. Consumers SHOULD ignore an event that would make
an illegal transition (and MAY log it) rather than abort.

Tasks that cannot run because an earlier task failed end as `cancelled` with
`reason: "upstream_failed"`. `skipped` is reserved for "this work was not
needed"; `cancelled` means "this work was needed and did not happen".

### 2.7 Progress semantics

1. **Determinate**: `completed` and `total`. A bar is `completed / total`.
2. **Indeterminate with a count**: `completed` only (e.g. tokens so far with no
   known end). Render a counter.
3. **Revisable total**: `estimate: true`. `total` may move up or down between
   events. The final `task:progress` before `task:end` SHOULD have
   `completed == total`.
4. **Heartbeat**: neither `completed` nor `total`, usually with `elapsed_ms` and a
   `message` ("loading weights", "provider reports no progress"). A running task
   SHOULD emit some event at least every 30 seconds; a consumer MAY flag a task
   silent for much longer as stalled.
5. `completed` never decreases between a `task:start` and the task's next
   `task:queued` or terminal event. A retry that restarts from zero goes back
   through `task:queued` and a fresh `task:start`, preceded by a `log` with a
   code saying why.
6. Throttle: at most 10 `task:progress` per second per task (a few per second is
   plenty). Always emit the final value. Because values are absolute, a
   consumer may also coalesce and keep only the latest.
7. Tasks do not aggregate. A parent's bar is whatever the producer emits for the
   parent; consumers do not sum children.

### 2.8 Errors, cancellation, exit codes

**Error codes** are stable strings. Generic codes any tool may use:

| `code` | Meaning |
|---|---|
| `usage` | Invalid arguments or combination of arguments. |
| `config` | Missing or invalid configuration (key, file, env var). |
| `auth` | Credentials rejected. |
| `not_found` | Input or remote resource does not exist. |
| `rejected` | Upstream refused the request (validation, content policy). |
| `upstream` | Upstream failed (5xx, provider error). |
| `rate_limited` | Upstream throttled and retries were exhausted. |
| `timeout` | A deadline passed. |
| `resource` | Local resource exhausted (memory, disk, device). |
| `interrupted` | The run was cancelled by a signal. |
| `internal` | Bug in the tool. |

Tool-specific codes are prefixed with the tool name and a dot
(`yue.attention_fallback`, `vid.envelope_dropped`). Consumers treat any unknown
code like `internal` for retry decisions and still show `message`.

**Cancellation.** On the first SIGINT (or SIGTERM) the producer:

1. stops starting new work,
2. emits `task:end` with `status: "cancelled"`, `reason: "interrupted"` for every
   task not yet terminal (running, queued and pending alike), with
   `detached: true` and `remote_id` on any task whose remote work keeps running,
3. emits `result` with `status: "cancelled"` and `error.code: "interrupted"`,
4. exits with 130 (SIGINT) or 143 (SIGTERM), 128 + signal number.

A second SIGINT MAY exit immediately without a result. A consumer wanting a
clean stop sends SIGINT once and waits; SIGKILL guarantees a truncated stream.

**Partial outputs.** An artifact event means the file is complete. Producers
write to a temporary name and rename, and never announce a partial file.

**Exit codes.**

| Exit | `result.status` | Notes |
|---|---|---|
| 0 | `succeeded` | Includes a run where every task was skipped. |
| 1 | `failed` | Any failure. |
| 2 | `failed` with `error.code: "usage"` | Recommended for usage errors. A tool with an established convention (e.g. `img`/`vid`/`snd` exit 1 on an unknown switch) MAY keep it; `error.code` is what consumers key on. |
| 130 / 143 | `cancelled` | SIGINT / SIGTERM. |

`result.exit_code` MUST equal the actual exit status.

**Failures before the stream exists.** If argument parsing fails before the tool
knows it is in stream mode, it writes plain text to stderr and exits non-zero
with no stream. Once `--output-format stream-json` has been recognised, even a
usage error produces `run:start` followed by `result`.

**Consumer rule for EOF.** Stdout closing without a `result` line means the
producer crashed or was killed. Use the process exit status, and mark every
non-terminal task as unknown.

### 2.9 Forward compatibility and versioning

1. Consumers MUST ignore unknown fields on known events.
2. Consumers MUST ignore unknown event types (displaying `message` if present).
3. Consumers MUST tolerate unknown values in open enums: task `kind`, skip
   `reason`, cancel `reason`, `unit`, artifact `kind`, error `code`,
   `capabilities`.
4. Closed enums, where a new value is a breaking change and bumps `v`: `task:end.status`,
   `result.status`, `log.level`, `artifact.role`.
5. New optional fields and new event types do not bump `v`. Removing or renaming
   a field, changing a type, or changing the meaning of a field does.
6. A consumer that receives a `v` it does not support SHOULD stop interpreting
   events and fall back to waiting for the process to exit.

**`--output-format json`** prints only the `result` object (the envelope fields
are optional there), once, at the end. It is the same shape so a consumer can
switch modes without a second parser.

### 2.10 Changes from the original sketch

The sketch was:

```jsonl
{"type":"task:declare","tasks":[{"id":"some-id","type":"render","description":"Pre process"}]}
{"type":"task:start","id":"some-id"}
{"type":"progress","id":"some-id","completed":{"stage":5,"total":12345}}
{"type":"log","message":"some message"}
```

| Sketch | v1 | Why |
|---|---|---|
| `task:declare`, `task:start`, `log`, `id`, `description` | kept | Sound. `noun:verb` naming matches Node's `test:*`, and declare-before-start is Bazel's announcement rule. |
| task `type: "render"` | `kind` | `type` is the event discriminator. Reusing the name one level down makes `jq '.type'` and `select(.type == ...)` filters ambiguous and invites bugs. |
| `progress` | `task:progress` | Every task-scoped event lives in the `task:` namespace; `progress` about something other than a task has no meaning in this model. |
| `completed: {stage, total}` | flat `completed`, `total`, `unit` | "completed" containing a "total" reads as a contradiction, and `stage` duplicates what `id` already says. Flat numbers are what BuildKit (`current`/`total`) and ffmpeg use, need no null-checking of a wrapper, and leave room for `unit`, `rate`, `eta_ms`. |
| no envelope | `v`, `seq`, `ts` on every line | Versioning, loss detection and ordering (L1, L10). |
| no end / no result | `task:end`, `task:skip`, `result` | Without terminal events a consumer cannot tell done from dead (L3, L8). |
| no skip | `task:skip` with `reason` | yue skips fresh stages constantly; it is an outcome, not an absence (L7). |
| `log` without level | `level` required | A UI must be able to hide `debug` and highlight `warn`. |

---

## 3. Worked examples

### 3.1 yue: 4 local stages, plan is fresh

`yue render --resume --output-format stream-json` on the workspace
`/Users/me/songs/lighthouse`.
The plan stage is reused; tokens generates with a revisable total (the token
budget is an upper bound, the song ends at its end token); synth runs 32 ODE
steps; decode writes chunks, then the stage audio and the exported song.

```jsonl
{"v":1,"seq":1,"ts":"2026-09-17T14:02:00.012Z","type":"run:start","tool":"yue","version":"0.1.0","command":"render","run_id":"9b2e4c1a","capabilities":["heartbeat"],"data":{"workspace":"/Users/me/songs/lighthouse"}}
{"v":1,"seq":2,"ts":"2026-09-17T14:02:00.020Z","type":"task:declare","tasks":[{"id":"plan","kind":"stage","description":"Plan score"},{"id":"tokens","kind":"stage","description":"Performance tokens","depends_on":["plan"]},{"id":"synth","kind":"stage","description":"Acoustic synthesis","depends_on":["tokens"]},{"id":"decode","kind":"stage","description":"Decode audio","depends_on":["synth"]}]}
{"v":1,"seq":3,"ts":"2026-09-17T14:02:00.100Z","type":"artifact","id":"plan","path":"/Users/me/songs/lighthouse/score.abc","kind":"text","mime":"text/vnd.abc","role":"intermediate","reused":true}
{"v":1,"seq":4,"ts":"2026-09-17T14:02:00.110Z","type":"task:skip","id":"plan","reason":"fresh","message":"plan is fresh, reusing 1-plan/"}
{"v":1,"seq":5,"ts":"2026-09-17T14:02:00.140Z","type":"task:start","id":"tokens","message":"Generating performance tokens"}
{"v":1,"seq":6,"ts":"2026-09-17T14:02:15.140Z","type":"task:progress","id":"tokens","elapsed_ms":15000,"message":"loading weights"}
{"v":1,"seq":7,"ts":"2026-09-17T14:02:31.500Z","type":"task:progress","id":"tokens","completed":0,"total":2250,"unit":"tokens","estimate":true,"elapsed_ms":31360}
{"v":1,"seq":8,"ts":"2026-09-17T14:02:51.450Z","type":"task:progress","id":"tokens","completed":900,"total":2250,"unit":"tokens","estimate":true,"rate":45.1,"eta_ms":29900,"elapsed_ms":51310}
{"v":1,"seq":9,"ts":"2026-09-17T14:03:11.300Z","type":"task:progress","id":"tokens","completed":2221,"total":2221,"unit":"tokens","elapsed_ms":71160}
{"v":1,"seq":10,"ts":"2026-09-17T14:03:11.500Z","type":"artifact","id":"tokens","path":"/Users/me/songs/lighthouse/2-tokens/semantic.npy","kind":"data","role":"intermediate","bytes":17896}
{"v":1,"seq":11,"ts":"2026-09-17T14:03:11.540Z","type":"task:end","id":"tokens","status":"succeeded","duration_ms":71400,"data":{"tokens":2221,"seed":184467}}
{"v":1,"seq":12,"ts":"2026-09-17T14:03:11.600Z","type":"task:start","id":"synth","message":"Synthesising latents (midpoint, 32 steps)"}
{"v":1,"seq":13,"ts":"2026-09-17T14:03:11.900Z","type":"log","level":"warn","id":"synth","code":"yue.attention_fallback","message":"flash attention unavailable on mps, using sdpa"}
{"v":1,"seq":14,"ts":"2026-09-17T14:04:28.000Z","type":"task:progress","id":"synth","completed":16,"total":32,"unit":"steps","rate":0.21,"eta_ms":76000,"elapsed_ms":76400}
{"v":1,"seq":15,"ts":"2026-09-17T14:05:44.300Z","type":"task:progress","id":"synth","completed":32,"total":32,"unit":"steps","elapsed_ms":152700}
{"v":1,"seq":16,"ts":"2026-09-17T14:05:44.500Z","type":"artifact","id":"synth","path":"/Users/me/songs/lighthouse/3-latents/latent.npy","kind":"data","role":"intermediate","bytes":1146976}
{"v":1,"seq":17,"ts":"2026-09-17T14:05:44.600Z","type":"task:end","id":"synth","status":"succeeded","duration_ms":153000}
{"v":1,"seq":18,"ts":"2026-09-17T14:05:44.700Z","type":"task:start","id":"decode","message":"Decoding audio (tiled)"}
{"v":1,"seq":19,"ts":"2026-09-17T14:05:56.000Z","type":"task:progress","id":"decode","completed":4,"total":9,"unit":"chunks","elapsed_ms":11300}
{"v":1,"seq":20,"ts":"2026-09-17T14:06:10.100Z","type":"task:progress","id":"decode","completed":9,"total":9,"unit":"chunks","elapsed_ms":25400}
{"v":1,"seq":21,"ts":"2026-09-17T14:06:10.600Z","type":"artifact","id":"decode","path":"/Users/me/songs/lighthouse/4-audio/audio.flac","kind":"audio","mime":"audio/flac","role":"intermediate","bytes":31744012}
{"v":1,"seq":22,"ts":"2026-09-17T14:06:11.900Z","type":"artifact","id":"decode","path":"/Users/me/songs/lighthouse/song.flac","kind":"audio","mime":"audio/flac","role":"final","bytes":31744012}
{"v":1,"seq":23,"ts":"2026-09-17T14:06:12.000Z","type":"task:end","id":"decode","status":"succeeded","duration_ms":27300}
{"v":1,"seq":24,"ts":"2026-09-17T14:06:12.050Z","type":"result","status":"succeeded","exit_code":0,"duration_ms":252038,"artifacts":[{"id":"decode","path":"/Users/me/songs/lighthouse/song.flac","kind":"audio","mime":"audio/flac","role":"final","bytes":31744012}],"summary":{"succeeded":3,"skipped":1},"data":{"workspace":"/Users/me/songs/lighthouse"}}
```

The same run failing in tokens (out of memory) ends like this instead; the
stages that never ran are `cancelled`, because they were needed:

```jsonl
{"v":1,"seq":9,"ts":"2026-09-17T14:02:58.010Z","type":"task:end","id":"tokens","status":"failed","duration_ms":57870,"error":{"code":"resource","message":"MPS backend out of memory (allocated 17.2 GB)","retryable":false,"hints":["Close other GPU-heavy apps","Lower --max-duration"]}}
{"v":1,"seq":10,"ts":"2026-09-17T14:02:58.020Z","type":"task:end","id":"synth","status":"cancelled","reason":"upstream_failed"}
{"v":1,"seq":11,"ts":"2026-09-17T14:02:58.030Z","type":"task:end","id":"decode","status":"cancelled","reason":"upstream_failed"}
{"v":1,"seq":12,"ts":"2026-09-17T14:02:58.040Z","type":"result","status":"failed","exit_code":1,"duration_ms":58028,"error":{"code":"resource","message":"tokens failed: MPS backend out of memory (allocated 17.2 GB)","retryable":false},"summary":{"skipped":1,"failed":1,"cancelled":2}}
```

### 3.2 Remote queue tool: submit, queue, render, download

`vid generate --prompt "..." --model rw-pvideo --duration 2 --output-format stream-json`.

```jsonl
{"v":1,"seq":1,"ts":"2026-09-17T14:10:02.000Z","type":"run:start","tool":"vid","version":"0.9.0","command":"generate","run_id":"c7d1e0f2","capabilities":["heartbeat","cost"],"data":{"model":"rw-pvideo"}}
{"v":1,"seq":2,"ts":"2026-09-17T14:10:02.010Z","type":"task:declare","tasks":[{"id":"submit","kind":"submit","description":"Submit job"},{"id":"render","kind":"render","description":"Render on Runware","depends_on":["submit"]},{"id":"download","kind":"download","description":"Download video","depends_on":["render"]}]}
{"v":1,"seq":3,"ts":"2026-09-17T14:10:02.020Z","type":"task:start","id":"submit"}
{"v":1,"seq":4,"ts":"2026-09-17T14:10:02.660Z","type":"task:end","id":"submit","status":"succeeded","duration_ms":640,"remote_id":"a1b2c3d4-5e6f-4a1b-9c2d-3e4f5a6b7c8d"}
{"v":1,"seq":5,"ts":"2026-09-17T14:10:03.400Z","type":"task:queued","id":"render","position":3,"remote_id":"a1b2c3d4-5e6f-4a1b-9c2d-3e4f5a6b7c8d"}
{"v":1,"seq":6,"ts":"2026-09-17T14:10:08.400Z","type":"task:queued","id":"render","position":1,"remote_id":"a1b2c3d4-5e6f-4a1b-9c2d-3e4f5a6b7c8d"}
{"v":1,"seq":7,"ts":"2026-09-17T14:10:11.000Z","type":"task:start","id":"render","remote_id":"a1b2c3d4-5e6f-4a1b-9c2d-3e4f5a6b7c8d"}
{"v":1,"seq":8,"ts":"2026-09-17T14:10:21.000Z","type":"task:progress","id":"render","completed":40,"total":100,"unit":"percent","elapsed_ms":10000}
{"v":1,"seq":9,"ts":"2026-09-17T14:10:51.000Z","type":"task:progress","id":"render","elapsed_ms":40000,"message":"provider reports no progress"}
{"v":1,"seq":10,"ts":"2026-09-17T14:10:58.000Z","type":"task:progress","id":"render","completed":100,"total":100,"unit":"percent","elapsed_ms":47000}
{"v":1,"seq":11,"ts":"2026-09-17T14:10:58.100Z","type":"task:end","id":"render","status":"succeeded","duration_ms":47100,"remote_id":"a1b2c3d4-5e6f-4a1b-9c2d-3e4f5a6b7c8d"}
{"v":1,"seq":12,"ts":"2026-09-17T14:10:58.200Z","type":"task:start","id":"download"}
{"v":1,"seq":13,"ts":"2026-09-17T14:10:58.700Z","type":"task:progress","id":"download","completed":1048576,"total":2811904,"unit":"bytes"}
{"v":1,"seq":14,"ts":"2026-09-17T14:10:59.300Z","type":"artifact","id":"download","path":"/Users/me/clips/rw-pvideo-20260917-141059.mp4","url":"https://vm.runware.ai/video/os/a1b2c3d4.mp4","kind":"video","mime":"video/mp4","bytes":2811904,"role":"final"}
{"v":1,"seq":15,"ts":"2026-09-17T14:10:59.310Z","type":"task:end","id":"download","status":"succeeded","duration_ms":1110}
{"v":1,"seq":16,"ts":"2026-09-17T14:10:59.320Z","type":"result","status":"succeeded","exit_code":0,"duration_ms":57320,"artifacts":[{"id":"download","path":"/Users/me/clips/rw-pvideo-20260917-141059.mp4","url":"https://vm.runware.ai/video/os/a1b2c3d4.mp4","kind":"video","mime":"video/mp4","bytes":2811904,"role":"final"}],"cost":{"amount":0.0306,"currency":"USD","basis":"billed"},"summary":{"succeeded":3}}
```

### 3.3 Remote tool interrupted during render

Ctrl-C after seq 8 above. The render cannot be cancelled upstream, so it is
marked `detached` and the result says so.

```jsonl
{"v":1,"seq":9,"ts":"2026-09-17T14:10:24.500Z","type":"task:end","id":"render","status":"cancelled","reason":"interrupted","duration_ms":13500,"remote_id":"a1b2c3d4-5e6f-4a1b-9c2d-3e4f5a6b7c8d","detached":true,"message":"stopped polling; the render continues upstream and is billed"}
{"v":1,"seq":10,"ts":"2026-09-17T14:10:24.510Z","type":"task:end","id":"download","status":"cancelled","reason":"interrupted"}
{"v":1,"seq":11,"ts":"2026-09-17T14:10:24.520Z","type":"result","status":"cancelled","exit_code":130,"duration_ms":22520,"error":{"code":"interrupted","message":"Interrupted. Remote job a1b2c3d4-5e6f-4a1b-9c2d-3e4f5a6b7c8d is still running and will be billed.","retryable":false},"summary":{"succeeded":1,"cancelled":2}}
```

---

## 4. JSON Schema (draft 2020-12)

The schema validates one line. Known types are checked with `if`/`then`, so an
event with an unknown `type` passes as long as its envelope is valid (rule
2.9.2). Objects allow additional properties (rule 2.9.1). Stream-level rules
(first/last line, `seq` continuity, declare-before-use, state transitions) cannot
be expressed per line and belong in a conformance test that replays a stream.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:run-events:v1:event",
  "title": "run-events v1 event (one JSON Lines record)",
  "type": "object",
  "required": ["v", "seq", "ts", "type"],
  "properties": {
    "v": { "const": 1 },
    "seq": { "type": "integer", "minimum": 1 },
    "ts": { "type": "string", "format": "date-time" },
    "type": { "type": "string", "minLength": 1 },
    "message": { "type": "string" }
  },
  "allOf": [
    {
      "if": { "properties": { "type": { "const": "run:start" } } },
      "then": {
        "required": ["tool", "command"],
        "properties": {
          "tool": { "type": "string", "minLength": 1 },
          "command": { "type": "string", "minLength": 1 },
          "version": { "type": "string" },
          "run_id": { "type": "string" },
          "capabilities": { "type": "array", "items": { "type": "string" } },
          "data": { "type": "object" }
        }
      }
    },
    {
      "if": { "properties": { "type": { "const": "task:declare" } } },
      "then": {
        "required": ["tasks"],
        "properties": {
          "tasks": { "type": "array", "minItems": 1, "items": { "$ref": "#/$defs/taskDeclaration" } }
        }
      }
    },
    {
      "if": { "properties": { "type": { "const": "task:queued" } } },
      "then": {
        "required": ["id"],
        "properties": {
          "id": { "$ref": "#/$defs/id" },
          "position": { "type": "integer", "minimum": 0 },
          "remote_id": { "type": "string" }
        }
      }
    },
    {
      "if": { "properties": { "type": { "const": "task:start" } } },
      "then": {
        "required": ["id"],
        "properties": {
          "id": { "$ref": "#/$defs/id" },
          "remote_id": { "type": "string" }
        }
      }
    },
    {
      "if": { "properties": { "type": { "const": "task:progress" } } },
      "then": {
        "required": ["id"],
        "properties": {
          "id": { "$ref": "#/$defs/id" },
          "completed": { "type": "number", "minimum": 0 },
          "total": { "type": "number", "minimum": 0 },
          "unit": { "type": "string" },
          "estimate": { "type": "boolean" },
          "rate": { "type": "number", "minimum": 0 },
          "eta_ms": { "type": "integer", "minimum": 0 },
          "elapsed_ms": { "type": "integer", "minimum": 0 }
        },
        "dependentRequired": { "estimate": ["total"] }
      }
    },
    {
      "if": { "properties": { "type": { "const": "log" } } },
      "then": {
        "required": ["level", "message"],
        "properties": {
          "level": { "enum": ["debug", "info", "warn", "error"] },
          "id": { "$ref": "#/$defs/id" },
          "code": { "type": "string" },
          "stream": { "enum": ["stdout", "stderr"] },
          "data": { "type": "object" }
        }
      }
    },
    {
      "if": { "properties": { "type": { "const": "artifact" } } },
      "then": { "$ref": "#/$defs/artifact" }
    },
    {
      "if": { "properties": { "type": { "const": "task:skip" } } },
      "then": {
        "required": ["id", "reason"],
        "properties": {
          "id": { "$ref": "#/$defs/id" },
          "reason": { "type": "string", "minLength": 1 },
          "data": { "type": "object" }
        }
      }
    },
    {
      "if": { "properties": { "type": { "const": "task:end" } } },
      "then": {
        "required": ["id", "status"],
        "properties": {
          "id": { "$ref": "#/$defs/id" },
          "status": { "enum": ["succeeded", "failed", "cancelled"] },
          "duration_ms": { "type": "integer", "minimum": 0 },
          "reason": { "type": "string" },
          "error": { "$ref": "#/$defs/error" },
          "remote_id": { "type": "string" },
          "detached": { "type": "boolean" },
          "data": { "type": "object" }
        },
        "if": { "properties": { "status": { "const": "failed" } } },
        "then": { "required": ["error"] }
      }
    },
    {
      "if": { "properties": { "type": { "const": "result" } } },
      "then": {
        "required": ["status", "exit_code"],
        "properties": {
          "status": { "enum": ["succeeded", "failed", "cancelled"] },
          "exit_code": { "type": "integer", "minimum": 0, "maximum": 255 },
          "duration_ms": { "type": "integer", "minimum": 0 },
          "error": { "$ref": "#/$defs/error" },
          "artifacts": { "type": "array", "items": { "$ref": "#/$defs/artifact" } },
          "cost": { "$ref": "#/$defs/cost" },
          "usage": { "type": "object" },
          "summary": {
            "type": "object",
            "additionalProperties": { "type": "integer", "minimum": 0 }
          },
          "data": { "type": "object" }
        },
        "allOf": [
          {
            "if": { "properties": { "status": { "const": "succeeded" } } },
            "then": { "properties": { "exit_code": { "const": 0 } } }
          },
          {
            "if": { "properties": { "status": { "enum": ["failed", "cancelled"] } } },
            "then": {
              "required": ["error"],
              "properties": { "exit_code": { "not": { "const": 0 } } }
            }
          }
        ]
      }
    }
  ],
  "$defs": {
    "id": {
      "type": "string",
      "pattern": "^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"
    },
    "taskDeclaration": {
      "type": "object",
      "required": ["id"],
      "properties": {
        "id": { "$ref": "#/$defs/id" },
        "description": { "type": "string" },
        "kind": { "type": "string" },
        "parent": { "$ref": "#/$defs/id" },
        "depends_on": { "type": "array", "items": { "$ref": "#/$defs/id" }, "uniqueItems": true }
      }
    },
    "artifact": {
      "type": "object",
      "anyOf": [{ "required": ["path"] }, { "required": ["url"] }],
      "properties": {
        "path": { "type": "string", "minLength": 1 },
        "url": { "type": "string", "format": "uri" },
        "id": { "$ref": "#/$defs/id" },
        "kind": { "type": "string" },
        "mime": { "type": "string" },
        "bytes": { "type": "integer", "minimum": 0 },
        "sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
        "role": { "enum": ["final", "intermediate"] },
        "reused": { "type": "boolean" },
        "data": { "type": "object" }
      }
    },
    "error": {
      "type": "object",
      "required": ["code", "message"],
      "properties": {
        "code": { "type": "string", "minLength": 1 },
        "message": { "type": "string" },
        "retryable": { "type": "boolean" },
        "hints": { "type": "array", "items": { "type": "string" } },
        "data": { "type": "object" }
      }
    },
    "cost": {
      "type": "object",
      "required": ["amount", "currency"],
      "properties": {
        "amount": { "type": "number", "minimum": 0 },
        "currency": { "type": "string", "pattern": "^[A-Z]{3}$" },
        "basis": { "enum": ["billed", "quoted"] }
      }
    }
  }
}
```

"""The four stages, driven through yue2-infer.

THIS IS THE ONE MODULE THAT REACHES PAST THE PUBLIC PIPELINE API. The public
`YuE2Pipeline.__call__` runs all four stages with one seed and fixed solver,
which cannot express: a per-stage seed, a token continuation (render --bars /
extend), an ODE that starts part-way from existing audio (--strength), a
solver other than midpoint, or anchoring latents outside an edited region. So:

    stage 1  plan    pipe.plan()                              public
    stage 2  tokens  pipe._generate() + protocol prefixes     private
    stage 3  synth   nar.song_chunks / CachedNAR.velocity     private
    stage 4  decode  YuE2VAE.decode / decode_tiled / encode   public

tests/test_upstream_contract.py pins every private name used here, so a
`uv lock --upgrade-package yue2-infer` that moves one fails the suite instead of
silently rendering noise.

THE DEFAULT PATH IS UPSTREAM'S PATH. With --solver midpoint, strength 1, no
anchors and the same seed, synth() performs exactly the arithmetic of
yue2.nar.synthesize (same full-song CPU noise draw, same logit timesteps, same
chunking). That equivalence is checked against a real upstream render in
tests/live/, not assumed.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "m-a-p/YuE2-3B"
VAES = {"standard": "m-a-p/YuE2-Vae", "legacy": "m-a-p/YuE2-Vae-legacy"}
SAMPLE_RATE = 48000
HOP = 1920


def vae_repo(value: str) -> str:
    return VAES.get(value, value)


@dataclass
class Runtime:
    model: str = DEFAULT_MODEL
    revision: str | None = None
    device: str = "auto"
    backend: str = "torch"
    quantization: str = "none"
    budget: float = 24.0
    offload_ar: bool = False
    offline: bool = False
    verify_hashes: bool = True
    quiet: bool = False


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    kind = requested.split(":", 1)[0]
    if kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {requested}: CUDA is not available on this machine")
    if kind == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps: Apple MPS is not available on this machine")
    return requested


def check_runtime(rt: Runtime, device: str) -> None:
    """Refuse CUDA-only options before a multi-GB load, in our words."""
    cuda = device.startswith("cuda")
    if rt.backend == "vllm" and not cuda:
        raise RuntimeError("--backend vllm needs CUDA (and `uv sync --extra cuda`)")
    if rt.quantization == "fp8" and not cuda:
        raise RuntimeError("--quantization fp8 needs a CUDA GPU with compute capability >= 8.9")
    if rt.backend == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("--backend vllm: vllm is not installed; run `uv sync --extra cuda`") from exc


class Engine:
    """Lazily loads the pipeline once per process; every stage shares it."""

    def __init__(self, rt: Runtime, on_load=None):
        self.rt = rt
        self._pipe = None
        self.device = None
        self.on_load = on_load  # called with the pipeline once it exists (progress bridge)

    @property
    def pipe(self):
        if self._pipe is None:
            from yue2 import YuE2Pipeline
            self.device = resolve_device(self.rt.device)
            check_runtime(self.rt, self.device)
            self._pipe = YuE2Pipeline.from_pretrained(
                self.rt.model, vae=VAES["standard"], revision=self.rt.revision,
                local_files_only=self.rt.offline, device=self.device,
                memory_budget_gib=self.rt.budget, backend=self.rt.backend,
                quantization=self.rt.quantization, offload_ar=self.rt.offload_ar,
                verify_hashes=self.rt.verify_hashes, progress=not self.rt.quiet)
            if self.on_load is not None:
                self.on_load(self._pipe)
        return self._pipe

    def close(self):
        if self._pipe is not None:
            self._pipe.close()
            self._pipe = None

    # ------------------------------------------------------------------ stage 1
    def plan(self, *, style, lyrics, cot, seed, abc, sampling: dict):
        from yue2.protocol import SongRequest
        request = SongRequest(style=style, lyrics=lyrics, cot=cot, seed=seed, abc=abc)
        start = time.perf_counter()
        plan = self.pipe.plan(request=request, abc_sampling=sampling)
        return plan, time.perf_counter() - start

    # ------------------------------------------------------------------ stage 2
    def tokens(self, plan, *, seed: int, cfg: float | None, sampling: dict,
               keep: list[int] | None = None, tail: list[int] | None = None,
               exact: int | None = None, extend_frames: int | None = None):
        """Sample semantic tokens after `plan.prefix`.

        keep          tokens forced before sampling starts (a continuation)
        exact         sample exactly this many new tokens (end token suppressed)
        extend_frames with `keep`: forbid the end token for this many new frames
        tail          tokens appended after the sampled ones (a middle repaint)
        """
        from yue2.protocol import CODEC_OFFSET, negative_prefix, resolve_sampling
        pipe = self.pipe
        request = plan.request
        guidance = cfg if cfg is not None else (1.01 if request.cot == "off" else 1.0)
        base = resolve_sampling(sampling, pipe.generation_config.semantic)
        keep = list(keep or [])
        forced = [t + CODEC_OFFSET for t in keep]
        prefix = list(plan.prefix) + forced
        negative = None
        if guidance != 1:
            negative = negative_prefix(request, pipe.tokenizer, plan.abc_ids) + forced

        from dataclasses import replace
        budget = 24576 - max(len(prefix), len(negative or [])) - 1
        if exact is not None:
            s = replace(base, min_tokens=exact, max_tokens=exact)
        else:
            max_tokens = min(base.max_tokens - len(keep), budget) if keep else min(base.max_tokens, budget)
            min_tokens = min(max(base.min_tokens - len(keep), extend_frames or 0), max_tokens)
            s = replace(base, min_tokens=max(0, min_tokens), max_tokens=max(1, max_tokens))
        if s.max_tokens > budget:
            raise ValueError(f"prefix ({len(prefix)} tokens) leaves room for {budget} tokens, "
                             f"{s.max_tokens} requested; shorten the lyrics/score or the song")
        ids, timing, truncated = pipe._generate(prefix, s, seed, "semantic", negative=negative,
                                                cfg_scale=guidance, legacy_off=request.cot == "off")
        new = [int(t) - CODEC_OFFSET for t in ids]
        if exact is not None:
            truncated = False  # the budget IS the requested length
        return keep + new + list(tail or []), {**timing, "guidance": guidance, "kept": len(keep),
                                               "sampled": len(new), "tail": len(tail or []),
                                               "sampling": s.__dict__}, truncated

    # ------------------------------------------------------------------ stage 3
    def synth(self, plan, tokens: list[int], *, seed: int, steps: int = 32, solver: str = "midpoint",
              strength: float = 1.0, init: np.ndarray | None = None,
              anchor: np.ndarray | None = None, attention: str = "sdpa",
              query_chunk: int | None = None) -> tuple[np.ndarray, dict]:
        """Flow-match latents for `tokens`.

        The ODE runs x from t=1 (noise) to t=0 (latents), x_t = t*noise + (1-t)*x0,
        velocity v = noise - x0, so each step is x <- x - v*dt.

        init + strength<1  start at t=strength from strength*noise + (1-strength)*init
        anchor (bool mask) frames that are RE-IMPOSED from `init` after every step,
                            at the current noise level -- RePaint for a flow ODE.
                            That keeps an unedited region's latents while the edited
                            region is solved with it as context.
        """
        import torch
        from yue2 import nar
        pipe = self.pipe
        if len(tokens) < 1:
            raise ValueError("no semantic tokens to synthesize")
        if not 0 < strength <= 1:
            raise ValueError("strength must be in (0, 1]")
        if (strength < 1 or anchor is not None) and init is None:
            raise ValueError("--strength < 1 and anchoring need source latents")
        if init is not None and len(init) != len(tokens):
            raise ValueError(f"source latents have {len(init)} frames, tokens have {len(tokens)}")
        if pipe.backend == "vllm":
            from yue2.fast import close_vllm
            close_vllm(pipe)
        if pipe.quantization != "none" and pipe._model is not None:
            from yue2.quantization import restore_ar
            restore_ar(pipe._model)
        model = pipe._load_model(for_nar=True)
        chunks = nar.song_chunks(plan.prefix, tokens, seed, pipe.generation_config.context)
        x0 = torch.as_tensor(init, dtype=torch.float32) if init is not None else None
        mask = torch.as_tensor(anchor, dtype=torch.bool) if anchor is not None else None
        t_start = float(strength)
        grid = [t_start * (1 - i / steps) for i in range(steps + 1)]  # t_start ... 0
        evals = {"euler": 1, "midpoint": 2, "heun": 2}[solver]
        total = steps * len(chunks)
        start_time = time.perf_counter()
        output = []
        offset = 0
        with pipe._status("Synthesizing audio", unit="steps") as status:
            for index, chunk in enumerate(chunks):
                n = len(chunk.noise)
                engine = nar.CachedNAR(model, chunk, attention, query_chunk)
                sl = slice(offset, offset + n)
                noise = chunk.noise
                c_x0 = x0[sl] if x0 is not None else None
                c_mask = mask[sl] if mask is not None else None
                with nar._offload_ar(model, pipe.offload_ar):
                    try:
                        dtype, device = engine.dtype, engine.device
                        if t_start < 1:
                            state = (t_start * noise + (1 - t_start) * c_x0).to(device=device, dtype=dtype)
                        else:
                            state = noise.to(device=device, dtype=dtype)
                        dev_noise = noise.to(device=device, dtype=dtype) if c_mask is not None else None
                        dev_x0 = c_x0.to(device=device, dtype=dtype) if c_mask is not None else None
                        dev_mask = c_mask.to(device=device)[:, None] if c_mask is not None else None
                        for step in range(steps):
                            t, t_next = grid[step], grid[step + 1]
                            dt = t - t_next
                            if solver == "euler":
                                state = state - engine.velocity(state, _raw(t)) * dt
                            elif solver == "midpoint":
                                first = engine.velocity(state, _raw(t))
                                mid = state - first * (dt / 2)
                                state = state - engine.velocity(mid, _raw(t - dt / 2)) * dt
                            else:  # heun
                                first = engine.velocity(state, _raw(t))
                                guess = state - first * dt
                                if t_next <= 0:
                                    state = guess
                                else:
                                    second = engine.velocity(guess, _raw(t_next))
                                    state = state - (first + second) * (dt / 2)
                            if dev_mask is not None:
                                known = t_next * dev_noise + (1 - t_next) * dev_x0
                                state = torch.where(dev_mask, known, state)
                            status.update(index * steps + step + 1, total=total)
                        result = state.float().cpu()
                        if not torch.isfinite(result).all():
                            raise FloatingPointError("acoustic flow matching produced non-finite latents")
                        output.append(result)
                    finally:
                        engine.close()
                del engine
                offset += n
        latents = torch.cat(output, 0).numpy()
        return latents, {"seconds": time.perf_counter() - start_time, "chunks": len(chunks),
                         "model_evaluations": steps * evals * len(chunks)}

    # ------------------------------------------------------------------ stage 4
    def load_vae(self, vae: str, revision: str | None, *, encoder: bool):
        from yue2.modeling_vae import YuE2VAE
        from yue2.storage import resolve_model
        path = resolve_model(vae_repo(vae), revision=revision, local_files_only=self.rt.offline)
        device = self.device or resolve_device(self.rt.device)
        return YuE2VAE.from_pretrained(path, decoder_only=not encoder, device=device,
                                       local_files_only=True), path

    def decode(self, latents: np.ndarray, *, vae: str, revision: str | None, mode: str,
               tile_frames: int, halo_frames: int | None) -> tuple[np.ndarray, dict]:
        import torch
        self._free_model()
        model, path = self.load_vae(vae, revision, encoder=False)
        start = time.perf_counter()
        z = torch.as_tensor(latents, dtype=torch.float32).T.unsqueeze(0)
        halo = model.required_halo(tile_frames) if halo_frames is None else halo_frames
        try:
            with torch.inference_mode():
                if mode == "full":
                    audio = model.decode(z.to(model.decoder_device)).cpu()
                else:
                    audio = model.decode_tiled(z, core_frames=tile_frames, halo_frames=halo, output_device="cpu")
            if not torch.isfinite(audio).all():
                raise ValueError("VAE produced non-finite audio")
            out = audio[0].float().clamp(-1, 1).T.contiguous().numpy()
        finally:
            model.to("cpu")
            del model
            self._empty_cache()
        return out, {"seconds": time.perf_counter() - start, "vae_path": str(path),
                     "halo_frames": halo if mode == "tiled" else None}

    def encode(self, audio: np.ndarray, *, vae: str, revision: str | None,
               tile_frames: int = 1024, halo_frames: int = 64) -> np.ndarray:
        """48 kHz stereo float [samples, 2] -> latents [frames, 64], tiled with cropped halos."""
        import torch
        self._free_model()
        model, _ = self.load_vae(vae, revision, encoder=True)
        device = next(model.encoder.parameters()).device
        wave = torch.as_tensor(audio.T, dtype=torch.float32)[None]
        frames = wave.shape[-1] // HOP
        wave = wave[..., :frames * HOP]
        chunks = []
        try:
            with torch.inference_mode():
                for start in range(0, frames, tile_frames):
                    end = min(frames, start + tile_frames)
                    left, right = max(0, start - halo_frames), min(frames, end + halo_frames)
                    z = model.encode(wave[..., left * HOP:right * HOP].to(device)).cpu()
                    chunks.append(z[..., start - left:start - left + (end - start)])
        finally:
            model.to("cpu")
            del model
            self._empty_cache()
        return torch.cat(chunks, -1)[0].T.contiguous().numpy()

    def _free_model(self):
        if self._pipe is not None and self._pipe._model is not None:
            self._pipe._model.to("cpu")
        self._empty_cache()

    @staticmethod
    def _empty_cache():
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _raw(t: float) -> float:
    """Upstream feeds the model logit(t) clamped to [-20, 20]; t=0 maps to -20."""
    import torch
    return torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()


def load_audio(path: Path) -> np.ndarray:
    """Any file ffmpeg reads -> 48 kHz stereo float32 [samples, 2]."""
    import soundfile as sf
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "in.wav"
        result = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
                                 "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(wav)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"ffmpeg could not read {path}: {result.stderr.strip()}")
        data, rate = sf.read(wav, dtype="float32", always_2d=True)
    assert rate == SAMPLE_RATE
    return data


def export_audio(audio: np.ndarray, path: Path, fmt: str, master: Path | None = None) -> Path:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "flac":
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_24")
    elif fmt == "wav":
        sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
    else:
        source = master
        tmp = None
        if source is None:
            tmp = tempfile.TemporaryDirectory()
            source = Path(tmp.name) / "master.flac"
            sf.write(source, audio, SAMPLE_RATE, subtype="PCM_24")
        result = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
                                 "-codec:a", "libmp3lame", "-b:a", "320k", str(path)],
                                capture_output=True, text=True)
        if tmp:
            tmp.cleanup()
        if result.returncode:
            raise RuntimeError(f"ffmpeg mp3 export failed: {result.stderr.strip()}")
    return path


def anchor_mask(frames: int, edit_start: int, edit_end: int | None, margin: int) -> np.ndarray:
    """True where a frame keeps its source latent, in the NEW timeline.

    render --bars A-B keeps the timeline (the edit samples exactly B-A frames),
    so the source tail after `edit_end` lines up frame for frame. A tail repaint
    or an extension (edit_end None) keeps only the head. The edit region is
    widened by `margin` frames each side so the solver blends into it rather
    than meeting a hard seam."""
    mask = np.zeros(frames, dtype=bool)
    mask[:max(0, edit_start - margin)] = True
    if edit_end is not None and edit_end + margin < frames:
        mask[edit_end + margin:] = True
    return mask

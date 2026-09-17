"""Every PRIVATE name engine.py uses from yue2-infer, pinned.

After `uv lock --upgrade-package yue2-infer`, this file is what says whether the
CLI still fits. A failure here means an internal moved: fix engine.py, do not
loosen the test.
"""
import inspect

import yue2
from yue2 import nar, pipeline, protocol
from yue2.modeling_vae import YuE2VAE


def params(fn):
    return list(inspect.signature(fn).parameters)


def test_pipeline_private_surface():
    P = pipeline.YuE2Pipeline
    assert params(P._generate)[:6] == ["self", "prefix", "sampling", "seed", "phase", "kwargs"]
    assert "for_nar" in params(P._load_model)
    assert {"label", "total", "unit"} <= set(params(P._status))
    for name in ("plan", "close", "tokenizer"):
        assert hasattr(P, name) or name == "tokenizer"
    assert {"verify_hashes", "offload_ar", "quantization", "backend", "memory_budget_gib", "progress"} <= set(params(P.__init__))


def test_sampling_loop_accepts_what_we_pass():
    from yue2.sampling import generate_tokens
    assert {"negative", "cfg_scale", "legacy_off"} <= set(params(generate_tokens))


def test_nar_internals():
    assert params(nar.song_chunks)[:4] == ["prefix", "codec", "seed", "context"]
    assert params(nar.CachedNAR.__init__)[:5] == ["self", "model", "chunk", "attention", "query_chunk_size"]
    assert params(nar.CachedNAR.velocity) == ["self", "state", "raw_t"]
    assert hasattr(nar.CachedNAR, "close") and hasattr(nar, "_offload_ar")
    # the upstream solver we mirror: 32-step midpoint from t=1 with logit timesteps clamped to +-20
    source = inspect.getsource(nar.CachedNAR.solve)
    assert "torch.logit" in source and "clamp(-20, 20)" in source and "dt / 2" in source


def test_protocol_constants_and_prefixes():
    assert protocol.CODEC_OFFSET == 151853 and protocol.CODEC_SIZE == 32768 and protocol.CONTEXT == 24576
    assert params(protocol.negative_prefix)[:3] == ["request", "tokenizer", "abc_ids"]
    assert callable(protocol.resolve_sampling)
    assert {f for f in protocol.Sampling.__dataclass_fields__} == {
        "temperature", "top_p", "top_k", "repetition_penalty", "penalty_window", "min_tokens", "max_tokens"}


def test_vae_surface():
    assert "decoder_only" in params(YuE2VAE.from_pretrained)
    assert {"core_frames", "halo_frames", "output_device"} <= set(params(YuE2VAE.decode_tiled))
    for name in ("encode", "decode", "required_halo", "decoder_device"):
        assert hasattr(YuE2VAE, name)


def test_symbolic_plan_roundtrip_api():
    assert hasattr(yue2.SymbolicPlan, "load") and hasattr(yue2.SymbolicPlan, "save")

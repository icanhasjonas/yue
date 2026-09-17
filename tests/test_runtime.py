"""CUDA-only options are refused in our words on a machine without CUDA -- and allowed with it.

We cannot run CUDA here; these tests pin the DECISIONS so the CUDA path is ready
the day it meets a GPU (and `yue doctor` there reports what it found).
"""
import pytest

from yuecli.engine import Runtime, check_runtime


def test_vllm_needs_cuda():
    with pytest.raises(RuntimeError, match="needs CUDA"):
        check_runtime(Runtime(backend="vllm"), "mps")


def test_fp8_needs_cuda():
    with pytest.raises(RuntimeError, match="compute capability"):
        check_runtime(Runtime(quantization="fp8"), "cpu")


def test_cuda_device_with_default_backend_passes():
    check_runtime(Runtime(), "cuda:0")
    check_runtime(Runtime(quantization="fp8"), "cuda")


def test_vllm_on_cuda_without_the_extra_says_how_to_install(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "vllm":
            raise ImportError("no vllm")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="--extra cuda"):
        check_runtime(Runtime(backend="vllm"), "cuda")
